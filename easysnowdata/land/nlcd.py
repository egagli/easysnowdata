"""NLCD — the USGS National Land Cover Database (§4.4, companion §B.7).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    nlcd = esd.land.nlcd.load(aoi)                          # Annual NLCD, latest year
    ts = esd.land.nlcd.load(aoi, time="1990/2024")          # one map per year
    rel = esd.land.nlcd.load(aoi, source="gee", layer="impervious")   # 2021 release

Two Earth Engine routes: **Annual NLCD** (community asset, 1985-2024, the
default because the official release is frozen at 2021) and the official
**2021 release**, which is where the science products and the impervious
descriptors live. Class tables are read from the asset properties at run time,
so they follow whatever the asset ships.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers, temporal
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract
from easysnowdata.processing.categorical import set_flags

__all__ = ["PRODUCT_ID", "ANNUAL_ASSETS", "RELEASE_ASSET", "RELEASE_LAYERS", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "nlcd"
#: The official release: one image per epoch, every layer a band.
RELEASE_ASSET = "USGS/NLCD_RELEASES/2021_REL/NLCD"
RELEASE_LAYERS = (
    "landcover",
    "impervious",
    "impervious_descriptor",
    "science_products_land_cover_change_count",
    "science_products_land_cover_change_first_disturbance_date",
    "science_products_land_cover_change_index",
    "science_products_land_cover_science_product",
    "science_products_forest_disturbance_date",
)
#: Annual NLCD (community mirror): one collection per layer, one image per year.
ANNUAL_ROOT = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD"
ANNUAL_ASSETS = {
    "landcover": f"{ANNUAL_ROOT}/LANDCOVER",
    "landcover_confidence": f"{ANNUAL_ROOT}/LANDCOVER_CONFIDENCE",
    "landcover_change": f"{ANNUAL_ROOT}/LANDCOVER_CHANGE",
    "impervious": f"{ANNUAL_ROOT}/FRACTIONAL_IMPERVIOUS_SURFACE",
    "impervious_descriptor": f"{ANNUAL_ROOT}/IMPERVIOUS_DESCRIPTOR",
    "spectral_change_doy": f"{ANNUAL_ROOT}/SPECTRAL_CHANGE_DOY",
}


def _asset(source_id: str, layer: str) -> str:
    if source_id == "gee":
        if layer not in RELEASE_LAYERS:
            raise ValueError(
                f"Unknown NLCD layer {layer!r} for the 2021 release; "
                f"available: {', '.join(RELEASE_LAYERS)}."
            )
        return RELEASE_ASSET
    try:
        return ANNUAL_ASSETS[layer]
    except KeyError:
        raise ValueError(
            f"Unknown Annual NLCD layer {layer!r}; "
            f"available: {', '.join(ANNUAL_ASSETS)}."
        ) from None


def _entries(value: Any) -> list[str]:
    """One class table field as a list.

    Earth Engine's own NLCD assets store these as lists. The community Annual
    NLCD asset stores them as delimited strings instead — ``"11,12,21,…"`` and
    ``"Open Water|Perennial Ice/Snow|…"`` — so a class *name* may contain a
    comma ("Developed, Open Space") and the pipe has to win where both appear.
    """
    if value is None:
        return []
    if isinstance(value, str):
        separator = "|" if "|" in value else ","
        return [part.strip() for part in value.split(separator) if part.strip()]
    return [v for v in value]


def _class_table(
    image: Any, collection: Any, layer: str, band: str
) -> tuple[list, list, list] | None:
    """Read ``<layer>_class_values`` / ``_names`` / ``_palette`` from the asset.

    The image first, then the collection: the official USGS releases put the
    table on each image, while the community Annual NLCD asset puts it on the
    collection only — which is why the default route came back as float32 with
    no legend until this looked in both places.
    """
    sources = []
    for obj in (image, collection):
        try:
            sources.append(obj.getInfo().get("properties") or {})
        except Exception as exc:  # noqa: BLE001 — a missing table is not a failure
            _logger.debug("NLCD class table unavailable: %s", exc)
    for properties in sources:
        for prefix in (layer, band, "landcover"):
            values = _entries(properties.get(f"{prefix}_class_values"))
            names = _entries(properties.get(f"{prefix}_class_names"))
            palette = _entries(properties.get(f"{prefix}_class_palette"))
            if (
                values
                and names
                and palette
                and len({len(values), len(names), len(palette)}) == 1
            ):
                return (
                    [int(v) for v in values],
                    [str(n).split(":")[0].split(".")[0] for n in names],
                    [c if str(c).startswith("#") else f"#{c}" for c in palette],
                )
    _logger.info("The %s asset carries no class table for %r.", PRODUCT_ID, layer)
    return None


def _collection(ee: Any, asset: str, time: Any):
    """The images to load: the whole period when *time* is given, else the newest."""
    collection = ee.ImageCollection(asset)
    if time is None:
        return collection.sort("system:time_start", False).limit(1), False
    start, end = temporal.parse_time(time)
    start_text = "1985-01-01" if start is None else start.strftime("%Y-%m-%d")
    end_text = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return collection.filterDate(start_text, end_text), True


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    layer: str = "landcover",
    time: Any = None,
    chunks: Any = contract.DEFAULT,
    mask: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Load one NLCD layer for *aoi* on the asset's native 30 m Albers grid.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. NLCD covers CONUS.
    source
        ``"gee-annual"`` (default; Annual NLCD 1985-2024) or ``"gee"`` (the
        official 2021 release, which has the science products).
    layer
        For the annual route one of :data:`ANNUAL_ASSETS`, for the release one
        of :data:`RELEASE_LAYERS`.
    time
        Any form :func:`easysnowdata.temporal.parse_time` accepts. ``None``
        (default) returns the newest year as a 2-D map with the year as a
        scalar coordinate; a range keeps the ``time`` dimension.
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        ``False`` (default, §2.5 for categorical products) keeps the source
        values; ``True`` masks the layer's nodata to NaN.
    **kwargs
        Passed to ``xarray.open_dataset(engine="ee")``.

    Returns
    -------
    xarray.DataArray
        The layer on its native grid (dims ``y``/``x``, plus ``time`` when a
        range was asked for), with CF flag attributes when the asset ships a
        class table.

    Raises
    ------
    easysnowdata.auth.CredentialError
        When Earth Engine is not configured.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    asset = _asset(src.id, layer)
    ee = providers.gee.ee()
    collection, keep_time = _collection(ee, asset, time)
    image = collection.first()
    if src.id == "gee":
        collection = collection.select([layer])
        image = image.select([layer])
    grid = providers.gee.grid_params(image, aoi)
    params: dict[str, Any] = {"grid": grid}
    if chunks is not contract.DEFAULT:
        params["chunks"] = chunks
    ds = providers.gee.open_dataset(collection, aoi, **params, **kwargs)
    names = list(ds.data_vars)
    if layer in names:
        band = layer
    elif len(names) == 1:  # the annual assets name their single band themselves
        band = names[0]
    else:
        raise ValueError(
            f"The {asset} images have no {layer!r} band; available: {', '.join(names)}."
        )
    da = ds[band]
    if "time" in da.dims and (not keep_time or da.sizes["time"] == 1):
        da = da.isel(time=0)  # keep the year as a scalar coordinate
    da = contract.write_crs(da, grid["crs"])
    table = _class_table(image, collection, layer, band)
    nodata = 0 if table is not None and 0 not in table[0] else None
    if table is not None:
        if da.dtype.kind == "f" and max(table[0]) < 256:
            # xee hands back float32 with NaN wherever the asset is masked.
            # The sentinel has to replace NaN *before* the cast, or a masked
            # pixel becomes an arbitrary class rather than nodata.
            da = da.fillna(nodata if nodata is not None else 0).astype("uint8")
        da = set_flags(da, *table, long_name=f"NLCD {layer.replace('_', ' ')}")
    da = (
        contract.mask_continuous(da, nodata)
        if mask
        else contract.set_categorical_nodata(da, nodata)
    )
    da = da.rename(layer)
    da.attrs.update(
        contract.provenance(
            product,
            src,
            source_url=f"https://code.earthengine.google.com/?asset={asset}",
            asset=asset,
            layer=layer,
            extent="CONUS",
        )
    )
    if chunks is None:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="land",
    title="National Land Cover Database (NLCD)",
    description=(
        "Land cover, impervious surface and change products for the conterminous "
        "United States at 30 m. Annual NLCD covers 1985-2024; the official 2021 "
        "release covers the 2001-2021 epochs and the science products."
    ),
    sources=(
        Source(
            id="gee-annual",
            provider="gee",
            location=f"{ANNUAL_ROOT}/<LAYER>",
            requires=("earthengine",),
            resolution_m=30,
            extent="CONUS",
            temporal="1985/2024 (annual)",
            latency="annual",
            notes=(
                "community mirror of Annual NLCD collection 1.x on Earth Engine; "
                "the official release is frozen at 2021, so this is the default "
                "for a land-cover time series"
            ),
            title="Earth Engine (Annual NLCD)",
            health=Probe(
                "Annual NLCD (GEE community asset)",
                partial(health.gee_asset, ANNUAL_ASSETS["landcover"]),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location=RELEASE_ASSET,
            requires=("earthengine",),
            resolution_m=30,
            extent="CONUS",
            temporal="2001-2021 epochs",
            notes="the official release; class tables come from the asset properties at run time",
            title="Earth Engine (2021 release)",
            health=Probe(
                "NLCD (GEE/USGS)",
                partial(health.gee_asset, RELEASE_ASSET),
            ),
        ),
    ),
    variables=(
        Variable(
            "landcover",
            dtype="uint8",
            long_name="NLCD land cover class",
        ),
        Variable(
            "impervious", units="%", dtype="uint8", long_name="impervious surface"
        ),
    ),
    citation=(
        "Dewitz, J. (2023). National Land Cover Database (NLCD) 2021 Products. "
        "U.S. Geological Survey data release. Annual NLCD: U.S. Geological Survey "
        "(2024), Annual NLCD Collection 1 Science Products."
    ),
    license="Public domain (US government data)",
    doi="10.5066/P9JZ7AO3",
    loader="easysnowdata.land.nlcd.load",
    examples=("land/plot_nlcd.py",),
    references=(
        "https://www.mrlc.gov/data",
        "https://gee-community-catalog.org/projects/annual_nlcd/",
    ),
    tags=("land cover", "nlcd", "impervious"),
)

catalog.register(PRODUCT, replace=True)
