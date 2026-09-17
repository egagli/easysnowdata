"""MODIS snow cover: MOD10A1, MOD10A2 and the cloud-gap-filled MOD10A1F.

Sources (companion file §B.4):

``nsidc`` (default)
    NSIDC via ``earthaccess``: the archive of record and the only route that
    carries the cloud-gap-filled product and the Aqua siblings. The granules
    are HDF-EOS2 (HDF4), which cannot be read through fsspec file objects, so
    they are downloaded once into the easysnowdata cache and opened as GDAL
    subdatasets. Needs Earthdata Login and a GDAL with the HDF4 driver
    (``libgdal-hdf4`` on conda-forge).
``planetary-computer``
    ``modis-10A1-061`` / ``modis-10A2-061`` as COGs, credential-free, kept as
    the historical route. Archiving of MOD10A2 there reportedly stopped in
    2025, which is an access-route failure rather than a product failure: the
    same granules are still at NSIDC (verified 2026-09-16).

::

    import easysnowdata as esd
    granules = esd.snow.modis.search(aoi, "2023-03", product="MOD10A1F")
    snow = esd.snow.modis.load(aoi, "2023-03", product="MOD10A1F")
    binary = esd.processing.binary_snow(snow["CGF_NDSI_Snow_Cover"], product="MOD10A1F")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.auth.planetary_computer import STAC_URL as PC_STAC
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.processing import snow as snow_processing

__all__ = [
    "PRODUCT",
    "PRODUCTS",
    "DEFAULT_VARIABLES",
    "GRIDS",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

#: Supported granule short names → (grid name, default variable).
PRODUCTS: dict[str, tuple[str, str]] = {
    "MOD10A1": ("MOD_Grid_Snow_500m", "NDSI_Snow_Cover"),
    "MYD10A1": ("MOD_Grid_Snow_500m", "NDSI_Snow_Cover"),
    "MOD10A2": ("MOD_Grid_Snow_500m", "Maximum_Snow_Extent"),
    "MYD10A2": ("MOD_Grid_Snow_500m", "Maximum_Snow_Extent"),
    # Verified against a v61 granule on 2026-09-17: the cloud-gap-filled
    # products store their fields in the same MOD_Grid_Snow_500m grid as the
    # daily ones, not in a MOD_CGF_NDSI_500m grid of their own.
    "MOD10A1F": ("MOD_Grid_Snow_500m", "CGF_NDSI_Snow_Cover"),
    "MYD10A1F": ("MOD_Grid_Snow_500m", "CGF_NDSI_Snow_Cover"),
}
#: The HDF-EOS grid each product stores its data fields in.
GRIDS = {name: grid for name, (grid, _) in PRODUCTS.items()}
#: What ``load`` reads when ``variables=None``.
DEFAULT_VARIABLES = {name: (variable,) for name, (_, variable) in PRODUCTS.items()}
#: Products Planetary Computer mirrors, and under which collection id.
PC_COLLECTIONS = {"MOD10A1": "modis-10A1-061", "MOD10A2": "modis-10A2-061"}

MODIS_VERSION = "61"
_NSIDC_DOCS = "https://nsidc.org/data/mod10a1f/versions/61"
_PC_DOCS = "https://planetarycomputer.microsoft.com/dataset/modis-10A1-061"
_GRANULE_DATE = re.compile(r"\.A(\d{4})(\d{3})\.")

_NDSI_VARIABLE = Variable(
    "NDSI_Snow_Cover",
    units="%",
    dtype="uint8",
    nodata=255,
    long_name="NDSI snow cover",
    flag_values=tuple(sorted(snow_processing.NDSI_FLAGS)),
    flag_meanings=tuple(
        snow_processing.NDSI_FLAGS[v][0] for v in sorted(snow_processing.NDSI_FLAGS)
    ),
    flag_colors=tuple(
        snow_processing.NDSI_FLAGS[v][1] for v in sorted(snow_processing.NDSI_FLAGS)
    ),
)
_CGF_VARIABLE = Variable(
    "CGF_NDSI_Snow_Cover",
    units="%",
    dtype="uint8",
    nodata=255,
    long_name="cloud-gap-filled NDSI snow cover",
    flag_values=_NDSI_VARIABLE.flag_values,
    flag_meanings=_NDSI_VARIABLE.flag_meanings,
    flag_colors=_NDSI_VARIABLE.flag_colors,
)
_MAX_EXTENT_VARIABLE = Variable(
    "Maximum_Snow_Extent",
    dtype="uint8",
    nodata=255,
    long_name="8-day maximum snow extent",
    flag_values=tuple(snow_processing.MOD10A2_CLASSES),
    flag_meanings=tuple(name for name, _ in snow_processing.MOD10A2_CLASSES.values()),
    flag_colors=tuple(color for _, color in snow_processing.MOD10A2_CLASSES.values()),
)

PRODUCT = Product(
    id="modis-snow",
    theme="snow",
    title="MODIS snow cover (MOD10A1, MOD10A2, MOD10A1F)",
    description=(
        "Terra and Aqua MODIS daily NDSI snow cover, the 8-day maximum snow extent "
        "and the cloud-gap-filled daily product at 500 m, from NSIDC (all six "
        "products) or Planetary Computer (the two COG mirrors)."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="earthdata",
            location="MOD10A1 / MOD10A2 / MOD10A1F (and the MYD Aqua siblings)",
            requires=("earthdata",),
            resolution_m=500,
            temporal="2000-02/present",
            latency="~1 day",
            notes=(
                "HDF-EOS2 granules: downloaded to the cache and opened as GDAL "
                "subdatasets; needs the HDF4 driver (conda-forge libgdal-hdf4)"
            ),
            title="NSIDC (Earthdata)",
            health=(
                Probe(
                    "MODIS snow cover MOD10A1F (NASA NSIDC)",
                    partial(
                        health.earthdata_search,
                        "MOD10A1F",
                        temporal=("2023-01-01", "2023-01-07"),
                    ),
                ),
                # CMR's metadata search needs no Earthdata Login, so both of
                # these answer even in a run with no credentials — which is
                # the point: the MOD10A2 archiving stop would have shown up
                # here as a frozen date (§8).
                Probe(
                    "MODIS snow cover MOD10A1F latency (CMR)",
                    partial(health.cmr_latest, "MOD10A1F"),
                    requires=(),
                    kind="latency",
                ),
                Probe(
                    "MODIS snow cover MOD10A1F DMR++ (CMR)",
                    partial(
                        health.dmrpp_status,
                        "MOD10A1F",
                        fallback=health.FALLBACK_PARSERS["HDF-EOS2"],
                    ),
                    requires=(),
                    kind="virtualization",
                ),
            ),
        ),
        Source(
            id="planetary-computer",
            provider="stac",
            location="modis-10A1-061 / modis-10A2-061",
            resolution_m=500,
            temporal="2000-02/~2025",
            notes=(
                "COG mirror, credential-free; MOD10A2 archiving reportedly stopped "
                "in 2025 and the cloud-gap-filled product was never mirrored"
            ),
            title="Planetary Computer",
            health=Probe(
                "MODIS snow cover MOD10A1 (Planetary Computer)",
                partial(
                    health.stac_search,
                    PC_STAC,
                    "modis-10A1-061",
                    datetime_range="2023-01-01/2023-01-31",
                    sign=True,
                ),
            ),
        ),
    ),
    variables=(_NDSI_VARIABLE, _MAX_EXTENT_VARIABLE, _CGF_VARIABLE),
    citation=(
        "Hall, D. K. and Riggs, G. A. (2021). MODIS/Terra Snow Cover Daily L3 Global "
        "500m SIN Grid, Version 61 (MOD10A1); 8-Day (MOD10A2); Cloud-Gap-Filled "
        "(MOD10A1F). NASA NSIDC DAAC."
    ),
    license="NASA Earthdata (free registration)",
    doi="10.5067/MODIS/MOD10A1.061",
    references=(_NSIDC_DOCS, _PC_DOCS),
    loader="easysnowdata.snow.modis.load",
    examples=(
        "snow/plot_modis_snow.py",
        "snow/plot_viirs_snow.py",
    ),
    tags=("snow cover", "ndsi", "modis", "terra", "aqua"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _product(name: str) -> str:
    key = str(name).upper()
    if key not in PRODUCTS:
        raise ValueError(f"product must be one of {list(PRODUCTS)}, got {name!r}.")
    return key


def _variables(product: str, variables: str | Sequence[str] | None) -> list[str]:
    if variables is None:
        return list(DEFAULT_VARIABLES[product])
    return [variables] if isinstance(variables, str) else list(variables)


def _grids_in(path: Path | str) -> list[str]:
    """Every subdataset GDAL reports for *path* (empty when it cannot be read)."""
    import rasterio  # noqa: PLC0415

    try:
        with rasterio.open(str(path)) as src:
            return list(src.subdatasets)
    except Exception as exc:  # noqa: BLE001 — fall back to the declared grid
        _logger.debug("Could not list subdatasets of %s: %s", path, exc)
        return []


def granule_date(name: str) -> pd.Timestamp | None:
    """The acquisition date encoded in a granule name (``.AYYYYDDD.``)."""
    match = _GRANULE_DATE.search(str(name))
    if not match:
        return None
    year, doy = int(match.group(1)), int(match.group(2))
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def subdataset(path: Path | str, product: str, variable: str) -> str:
    """The GDAL subdataset path for one variable of an HDF-EOS2 granule.

    The grid name is taken from the granule itself when it can be read, and
    from :data:`GRIDS` otherwise. Asking GDAL for a grid the file does not
    have is worth avoiding: it does not raise, it returns a 1×0 dataset, and
    the failure surfaces several layers away as ``ValueError: Unknown dims``
    out of rioxarray. That is how MOD10A1F stayed broken — the table said
    ``MOD_CGF_NDSI_500m`` and the granules use ``MOD_Grid_Snow_500m``.
    """
    grids = _grids_in(path)
    if grids:
        wanted = [name for name in grids if name.endswith(f":{variable}")]
        if wanted:
            return wanted[0]
        raise ValueError(
            f"{Path(path).name} has no {variable!r} field; available: "
            + ", ".join(sorted(n.rsplit(":", 1)[-1] for n in grids))
        )
    return f'HDF4_EOS:EOS_GRID:"{path}":{GRIDS[product]}:{variable}'


def _variable_definition(name: str) -> Variable | None:
    for variable in PRODUCT.variables:
        if variable.name == name:
            return variable
    return None


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    product: str = "MOD10A1",
    source: str | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search for MODIS snow granules and return them as a ``GeoDataFrame``.

    On the NSIDC route the frame has one row per granule (``id``, ``start``,
    ``end``, ``size_mb``, ``data_links``); on Planetary Computer it is the
    usual STAC item frame.
    """
    src = resolve_source(PRODUCT, source)
    name = _product(product)
    if src.id == "planetary-computer":
        if name not in PC_COLLECTIONS:
            raise ValueError(
                f"Planetary Computer mirrors only {list(PC_COLLECTIONS)}; "
                f"{name} is on NSIDC (source='nsidc')."
            )
        items = providers.stac.search(
            "planetary-computer",
            PC_COLLECTIONS[name],
            aoi,
            time,
            max_items=max_items,
            **kwargs,
        )
        gdf = providers.stac.items_to_geodataframe(items)
    else:
        ensure_source(PRODUCT, src)
        granules = providers.earthdata.search(
            name, aoi, time, version=MODIS_VERSION, **kwargs
        )
        if max_items is not None:
            granules = granules[:max_items]
        gdf = providers.earthdata.granules_to_geodataframe(granules)
        if len(gdf):
            gdf["date"] = [granule_date(i) for i in gdf["id"]]
    gdf.attrs = {"source": src.id, "product": name}
    return gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    product: str = "MOD10A1",
    variables: str | Sequence[str] | None = None,
    source: str | None = None,
    granules: Any = None,
    resolution: float | None = None,
    crs: Any = None,
    chunks: Any = True,
    mask: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """Load MODIS snow cover as a lazy ``xarray.Dataset`` (``time``, ``y``, ``x``).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs. ``clip=False`` on the AOI keeps
        whole MODIS tiles.
    product
        ``"MOD10A1"`` (default), ``"MOD10A2"``, ``"MOD10A1F"`` or an Aqua
        sibling (``"MYD…"``).
    variables
        Which data fields to read; the product's main field by default.
    source
        ``"nsidc"`` (default) or ``"planetary-computer"``.
    granules
        A search result to load instead of searching again.
    mask
        ``True`` NaN-masks the sentinel values (cloud, night, fill …).
        The default keeps the raw byte with its CF flags, so nothing is lost;
        :func:`easysnowdata.processing.binary_snow` turns it into a mask.
    crs, resolution
        Reproject the native sinusoidal grid (default: keep it).
    """
    src = resolve_source(PRODUCT, source)
    name = _product(product)
    fields = _variables(name, variables)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if src.id == "planetary-computer":
        ds = _load_planetary_computer(
            name, fields, granules, parsed, aoi, time, resolution, crs, chunks, **kwargs
        )
    else:
        ds = _load_nsidc(name, fields, granules, parsed, aoi, time, chunks, **kwargs)

    ds = contract.apply_variables(
        ds,
        [v for v in PRODUCT.variables if v.name in ds.data_vars],
        mask=True if mask else False,
    )
    ds = contract.finalize(
        ds,
        PRODUCT,
        src,
        variables=(),
        source_url=_NSIDC_DOCS if src.id == "nsidc" else _PC_DOCS,
        attrs={"modis_product": name, "version": MODIS_VERSION},
    )
    return ds


def _load_planetary_computer(
    product: str,
    fields: Sequence[str],
    items: Any,
    parsed: Any,
    aoi: Any,
    time: Any,
    resolution: float | None,
    crs: Any,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    if product not in PC_COLLECTIONS:
        raise ValueError(
            f"Planetary Computer mirrors only {list(PC_COLLECTIONS)}; "
            f"{product} is on NSIDC (source='nsidc')."
        )
    if items is None:
        items = search(aoi, time, product=product, source="planetary-computer")
    if not len(items):
        raise ValueError(f"No {product} items on Planetary Computer for this AOI/time.")
    return providers.stac.load(
        items,
        parsed,
        bands=list(fields),
        resolution=resolution,
        crs=crs,
        chunks={} if chunks is True else chunks,
        groupby="solar_day",
        catalog="planetary-computer",
        **kwargs,
    )


def _load_nsidc(
    product: str,
    fields: Sequence[str],
    granules: Any,
    parsed: Any,
    aoi: Any,
    time: Any,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    """Download the HDF-EOS2 granules once and open their subdatasets."""
    providers.earthdata.require_hdf4(product)
    if granules is None:
        granules = providers.earthdata.search(product, aoi, time, version=MODIS_VERSION)
    elif isinstance(granules, gpd.GeoDataFrame):
        raise TypeError(
            "Pass the earthaccess granules themselves (what providers.earthdata.search "
            "returns), not the GeoDataFrame from snow.modis.search()."
        )
    if not len(granules):
        raise ValueError(f"No {product} granules at NSIDC for this AOI and time.")
    paths = providers.earthdata.download(granules, product)

    per_date: dict[pd.Timestamp, list[xr.Dataset]] = {}
    for path in paths:
        when = granule_date(Path(path).name)
        if when is None:  # pragma: no cover — defensive
            _logger.warning("Skipping %s: no date in the granule name.", path)
            continue
        arrays = {}
        for field in fields:
            da = providers.raster_http.open(
                subdataset(path, product, field), None, chunks=chunks, **kwargs
            )
            if parsed is not None and parsed.clip and not parsed.is_global:
                bounds = parsed.to_crs(da.rio.crs).total_bounds
                da = da.rio.clip_box(*bounds, crs=da.rio.crs, auto_expand=True)
            arrays[field] = da
        per_date.setdefault(when, []).append(xr.Dataset(arrays))

    datasets = []
    for when in sorted(per_date):
        tiles = per_date[when]
        merged = tiles[0] if len(tiles) == 1 else _merge_tiles(tiles)
        datasets.append(merged.expand_dims(time=[when]))
    combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    return combined.sortby("time")


def _merge_tiles(tiles: Sequence[xr.Dataset]) -> xr.Dataset:
    """Mosaic same-date MODIS tiles onto one grid."""
    from rioxarray.merge import merge_arrays  # noqa: PLC0415

    fields = list(tiles[0].data_vars)
    merged = {}
    for field in fields:
        arrays = [tile[field] for tile in tiles]
        merged[field] = merge_arrays(arrays)
    return xr.Dataset(merged)
