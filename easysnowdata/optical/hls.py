"""Harmonized Landsat Sentinel-2 (HLS) v2.0.

Sources (companion file §B.3):

``lpcloud-cmr-stac`` (default)
    NASA's CMR-STAC ``LPCLOUD`` catalog: ``HLSL30_2.0`` (Landsat 8/9) and
    ``HLSS30_2.0`` (Sentinel-2), the archive of record, same-day with the LP
    DAAC. The *search* is open; the COG reads are behind Earthdata Login, so
    they go through the ``earthdata`` provider's GDAL environment (netrc plus
    a cookie jar, or a bearer token), which Dask workers inherit.
``planetary-computer``
    ``hls2-l30`` / ``hls2-s30``: a credential-free mirror, ids verified live
    on 2026-09-16 (the plan could not check them during the PC outage).

Scene metadata comes from STAC properties, not from the per-granule XML the
old ``HLS`` class fetched one request at a time::

    import easysnowdata as esd
    items_gdf = esd.optical.hls.search(aoi, "2023-08", cloud_cover=40)
    hls_ds = esd.optical.hls.load(aoi, "2023-08", mask="fmask-default")

``load`` returns one ``Dataset`` with both products stacked on ``time`` and a
``product`` coordinate (``L30``/``S30``) saying where each scene came from.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.auth.planetary_computer import STAC_URL as PC_STAC
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract, masks

__all__ = [
    "PRODUCT",
    "COLLECTIONS",
    "PRODUCTS",
    "DEFAULT_BANDS",
    "ALL_BANDS",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

CMR_LPCLOUD_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

#: The two HLS products: Landsat 8/9 (L30) and Sentinel-2 (S30).
PRODUCTS = ("L30", "S30")

#: catalog id → {product → collection id}
COLLECTIONS: dict[str, dict[str, str]] = {
    "lpcloud-cmr-stac": {"L30": "HLSL30_2.0", "S30": "HLSS30_2.0"},
    "planetary-computer": {"L30": "hls2-l30", "S30": "hls2-s30"},
}
_CATALOG_FOR_SOURCE = {
    "lpcloud-cmr-stac": "cmr-lpcloud",
    "planetary-computer": "planetary-computer",
}

#: Common band name → asset name, per product (HLS User Guide v2.0, table 4).
BAND_ASSETS: dict[str, dict[str, str]] = {
    "L30": {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir08": "B05",
        "swir16": "B06",
        "swir22": "B07",
        "cirrus": "B09",
        "lwir11": "B10",
        "lwir12": "B11",
    },
    "S30": {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "rededge1": "B05",
        "rededge2": "B06",
        "rededge3": "B07",
        "nir": "B08",
        "nir08": "B8A",
        "watervapor": "B09",
        "cirrus": "B10",
        "swir16": "B11",
        "swir22": "B12",
    },
}
#: Bands present in both products (what a two-product load can stack).
COMMON_BANDS = tuple(name for name in BAND_ASSETS["L30"] if name in BAND_ASSETS["S30"])
#: Quality and geometry layers, named the same in both products.
QUALITY_BANDS = ("Fmask", "SZA", "SAA", "VZA", "VAA")

DEFAULT_BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22", "Fmask")
ALL_BANDS = tuple(COMMON_BANDS) + QUALITY_BANDS

REFLECTANCE_NODATA = -9999
REFLECTANCE_SCALE = 1e-4
ANGLE_NODATA = 40000
ANGLE_SCALE = 1e-2
FMASK_NODATA = 255

_LPDAAC_DOCS = "https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf"
_PC_DOCS = "https://planetarycomputer.microsoft.com/dataset/hls2-s30"

_FMASK_VARIABLE = Variable(
    "Fmask",
    dtype="uint8",
    nodata=FMASK_NODATA,
    long_name="Fmask quality bits (cirrus, cloud, adjacent, shadow, snow/ice, water, aerosol)",
)

PRODUCT = Product(
    id="hls",
    theme="optical",
    title="Harmonized Landsat Sentinel-2 (HLS) v2.0",
    description=(
        "NASA HLS L30 (Landsat 8/9) and S30 (Sentinel-2) 30 m surface reflectance "
        "with Fmask and the sun/view angles, harmonized to a common grid and "
        "stacked on one time axis."
    ),
    sources=(
        Source(
            id="lpcloud-cmr-stac",
            provider="stac",
            location="HLSL30_2.0 / HLSS30_2.0",
            requires=("earthdata",),
            resolution_m=30,
            temporal="2013-04/present",
            latency="~2-3 days",
            notes=(
                "the archive of record; search is open, the COG reads need Earthdata "
                "Login (netrc plus a cookie jar, or a bearer token)"
            ),
            title="CMR-STAC LPCLOUD",
            health=Probe(
                "HLS L30 (CMR-STAC LPCLOUD)",
                partial(
                    health.stac_search,
                    CMR_LPCLOUD_STAC,
                    "HLSL30_2.0",
                    datetime_range="2024-07-01T00:00:00Z/2024-07-31T23:59:59Z",
                ),
                requires=(),
            ),
        ),
        Source(
            id="planetary-computer",
            provider="stac",
            location="hls2-l30 / hls2-s30",
            resolution_m=30,
            temporal="2013-04/present",
            latency="mirror; lag unknown",
            notes="credential-free mirror; collection ids verified live 2026-09-16",
            title="Planetary Computer",
            health=Probe(
                "HLS S30 (Planetary Computer)",
                partial(
                    health.stac_search,
                    PC_STAC,
                    "hls2-s30",
                    datetime_range="2024-07-01/2024-07-31",
                    sign=True,
                ),
            ),
        ),
    ),
    variables=(
        Variable("blue", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        Variable("green", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        Variable("red", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        Variable("nir08", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        Variable("swir16", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        Variable("swir22", units="1", dtype="int16", nodata=REFLECTANCE_NODATA),
        _FMASK_VARIABLE,
    ),
    citation=(
        "Masek, J., Ju, J., Roger, J.-C., et al. (2021). HLS Operational Land Imager "
        "Surface Reflectance and TOA Brightness Daily Global 30 m v2.0; HLS Sentinel-2 "
        "MSI Surface Reflectance Daily Global 30 m v2.0. NASA EOSDIS LP DAAC."
    ),
    license="NASA Earthdata (free registration)",
    doi="10.5067/HLS/HLSL30.002",
    references=(_LPDAAC_DOCS, _PC_DOCS),
    loader="easysnowdata.optical.hls.load",
    examples=("optical/plot_hls.py",),
    tags=("optical", "reflectance", "landsat", "sentinel-2", "hls"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _products(products: str | Sequence[str] | None) -> list[str]:
    if products is None:
        return list(PRODUCTS)
    names = [products] if isinstance(products, str) else list(products)
    unknown = [p for p in names if p not in PRODUCTS]
    if unknown:
        raise ValueError(f"products must be in {PRODUCTS}, got {unknown}.")
    return names


def _bands(bands: str | Sequence[str] | None) -> list[str]:
    if bands is None:
        return list(DEFAULT_BANDS)
    return [bands] if isinstance(bands, str) else list(bands)


def _stac_cfg(collection: str, product: str, bands: Sequence[str]) -> dict[str, Any]:
    """Per-collection dtypes, nodata and band aliases.

    Neither catalog publishes ``raster:bands`` for HLS, so the loader supplies
    them: reflectance is ``int16``/−9999, Fmask ``uint8``/255 and the angles
    ``uint16``/40000.
    """
    assets: dict[str, Any] = {
        "*": {"data_type": "int16", "nodata": REFLECTANCE_NODATA, "unit": "1"},
        "Fmask": {"data_type": "uint8", "nodata": FMASK_NODATA, "unit": "1"},
    }
    for angle in ("SZA", "SAA", "VZA", "VAA"):
        assets[angle] = {"data_type": "uint16", "nodata": ANGLE_NODATA, "unit": "deg"}
    aliases = {
        alias: asset
        for alias, asset in BAND_ASSETS[product].items()
        if alias in bands or not bands
    }
    return {collection: {"assets": assets, "aliases": aliases}}


def _asset_names(product: str, bands: Sequence[str]) -> list[str]:
    """Translate common band names to this product's asset names."""
    mapping = BAND_ASSETS[product]
    out = []
    for band in bands:
        if band in QUALITY_BANDS:
            out.append(band)
        elif band in mapping:
            out.append(band)  # odc-stac resolves the alias from stac_cfg
        else:
            _logger.debug("%s has no band %r; skipping it.", product, band)
    return out


def _product_of(item: dict[str, Any], collection: str) -> str:
    """``"L30"`` or ``"S30"`` from the granule id, else from the collection."""
    item_id = str(item.get("id", ""))
    for product in PRODUCTS:
        if f".{product}." in item_id:
            return product
    for product, name in COLLECTIONS["lpcloud-cmr-stac"].items():
        if name == collection:
            return product
    for product, name in COLLECTIONS["planetary-computer"].items():
        if name == collection:
            return product
    return ""


def _platform_of(item: dict[str, Any], product: str) -> str:
    """The satellite, from STAC properties where present, else from the product."""
    platform = (item.get("properties", {}) or {}).get("platform")
    if platform:
        return str(platform)
    return "landsat-8/9" if product == "L30" else "sentinel-2a/b"


def _apply_mask(ds: xr.Dataset, mask: Any) -> xr.Dataset:
    if mask in (None, False):
        return ds
    if "Fmask" not in ds.data_vars:
        raise ValueError(
            "Masking needs the 'Fmask' band; add it to bands= or pass mask=False."
        )
    flags = masks.DEFAULT_FMASK_REMOVE
    aerosol: tuple[str, ...] = ()
    if isinstance(mask, str):
        if mask not in ("fmask-default", "default", "fmask"):
            raise ValueError(
                f"Unknown mask {mask!r}; use True, 'fmask-default', or a list of Fmask flags."
            )
    elif mask is not True:
        flags = tuple(f for f in mask if f in masks.FMASK_BITS)
        aerosol = tuple(f for f in mask if f in masks.FMASK_AEROSOL_LEVELS)
        unknown = set(mask) - set(flags) - set(aerosol)
        if unknown:
            raise ValueError(
                f"Unknown Fmask flags {sorted(unknown)}; known: {list(masks.FMASK_BITS)} "
                f"and aerosol levels {list(masks.FMASK_AEROSOL_LEVELS)}."
            )
    keep_da = masks.fmask_mask(ds["Fmask"], flags, aerosol)
    masked_ds = ds.copy()
    for band in ds.data_vars:
        if band != "Fmask":
            masked_ds[band] = ds[band].where(keep_da)
    masked_ds.attrs["masked_fmask_flags"] = " ".join([*flags, *aerosol])
    _logger.info(
        "Fmask cloud detection is unreliable over snow and ice; masked %s.",
        ", ".join([*flags, *aerosol]),
    )
    return masked_ds


def _scale(ds: xr.Dataset) -> xr.Dataset:
    scaled_ds = ds.copy()
    for band in ds.data_vars:
        if band == "Fmask":
            continue
        scale = ANGLE_SCALE if band in QUALITY_BANDS else REFLECTANCE_SCALE
        scaled_ds[band] = (scaled_ds[band] * scale).assign_attrs(
            {
                **ds[band].attrs,
                "units": "deg" if band in QUALITY_BANDS else "1",
                "scale": scale,
            }
        )
    return scaled_ds


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    products: str | Sequence[str] | None = None,
    cloud_cover: float | None = None,
    query: dict[str, Any] | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search both HLS products and return the items as one ``GeoDataFrame``.

    The frame carries the STAC properties (including ``eo:cloud_cover``) plus
    ``product`` (``L30``/``S30``) and ``platform`` — the metadata the old
    class fetched by scraping one XML file per granule.
    """
    src = resolve_source(PRODUCT, source)
    # No ensure_source here: a CMR-STAC search is open, only the reads need
    # Earthdata Login (which is why the health probe overrides `requires`).
    names = _products(products)
    stac_query = dict(query or {})
    if cloud_cover is not None:
        stac_query["eo:cloud_cover"] = {"lt": float(cloud_cover)}
    frames = []
    for product in names:
        collection = COLLECTIONS[src.id][product]
        items = providers.stac.search(
            _CATALOG_FOR_SOURCE[src.id],
            collection,
            aoi,
            time,
            query=stac_query or None,
            max_items=max_items,
            **kwargs,
        )
        gdf = providers.stac.items_to_geodataframe(items)
        if len(gdf):
            gdf["product"] = product
            gdf["platform"] = [_platform_of(item, product) for item in gdf["stac_item"]]
            frames.append(gdf)
    if not frames:
        empty_gdf = providers.stac.items_to_geodataframe([])
        empty_gdf["product"] = pd.Series(dtype="object")
        empty_gdf["platform"] = pd.Series(dtype="object")
        empty_gdf.attrs = {"source": src.id, "products": names}
        return empty_gdf
    combined_gdf = pd.concat(frames)
    combined_gdf = combined_gdf.sort_values("datetime")
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs="EPSG:4326")
    combined_gdf.attrs = {"source": src.id, "products": names}
    return combined_gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    items: Any = None,
    bands: str | Sequence[str] | None = None,
    source: str | None = None,
    products: str | Sequence[str] | None = None,
    resolution: float | None = 30,
    crs: Any = "utm",
    groupby: str = "solar_day",
    mask: Any = None,
    scale: bool = True,
    mask_nodata: bool = True,
    cloud_cover: float | None = None,
    chunks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load HLS as a lazy ``xarray.Dataset`` (``time``, ``y``, ``x``).

    Both products are loaded and stacked on ``time``, with a ``product``
    coordinate (``L30``/``S30``) and a ``platform`` coordinate.

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    items
        A search result to load instead of searching again.
    bands
        Common band names (:data:`DEFAULT_BANDS` by default). Bands that
        exist in only one product are loaded for that product alone.
    source
        ``"lpcloud-cmr-stac"`` (default, Earthdata Login) or
        ``"planetary-computer"`` (credential-free mirror).
    products
        ``"L30"``, ``"S30"`` or both (default).
    mask
        ``None``/``False`` (default) keeps every pixel; ``True`` or
        ``"fmask-default"`` removes cirrus, cloud, cloud-adjacent and shadow
        pixels; a sequence selects Fmask flags and aerosol levels. Fmask is
        unreliable over snow and ice.
    scale
        Convert reflectance to 0–1 floats and the angles to degrees.
    mask_nodata
        NaN-mask the fill values and keep them as the encoded nodata.
    resolution, crs, groupby, chunks, **kwargs
        Passed to ``odc.stac.load``.
    """
    src = resolve_source(PRODUCT, source)
    band_names = _bands(bands)
    names = _products(products)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if items is None:
        items = search(
            aoi, time, source=src.id, products=names, cloud_cover=cloud_cover
        )
    per_product = _split_items(items)
    if not any(len(v) for v in per_product.values()):
        raise ValueError(
            "No HLS items for this AOI and time; widen the search or raise cloud_cover."
        )

    datasets = []
    for product, product_items in per_product.items():
        if not len(product_items):
            continue
        collection = COLLECTIONS[src.id][product]
        ds = providers.stac.load(
            product_items,
            parsed,
            bands=_asset_names(product, band_names),
            resolution=resolution,
            crs=crs,
            chunks=chunks,
            groupby=groupby,
            stac_cfg=_stac_cfg(collection, product, band_names),
            catalog=_CATALOG_FOR_SOURCE[src.id],
            # odc-stac's default raises on a failed read. The 0.2 loader passed
            # fail_on_error=False, which turned a 401 on every band into a
            # silently blank Dataset (the first full docs build on main).
            **kwargs,
        )
        if "time" in ds.dims:
            ds = ds.assign_coords(
                product=("time", np.full(ds.sizes["time"], product)),
                platform=(
                    "time",
                    np.full(ds.sizes["time"], _platform_of({}, product)),
                ),
            )
        datasets.append(ds)

    combined_ds = (
        datasets[0]
        if len(datasets) == 1
        else xr.concat(
            datasets, dim="time", join="outer", combine_attrs="drop_conflicts"
        )
    )
    combined_ds = combined_ds.sortby("time")
    combined_ds = contract.apply_variables(
        combined_ds,
        [v for v in PRODUCT.variables if v.name in combined_ds.data_vars],
        mask=False,
    )
    if mask_nodata:
        for band in list(combined_ds.data_vars):
            if band == "Fmask":
                continue
            nodata = ANGLE_NODATA if band in QUALITY_BANDS else REFLECTANCE_NODATA
            combined_ds[band] = contract.mask_continuous(combined_ds[band], nodata)
    if scale:
        combined_ds = _scale(combined_ds)
    combined_ds = _apply_mask(combined_ds, mask)
    combined_ds = contract.finalize(
        combined_ds,
        PRODUCT,
        src,
        variables=(),
        source_url=_LPDAAC_DOCS if src.id == "lpcloud-cmr-stac" else _PC_DOCS,
        attrs={
            "collections": " ".join(COLLECTIONS[src.id][p] for p in names),
            "scaled_to_reflectance": str(bool(scale)),
        },
    )
    return combined_ds


def _split_items(items: Any) -> dict[str, Any]:
    """Group a search result (frame or item collection) by HLS product."""
    if isinstance(items, gpd.GeoDataFrame):
        if "product" in items.columns:
            return {product: items[items["product"] == product] for product in PRODUCTS}
        dicts = list(items["stac_item"]) if "stac_item" in items.columns else []
    else:
        dicts = [
            item.to_dict(transform_hrefs=False) if hasattr(item, "to_dict") else item
            for item in items
        ]
    frame_gdf = providers.stac.items_to_geodataframe(dicts) if dicts else None
    if frame_gdf is None or not len(frame_gdf):
        return {product: [] for product in PRODUCTS}
    frame_gdf["product"] = [
        _product_of(item, str(item.get("collection", "")))
        for item in frame_gdf["stac_item"]
    ]
    return {product: frame_gdf[frame_gdf["product"] == product] for product in PRODUCTS}
