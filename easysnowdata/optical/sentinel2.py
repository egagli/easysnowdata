"""Sentinel-2 Level-2A surface reflectance.

Sources (companion file §B.2):

``planetary-computer`` (default)
    ``sentinel-2-l2a`` on Microsoft Planetary Computer: credential-free
    (hrefs are signed for you), one catalog for Sentinel-1, Sentinel-2, the
    DEM and WorldCover. Its items historically carry no ``raster:bands``, so
    the band aliases and dtypes come from a ``stac_cfg`` fallback and the
    post-2022-01-25 baseline offset is undone by the date rule.
``earth-search``
    Element 84's ``sentinel-2-l2a``, ``sentinel-2-c1-l2a`` and
    ``sentinel-2-pre-c1-l2a`` on AWS: unsigned S3, assets already named by
    common band name, and items that carry ``raster:bands`` with ``scale``
    and ``offset`` — so the baseline offset is applied from metadata rather
    than from a date.

::

    import easysnowdata as esd
    items = esd.optical.sentinel2.search(aoi, "2024-05", cloud_cover=30)
    s2 = esd.optical.sentinel2.load(aoi, "2024-05", mask="scl-default")
    ndsi = esd.processing.ndsi(s2)

``load`` returns reflectance (scaled, harmonized to the pre-2022 baseline)
with dims ``time``, ``y``, ``x`` in the AOI's UTM zone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.auth.planetary_computer import STAC_URL as PC_STAC
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract, masks
from easysnowdata.processing import optical as optical_processing

__all__ = [
    "PRODUCT",
    "COLLECTIONS",
    "DEFAULT_BANDS",
    "ALL_BANDS",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1"

#: catalog id → (default collection, every collection it serves)
COLLECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "planetary-computer": ("sentinel-2-l2a", ("sentinel-2-l2a",)),
    "earth-search": (
        "sentinel-2-l2a",
        ("sentinel-2-l2a", "sentinel-2-c1-l2a", "sentinel-2-pre-c1-l2a"),
    ),
}

#: What ``load`` reads when ``bands=None``: the snow-relevant reflectance bands
#: plus the scene classification layer that drives masking.
DEFAULT_BANDS = ("blue", "green", "red", "nir", "swir16", "swir22", "scl")
#: Every band the old ``Sentinel2`` class loaded (used by the deprecation shim).
ALL_BANDS = (
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09", "swir16", "swir22", "aot", "scl", "wvp",
)  # fmt: skip

#: Bands that carry reflectance (scaled by 1e-4 and baseline-offset).
REFLECTANCE_BANDS = frozenset(
    {
        "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
        "nir", "nir08", "nir09", "swir16", "swir22",
    }
)  # fmt: skip
_REFLECTANCE_SCALE = 1e-4
_BASELINE_OFFSET_DN = 1000

_PC_DOCS = "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a"
_EARTH_SEARCH_DOCS = "https://element84.com/earth-search/"

_SCL_VARIABLE = Variable(
    "scl",
    dtype="uint8",
    nodata=0,
    long_name="scene classification layer",
    flag_values=tuple(optical_processing.SCL_CLASSES),
    flag_meanings=tuple(name for name, _ in optical_processing.SCL_CLASSES.values()),
    flag_colors=tuple(color for _, color in optical_processing.SCL_CLASSES.values()),
)

PRODUCT = Product(
    id="sentinel-2-l2a",
    theme="optical",
    title="Sentinel-2 Level-2A surface reflectance",
    description=(
        "Sentinel-2 MSI bottom-of-atmosphere reflectance with the Sen2Cor scene "
        "classification layer, 10–60 m, from Planetary Computer or Earth Search."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="sentinel-2-l2a",
            resolution_m=10,
            temporal="2015-06/present",
            latency="~1-2 days",
            notes=(
                "raw ESA values; the post-2022-01-25 baseline offset is undone by the "
                "date rule and the band aliases come from the stac_cfg fallback"
            ),
            title="Planetary Computer",
            health=(
                Probe(
                    "Sentinel-2 L2A (Planetary Computer)",
                    partial(
                        health.stac_search,
                        PC_STAC,
                        "sentinel-2-l2a",
                        datetime_range="2024-07-01/2024-07-31",
                        sign=True,
                    ),
                ),
                Probe(
                    "Sentinel-2 L2A latency (Planetary Computer)",
                    partial(health.stac_latest, PC_STAC, "sentinel-2-l2a"),
                    kind="latency",
                ),
            ),
        ),
        Source(
            id="earth-search",
            provider="stac",
            location="sentinel-2-l2a / sentinel-2-c1-l2a / sentinel-2-pre-c1-l2a",
            resolution_m=10,
            temporal="2015-06/present",
            latency="~1 day",
            notes=(
                "items carry raster:bands scale and offset, so the baseline offset "
                "is applied from metadata; unsigned S3 reads"
            ),
            title="Earth Search (AWS)",
            health=Probe(
                "Sentinel-2 L2A (Earth Search)",
                partial(
                    health.stac_search,
                    EARTH_SEARCH_STAC,
                    "sentinel-2-l2a",
                    datetime_range="2024-07-01/2024-07-31",
                ),
            ),
        ),
    ),
    variables=(
        Variable("blue", units="1", dtype="uint16", nodata=0, long_name="blue (B02)"),
        Variable("green", units="1", dtype="uint16", nodata=0, long_name="green (B03)"),
        Variable("red", units="1", dtype="uint16", nodata=0, long_name="red (B04)"),
        Variable("nir", units="1", dtype="uint16", nodata=0, long_name="NIR (B08)"),
        Variable(
            "swir16", units="1", dtype="uint16", nodata=0, long_name="SWIR 1.6 µm (B11)"
        ),
        Variable(
            "swir22", units="1", dtype="uint16", nodata=0, long_name="SWIR 2.2 µm (B12)"
        ),
        _SCL_VARIABLE,
    ),
    citation="European Space Agency. Copernicus Sentinel-2 MSI Level-2A products.",
    license="Copernicus Sentinel data licence",
    references=(_PC_DOCS, _EARTH_SEARCH_DOCS),
    loader="easysnowdata.optical.sentinel2.load",
    examples=(
        "optical/plot_sentinel2.py",
        "optical/plot_planetscope.py",
    ),
    tags=("optical", "reflectance", "ndsi", "sentinel-2"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _catalog_and_collection(
    source: str | Source | None, collection: str | None
) -> tuple[Source, str]:
    src = resolve_source(PRODUCT, source)
    default, known = COLLECTIONS[src.id]
    if collection is None:
        return src, default
    if collection not in known:
        raise ValueError(
            f"{src.id} serves {known}, not {collection!r}. "
            "Collection-1 products are on Earth Search only."
        )
    return src, collection


#: Band aliases and dtypes for catalogs whose items lack ``raster:bands``.
#:
#: Planetary Computer names its assets ``B02``/``SCL``, so the per-asset
#: entries are keyed by the *asset* name (the legacy ``utils.get_stac_cfg``
#: keyed them by the alias, which silently left SCL at the ``'*'`` dtype of
#: ``uint16``). ``coastal`` is spelled both ways: the old config had ``costal``.
_PC_ASSET_ALIASES: dict[str, str] = {
    "coastal": "B01",
    "costal": "B01",  # the legacy spelling, kept working
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
    "aot": "AOT",
    "wvp": "WVP",
}
_PC_STAC_CFG: dict[str, Any] = {
    "*": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0, "unit": "1"},
            "SCL": {"data_type": "uint8", "nodata": 0, "unit": "1"},
            "visual": {"data_type": "uint8", "nodata": 0, "unit": "1"},
        },
        "aliases": _PC_ASSET_ALIASES,
    }
}


def _stac_cfg(src: Source, collection: str) -> dict[str, Any] | None:
    """The band-alias fallback, needed only where items lack ``raster:bands``."""
    if src.id != "planetary-computer":
        return None
    return {collection: _PC_STAC_CFG["*"]}


def _bands(bands: str | Sequence[str] | None) -> list[str]:
    if bands is None:
        return list(DEFAULT_BANDS)
    return [bands] if isinstance(bands, str) else list(bands)


def _asset_scale_offset(item: Any, band: str) -> tuple[float | None, float | None]:
    """``(scale, offset)`` from an item's ``raster:bands``, if it has any."""
    assets = item.get("assets", {}) if isinstance(item, dict) else {}
    asset = assets.get(band)
    if not asset:
        return None, None
    raster = asset.get("raster:bands") or asset.get("bands") or []
    if not raster:
        return None, None
    entry = raster[0]
    return entry.get("scale"), entry.get("offset")


def _offset_days(items: Sequence[Any]) -> set[pd.Timestamp] | None:
    """Days whose pixels still carry the ESA baseline offset, per item metadata.

    ``None`` when no item carries the metadata at all (Planetary Computer),
    which is the signal to fall back on the acquisition-date rule.

    Two signals are read, in this order:

    ``earthsearch:boa_offset_applied``
        ``True`` means Element 84 already removed the ``+1000`` offset from
        the pixels, so the day needs nothing — even though the same item still
        advertises ``offset: -0.1`` in ``raster:bands``. Verified 2026-09-16
        on tile 10TES, 2023-08-10: Earth Search ``sentinel-2-l2a`` reads
        1025 DN below Planetary Computer for the same pixels, while
        ``sentinel-2-c1-l2a`` matches it to within 26 DN. Applying the
        metadata offset there would subtract the offset twice.
    ``raster:bands.offset``
        A non-zero offset on a reflectance asset means the pixels are raw ESA
        values (Collection-1 on Earth Search).
    """
    days: set[pd.Timestamp] = set()
    seen_metadata = False
    for item in items:
        properties = item.get("properties", {}) if isinstance(item, dict) else {}
        when = properties.get("datetime") or properties.get("start_datetime")
        offsets = [
            _asset_scale_offset(item, band)[1]
            for band in item.get("assets", {})
            if band in REFLECTANCE_BANDS
        ]
        offsets = [o for o in offsets if o is not None]
        if not offsets:
            continue
        seen_metadata = True
        if properties.get("earthsearch:boa_offset_applied") is True:
            continue  # the pixels are already on the old baseline
        if when is not None and any(abs(float(o)) > 1e-9 for o in offsets):
            days.add(pd.Timestamp(when).tz_localize(None).normalize())
    return days if seen_metadata else None


def _harmonize(
    ds: xr.Dataset, items: Sequence[Any], bands: Sequence[str]
) -> xr.Dataset:
    """Put every scene on the pre-2022 baseline (metadata first, date rule second)."""
    names = [b for b in bands if b in ds.data_vars and b in REFLECTANCE_BANDS]
    if not names or "time" not in ds.dims:
        return ds
    days = _offset_days(items)
    if days is None:
        return optical_processing.harmonize_s2_baseline(ds, bands=names)
    if not days:
        return ds
    times = pd.DatetimeIndex(ds["time"].values).normalize()
    where = xr.DataArray(
        times.isin(sorted(days)), dims="time", coords={"time": ds["time"]}
    )
    out = ds.copy()
    for name in names:
        shifted = out[name].clip(min=_BASELINE_OFFSET_DN) - _BASELINE_OFFSET_DN
        out[name] = xr.where(where, shifted, out[name], keep_attrs=True).transpose(
            *out[name].dims
        )
    return out


def _scale(ds: xr.Dataset, items: Sequence[Any], bands: Sequence[str]) -> xr.Dataset:
    """Scale the reflectance bands to 0–1 floats (metadata scale where present)."""
    scales: dict[str, float] = {}
    for band in bands:
        if band not in ds.data_vars or band not in REFLECTANCE_BANDS:
            continue
        found = next(
            (
                s
                for s in (_asset_scale_offset(item, band)[0] for item in items)
                if s is not None
            ),
            None,
        )
        scales[band] = float(found) if found else _REFLECTANCE_SCALE
    if not scales:
        return ds
    out = ds.copy()
    for band, scale in scales.items():
        # Nodata (0) is already NaN at this point; keep the attrs.
        out[band] = (out[band] * scale).assign_attrs(
            {**ds[band].attrs, "units": "1", "scale": scale}
        )
    return out


def _apply_mask(ds: xr.Dataset, mask: Any, bands: Sequence[str]) -> xr.Dataset:
    if mask in (None, False):
        return ds
    if "scl" not in ds.data_vars:
        raise ValueError(
            "Masking needs the 'scl' band; add it to bands= or pass mask=False."
        )
    classes = masks.DEFAULT_SCL_REMOVE
    if isinstance(mask, str):
        if mask not in ("scl-default", "default", "scl"):
            raise ValueError(
                f"Unknown mask {mask!r}; use True, 'scl-default', or a list of SCL classes."
            )
    elif mask is not True:
        classes = tuple(mask)
    data_bands = [b for b in ds.data_vars if b != "scl"]
    masked = ds.copy()
    keep = masks.scl_mask(ds["scl"], classes)
    for band in data_bands:
        masked[band] = ds[band].where(keep)
    masked.attrs["masked_scl_classes"] = " ".join(str(c) for c in classes)
    return masked


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    collection: str | None = None,
    cloud_cover: float | None = None,
    query: dict[str, Any] | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search one of the catalogs and return the items as a ``GeoDataFrame``.

    The frame carries every STAC property as a column plus ``stac_item``, so
    it can be filtered and handed straight to :func:`load`.

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    source
        ``"planetary-computer"`` (default) or ``"earth-search"``.
    collection
        A collection of that catalog (Earth Search also has
        ``"sentinel-2-c1-l2a"`` and ``"sentinel-2-pre-c1-l2a"``).
    cloud_cover
        Keep scenes with ``eo:cloud_cover`` below this percentage.
    query, max_items, **kwargs
        Passed to :func:`easysnowdata.providers.stac.search`.
    """
    src, collection_id = _catalog_and_collection(source, collection)
    ensure_source(PRODUCT, src)
    stac_query = dict(query or {})
    if cloud_cover is not None:
        stac_query["eo:cloud_cover"] = {"lt": float(cloud_cover)}
    items = providers.stac.search(
        src.id,
        collection_id,
        aoi,
        time,
        query=stac_query or None,
        max_items=max_items,
        **kwargs,
    )
    gdf = providers.stac.items_to_geodataframe(items)
    gdf.attrs = {"source": src.id, "collection": collection_id}
    return gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    items: Any = None,
    bands: str | Sequence[str] | None = None,
    source: str | None = None,
    collection: str | None = None,
    resolution: float | None = None,
    crs: Any = "utm",
    groupby: str = "solar_day",
    mask: Any = None,
    harmonize: bool = True,
    scale: bool = True,
    mask_nodata: bool = True,
    cloud_cover: float | None = None,
    chunks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load Sentinel-2 L2A as a lazy ``xarray.Dataset`` (``time``, ``y``, ``x``).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs. ``clip=False`` on the AOI
        returns the covering tiles instead of the cut grid.
    items
        A search result (``GeoDataFrame`` or ``ItemCollection``) to load
        instead of searching again.
    bands
        Common band names (:data:`DEFAULT_BANDS` by default). ``"scl"`` is
        needed for masking.
    source, collection
        Which catalog and collection to read (see :func:`search`).
    resolution, crs, groupby, chunks, **kwargs
        Passed to ``odc.stac.load``; ``crs="utm"`` (default) picks the AOI's
        UTM zone.
    mask
        ``None``/``False`` (default) keeps every pixel; ``True`` or
        ``"scl-default"`` removes the legacy default SCL classes (no data,
        saturated, shadows, clouds, cirrus); a sequence selects classes by
        name or value.
    harmonize
        Put post-2022-01-25 scenes back on the old baseline: from the items'
        ``raster:bands`` offsets where the catalog provides them, otherwise
        from the acquisition date (Planetary Computer).
    scale
        Convert the reflectance bands to 0–1 floats.
    mask_nodata
        NaN-mask the reflectance nodata value (0) and keep it as the encoded
        nodata. ``False`` returns the raw integers with ``rio.nodata`` set.
    cloud_cover
        Passed to :func:`search` when *items* is not given.
    """
    src, collection_id = _catalog_and_collection(source, collection)
    band_names = _bands(bands)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if items is None:
        items = search(
            aoi,
            time,
            source=src.id,
            collection=collection_id,
            cloud_cover=cloud_cover,
        )
    item_dicts = _item_dicts(items)
    if not item_dicts:
        raise ValueError(
            "No Sentinel-2 items for this AOI and time; widen the search or raise "
            "cloud_cover."
        )

    ds = providers.stac.load(
        items,
        parsed,
        bands=band_names,
        resolution=resolution,
        crs=crs,
        chunks=chunks,
        groupby=groupby,
        stac_cfg=_stac_cfg(src, collection_id),
        catalog=src.id,
        **kwargs,
    )
    ds = contract.apply_variables(
        ds, [v for v in PRODUCT.variables if v.name in ds.data_vars], mask=False
    )
    if harmonize:
        ds = _harmonize(ds, item_dicts, band_names)
    # Nodata → NaN for the reflectance bands; the SCL sentinel (0) is kept.
    if mask_nodata:
        for band in list(ds.data_vars):
            if band in REFLECTANCE_BANDS:
                ds[band] = contract.mask_continuous(ds[band], 0)
    if scale:
        ds = _scale(ds, item_dicts, band_names)
    ds = _apply_mask(ds, mask, band_names)
    ds = contract.finalize(
        ds,
        PRODUCT,
        src,
        variables=(),
        source_url=_PC_DOCS if src.id == "planetary-computer" else _EARTH_SEARCH_DOCS,
        attrs={
            "collection": collection_id,
            "harmonized_to_old_baseline": str(bool(harmonize)),
            "scaled_to_reflectance": str(bool(scale)),
        },
    )
    return ds


def _item_dicts(items: Any) -> list[dict[str, Any]]:
    """The items as plain dicts, whatever form they arrived in."""
    if isinstance(items, gpd.GeoDataFrame):
        return list(items["stac_item"]) if "stac_item" in items.columns else []
    out = []
    for item in items:
        out.append(
            item.to_dict(transform_hrefs=False) if hasattr(item, "to_dict") else item
        )
    return out
