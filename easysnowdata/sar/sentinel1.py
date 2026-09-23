"""Sentinel-1 radiometrically terrain-corrected (RTC) backscatter.

Sources (companion file §B.1):

``planetary-computer`` (default)
    ``sentinel-1-rtc``: 10 m gamma0 from 2014-10, credential-free, scene
    based. No incidence-angle layer.
``opera-rtc-s1``
    OPERA RTC-S1 on ASF via CMR-STAC ``cloudstac/ASF``: 30 m, burst based,
    2016-04 onward, one single-band COG per polarization plus a layover and
    shadow mask. Earthdata Login. Its companion static product carries the
    local incidence angle.
``gee``
    ``OPERA/RTC/L2_V1/S1`` re-served by Earth Engine, without the static
    layers.

Local incidence angle has its own three routes, and the credential-free one
is a computation rather than a download. The angle belongs to one acquisition
geometry, so the routes never fuse tracks: without ``relative_orbit=`` the
result carries a ``relative_orbit`` dimension with one raster per track that
crosses the AOI::

    import easysnowdata as esd
    s1_ds = esd.sar.sentinel1.load(aoi, "2023-10/2024-06", units="dB")
    lia_ds = esd.sar.sentinel1.local_incidence_angle(aoi)  # every track
    lia_ds = esd.sar.sentinel1.local_incidence_angle(aoi, relative_orbit=137)  # one
    lia_ds = esd.sar.sentinel1.local_incidence_angle(aoi, source="dem")  # no account
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import partial
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.auth.planetary_computer import STAC_URL as PC_STAC
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.processing import sar as sar_processing

__all__ = [
    "PRODUCT",
    "LIA_PRODUCT",
    "OPERA_COLLECTION",
    "OPERA_STATIC_COLLECTION",
    "OPERA_MASK_CLASSES",
    "IW_INCIDENCE_RANGE",
    "search",
    "load",
    "local_incidence_angle",
    "scene_geometry",
    "track_geometries",
    "incidence_angle_field",
]

_logger = logging.getLogger(__name__)

OPERA_COLLECTION = "OPERA_L2_RTC-S1_V1_1"
OPERA_STATIC_COLLECTION = "OPERA_L2_RTC-S1-STATIC_V1_1"
GEE_COLLECTION = "OPERA/RTC/L2_V1/S1"
COPERNICUS_DEM_COLLECTION = "cop-dem-glo-30"

_PC_DOCS = "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc"
_OPERA_DOCS = "https://www.jpl.nasa.gov/go/opera/products/rtc-product/"
_GEE_DOCS = (
    "https://developers.google.com/earth-engine/datasets/catalog/OPERA_RTC_L2_V1_S1"
)

PC_NODATA = -32768
OPERA_NODATA = float("nan")

#: OPERA RTC-S1 layover/shadow mask classes (the ``_mask.tif`` layer).
OPERA_MASK_CLASSES: dict[int, tuple[str, str]] = {
    0: ("not_layover_shadow_or_water", "#d9d9d9"),
    1: ("shadow", "#253494"),
    2: ("layover", "#e31a1c"),
    3: ("layover_and_shadow", "#6a3d9a"),
    255: ("fill", "#ffffff"),
}

_VV = Variable(
    "vv", units="linear power", dtype="float32", nodata=PC_NODATA, long_name="gamma0 VV"
)
_VH = Variable(
    "vh", units="linear power", dtype="float32", nodata=PC_NODATA, long_name="gamma0 VH"
)

PRODUCT = Product(
    id="sentinel-1-rtc",
    theme="sar",
    title="Sentinel-1 radiometrically terrain-corrected backscatter",
    description=(
        "Sentinel-1 C-band gamma0 backscatter (VV, VH) terrain-corrected with the "
        "Copernicus GLO-30 DEM: 10 m scene-based from Planetary Computer (2014-10 "
        "onward) or 30 m burst-based OPERA RTC-S1 from ASF (2016-04 onward, with "
        "layover/shadow masks and static incidence-angle layers)."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="sentinel-1-rtc",
            resolution_m=10,
            temporal="2014-10/present",
            latency="~2 days",
            notes="scene based, Catalyst processing; no incidence-angle layer",
            title="Planetary Computer",
            health=(
                Probe(
                    "Sentinel-1 RTC (Planetary Computer)",
                    partial(
                        health.stac_search,
                        PC_STAC,
                        "sentinel-1-rtc",
                        datetime_range="2024-07-01/2024-07-31",
                        sign=True,
                    ),
                ),
                Probe(
                    "Sentinel-1 RTC latency (Planetary Computer)",
                    partial(health.stac_latest, PC_STAC, "sentinel-1-rtc"),
                    kind="latency",
                ),
            ),
        ),
        Source(
            id="opera-rtc-s1",
            provider="stac",
            location=OPERA_COLLECTION,
            requires=("earthdata",),
            resolution_m=30,
            temporal="2016-04/present",
            latency="~1-2 days",
            notes=(
                "burst based, JPL OPERA; single-band COGs per polarization plus a "
                "layover/shadow mask. CMR-STAC cloudstac/ASF, not stac.asf.alaska.edu. "
                "ASF's datapool refuses a bearer token alone (EARTHDATA_TOKEN); the "
                "reads need a netrc entry or EARTHDATA_USERNAME/EARTHDATA_PASSWORD"
            ),
            title="OPERA RTC-S1 (ASF)",
            health=Probe(
                "Sentinel-1 RTC OPERA (CMR-STAC ASF)",
                partial(
                    health.stac_search,
                    "https://cmr.earthdata.nasa.gov/cloudstac/ASF",
                    OPERA_COLLECTION,
                    datetime_range="2024-07-01T00:00:00Z/2024-07-31T23:59:59Z",
                ),
                requires=(),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location=GEE_COLLECTION,
            requires=("earthengine",),
            resolution_m=30,
            temporal="2016-04/present",
            notes="the OPERA product re-served by Earth Engine; no static layers",
            title="Earth Engine (OPERA)",
            health=Probe(
                "Sentinel-1 RTC OPERA (Earth Engine)",
                partial(
                    health.gee_asset,
                    GEE_COLLECTION,
                    start="2024-07-01",
                    end="2024-07-31",
                ),
            ),
        ),
    ),
    variables=(_VV, _VH),
    citation=(
        "European Space Agency. Copernicus Sentinel-1 GRD. RTC processing by Catalyst "
        "for Microsoft Planetary Computer; OPERA RTC-S1 by NASA/JPL, distributed by ASF DAAC."
    ),
    license="Copernicus Sentinel data licence",
    references=(_PC_DOCS, _OPERA_DOCS, _GEE_DOCS),
    loader="easysnowdata.sar.sentinel1.load",
    examples=("sar/plot_sentinel1.py",),
    tags=("sar", "backscatter", "gamma0", "sentinel-1"),
)
catalog.register(PRODUCT, replace=True)

LIA_PRODUCT = Product(
    id="sentinel-1-local-incidence-angle",
    theme="sar",
    title="Sentinel-1 local incidence angle and layover/shadow mask",
    description=(
        "The angle between the radar look vector and the terrain normal, with the "
        "layover and shadow mask: read from the OPERA RTC-S1-STATIC per-burst "
        "layers, or computed from any DEM with no account at all (issue #10). One "
        "raster per relative orbit; tracks are never fused."
    ),
    sources=(
        Source(
            id="opera-static",
            provider="stac",
            location=OPERA_STATIC_COLLECTION,
            requires=("earthdata",),
            resolution_m=30,
            temporal="static, one granule per burst",
            notes=(
                "ships local_incidence_angle, incidence_angle, mask (layover/shadow), "
                "number_of_looks and the gamma0→beta0/sigma0 factors as COGs. ASF's "
                "datapool refuses a bearer token alone (EARTHDATA_TOKEN); the reads "
                "need a netrc entry or EARTHDATA_USERNAME/EARTHDATA_PASSWORD"
            ),
            title="OPERA RTC-S1-STATIC (ASF)",
            health=Probe(
                "Sentinel-1 static layers (CMR-STAC ASF)",
                partial(
                    health.stac_search,
                    "https://cmr.earthdata.nasa.gov/cloudstac/ASF",
                    OPERA_STATIC_COLLECTION,
                    datetime_range=None,
                ),
                requires=(),
            ),
        ),
        Source(
            id="dem",
            provider="stac",
            location=COPERNICUS_DEM_COLLECTION,
            resolution_m=30,
            temporal="static (DEM epoch)",
            notes=(
                "an approximation: processing.sar.local_incidence_angle on the "
                "Copernicus GLO-30 DEM, with the track's heading and a linear "
                "across-swath incidence field read from a Planetary Computer RTC "
                "scene footprint; no credentials, no layover/shadow mask. For the "
                "exact geometry without OPERA, see "
                "github.com/egagli/generate_sentinel1_local_incidence_angle_maps"
            ),
            title="computed from the Copernicus DEM",
            health=Probe(
                "Copernicus DEM for the incidence angle (Planetary Computer)",
                partial(
                    health.stac_search, PC_STAC, COPERNICUS_DEM_COLLECTION, sign=True
                ),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location="COPERNICUS/S1_GRD angle band + COPERNICUS/DEM/GLO30",
            requires=("earthengine",),
            resolution_m=30,
            temporal="2014-10/present",
            notes=(
                "the S1_GRD angle band is the real per-pixel ellipsoidal incidence "
                "angle of the chosen track; the terrain part is the same DEM "
                "computation as source='dem'"
            ),
            title="Earth Engine (S1_GRD angle band)",
            health=Probe(
                "Sentinel-1 GRD angle band (Earth Engine)",
                partial(
                    health.gee_asset,
                    "COPERNICUS/S1_GRD",
                    start="2024-07-01",
                    end="2024-07-31",
                ),
            ),
        ),
    ),
    variables=(
        Variable(
            "local_incidence_angle",
            units="degrees",
            dtype="float32",
            long_name="local incidence angle",
        ),
        Variable(
            "incidence_angle",
            units="degrees",
            dtype="float32",
            long_name="ellipsoidal incidence angle",
        ),
        Variable(
            "mask",
            dtype="uint8",
            nodata=255,
            long_name="layover and shadow mask",
            flag_values=tuple(OPERA_MASK_CLASSES),
            flag_meanings=tuple(name for name, _ in OPERA_MASK_CLASSES.values()),
            flag_colors=tuple(color for _, color in OPERA_MASK_CLASSES.values()),
        ),
    ),
    citation=(
        "OPERA RTC-S1-STATIC, NASA/JPL, distributed by ASF DAAC; European Space Agency, "
        "Sinergise (2021), Copernicus Global Digital Elevation Model."
    ),
    license="NASA Earthdata (free registration) / Copernicus DEM licence",
    references=(_OPERA_DOCS, "https://gis.stackexchange.com/a/352658"),
    loader="easysnowdata.sar.sentinel1.local_incidence_angle",
    examples=("sar/plot_sentinel1.py",),
    tags=("sar", "incidence angle", "layover", "shadow", "terrain"),
)
catalog.register(LIA_PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _polarisations(bands: str | Sequence[str] | None) -> list[str]:
    if bands is None:
        return ["vv", "vh"]
    names = [bands] if isinstance(bands, str) else list(bands)
    return [str(b).lower() for b in names]


def _opera_assets(polarisations: Sequence[str], mask: bool) -> list[str]:
    """OPERA asset keys for the requested polarizations (``0_VV``, ``0_mask``…)."""
    assets = [f"0_{p.upper()}" for p in polarisations]
    if mask:
        assets.append("0_mask")
    return assets


def _rename_opera(ds: xr.Dataset) -> xr.Dataset:
    """``0_VV`` → ``vv``, ``0_mask`` → ``mask``."""
    mapping = {
        name: name.split("_", 1)[1].lower()
        for name in ds.data_vars
        if name.startswith("0_")
    }
    return ds.rename(mapping) if mapping else ds


def _group_by_track(items: Sequence[Any]) -> dict[int, list[Any]]:
    """OPERA items keyed by the relative orbit in their burst id (``_T137-``).

    Items whose id carries no track are kept together under ``-1`` and
    logged; in practice every OPERA RTC-S1 granule names its track.
    """
    groups: dict[int, list[Any]] = {}
    for item in items:
        track = _track_of(_item_id(item))
        groups.setdefault(-1 if track is None else track, []).append(item)
    if -1 in groups:
        _logger.warning(
            "%d OPERA item(s) carry no track in their id; kept as relative orbit -1.",
            len(groups[-1]),
        )
    return dict(sorted(groups.items()))


def _align_to(reference_ds: xr.Dataset, other_ds: xr.Dataset) -> xr.Dataset:
    """*other_ds* on *reference_ds*'s grid (they are loaded on the same geobox,
    so this is a no-op unless a float coordinate differs in the last place)."""
    if all(
        other_ds[d].equals(reference_ds[d]) for d in ("x", "y") if d in other_ds.dims
    ):
        return other_ds
    return other_ds.reindex_like(reference_ds[["x", "y"]], method="nearest")


def _as_items(items: Sequence[Any]) -> list[Any]:
    """STAC item dicts (from ``search``'s GeoDataFrame) back to ``pystac.Item``
    for odc-stac; anything that is not a full item dict is passed through."""
    import pystac  # noqa: PLC0415

    out = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "Feature":
            out.append(pystac.Item.from_dict(item))
        else:
            out.append(item)
    return out


def _load_opera_by_track(
    items: Sequence[Any],
    parsed: Any,
    *,
    bands: Sequence[str],
    resolution: float | None,
    crs: Any,
    chunks: Any,
    groupby: Any,
    **kwargs: Any,
) -> xr.Dataset:
    """OPERA RTC-S1 bursts loaded one track at a time and stacked along ``time``.

    Grouping by solar day alone would fuse bursts of two tracks that image
    the AOI on the same day (an ascending evening pass and a descending
    morning pass, or two adjacent tracks at a swath edge) into one raster
    with two geometries. Each track is loaded on its own and tagged with
    ``sat:relative_orbit`` before the pieces are concatenated in time order.
    """
    parts: list[xr.Dataset] = []
    for track, group in _group_by_track(items).items():
        part_ds = providers.stac.load(
            _as_items(group),
            parsed,
            bands=list(bands),
            resolution=resolution,
            crs=crs,
            chunks=chunks,
            groupby=groupby,
            catalog="cmr-asf",
            **kwargs,
        )
        part_ds = _rename_opera(part_ds)
        if "time" in part_ds.dims:
            part_ds = part_ds.assign_coords(
                {
                    "sat:relative_orbit": (
                        "time",
                        np.full(part_ds.sizes["time"], track, dtype="int16"),
                    )
                }
            )
        parts.append(part_ds)
    if len(parts) == 1:
        ds = parts[0]
    else:
        parts = [parts[0], *(_align_to(parts[0], p_ds) for p_ds in parts[1:])]
        ds = xr.concat(parts, dim="time", combine_attrs="override")
    return ds.sortby("time") if "time" in ds.dims else ds


def _add_orbit_coords(ds: xr.Dataset, items: Sequence[dict[str, Any]]) -> xr.Dataset:
    """Attach ``sat:orbit_state`` and ``sat:relative_orbit`` along ``time``."""
    if "time" not in ds.dims or not items:
        return ds
    rows = []
    for item in items:
        properties = item.get("properties", {}) if isinstance(item, dict) else {}
        when = properties.get("datetime") or properties.get("start_datetime")
        rows.append(
            {
                "time": pd.Timestamp(when).tz_localize(None) if when else pd.NaT,
                "orbit_state": properties.get("sat:orbit_state"),
                "relative_orbit": properties.get("sat:relative_orbit"),
                "absolute_orbit": properties.get("sat:absolute_orbit"),
            }
        )
    frame_df = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time")
    if frame_df.empty:
        return ds
    times = pd.DatetimeIndex(ds["time"].values)
    matched_df = frame_df.set_index("time").reindex(
        times, method="nearest", tolerance=pd.Timedelta("1D")
    )
    coords = {}
    if matched_df["orbit_state"].notna().any():
        coords["sat:orbit_state"] = (
            "time",
            matched_df["orbit_state"].astype("object").values,
        )
    if matched_df["relative_orbit"].notna().any():
        coords["sat:relative_orbit"] = (
            "time",
            matched_df["relative_orbit"].fillna(-1).astype("int16").values,
        )
    return ds.assign_coords(coords) if coords else ds


def _item_dicts(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, gpd.GeoDataFrame):
        return list(items["stac_item"]) if "stac_item" in items.columns else []
    return [
        item.to_dict(transform_hrefs=False) if hasattr(item, "to_dict") else item
        for item in items
    ]


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    orbit_state: str | None = None,
    query: dict[str, Any] | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search a Sentinel-1 RTC catalog and return the items as a ``GeoDataFrame``.

    Parameters
    ----------
    source
        ``"planetary-computer"`` (default) or ``"opera-rtc-s1"``. The Earth
        Engine route has no item search; use :func:`load` with
        ``source="gee"``.
    orbit_state
        ``"ascending"`` or ``"descending"`` (Planetary Computer only: OPERA
        items carry no orbit properties).
    """
    src = resolve_source(PRODUCT, source)
    if src.id == "gee":
        raise ValueError(
            'source="gee" has no item search; call load(..., source="gee") instead.'
        )
    stac_query = dict(query or {})
    if orbit_state is not None:
        stac_query["sat:orbit_state"] = {"eq": orbit_state}
    catalog_id = "planetary-computer" if src.id == "planetary-computer" else "cmr-asf"
    collection = (
        "sentinel-1-rtc" if src.id == "planetary-computer" else OPERA_COLLECTION
    )
    items = providers.stac.search(
        catalog_id,
        collection,
        aoi,
        time,
        query=stac_query or None,
        max_items=max_items,
        **kwargs,
    )
    gdf = providers.stac.items_to_geodataframe(items)
    gdf.attrs = {"source": src.id, "collection": collection}
    return gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    items: Any = None,
    bands: str | Sequence[str] | None = None,
    source: str | None = None,
    units: str = "dB",
    resolution: float | None = None,
    crs: Any = "utm",
    groupby: str | None = None,
    border_noise: bool = True,
    mask: bool = True,
    orbit_state: str | None = None,
    chunks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load Sentinel-1 RTC backscatter as a lazy ``xarray.Dataset``.

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    bands
        Polarizations (``"vv"``, ``"vh"``; OPERA also has ``"hh"``/``"hv"``).
    source
        ``"planetary-computer"`` (default), ``"opera-rtc-s1"`` or ``"gee"``.
    units
        ``"dB"`` (default) or ``"linear power"``.
    groupby
        odc-stac grouping; defaults to ``"sat:absolute_orbit"`` on Planetary
        Computer (one raster per pass) and ``"solar_day"`` for OPERA bursts.
        OPERA bursts are grouped **within each relative orbit** first, so a
        time step never mixes two tracks' geometries; the track rides along as
        the ``sat:relative_orbit`` coordinate.
    border_noise
        Mask the falsely low values along scene edges in pre-2018 scenes
        (:func:`easysnowdata.processing.remove_border_noise`).
    mask
        Include the OPERA layover/shadow mask (OPERA route only).
    orbit_state
        Keep only ascending or descending passes (Planetary Computer).
    """
    src = resolve_source(PRODUCT, source)
    polarisations = _polarisations(bands)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if src.id == "gee":
        ds = _load_gee(parsed, time, polarisations, chunks, **kwargs)
        item_dicts: list[dict[str, Any]] = []
    else:
        if items is None:
            items = search(aoi, time, source=src.id, orbit_state=orbit_state)
        item_dicts = _item_dicts(items)
        if not item_dicts:
            raise ValueError(
                "No Sentinel-1 RTC items for this AOI and time; widen the search."
            )
        if src.id == "planetary-computer":
            ds = providers.stac.load(
                items,
                parsed,
                bands=polarisations,
                resolution=resolution,
                crs=crs,
                chunks=chunks,
                groupby=groupby or "sat:absolute_orbit",
                catalog="planetary-computer",
                **kwargs,
            ).sortby("time")  # the PC Sentinel-1 items are not time-ordered
        else:
            ds = _load_opera_by_track(
                item_dicts,
                parsed,
                bands=_opera_assets(polarisations, mask),
                resolution=resolution,
                crs=crs,
                chunks=chunks,
                groupby=groupby or "solar_day",
                **kwargs,
            )

    for name in list(ds.data_vars):
        if name == "mask":
            ds[name] = contract.set_categorical_nodata(ds[name], 255)
            ds[name].attrs.update(LIA_PRODUCT.variables[-1].cf_attrs())
        else:
            ds[name] = contract.mask_continuous(
                ds[name], PC_NODATA if src.id == "planetary-computer" else None
            )
    ds.attrs["units"] = "linear power"

    if border_noise:
        backscatter = [n for n in ds.data_vars if n != "mask"]
        for name in backscatter:
            ds[name] = sar_processing.remove_border_noise(ds[name])
    if str(units).lower() in ("db", "decibel", "decibels"):
        for name in [n for n in ds.data_vars if n != "mask"]:
            ds[name] = sar_processing.linear_to_db(ds[name])
        ds.attrs["units"] = "dB"
    ds = _add_orbit_coords(ds, item_dicts)
    ds = contract.finalize(
        ds,
        PRODUCT,
        src,
        variables=(),
        source_url={
            "planetary-computer": _PC_DOCS,
            "opera-rtc-s1": _OPERA_DOCS,
            "gee": _GEE_DOCS,
        }[src.id],
        attrs={"units": ds.attrs.get("units")},
    )
    return ds


def _load_gee(
    parsed: Any, time: Any, polarisations: Sequence[str], chunks: Any, **kwargs: Any
) -> xr.Dataset:
    from easysnowdata import temporal  # noqa: PLC0415

    ee = providers.gee.ee()
    collection = ee.ImageCollection(GEE_COLLECTION)
    start, end = temporal.parse_time(time)
    if time is not None:
        collection = collection.filterDate(
            (start or pd.Timestamp("2016-04-01")).strftime("%Y-%m-%d"),
            (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
    bands = [p.upper() for p in polarisations]
    collection = collection.select(bands)
    ds = providers.gee.open_dataset(
        collection, parsed, chunks={} if chunks is None else chunks, **kwargs
    )
    return ds.rename({b: b.lower() for b in ds.data_vars if b in bands})


def local_incidence_angle(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    relative_orbit: int | None = None,
    orbit_state: str | None = None,
    incidence_angle: float | xr.DataArray | None = None,
    resolution: float | None = None,
    crs: Any = "utm",
    dem: xr.DataArray | None = None,
    mask: bool = True,
    chunks: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Local incidence angle (and, where available, the layover/shadow mask).

    The local incidence angle is a property of one acquisition geometry, not
    of a place: it depends on which relative orbit (track) the scene came
    from, whether the pass was ascending or descending, and where in the
    swath a pixel sits (the ellipsoidal incidence angle runs from about 29°
    at near range to 46° at far range in IW mode). Rasters of different
    tracks are therefore **never fused**. With ``relative_orbit=`` the result
    is one ``(y, x)`` raster for that track; without it, one raster per track
    that crosses the AOI, stacked along a ``relative_orbit`` dimension with
    ``sat:orbit_state`` (and, on the computed routes, ``platform_heading``
    and ``look_azimuth``) as coordinates along it. An INFO log line says so.

    The three routes trade exactness for access:

    ``"opera-static"`` (default, Earthdata Login)
        The published answer. OPERA's RTC-S1-STATIC layers carry the local
        incidence angle, the ellipsoidal incidence angle and the
        layover/shadow mask **per burst**, computed from the orbit state
        vectors and the Copernicus DEM. The bursts of one track are mosaicked;
        bursts of different tracks go to different ``relative_orbit`` slices.
    ``"dem"`` (no account)
        An approximation from a DEM. The geometry of each track is taken from
        a representative Planetary Computer RTC scene: the platform heading
        from the scene footprint's along-track edge, the look azimuth from
        the heading and the look side, and the ellipsoidal incidence angle as
        a field that grows linearly from the near-range edge to the far-range
        edge of the swath. Every pass of a relative orbit is assumed to share
        this geometry (the same assumption the standalone LIA tool at
        https://github.com/egagli/generate_sentinel1_local_incidence_angle_maps
        makes; that tool computes the exact geometry from the GRD orbit file
        with ``sarsen``, at the price of a SAFE download per track). The
        result is within a few degrees of OPERA's on open slopes and does not
        know about layover or shadow. When no RTC scene of a requested track
        or pass can be found over the AOI the route raises ``ValueError``
        rather than guess: the heading varies with latitude and the incidence
        angle with the position in the swath, so there is no defensible
        default.
    ``"gee"`` (Earth Engine)
        The ``COPERNICUS/S1_GRD`` ``angle`` band — the real per-pixel
        ellipsoidal incidence angle of each track — combined with the
        Copernicus DEM and the same footprint-derived heading.

    Parameters
    ----------
    aoi, time
        The area, and (OPERA route) a time window for the search.
    source
        ``"opera-static"``, ``"dem"`` or ``"gee"``.
    relative_orbit
        The Sentinel-1 track (``sat:relative_orbit``) to compute or select.
        ``None`` (default) returns every track over *aoi* along a
        ``relative_orbit`` dimension.
    orbit_state
        ``"ascending"`` or ``"descending"`` restricts the tracks to one pass
        direction on the ``"dem"`` and ``"gee"`` routes; ``None`` (default)
        keeps both. Ignored when *relative_orbit* is given (a track has one
        pass direction) and on the OPERA route.
    incidence_angle
        Override for the ellipsoidal incidence angle on the ``"dem"`` route:
        a scalar for the whole AOI or a raster aligned with the DEM.
    dem
        A DEM to use instead of fetching the Copernicus DEM (``"dem"`` route).

    Returns
    -------
    xarray.Dataset
        ``local_incidence_angle`` and ``incidence_angle`` in degrees, plus the
        categorical ``mask`` on the OPERA route. With *relative_orbit* the
        geometry used is in the attrs (``relative_orbit``, ``orbit_state``,
        ``platform_heading``, ``look_azimuth``, ``incidence_angle_model``);
        without it, the per-track values are coordinates along
        ``relative_orbit``.
    """
    if orbit_state is not None and orbit_state not in ORBIT_STATES:
        raise ValueError(
            f"orbit_state must be one of {list(ORBIT_STATES)}, got {orbit_state!r}."
        )
    src = resolve_source(LIA_PRODUCT, source or "opera-static")
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(LIA_PRODUCT, src)

    if src.id == "opera-static":
        tracks = _lia_from_opera(
            parsed, time, resolution, crs, mask, chunks, relative_orbit, **kwargs
        )
    elif src.id == "dem":
        tracks = _lia_from_dem(
            parsed,
            orbit_state,
            relative_orbit,
            incidence_angle,
            resolution,
            crs,
            dem,
            chunks,
            **kwargs,
        )
    else:
        tracks = _lia_from_gee(parsed, orbit_state, relative_orbit, chunks, **kwargs)

    if relative_orbit is None:
        _logger.info(
            "No relative_orbit specified: returning a local incidence angle raster "
            "for each relative orbit within the AOI — tracks %s — along the "
            "relative_orbit dimension. Tracks are never fused; select one with "
            ".sel(relative_orbit=...) or pass relative_orbit=.",
            ", ".join(str(t) for t, _ in tracks),
        )
    ds = _stack_tracks(tracks, single=relative_orbit is not None)
    return contract.finalize(
        ds,
        LIA_PRODUCT,
        src,
        variables=(),
        source_url=_OPERA_DOCS if src.id == "opera-static" else _PC_DOCS,
    )


#: The per-track pieces every route returns: ``(track, dataset)`` in track order,
#: each dataset ``(y, x)`` with the geometry it used in ``attrs``.
TrackRasters = list[tuple[int, xr.Dataset]]

_TRACK_COORDS = (
    ("sat:orbit_state", "orbit_state", object, "pass direction of the track"),
    ("platform_heading", "platform_heading", "float32", "degrees clockwise from north"),
    ("look_azimuth", "look_azimuth", "float32", "degrees clockwise from north"),
    ("scene_id", "scene_id", object, "scene the geometry was read from"),
)


def _stack_tracks(tracks: TrackRasters, *, single: bool) -> xr.Dataset:
    """One ``(y, x)`` Dataset for a requested track, or every track stacked
    along a new ``relative_orbit`` dimension with the geometry as coordinates."""
    if not tracks:  # pragma: no cover — every route raises before this
        raise ValueError("No Sentinel-1 track covers this AOI.")
    if single:
        track, ds = tracks[0]
        ds = ds.assign_coords(relative_orbit=track)
        ds["relative_orbit"].attrs["long_name"] = "Sentinel-1 relative orbit"
        ds.attrs["relative_orbit"] = track
        return ds
    reference_ds = tracks[0][1]
    parts = [reference_ds, *(_align_to(reference_ds, ds) for _, ds in tracks[1:])]
    ds = xr.concat(
        parts,
        dim=pd.Index([t for t, _ in tracks], name="relative_orbit"),
        combine_attrs="drop_conflicts",
    )
    ds["relative_orbit"].attrs["long_name"] = "Sentinel-1 relative orbit"
    for coord, attr, dtype, description in _TRACK_COORDS:
        values = [part_ds.attrs.get(attr) for part_ds in parts]
        if all(v is None for v in values):
            continue
        ds = ds.assign_coords(
            {coord: ("relative_orbit", np.asarray(values, dtype=dtype))}
        )
        ds[coord].attrs["description"] = description
        ds.attrs.pop(attr, None)
    for attr in ("relative_orbit", "orbit_state", "scene_id"):
        ds.attrs.pop(attr, None)
    return ds


def _lia_from_opera(
    parsed: Any,
    time: Any,
    resolution: float | None,
    crs: Any,
    mask: bool,
    chunks: Any,
    relative_orbit: int | None = None,
    **kwargs: Any,
) -> TrackRasters:
    items = list(
        providers.stac.search(
            "cmr-asf", OPERA_STATIC_COLLECTION, parsed, time, **kwargs
        )
    )
    if not items:
        raise ValueError(
            "No OPERA RTC-S1-STATIC granules cover this AOI; try source='dem'."
        )
    groups = _group_by_track(items)
    if relative_orbit is not None:
        # The burst id carries the track: OPERA_L2_RTC-S1-STATIC_T137-292394-IW3_…
        if int(relative_orbit) not in groups:
            raise ValueError(
                f"No OPERA static bursts of track {relative_orbit} cover this AOI; "
                f"the tracks that do: {[t for t in groups if t >= 0]}."
            )
        groups = {int(relative_orbit): groups[int(relative_orbit)]}
    bands = ["0_local_incidence_angle", "0_incidence_angle"]
    if mask:
        bands.append("0_mask")
    # The pass direction of each track, from Planetary Computer scenes of the
    # same tracks (OPERA static items carry no orbit properties). Best effort.
    states = (
        {t: g["orbit_state"] for t, g in track_geometries(parsed).items()}
        if parsed is not None
        else {}
    )
    tracks: TrackRasters = []
    for track, group in groups.items():
        ds = providers.stac.load(
            group,
            parsed,
            bands=bands,
            resolution=resolution,
            crs=crs,
            chunks=chunks,
            groupby="solar_day",
            catalog="cmr-asf",
        )
        ds = _rename_opera(ds)
        if "time" in ds.dims:  # static layers: collapse the (degenerate) time axis
            ds = ds.max(dim="time", keep_attrs=True)
        # The COGs carry no band metadata, so name and unit the layers here the
        # same way the DEM route does; plotting labels read these.
        ds["local_incidence_angle"].attrs.update(
            {"long_name": "local incidence angle", "units": "degrees"}
        )
        ds["incidence_angle"].attrs.update(
            {"long_name": "ellipsoidal incidence angle", "units": "degrees"}
        )
        if "mask" in ds:
            ds["mask"] = contract.set_categorical_nodata(ds["mask"], 255)
            ds["mask"].attrs.update(LIA_PRODUCT.variables[-1].cf_attrs())
        ds.attrs["incidence_angle_model"] = "OPERA RTC-S1-STATIC per-burst layers"
        if track in states:
            ds.attrs["orbit_state"] = states[track]
        tracks.append((track, ds))
    return tracks


def _item_id(item: Any) -> str:
    return str(
        getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else "")
    )


def _properties(item: Any) -> dict[str, Any]:
    """A STAC item's properties, whether it is a ``pystac.Item`` or a dict."""
    if isinstance(item, dict):
        return item.get("properties") or {}
    return getattr(item, "properties", None) or {}


def _geometry(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item.get("geometry")
    return getattr(item, "geometry", None)


def _track_of(item_id: str) -> int | None:
    """The relative orbit encoded in an OPERA burst id (``…_T137-292394-IW3_…``)."""
    match = re.search(r"_T(\d{3})-", str(item_id))
    return int(match.group(1)) if match else None


#: Ellipsoidal incidence angle at the near and far edges of the IW swath.
IW_INCIDENCE_RANGE = (29.1, 46.0)
#: The two Sentinel-1 pass directions, as ``sat:orbit_state`` spells them.
ORBIT_STATES = ("ascending", "descending")


def track_geometries(
    aoi: Any,
    *,
    orbit_state: str | None = None,
    relative_orbit: int | None = None,
    time: Any = "2023-01-01/2024-12-31",
) -> dict[int, dict[str, Any]]:
    """The acquisition geometry of every Sentinel-1 track over *aoi*, by track.

    Searches the Planetary Computer RTC collection for scenes over *aoi*
    (restricted to *orbit_state* and/or *relative_orbit* when given), groups
    them by ``sat:relative_orbit`` and reads each track's geometry off a
    representative footprint:

    ``orbit_state``
        The pass direction, from the scene.
    ``platform_heading``
        Azimuth of the footprint's along-track edge, oriented by the pass
        (southward for descending, northward for ascending). Sentinel-1's
        heading varies with latitude, so this beats a constant.
    ``look_azimuth``
        Heading plus 90° for the right-looking Sentinel-1
        (``sar:observation_direction``).
    ``near_range``, ``swath_width``
        Position of the near-range edge along the look direction and the
        swath's width, in the AOI's UTM metres — the inputs
        :func:`incidence_angle_field` converts to a per-pixel ellipsoidal
        incidence angle.
    ``scene_id``, ``scenes_found``
        Which scene the geometry came from, and how many the track had.

    Returns an empty dict when no scene is found (or the search fails).
    """
    parsed = parse_aoi(aoi)
    query: dict[str, Any] = {}
    if orbit_state is not None:
        query["sat:orbit_state"] = {"eq": orbit_state}
    if relative_orbit is not None:
        query["sat:relative_orbit"] = {"eq": int(relative_orbit)}
    try:
        items = providers.stac.search(
            "planetary-computer", "sentinel-1-rtc", parsed, time, query=query or None
        )
    except Exception as exc:  # noqa: BLE001 — the callers raise a clear error
        _logger.warning("Could not search Sentinel-1 scenes for the geometry: %s", exc)
        return {}
    by_track: dict[int, list[Any]] = {}
    for item in items:
        properties = _properties(item)
        track = properties.get("sat:relative_orbit")
        state = properties.get("sat:orbit_state") or orbit_state
        if track is None or state not in ORBIT_STATES or _geometry(item) is None:
            continue
        if orbit_state is not None and state != orbit_state:
            continue
        if relative_orbit is not None and int(track) != int(relative_orbit):
            continue
        by_track.setdefault(int(track), []).append(item)
    utm = parsed.utm_crs
    geometries: dict[int, dict[str, Any]] = {}
    for track in sorted(by_track):
        item = by_track[track][0]
        properties = _properties(item)
        state = str(properties.get("sat:orbit_state") or orbit_state)
        footprint = gpd.GeoSeries(
            [shapely.geometry.shape(_geometry(item))], crs="EPSG:4326"
        )
        ring = np.asarray(footprint.to_crs(utm).iloc[0].exterior.coords)
        heading = _along_track_heading(ring, state)
        right = str(properties.get("sar:observation_direction", "right")) == "right"
        look = sar_processing.look_azimuth(
            heading, looking="right" if right else "left"
        )
        ux, uy = np.sin(np.radians(look)), np.cos(np.radians(look))
        along_look = ring[:, 0] * ux + ring[:, 1] * uy
        geometries[track] = {
            "relative_orbit": track,
            "orbit_state": state,
            "platform_heading": float(heading),
            "look_azimuth": float(look),
            "near_range": float(along_look.min()),
            "swath_width": float(along_look.max() - along_look.min()),
            "crs": str(utm),
            "scene_id": _item_id(item),
            "scenes_found": len(by_track[track]),
        }
    return geometries


def scene_geometry(
    aoi: Any,
    *,
    orbit_state: str | None = None,
    relative_orbit: int | None = None,
    time: Any = "2023-01-01/2024-12-31",
) -> dict[str, Any]:
    """The geometry of one track over *aoi*: :func:`track_geometries` reduced
    to the requested *relative_orbit*, or else the track with the most scenes.

    Returns an empty dict when no scene is found.
    """
    geometries = track_geometries(
        aoi, orbit_state=orbit_state, relative_orbit=relative_orbit, time=time
    )
    if not geometries:
        return {}
    if relative_orbit is not None:
        return geometries.get(int(relative_orbit), {})
    return max(geometries.values(), key=lambda g: g["scenes_found"])


def _along_track_heading(ring: np.ndarray, orbit_state: str) -> float:
    """Heading (° clockwise from north) of the footprint edge that runs along track.

    Sentinel-1 flies a near-polar orbit (98.2° inclination), so at any
    latitude a GRD footprint's along-track edges are the two that run closest
    to north–south and its across-track edges the two closest to east–west.
    The along-track azimuths are averaged and then oriented so a descending
    pass heads south. No fixed heading is assumed: the real one swings by
    tens of degrees between the equator and the poles.
    """
    nominal = 0.0  # north–south, modulo 180°
    azimuths = []
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=True):
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 1.0:
            continue
        az = float(np.degrees(np.arctan2(x1 - x0, y1 - y0))) % 180.0
        diff = min(abs(az - nominal), 180.0 - abs(az - nominal))
        azimuths.append((diff, az, length))
    azimuths.sort()
    along = [az for _, az, _ in azimuths[:2]] or [nominal]
    # average on the circle mod 180
    doubled = np.radians(2 * np.asarray(along))
    mean = (
        float(
            np.degrees(np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))))
        )
        / 2.0
    ) % 180.0
    # Orient the line: a descending pass heads south (90°–270°), an ascending
    # pass north (270°–90° through 0°).
    if orbit_state == "descending":
        return mean if mean >= 90.0 else mean + 180.0
    return mean if mean < 90.0 else mean + 180.0


def incidence_angle_field(
    dem: xr.DataArray,
    geometry: dict[str, Any],
    *,
    swath: tuple[float, float] = IW_INCIDENCE_RANGE,
) -> xr.DataArray:
    """Ellipsoidal incidence angle across the swath, on the DEM's grid.

    Linear from ``swath[0]`` at the near-range edge to ``swath[1]`` at the
    far-range edge of the footprint in *geometry* (the true relation is
    slightly convex; the linear model is within about a degree). Pixels
    outside the footprint are clipped to the nearest edge.
    """
    from pyproj import Transformer  # noqa: PLC0415

    x_dim, y_dim = ("x", "y") if "x" in dem.dims else ("longitude", "latitude")
    xx, yy = np.meshgrid(dem[x_dim].values, dem[y_dim].values)
    crs = dem.rio.crs
    if crs is None or str(crs) != geometry["crs"]:
        to_utm = Transformer.from_crs(
            crs or "EPSG:4326", geometry["crs"], always_xy=True
        )
        xx, yy = to_utm.transform(xx, yy)
    look = np.radians(geometry["look_azimuth"])
    along_look = xx * np.sin(look) + yy * np.cos(look)
    frac = (along_look - geometry["near_range"]) / max(geometry["swath_width"], 1.0)
    frac = np.clip(frac, 0.0, 1.0)
    values = swath[0] + frac * (swath[1] - swath[0])
    field_da = xr.DataArray(
        values.astype("float32"),
        dims=(y_dim, x_dim),
        coords={y_dim: dem[y_dim], x_dim: dem[x_dim]},
    )
    field_da.attrs = {
        "long_name": "ellipsoidal incidence angle",
        "units": "degrees",
        "description": (
            f"linear across the IW swath from {swath[0]}° at near range to "
            f"{swath[1]}° at far range, from the footprint of {geometry.get('scene_id')}"
        ),
    }
    if crs is not None:
        field_da = field_da.rio.write_crs(crs)
    return field_da


def _no_scene_error(orbit_state: str | None, relative_orbit: int | None) -> ValueError:
    if relative_orbit is not None:
        which = f"relative orbit {relative_orbit}"
    elif orbit_state is not None:
        which = f"the {orbit_state} pass"
    else:
        which = "any track"
    return ValueError(
        f"No Sentinel-1 RTC scene of {which} found over this AOI on Planetary "
        "Computer, so there is no pass geometry to compute the local incidence "
        "angle from. The heading depends on latitude and the incidence angle on "
        "the position in the swath, so nothing is assumed in their place: check "
        "orbit_state= and relative_orbit=, or use source='opera-static' "
        "(Earthdata Login), which reads the published layer."
    )


def _lia_from_dem(
    parsed: Any,
    orbit_state: str | None,
    relative_orbit: int | None,
    incidence_angle: float | xr.DataArray | None,
    resolution: float | None,
    crs: Any,
    dem_da: xr.DataArray | None,
    chunks: Any,
    **kwargs: Any,
) -> TrackRasters:
    if parsed is None:
        raise ValueError(
            "source='dem' needs an aoi: the pass geometry (heading, look "
            "direction and the incidence angle across the swath) is read from a "
            "Sentinel-1 RTC scene found over it. To use a geometry of your own, "
            "call easysnowdata.processing.sar.local_incidence_angle on the DEM."
        )
    geometries = track_geometries(
        parsed, orbit_state=orbit_state, relative_orbit=relative_orbit
    )
    if not geometries:
        raise _no_scene_error(orbit_state, relative_orbit)
    if dem_da is None:
        dem_da = _copernicus_dem(parsed, resolution, crs, chunks, **kwargs)
    tracks: TrackRasters = []
    for track, geometry in geometries.items():
        heading = geometry["platform_heading"]
        look = geometry["look_azimuth"]
        model = (
            f"track {track} geometry from scene {geometry['scene_id']}; incidence "
            f"linear {IW_INCIDENCE_RANGE[0]}°–{IW_INCIDENCE_RANGE[1]}° across the swath"
        )
        if incidence_angle is not None:
            angle: Any = incidence_angle
            model += "; incidence angle supplied by the caller"
        else:
            angle = incidence_angle_field(dem_da, geometry)
        lia_da = sar_processing.local_incidence_angle(dem_da, angle, look)
        ds = lia_da.to_dataset(name="local_incidence_angle")
        ds["incidence_angle"] = (
            xr.full_like(lia_da, float(angle))
            if not isinstance(angle, xr.DataArray)
            else angle
        )
        ds["incidence_angle"].attrs.setdefault(
            "long_name", "ellipsoidal incidence angle"
        )
        ds["incidence_angle"].attrs.setdefault("units", "degrees")
        ds.attrs.update(
            {
                "orbit_state": geometry["orbit_state"],
                "look_azimuth": float(look),
                "platform_heading": float(heading),
                "scene_id": geometry["scene_id"],
                "incidence_angle_model": model,
            }
        )
        tracks.append((track, ds))
    return tracks


def _copernicus_dem(
    parsed: Any, resolution: float | None, crs: Any, chunks: Any, **kwargs: Any
) -> xr.DataArray:
    """The Copernicus GLO-30 DEM over the AOI, on a metric grid."""
    items = providers.stac.search(
        "planetary-computer", COPERNICUS_DEM_COLLECTION, parsed, None
    )
    if not len(items):
        raise ValueError("No Copernicus DEM tiles cover this AOI.")
    dem_da = providers.stac.load(
        items,
        parsed,
        bands=["data"],
        resolution=resolution or 30,
        crs=crs,
        chunks=chunks,
        groupby="solar_day",
        catalog="planetary-computer",
        # Elevation is continuous: interpolate it onto the metric grid rather
        # than copying the nearest 1" pixel (which would fold its staircase
        # into every slope and aspect).
        resampling=kwargs.pop("resampling", "bilinear"),
        **kwargs,
    )["data"]
    if "time" in dem_da.dims:
        dem_da = dem_da.max(dim="time", keep_attrs=True)
    return dem_da


def _lia_from_gee(
    parsed: Any,
    orbit_state: str | None,
    relative_orbit: int | None,
    chunks: Any,
    **kwargs: Any,
) -> TrackRasters:
    """The Earth Engine route: the S1_GRD angle band of each track plus GLO-30."""
    ee = providers.gee.ee()
    geometry_ee = providers.gee.geometry(parsed)
    geometries = track_geometries(
        parsed, orbit_state=orbit_state, relative_orbit=relative_orbit
    )
    if not geometries:
        raise _no_scene_error(orbit_state, relative_orbit)
    dem_da = _copernicus_dem(parsed, 30, "utm", chunks)
    tracks: TrackRasters = []
    for track, geometry in geometries.items():
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry_ee)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.eq("orbitProperties_pass", geometry["orbit_state"].upper())
            )
            .filter(ee.Filter.eq("relativeOrbitNumber_start", int(track)))
        )
        angle = collection.select("angle").median()
        ds = providers.gee.open_dataset(
            ee.ImageCollection([angle]),
            parsed,
            chunks={} if chunks is None else chunks,
            **kwargs,
        )
        if "angle" not in ds.data_vars:  # pragma: no cover — defensive
            raise ValueError("Earth Engine returned no angle band for this AOI.")
        ds = ds.rename({"angle": "incidence_angle"})
        if "time" in ds.dims:
            ds = ds.max(dim="time", keep_attrs=True)
        incidence_da = ds["incidence_angle"]
        if incidence_da.shape != dem_da.shape:
            incidence_da = incidence_da.rio.reproject_match(dem_da)
        heading = geometry["platform_heading"]
        look = geometry["look_azimuth"]
        lia_da = sar_processing.local_incidence_angle(dem_da, incidence_da, look)
        track_ds = lia_da.to_dataset(name="local_incidence_angle")
        track_ds["incidence_angle"] = incidence_da
        track_ds["incidence_angle"].attrs.update(
            {"long_name": "ellipsoidal incidence angle", "units": "degrees"}
        )
        track_ds.attrs.update(
            {
                "orbit_state": geometry["orbit_state"],
                "look_azimuth": float(look),
                "platform_heading": float(heading),
                "scene_id": geometry["scene_id"],
                "incidence_angle_model": (
                    "COPERNICUS/S1_GRD angle band (median over the track's scenes)"
                ),
            }
        )
        tracks.append((track, track_ds))
    return tracks
