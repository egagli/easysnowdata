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
is a computation rather than a download::

    import easysnowdata as esd
    s1 = esd.sar.sentinel1.load(aoi, "2023-10/2024-06", units="dB")
    lia = esd.sar.sentinel1.local_incidence_angle(aoi)             # OPERA static
    lia = esd.sar.sentinel1.local_incidence_angle(aoi, source="dem")   # no account
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
                "layover/shadow mask. CMR-STAC cloudstac/ASF, not stac.asf.alaska.edu"
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
        "layers, or computed from any DEM with no account at all (issue #10)."
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
                "number_of_looks and the gamma0→beta0/sigma0 factors as COGs"
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
    frame = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time")
    if frame.empty:
        return ds
    times = pd.DatetimeIndex(ds["time"].values)
    matched = frame.set_index("time").reindex(
        times, method="nearest", tolerance=pd.Timedelta("1D")
    )
    coords = {}
    if matched["orbit_state"].notna().any():
        coords["sat:orbit_state"] = (
            "time",
            matched["orbit_state"].astype("object").values,
        )
    if matched["relative_orbit"].notna().any():
        coords["sat:relative_orbit"] = (
            "time",
            matched["relative_orbit"].fillna(-1).astype("int16").values,
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
            ds = providers.stac.load(
                items,
                parsed,
                bands=_opera_assets(polarisations, mask),
                resolution=resolution,
                crs=crs,
                chunks=chunks,
                groupby=groupby or "solar_day",
                catalog="cmr-asf",
                **kwargs,
            ).sortby("time")
            ds = _rename_opera(ds)

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
    orbit_state: str = "ascending",
    relative_orbit: int | None = None,
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
    at near range to 46° at far range in IW mode). The three routes trade
    exactness for access:

    ``"opera-static"`` (default, Earthdata Login)
        The published answer. OPERA's RTC-S1-STATIC layers carry the local
        incidence angle, the ellipsoidal incidence angle and the
        layover/shadow mask **per burst**, computed from the orbit state
        vectors and the Copernicus DEM. Bursts from several tracks cover most
        places; pass *relative_orbit* to keep one track's bursts, otherwise
        overlapping bursts are fused (max) into a single raster and the
        geometry is mixed.
    ``"dem"`` (no account)
        An approximation from a DEM. The geometry is taken from a
        representative Planetary Computer RTC scene of the chosen track:
        the platform heading from the scene footprint's along-track edge,
        the look azimuth from the heading and the look side, and the
        ellipsoidal incidence angle as a field that grows linearly from the
        near-range edge to the far-range edge of the swath. Every pass of a
        relative orbit is assumed to share this geometry (the same assumption
        the standalone LIA tool at
        https://github.com/egagli/generate_sentinel1_local_incidence_angle_maps
        makes; that tool computes the exact geometry from the GRD orbit file
        with ``sarsen``, at the price of a SAFE download per track). The
        result is within a few degrees of OPERA's on open slopes and does not
        know about layover or shadow. When no RTC scene of that pass can be
        found over the AOI the route raises ``ValueError`` rather than guess:
        the heading varies with latitude and the incidence angle with the
        position in the swath, so there is no defensible default.
    ``"gee"`` (Earth Engine)
        The ``COPERNICUS/S1_GRD`` ``angle`` band — the real per-pixel
        ellipsoidal incidence angle of the chosen track — combined with the
        Copernicus DEM and the same footprint-derived heading.

    Parameters
    ----------
    aoi, time
        The area, and (OPERA route) a time window for the search.
    source
        ``"opera-static"``, ``"dem"`` or ``"gee"``.
    orbit_state
        ``"ascending"`` (default) or ``"descending"``: which pass geometry,
        on the ``"dem"`` and ``"gee"`` routes.
    relative_orbit
        The Sentinel-1 track (``sat:relative_orbit``) to compute or select
        for. ``None`` picks, on the ``"dem"``/``"gee"`` routes, the track of
        that *orbit_state* with the most scenes over *aoi*; on the OPERA
        route it keeps every burst.
    incidence_angle
        Override for the ellipsoidal incidence angle on the ``"dem"`` route:
        a scalar for the whole AOI or a raster aligned with the DEM.
    dem
        A DEM to use instead of fetching the Copernicus DEM (``"dem"`` route).

    Returns
    -------
    xarray.Dataset
        ``local_incidence_angle`` and ``incidence_angle`` in degrees, plus the
        categorical ``mask`` on the OPERA route; the geometry used is in the
        attrs (``relative_orbit``, ``platform_heading``, ``look_azimuth``,
        ``incidence_angle_model``).
    """
    if orbit_state not in ORBIT_STATES:
        raise ValueError(
            f"orbit_state must be one of {list(ORBIT_STATES)}, got {orbit_state!r}."
        )
    src = resolve_source(LIA_PRODUCT, source or "opera-static")
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(LIA_PRODUCT, src)

    if src.id == "opera-static":
        ds = _lia_from_opera(
            parsed, time, resolution, crs, mask, chunks, relative_orbit, **kwargs
        )
    elif src.id == "dem":
        ds = _lia_from_dem(
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
        ds = _lia_from_gee(parsed, orbit_state, relative_orbit, chunks, **kwargs)
    return contract.finalize(
        ds,
        LIA_PRODUCT,
        src,
        variables=(),
        source_url=_OPERA_DOCS if src.id == "opera-static" else _PC_DOCS,
        attrs={"orbit_state": orbit_state if src.id != "opera-static" else None},
    )


def _lia_from_opera(
    parsed: Any,
    time: Any,
    resolution: float | None,
    crs: Any,
    mask: bool,
    chunks: Any,
    relative_orbit: int | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    items = providers.stac.search(
        "cmr-asf", OPERA_STATIC_COLLECTION, parsed, time, **kwargs
    )
    if not len(items):
        raise ValueError(
            "No OPERA RTC-S1-STATIC granules cover this AOI; try source='dem'."
        )
    if relative_orbit is not None:
        # The burst id carries the track: OPERA_L2_RTC-S1-STATIC_T137-292394-IW3_…
        wanted = f"_T{int(relative_orbit):03d}-"
        kept = [it for it in items if wanted in _item_id(it)]
        if not kept:
            tracks = sorted({_track_of(_item_id(it)) for it in items} - {None})
            raise ValueError(
                f"No OPERA static bursts of track {relative_orbit} cover this AOI; "
                f"the tracks that do: {tracks}."
            )
        items = kept
    bands = ["0_local_incidence_angle", "0_incidence_angle"]
    if mask:
        bands.append("0_mask")
    ds = providers.stac.load(
        items,
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
        ds["mask"].attrs.setdefault("long_name", "layover and shadow mask")
    tracks = sorted({_track_of(_item_id(it)) for it in items} - {None})
    ds.attrs["relative_orbit"] = (
        tracks[0] if len(tracks) == 1 else " ".join(map(str, tracks))
    )
    ds.attrs["incidence_angle_model"] = "OPERA RTC-S1-STATIC per-burst layers"
    return ds


def _item_id(item: Any) -> str:
    return str(
        getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else "")
    )


def _track_of(item_id: str) -> int | None:
    """The relative orbit encoded in an OPERA burst id (``…_T137-292394-IW3_…``)."""
    match = re.search(r"_T(\d{3})-", str(item_id))
    return int(match.group(1)) if match else None


#: Ellipsoidal incidence angle at the near and far edges of the IW swath.
IW_INCIDENCE_RANGE = (29.1, 46.0)
#: The two Sentinel-1 pass directions, as ``sat:orbit_state`` spells them.
ORBIT_STATES = ("ascending", "descending")


def scene_geometry(
    aoi: Any,
    *,
    orbit_state: str = "descending",
    relative_orbit: int | None = None,
    time: Any = "2023-01-01/2024-12-31",
) -> dict[str, Any]:
    """The acquisition geometry of one Sentinel-1 track over *aoi*, from a scene footprint.

    Searches the Planetary Computer RTC collection for scenes of
    *orbit_state* (and *relative_orbit* when given), takes the track with the
    most scenes, and reads the geometry off a representative footprint:

    ``platform_heading``
        Azimuth of the footprint's along-track edge, oriented by
        *orbit_state* (southward for descending, northward for ascending).
        Sentinel-1's heading varies with latitude, so this beats a constant.
    ``look_azimuth``
        Heading plus 90° for the right-looking Sentinel-1
        (``sar:observation_direction``).
    ``near_range``, ``swath_width``
        Position of the near-range edge along the look direction and the
        swath's width, in the AOI's UTM metres — the inputs
        :func:`incidence_angle_field` converts to a per-pixel ellipsoidal
        incidence angle.

    Returns an empty dict when no scene is found.
    """
    parsed = parse_aoi(aoi)
    query: dict[str, Any] = {"sat:orbit_state": {"eq": orbit_state}}
    if relative_orbit is not None:
        query["sat:relative_orbit"] = {"eq": int(relative_orbit)}
    try:
        items = providers.stac.search(
            "planetary-computer", "sentinel-1-rtc", parsed, time, query=query
        )
    except Exception as exc:  # noqa: BLE001 — the caller falls back to constants
        _logger.warning("Could not search Sentinel-1 scenes for the geometry: %s", exc)
        return {}
    items = list(items)
    if not items:
        return {}
    by_track: dict[int, list[Any]] = {}
    for item in items:
        track = item.properties.get("sat:relative_orbit")
        if track is not None:
            by_track.setdefault(int(track), []).append(item)
    if not by_track:
        return {}
    track = max(by_track, key=lambda t: len(by_track[t]))
    item = by_track[track][0]
    utm = parsed.utm_crs
    footprint = gpd.GeoSeries([shapely.geometry.shape(item.geometry)], crs="EPSG:4326")
    ring = np.asarray(footprint.to_crs(utm).iloc[0].exterior.coords)
    heading = _along_track_heading(ring, orbit_state)
    right = str(item.properties.get("sar:observation_direction", "right")) == "right"
    look = sar_processing.look_azimuth(heading, looking="right" if right else "left")
    ux, uy = np.sin(np.radians(look)), np.cos(np.radians(look))
    along_look = ring[:, 0] * ux + ring[:, 1] * uy
    return {
        "relative_orbit": track,
        "orbit_state": orbit_state,
        "platform_heading": float(heading),
        "look_azimuth": float(look),
        "near_range": float(along_look.min()),
        "swath_width": float(along_look.max() - along_look.min()),
        "crs": str(utm),
        "scene_id": _item_id(item),
        "scenes_found": len(by_track[track]),
    }


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
    field = xr.DataArray(
        values.astype("float32"),
        dims=(y_dim, x_dim),
        coords={y_dim: dem[y_dim], x_dim: dem[x_dim]},
    )
    field.attrs = {
        "long_name": "ellipsoidal incidence angle",
        "units": "degrees",
        "description": (
            f"linear across the IW swath from {swath[0]}° at near range to "
            f"{swath[1]}° at far range, from the footprint of {geometry.get('scene_id')}"
        ),
    }
    if crs is not None:
        field = field.rio.write_crs(crs)
    return field


def _no_scene_error(orbit_state: str, relative_orbit: int | None) -> ValueError:
    which = (
        f"relative orbit {relative_orbit}"
        if relative_orbit is not None
        else f"the {orbit_state} pass"
    )
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
    orbit_state: str,
    relative_orbit: int | None,
    incidence_angle: float | xr.DataArray | None,
    resolution: float | None,
    crs: Any,
    dem: xr.DataArray | None,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    if parsed is None:
        raise ValueError(
            "source='dem' needs an aoi: the pass geometry (heading, look "
            "direction and the incidence angle across the swath) is read from a "
            "Sentinel-1 RTC scene found over it. To use a geometry of your own, "
            "call easysnowdata.processing.sar.local_incidence_angle on the DEM."
        )
    if dem is None:
        dem = _copernicus_dem(parsed, resolution, crs, chunks, **kwargs)
    geometry = scene_geometry(
        parsed, orbit_state=orbit_state, relative_orbit=relative_orbit
    )
    if not geometry:
        raise _no_scene_error(orbit_state, relative_orbit)
    heading = geometry["platform_heading"]
    look = geometry["look_azimuth"]
    model = (
        f"track {geometry['relative_orbit']} geometry from scene "
        f"{geometry['scene_id']}; incidence linear {IW_INCIDENCE_RANGE[0]}°–"
        f"{IW_INCIDENCE_RANGE[1]}° across the swath"
    )
    if incidence_angle is not None:
        angle: Any = incidence_angle
        model += "; incidence angle supplied by the caller"
    else:
        angle = incidence_angle_field(dem, geometry)
    lia = sar_processing.local_incidence_angle(dem, angle, look)
    ds = lia.to_dataset(name="local_incidence_angle")
    ds["incidence_angle"] = (
        xr.full_like(lia, float(angle))
        if not isinstance(angle, xr.DataArray)
        else angle
    )
    ds["incidence_angle"].attrs.setdefault("long_name", "ellipsoidal incidence angle")
    ds["incidence_angle"].attrs.setdefault("units", "degrees")
    ds.attrs["look_azimuth"] = float(look)
    ds.attrs["platform_heading"] = float(heading)
    ds.attrs["relative_orbit"] = geometry["relative_orbit"]
    ds.attrs["incidence_angle_model"] = model
    return ds


def _copernicus_dem(
    parsed: Any, resolution: float | None, crs: Any, chunks: Any, **kwargs: Any
) -> xr.DataArray:
    """The Copernicus GLO-30 DEM over the AOI, on a metric grid."""
    items = providers.stac.search(
        "planetary-computer", COPERNICUS_DEM_COLLECTION, parsed, None
    )
    if not len(items):
        raise ValueError("No Copernicus DEM tiles cover this AOI.")
    dem = providers.stac.load(
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
    if "time" in dem.dims:
        dem = dem.max(dim="time", keep_attrs=True)
    return dem


def _lia_from_gee(
    parsed: Any,
    orbit_state: str,
    relative_orbit: int | None,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    """The Earth Engine route: the S1_GRD angle band of one track plus GLO-30."""
    ee = providers.gee.ee()
    geometry_ee = providers.gee.geometry(parsed)
    geometry = scene_geometry(
        parsed, orbit_state=orbit_state, relative_orbit=relative_orbit
    )
    if not geometry:
        raise _no_scene_error(orbit_state, relative_orbit)
    track = (
        relative_orbit if relative_orbit is not None else geometry.get("relative_orbit")
    )
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry_ee)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_state.upper()))
    )
    if track is not None:
        collection = collection.filter(
            ee.Filter.eq("relativeOrbitNumber_start", int(track))
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
    dem = _copernicus_dem(parsed, 30, "utm", chunks)
    incidence = ds["incidence_angle"]
    if incidence.shape != dem.shape:
        incidence = incidence.rio.reproject_match(dem)
    heading = geometry["platform_heading"]
    look = geometry["look_azimuth"]
    lia = sar_processing.local_incidence_angle(dem, incidence, look)
    out = lia.to_dataset(name="local_incidence_angle")
    out["incidence_angle"] = incidence
    out["incidence_angle"].attrs.update(
        {"long_name": "ellipsoidal incidence angle", "units": "degrees"}
    )
    out.attrs["look_azimuth"] = float(look)
    out.attrs["platform_heading"] = float(heading)
    out.attrs["relative_orbit"] = track if track is not None else "unknown"
    out.attrs["incidence_angle_model"] = (
        "COPERNICUS/S1_GRD angle band (median over the track's scenes)"
    )
    return out
