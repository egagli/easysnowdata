"""UCLA snow reanalysis: Western US (WUS_UCLA_SR) and High Mountain Asia (HMA_SR_D).

Daily posterior SWE, snow depth and snow-covered area at 480 m from a
particle-batch-smoother reanalysis, one NetCDF-4 granule per water year and
1°×1° tile. Two regions, the same layout:

``wus`` (default)
    ``WUS_UCLA_SR`` v1, water years 1985–2021 over the western United States.
``hma``
    ``HMA_SR_D`` v1, the High Mountain Asia sibling — new here (plan §4.3).

Both come from NSIDC through ``earthaccess`` and need Earthdata Login.

Virtualization (plan §4.9)
--------------------------
These granules are NetCDF-4 without DMR++ sidecars, so
``earthaccess.virtualize()`` falls back to scanning each file's HDF5 metadata
once and caching kerchunk references. That is worth it for a long series and
wasted on one water year, which is what ``virtualize="auto"`` encodes: it
turns on past ``VIRTUALIZE_THRESHOLD`` granules. ``True``/``False`` force it.
The reference cache lives under :func:`easysnowdata.config.cache_dir`, so a
second call over the same granules skips the scan.

::

    import easysnowdata as esd
    swe = esd.snow.ucla_sr.load(aoi, "2019-10/2021-09")                  # WUS
    swe = esd.snow.ucla_sr.load(aoi, "2000-10/2001-09", region="hma")    # HMA
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
import xarray as xr

from easysnowdata import catalog, config, providers, temporal
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT",
    "REGIONS",
    "STATS",
    "VARIABLES",
    "VIRTUALIZE_THRESHOLD",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

#: region → (short name, extent, temporal coverage)
REGIONS: dict[str, tuple[str, str, str]] = {
    "wus": ("WUS_UCLA_SR", "western US", "1984-10/2021-09"),
    "hma": ("HMA_SR_D", "High Mountain Asia", "1999-10/2017-09"),
}
#: Position of each ensemble statistic along the ``Stats`` dimension.
#: (Phase 0 fixed the old mapping, where "median" and "25pct" shared index 2.)
STATS: dict[str, int] = {"mean": 0, "std": 1, "median": 2, "25pct": 3, "75pct": 4}
#: The posterior variables the SWE/SCA granules carry.
VARIABLES = ("SWE_Post", "SCA_Post", "SD_Post")
#: More granules than this and ``virtualize="auto"`` turns virtualization on.
VIRTUALIZE_THRESHOLD = 6

_NSIDC_DOCS = "https://nsidc.org/data/wus_ucla_sr/versions/1"
_HMA_DOCS = "https://nsidc.org/data/hma_sr_d/versions/1"
_WATER_YEAR = re.compile(r"_WY(\d{4})_")
_DATE_IN_PATH = re.compile(r"(\d{4})[./](\d{2})[./](\d{2})")

PRODUCT = Product(
    id="ucla-snow-reanalysis",
    theme="snow",
    title="UCLA snow reanalysis (Western US and High Mountain Asia)",
    description=(
        "Daily 480 m posterior SWE, snow depth and snow-covered area from the UCLA "
        "particle-batch-smoother reanalysis: water years 1985–2021 over the western "
        "United States (WUS_UCLA_SR v1) and 2000–2017 over High Mountain Asia "
        "(HMA_SR_D v1), each with five ensemble statistics."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="earthdata",
            location="WUS_UCLA_SR",
            requires=("earthdata",),
            resolution_m=480,
            extent="western US",
            temporal="1984-10/2021-09",
            latency="static",
            notes=(
                "one NetCDF-4 granule per water year and 1° tile; no DMR++ sidecars, "
                "so virtualize='auto' scans HDF5 metadata once and caches references"
            ),
            title="NSIDC (Earthdata)",
            health=(
                Probe(
                    "UCLA Snow Reanalysis (NASA NSIDC)",
                    partial(health.earthdata_search, "WUS_UCLA_SR"),
                ),
                # The one that would change the loader's default: a DMR++
                # sidecar turns virtualize='auto' from an HDF5 metadata scan
                # into a sidecar read (§4.9). NSIDC published none as of
                # 2026-09-15; this is the weekly check for that changing.
                Probe(
                    "UCLA Snow Reanalysis DMR++ (CMR)",
                    partial(
                        health.dmrpp_status,
                        "WUS_UCLA_SR",
                        fallback=health.FALLBACK_PARSERS["NETCDF-4"],
                    ),
                    requires=(),
                    kind="virtualization",
                ),
            ),
        ),
        Source(
            id="nsidc-hma",
            provider="earthdata",
            location="HMA_SR_D",
            requires=("earthdata",),
            resolution_m=480,
            extent="High Mountain Asia",
            temporal="1999-10/2017-09",
            latency="static",
            notes="the High Mountain Asia sibling of the same reanalysis",
            title="NSIDC HMA (Earthdata)",
            health=Probe(
                "HMA Snow Reanalysis (NASA NSIDC)",
                partial(
                    health.earthdata_search,
                    "HMA_SR_D",
                    bbox=(80.0, 30.0, 81.0, 31.0),
                    temporal=("2000-03-01", "2000-03-07"),
                ),
            ),
        ),
    ),
    variables=(
        Variable(
            "SWE_Post",
            units="m",
            dtype="float32",
            long_name="posterior snow water equivalent",
        ),
        Variable(
            "SCA_Post",
            units="1",
            dtype="float32",
            long_name="posterior snow-covered area fraction",
        ),
        Variable(
            "SD_Post", units="m", dtype="float32", long_name="posterior snow depth"
        ),
    ),
    citation=(
        "Fang, Y., Liu, Y. and Margulis, S. A. (2022). Western United States UCLA Daily "
        "Snow Reanalysis, Version 1. NASA NSIDC DAAC. Liu, Y., Fang, Y. and Margulis, "
        "S. A. (2021). High Mountain Asia UCLA Daily Snow Reanalysis, Version 1."
    ),
    license="NASA Earthdata (free registration)",
    doi="10.5067/PP7T2GBI52I2",
    references=(_NSIDC_DOCS, _HMA_DOCS),
    loader="easysnowdata.snow.ucla_sr.load",
    examples=("snow/plot_ucla_sr.py",),
    tags=("swe", "reanalysis", "snow depth", "ucla"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _region(region: str) -> str:
    key = str(region).lower()
    if key not in REGIONS:
        raise ValueError(f"region must be one of {list(REGIONS)}, got {region!r}.")
    return key


def _stats_index(stats: str) -> int:
    if stats not in STATS:
        raise ValueError(f"stats must be one of {list(STATS)}, got {stats!r}.")
    return STATS[stats]


def water_year_start(path: str) -> pd.Timestamp:
    """The 1 October a granule's daily record starts on.

    Taken from the ``_WY1999_`` field of the file name, or from a date in the
    archive path (``.../1999/10/01/...``) when the name does not carry one.
    """
    text = str(path)
    if match := _WATER_YEAR.search(text):
        return pd.Timestamp(year=int(match.group(1)), month=10, day=1)
    if match := _DATE_IN_PATH.search(text):
        return pd.Timestamp(*(int(g) for g in match.groups()))
    raise ValueError(f"Could not determine the water-year start date from: {path}")


def _should_virtualize(virtualize: Any, granule_count: int) -> bool:
    if isinstance(virtualize, str):
        if virtualize.lower() != "auto":
            raise ValueError(
                f"virtualize must be True, False or 'auto', got {virtualize!r}."
            )
        return granule_count > VIRTUALIZE_THRESHOLD
    return bool(virtualize)


def _add_time_coord(ds: xr.Dataset, start: pd.Timestamp) -> xr.Dataset:
    """Turn the granule's ``Day`` index into a real time axis."""
    if "Day" not in ds.dims:
        return ds
    times = pd.date_range(start, periods=ds.sizes["Day"])
    return ds.assign_coords(time=("Day", times)).swap_dims({"Day": "time"})


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    region: str = "wus",
    source: str | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search the reanalysis granules and return them as a ``GeoDataFrame``."""
    key = _region(region)
    src = resolve_source(PRODUCT, source or ("nsidc" if key == "wus" else "nsidc-hma"))
    ensure_source(PRODUCT, src)
    granules = providers.earthdata.search(REGIONS[key][0], aoi, time, **kwargs)
    if max_items is not None:
        granules = granules[:max_items]
    gdf = providers.earthdata.granules_to_geodataframe(granules)
    gdf.attrs = {"source": src.id, "region": key, "short_name": REGIONS[key][0]}
    return gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    variable: str = "SWE_Post",
    stats: str = "mean",
    region: str = "wus",
    source: str | None = None,
    granules: Any = None,
    virtualize: bool | str = "auto",
    access: str = "auto",
    chunks: Any = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Load the reanalysis as a lazy ``xarray.DataArray`` (``time``, ``y``, ``x``).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    variable
        ``"SWE_Post"`` (default), ``"SCA_Post"`` or ``"SD_Post"``.
    stats
        Ensemble statistic: ``"mean"``, ``"std"``, ``"median"``, ``"25pct"``
        or ``"75pct"``.
    region
        ``"wus"`` (default) or ``"hma"``.
    virtualize
        ``"auto"`` (default) virtualizes past
        :data:`VIRTUALIZE_THRESHOLD` granules, which is where the one-off
        HDF5 metadata scan pays for itself; ``True``/``False`` force it.
    access
        ``"auto"`` (default) uses NASA's in-region S3 credentials when this
        process runs in ``us-west-2`` and HTTPS everywhere else;
        ``"direct"``/``"indirect"`` force it.

    Returns
    -------
    xarray.DataArray
        The chosen variable and statistic, clipped to the AOI.
    """
    key = _region(region)
    src = resolve_source(PRODUCT, source or ("nsidc" if key == "wus" else "nsidc-hma"))
    if variable not in VARIABLES:
        raise ValueError(f"variable must be one of {VARIABLES}, got {variable!r}.")
    index = _stats_index(stats)  # validated before any network call
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if granules is None:
        granules = providers.earthdata.search(REGIONS[key][0], aoi, time, **kwargs)
    if not len(granules):
        raise ValueError(
            f"No {REGIONS[key][0]} granules for this AOI and time "
            f"({REGIONS[key][2]} is the coverage)."
        )
    mode = _resolve_access(access)
    use_virtual = _should_virtualize(virtualize, len(granules))
    _logger.info(
        "Loading %d %s granule(s) (virtualize=%s, access=%s).",
        len(granules),
        REGIONS[key][0],
        use_virtual,
        mode,
    )
    ds = (
        _open_virtual(granules, mode, chunks)
        if use_virtual
        else _open_direct(granules, chunks)
    )

    da = ds[variable]
    if "Stats" in da.dims:
        da = da.isel(Stats=index, drop=True)
    da = da.rename({"Latitude": "latitude", "Longitude": "longitude"})
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    da = da.rio.write_crs("EPSG:4326")
    if parsed is not None and parsed.clip and not parsed.is_global:
        west, south, east, north = parsed.total_bounds()
        da = da.rio.clip_box(
            west, south, east, north, allow_one_dimensional_raster=True
        )
    if time is not None:
        start, end = temporal.parse_time(time)
        da = da.sel(time=slice(start, end))
    if chunks is not None:
        da = da.chunk(chunks)
    elif da.chunks is None:
        da = da.chunk({"time": 31})

    da = contract.finalize(
        da,
        PRODUCT,
        src,
        crs="EPSG:4326",
        variables=(),
        source_url=_NSIDC_DOCS if key == "wus" else _HMA_DOCS,
        attrs={
            "region": key,
            "short_name": REGIONS[key][0],
            "statistic": stats,
            "variable": variable,
            "virtualized": str(use_virtual),
            "access": mode,
            **_variable_attrs(variable),
        },
    )
    return da


def _variable_attrs(variable: str) -> dict[str, Any]:
    """The catalog's CF attrs for one reanalysis variable."""
    for definition in PRODUCT.variables:
        if definition.name == variable:
            return definition.cf_attrs()
    return {}


def _resolve_access(access: str) -> str:
    """``"direct"`` only where NASA's in-region S3 credentials work (us-west-2)."""
    if access not in ("auto", "direct", "indirect"):
        raise ValueError(
            f"access must be 'auto', 'direct' or 'indirect', got {access!r}."
        )
    if access != "auto":
        return access
    return "direct" if config.region(probe=True).direct_s3 else "indirect"


def _open_direct(granules: Sequence[Any], chunks: Any) -> xr.Dataset:
    """``earthaccess.open`` plus ``open_mfdataset`` — the path for a few granules."""
    files = providers.earthdata.open(list(granules))
    datasets = []
    for handle in files:
        path = getattr(handle, "path", None) or getattr(
            handle, "full_name", str(handle)
        )
        ds = xr.open_dataset(handle, chunks={} if chunks is None else chunks)
        datasets.append(_add_time_coord(ds, water_year_start(path)))
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")


def _open_virtual(granules: Sequence[Any], access: str, chunks: Any) -> xr.Dataset:
    """``earthaccess.virtualize`` with a reference cache in the easysnowdata cache.

    The granules carry no DMR++ sidecars, so earthaccess falls back to the
    HDF5 parser: one metadata scan per granule, written to ``reference_dir``
    and reused on the next call.
    """
    import earthaccess  # noqa: PLC0415

    reference_dir = config.cache_dir("virtual", "ucla_sr")
    try:
        ds = earthaccess.virtualize(
            list(granules),
            access=access,
            load=True,
            concat_dim="Day",
            reference_dir=str(reference_dir),
        )
    except Exception as exc:  # noqa: BLE001 — virtualization is an optimisation
        _logger.warning(
            "Virtualization failed (%s); falling back to earthaccess.open().", exc
        )
        return _open_direct(granules, chunks)
    starts = sorted(water_year_start(_granule_path(g)) for g in granules)
    if "Day" in ds.dims:
        days = ds.sizes["Day"]
        per_year = days // max(len(starts), 1)
        times = np.concatenate(
            [pd.date_range(start, periods=per_year).values for start in starts]
        )[:days]
        ds = ds.assign_coords(time=("Day", times)).swap_dims({"Day": "time"})
    return ds


def _granule_path(granule: Any) -> str:
    """A file path or URL for a granule, whatever earthaccess handed back."""
    for attribute in ("path", "full_name"):
        value = getattr(granule, attribute, None)
        if value:
            return str(value)
    try:
        links = granule.data_links()
    except Exception:  # noqa: BLE001
        links = []
    if links:
        return str(links[0])
    if isinstance(granule, dict):
        return str(granule.get("id") or granule.get("path") or granule)
    return str(granule)
