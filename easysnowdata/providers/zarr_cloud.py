"""Anonymous cloud Zarr stores (GCS, S3, Azure, HTTPS) with xarray."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from easysnowdata import temporal
from easysnowdata.aoi import parse_aoi

__all__ = ["storage_options_for", "open", "select"]

_logger = logging.getLogger(__name__)


def storage_options_for(url: str) -> dict[str, Any]:
    """Anonymous-access ``storage_options`` for the store's scheme."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme in ("gs", "gcs"):
        return {"token": "anon"}
    if scheme in ("s3", "s3a"):
        return {"anon": True}
    if scheme in ("az", "abfs", "abfss"):
        return {"anon": True}
    return {}


def open(  # noqa: A001 — mirrors xarray.open_zarr
    url: str,
    *,
    storage_options: dict[str, Any] | None = None,
    chunks: Any = None,
    consolidated: bool | None = None,
    **kwargs: Any,
) -> Any:
    """``xarray.open_zarr`` with anonymous defaults for public buckets.

    ``chunks=None`` (default) opens lazily without Dask; pass ``chunks={}`` for
    the store's native chunking. Extra keyword arguments go to ``open_zarr``.
    """
    import xarray as xr  # noqa: PLC0415

    options = {**storage_options_for(url), **(storage_options or {})}
    params: dict[str, Any] = {"chunks": chunks, "consolidated": consolidated, **kwargs}
    if options:
        params.setdefault("storage_options", options)
    return xr.open_zarr(url, **params)


def select(
    ds: Any,
    aoi: Any = None,
    time: Any = None,
    *,
    lon: str = "longitude",
    lat: str = "latitude",
    time_dim: str = "time",
) -> Any:
    """Subset a global lat/lon dataset to an AOI and time range.

    Handles 0–360° longitude grids (ARCO-ERA5) and descending latitudes, and
    an AOI across the antimeridian.
    """
    if time is not None and time_dim in ds.dims:
        start, end = temporal.parse_time(time)
        ds = ds.sel({time_dim: slice(start, end)})
    if aoi is None:
        return ds
    parsed = parse_aoi(aoi)
    if parsed.is_global:
        return ds
    west, south, east, north = parsed.total_bounds()  # unwrapped across ±180°
    lons = ds[lon].values
    if lons.size and float(np.nanmax(lons)) > 180.0:  # 0..360 grid
        west %= 360.0
        east %= 360.0
        if east < west:  # crosses 0° on a 0..360 grid
            east += 360.0
        if east > 360.0:
            return _concat_lon(
                ds, (west, 360.0), (0.0, east - 360.0), lon, lat, south, north
            )
    elif east > 180.0:  # AOI across the antimeridian on a -180..180 grid
        return _concat_lon(
            ds, (west, 180.0), (-180.0, east - 360.0), lon, lat, south, north
        )
    return _sel_box(ds, lon, lat, west, east, south, north)


def _sel_box(
    ds: Any, lon: str, lat: str, west: float, east: float, south: float, north: float
) -> Any:
    lats = ds[lat].values
    lat_slice = (
        slice(north, south)
        if lats.size > 1 and lats[0] > lats[-1]
        else slice(south, north)
    )
    return ds.sel({lon: slice(west, east), lat: lat_slice})


def _concat_lon(
    ds: Any,
    a: tuple[float, float],
    b: tuple[float, float],
    lon: str,
    lat: str,
    south: float,
    north: float,
) -> Any:
    import xarray as xr  # noqa: PLC0415

    parts = [_sel_box(ds, lon, lat, w, e, south, north) for w, e in (a, b)]
    return xr.concat(parts, dim=lon)
