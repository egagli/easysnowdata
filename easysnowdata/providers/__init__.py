"""Thin, generic adapters over the access libraries — no snow knowledge here.

* :mod:`~easysnowdata.providers.stac` — pystac-client + odc-stac (Planetary
  Computer, Earth Search, CMR-STAC)
* :mod:`~easysnowdata.providers.earthdata` — earthaccess search / open / download
* :mod:`~easysnowdata.providers.gee` — Earth Engine via xee at the native grid
* :mod:`~easysnowdata.providers.raster_http` — COGs and zipped GeoTIFFs over
  HTTPS, with a pooch cache
* :mod:`~easysnowdata.providers.zarr_cloud` — anonymous cloud Zarr stores
* :mod:`~easysnowdata.providers.vector_http` — remote GDB / GeoJSON / GeoParquet
  with bbox or mask pushdown

Every read happens inside :func:`easysnowdata._gdal.gdal_env` (and the auth
providers' ``env()``); nothing here configures GDAL, xarray or "today" at
import time (§2.9).
"""

from __future__ import annotations

from easysnowdata._gdal import CLOUD_DEFAULTS, gdal_env
from easysnowdata.providers import (
    earthdata,
    gee,
    raster_http,
    stac,
    vector_http,
    zarr_cloud,
)

__all__ = [
    "CLOUD_DEFAULTS",
    "gdal_env",
    "stac",
    "earthdata",
    "gee",
    "raster_http",
    "zarr_cloud",
    "vector_http",
]
