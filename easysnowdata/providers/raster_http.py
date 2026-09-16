"""COGs and zipped GeoTIFFs over HTTPS (rioxarray), plus a pooch download cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from easysnowdata import auth, config
from easysnowdata._gdal import gdal_env
from easysnowdata.aoi import parse_aoi

__all__ = ["open", "zip_url", "fetch"]

_logger = logging.getLogger(__name__)


def zip_url(url: str, member: str) -> str:
    """GDAL path for *member* inside the remote zip at *url* (``zip+https://…!/member``)."""
    return f"zip+{url}!/{member.lstrip('/')}"


def open(  # noqa: A001 — mirrors rioxarray.open_rasterio
    url: str | Path,
    aoi: Any = None,
    *,
    chunks: Any = True,
    mask_and_scale: bool = False,
    requires: tuple[str, ...] = (),
    gdal: dict[str, Any] | None = None,
    squeeze: bool = True,
    **kwargs: Any,
) -> Any:
    """Open a raster with ``rioxarray.open_rasterio`` inside the read environment.

    Parameters
    ----------
    url
        Local path, ``https://`` URL or ``zip+https://…!/member`` path.
    aoi
        When given and ``clip=True`` the raster is clipped (``rio.clip_box``)
        to the AOI bounds; ``clip=False`` returns the whole raster.
    chunks, mask_and_scale, **kwargs
        Passed to ``rioxarray.open_rasterio`` (``chunks=True`` → Dask).
    requires, gdal
        Auth providers whose ``env()`` the read needs, and extra GDAL options.
    squeeze
        Drop the length-one ``band`` dimension.
    """
    import rioxarray as rxr  # noqa: PLC0415

    params: dict[str, Any] = {
        "chunks": chunks,
        "mask_and_scale": mask_and_scale,
        **kwargs,
    }
    with auth.env(*requires), gdal_env(**(gdal or {})):
        da = rxr.open_rasterio(str(url), **params)
        if aoi is not None:
            parsed = parse_aoi(aoi)
            if parsed.clip and not parsed.is_global:
                da = da.rio.clip_box(
                    *parsed.footprint.total_bounds,
                    crs=parsed.footprint.crs,
                    auto_expand=True,
                )
    if squeeze and "band" in getattr(da, "dims", ()) and da.sizes.get("band") == 1:
        da = da.squeeze("band", drop=True)
    return da


def fetch(
    url: str,
    fname: str | None = None,
    *,
    subdir: str | None = None,
    known_hash: str | None = None,
    progressbar: bool = True,
) -> Path:
    """Download *url* once into the cache with a plain GET and return the local path.

    For static archives whose hosts reject range requests or the HEAD request
    GDAL's ``/vsicurl`` sends first (GRDC). ``EASYSNOWDATA_CACHE_DIR`` moves
    the cache root.
    """
    import pooch  # noqa: PLC0415

    path = config.cache_dir(*([subdir] if subdir else []))
    local = pooch.retrieve(
        url,
        known_hash=known_hash,
        fname=fname or url.rsplit("/", 1)[-1],
        path=path,
        progressbar=progressbar,
    )
    return Path(local)
