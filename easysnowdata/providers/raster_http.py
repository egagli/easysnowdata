"""COGs and zipped GeoTIFFs over HTTPS (rioxarray), plus a pooch download cache."""

from __future__ import annotations

import logging
import re
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import url2pathname

from easysnowdata import auth, config
from easysnowdata._gdal import gdal_env
from easysnowdata.aoi import parse_aoi

__all__ = ["open", "zip_url", "fetch"]

_logger = logging.getLogger(__name__)

#: "C:/…" — a Windows drive-lettered path, which must not keep a leading slash.
_DRIVE = re.compile(r"^[A-Za-z]:/")


def zip_url(url: str, member: str) -> str:
    """GDAL path for *member* inside the zip at *url*.

    Remote archives get rasterio's ``zip+https://…!/member`` spelling, which it
    expands to ``/vsizip/vsicurl/https://…``.

    A **local** archive is spelled as ``/vsizip/`` directly instead of going
    through a ``file://`` URI, because rasterio renders
    ``zip+file:///C:/x.zip!/m.tif`` as ``/vsizip//C:/x.zip/m.tif`` — with two
    slashes. On POSIX that extra slash is harmless (``//tmp/x.zip`` is still
    ``/tmp/x.zip``); on Windows ``//C:/…`` is not a path GDAL can open. Only
    local archives are affected, so this never reached a user — the products
    all read remote zips — but it is what made the offline tier fail on
    Windows the first time CI ran it.
    """
    member = member.lstrip("/")
    text = str(url)
    scheme = urllib.parse.urlsplit(text).scheme
    if scheme == "file":
        local = url2pathname(urllib.parse.urlsplit(text).path)
    elif scheme == "" or len(scheme) == 1:  # a bare path, or a Windows drive letter
        local = text
    else:
        return f"zip+{text}!/{member}"
    path = PurePosixPath(Path(local).as_posix())
    # A POSIX absolute path keeps its leading slash — /vsizip//tmp/x.zip is how
    # an absolute path is spelled, and /vsizip/tmp/x.zip would be relative. A
    # drive-lettered path must not gain one. url2pathname only strips the
    # leading slash from a drive letter when it runs on Windows, so do it here
    # rather than depending on the platform.
    text_path = str(path)
    if _DRIVE.match(text_path.lstrip("/")):
        text_path = text_path.lstrip("/")
    return f"/vsizip/{text_path}/{member}"


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


USER_AGENT = "easysnowdata (+https://github.com/egagli/easysnowdata)"


def fetch(
    url: str,
    fname: str | None = None,
    *,
    subdir: str | None = None,
    known_hash: str | None = None,
    progressbar: bool = True,
    max_age: float | None = None,
) -> Path:
    """Download *url* once into the cache with a plain GET and return the local path.

    For static archives whose hosts reject range requests or the HEAD request
    GDAL's ``/vsicurl`` sends first (GRDC). ``EASYSNOWDATA_CACHE_DIR`` moves
    the cache root.

    Without a *known_hash* pooch never re-downloads an existing file, which is
    right for a static release and wrong for an artefact that is republished:
    *max_age* (in seconds) re-fetches a cached copy older than that, so the
    daily-rebuilt station archive cannot lag its own inventory for ever on a
    machine that downloaded it once.
    """
    import time  # noqa: PLC0415

    import pooch  # noqa: PLC0415

    path = config.cache_dir(*([subdir] if subdir else []))
    target = Path(path) / (fname or url.rsplit("/", 1)[-1])
    if max_age is not None and target.exists():
        age = time.time() - target.stat().st_mtime
        if age > max_age:
            _logger.info(
                "Cached %s is %.0f h old (limit %.0f h); fetching it again.",
                target.name,
                age / 3600,
                max_age / 3600,
            )
            target.unlink()
    local = pooch.retrieve(
        url,
        known_hash=known_hash,
        fname=fname or url.rsplit("/", 1)[-1],
        path=path,
        # Name ourselves: some hosts (HydroSHEDS from cloud runners) answer
        # 403 to the default python-requests agent.
        downloader=pooch.HTTPDownloader(
            progressbar=progressbar, headers={"User-Agent": USER_AGENT}
        ),
    )
    return Path(local)
