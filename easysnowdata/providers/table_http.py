"""Small text tables over HTTPS (climate-index files), cached with pooch.

The files are a few kilobytes and are republished in place every month, so
they are downloaded whole with a plain GET and re-fetched once the cached copy
is older than *max_age*. Parsing is the product module's job; this layer only
moves bytes, like the other providers.
"""

from __future__ import annotations

from pathlib import Path

from easysnowdata.providers import raster_http

__all__ = ["fetch"]


def fetch(
    url: str,
    fname: str | None = None,
    *,
    subdir: str | None = None,
    max_age: float | None = 86400,
) -> Path:
    """Download *url* into the cache and return the local path.

    A thin wrapper over :func:`easysnowdata.providers.raster_http.fetch` with
    the progress bar off and a one-day *max_age* by default, because these
    files change in place (``max_age=0`` always downloads).
    """
    return raster_http.fetch(
        url, fname, subdir=subdir, progressbar=False, max_age=max_age
    )
