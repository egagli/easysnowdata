"""GDAL/rasterio configuration, applied only inside context managers (§2.9).

Two mechanisms are needed because reads happen at two different times:

* :class:`rasterio.Env` covers eager reads made *inside* the ``with`` block
  (``rioxarray.open_rasterio`` opening a ``/vsicurl/`` COG).
* :func:`odc.stac.configure_rio` registers the same options with odc-stac /
  odc-loader, which wraps every lazy Dask chunk read in its own
  ``rasterio.Env`` — and forwards them to Dask workers when a distributed
  client is given. That is how lazily-loaded EDL-protected COGs (HLS, OPERA)
  keep working after the ``with`` block ends.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

__all__ = ["CLOUD_DEFAULTS", "gdal_env", "dask_client"]

_logger = logging.getLogger(__name__)

# GDAL's default is zero HTTP retries; cloud object stores throttle (429) and
# hiccup (5xx) often enough that a few retries turn flaky reads into slow ones.
CLOUD_DEFAULTS: dict[str, Any] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "CPL_VSIL_CURL_CHUNK_SIZE": str(4 * 1024 * 1024),
}


def dask_client() -> Any | None:
    """Return the default ``distributed.Client`` if one exists, else ``None``."""
    try:
        import distributed  # noqa: PLC0415

        return distributed.default_client()
    except Exception:
        return None


@contextlib.contextmanager
def gdal_env(
    *, cloud_defaults: bool = True, lazy: bool = True, **options: Any
) -> Iterator[dict[str, Any]]:
    """Apply GDAL options for the duration of the block (and to lazy reads).

    Parameters
    ----------
    cloud_defaults
        Start from :data:`CLOUD_DEFAULTS`.
    lazy
        Also register the options with odc-stac (``configure_rio``) so Dask
        chunk reads scheduled inside the block — and on a distributed client,
        if one exists — see them later.
    **options
        Extra GDAL configuration options; ``None`` values are dropped.

    Yields
    ------
    dict
        The effective options.
    """
    import rasterio  # noqa: PLC0415

    effective: dict[str, Any] = dict(CLOUD_DEFAULTS) if cloud_defaults else {}
    effective.update({k: v for k, v in options.items() if v is not None})
    if lazy:
        _configure_odc(effective)
    with rasterio.Env(**effective):
        yield effective


def _configure_odc(options: dict[str, Any]) -> None:
    try:
        import odc.stac  # noqa: PLC0415
    except ImportError:  # pragma: no cover — odc-stac is a hard dependency
        return
    kwargs: dict[str, Any] = dict(options)
    client = dask_client()
    if client is not None:
        kwargs["client"] = client
    try:
        odc.stac.configure_rio(cloud_defaults=False, **kwargs)
    except Exception as exc:  # pragma: no cover — never fail a read on config
        _logger.debug("odc.stac.configure_rio failed: %s", exc)
