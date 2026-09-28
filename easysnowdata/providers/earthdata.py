"""NASA Earthdata via ``earthaccess``: search, open, download (cached)."""

from __future__ import annotations

import contextlib
import logging
import urllib.parse
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from easysnowdata import auth, config, temporal
from easysnowdata.aoi import parse_aoi
from easysnowdata.auth.earthdata import NETRC_HOST

__all__ = [
    "ensure",
    "env",
    "search",
    "granules_to_geodataframe",
    "open",
    "download",
    "download_with_password",
    "auth_host_in",
    "hdf4_available",
    "require_hdf4",
]

_logger = logging.getLogger(__name__)


def ensure() -> Any:
    """Log in (once) and return the ``earthaccess.Auth``."""
    return auth.get("earthdata").ensure()


@contextlib.contextmanager
def env() -> Iterator[dict[str, Any]]:
    """GDAL environment for ``/vsicurl`` reads of EDL-protected files."""
    with auth.get("earthdata").env() as options:
        yield options


def search(
    short_name: str,
    aoi: Any = None,
    time: Any = None,
    *,
    cloud_hosted: bool = True,
    count: int = -1,
    **kwargs: Any,
) -> list[Any]:
    """``earthaccess.search_data`` with the package's AOI/time inputs.

    The bounding box is the AOI's EPSG:4326 bounds (CMR cannot take a polygon
    across the antimeridian in one query, so such AOIs search the unwrapped
    bounds); ``time`` becomes ``temporal=(start, end)``.
    """
    ensure()
    import earthaccess  # noqa: PLC0415

    params: dict[str, Any] = {"short_name": short_name, "count": count}
    if cloud_hosted:
        params["cloud_hosted"] = True
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            west, south, east, north = parsed.bounds
            if parsed.crosses_antimeridian:
                west, south, east, north = parsed.total_bounds()
                east = min(east, 180.0)  # CMR wants -180..180
            params["bounding_box"] = (west, south, east, north)
    if time is not None:
        start, end = temporal.parse_time(time)
        params["temporal"] = (
            None if start is None else start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
    params.update(kwargs)
    _logger.debug("earthaccess.search_data(%s)", params)
    return earthaccess.search_data(**params)


def _granule_geometry(granule: Any) -> shapely.Geometry | None:
    try:
        extent = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
    except (KeyError, TypeError):
        return None
    polys = []
    for gpoly in extent.get("GPolygons", []):
        points = gpoly.get("Boundary", {}).get("Points", [])
        if len(points) >= 3:
            polys.append(
                shapely.Polygon([(p["Longitude"], p["Latitude"]) for p in points])
            )
    for box in extent.get("BoundingRectangles", []):
        polys.append(
            shapely.box(
                box["WestBoundingCoordinate"],
                box["SouthBoundingCoordinate"],
                box["EastBoundingCoordinate"],
                box["NorthBoundingCoordinate"],
            )
        )
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def granules_to_geodataframe(granules: Iterable[Any]) -> gpd.GeoDataFrame:
    """Granules → GeoDataFrame: ``id``, ``start``, ``end``, ``size_mb``, ``data_links``, geometry."""
    rows = []
    for g in granules:
        umm = g.get("umm", {}) if hasattr(g, "get") else {}
        temporal_extent = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        try:
            links = g.data_links()
        except Exception:  # noqa: BLE001
            links = []
        try:
            size = (
                float(g.size())
                if callable(getattr(g, "size", None))
                else float(g.get("size", "nan"))
            )
        except Exception:  # noqa: BLE001
            size = float("nan")
        rows.append(
            {
                "id": umm.get("GranuleUR") or g.get("meta", {}).get("native-id"),
                "start": temporal_extent.get("BeginningDateTime"),
                "end": temporal_extent.get("EndingDateTime"),
                "size_mb": size,
                "data_links": links,
                "geometry": _granule_geometry(g),
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            {"id": [], "start": [], "end": [], "size_mb": [], "data_links": []},
            geometry=[],
            crs="EPSG:4326",
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    for col in ("start", "end"):
        gdf[col] = pd.to_datetime(gdf[col], utc=True, errors="coerce")
    return gdf


def open(granules: list[Any], **kwargs: Any) -> list[Any]:  # noqa: A001 — mirrors earthaccess.open
    """``earthaccess.open`` (fsspec file objects) after an explicit login."""
    ensure()
    import earthaccess  # noqa: PLC0415

    return earthaccess.open(granules, **kwargs)


def download(granules: list[Any], subdir: str, **kwargs: Any) -> list[Path]:
    """``earthaccess.download`` into the easysnowdata cache (``<cache>/<subdir>``).

    Files already present are skipped by earthaccess. Override the cache root
    with ``EASYSNOWDATA_CACHE_DIR``.
    """
    ensure()
    import earthaccess  # noqa: PLC0415

    target = config.cache_dir(subdir)
    return [Path(p) for p in earthaccess.download(granules, target, **kwargs)]


def download_with_password(
    url: str,
    subdir: str,
    *,
    valid: Callable[[Path], bool] | None = None,
    timeout: float = 120,
) -> Path:
    """Download one file from a server that refuses a bearer token.

    NSIDC's on-premises ``daacdata`` tree answers a token-only request with the
    URS login page and HTTP 200, which ``earthaccess.download`` saves as if it
    were the file. This goes through
    :meth:`~easysnowdata.auth.earthdata.EarthdataProvider.password_session`
    instead, so the username and password are used even when
    ``EARTHDATA_TOKEN`` is also set.

    The file lands in ``<cache>/<subdir>`` and is reused on the next call. A
    cached file that fails *valid*, such as a login page an older version saved,
    is fetched again. A 404 raises :class:`FileNotFoundError`; a download that
    ends on URS, or that fails *valid*, raises instead of being cached.
    """
    target = config.cache_dir(subdir) / url.rsplit("/", 1)[-1]
    if target.is_file() and target.stat().st_size and (valid is None or valid(target)):
        return target
    session = auth.get("earthdata").password_session()
    partial_path = target.with_name(target.name + ".part")
    with session.get(
        url, stream=True, timeout=timeout, allow_redirects=True
    ) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"{url} does not exist (HTTP 404).")
        if auth_host_in(response.url) or response.status_code != 200:
            raise RuntimeError(
                f"Earthdata Login refused {url} (HTTP {response.status_code}, ended at "
                f"{response.url.split('?', 1)[0]}); check the username and password."
            )
        with partial_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    if valid is not None and not valid(partial_path):
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"{url} did not return the expected file.")
    partial_path.replace(target)
    return target


def auth_host_in(url: str) -> bool:
    """Whether *url* is on the Earthdata Login host (a request that ended at the login page)."""
    return urllib.parse.urlparse(url).hostname == NETRC_HOST


def hdf4_available() -> bool:
    """Whether this GDAL build can read HDF4 (HDF-EOS2) files."""
    import rasterio  # noqa: PLC0415

    with rasterio.Env() as env:
        return "HDF4" in env.drivers()


def require_hdf4(product: str) -> None:
    """Fail fast when *product* is HDF4 and GDAL has no HDF4 driver."""
    if not hdf4_available():
        raise RuntimeError(
            f"{product} granules are HDF4 (HDF-EOS2) files, but this GDAL build has no "
            "HDF4 driver. rasterio's PyPI wheels omit it; install easysnowdata from "
            "conda-forge (which pulls in libgdal-hdf4), or `conda install -c conda-forge "
            "libgdal-hdf4` into a conda environment that provides GDAL."
        )
