"""Shared utility functions used across easysnowdata modules.

Credential handling lives in :mod:`easysnowdata.auth` and the cache directory
in :mod:`easysnowdata.config`; the names kept here are thin delegates so the
existing modules and tests keep working.
"""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import requests
import shapely
import yaml
from bs4 import BeautifulSoup

from easysnowdata import auth, config
from easysnowdata.auth import CredentialError
from easysnowdata.auth.earthengine import (
    credentials_from_token as _ee_credentials_from_token,
)
from easysnowdata.auth.earthengine import decode_token as _decode_ee_token

if TYPE_CHECKING:
    import ee

__all__ = [
    "CredentialError",
    "requires_earthengine",
    "requires_earthaccess",
    "initialize_earthengine",
    "suppress_stdout",
    "convert_bbox_to_geodataframe",
    "get_ee_grid_params",
    "get_stac_cfg",
    "get_water_year_start",
    "datetime_to_DOWY",
    "datetime_to_WY",
    "HLS_xml_url_to_metadata_df",
]
_ = (_decode_ee_token, _ee_credentials_from_token)  # re-exported for back-compat

_logger = logging.getLogger(__name__)

# Setup texts now live on the providers; kept under the old names.
_EE_SETUP_MSG = auth.get("earthengine").setup_instructions
_EARTHACCESS_SETUP_MSG = auth.get("earthdata").setup_instructions
_EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"


# ── Credential detection (delegates to easysnowdata.auth) ─────────────────────


def _has_earthengine_credentials() -> bool:
    """Return True if EE credentials can be found (env var, ADC, or credential file)."""
    return bool(auth.detect("earthengine"))


def _has_earthaccess_credentials() -> bool:
    """Return True if NASA Earthdata credentials can be found (env vars or ~/.netrc)."""
    return bool(auth.detect("earthdata"))


def _earthaccess_login():
    """Log in to NASA Earthdata through ``earthaccess`` if not already done.

    Delegates to the ``earthdata`` auth provider (explicit ``earthaccess.login``
    with the environment-then-netrc strategy, store check, one retry). Never
    prompts.

    Raises
    ------
    CredentialError
        If no credentials are found or Earthdata Login rejects them.
    """
    if not _has_earthaccess_credentials():
        raise CredentialError(
            f"NASA EarthData credentials are required.\n\n{_EARTHACCESS_SETUP_MSG}",
            provider="earthdata",
        )
    return auth.get("earthdata").ensure()


def initialize_earthengine(**kwargs) -> None:
    """Initialise Google Earth Engine once, honouring ``EARTHENGINE_TOKEN`` if set.

    Delegates to the ``earthengine`` auth provider: token (service-account or
    OAuth JSON, raw or base64), Application Default Credentials, or the
    ``ee.Authenticate()`` file; high-volume endpoint unless ``opt_url``/``url``
    is given; a Cloud project from ``EE_PROJECT_ID``/``EARTHENGINE_PROJECT``,
    the token, or the credentials file. Extra keyword arguments are passed to
    ``ee.Initialize`` and force a re-initialisation.
    """
    auth.get("earthengine").ensure(**kwargs)


# ── Local cache (delegates to easysnowdata.config) ────────────────────────────


def _cache_dir(*subdirs: str) -> Path:
    """Return (and create) the easysnowdata cache directory (see ``config.cache_dir``)."""
    return config.cache_dir(*subdirs)


def _fetch_to_cache(
    url: str, fname: str, subdir: str | None = None, progressbar: bool = True
) -> Path:
    """Download *url* into the cache with a plain GET and return the local path.

    The download is skipped when the file is already present. Used for static
    archives whose hosts do not support range requests or reject the HEAD
    request that GDAL's ``/vsicurl`` sends first (e.g. GRDC).
    """
    path = _cache_dir(*([subdir] if subdir else []))
    local = pooch.retrieve(
        url, known_hash=None, fname=fname, path=path, progressbar=progressbar
    )
    return Path(local)


# ── Auth decorators ───────────────────────────────────────────────────────────


def requires_earthengine(func):
    """Decorator: raise CredentialError with setup instructions if EE credentials are missing."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_earthengine_credentials():
            raise CredentialError(
                f"`{func.__qualname__}` requires Google Earth Engine.\n\n{_EE_SETUP_MSG}",
                provider="earthengine",
            )
        return func(*args, **kwargs)

    return wrapper


def requires_earthaccess(func):
    """Decorator: raise CredentialError with setup instructions if EarthData credentials are missing."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_earthaccess_credentials():
            raise CredentialError(
                f"`{func.__qualname__}` requires NASA EarthData credentials.\n\n{_EARTHACCESS_SETUP_MSG}",
                provider="earthdata",
            )
        return func(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def suppress_stdout():
    """Context manager that silences stdout for noisy third-party calls."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def convert_bbox_to_geodataframe(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None,
) -> gpd.GeoDataFrame:
    """Convert a bounding-box input of any supported type to a GeoDataFrame.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.geometry or None
        Accepted forms:

        * ``geopandas.GeoDataFrame`` — returned unchanged.
        * 4-element tuple ``(xmin, ymin, xmax, ymax)`` in EPSG:4326.
        * Any Shapely geometry — wrapped in a single-row GeoDataFrame.
        * ``None`` — returns a GeoDataFrame covering the entire world.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame in EPSG:4326.
    """
    if bbox_input is None:
        _logger.debug("No bbox_input provided — using global extent.")
        return gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(-180, -90, 180, 90)], crs="EPSG:4326"
        )
    if isinstance(bbox_input, gpd.GeoDataFrame):
        return bbox_input
    if isinstance(bbox_input, tuple) and len(bbox_input) == 4:
        return gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(*bbox_input)], crs="EPSG:4326"
        )
    if isinstance(bbox_input, shapely.geometry.base.BaseGeometry):
        return gpd.GeoDataFrame(geometry=[bbox_input], crs="EPSG:4326")
    raise TypeError(
        f"Unsupported bbox_input type: {type(bbox_input)}. "
        "Expected GeoDataFrame, 4-tuple, Shapely geometry, or None."
    )


def get_ee_grid_params(
    ee_obj: ee.Image | ee.ImageCollection,
    bbox_gdf: gpd.GeoDataFrame | None = None,
) -> dict:
    """Build the pixel-grid kwargs required by ``xarray.open_dataset(engine="ee")``.

    xee >= 0.1 no longer accepts ``geometry`` / ``scale`` / ``projection``; the
    output grid must instead be given explicitly as ``crs``, ``crs_transform``
    and ``shape_2d``. This helper derives those from the *native* grid of an
    Earth Engine object and, optionally, crops the grid to a bounding box.

    Parameters
    ----------
    ee_obj : ee.Image or ee.ImageCollection
        Object whose native projection defines the grid. For a collection the
        first band of the first image is used (via
        ``xee.helpers.extract_grid_params``).
    bbox_gdf : geopandas.GeoDataFrame, optional
        Area of interest, in any CRS. The grid is cropped to the smallest block
        of native pixels that fully covers it, so the returned pixels are exact
        native values rather than a resampled copy. ``None`` returns the full
        native grid.

    Returns
    -------
    dict
        ``{"crs": str, "crs_transform": tuple, "shape_2d": (width, height)}`` —
        unpack directly into ``xarray.open_dataset(..., engine="ee", **grid)``.
    """
    from xee import helpers as xee_helpers  # noqa: PLC0415

    native = xee_helpers.extract_grid_params(ee_obj)
    if bbox_gdf is None:
        return dict(native)

    a, b, c, d, e, f = native["crs_transform"][:6]
    if b or d:
        raise ValueError("Rotated Earth Engine grids are not supported.")

    geom = bbox_gdf.geometry
    if geom.crs is None:
        geom = geom.set_crs("EPSG:4326")
    # Densify the outline so curved edges survive reprojection to projected CRSs.
    xmin0, ymin0, xmax0, ymax0 = geom.total_bounds
    seg = max(xmax0 - xmin0, ymax0 - ymin0) / 100 or 1.0
    x_min, y_min, x_max, y_max = geom.segmentize(seg).to_crs(native["crs"]).total_bounds

    # Pixel indices of the bbox edges on the native grid, expanded outward.
    eps = 1e-9  # tolerate float noise when an edge sits exactly on a pixel boundary
    cols = ((x_min - c) / a, (x_max - c) / a)
    rows = ((y_min - f) / e, (y_max - f) / e)
    col0 = math.floor(min(cols) + eps)
    col1 = math.ceil(max(cols) - eps)
    row0 = math.floor(min(rows) + eps)
    row1 = math.ceil(max(rows) - eps)

    # Pixels outside the asset footprint come back as NaN (as with xee < 0.1),
    # so the grid is not clamped to the native extent; just guarantee >= 1 pixel.
    col1 = max(col1, col0 + 1)
    row1 = max(row1, row0 + 1)

    return {
        "crs": native["crs"],
        "crs_transform": (a, 0.0, c + col0 * a, 0.0, e, f + row0 * e),
        "shape_2d": (col1 - col0, row1 - row0),
    }


def get_stac_cfg(sensor: str = "sentinel-2-l2a") -> dict:
    """Return an ODC-STAC band configuration dict for common sensors.

    Parameters
    ----------
    sensor : str, optional
        Sensor identifier. Supported values: ``"sentinel-2-l2a"``,
        ``"HLSL30_2.0"``, ``"HLSS30_2.0"``. Default is ``"sentinel-2-l2a"``.

    Returns
    -------
    dict
        STAC configuration dict suitable for ``odc.stac.load(stac_cfg=...)``.

    Raises
    ------
    ValueError
        If *sensor* is not a recognised identifier.
    """
    if sensor == "sentinel-2-l2a":
        cfg = """---
        sentinel-2-l2a:
            assets:
                '*':
                    data_type: uint16
                    nodata: 0
                    unit: '1'
                scl:
                    data_type: uint8
                    nodata: 0
                    unit: '1'
                visual:
                    data_type: uint8
                    nodata: 0
                    unit: '1'
            aliases:
                costal: B01
                blue: B02
                green: B03
                red: B04
                rededge1: B05
                rededge2: B06
                rededge3: B07
                nir: B08
                nir08: B8A
                nir09: B09
                swir16: B11
                swir22: B12
                scl: SCL
                aot: AOT
                wvp: WVP
        """
    elif sensor == "HLSL30_2.0":
        cfg = """---
        HLSL30_2.0:
            assets:
                '*':
                    data_type: int16
                    nodata: -9999
                    scale: 0.0001
                Fmask:
                    data_type: uint8
                    nodata: 255
                    scale: 1
                SZA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                SAA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                VZA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                VAA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                thermal infrared 1:
                    data_type: int16
                    nodata: -9999
                    scale: 0.01
                thermal:
                    data_type: int16
                    nodata: -9999
                    scale: 0.01
            aliases:
                coastal: B01
                blue: B02
                green: B03
                red: B04
                nir08: B05
                swir16: B06
                swir22: B07
                cirrus: B09
                lwir11: B10
                lwir12: B11
        """
    elif sensor == "HLSS30_2.0":
        cfg = """---
        HLSS30_2.0:
            assets:
                '*':
                    data_type: int16
                    nodata: -9999
                    scale: 0.0001
                Fmask:
                    data_type: uint8
                    nodata: 255
                    scale: 1
                SZA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                SAA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                VZA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
                VAA:
                    data_type: uint16
                    nodata: 40000
                    scale: 0.01
            aliases:
                coastal: B01
                blue: B02
                green: B03
                red: B04
                rededge071: B05
                rededge075: B06
                rededge078: B07
                nir: B08
                nir08: B8A
                water vapor: B09
                cirrus: B10
                swir16: B11
                swir22: B12
        """
    else:
        raise ValueError(
            f"Unknown sensor '{sensor}'. "
            "Supported sensors: 'sentinel-2-l2a', 'HLSL30_2.0', 'HLSS30_2.0'."
        )
    return yaml.load(cfg, Loader=yaml.CSafeLoader)


def get_water_year_start(date: pd.Timestamp, hemisphere: str) -> pd.Timestamp:
    """Return the start date of the water year containing *date*.

    Parameters
    ----------
    date : pandas.Timestamp
        Any date within the water year of interest.
    hemisphere : str
        ``"northern"`` (water year starts Oct 1) or
        ``"southern"`` (water year starts Apr 1).

    Returns
    -------
    pandas.Timestamp
        The first day of the corresponding water year.
    """
    year = date.year
    month = 10 if hemisphere == "northern" else 4
    if (hemisphere == "northern" and date.month < 10) or (
        hemisphere == "southern" and date.month < 4
    ):
        year -= 1
    return pd.Timestamp(year=year, month=month, day=1)


def datetime_to_DOWY(
    date: pd.Timestamp | str, hemisphere: str = "northern"
) -> int | float:
    """Convert a date to the day-of-water-year (DOWY).

    Parameters
    ----------
    date : pandas.Timestamp or str
        The date to convert. Strings are parsed by :func:`pandas.to_datetime`.
    hemisphere : str, optional
        ``"northern"`` or ``"southern"``. Default is ``"northern"``.

    Returns
    -------
    int or float
        Day of the water year (1-indexed), or ``np.nan`` on parse failure.
    """
    try:
        date = pd.to_datetime(date)
        start = get_water_year_start(date, hemisphere)
        return (date - start).days + 1
    except Exception as exc:
        _logger.warning("Could not compute DOWY for %s: %s", date, exc)
        return np.nan


def datetime_to_WY(
    date: pd.Timestamp | str, hemisphere: str = "northern"
) -> int | float:
    """Convert a date to its water year (WY).

    Parameters
    ----------
    date : pandas.Timestamp or str
        The date to convert. Strings are parsed by :func:`pandas.to_datetime`.
    hemisphere : str, optional
        ``"northern"`` or ``"southern"``. Default is ``"northern"``.

    Returns
    -------
    int or float
        The water year as a calendar year integer, or ``np.nan`` on failure.

    Notes
    -----
    For the northern hemisphere, the water year is the calendar year in which
    the water year *ends* (i.e. WY 2021 runs Oct 1 2020 – Sep 30 2021).
    """
    try:
        date = pd.to_datetime(date)
        start = get_water_year_start(date, hemisphere)
        return start.year + (1 if hemisphere == "northern" else 0)
    except Exception as exc:
        _logger.warning("Could not compute WY for %s: %s", date, exc)
        return np.nan


def HLS_xml_url_to_metadata_df(url: str) -> pd.DataFrame:
    """Parse an HLS granule XML metadata URL into a one-row DataFrame.

    Parameters
    ----------
    url : str
        Full URL to an HLS XML metadata file (NASA CMR or direct link).

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame with columns:
        ``ProducerGranuleId``, ``Temporal``, ``Platform``,
        ``AssociatedBrowseImageUrls``.

    Notes
    -----
    HLS (Harmonized Landsat Sentinel) metadata is produced by NASA LP DAAC.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml-xml")
    data = {
        tag.name: tag.text.strip().replace("\n", " ")
        for tag in soup.find_all()
        if tag.text.strip()
    }
    df = pd.DataFrame([data]).iloc[0][
        ["ProducerGranuleId", "Temporal", "Platform", "AssociatedBrowseImageUrls"]
    ]
    df["Platform"] = df["Platform"].split(" ")[0]
    df["AssociatedBrowseImageUrls"] = df["AssociatedBrowseImageUrls"].split(" ")[0]
    df["Temporal"] = df["Temporal"].split(" ")[0]
    return df
