"""Shared utility functions used across easysnowdata modules."""

from __future__ import annotations

import contextlib
import io
import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import yaml
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    "suppress_stdout",
    "convert_bbox_to_geodataframe",
    "get_stac_cfg",
    "get_water_year_start",
    "datetime_to_DOWY",
    "datetime_to_WY",
    "HLS_xml_url_to_metadata_df",
]

_logger = logging.getLogger(__name__)


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
