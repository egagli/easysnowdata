"""The utils module contains common functions and classes used by the other modules.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import yaml
import datetime
import typing
import requests
from bs4 import BeautifulSoup
import sys
import os


def hello_world():
    """Prints "Hello World!" to the console."""
    print("Hello World!")


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def convert_bbox_to_geodataframe(bbox_input):
    """
    Converts the input to a GeoDataFrame.

    Parameters:
    bbox_input (GeoDataFrame or tuple or Shapely geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    GeoDataFrame: The converted bounding box.
    """
    if isinstance(bbox_input, tuple) and len(bbox_input) == 4:
        # If it's a tuple of four elements, treat it as (xmin, ymin, xmax, ymax)
        bbox_input = gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(*bbox_input)], crs="EPSG:4326"
        )
    elif isinstance(bbox_input, shapely.geometry.base.BaseGeometry):
        # If it's a Shapely geometry, convert it to a GeoDataFrame
        bbox_input = gpd.GeoDataFrame(geometry=[bbox_input], crs="EPSG:4326")

    return bbox_input


# def get_stac_cfg(sensor='sentinel-2-l2a'):
#     if sensor == 'sentinel-2-l2a':
#         cfg = """---
#         sentinel-2-l2a:
#             assets:
#                 '*':
#                     data_type: uint16
#                     nodata: 0
#                     unit: '1'
#                 SCL:
#                     data_type: uint8
#                     nodata: 0
#                     unit: '1'
#                 visual:
#                     data_type: uint8
#                     nodata: 0
#                     unit: '1'
#             aliases:  # Alias -> Canonical Name
#                 costal: B01
#                 blue: B02
#                 green: B03
#                 red: B04
#                 rededge1: B05
#                 rededge2: B06
#                 rededge3: B07
#                 nir: B08
#                 nir08: B8A
#                 nir09: B09
#                 swir16: B11
#                 swir22: B12
#                 scl: SCL
#                 aot: AOT
#                 wvp: WVP
#         """
#     cfg = yaml.load(cfg, Loader=yaml.CSafeLoader)

#     return cfg


def get_stac_cfg(sensor="sentinel-2-l2a"):
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
            aliases:  # Alias -> Canonical Name
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
    elif sensor == "HLSL30.v2.0":
        cfg = """---
        HLSL30.v2.0:
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
                coastal aerosol: B01
                blue: B02
                green: B03
                red: B04
                nir narrow: B05
                swir 1: B06
                swir 2: B07
                cirrus: B09
                thermal infrared 1: B10
                thermal: B11
        """
    elif sensor == "HLSS30.v2.0":
        cfg = """---
        HLSS30.v2.0:
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
                coastal aerosol: B01
                blue: B02
                green: B03
                red: B04
                red-edge 1: B05
                red-edge 2: B06
                red-edge 3: B07
                nir broad: B08
                nir narrow: B8A
                water vapor: B09
                cirrus: B10
                swir 1: B11
                swir 2: B12
        """
    cfg = yaml.load(cfg, Loader=yaml.CSafeLoader)

    return cfg


def get_water_year_start(date, hemisphere):
    year = date.year
    month = 10 if hemisphere == "northern" else 4
    if (hemisphere == "northern" and date.month < 10) or (
        hemisphere == "southern" and date.month < 4
    ):
        year -= 1
    return pd.Timestamp(year=year, month=month, day=1)


def datetime_to_DOWY(date, hemisphere="northern"):
    """
    Convert a datetime-like object to the day of water year (DOWY).

    Parameters:
    date: datetime-like object (string, datetime, Timestamp, etc.)
    hemisphere: str, 'northern' or 'southern'

    Returns:
    int: The day of the water year, or np.nan if the date is not valid.
    """
    try:
        date = pd.to_datetime(date)
        start = get_water_year_start(date, hemisphere)
        return (date - start).days + 1
    except:
        return np.nan


def datetime_to_WY(date, hemisphere="northern"):
    """
    Convert a datetime-like object to the water year (WY).
    Parameters:
    date: datetime-like object (string, datetime, Timestamp, etc.)
    hemisphere: str, 'northern' or 'southern'

    Returns:
    int: The water year.
    """
    try:
        date = pd.to_datetime(date)
        start = get_water_year_start(date, hemisphere)
        return start.year + (1 if hemisphere == "northern" else 0)
    except:
        return np.nan


def HLS_xml_url_to_metadata_df(url):
    # URL of the XML file

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the XML content of the response with BeautifulSoup
    soup = BeautifulSoup(
        response.content, "lxml-xml"
    )  # 'lxml-xml' parser is used for parsing XML

    # Create a dictionary to hold the data
    data = {}

    # Iterate over all tags in the soup object
    for tag in soup.find_all():
        # If the tag has a text value, add it to the dictionary
        if tag.text.strip():
            data[tag.name] = tag.text.strip().replace("\n", " ")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data]).iloc[0][
        ["ProducerGranuleId", "Temporal", "Platform", "AssociatedBrowseImageUrls"]
    ]

    df["Platform"] = df["Platform"].split(" ")[0]
    df["AssociatedBrowseImageUrls"] = df["AssociatedBrowseImageUrls"].split(" ")[0]
    df["Temporal"] = df["Temporal"].split(" ")[0]

    return df
