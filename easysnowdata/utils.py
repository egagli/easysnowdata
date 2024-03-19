"""The common module contains common functions and classes used by the other modules.
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


def hello_world():
    """Prints "Hello World!" to the console.
    """
    print("Hello World!")


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
        bbox_input = gpd.GeoDataFrame(geometry=[shapely.geometry.box(*bbox_input)], crs="EPSG:4326")
    elif isinstance(bbox_input, shapely.geometry.base.BaseGeometry):
        # If it's a Shapely geometry, convert it to a GeoDataFrame
        bbox_input = gpd.GeoDataFrame(geometry=[bbox_input], crs="EPSG:4326")

    return bbox_input

def get_stac_cfg(sensor='sentinel-2-l2a'):
    if sensor == 'sentinel-2-l2a':
        cfg = """---
        sentinel-2-l2a:
            assets:
                '*':
                    data_type: uint16
                    nodata: 0
                    unit: '1'
                SCL:
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
    cfg = yaml.load(cfg, Loader=yaml.CSafeLoader)

    return cfg


def datetime_to_DOWY(date: datetime.datetime) -> typing.Union[int, np.nan]:
    """
    Convert a datetime to the day of the water year (DOWY).
    
    The water year starts on October 1. If the month of the date is less than 10 (October), 
    the start of the water year is considered to be October 1 of the previous year. 
    Otherwise, the start of the water year is October 1 of the current year.
    
    Parameters:
    date (datetime): The date to convert.
    
    Returns:
    int: The day of the water year, or np.nan if the date is not valid.
    """
    try:
        if date.month < 10:
            start_of_water_year = pd.Timestamp(year=date.year-1, month=10, day=1)
        else:
            start_of_water_year = pd.Timestamp(year=date.year, month=10, day=1)
        return (date - start_of_water_year).days + 1
    except:
        return np.nan

def datetime_to_WY(date: datetime) -> int:
    """
    Convert a datetime to the water year (WY).
    
    The water year starts on October 1. If the month of the date is less than 10 (October), 
    the water year is considered to be the current year. Otherwise, the water year is the next year.
    
    Parameters:
    date (datetime): The date to convert.
    
    Returns:
    int: The water year.
    """
    if date.month < 10:
        return date.year
    else:
        return date.year + 1