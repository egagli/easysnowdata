"""The common module contains common functions and classes used by the other modules.
"""

import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import yaml


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