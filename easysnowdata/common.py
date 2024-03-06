"""The common module contains common functions and classes used by the other modules.
"""

import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask


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