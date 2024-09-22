import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import pystac_client
import planetary_computer
import os

import odc.stac
odc.stac.configure_rio(cloud_defaults=True)  

import datetime
today = datetime.datetime.now().strftime('%Y-%m-%d')

from easysnowdata.utils import convert_bbox_to_geodataframe, get_stac_cfg




def get_copernicus_dem(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
                       resolution: int = 30
) -> xr.DataArray:
    """
    Fetches 30m or 90m Copernicus DEM from Microsoft Planetary Computer.

    This function retrieves the Copernicus Digital Elevation Model (DEM) data for a specified
    bounding box and resolution. The DEM represents the surface of the Earth including buildings,
    infrastructure, and vegetation.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    resolution : int, optional
        The resolution of the DEM, either 30 or 90 meters. Default is 30.

    Returns
    -------
    xarray.DataArray
        A DataArray containing the Copernicus DEM data for the specified area.

    Raises
    ------
    ValueError
        If the resolution is not 30 or 90 meters.

    Notes
    -----
    The Copernicus DEM is a Digital Surface Model (DSM) derived from the WorldDEM, with additional
    editing applied to water bodies, coastlines, and other special features.

    Data citation:
    European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model.
    Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed: 2024-03-18
    """
    if resolution != 30 and resolution != 90:
        raise ValueError("Copernicus DEM resolution is available in 30m and 90m. Please select either 30 or 90.")

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[f"cop-dem-glo-{resolution}"],bbox=bbox_gdf.total_bounds)
    cop_dem_da = odc.stac.load(search.items(),bbox=bbox_gdf.total_bounds,chunks={})['data'].squeeze()
    cop_dem_da = cop_dem_da.rio.write_nodata(-32767,encoded=True)

    return cop_dem_da

# find a way to say given bounding box, best and available DEMs?
#https://github.com/OpenTopography/OT_3DEP_Workflows
#https://github.com/OpenTopography/OT_BulkAccess_COGs/blob/main/OT_BulkAccessCOGs.ipynb

def get_3dep_dem(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
                 dem_type: str = 'DSM',
) -> xr.DataArray:
    """
    Fetches 3DEP DEM data from Microsoft Planetary Computer.

    This function retrieves the 3D Elevation Program (3DEP) Digital Elevation Model (DEM) data
    for a specified bounding box and DEM type.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    dem_type : str, optional
        The DEM type, either 'DSM' (Digital Surface Model) or 'DTM' (Digital Terrain Model). Default is 'DSM'.

    Returns
    -------
    pystac_client.ItemCollection
        A STAC ItemCollection containing the search results for the 3DEP DEM data.

    Raises
    ------
    ValueError
        If the dem_type is not 'DSM' or 'DTM'.

    Notes
    -----
    This function currently returns the search results rather than the actual DEM data.
    Further processing would be needed to load and process the DEM data from the search results.

    The 3DEP program provides high-quality elevation data for the United States.
    """
    if dem_type != 'DSM' and dem_type != 'DTM':
        raise ValueError("3DEP DEM type is available as DSM and DTM. Please select either DSM or DTM.")

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[f"3dep-lidar-{dem_type.lower()}"],bbox=bbox_gdf.total_bounds)
    #dep_dem_da = odc.stac.load(search.items(),bbox=bbox_gdf.total_bounds,chunks={})
    #dep_dem_da = dep_dem_da.rio.write_nodata(-32767,encoded=True)

    return search
