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




def get_copernicus_dem(bbox_input, resolution: int = 30) -> xr.DataArray:
    """
    Fetches 30m or 90m Copernicus DEM from Microsoft Planetary Computer.

    Description:
    "The Copernicus DEM is a Digital Surface Model (DSM) which represents the surface of the Earth including buildings, infrastructure and vegetation. This DSM is derived from an edited DSM named WorldDEM, where flattening of water bodies and consistent flow of rivers has been included. In addition, editing of shore- and coastlines, special features such as airports, and implausible terrain structures has also been applied." From https://doi.org/10.5069/G9028PQB
    Citation:
    European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed: 2024-03-18
    Parameters:
    bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    resolution (int): The resolution of the DEM, either 30 or 90. Default is 30.
    Returns:
    cop_dem_da (xarray.DataArray): Copernicus DEM DataArray.
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

def get_3dep_dem(bbox_input, dem_type: str = 'DSM') -> xr.DataArray:
    """
    XXXXXFetches 30m or 90m Copernicus DEM from Microsoft Planetary Computer.

    Description:
    XXXXX"The Copernicus DEM is a Digital Surface Model (DSM) which represents the surface of the Earth including buildings, infrastructure and vegetation. This DSM is derived from an edited DSM named WorldDEM, where flattening of water bodies and consistent flow of rivers has been included. In addition, editing of shore- and coastlines, special features such as airports, and implausible terrain structures has also been applied." From https://doi.org/10.5069/G9028PQB
    Citation:
    XXXXXEuropean Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed: 2024-03-18
    Parameters:
    bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    dem_type (str): The DEM type, DSM or DTM. Default is DSM.
    Returns:
    dep_dem_da (xarray.DataArray): 3DEP DEM DataArray.
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
