import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import pystac_client
import planetary_computer
import os
import ee
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

def get_chili(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, initialize_ee: bool = True,
) -> xr.DataArray:
    """
    Fetches Continuous Heat-Insolation Load Index (CHILI) data for a given bounding box. 

    Description:
    CHILI is a topographic index that quantifies the combined effect of solar radiation and surface temperature.
    It is derived from the ALOS World 3D - 30m (AW3D30) dataset and is available globally between 70°N and 70°S.
    The values range from 0 to 1, with classifications warm (0.767,1], cool [0,0.448), and neutral [0.448,0.767].

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns
    -------
    xarray.DataArray
        CHILI DataArray for the specified region.

    Examples
    --------
    >>> import geopandas as gpd
    >>> easysnowdata
    >>> 
    >>> # Define a bounding box for an area of interest
    >>> bbox = (-122.5, 47.0, -121.5, 48.0)
    >>> 
    >>> # Fetch CHILI data
    >>> chili_data = easysnowdata.remote_sensing.get_chili(bbox)
    >>> 
    >>> # Plot the data
    >>> chili_data.plot(cmap='viridis')

    Notes
    -----
    - CHILI data is only available for latitudes between 70°N and 70°S.
    - The function uses Google Earth Engine to access the dataset.

    Data citation:
    Theobald, D.M., Harrison-Atlas, D., Monahan, W.B., Albano, C.M. (2015).
    Ecologically-Relevant Maps of Landforms and Physiographic Diversity for Climate Adaptation Planning.
    PLoS ONE 10(12): e0143619. https://doi.org/10.1371/journal.pone.0143619
    """

    # Initialize Earth Engine with high-volume endpoint
    if initialize_ee == True:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    else:
        print(f'Initialization turned off. If you haven\'t already, please sign in to Google Earth Engine by running the following code:\n\nimport ee\nee.Authenticate()\nee.Initialize()\n\n')

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
    # Access the CHILI dataset
    image = ee.Image("CSP/ERGo/1_0/Global/ALOS_CHILI")
    image_collection = ee.ImageCollection(image)

    # Get projection information
    crs = image.projection().getInfo()['crs']
    transform = image.projection().getInfo()['transform']

    # Load the data using xarray and Earth Engine
    chili_da = xr.open_dataset(image_collection, engine='ee', geometry=tuple(bbox_gdf.total_bounds), projection=ee.Projection(crs=crs,transform=transform)).drop_vars('time').squeeze()['constant'].squeeze().transpose().rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    # maybe add chunks={} here to lazy load data

    # Clip the data to the bounding box
    chili_da = chili_da.rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs)

    # Check if data is available for the specified region
    if chili_da.isnull().all().item():
        print("No CHILI data available for this location. CHILI data is only available for latitudes between 70°N and 70°S.")

    chili_da = (chili_da - chili_da.min()) / (chili_da.max() - chili_da.min())
    
    chili_da.attrs['data_citation'] = "Theobald, D.M., Harrison-Atlas, D., Monahan, W.B., Albano, C.M. (2015). Ecologically-Relevant Maps of Landforms and Physiographic Diversity for Climate Adaptation Planning. PLoS ONE 10(12): e0143619. https://doi.org/10.1371/journal.pone.0143619"

    return chili_da

# find a way to say given bounding box, best and available DEMs?
#https://github.com/OpenTopography/OT_3DEP_Workflows
#https://github.com/OpenTopography/OT_BulkAccess_COGs/blob/main/OT_BulkAccessCOGs.ipynb

# def get_3dep_dem(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
#                  dem_type: str = 'DSM',
# ) -> xr.DataArray:
#     """
#     Fetches 3DEP DEM data from Microsoft Planetary Computer.

#     This function retrieves the 3D Elevation Program (3DEP) Digital Elevation Model (DEM) data
#     for a specified bounding box and DEM type.

#     Parameters
#     ----------
#     bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
#         GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
#     dem_type : str, optional
#         The DEM type, either 'DSM' (Digital Surface Model) or 'DTM' (Digital Terrain Model). Default is 'DSM'.

#     Returns
#     -------
#     pystac_client.ItemCollection
#         A STAC ItemCollection containing the search results for the 3DEP DEM data.

#     Raises
#     ------
#     ValueError
#         If the dem_type is not 'DSM' or 'DTM'.

#     Notes
#     -----
#     This function currently returns the search results rather than the actual DEM data.
#     Further processing would be needed to load and process the DEM data from the search results.

#     The 3DEP program provides high-quality elevation data for the United States.
#     """
#     if dem_type != 'DSM' and dem_type != 'DTM':
#         raise ValueError("3DEP DEM type is available as DSM and DTM. Please select either DSM or DTM.")

#     # Convert the input to a GeoDataFrame if it's not already one
#     bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

#     catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
#     search = catalog.search(collections=[f"3dep-lidar-{dem_type.lower()}"],bbox=bbox_gdf.total_bounds)
#     #dep_dem_da = odc.stac.load(search.items(),bbox=bbox_gdf.total_bounds,chunks={})
#     #dep_dem_da = dep_dem_da.rio.write_nodata(-32767,encoded=True)

#     return search
