import re
import pandas as pd
import geopandas as gpd
import ee
import json
import xarray as xr
import numpy as np
import earthaccess
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Union
import shapely


from easysnowdata.utils import convert_bbox_to_geodataframe

# ee.Authenticate() need to figure out https://developers.google.com/earth-engine/guides/auth
# ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def get_huc_geometries(
        bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, 
        huc_level: str = "02",
) -> gpd.GeoDataFrame:
    """
    Retrieves Hydrologic Unit Code (HUC) geometries within a specified bounding box and HUC level.

    This function queries the USGS Water Boundary Dataset (WBD) for HUC geometries. It can retrieve
    HUC geometries at different levels for a specified region defined by a bounding box. If no 
    bounding box is provided, it retrieves HUC geometries for the entire United States.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire US dataset is returned.
    huc_level : str, optional
        The HUC level to retrieve geometries for. Valid levels are '02', '04', '06', '08', '10', '12'.
        Default is '02'.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the retrieved HUC geometries along with associated attributes
        such as name, area in square kilometers, states, TNMID, and geometry.

    Examples
    --------
    Get HUC geometries for a specific region at HUC level 08...

    >>> huc_data = get_huc_geometries(bbox_input=(-121.94, 46.72, -121.54, 46.99), huc_level="08")
    >>> huc_data.plot()

    Notes
    -----
    This function requires an active Earth Engine session. Make sure to authenticate 
    with Earth Engine before using this function.

    Data citation: 
    Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, 
    Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): 
    U.S. Geological Survey Techniques and Methods 11-A3, 54 p., 
    https://doi.org/10.3133/tm11A3
    """

    # Convert bounding box to feature collection to use as region for querying HUC geometries
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
    bbox_json = bbox_gdf.to_json()
    featureCollection = ee.FeatureCollection(json.loads(bbox_json))

    # Search Earth Engine USGS WBD collection for HUC geometries
    huc_gdf = ee.data.listFeatures(
        {
            "assetId": f"USGS/WBD/2017/HUC{huc_level}",
            "region": featureCollection.geometry().getInfo(),
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        }
    )

    # Add crs to geodataframe and select relevant columns
    huc_gdf.crs = "EPSG:4326"
    huc_gdf = huc_gdf[
        [
            "name",
            f'huc{huc_level.lstrip("0")}',
            "areasqkm",
            "states",
            "tnmid",
            "geometry",
        ]
    ]

    huc_gdf.attrs = {"Data citation": "Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): U.S. Geological Survey Techniques and Methods 11-A3, 54 p., https://doi.org/10.3133/tm11A3"}
    
    return huc_gdf


def get_era5(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
) -> xr.Dataset:
    """
    Retrieves ERA5 data for a given bounding box.

    This function fetches ERA5 reanalysis data from Google Cloud Storage, allowing for optional
    spatial subsetting. The data is returned as an xarray dataset with time-chunked data.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the global dataset is returned.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing ERA5 reanalysis data for the specified region.

    Examples
    --------
    Get ERA5 data for the globe, plot the mean 2m temperature on May 26th, 2020...

    >>> era5_global_ds = easysnowdata.hydroclimatology.get_era5()
    >>> era5_global_ds["2m_temperature"].sel(time="2020-05-26").mean(dim="time").plot.imshow(ax=ax)

    Get ERA5 data for a specific region, plot the mean 2m temperature on May 26th, 2020...

    >>> era5_bbox_ds = easysnowdata.hydroclimatology.get_era5(bbox_input=(-121.94, 46.72, -121.54, 46.99))
    >>> era5_bbox_ds["2m_temperature"].sel(time="2020-05-26").mean(dim="time").plot.imshow(ax=ax)

    Notes
    -----
    The ERA5 data is sourced from Google Cloud Storage and is time-chunked for efficient processing.

    Data citation:

    Carver, Robert W, and Merose, Alex. (2023):
    ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset.
    22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1,
    https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842
    """

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    era5 = xr.open_zarr(  # https://cloud.google.com/storage/docs/public-datasets/era5
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        chunks={"time": 48},
        consolidated=True,
    )
    era5.rio.write_crs("EPSG:4326", inplace=True)
    era5.coords["longitude"] = (era5.coords["longitude"] + 180) % 360 - 180
    era5 = era5.sortby(era5.longitude)

    era5_ds = era5.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs)

    # ar_full_37_1h = xr.open_zarr( #https://github.com/google-research/arco-era5
    #   'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    #   chunks=None,
    #   storage_options=dict(token='anon'),
    # )

    era5.attrs["data_citation"] = "Carver, Robert W, and Merose, Alex. (2023): ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset. 22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1, https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842"

    return era5_ds


def get_ucla_snow_reanalysis(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
                             variable: str = 'SWE_Post',
                             stats: str = 'mean',
                             start_date: str = '1984-10-01',
                             end_date: str = '2021-09-30',
) -> xr.DataArray:
    """
    Fetches the Margulis UCLA snow reanalysis product for a specified bounding box and time range.

    This function retrieves snow reanalysis data from the UCLA dataset, allowing users to specify
    the type of snow data variable, statistical measure, and the temporal range for the data retrieval.
    The data is then clipped to the specified bounding box and returned as an xarray DataArray.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire dataset is returned.
    variable : str, optional
        The type of snow data variable to retrieve. Options include 'SWE_Post' (Snow Water Equivalent),
        'SCA_Post' (Snow Cover Area), and 'SD_Post' (Snow Depth). Default is 'SWE_Post'.
    stats : str, optional
        The ensemble statistic. Options are 'mean', 'std' (standard deviation),
        'median', '25pct' (25th percentile), and '75pct' (75th percentile). Default is 'mean'.
    start_date : str, optional
        The start date for the data retrieval in 'YYYY-MM-DD' format. Default is '1984-10-01'.
    end_date : str, optional
        The end date for the data retrieval in 'YYYY-MM-DD' format. Default is '2021-09-30'.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the requested snow reanalysis data, clipped to the specified bounding box.

    Examples
    --------
    Get mean Snow Water Equivalent data for a specific region and time period...

    >>> swe_reanalysis_da = easysnowdata.hydroclimatology.get_ucla_snow_reanalysis(bbox_input=(-121.94, 46.72, -121.54, 46.99), 
    ...                                     variable='SWE_Post', 
    ...                                     start_date='2000-01-01', 
    ...                                     end_date='2000-12-31')
    >>> snow_reanalysis_da.isel(time=slice(0, 365, 30)).plot.imshow(col="time",col_wrap=5,cmap="Blues",vmin=0,vmax=3)

    Notes
    -----
    Data citation:

    Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2
    """

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    search = earthaccess.search_data(
                short_name="WUS_UCLA_SR",
                cloud_hosted=True,
                bounding_box=tuple(bbox_gdf.total_bounds),
                temporal=(start_date, end_date),
            )
    
    files = earthaccess.open(search) # cant disable progress bar yet https://github.com/nsidc/earthaccess/issues/612
    snow_reanalysis_ds = xr.open_mfdataset(files)

    url = files[0].path
    date_pattern = r'\d{4}\.\d{2}\.\d{2}'
    WY_start_date = pd.to_datetime(re.search(date_pattern, url).group())

    snow_reanalysis_ds.coords['time'] = ("Day", pd.date_range(WY_start_date, periods=snow_reanalysis_ds.sizes['Day']))
    snow_reanalysis_ds = snow_reanalysis_ds.swap_dims({'Day':'time'})

    snow_reanalysis_ds = snow_reanalysis_ds.sel(time=slice(start_date, end_date))

    stats_dictionary = {'mean':0, 'std':1, 'median':2, '25pct':2, '75pct':3}
    stats_index = stats_dictionary[stats]

    snow_reanalysis_da = snow_reanalysis_ds[variable].sel(Stats=stats_index)
    snow_reanalysis_da = snow_reanalysis_da.rio.write_crs("EPSG:4326")
    snow_reanalysis_da = snow_reanalysis_da.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude")
    snow_reanalysis_da = snow_reanalysis_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs)

    snow_reanalysis_da.attrs["data_citation"] = "Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2"
    
    return snow_reanalysis_da


def get_koppen_geiger_classes(
        bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
        resolution: str = "0.1 degree",
) -> xr.DataArray:
    """
    Retrieves Köppen-Geiger climate classification data for a given bounding box and resolution.

    This function fetches global Köppen-Geiger climate classification data from a high-resolution dataset
    based on constrained CMIP6 projections. It allows for optional spatial subsetting and provides
    multiple resolution options. The returned DataArray includes a custom plotting function as an attribute.

    Parameters
    ----------
    bbox_input:
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    resolution:
        The spatial resolution of the data. Options are "1 degree", "0.5 degree", "0.1 degree", or "1 km".
        Default is "0.1 degree".

    Returns
    -------
    xarray.DataArray
        A DataArray containing the Köppen-Geiger climate classification data, with class information,
        color map, data citation, and a custom plotting function included as attributes.

    Examples
    --------
    Get Köppen-Geiger climate classification data for the entire globe with a 1-degree resolution, use custom plotting function...

    >>> koppen_data = get_koppen_geiger_classes(bbox_input=None, resolution="1 degree")
    >>> koppen_data.attrs['plot_classes'](koppen_data)

        Get Köppen-Geiger climate classification data for a specific region with a 1 km resolution, plot using xarray's built-in plotting function...

    >>> koppen_geiger_da = get_koppen_geiger_classes(bbox_input=(-121.94224976, 46.72842173, -121.54136001, 46.99728203), resolution="1 km")
    >>> koppen_data.plot(cmap=koppen_data.attrs["cmap"])

    Notes
    -----
    Data citation:

    Beck, H.E., McVicar, T.R., Vergopolan, N. et al. High-resolution (1 km) Köppen-Geiger maps
    for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023).
    https://doi.org/10.1038/s41597-023-02549-6
    """

    def get_koppen_geiger_class_info():
        koppen_geiger_classes = {
            1: {"name": "Af", "description": "Tropical, rainforest", "color": [0, 0, 255]},
            2: {"name": "Am", "description": "Tropical, monsoon", "color": [0, 120, 255]},
            3: {"name": "Aw", "description": "Tropical, savannah", "color": [70, 170, 250]},
            4: {"name": "BWh", "description": "Arid, desert, hot", "color": [255, 0, 0]},
            5: {"name": "BWk", "description": "Arid, desert, cold", "color": [255, 150, 150]},
            6: {"name": "BSh", "description": "Arid, steppe, hot", "color": [245, 165, 0]},
            7: {"name": "BSk", "description": "Arid, steppe, cold", "color": [255, 220, 100]},
            8: {"name": "Csa", "description": "Temperate, dry summer, hot summer", "color": [255, 255, 0]},
            9: {"name": "Csb", "description": "Temperate, dry summer, warm summer", "color": [200, 200, 0]},
            10: {"name": "Csc", "description": "Temperate, dry summer, cold summer", "color": [150, 150, 0]},
            11: {"name": "Cwa", "description": "Temperate, dry winter, hot summer", "color": [150, 255, 150]},
            12: {"name": "Cwb", "description": "Temperate, dry winter, warm summer", "color": [100, 200, 100]},
            13: {"name": "Cwc", "description": "Temperate, dry winter, cold summer", "color": [50, 150, 50]},
            14: {"name": "Cfa", "description": "Temperate, no dry season, hot summer", "color": [200, 255, 80]},
            15: {"name": "Cfb", "description": "Temperate, no dry season, warm summer", "color": [100, 255, 80]},
            16: {"name": "Cfc", "description": "Temperate, no dry season, cold summer", "color": [50, 200, 0]},
            17: {"name": "Dsa", "description": "Cold, dry summer, hot summer", "color": [255, 0, 255]},
            18: {"name": "Dsb", "description": "Cold, dry summer, warm summer", "color": [200, 0, 200]},
            19: {"name": "Dsc", "description": "Cold, dry summer, cold summer", "color": [150, 50, 150]},
            20: {"name": "Dsd", "description": "Cold, dry summer, very cold winter", "color": [150, 100, 150]},
            21: {"name": "Dwa", "description": "Cold, dry winter, hot summer", "color": [170, 175, 255]},
            22: {"name": "Dwb", "description": "Cold, dry winter, warm summer", "color": [90, 120, 220]},
            23: {"name": "Dwc", "description": "Cold, dry winter, cold summer", "color": [75, 80, 180]},
            24: {"name": "Dwd", "description": "Cold, dry winter, very cold winter", "color": [50, 0, 135]},
            25: {"name": "Dfa", "description": "Cold, no dry season, hot summer", "color": [0, 255, 255]},
            26: {"name": "Dfb", "description": "Cold, no dry season, warm summer", "color": [55, 200, 255]},
            27: {"name": "Dfc", "description": "Cold, no dry season, cold summer", "color": [0, 125, 125]},
            28: {"name": "Dfd", "description": "Cold, no dry season, very cold winter", "color": [0, 70, 95]},
            29: {"name": "ET", "description": "Polar, tundra", "color": [178, 178, 178]},
            30: {"name": "EF", "description": "Polar, frost", "color": [102, 102, 102]}
        }
        return koppen_geiger_classes


    def get_koppen_geiger_cmap(koppen_geiger_classes):
        colors = {k: [c/255 for c in v["color"]] for k, v in koppen_geiger_classes.items()}
        return matplotlib.colors.ListedColormap([colors[i] for i in range(1, 31)])
    


    def plot_classes(self, ax=None, figsize=(10, 10), cbar_orientation='horizontal'):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        bounds = np.arange(0.5, 31.5, 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)

        ax.set_aspect("equal")

        cbar = f.colorbar(im, ax=ax, orientation=cbar_orientation, aspect=30, pad=0.08)

        cbar.set_ticks(np.arange(1, 31))
        cbar.set_ticklabels([f"{v['name']}: {v['description']}" for k, v in self.attrs["class_info"].items()], fontsize=8)

        if cbar_orientation == 'horizontal':
            plt.setp(cbar.ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
        else:
            plt.setp(cbar.ax.get_yticklabels(), rotation=0, ha='right')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Köppen-Geiger Climate Classification")
        f.tight_layout()

        return f, ax
    
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    resolution_dict = {"1 degree": "1p0", "0.5 degree": "0p5", "0.1 degree": "0p1", "1 km": "0p00833333"}
    resolution = resolution_dict[resolution]

    koppen_geiger_da = rxr.open_rasterio(f"zip+https://figshare.com/ndownloader/files/45057352/koppen_geiger_tif.zip/1991_2020/koppen_geiger_{resolution}.tif").squeeze()

    koppen_geiger_da = koppen_geiger_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs)
        

    koppen_geiger_da.attrs["class_info"] = get_koppen_geiger_class_info()
    koppen_geiger_da.attrs["cmap"] = get_koppen_geiger_cmap(koppen_geiger_da.attrs["class_info"])
    koppen_geiger_da.attrs["data_citation"] = "Beck, H.E., McVicar, T.R., Vergopolan, N. et al. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023). https://doi.org/10.1038/s41597-023-02549-6"

    koppen_geiger_da.attrs['plot_classes'] = plot_classes

    return koppen_geiger_da




















# huc map, from gee?

# hydroatlas? https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroATLAS_v1_Basins_level03

# maybe seperate climate module,or use https://github.com/hyriver/pydaymet

# https://github.com/OpenTopography/OT_3DEP_Workflows/blob/main/notebooks/03_3DEP_Generate_DEM_USGS_HUCs.ipynb
# from opentopo....
# Now exectute the call to the USGS Watershed Boundary Dataset REST API. There are several possibilities here if the following step does not execute properly. There is a print statement implemented that should provide some indication of which it is.

# If you receive the error Error with Service Call or Error loading JSON output, this most likely indicates that the region you selected does coincide with a USGS hydrologic unit from that particular service.

# If you recieve the error {'message': 'Endpoint request timed out'}, this likely indicates the Watershed Boundary Dataset service may be down. To check, click this link (https://stats.uptimerobot.com/gxzRZFARLZ/783928857). If you are greeted with the message ScienceBase is down., the service is temporarily down and it is not possible to query the service at this time.

# If the cell executes successfully, the interesecting watershed boundary geometry will be printed.

# #url for 12-digit HUCs is ../MapServer/6/
# #url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/7/query?'   #14-Digit HUC
# url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?'   #12-Digit HUC

# #the parameters here will query the map server for the appropriate HU boundary
# params = dict(geometry=user_AOI,geometryType='esriGeometryEnvelope',inSR='4326',
#               spatialRel='esriSpatialRelIntersects',f='geojson')

# #Execute REST API call.
# try:
#     r = requests.get(url,params=params)
# except:
#     print('Error with Service Call. This could mean that there is no hydrologic unit polygon where you selected.')

# #load API JSON output into a variable
# try:
#     wbd_geojson = json.loads(r.content)
#     print(wbd_geojson)
# except:
#     print('Error loading JSON output')

# #To write out a JSON file...
# with open('WBD_API_Query.geojson', 'w') as outfile:
#     json.dump(wbd_geojson, outfile)
