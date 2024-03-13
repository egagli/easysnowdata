import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import pystac_client
import planetary_computer
import odc.stac

from easysnowdata.utils import convert_bbox_to_geodataframe


def get_forest_cover_fraction(bbox_input) -> xr.DataArray:
    """
    Fetches ~100m forest cover fraction data for a given bounding box.

    Description:
    The data is obtained from the Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe dataset, available at https://zenodo.org/records/3939050. The specific layer used is the Tree-CoverFraction-layer, which provides the fractional cover (%) for the forest class.

    Citation:
    Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050

    Parameters:
    bbox_input (GeoPandas GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    fcf_da (Xarray DataArray): Forest cover fraction DataArray.
    """

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    fcf_da = rxr.open_rasterio('https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif', chunks=True, mask_and_scale=True)
    fcf_da = fcf_da.rio.clip_box(xmin, ymin, xmax, ymax).squeeze()
    
    return fcf_da


def get_seasonal_snow_classification(bbox_input) -> xr.DataArray:
    """
    Fetches 10arcsec (~300m) Sturm & Liston 2021 seasonal snow classification data for a given bounding box. Class info in attributes.

    Description:
    "This data set consists of global, seasonal snow classifications—e.g., tundra, boreal forest, maritime, ephemeral, prairie, montane forest, and ice—determined from air temperature, precipitation, and wind speed climatologies." This is the 10 arcsec (~300m) product in EPSG:4326. The data is available on NSIDC at http://dx.doi.org/10.5067/99FTCYYYLAQ0, but this function pulls from a file I've hosted on blob storage (due to inability to stream from NSIDC source). 

    Citation:
    Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0. Date Accessed 03-06-2024.

    Parameters:
    bbox_input (GeoPandas GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    snow_classification_da (Xarray DataArray): Seasonal snow class DataArray.
    """    

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    snow_classification_da = rxr.open_rasterio('https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif', chunks=True, mask_and_scale=True)
    snow_classification_da = snow_classification_da.rio.clip_box(xmin, ymin, xmax, ymax).squeeze().rio.write_nodata(9, encoded=True)
    
    snow_classification_da.attrs['class_info'] = {
        1: {'name': "Tundra", 'color': '#a100c8'},
        2: {'name': "Boreal Forest", 'color': '#00a0fe'},
        3: {'name': "Maritime", 'color': '#fe0000'},
        4: {'name': "Ephemeral (includes no snow)", 'color': '#e7dc32'},
        5: {'name': "Prairie", 'color': '#f08328'},
        6: {'name': "Montane Forest", 'color': '#00dc00'},
        7: {'name': "Ice (glaciers and ice sheets)", 'color': '#aaaaaa'},
        8: {'name': "Ocean", 'color': '#0000ff'},
        9: {'name': "Fill", 'color': '#ffffff'}
    }

    return snow_classification_da


def get_esa_worldcover(bbox_input, version: str = 'v200') -> xr.DataArray:
    """
    Fetches 10m ESA WorldCover global land cover data (2020 v100 or 2021 v200) for a given bounding box. Class info in attributes.

    Description:
    "The discrete classification maps provide 11 classes defined using the Land Cover Classification System (LCCS) developed by the United Nations (UN) Food and Agriculture Organization (FAO)". Available at https://planetarycomputer.microsoft.com/dataset/esa-worldcover.

    Citation:
    Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936.

    Parameters:
    bbox_input (GeoPandas GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    version (str): Version of the WorldCover data. The two versions are v100 (2020) and v200 (2021). Default is 'v200'.

    Returns:
    worldcover_da (Xarray DataArray): WorldCover DataArray.
    """

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    if version == 'v100':
        version_year = '2020'
    elif version == 'v200':
        version_year = '2021'
    else:
        raise ValueError("Incorrect version number. Please provide 'v100' or 'v200'.")

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["esa-worldcover"],bbox=bbox_gdf.total_bounds)
    worldcover_da = odc.stac.load(search.items(),bbox=bbox_gdf.total_bounds,bands='map',chunks={})['map'].sel(time=version_year).squeeze()
    worldcover_da = worldcover_da.rio.write_nodata(0,encoded=True)

    worldcover_da.attrs['class_info'] = {
        10: {'name': 'Tree cover', 'color': '#006400'},
        20: {'name': 'Shrubland', 'color': '#FFBB22'},
        30: {'name': 'Grassland', 'color': '#FFFF4C'},
        40: {'name': 'Cropland', 'color': '#F096FF'},
        50: {'name': 'Built-up', 'color': '#FA0000'},
        60: {'name': 'Bare / sparse vegetation', 'color': '#B4B4B4'},
        70: {'name': 'Snow and ice', 'color': '#F0F0F0'},
        80: {'name': 'Permanent water bodies', 'color': '#0064C8'},
        90: {'name': 'Herbaceous wetland', 'color': '#0096A0'},
        95: {'name': 'Mangroves', 'color': '#00CF75'},
        100: {'name': 'Moss and lichen', 'color': '#FAE6A0'}
    }

    return worldcover_da





























# def get_sentinel2(bbox_input, start_date, end_date, catalog_choice='planetarycomputer', bands=None, resolution=None, crs=None, groupby='solar_day',config=None):
#     """
#     Fetch Sentinel-2 data from a specified catalog for a given bounding box and date range.

#     Parameters:
#     bbox_input (GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
#     start_date (str): Start date in the format 'YYYY-MM-DD'.
#     end_date (str): End date in the format 'YYYY-MM-DD'.
#     catalog_choice (str, optional): Catalog to fetch data from. Options are 'planetarycomputer' and 'earthsearch'. Defaults to 'planetarycomputer'.
#     bands (list, optional): List of bands to be included in the output data. If not specified, all bands are included.
#     resolution (tuple, optional): Resolution for the output data. If not specified, the data is returned at its original resolution.
#     crs (str, optional): Coordinate Reference System to be used for the output data. If not specified, the data is returned in its original CRS.
#     groupby (str, optional): Method to group by when loading data. Defaults to 'solar_day'.

#     Returns:
#     s2_ds (Xarray DataSet): An xarray Dataset containing the Sentinel-2 data.
#     """

#     band_info = {
#     "B01": {"Name": "coastal", "Description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)", "Resolution": "60m"},
#     "B02": {"Name": "blue", "Description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)", "Resolution": "10m"},
#     "B03": {"Name": "green", "Description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)", "Resolution": "10m"},
#     "B04": {"Name": "red", "Description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)", "Resolution": "10m"},
#     "B05": {"Name": "rededge", "Description": "Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)", "Resolution": "20m"},
#     "B06": {"Name": "rededge2", "Description": "Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)", "Resolution": "20m"},
#     "B07": {"Name": "rededge3", "Description": "Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)", "Resolution": "20m"},
#     "B08": {"Name": "nir", "Description": "NIR, 832.8 nm (S2A), 833.0 nm (S2B)", "Resolution": "10m"},
#     "B8A": {"Name": "nir08", "Description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)", "Resolution": "20m"},
#     "B09": {"Name": "nir09", "Description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)", "Resolution": "60m"},
#     "B11": {"Name": "swir16", "Description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)", "Resolution": "20m"},
#     "B12": {"Name": "swir22", "Description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)", "Resolution": "20m"},
#     "AOT": {"Name": "aot", "Description": "Aerosol Optical Thickness map, based on Sen2Cor processor", "Resolution": "10m"},
#     "SCL": {"Name": "scl", "Description": "Scene classification data, based on Sen2Cor processor: 0 - No data, 1 - Saturated / Defective, 2 - Dark Area Pixels, 3 - Cloud Shadows 4 - Vegetation, 5 - Bare Soils, 6 - Water, 7 - Clouds low probability / Unclassified, 8 - Clouds medium probability, 9 - Clouds high probability, 10 - Cirrus, 11 - Snow / Ice", "Resolution": "20m"},
#     "WVP": {"Name": "wvp", "Description": "Water Vapour map", "Resolution": "10m"},
#     "visual": {"Name": "visual", "Description": "True color image", "Resolution": "10m"},
#     } 

#     # Convert bbox_input to bbox_gdf
#     bbox_gdf = easysnowdata.utils.convert_bbox_to_geodataframe(bbox_input)
    
#     # Choose the catalog URL based on catalog_choice
#     if catalog_choice == "planetarycomputer":
#         catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
#         catalog = pystac_client.Client.open(catalog_url, modifier=planetary_computer.sign_inplace)
#         config = {
#             "sentinel-2-l2a": {
#                 "aliases": {
#                     "costal": "B01",
#                     "blue": "B02",
#                     "green": "B03",
#                     "red": "B04",
#                     "rededge": "B05",
#                     "rededge2": "B06",
#                     "rededge3": "B07",
#                     "nir": "B08",
#                     "nir08": "B8A",
#                     "nir09": "B09",
#                     "swir16": "B11",
#                     "swir22": "B12",
#                     "scl": "SCL",
#                     "aot": "AOT",
#                     "wvp": "WVP",},}}
#     elif catalog_choice == "earthsearch":
#         catalog_url = "https://earth-search.aws.element84.com/v1"
#         catalog = pystac_client.Client.open(catalog_url)
#     else:
#         raise ValueError("Invalid catalog_choice. Choose either 'planetarycomputer' or 'earthsearch'.")
     
#     # Search for items within the specified bbox and date range
#     search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox_gdf.total_bounds, datetime=(start_date, end_date))
    
#     # Prepare the parameters for odc.stac.load
#     load_params = {
#         'items': search.items(),
#         'bbox': bbox_gdf.total_bounds,
#         'nodata': 0,
#         'chunks': {},
#         'groupby': groupby
#     }
#     if bands:
#         load_params['bands'] = bands
#     else:
#         load_params['bands'] = [info['Name'] for info in band_info.values()]
#     if crs:
#         load_params['crs'] = crs
#     if resolution:
#         load_params['resolution'] = resolution
#     if config:
#         load_params['stac_cfg'] = config


#     # Load the data lazily using odc.stac
#     s2_ds = odc.stac.load(**load_params)

#     s2_ds.attrs['band_info'] = band_info
    
#     return s2_ds
