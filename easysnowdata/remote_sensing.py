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




def get_forest_cover_fraction(bbox_input) -> xr.DataArray:
    """
    Fetches ~100m forest cover fraction data for a given bounding box.

    Description:
    The data is obtained from the Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe dataset, available at https://zenodo.org/records/3939050. The specific layer used is the Tree-CoverFraction-layer, which provides the fractional cover (%) for the forest class.

    Citation:
    Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050

    Parameters:
    bbox_input (geopandas.GeoDataFrame or tuple or shapely.Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    fcf_da (xarray.DataArray): Forest cover fraction DataArray.
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
    bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    snow_classification_da (xarray.DataArray): Seasonal snow class DataArray.
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
    bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    version (str): Version of the WorldCover data. The two versions are v100 (2020) and v200 (2021). Default is 'v200'.

    Returns:
    worldcover_da (xarray.DataArray): WorldCover DataArray.
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





class Sentinel2:
    """
    A class to handle Sentinel-2 satellite data.

    Attributes:
        bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
        start_date (str): The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
        end_date (str): The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
        catalog_choice (str): The catalog choice for the data. Can choose between 'planetarycomputer' and 'earthsearch', default is 'planetarycomputer'.
        bands (list): The bands to be used. Default is all bands. Must include SCL for data masking. Each band should be a string like 'B01', 'B02', etc.
        resolution (str): The resolution of the data. Defaults to native resolution, 10m.
        crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
        groupby (str): The groupby parameter for the data. Default is "solar_day".

        band_info (dict): Information about the bands.
        scl_class_info (dict): Information about the scene classification. This should be a dictionary with keys being the class values and values being another dictionary with keys like 'name', 'description', etc.
        
        data (xarray.Dataset): The loaded data.
        rgb (xarray.DataArray): The RGB data.
        ndvi (xarray.DataArray): The NDVI data.
        ndsi (xarray.DataArray): The NDSI data.
        ndwi (xarray.DataArray): The NDWI data.
        evi (xarray.DataArray): The EVI data.
        ndbi (xarray.DataArray): The NDBI data.
    """


    def __init__(self, bbox_input, start_date='2014-01-01', end_date=datetime.datetime.now().strftime('%Y-%m-%d'), catalog_choice='planetarycomputer', bands=None, resolution=None, crs=None, groupby='solar_day'):
        """
        The constructor for the Sentinel2 class.

        Parameters:
            bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
            start_date (str): The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
            end_date (str): The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
            catalog_choice (str): The catalog choice for the data. Can choose between 'planetarycomputer' and 'earthsearch', default is 'planetarycomputer'.
            bands (list): The bands to be used. Default is all bands. Must include SCL for data masking. Each band should be a string like 'B01', 'B02', etc.
            resolution (str): The resolution of the data. Defaults to native resolution, 10m.
            crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
            groupby (str): The groupby parameter for the data. Default is "solar_day".
        """
        # Initialize the attributes
        self.bbox_input = bbox_input
        self.start_date = start_date
        self.end_date = end_date
        self.catalog_choice = catalog_choice
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.groupby = groupby
            
        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs == None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # Define the band information
        self.band_info = {
            "B01": {"name": "coastal", "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)", "resolution": "60m"},
            "B02": {"name": "blue", "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)", "resolution": "10m"},
            "B03": {"name": "green", "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)", "resolution": "10m"},
            "B04": {"name": "red", "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)", "resolution": "10m"},
            "B05": {"name": "rededge", "description": "Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)", "resolution": "20m"},
            "B06": {"name": "rededge2", "description": "Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)", "resolution": "20m"},
            "B07": {"name": "rededge3", "description": "Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)", "resolution": "20m"},
            "B08": {"name": "nir", "description": "NIR, 832.8 nm (S2A), 833.0 nm (S2B)", "resolution": "10m"},
            "B8A": {"name": "nir08", "description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)", "resolution": "20m"},
            "B09": {"name": "nir09", "description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)", "resolution": "60m"},
            "B11": {"name": "swir16", "description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)", "resolution": "20m"},
            "B12": {"name": "swir22", "description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)", "resolution": "20m"},
            "AOT": {"name": "aot", "description": "Aerosol Optical Thickness map, based on Sen2Cor processor", "resolution": "10m"},
            "SCL": {"name": "scl", "description": "Scene classification data, based on Sen2Cor processor", "resolution": "20m"},
            "WVP": {"name": "wvp", "description": "Water Vapour map", "resolution": "10m"},
            "visual": {"name": "visual", "description": "True color image", "resolution": "10m"},
        } 

        # Define the scene classification information, classes here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
        self.scl_class_info = { 
            0: {"name": "No Data (Missing data)", "color": "#000000"},
            1: {"name": "Saturated or defective pixel", "color": "#ff0000"},
            2: {"name": "Topographic casted shadows", "color": "#2f2f2f"}, # (called 'Dark features/Shadows' for data before 2022-01-25)
            3: {"name": "Cloud shadows", "color": "#643200"},
            4: {"name": "Vegetation", "color": "#00a000"},
            5: {"name": "Not-vegetated", "color": "#ffe65a"},
            6: {"name": "Water", "color": "#0000ff"},
            7: {"name": "Unclassified", "color": "#808080"},
            8: {"name": "Cloud medium probability", "color": "#c0c0c0"},
            9: {"name": "Cloud high probability", "color": "#ffffff"},
            10: {"name": "Thin cirrus", "color": "#64c8ff"},
            11: {"name": "Snow or ice", "color": "#ff96ff"},
        }

        # Initialize the data attributes 
        self.search = None
        self.data = None
        self.metadata = None

        self.rgb = None
        self.ndvi = None
        self.ndsi = None
        self.ndwi = None
        self.evi = None
        self.ndbi = None

        self.search_data()
        self.get_data()
        self.get_metadata()

    def search_data(self):
        """
        The method to search the data.
        """
        
        # Choose the catalog URL based on catalog_choice
        if self.catalog_choice == "planetarycomputer":
            catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
            catalog = pystac_client.Client.open(catalog_url, modifier=planetary_computer.sign_inplace)
        elif self.catalog_choice == "earthsearch":
            os.environ['AWS_REGION']='us-west-2'
            os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR' 
            os.environ['AWS_NO_SIGN_REQUEST']='YES'
            catalog_url = "https://earth-search.aws.element84.com/v1"
            catalog = pystac_client.Client.open(catalog_url)
        else:
            raise ValueError("Invalid catalog_choice. Choose either 'planetarycomputer' or 'earthsearch'.")
        
        # Search for items within the specified bbox and date range
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=self.bbox_gdf.total_bounds, datetime=(self.start_date, self.end_date))
        self.search = search
        print(f'Data searched. Access the returned seach with the .search attribute.')


    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params = {
            'items': self.search.items(),
            'bbox': self.bbox_gdf.total_bounds,
            'nodata': 0,
            'chunks': {},
            'crs': self.crs,
            'groupby': self.groupby,
            'stac_cfg': get_stac_cfg(sensor="sentinel-2-l2a")
        }
        if self.bands:
            load_params['bands'] = self.bands
        else:
            load_params['bands'] = [info['name'] for info in self.band_info.values()]
        if self.resolution:
            load_params['resolution'] = self.resolution


        # Load the data lazily using odc.stac
        self.data = odc.stac.load(**load_params)

        self.data.attrs['band_info'] = self.band_info
        self.data.attrs['scl_class_info'] = self.scl_class_info

        if 'scl' in self.data.variables:
            self.data.scl.attrs['scl_class_info'] = self.scl_class_info
        
        print(f'Data retrieved. Access with the .data attribute. Data CRS: {self.bbox_gdf.estimate_utm_crs().name}.')

    def get_metadata(self):
        """
        The method to get the metadata.
        """

        stac_json = self.search.item_collection_as_dict()
        metadata_gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")
        if self.catalog_choice == "earthsearch":
            metadata_gdf["s2:mgrs_tile"] = (
                metadata_gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
                + metadata_gdf["mgrs:latitude_band"]
                + metadata_gdf["mgrs:grid_square"]
            )

        self.metadata = metadata_gdf
        print(f'Metadata retrieved. Access with the .metadata attribute.')

    def mask_data(self,remove_nodata=True, remove_saturated_defective=True, remove_topo_shadows=True, remove_cloud_shadows=True, remove_vegetation=False, remove_not_vegetated=False, remove_water=False, remove_unclassified=False, remove_medium_prob_clouds=True,remove_high_prob_clouds=True, remove_thin_cirrus_clouds=True, remove_snow_ice=False):
        """
        The method to mask the data.

        Parameters:
            remove_nodata (bool): Whether to remove no data pixels.
            remove_saturated_defective (bool): Whether to remove saturated or defective pixels.
            remove_topo_shadows (bool): Whether to remove topographic shadow pixels.
            remove_cloud_shadows (bool): Whether to remove cloud shadow pixels.
            remove_vegetation (bool): Whether to remove vegetation pixels.
            remove_not_vegetated (bool): Whether to remove not vegetated pixels.
            remove_water (bool): Whether to remove water pixels.
            remove_unclassified (bool): Whether to remove unclassified pixels.
            remove_medium_prob_clouds (bool): Whether to remove medium probability cloud pixels.
            remove_high_prob_clouds (bool): Whether to remove high probability cloud pixels.
            remove_thin_cirrus_clouds (bool): Whether to remove thin cirrus cloud pixels.
            remove_snow_ice (bool): Whether to remove snow or ice pixels.
        """

        # Mask the data based on the Scene Classification (SCL) band (see definitions above)
        mask_list = []
        if remove_nodata:
            mask_list.append(0)
        if remove_saturated_defective:
            mask_list.append(1)
        if remove_topo_shadows:
            mask_list.append(2)
        if remove_cloud_shadows:
            mask_list.append(3)
        if remove_vegetation:
            mask_list.append(4)
        if remove_not_vegetated:
            mask_list.append(5)
        if remove_water:
            mask_list.append(6)
        if remove_unclassified:
            mask_list.append(7)
        if remove_medium_prob_clouds:
            mask_list.append(8)
        if remove_high_prob_clouds:
            mask_list.append(9)
        if remove_thin_cirrus_clouds:
            mask_list.append(10)
        if remove_snow_ice:
            mask_list.append(11)
        
        print(f'Removed pixels with the following scene classification values:')
        for val in mask_list:
            print(self.scl_class_info[val]["name"])

        scl = self.data.scl
        mask = scl.where(scl.isin(mask_list) == False, 0)
        self.data = self.data.where(mask != 0)

    def harmonize_to_old(self):
        """
        Harmonize new Sentinel-2 data to the old baseline.
        Adapted from: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
        Returns
        -------
        harmonized: xarray.Dataset
            A Dataset with all values harmonized to the old
            processing baseline.
        """
        cutoff = datetime.datetime(2022, 1, 25)
        offset = 1000
        bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
        bands = [self.data.band_info[band]["name"] for band in bands]
        old = self.data.sel(time=slice(None, cutoff))

        to_process = list(set(bands) & set(self.data.data_vars))
        new = self.data.sel(time=slice(cutoff, None))

        for band in to_process:
            if band in new.data_vars:
                new[band] = new[band].clip(offset) - offset

        self.data = xr.concat([old, new], dim="time")

        print(f'Data acquired after January 25th, 2022 harmonized to old baseline.')

    def get_rgb(self):
        """
        The method to get the RGB data.

        Returns:
            xarray.DataArray: The RGB data.
        """
        # Convert the red, green, and blue bands to an RGB DataArray
        rgb_da = self.data[['red','green','blue']].to_dataarray(dim='band')
        self.rgb = rgb_da

        print(f'RGB data retrieved. Access with the .rgb attribute.')

    # Indicies
    # find indicies Sentinel-2 indicies here: https://www.indexdatabase.de/db/is.php?sensor_id=96 and https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/

    def get_ndvi(self):
        """
        The method to get the NDVI data. 
        S2 NDVI definition: (B08 - B04) / (B08 + B04) [https://www.indexdatabase.de/db/i-single.php?id=58]

        Returns:
            ndvi_da (xarray.DataArray): The NDVI data.
        """
        red = self.data.red
        nir = self.data.nir
        ndvi_da = (nir - red) / (nir + red)

        self.ndvi = ndvi_da

        print(f'NDVI data calculated. Access with the .ndvi attribute.')

    def get_ndsi(self):
        """
        The method to get the NDSI data.
        S2 NDSI definition: (B03 - B11) / (B03 + B11) 

        Returns:
            ndsi_da (xarray.DataArray): The NDSI data.
        """
        green = self.data.green
        swir16 = self.data.swir16
        ndsi_da = (green - swir16) / (green + swir16)

        self.ndsi = ndsi_da

        print(f'NDSI data calculated. Access with the .ndsi attribute.')


    def get_ndwi(self):
        """
        The method to get the NDWI data.
        S2 NDWI definition: (B03 - B08) / (B03 + B08)

        Returns:
            ndwi_da (xarray.DataArray): The NDWI data.
        """
        green = self.data.green
        nir = self.data.nir
        ndwi_da = (green - nir) / (green + nir)

        self.ndwi = ndwi_da
        
        print(f'NDWI data calculated. Access with the .ndwi attribute.')

    def get_evi(self):
        """
        The method to get the EVI data.
        S2 EVI definition: 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1) [https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=16]

        Returns:
            xarray.DataArray: The EVI data.
        """ 
        red = self.data.red
        nir = self.data.nir
        blue = self.data.blue

        evi_da = 2.5 * (nir - red) / (nir+6*red-7.5*blue+1)

        self.evi = evi_da

        print(f'EVI data calculated. Access with the .evi attribute.')

    def get_ndbi(self):
        """
        The method to get the NDBI data.
        S2 NDBI definition: (B08 - B12) / (B08 + B12)

        Returns:
            xarray.DataArray: The NDBI data.
        """
        nir = self.data.nir
        swir22 = self.data.swir22
        ndbi_da = (nir - swir22) / (nir + swir22)

        self.ndbi = ndbi_da
        
        print(f'NDBI data calculated. Access with the .ndbi attribute.')





class Sentinel1:
    """
    A class to handle Sentinel-1 RTC satellite data.

    Attributes:
        bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
        start_date (str): The start date for the data in the format 'YYYY-MM-DD'.
        end_date (str): The end date for the data in the format 'YYYY-MM-DD'.
        catalog_choice (str): The catalog choice for the data. Can choose between 'planetarycomputer' and 'earthsearch', default is 'planetarycomputer'.
        bands (list): The bands to be used. Default is all bands. Must include SCL for data masking. Each band should be a string like 'B01', 'B02', etc.
        resolution (str): The resolution of the data. Defaults to native resolution, 10m.
        crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
        groupby (str): The groupby parameter for the data. Default is "sat:absolute_orbit".

    """


    def __init__(self, bbox_input, start_date='2014-01-01', end_date=today, catalog_choice='planetarycomputer', bands=None, resolution=None, crs=None, groupby='sat:absolute_orbit'):
        """
        The constructor for the Sentinel1 class.

        Parameters:
            bbox_input (geopandas.GeoDataFrame or tuple or shapely.Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
            start_date (str): The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
            end_date (str): The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
            catalog_choice (str): The catalog choice for the data. Can choose between 'planetarycomputer' and <unimplemented>, default is 'planetarycomputer'.
            bands (list): The bands to be used. Default is all bands. 
            resolution (str): The resolution of the data. Defaults to native resolution, 10m.
            crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
            groupby (str): The groupby parameter for the data. Default is "sat:absolute_orbit".
        """
        # Initialize the attributes
        self.bbox_input = bbox_input
        self.start_date = start_date
        self.end_date = end_date
        self.catalog_choice = catalog_choice
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.groupby = groupby

        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs == None:
            self.crs = self.bbox_gdf.estimate_utm_crs()


        self.search = None
        self.data = None
        self.metadata = None

        self.search_data()
        self.get_data()
        self.get_metadata()
        self.remove_border_noise()
        self.add_orbit_info()
        self.linear_to_db()

    def search_data(self):
        """
        The method to search the data.
        """
        
        # Choose the catalog URL based on catalog_choice
        if self.catalog_choice == "planetarycomputer":
            catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
            catalog = pystac_client.Client.open(catalog_url, modifier=planetary_computer.sign_inplace)
        # elif self.catalog_choice == "aws":
        #     catalog_url = indigo
        #     catalog = pystac_client.Client.open(catalog_url)
        else:
            raise ValueError("Invalid catalog_choice. Choose either 'planetarycomputer' or <unimplemented>.")
        
        # Search for items within the specified bbox and date range
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=self.bbox_gdf.total_bounds, datetime=(self.start_date, self.end_date))
        self.search = search
        print(f'Data searched. Access the returned seach with the .search attribute.')

    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params = {
            'items': self.search.items(),
            'bbox': self.bbox_gdf.total_bounds,
            'nodata': -32768,
            'chunks': {'x':512, 'y':512,'time':-1},
            'groupby': self.groupby
        }
        if self.bands:
            load_params['bands'] = self.bands
        if self.crs:
            load_params['crs'] = self.crs
        if self.resolution:
            load_params['resolution'] = self.resolution

        # Load the data lazily using odc.stac
        self.data = odc.stac.load(**load_params).sortby('time') # sorting by time because of known issue in s1 mpc stac catalog
        self.data.attrs['units'] = 'linear power'
        print(f'Data retrieved. Access with the .data attribute. Data CRS: {self.bbox_gdf.estimate_utm_crs().name}.')



    def get_metadata(self):
        """
        The method to get the metadata.
        """
        stac_json = self.search.item_collection_as_dict()
        metadata_gdf = gpd.GeoDataFrame.from_features(stac_json,"epsg:4326")

        self.metadata = metadata_gdf
        print(f'Metadata retrieved. Access with the .metadata attribute.')

    def remove_border_noise(self):
        """
        The method to remove border noise from the data.
        """
        self.data = self.data.where(self.data>0.006)
        print(f'Border noise removed from the data.')
    
    def linear_to_db(self):
        """
        The method to convert the linear power data to dB.
        """
        self.data = 10 * np.log10(self.data)
        self.data.attrs['units'] = 'dB'
        print(f'Linear power units converted to dB. Convert back to linear power units using the .db_to_linear() method.')

    def db_to_linear(self):
        """
        The method to convert the dB data to linear power.
        """
        self.data = 10 ** (self.data / 10)
        self.data.attrs['units'] = 'linear power'
        print(f'dB converted to linear power units. Convert back to dB using the .linear_to_db() method.')


    def add_orbit_info(self):
        """
        The method to add the relative orbit number to the data.
        """
        metadata_groupby_gdf = self.metadata.groupby([f'{self.groupby}']).first().sort_values('datetime')
        self.data = self.data.assign_coords({'sat:orbit_state':('time',metadata_groupby_gdf['sat:orbit_state'])})
        self.data = self.data.assign_coords({'sat:relative_orbit':('time',metadata_groupby_gdf['sat:relative_orbit'].astype('int16'))})
        print(f'Added relative orbit number and orbit state as coordinates to the data.')












# import pystac

# catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
# catalog = pystac_client.Client.open(catalog_url, modifier=planetary_computer.sign_inplace)


# items = search.get_all_items()

# # Initialize an empty list to store the patched items
# patched_items = []

# # Iterate over the items
# for item in items:
#     # Convert the item to a dictionary
#     item_dict = item.to_dict()

#     # Modify the 'raster:bands' attribute of the 'visual' asset
#     item_dict['assets']['visual']['raster:bands'] = [{'data_type': 'uint16'}]*3

#     # Convert the dictionary back to an Item
#     item_patched = pystac.Item.from_dict(item_dict)

#     # Append the patched item to the list
#     patched_items.append(item_patched)

#     odc.stac.load(patched_items,bbox=bbox_gdf.total_bounds,chunks={},bands=("visual.1","visual.2","visual.3"),nodata=0).to_dataarray(dim='band').isel(time=10).plot.imshow(rgb='band',robust=True)





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
