import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask
import pystac_client
import planetary_computer
import os
import earthaccess
import ee

import rasterio as rio

rio_env = rio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
    CPL_VSIL_CURL_USE_HEAD="FALSE",
    GDAL_HTTP_NETRC="TRUE",
    GDAL_HTTP_COOKIEFILE=os.path.expanduser("~/cookies.txt"),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser("~/cookies.txt"),
)
rio_env.__enter__()

import odc.stac

odc.stac.configure_rio(cloud_defaults=True)

import datetime

today = datetime.datetime.now().strftime("%Y-%m-%d")

from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    get_stac_cfg,
    HLS_xml_url_to_metadata_df,
    blockPrint,
    enablePrint
)


def authenticate_all():
    """
    Authenticates with all potential data providers.

    This function authenticates with NASA EarthData, Planetary Computer, and Earth Engine.
    It prints the authentication status for each provider.
    """
    print("Authenticating for all potential data providers...")

    print("Authenticating for NASA EarthData...")
    earthaccess.login(persist=True)
    # print("Authenticating for Planetary Computer...")
    # planetary_computer.set_subscription_key()
    print("Authenticating for Earth Engine...")
    ee.Authenticate()


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

    fcf_da = rxr.open_rasterio(
        "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        chunks=True,
        mask_and_scale=True,
    )
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
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    snow_classification_da = rxr.open_rasterio(
        "https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif",
        chunks=True,
        mask_and_scale=True,
    )
    snow_classification_da = (
        snow_classification_da.rio.clip_box(xmin, ymin, xmax, ymax)
        .squeeze()
        .rio.write_nodata(9, encoded=True)
    )

    snow_classification_da.attrs["class_info"] = {
        1: {"name": "Tundra", "color": "#a100c8"},
        2: {"name": "Boreal Forest", "color": "#00a0fe"},
        3: {"name": "Maritime", "color": "#fe0000"},
        4: {"name": "Ephemeral (includes no snow)", "color": "#e7dc32"},
        5: {"name": "Prairie", "color": "#f08328"},
        6: {"name": "Montane Forest", "color": "#00dc00"},
        7: {"name": "Ice (glaciers and ice sheets)", "color": "#aaaaaa"},
        8: {"name": "Ocean", "color": "#0000ff"},
        9: {"name": "Fill", "color": "#ffffff"},
    }

    return snow_classification_da


def get_seasonal_mountain_snow_mask(
    bbox_input, data_product="mountain_snow"
) -> xr.DataArray:
    """
    Fetches ~1km static global seasonal (mountain snow / snow) mask for a given bounding box.

    Description:
    "Seasonal Mountain Snow (SMS) mask derived from MODIS MOD10A2 snow cover extent and GTOPO30 digital elevation model produced at 30 arcsecond spatial resolution.
    Three datasets are provided: the Seasonal Mountain Snow mask (MODIS_mtnsnow_classes), a seasonal snow cover classification (MODIS_snow_classes), and cool-season cloud percentages (MODIS_clouds).

    The classification systems are as follows:

    MODIS_snow_classes:
    0: Little-to-no snow
    1: Indeterminate due to clouds
    2: Ephemeral snow
    3: Seasonal snow

    MODIS_mtnsnow_classes:
    0: Mountains with little-to-no snow
    1: Indeterminate due to clouds
    2: Mountains with ephemeral snow
    3: Mountains with seasonal snow

    Citation:
    Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019). Global Seasonal Mountain Snow Mask from MODIS MOD10A2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2626737

    Parameters:
    bbox_input (geopandas.GeoDataFrame or tuple or shapely.Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    data_product (str): Data product to fetch. Choose from 'snow' or 'mountain_snow'. Default is 'mountain_snow'.

    Returns:
    mountain_snow_da (xarray.DataArray): Mountain snow DataArray.
    """

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    if data_product == "snow":
        url = "zip+https://zenodo.org/records/2626737/files/MODIS_snow_classes.zip!/MODIS_snow_classes.tif"
        class_dict = {
            0: {"name": "Little-to-no snow", "color": "#FFFFFF"},
            1: {"name": "Indeterminate due to clouds", "color": "#FF0000"},
            2: {"name": "Ephemeral snow", "color": "#00FF00"},
            3: {"name": "Seasonal snow", "color": "#0000FF"},
        }
    elif data_product == "mountain_snow":
        url = "zip+https://zenodo.org/records/2626737/files/MODIS_mtnsnow_classes.zip!/MODIS_mtnsnow_classes.tif"
        class_dict = {
            0: {"name": "Mountains with little-to-no snow", "color": "#FFFFFF"},
            1: {"name": "Indeterminate due to clouds", "color": "#FF0000"},
            2: {"name": "Mountains with ephemeral snow", "color": "#00FF00"},
            3: {"name": "Mountains with seasonal snow", "color": "#0000FF"},
        }
    else:
        raise ValueError('Invalid data_product. Choose from "snow" or "mountain_snow".')

    mountain_snow_da = rxr.open_rasterio(
        url,
        chunks=True,
        mask_and_scale=True,
    )
    mountain_snow_da = (
        mountain_snow_da.rio.clip_box(xmin, ymin, xmax, ymax, crs="EPSG:4326")
        .squeeze()
        .astype("float32")
    )

    mountain_snow_da.attrs["class_info"] = class_dict

    return mountain_snow_da


def get_esa_worldcover(bbox_input, version: str = "v200") -> xr.DataArray:
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
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    if version == "v100":
        version_year = "2020"
    elif version == "v200":
        version_year = "2021"
    else:
        raise ValueError("Incorrect version number. Please provide 'v100' or 'v200'.")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox_gdf.total_bounds)
    worldcover_da = (
        odc.stac.load(
            search.items(), bbox=bbox_gdf.total_bounds, bands="map", chunks={}
        )["map"]
        .sel(time=version_year)
        .squeeze()
    )
    worldcover_da = worldcover_da.rio.write_nodata(0, encoded=True)

    worldcover_da.attrs["class_info"] = {
        10: {"name": "Tree cover", "color": "#006400"},
        20: {"name": "Shrubland", "color": "#FFBB22"},
        30: {"name": "Grassland", "color": "#FFFF4C"},
        40: {"name": "Cropland", "color": "#F096FF"},
        50: {"name": "Built-up", "color": "#FA0000"},
        60: {"name": "Bare / sparse vegetation", "color": "#B4B4B4"},
        70: {"name": "Snow and ice", "color": "#F0F0F0"},
        80: {"name": "Permanent water bodies", "color": "#0064C8"},
        90: {"name": "Herbaceous wetland", "color": "#0096A0"},
        95: {"name": "Mangroves", "color": "#00CF75"},
        100: {"name": "Moss and lichen", "color": "#FAE6A0"},
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

    def __init__(
        self,
        bbox_input,
        start_date="2014-01-01",
        end_date=today,
        catalog_choice="planetarycomputer",
        bands=None,
        resolution=None,
        crs=None,
        remove_nodata=True,
        harmonize_to_old=True,
        scale_data=True,
        groupby="solar_day",
    ):
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
        self.remove_nodata = remove_nodata
        self.harmonize_to_old = harmonize_to_old
        self.scale_data = scale_data
        self.groupby = groupby

        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs == None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # Define the band information
        self.band_info = {
            "B01": {
                "name": "coastal",
                "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)",
                "resolution": "60m",
                "scale": "0.0001",
            },
            "B02": {
                "name": "blue",
                "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)",
                "resolution": "10m",
                "scale": "0.0001",
            },
            "B03": {
                "name": "green",
                "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)",
                "resolution": "10m",
                "scale": "0.0001",
            },
            "B04": {
                "name": "red",
                "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)",
                "resolution": "10m",
                "scale": "0.0001",
            },
            "B05": {
                "name": "rededge",
                "description": "Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "B06": {
                "name": "rededge2",
                "description": "Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "B07": {
                "name": "rededge3",
                "description": "Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "B08": {
                "name": "nir",
                "description": "NIR, 832.8 nm (S2A), 833.0 nm (S2B)",
                "resolution": "10m",
                "scale": "0.0001",
            },
            "B8A": {
                "name": "nir08",
                "description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "B09": {
                "name": "nir09",
                "description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)",
                "resolution": "60m",
                "scale": "0.0001",
            },
            "B11": {
                "name": "swir16",
                "description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "B12": {
                "name": "swir22",
                "description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)",
                "resolution": "20m",
                "scale": "0.0001",
            },
            "AOT": {
                "name": "aot",
                "description": "Aerosol Optical Thickness map, based on Sen2Cor processor",
                "resolution": "10m",
                "scale": "1",
            },
            "SCL": {
                "name": "scl",
                "description": "Scene classification data, based on Sen2Cor processor",
                "resolution": "20m",
                "scale": "1",
            },
            "WVP": {
                "name": "wvp",
                "description": "Water Vapour map",
                "resolution": "10m",
                "scale": "1",
            },
            "visual": {
                "name": "visual",
                "description": "True color image",
                "resolution": "10m",
                "scale": "0.0001",
            },
        }

        # Define the scene classification information, classes here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
        self.scl_class_info = {
            0: {"name": "No Data (Missing data)", "color": "#000000"},
            1: {"name": "Saturated or defective pixel", "color": "#ff0000"},
            2: {
                "name": "Topographic casted shadows",
                "color": "#2f2f2f",
            },  # (called 'Dark features/Shadows' for data before 2022-01-25)
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

        if self.remove_nodata:
            self.remove_nodata_inplace()

        if self.harmonize_to_old:
            self.harmonize_to_old_inplace()

        if self.scale_data:
            self.scale_data_inplace()

        self.get_metadata()

    def search_data(self):
        """
        The method to search the data.
        """

        # Choose the catalog URL based on catalog_choice
        if self.catalog_choice == "planetarycomputer":
            catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
            catalog = pystac_client.Client.open(
                catalog_url, modifier=planetary_computer.sign_inplace
            )
        elif self.catalog_choice == "earthsearch":
            os.environ["AWS_REGION"] = "us-west-2"
            os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
            os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            catalog_url = "https://earth-search.aws.element84.com/v1"
            catalog = pystac_client.Client.open(catalog_url)
        else:
            raise ValueError(
                "Invalid catalog_choice. Choose either 'planetarycomputer' or 'earthsearch'."
            )

        # Search for items within the specified bbox and date range
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=self.bbox_gdf.total_bounds,
            datetime=(self.start_date, self.end_date),
        )
        self.search = search
        print(f"Data searched. Access the returned seach with the .search attribute.")

    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params = {
            "items": self.search.items(),
            "bbox": self.bbox_gdf.total_bounds,
            "nodata": 0,
            "chunks": {},
            "crs": self.crs,
            "groupby": self.groupby,
            "stac_cfg": get_stac_cfg(sensor="sentinel-2-l2a"),
        }
        if self.bands:
            load_params["bands"] = self.bands
        else:
            load_params["bands"] = [info["name"] for info in self.band_info.values()]
        if self.resolution:
            load_params["resolution"] = self.resolution

        # Load the data lazily using odc.stac
        self.data = odc.stac.load(**load_params)

        self.data.attrs["band_info"] = self.band_info
        self.data.attrs["scl_class_info"] = self.scl_class_info

        if "scl" in self.data.variables:
            self.data.scl.attrs["scl_class_info"] = self.scl_class_info

        print(
            f"Data retrieved. Access with the .data attribute. Data CRS: {self.bbox_gdf.estimate_utm_crs().name}."
        )

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
        print(f"Metadata retrieved. Access with the .metadata attribute.")

    def remove_nodata_inplace(self):
        """
        The method to remove no data values from the data.
        """
        for band in self.data.data_vars:
            nodata_value = self.data[band].attrs.get("nodata")
            if nodata_value is not None:
                self.data[band] = self.data[band].where(self.data[band] != nodata_value)
        print(
            f"Nodata values removed from the data. In doing so, all bands converted to float32. To turn this behavior off, set remove_nodata=False."
        )

    def mask_data(
        self,
        remove_nodata=True,
        remove_saturated_defective=True,
        remove_topo_shadows=True,
        remove_cloud_shadows=True,
        remove_vegetation=False,
        remove_not_vegetated=False,
        remove_water=False,
        remove_unclassified=False,
        remove_medium_prob_clouds=True,
        remove_high_prob_clouds=True,
        remove_thin_cirrus_clouds=True,
        remove_snow_ice=False,
    ):
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

        print(f"Removed pixels with the following scene classification values:")
        for val in mask_list:
            print(self.scl_class_info[val]["name"])

        scl = self.data.scl
        mask = scl.where(scl.isin(mask_list) == False, 0)
        self.data = self.data.where(mask != 0)

    def harmonize_to_old_inplace(self):
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

        print(
            f"Data acquired after January 25th, 2022 harmonized to old baseline. To turn this behavior off, set harmonize_to_old=False."
        )

    def scale_data_inplace(self):
        """
        The method to scale the data.
        """
        for band in self.data.data_vars:
            scale_factor = self.data[band].attrs.get("scale")
            if scale_factor is not None:
                self.data[band] = self.data[band] * scale_factor
        print(
            f"Data scaled to reflectance. To turn this behavior off, set scale_data=False."
        )

    def get_rgb(self):
        """
        The method to get the RGB data.

        Returns:
            xarray.DataArray: The RGB data.
        """
        # Convert the red, green, and blue bands to an RGB DataArray
        rgb_da = self.data[["red", "green", "blue"]].to_dataarray(dim="band")
        self.rgb = rgb_da.squeeze()
        self.rgb_norm = (self.rgb - self.rgb.min(dim=['x', 'y'])) / (self.rgb.max(dim=['x', 'y']) - self.rgb.min(dim=['x', 'y']))

        print(f"RGB data retrieved. Access with the .rgb attribute, or .rgb_norm for normalized RGB.")

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

        print(f"NDVI data calculated. Access with the .ndvi attribute.")

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

        print(f"NDSI data calculated. Access with the .ndsi attribute.")

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

        print(f"NDWI data calculated. Access with the .ndwi attribute.")

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

        evi_da = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

        self.evi = evi_da

        print(f"EVI data calculated. Access with the .evi attribute.")

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

        print(f"NDBI data calculated. Access with the .ndbi attribute.")


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

    def __init__(
        self,
        bbox_input,
        start_date="2014-01-01",
        end_date=today,
        catalog_choice="planetarycomputer",
        bands=None,
        units='dB', # linear power or dB
        resolution=None,
        crs=None,
        groupby="sat:absolute_orbit",
        chunks={}, # {"x": 512, "y": 512} or # {"x": 512, "y": 512, "time": -1}
    ):
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
        self.chunks = chunks
        self.groupby = groupby

        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs == None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # if resolution == None:
        #     self.resolution = 10

        self.search = None
        self.data = None
        self.metadata = None

        self.search_data()
        self.get_data()
        self.get_metadata()
        self.remove_border_noise()
        self.add_orbit_info()
        if units == 'dB':
            self.linear_to_db()
        else:
            print('Units remain in linear power. Convert to dB using the .linear_to_db() method.')

    def search_data(self):
        """
        The method to search the data.
        """

        # Choose the catalog URL based on catalog_choice
        if self.catalog_choice == "planetarycomputer":
            catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
            catalog = pystac_client.Client.open(
                catalog_url, modifier=planetary_computer.sign_inplace
            )
        # elif self.catalog_choice == "aws":
        #     catalog_url = indigo
        #     catalog = pystac_client.Client.open(catalog_url)
        else:
            raise ValueError(
                "Invalid catalog_choice. Choose either 'planetarycomputer' or <unimplemented>."
            )

        # Search for items within the specified bbox and date range
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=self.bbox_gdf.total_bounds,
            datetime=(self.start_date, self.end_date),
        )
        self.search = search
        print(f"Data searched. Access the returned seach with the .search attribute.")

    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params = {
            "items": self.search.items(),
            "bbox": self.bbox_gdf.total_bounds,
            "nodata": -32768,
            "chunks": self.chunks,
            "groupby": self.groupby,
        }
        if self.bands:
            load_params["bands"] = self.bands
        if self.crs:
            load_params["crs"] = self.crs
        if self.resolution:
            load_params["resolution"] = self.resolution

        # Load the data lazily using odc.stac
        self.data = odc.stac.load(**load_params).sortby(
            "time"
        )  # sorting by time because of known issue in s1 mpc stac catalog
        self.data.attrs["units"] = "linear power"
        print(
            f"Data retrieved. Access with the .data attribute. Data CRS: {self.bbox_gdf.estimate_utm_crs().name}."
        )

    def get_metadata(self):
        """
        The method to get the metadata.
        """
        stac_json = self.search.item_collection_as_dict()
        metadata_gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")

        self.metadata = metadata_gdf
        print(f"Metadata retrieved. Access with the .metadata attribute.")

    def remove_border_noise(self,threshold=0.001):
        """
        The method to remove border noise from the data.
        https://forum.step.esa.int/t/grd-border-noise-and-thermal-noise-removal-are-not-working-anymore-since-march-13-2018/9332
        https://www.mdpi.com/2072-4292/8/4/348
        https://forum.step.esa.int/t/nan-appears-at-the-edge-of-the-scene-after-applying-border-noise-removal-sentinel-1-grd/40627/2
        https://sentiwiki.copernicus.eu/__attachments/1673968/OI-MPC-OTH-MPC-0243%20-%20Sentinel-1%20masking%20no%20value%20pixels%20grd%20products%20note%202023%20-%202.2.pdf?inst-v=534578f3-fc04-48e9-bd69-3a45a681fe67#page=12.58
        https://ieeexplore.ieee.org/document/8255846
        https://www.mdpi.com/2504-3900/2/7/330
        """
        self.data.loc[dict(time=slice('2014-01-01','2018-03-14'))] = self.data.sel(time=slice('2014-01-01','2018-03-14')).where(lambda x: x > threshold)
        print(f"Border noise removed from the data.")

    def linear_to_db(self):
        """
        The method to convert the linear power data to dB.
        """
        self.data = 10 * np.log10(self.data)
        self.data.attrs["units"] = "dB"
        print(
            f"Linear power units converted to dB. Convert back to linear power units using the .db_to_linear() method."
        )

    def db_to_linear(self):
        """
        The method to convert the dB data to linear power.
        """
        self.data = 10 ** (self.data / 10)
        self.data.attrs["units"] = "linear power"
        print(
            f"dB converted to linear power units. Convert back to dB using the .linear_to_db() method."
        )

    def add_orbit_info(self):
        """
        The method to add the relative orbit number to the data.
        """
        metadata_groupby_gdf = (
            self.metadata.groupby([f"{self.groupby}"]).first().sort_values("datetime")
        )
        self.data = self.data.assign_coords(
            {"sat:orbit_state": ("time", metadata_groupby_gdf["sat:orbit_state"])}
        )
        self.data = self.data.assign_coords(
            {
                "sat:relative_orbit": (
                    "time",
                    metadata_groupby_gdf["sat:relative_orbit"].astype("int16"),
                )
            }
        )
        print(
            f"Added relative orbit number and orbit state as coordinates to the data."
        )


class HLS:
    """
    A class to handle Harmonlized Landsat Sentinel satellite data.

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

    # https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
    # https://lpdaac.usgs.gov/documents/842/HLS_Tutorial.html

    def __init__(
        self,
        bbox_input,
        start_date="2014-01-01",
        end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
        bands=None,
        resolution=None,
        crs="utm",
        remove_nodata=True,
        scale_data=True,
        add_metadata=True,
        add_platform=True,
        groupby="solar_day",
    ):  #'ProducerGranuleId'
        """
        The constructor for the HLS class.

        Parameters:
            bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
            start_date (str): The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
            end_date (str): The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
            bands (list): The bands to be used. Default is all bands. Must include SCL for data masking. Each band should be a string like 'B01', 'B02', etc.
            resolution (str): The resolution of the data. Defaults to native resolution, 10m.
            crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
            groupby (str): The groupby parameter for the data. Default is "solar_day".
        """
        # Initialize the attributes
        self.bbox_input = bbox_input
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.remove_nodata = remove_nodata
        self.scale_data = scale_data
        self.add_metadata = add_metadata
        self.add_platform = add_platform
        self.groupby = groupby

        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs == None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # Define the band information
        self.band_info = {
            "coastal aerosol": {
                "landsat_band": "B01",
                "sentinel_band": "B01",
                "description": "430-450 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "blue": {
                "landsat_band": "B02",
                "sentinel_band": "B02",
                "description": "450-510 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "green": {
                "landsat_band": "B03",
                "sentinel_band": "B03",
                "description": "530-590 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "red": {
                "landsat_band": "B04",
                "sentinel_band": "B04",
                "description": "640-670 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "red-edge 1": {
                "landsat_band": "-",
                "sentinel_band": "B05",
                "description": "690-710 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "red-edge 2": {
                "landsat_band": "-",
                "sentinel_band": "B06",
                "description": "730-750 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "red-edge 3": {
                "landsat_band": "-",
                "sentinel_band": "B07",
                "description": "770-790 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "nir broad": {
                "landsat_band": "-",
                "sentinel_band": "B08",
                "description": "780-880 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "nir narrow": {
                "landsat_band": "B05",
                "sentinel_band": "B8A",
                "description": "850-880 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "swir 1": {
                "landsat_band": "B06",
                "sentinel_band": "B11",
                "description": "1570-1650 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "swir 2": {
                "landsat_band": "B07",
                "sentinel_band": "B12",
                "description": "2110-2290 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "water vapor": {
                "landsat_band": "-",
                "sentinel_band": "B09",
                "description": "930-950 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "cirrus": {
                "landsat_band": "B09",
                "sentinel_band": "B10",
                "description": "1360-1380 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "thermal infrared 1": {
                "landsat_band": "B10",
                "sentinel_band": "-",
                "description": "10600-11190 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "thermal": {
                "landsat_band": "B11",
                "sentinel_band": "-",
                "description": "11500-12510 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "Fmask": {
                "landsat_band": "Fmask",
                "sentinel_band": "Fmask",
                "description": "quality bits",
                "data_type": "uint8",
                "nodata": "255",
                "scale": "1",
            },
            "SZA": {
                "landsat_band": "SZA",
                "sentinel_band": "SZA",
                "description": "Sun zenith degrees",
                "data_type": "uint16",
                "nodata": "40000",
                "scale": "0.01",
            },
            "SAA": {
                "landsat_band": "SAA",
                "sentinel_band": "SAA",
                "description": "Sun azimuth degrees",
                "data_type": "uint16",
                "nodata": "40000",
                "scale": "0.01",
            },
            "VZA": {
                "landsat_band": "VZA",
                "sentinel_band": "VZA",
                "description": "View zenith degrees",
                "data_type": "uint16",
                "nodata": "40000",
                "scale": "0.01",
            },
            "VAA": {
                "landsat_band": "VAA",
                "sentinel_band": "VAA",
                "description": "View azimuth degrees",
                "data_type": "uint16",
                "nodata": "40000",
                "scale": "0.01",
            },
        }

        self.Fmask_mask_info = {
            0: {"name": "Cirrus", "bit number": "0"},
            1: {"name": "Cloud", "bit number": "1"},
            2: {"name": "Adjacent to cloud / shadow", "bit number": "2"},
            3: {"name": "Cloud shadows", "bit number": "3"},
            4: {"name": "Snow / ice", "bit number": "4"},
            5: {"name": "Water", "bit number": "5"},
            6: {
                "name": "Aerosol level (00:climatology aersol,01:low aerosol,10:moderate aerosol, 11:high aerosol)",
                "bit number": "6-7",
            },
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
        if self.remove_nodata:
            self.remove_nodata_inplace()
        if self.scale_data:
            self.scale_data_inplace()
        if self.add_metadata:
            self.get_combined_metadata()
        if self.add_platform:
            self.add_platform_inplace()

    def search_data(self):
        """
        The method to search the data.
        """

        catalog = pystac_client.Client.open(
            "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
        )

        # Search for items within the specified bbox and date range
        landsat_search = catalog.search(
            collections=["HLSL30.v2.0"],
            bbox=self.bbox_gdf.total_bounds,
            datetime=(self.start_date, self.end_date),
        )
        sentinel_search = catalog.search(
            collections=["HLSS30.v2.0"],
            bbox=self.bbox_gdf.total_bounds,
            datetime=(self.start_date, self.end_date),
        )

        self.search_landsat = landsat_search
        self.search_sentinel = sentinel_search
        print(
            f"Data searched. Access the returned seach with the .search_landsat or .search_sentinel attribute."
        )

    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params_landsat = {
            "items": self.search_landsat.item_collection(),
            "bbox": self.bbox_gdf.total_bounds,
            "chunks": {"time": 1, "x": 512, "y": 512},
            "crs": self.crs,  # maybe put 'utm'?
            "groupby": self.groupby,
            "fail_on_error": False,
            "stac_cfg": get_stac_cfg(sensor="HLSL30.v2.0"),
        }
        if self.bands:
            load_params_landsat["bands"] = self.bands
        else:
            load_params_landsat["bands"] = [
                band
                for band, info in self.band_info.items()
                if info["landsat_band"] != "-"
            ]
        if self.resolution:
            load_params_landsat["resolution"] = self.resolution
        else:
            load_params_landsat["resolution"] = 30

        L30_ds = odc.stac.load(**load_params_landsat)

        load_params_sentinel = {
            "items": self.search_sentinel.item_collection(),
            "bbox": self.bbox_gdf.total_bounds,
            "chunks": {"time": 1, "x": 512, "y": 512},
            "crs": self.crs,
            "groupby": self.groupby,
            "fail_on_error": False,
            "stac_cfg": get_stac_cfg(sensor="HLSS30.v2.0"),
        }
        if self.bands:
            load_params_sentinel["bands"] = self.bands
        else:
            load_params_sentinel["bands"] = [
                band
                for band, info in self.band_info.items()
                if info["sentinel_band"] != "-"
            ]
        if self.resolution:
            load_params_sentinel["resolution"] = self.resolution
        else:
            load_params_sentinel["resolution"] = 30

        S30_ds = odc.stac.load(**load_params_sentinel)

        # Load the data lazily using odc.stac
        self.data = xr.concat((L30_ds, S30_ds), dim="time", fill_value=-9999).sortby(
            "time"
        )

        self.data.attrs["band_info"] = self.band_info
        # self.data.attrs['scl_class_info'] = self.scl_class_info

        # if 'scl' in self.data.variables:
        #    self.data.scl.attrs['scl_class_info'] = self.scl_class_info

        print(
            f"Data retrieved. Access with the .data attribute. Data CRS: {self.bbox_gdf.estimate_utm_crs().name}."
        )

    def remove_nodata_inplace(self):
        """
        The method to remove no data values from the data.
        """
        for band in self.data.data_vars:
            nodata_value = self.data[band].attrs.get("nodata")
            if nodata_value is not None:
                self.data[band] = self.data[band].where(self.data[band] != nodata_value)
        print(
            f"Nodata values removed from the data. In doing so, all bands converted to float32. To turn this behavior off, set remove_nodata=False."
        )

    def mask_data(
        self,
        remove_cirrus=True,
        remove_cloud=True,
        remove_adj_to_cloud=True,
        remove_cloud_shadows=True,
        remove_snow_ice=False,
        remove_water=False,
        remove_aerosol_low=False,
        remove_aerosol_moderate=False,
        remove_aerosol_high=False,
        remove_aerosol_climatology=False,
    ):
        """
        The method to mask the data using Fmask.

        Parameters:
            remove_cirrus (bool): Whether to remove cirrus pixels.
            remove_cloud (bool): Whether to remove cloud pixels.
            remove_adj_to_cloud (bool): Whether to remove pixels adjacent to clouds.
            remove_cloud_shadows (bool): Whether to remove cloud shadow pixels.
            remove_snow_ice (bool): Whether to remove snow and ice pixels.
            remove_water (bool): Whether to remove water pixels.
            remove_aerosol_low (bool): Whether to remove low aerosol pixels.
            remove_aerosol_moderate (bool): Whether to remove moderate aerosol pixels.
            remove_aerosol_high (bool): Whether to remove high aerosol pixels.
            remove_aerosol_climatology (bool): Whether to remove climatology aerosol pixels.


        """

        # Get value of QC bit based on location
        def get_qc_bit(ar, bit):
            # taken from Helen's fantastic repo https://github.com/UW-GDA/mekong-water-quality/blob/main/02_pull_hls.ipynb
            return (ar // (2**bit)) - ((ar // (2**bit)) // 2 * 2)

        mask = xr.DataArray()
        # Mask the data based on the Fmask values
        mask_list = []
        if remove_cirrus:
            mask_list.append(0)
        if remove_cloud:
            mask_list.append(1)
        if remove_adj_to_cloud:
            mask_list.append(2)
        if remove_cloud_shadows:
            mask_list.append(3)
        if remove_snow_ice:
            mask_list.append(4)
        if remove_water:
            mask_list.append(5)
        if (
            remove_aerosol_climatology
            | remove_aerosol_low
            | remove_aerosol_moderate
            | remove_aerosol_high
        ):
            mask_list.append(6)
            aerosol_mask = (
                get_qc_bit(self.data["Fmask"], 6)
                .astype(str)
                .str.cat(get_qc_bit(self.data["Fmask"], 7).astype(str))
            )
            if remove_aerosol_climatology:
                mask = xr.concat([mask, aerosol_mask == "00"], dim="masks")
            if remove_aerosol_low:
                mask = xr.concat([mask, aerosol_mask == "01"], dim="masks")
            if remove_aerosol_moderate:
                mask = xr.concat([mask, aerosol_mask == "10"], dim="masks")
            if remove_aerosol_high:
                mask = xr.concat([mask, aerosol_mask == "11"], dim="masks")

        for val in mask_list:
            if val != 6:
                mask = xr.concat(
                    [mask, get_qc_bit(self.data["Fmask"], val)], dim="masks"
                )

        mask = mask.sum(dim="masks")
        self.data = self.data.where(mask == 0)

        print(
            f"WARNING: The cloud masking is pretty bad over snow and ice. Use with caution."
        )
        print(f"Data masked. Using Fmask, removed pixels classified as:")
        for val in mask_list:
            print(self.Fmask_mask_info[val]["name"])

    def get_metadata(self, item_collection):

        HLS_metadata = gpd.GeoDataFrame.from_features(
            item_collection.to_dict(transform_hrefs=True), "EPSG:4326"
        )
        HLS_metdata = HLS_metadata.drop(
            columns=["start_datetime", "end_datetime"], inplace=True
        )
        HLS_metadata["datetime"] = pd.to_datetime(HLS_metadata["datetime"], utc=True)

        series_list = []
        for item in item_collection:
            url = item.assets["metadata"].href
            series = HLS_xml_url_to_metadata_df(url)
            series_list.append(series)

        extra_attributes = pd.DataFrame(series_list)
        extra_attributes["Temporal"] = pd.to_datetime(extra_attributes["Temporal"])
        extra_attributes["Platform"] = extra_attributes["Platform"].str.title()

        metadata_gdf = gpd.GeoDataFrame(
            pd.merge_asof(
                HLS_metadata,
                extra_attributes,
                left_on="datetime",
                right_on="Temporal",
                direction="nearest",
                tolerance=pd.Timedelta("100ms"),
            )
        ).drop(columns="Temporal")
        metadata_gdf = metadata_gdf[
            [
                "datetime",
                "ProducerGranuleId",
                "Platform",
                "eo:cloud_cover",
                "AssociatedBrowseImageUrls",
                "geometry",
            ]
        ]

        return metadata_gdf

    def get_combined_metadata(self):
        L30_metadata = self.get_metadata(self.search_landsat.item_collection())
        S30_metadata = self.get_metadata(self.search_sentinel.item_collection())
        combined_metadata_gdf = (
            pd.concat([L30_metadata, S30_metadata])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        self.metadata = combined_metadata_gdf
        print(
            f"Metadata retrieved. Access with the .metadata attribute. To turn this behavior off, set add_metadata=False."
        )

    def add_platform_inplace(self):
        temp_grouped_metadata = self.metadata
        temp_grouped_metadata["cluster"] = (
            temp_grouped_metadata["datetime"].diff().dt.total_seconds().gt(60).cumsum()
        )

        grouped_metadata = pd.DataFrame()
        grouped_metadata["datetime"] = temp_grouped_metadata.groupby("cluster")[
            "datetime"
        ].apply(np.mean)
        grouped_metadata["Platforms"] = temp_grouped_metadata.groupby("cluster")[
            "Platform"
        ].apply(np.unique)
        grouped_metadata["eo:cloud_cover_avg"] = (
            temp_grouped_metadata.groupby("cluster")["eo:cloud_cover"]
            .apply(np.mean)
            .astype(int)
        )
        grouped_metadata["BrowseUrls"] = self.metadata.groupby("cluster")[
            "AssociatedBrowseImageUrls"
        ].apply(list)
        grouped_metadata["geometry"] = (
            temp_grouped_metadata.groupby("cluster")["geometry"]
            .apply(list)
            .apply(shapely.geometry.MultiPolygon)
        )
        grouped_metadata_gdf = gpd.GeoDataFrame(grouped_metadata).sort_values(
            "datetime"
        )

        self.data = self.data.assign_coords(
            {
                "platform": (
                    "time",
                    [item[0] for item in grouped_metadata_gdf["Platforms"].values],
                )
            }
        )
        self.data = self.data.assign_coords(
            {
                "eo:cloud_cover_avg": (
                    "time",
                    grouped_metadata_gdf["eo:cloud_cover_avg"].values,
                )
            }
        )
        self.data = self.data.assign_coords(
            {
                "AssociatedBrowseImageUrls": (
                    "time",
                    grouped_metadata_gdf["BrowseUrls"].values,
                )
            }
        )
        self.data = self.data.assign_coords(
            {"geometry": ("time", grouped_metadata_gdf["geometry"].values)}
        )

        print(
            f"Platform, geometry, cloud cover, browse URLs added to data as coordinates. Access with the .data attribute. To turn this behavior off, set add_platform=False."
        )

    def scale_data_inplace(self):
        """
        The method to scale the data.
        """

        # Define a function to scale a data variable
        def scale_var(x):
            band = x.name
            if band in self.data.band_info:
                scale_factor_dict = self.data.band_info[band]
                # Extract the actual scale factor from the dictionary and convert it to a float
                scale_factor = float(scale_factor_dict["scale"])
                return x * scale_factor
            else:
                return x

        # Apply the function to each data variable in the Dataset
        self.data = self.data.apply(scale_var)
        print(
            f"Data scaled to reflectance. Access with the .data attribute. To turn this behavior off, set scale_data=False."
        )

    def get_rgb(self):
        """
        The method to get the RGB data.

        Returns:
            xarray.DataArray: The RGB data.
        """
        # Convert the red, green, and blue bands to an RGB DataArray
        rgb_da = self.data[["red", "green", "blue"]].to_dataarray(dim="band")
        self.rgb = rgb_da

        print(f"RGB data retrieved. Access with the .rgb attribute.")

    # Indicies

    def get_ndvi(self):
        """
        The method to get the NDVI data.

        Returns:
            ndvi_da (xarray.DataArray): The NDVI data.
        """
        red = self.data.red
        nir = self.data.nir
        ndvi_da = (nir - red) / (nir + red)

        self.ndvi = ndvi_da

        print(f"NDVI data calculated. Access with the .ndvi attribute.")


class MODIS_snow:
    """
    A class to handle MODIS snow data.

    Data product options:
    MOD10A1: "This global Level-3 (L3) data set provides a daily composite of snow cover and albedo derived from the 'MODIS/Terra Snow Cover 5-Min L2 Swath 500m' data set (DOI:10.5067/MODIS/MOD10_L2.061). Each data granule is a 10°x10° tile projected to a 500 m sinusoidal grid." https://planetarycomputer.microsoft.com/dataset/modis-10A1-061#overview
    MOD10A2: "This global Level-3 (L3) data set provides the maximum snow cover extent observed over an eight-day period within 10degx10deg MODIS sinusoidal grid tiles. Tiles are generated by compositing 500 m observations from the 'MODIS Snow Cover Daily L3 Global 500m Grid' data set. A bit flag index is used to track the eight-day snow/no-snow chronology for each 500 m cell." https://planetarycomputer.microsoft.com/dataset/modis-10A2-061#overview
    MOD10A1F: "his global Level-3 data set (MOD10A1F) provides daily cloud-free snow cover derived from the MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid data set (MOD10A1). Grid cells in MOD10A1 which are obscured by cloud cover are filled by retaining clear-sky views of the surface from previous days. A separate parameter is provided which tracks the number of days in each cell since the last clear-sky observation. Each data granule contains a 10° x 10° tile projected to the 500 m sinusoidal grid." https://nsidc.org/data/mod10a1f/versions/61


    Attributes:
        bbox_input (geopandas.GeoDataFrame or tuple or Shapely Geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
        start_date (str): The start date for the data in the format 'YYYY-MM-DD'. Default is '2000-01-01'.
        end_date (str): The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
        data_product (str): The MODIS data product to retrieve. Can choose between 'MOD10A1F', 'MOD10A1', or 'MOD10A2'. Default is 'MOD10A1F'.
        bands (list): The bands to be used. Default is all bands.
        resolution (str): The resolution of the data. Defaults to native resolution.
        crs (str): The coordinate reference system. This should be a string like 'EPSG:4326'. Default is None.

    Methods:
        search_data(): Searches for MODIS snow data based on the specified parameters.
        get_data(): Retrieves the MODIS snow data based on the search results.
        get_binary_snow(): Calculates the binary snow map based on the retrieved data.

    Citation:
    MOD10A1F: Hall, D. K. and G. A. Riggs. (2020). MODIS/Terra CGF Snow Cover Daily L3 Global 500m SIN Grid, Version 61 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD10A1F.061. Date Accessed 03-19-2024.
    MOD10A1: Hall, D. K. and G. A. Riggs. (2021). MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid, Version 61 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD10A1.061. Date Accessed 03-28-2024.
    MOD10A2: Hall, D. K. and G. A. Riggs. (2021). MODIS/Terra Snow Cover 8-Day L3 Global 500m SIN Grid, Version 61 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD10A2.061. Date Accessed 03-28-2024.
    """

    def __init__(
        self,
        bbox_input=None,
        clip_to_bbox=True,
        start_date="2000-01-01",
        end_date=today,
        data_product="MOD10A2",
        bands=None,
        resolution=None,
        crs=None,
        vertical_tile=None,
        horizontal_tile=None,
        mute=False,
    ):

        if mute:
            blockPrint()
            
        self.bbox_input = bbox_input
        self.bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
        self.clip_to_bbox = clip_to_bbox
        self.start_date = start_date
        self.end_date = end_date
        self.data_product = data_product
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.vertical_tile = vertical_tile
        self.horizontal_tile = horizontal_tile


        self.search_data()
        self.get_data()
        
        if mute:
            enablePrint()

    def search_data(self):

        if self.data_product == "MOD10A1" or self.data_product == "MOD10A2":
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )

            if self.bbox_input is not None:
                search = catalog.search(
                    collections=[f"modis-{self.data_product[3:]}-061"],
                    bbox=self.bbox_gdf.total_bounds,
                    datetime=(self.start_date, self.end_date),
                )

            else:
                search = catalog.search(
                    collections=[f"modis-{self.data_product[3:]}-061"],
                    datetime=(self.start_date, self.end_date),
                    query={
                        "modis:vertical-tile": {"eq": self.vertical_tile},
                        "modis:horizontal-tile": {"eq": self.horizontal_tile},
                    },
                )

        elif self.data_product == "MOD10A1F":
            search = earthaccess.search_data(
                short_name="MOD10A1F",
                cloud_hosted=False,
                bounding_box=tuple(self.bbox_gdf.total_bounds),
                temporal=(self.start_date, self.end_date),
            )

        else:
            raise ValueError(
                "Data product not recognized. Please choose 'MOD10A1', 'MOD10A2', or 'MOD10A1F'."
            )

        self.search = search

    def get_data(self):

        if self.data_product == "MOD10A1" or self.data_product == "MOD10A2":

            load_params = {
                "items": self.search.item_collection(),
                "chunks": {"time": 1, "x": 512, "y": 512},
            }
            if self.clip_to_bbox:
                load_params["bbox"] = self.bbox_gdf.total_bounds
            if self.bands:
                load_params["bands"] = self.bands
            if self.crs:
                load_params["crs"] = self.crs
            if self.resolution:
                load_params["resolution"] = self.resolution

            modis_snow = odc.stac.load(**load_params)

        elif self.data_product == "MOD10A1F":
            # files = earthaccess.open(results) # doesn't seem to work for .hdf files...
            # https://github.com/nsidc/earthaccess/blob/main/docs/tutorials/file-access.ipynb
            # https://github.com/nsidc/earthaccess/tree/main
            # https://earthaccess.readthedocs.io/en/latest/tutorials/emit-earthaccess/
            # https://nbviewer.org/urls/gist.githubusercontent.com/scottyhq/790bf19c7811b5c6243ce37aae252ca1/raw/e2632e928647fd91c797e4a23116d2ac3ff62372/0-load-hdf5.ipynb
            # https://docs.dask.org/en/latest/array-creation.html#concatenation-and-stacking
            # https://matthewrocklin.com/blog/work/2018/02/06/hdf-in-the-cloud

            # guess we'll download instead
            temp_download_fp = "/tmp/local_folder"  # do these auto delete, or should i delete when opened explicitly? shutil.rmtree(temp_download_fp)

            files = earthaccess.download(
                self.search, temp_download_fp
            )  # can i suppress the print output? https://earthaccess.readthedocs.io/en/latest/user-reference/api/api/

            xmin, ymin, xmax, ymax = self.bbox_gdf.total_bounds

            if self.clip_to_bbox:
                modis_snow = xr.concat(
                    [
                        rxr.open_rasterio(
                            file, variable="CGF_NDSI_Snow_Cover", chunks={}
                        )["CGF_NDSI_Snow_Cover"]
                        .squeeze()
                        .rio.clip_box(xmin, ymin, xmax, ymax, crs="EPSG:4326")
                        .assign_coords(
                            time=pd.to_datetime(
                                rxr.open_rasterio(
                                    file, variable="CGF_NDSI_Snow_Cover", chunks={}
                                )
                                .squeeze()
                                .attrs["RANGEBEGINNINGDATE"]
                            )
                        )
                        .drop_vars("band")
                        for file in files
                    ],
                    dim="time",
                )

            else:
                modis_snow = xr.concat(
                    [
                        rxr.open_rasterio(
                            file, variable="CGF_NDSI_Snow_Cover", chunks={}
                        )["CGF_NDSI_Snow_Cover"]
                        .squeeze()
                        .assign_coords(
                            time=pd.to_datetime(
                                rxr.open_rasterio(
                                    file, variable="CGF_NDSI_Snow_Cover", chunks={}
                                )
                                .squeeze()
                                .attrs["RANGEBEGINNINGDATE"]
                            )
                        )
                        .drop_vars("band")
                        for file in files
                    ],
                    dim="time",
                )

        else:
            raise ValueError(
                "Data product not recognized. Please choose 'MOD10A1', 'MOD10A2', or 'MOD10A1F'."
            )

        self.data = modis_snow

        if self.data_product == "MOD10A2":
            self.data.attrs["class_info"] = {
                0: {"name": "missing data", "color": "#006400"},
                1: {"name": "no decision", "color": "#FFBB22"},
                11: {"name": "night", "color": "#FFFF4C"},
                25: {"name": "no snow", "color": "#F096FF"},
                37: {"name": "lake", "color": "#FA0000"},
                39: {"name": "ocean / sparse vegetation", "color": "#B4B4B4"},
                50: {"name": "cloud", "color": "#F0F0F0"},
                100: {"name": "lake ice", "color": "#0064C8"},
                200: {"name": "snow", "color": "#0096A0"},
                254: {"name": "detector saturated", "color": "#00CF75"},
                255: {"name": "fill", "color": "#FAE6A0"},
            }

        print("Data retrieved. Access with the .data attribute.")

    def get_binary_snow(self):

        if self.data_product == "MOD10A2":
            self.binary_snow = xr.where(self.data["Maximum_Snow_Extent"] == 200, 1, 0)
            print("Binary snow map calculated. Access with the .binary_snow attribute.")
        else:
            print("This method is only available for the MOD10A2 product.")






# palsar2
# ic = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR').filterDate('2020-10-05', '2021-03-31')
# ds = xarray.open_dataset(ic, geometry=bbox_ee,engine='ee')