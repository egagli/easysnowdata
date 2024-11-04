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
import matplotlib
import matplotlib.pyplot as plt
import skimage

import rasterio as rio

rio_env = rio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
    CPL_VSIL_CURL_USE_HEAD="FALSE",
    GDAL_HTTP_NETRC="TRUE",
    GDAL_HTTP_COOKIEFILE=os.path.expanduser("~/cookies.txt"),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser("~/cookies.txt"),
)
rio_env.__enter__()

xr.set_options(keep_attrs=True)

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


def get_forest_cover_fraction(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, mask_nodata: bool = False,
) -> xr.DataArray:
    """
    Fetches ~100m forest cover fraction data for a given bounding box.

    Description:
    The data is obtained from the Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe dataset.
    The specific layer used is the Tree-CoverFraction-layer, which provides the fractional cover (%) for the forest class.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=255, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=255)

    Returns
    -------
    xarray.DataArray
        Forest cover fraction DataArray.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from easysnowdata import remote_sensing
    >>> 
    >>> # Define a bounding box for an area of interest
    >>> bbox = (-122.5, 47.0, -121.5, 48.0)
    >>> 
    >>> # Fetch forest cover fraction data
    >>> forest_cover = remote_sensing.get_forest_cover_fraction(bbox)
    >>> 
    >>> # Plot the data using the example plot function
    >>> f, ax = forest_cover.attrs['example_plot'](forest_cover)

    Notes
    -----
    Data citation:
    Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020).
    Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050
    """

    def plot_forest_cover(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        cmap = matplotlib.colormaps.get_cmap('Greens').copy()
        cmap.set_over('white')  # Set values over 100 (i.e., 255) to white

        im = self.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=100, add_colorbar=False)
        
        cbar = plt.colorbar(im, ax=ax, extend='max')
        cbar.set_label('Forest Cover Fraction (%)')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Copernicus Global Land Service Forest Cover Fraction\nLand Cover 100m: collection 3: epoch 2019")
        f.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
        f.dpi = 300

        return f, ax

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    fcf_da = rxr.open_rasterio(
        "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        chunks=True,
        mask_and_scale=mask_nodata,
    )

    fcf_da = fcf_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs).squeeze()



    fcf_da.attrs['example_plot'] = plot_forest_cover
    fcf_da.attrs['data_citation'] = "Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050"

    return fcf_da


def get_seasonal_snow_classification(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, mask_nodata: bool = False,
) -> xr.DataArray:
    """
    Fetches 10arcsec (~300m) Sturm & Liston 2021 seasonal snow classification data for a given bounding box.

    Description:
    This dataset consists of global, seasonal snow classifications determined from air temperature,
    precipitation, and wind speed climatologies. This is the 10 arcsec (~300m) product in EPSG:4326.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=9, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=9)

    Returns
    -------
    xarray.DataArray
        Seasonal snow class DataArray with class information in attributes.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import easysnowdata
    >>> 
    >>> # Define a bounding box for an area of interest
    >>> bbox = (-120.0, 40.0, -118.0, 42.0)
    >>> 
    >>> # Fetch seasonal snow classification data
    >>> snow_classification_da = easysnowdata.remote_sensing.get_seasonal_snow_classification(bbox)
    >>> 
    >>> # Plot the data using the example plot function
    >>> f,ax = snow_classification_da.attrs['example_plot'](snow_classification_da)

    Notes
    -----
    Data citation:
    Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set].
    Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0. Date Accessed 03-06-2024.
    """

    def get_class_info():
        classes = {
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
        return classes

    def get_class_cmap(classes):
        cmap = plt.cm.colors.ListedColormap(
            [classes[key]["color"] for key in classes.keys()]
        )
        return cmap
    
    def plot_classes(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        class_values = sorted(list(self.attrs["class_info"].keys()))
        bounds = [
            (class_values[i] + class_values[i + 1]) / 2 for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)
        #ax.set_aspect("equal")


        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black")
            )
            class_names.append(class_info["name"])

        legend_kwargs = legend_kwargs or {}
        default_legend_kwargs = {
            "bbox_to_anchor": (0.5, -0.1),
            "loc": "upper center",
            "ncol": len(class_names) // 3,
            "frameon": False,
            "handlelength": 3.5,
            "handleheight": 5,
        }
        legend_kwargs = {**default_legend_kwargs, **legend_kwargs}

        ax.legend(legend_handles, class_names, **legend_kwargs)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Seasonal snow classification\nfrom Sturm & Liston 2021")
        f.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
        f.dpi = 300

        return f, ax
    
    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    snow_classification_da = rxr.open_rasterio(
        "https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif",
        chunks=True,
        mask_and_scale=mask_nodata,
    )
    snow_classification_da = (
        snow_classification_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs).squeeze()
    )

    if mask_nodata:
        snow_classification_da.rio.write_nodata(9, encoded=True, inplace=True)
    else:
        snow_classification_da.rio.set_nodata(9, inplace=True)

    snow_classification_da.attrs["class_info"] = get_class_info()
    snow_classification_da.attrs["cmap"] = get_class_cmap(snow_classification_da.attrs["class_info"])
    snow_classification_da.attrs['data_citation'] = "Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0. Date Accessed 03-06-2024."
    
    snow_classification_da.attrs['example_plot'] = plot_classes

    return snow_classification_da


def get_seasonal_mountain_snow_mask(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, data_product: str = "mountain_snow", mask_nodata: bool = False,
) -> xr.DataArray:
    """
    Fetches ~1km static global seasonal (mountain snow / snow) mask for a given bounding box.

    Description:
    Seasonal Mountain Snow (SMS) mask derived from MODIS MOD10A2 snow cover extent and GTOPO30 digital elevation model
    produced at 30 arcsecond spatial resolution.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    data_product : str, optional
        Data product to fetch. Choose from 'snow' or 'mountain_snow'. Default is 'mountain_snow'.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=255, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=255)

    Returns
    -------
    xarray.DataArray
        Mountain snow DataArray with class information in attributes.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import easysnowdata
    >>> 
    >>> # Define a bounding box for a mountainous area
    >>> bbox = (-106.0, 39.0, -105.0, 40.0)
    >>> 
    >>> # Fetch mountain snow mask data
    >>> mountain_snow_da = easysnowdata.remote_sensing.get_seasonal_mountain_snow_mask(bbox)
    >>> 
    >>> # Plot the data using the example plot function
    >>> f, ax = mountain_snow_da.attrs['example_plot'](mountain_snow_da)

    Notes
    -----
    Data citation:
    Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019).
    Global Seasonal Mountain Snow Mask from MODIS MOD10A2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2626737
    """

    def get_class_info(data_product):
        if data_product == "snow":
            classes = {
                0: {"name": "Little-to-no snow", "color": "#030303"},
                1: {"name": "Indeterminate due to clouds", "color": "#755F4A"},
                2: {"name": "Ephemeral snow", "color": "#792B8E"},
                3: {"name": "Seasonal snow", "color": "#679ACF"},
                255: {"name": "Fill", "color": "#ffffff"},
            }
        elif data_product == "mountain_snow":
            classes = {
                0: {"name": "Mountains with little-to-no snow", "color": "#030303"},
                1: {"name": "Indeterminate due to clouds", "color": "#755F4A"},
                2: {"name": "Mountains with ephemeral snow", "color": "#792B8E"},
                3: {"name": "Mountains with seasonal snow", "color": "#679ACF"},
                255: {"name": "Fill", "color": "#ffffff"},
            }
        else:
            raise ValueError('Invalid data_product. Choose from "snow" or "mountain_snow".')
        return classes

    def get_class_cmap(classes):
        cmap = plt.cm.colors.ListedColormap(
            [classes[key]["color"] for key in classes.keys()]
        )
        return cmap
    
    def plot_classes(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        class_values = sorted(list(self.attrs["class_info"].keys()))
        bounds = [
            (class_values[i] + class_values[i + 1]) / 2 for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)
        #ax.set_aspect("equal")

        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black")
            )
            class_names.append(class_info["name"])

        legend_kwargs = legend_kwargs or {}
        default_legend_kwargs = {
            "bbox_to_anchor": (0.5, -0.1),
            "loc": "upper center",
            "ncol": len(class_names) // 2,
            "frameon": False,
            "handlelength": 3.5,
            "handleheight": 5,
        }
        legend_kwargs = {**default_legend_kwargs, **legend_kwargs}

        ax.legend(legend_handles, class_names, **legend_kwargs)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Global seasonal {'mountain ' if data_product == 'mountain_snow' else ''}snow mask\nfrom Wrzesien et al 2019")
        f.tight_layout(pad=5.5, w_pad=5.5, h_pad=1.5)
        f.dpi = 300

        return f, ax

    print(f'This function takes a moment, getting zipped file from zenodo...')
    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    url = f"zip+https://zenodo.org/records/2626737/files/MODIS_{'mtnsnow' if data_product == 'mountain_snow' else 'snow'}_classes.zip!/MODIS_{'mtnsnow' if data_product == 'mountain_snow' else 'snow'}_classes.tif"

    mountain_snow_da = rxr.open_rasterio(
        url,
        chunks=True,
        mask_and_scale=mask_nodata,
    ).rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs).squeeze()

    # looks like the creators accidently set no data to 256 and 265 instead of 255, therefore unmasked the data is of type uint32 :(
    # attempt to fix this by setting all invalid values to 255, then converting types
    mask = mountain_snow_da > 3
    mountain_snow_da = mountain_snow_da.where(~mask, 255)

    if mask_nodata:
        mountain_snow_da = mountain_snow_da.astype("float32").rio.write_nodata(255, encoded=True)
    else:
        mountain_snow_da = mountain_snow_da.astype("uint8").rio.set_nodata(255)


    mountain_snow_da.attrs["class_info"] = get_class_info(data_product)
    mountain_snow_da.attrs["cmap"] = get_class_cmap(mountain_snow_da.attrs["class_info"])
    mountain_snow_da.attrs['data_citation'] = "Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019). Global Seasonal Mountain Snow Mask from MODIS MOD10A2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2626737"
    
    mountain_snow_da.attrs['example_plot'] = plot_classes

    return mountain_snow_da


def get_esa_worldcover(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    version: str = "v200", mask_nodata: bool = False,
) -> xr.DataArray:
    """
    Fetches 10m ESA WorldCover global land cover data (2020 v100 or 2021 v200) for a given bounding box.

    Description:
    The discrete classification maps provide 11 classes defined using the Land Cover Classification System (LCCS)
    developed by the United Nations (UN) Food and Agriculture Organization (FAO).

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    version : str, optional
        Version of the WorldCover data. The two versions are v100 (2020) and v200 (2021). Default is 'v200'.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=0, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=0)

    Returns
    -------
    xarray.DataArray
        WorldCover DataArray with class information in attributes.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import easysnowdata
    >>> 
    >>> # Define a bounding box for Mount Rainier
    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> 
    >>> # Fetch WorldCover data for the area
    >>> worldcover_da = easysnowdata.remote_sensing.get_esa_worldcover(bbox)
    >>> 
    >>> # Plot the data using the example plot function
    >>> f, ax = worldcover_da.attrs['example_plot'](worldcover_da)

    Notes
    -----
    Data citation:
    Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A.,
    Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin,
    Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936.
    """

    def get_class_info():
        classes = {
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
        return classes

    def get_class_cmap(classes):
        cmap = plt.cm.colors.ListedColormap(
            [classes[key]["color"] for key in classes.keys()]
        )
        return cmap
    
    def plot_classes(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        class_values = sorted(list(self.attrs["class_info"].keys()))
        bounds = [
            (class_values[i] + class_values[i + 1]) / 2 for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)
        #ax.set_aspect("equal")

        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black")
            )
            class_names.append(class_info["name"])

        legend_kwargs = legend_kwargs or {}
        default_legend_kwargs = {
            "bbox_to_anchor": (0.5, -0.1),
            "loc": "upper center",
            "ncol": len(class_names) // 3,
            "frameon": False,
            "handlelength": 3.5,
            "handleheight": 5,
        }
        legend_kwargs = {**default_legend_kwargs, **legend_kwargs}

        ax.legend(legend_handles, class_names, **legend_kwargs)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"ESA WorldCover\n{version} ({version_year})")
        f.tight_layout(pad=5.5, w_pad=5.5, h_pad=1.5)
        f.dpi = 300

        return f, ax

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

    if mask_nodata:
        worldcover_da = worldcover_da.where(worldcover_da>0)
        worldcover_da.rio.write_nodata(0, encoded=True, inplace=True)

    worldcover_da.attrs["class_info"] = get_class_info()
    worldcover_da.attrs["cmap"] = get_class_cmap(worldcover_da.attrs["class_info"])
    worldcover_da.attrs['data_citation'] = "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936."
    
    worldcover_da.attrs['example_plot'] = plot_classes

    return worldcover_da


def get_nlcd_landcover(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, 
             layer: str = 'landcover',
             initialize_ee: bool = True) -> xr.DataArray:
    """
    Fetches National Land Cover Database (NLCD) data for a given bounding box.

    Description:
    The National Land Cover Database (NLCD) provides nationwide data on land cover and land cover change 
    at a 30m resolution. The dataset includes various layers such as land cover classification, 
    impervious surfaces, and urban intensity. Projection is an albers equal area conic projection.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    layer : str, optional
        The NLCD layer to retrieve. Options are:
        - 'landcover'
        - 'impervious'
        - 'impervious_descriptor'
        - 'science_products_land_cover_change_count'
        - 'science_products_land_cover_change_first_disturbance_date'
        - 'science_products_land_cover_change_index'
        - 'science_products_land_cover_science_product'
        - 'science_products_forest_disturbance_date'
        Default is 'landcover'.
    initialize_ee : bool, optional
        Whether to initialize Earth Engine. Default is True.

    Returns
    -------
    xarray.DataArray
        NLCD DataArray for the specified region and layer.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import easysnowdata
    >>> 
    >>> # Define a bounding box for an area of interest
    >>> bbox = (-122.5, 47.0, -121.5, 48.0)
    >>> 
    >>> # Fetch NLCD land cover data
    >>> nlcd_landcover_da = easysnowdata.remote_sensing.get_nlcd_landcover(bbox, layer='landcover')
    >>> 
    >>> # Plot the data
    >>> nlcd_landcover_da.attrs['example_plot'](nlcd_landcover_da)


    Notes
    -----
    - NLCD data is only available for the contiguous United States
    - The latest version (2021) includes data from 2001-2021
    - Resolution is 30 meters

    Data citation:
    Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, doi:10.5066/P9JZ7AO3
    """
    # Initialize Earth Engine with high-volume endpoint
    if initialize_ee:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    else:
        print("Initialization turned off. If you haven't already, please sign in to Google Earth Engine by running:\n\nimport ee\nee.Authenticate()\nee.Initialize()\n\n")

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    image_collection = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
    image = image_collection.first()
    
    projection = image.select(0).projection()
    
    ds = xr.open_dataset(
        image_collection, 
        engine='ee', 
        geometry=tuple(bbox_gdf.total_bounds), 
        projection=projection,
        chunks={},
    ).squeeze().transpose().rename({'X':'x','Y':'y'}).rio.set_spatial_dims(x_dim='x', y_dim='y').astype('uint8')
    
    nlcd_da = ds[layer]

    # would be nice for them to come in as ints
    # https://github.com/google/Xee/issues/86
    # https://github.com/google/Xee/issues/146
    
    
    def get_class_info():
        info = image.getInfo()['properties']
        
        if layer == 'landcover':
            return {
                value: {
                    "name": name.split(':')[0],
                    "color": f"#{palette}"
                } for value, name, palette in zip(
                    info['landcover_class_values'],
                    info['landcover_class_names'],
                    info['landcover_class_palette']
                )
            }
        elif layer == 'impervious':
            return None  
        elif layer == 'impervious_descriptor':
            return {
                value: {
                    "name": name.split('.')[0],
                    "color": f"#{palette}"
                } for value, name, palette in zip(
                    info['impervious_descriptor_class_values'],
                    info['impervious_descriptor_class_names'],
                    info['impervious_descriptor_class_palette']
                )
            }
        elif layer.startswith('science_products'):
            return {
                value: {
                    "name": name,
                    "color": f"#{palette}"
                } for value, name, palette in zip(
                    info[f'{layer}_class_values'],
                    info[f'{layer}_class_names'],
                    info[f'{layer}_class_palette']
                )
            }

    def get_class_cmap(classes):
        if classes is None: 
            return plt.cm.YlOrRd
        return plt.cm.colors.ListedColormap([classes[key]["color"] for key in classes.keys()])

    def plot_classes(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        if self.name != 'impervious':
            class_values = sorted(list(self.attrs["class_info"].keys()))
            bounds = [(class_values[i] + class_values[i + 1]) / 2 for i in range(len(class_values) - 1)]
            bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
            norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

            im = self.plot.imshow(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)

            legend_handles = []
            class_names = []
            for class_value, class_info in self.attrs["class_info"].items():
                legend_handles.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black")
                )
                class_names.append(class_info["name"])

            legend_kwargs = legend_kwargs or {}
            default_legend_kwargs = {
                "bbox_to_anchor": (0.5, -0.1),
                "loc": "upper center",
                "ncols": 4,
                "frameon": False,
                "handlelength": 3.5,
                "handleheight": 5,
            }
            legend_kwargs = {**default_legend_kwargs, **legend_kwargs}
            ax.legend(legend_handles, class_names, **legend_kwargs)

        else: 
            im = self.plot.imshow(ax=ax, cmap=self.attrs["cmap"], add_colorbar=False)
            f.colorbar(im, ax=ax, label='Percent impervious surface [%]')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        #ax.axis('equal')
        ax.set_title(f"NLCD {self.name.title()} (2021)")
        #f.tight_layout(pad=0, w_pad=0, h_pad=0)
        f.dpi = 300

        return f, ax

    class_info = get_class_info()
    nlcd_da.attrs["class_info"] = class_info
    nlcd_da.attrs["cmap"] = get_class_cmap(class_info)
    nlcd_da.attrs["example_plot"] = plot_classes

    nlcd_da.attrs['data_citation'] = "Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, doi:10.5066/P9JZ7AO3"


    return nlcd_da

class Sentinel2:
    """
    A class to handle Sentinel-2 satellite data.

    This class provides functionality to search, retrieve, and process Sentinel-2 satellite imagery.
    It supports various data operations including masking, scaling, and calculation of spectral indices.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    start_date : str, optional
        The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
    end_date : str, optional
        The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
    catalog_choice : str, optional
        The catalog choice for the data. Can choose between 'planetarycomputer' and 'earthsearch'. Default is 'planetarycomputer'.
    bands : list, optional
        The bands to be used. Default is all bands. Must include SCL for data masking.
    resolution : str, optional
        The resolution of the data. Defaults to native resolution, 10m.
    crs : str, optional
        The coordinate reference system. This should be a string like 'EPSG:4326'. Default CRS is UTM zone estimated from bounding box.
    remove_nodata : bool, optional
        Whether to remove no data values. Default is True.
    harmonize_to_old : bool, optional
        Whether to harmonize new data to the old baseline. Default is True.
    scale_data : bool, optional
        Whether to scale the data. Default is True.
    groupby : str, optional
        The groupby parameter for the data. Default is "solar_day".

    Attributes
    ----------
    data : xarray.Dataset
        The loaded Sentinel-2 data.
    metadata : geopandas.GeoDataFrame
        Metadata for the retrieved Sentinel-2 scenes.
    rgb : xarray.DataArray
        RGB composite of the Sentinel-2 data.
    ndvi : xarray.DataArray
        Normalized Difference Vegetation Index (NDVI) calculated from the data.
    ndsi : xarray.DataArray
        Normalized Difference Snow Index (NDSI) calculated from the data.
    ndwi : xarray.DataArray
        Normalized Difference Water Index (NDWI) calculated from the data.
    evi : xarray.DataArray
        Enhanced Vegetation Index (EVI) calculated from the data.
    ndbi : xarray.DataArray
        Normalized Difference Built-up Index (NDBI) calculated from the data.

    Methods
    -------
    search_data()
        Searches for Sentinel-2 data based on the specified parameters.
    get_data()
        Retrieves the Sentinel-2 data based on the search results.
    get_metadata()
        Retrieves metadata for the Sentinel-2 scenes.
    remove_nodata_inplace()
        Removes no data values from the data.
    mask_data()
        Masks the data based on the Scene Classification Layer (SCL).
    harmonize_to_old_inplace()
        Harmonizes new Sentinel-2 data to the old baseline.
    scale_data_inplace()
        Scales the data to reflectance values.
    get_rgb()
        Retrieves the RGB composite of the data.
    get_ndvi()
        Calculates the Normalized Difference Vegetation Index (NDVI).
    get_ndsi()
        Calculates the Normalized Difference Snow Index (NDSI).
    get_ndwi()
        Calculates the Normalized Difference Water Index (NDWI).
    get_evi()
        Calculates the Enhanced Vegetation Index (EVI).
    get_ndbi()
        Calculates the Normalized Difference Built-up Index (NDBI).
    """

    def __init__(
        self,
        bbox_input,
        start_date="2014-01-01",
        end_date=today,
        catalog_choice="planetarycomputer",
        collection= "sentinel-2-l2a", # could also choose "sentinel-2-c1-l2a" once published to https://github.com/Element84/earth-search
        bands=None,
        resolution=None,
        crs=None,
        remove_nodata=True,
        harmonize_to_old=None,
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
        self.collection = collection
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

        self.scl_cmap = plt.cm.colors.ListedColormap(
            [info["color"] for info in self.scl_class_info.values()]
        )

        # Initialize the data attributes
        self.search = None
        self.data = None
        self.metadata = None

        self.rgb = None
        self.rgba = None
        self.rgb_clahe = None
        self.rgb_percentile = None
        self.ndvi = None
        self.ndsi = None
        self.ndwi = None
        self.evi = None
        self.ndbi = None

        self.search_data()
        self.get_data()

        if self.remove_nodata:
            self.remove_nodata_inplace()

        if self.harmonize_to_old is None:
            if self.catalog_choice == "planetarycomputer":
                self.harmonize_to_old = True
            else:
                if self.collection == "sentinel-2-c1-l2a":
                    self.harmonize_to_old = True
                elif self.collection == "sentinel-2-l2a":
                    self.harmonize_to_old = False
                    print(f"Since {self.collection} on {self.catalog_choice} is used, harmonization step is not needed.")
                else:
                    raise ValueError(f"Unknown collection: {self.collection}")
                
        if self.harmonize_to_old:
            self.harmonize_to_old_inplace()

        if self.scale_data:
            self.scale_data_inplace()

        self.get_metadata()



        # Add the plot_scl method as an attribute to the SCL data variable
        if 'scl' in self.data.data_vars:
            self.data.scl.attrs['example_plot'] = self.plot_scl
            self.data.scl.attrs['class_info'] = self.scl_class_info
            self.data.scl.attrs['cmap'] = self.scl_cmap

    def plot_scl(self, scl_data, ax=None, figsize=None, col_wrap=5, legend_kwargs=None):
        
        if figsize is None:
            figsize = (8, 10) if scl_data.time.size == 1 else (12, 7)

        class_values = sorted(list(self.scl_class_info.keys()))
        bounds = [(class_values[i] + class_values[i + 1]) / 2 for i in range(len(class_values) - 1)]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.scl_cmap.N)

        if scl_data.time.size == 1:
            # Single image plot
            if ax is None:
                f, ax = plt.subplots(figsize=figsize)
            else:
                f = ax.get_figure()

            im = scl_data.plot.imshow(ax=ax, cmap=self.scl_cmap, norm=norm, add_colorbar=False)
            ax.set_aspect("equal")

            local_time = pd.to_datetime(scl_data.time.values).tz_localize('UTC').tz_convert('America/Los_Angeles')
            ax.set_title(f"Sentinel-2 Scene Classification Layer (SCL)\n{local_time.strftime('%B %d, %Y')}\n{local_time.strftime('%I:%M%p')}")

        else:
            # Multiple images plot
            f = scl_data.plot.imshow(
                col='time',
                col_wrap=col_wrap,
                cmap=self.scl_cmap,
                norm=matplotlib.colors.BoundaryNorm(bounds, self.scl_cmap.N),
                add_colorbar=False,
                #figsize=figsize
            )

            for ax, time in zip(f.axs.flat, scl_data.time.values):
                local_time = pd.to_datetime(time).tz_localize('UTC').tz_convert('America/Los_Angeles')
                ax.set_title(f'{local_time.strftime("%B %d, %Y")}\n{local_time.strftime("%I:%M%p")}')
                ax.axis('off')
                ax.set_aspect('equal')

            f.fig.suptitle('Sentinel-2 SCL time series', fontsize=16, y=1.02)

        # Add legend
        legend_handles = []
        class_names = []
        for class_value, class_info in self.scl_class_info.items():
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black")
            )
            class_names.append(class_info["name"])

        legend_kwargs = legend_kwargs or {}

        if scl_data.time.size == 1:
            default_legend_kwargs = {
                "bbox_to_anchor": (0.5, -0.1),
                "loc": "upper center",
                "ncol": len(class_names) // 4,
                "frameon": False,
                "handlelength": 3.5,
                "handleheight": 5,
            }
        else:
            default_legend_kwargs = {
                "bbox_to_anchor": (0.5, -0.1),
                "loc": "upper center",
                "ncol": len(class_names) // 4,
                "frameon": False,
                "handlelength": 5,
                "handleheight": 6,
                "fontsize": 16,
            }

        legend_kwargs = {**default_legend_kwargs, **legend_kwargs}

        if scl_data.time.size == 1:
            ax.legend(legend_handles, class_names, **legend_kwargs)
            f.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
            f.dpi = 300
        else:
            f.fig.legend(legend_handles, class_names, **legend_kwargs)
            f.fig.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
            f.fig.dpi = 300

        return f, ax if scl_data.time.size == 1 else f

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
            collections=[self.collection],
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
        data_removed = False
        for band in self.data.data_vars:
            nodata_value = None
            nodata_value = self.data[band].attrs.get("nodata")
            if nodata_value is not None:
                #print(f"Removing nodata {nodata_value} values for band {band}...")
                self.data[band] = self.data[band].where(self.data[band] != nodata_value)
                data_removed = True
        if data_removed:
            print(f"Nodata values removed from the data. In doing so, all bands converted to float32. To turn this behavior off, set remove_nodata=False.")
        else:
            print(f"Tried to remove nodata values and set them to nans, but no nodata values found in the data.")

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
            f"Data acquired after January 25th, 2022 harmonized to old baseline. To override this behavior, set harmonize_to_old=False."
        )

    def scale_data_inplace(self):
        """
        The method to scale the data.
        """
        for band in self.data.data_vars:
            scale_factor = self.data[band].attrs.get("scale")

            if scale_factor is None:
                scale_factor = next((info['scale'] for name, info in self.band_info.items() if info['name'] == band), None)

            scale_factor = int(scale_factor) if scale_factor == '1' else float(scale_factor)
            self.data[band] = self.data[band] * scale_factor

        print(
            f"Data scaled to float reflectance. To turn this behavior off, set scale_data=False."
        )

    def get_rgb(self, percentile_kwargs={'lower': 2, 'upper': 98}, clahe_kwargs={'clip_limit': 0.03, 'nbins': 256, 'kernel_size': None}):
        """
        Retrieve RGB data with optional percentile-based contrast stretching and CLAHE enhancement.

        This method calculates and stores three versions of RGB data: raw, percentile-stretched, and CLAHE-enhanced.

        Parameters
        ----------
        percentile_kwargs : dict, optional
            Parameters for percentile-based contrast stretching. Keys are:
            - 'lower': Lower percentile for contrast stretching (default: 2)
            - 'upper': Upper percentile for contrast stretching (default: 98)
        clahe_kwargs : dict, optional
            Parameters for CLAHE enhancement. Keys are:
            - 'clip_limit': Clipping limit for CLAHE (default: 0.03)
            - 'nbins': Number of bins for CLAHE histogram (default: 256)
            - 'kernel_size': Size of kernel for CLAHE (default: None)

        Returns
        -------
        None
            The method stores results in instance attributes.

        Notes
        -----
        Results are stored in the following attributes:
        - .rgb: Raw RGB data
        - .rgb_percentile: Percentile-stretched RGB data
        - .rgb_clahe: CLAHE-enhanced RGB data
        """

        rgba_da = self.data.odc.to_rgba(bands=('red','green','blue'),vmin=0, vmax=1.7)
        self.rgba = rgba_da

        rgb_da = rgba_da.isel(band=slice(0, 3))  #.where(self.data.scl>=0, other=255) if we want to make no data white
        self.rgb = rgb_da

        self.rgb_percentile = self.get_rgb_percentile(**percentile_kwargs)
        self.rgb_clahe = self.get_rgb_clahe(**clahe_kwargs)

        print(f"RGB data retrieved.\nAccess with the following attributes:\n.rgb for raw RGB,\n.rgba for RGBA,\n.rgb_percentile for percentile RGB,\n.rgb_clahe for CLAHE RGB.\nYou can pass in percentile_kwargs and clahe_kwargs to adjust RGB calculations, check documentation for options.")

    def get_rgb_percentile(self, **percentile_kwargs):
        """
        Apply percentile-based contrast stretching to the RGB bands of the Sentinel-2 data.

        This function creates a new DataArray with the contrast-stretched RGB bands.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for percentile calculation. Supported keys:
            - 'lower': Lower percentile for contrast stretching (default: 2)
            - 'upper': Upper percentile for contrast stretching (default: 98)

        Returns
        -------
        xarray.DataArray
            RGB data with percentile-based contrast stretching applied.

        Notes
        -----
        The function clips values to the range [0, 1] and masks areas where SCL < 0.
        """
        lower_percentile = percentile_kwargs.get('lower', 2)
        upper_percentile = percentile_kwargs.get('upper', 98)

        def stretch_percentile(da):
            p_low, p_high = np.nanpercentile(da.values, [lower_percentile, upper_percentile])
            return (da - p_low) / (p_high - p_low)

        rgb_da = self.rgb
        
        template = xr.zeros_like(rgb_da)
        rgb_percentile_da = xr.map_blocks(stretch_percentile, rgb_da, template=template)
        rgb_percentile_da = rgb_percentile_da.clip(0, 1).where(self.data.scl>=0)

        return rgb_percentile_da
    
    def get_rgb_clahe(self, **kwargs):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to RGB bands.

        This function creates a new DataArray with CLAHE applied to the RGB bands.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for CLAHE. Supported keys:
            - 'clip_limit': Clipping limit for CLAHE (default: 0.03)
            - 'nbins': Number of bins for CLAHE histogram (default: 256)
            - 'kernel_size': Size of kernel for CLAHE (default: None)

        Returns
        -------
        xarray.DataArray
            RGB data with CLAHE enhancement applied.

        Notes
        -----
        The function applies CLAHE to each band separately and masks areas where SCL < 0.
        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
        """

        # Custom wrapper to preserve xarray metadata
        def equalize_adapthist_da(da, **kwargs):
            # Apply the CLAHE function from skimage
            result = skimage.exposure.equalize_adapthist(da.values, **kwargs)
            #new_coords = {k: v for k, v in da.coords.items() if k != 'band' or len(v) == 3}

            # Convert the result back to a DataArray, preserving the original metadata
            return xr.DataArray(result, dims=da.dims, coords=da.coords, attrs=da.attrs)
        
        rgb_da = self.rgb
        
        #template = rgb_da.copy(data=np.empty_like(rgb_da).data)
        template = xr.zeros_like(rgb_da)
        rgb_clahe_da = xr.map_blocks(equalize_adapthist_da, rgb_da, template=template, kwargs=kwargs)
        rgb_clahe_da = rgb_clahe_da.where(self.data.scl>=0)

        return rgb_clahe_da

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

    This class provides functionality to search, retrieve, and process Sentinel-1 Radiometric Terrain Corrected (RTC) data.
    It supports various data operations including border noise removal and unit conversion.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    start_date : str, optional
        The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
    end_date : str, optional
        The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
    catalog_choice : str, optional
        The catalog choice for the data. Default is 'planetarycomputer'.
    bands : list, optional
        The bands to be used. Default is all bands.
    units : str, optional
        The units of the data. Can be 'dB' or 'linear power'. Default is 'dB'.
    resolution : str, optional
        The resolution of the data. Defaults to native resolution.
    crs : str, optional
        The coordinate reference system. Default is None.
    groupby : str, optional
        The groupby parameter for the data. Default is "sat:absolute_orbit".
    chunks : dict, optional
        The chunk size for dask arrays. Default is {}.
    remove_border_noise : bool, optional
        Whether to remove border noise from the data. Default is True.

    Attributes
    ----------
    data : xarray.Dataset
        The loaded Sentinel-1 data.
    metadata : geopandas.GeoDataFrame
        Metadata for the retrieved Sentinel-1 scenes.

    Methods
    -------
    search_data()
        Searches for Sentinel-1 data based on the specified parameters.
    get_data()
        Retrieves the Sentinel-1 data based on the search results.
    get_metadata()
        Retrieves metadata for the Sentinel-1 scenes.
    remove_border_noise()
        Removes border noise from the data.
    linear_to_db()
        Converts linear power units to decibels (dB).
    db_to_linear()
        Converts decibels (dB) to linear power units.
    add_orbit_info()
        Adds orbit information to the data as coordinates.
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
        remove_border_noise=True,
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
        self.remove_border_noise = remove_border_noise

        #if not self.geobox:
        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs is None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # if resolution == None:
        #     self.resolution = 10

        self.search = None
        self.data = None
        self.metadata = None

        self.search_data()
        self.get_data()
        self.get_metadata()
        if self.remove_border_noise:
            self.remove_bad_scenes_and_border_noise()
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
        # elif self.geobox:
        #     search = catalog.search(
        #         collections=["sentinel-1-rtc"],
        #         bbox=np.array(self.geobox.extent.boundingbox.to_crs('epsg:4326')),
        #         datetime=(self.start_date, self.end_date),
        #     )

        self.search = search
        print(f"Data searched. Access the returned seach with the .search attribute.")

    def get_data(self):
        """
        The method to get the data.
        """
        # Prepare the parameters for odc.stac.load
        load_params = {
            "items": self.search.items(),
            "nodata": -32768,
            "chunks": self.chunks,
            "groupby": self.groupby,
        }
        if self.bands:
            load_params["bands"] = self.bands
        load_params["crs"] = self.crs
        load_params["bbox"] = self.bbox_gdf.total_bounds
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

    # def remove_border_noise(self,threshold=0.001):
    #     """
    #     The method to remove border noise from the data.
    #     https://forum.step.esa.int/t/grd-border-noise-and-thermal-noise-removal-are-not-working-anymore-since-march-13-2018/9332
    #     https://www.mdpi.com/2072-4292/8/4/348
    #     https://forum.step.esa.int/t/nan-appears-at-the-edge-of-the-scene-after-applying-border-noise-removal-sentinel-1-grd/40627/2
    #     https://sentiwiki.copernicus.eu/__attachments/1673968/OI-MPC-OTH-MPC-0243%20-%20Sentinel-1%20masking%20no%20value%20pixels%20grd%20products%20note%202023%20-%202.2.pdf?inst-v=534578f3-fc04-48e9-bd69-3a45a681fe67#page=12.58
    #     https://ieeexplore.ieee.org/document/8255846
    #     https://www.mdpi.com/2504-3900/2/7/330
    #     """
    #     self.data.loc[dict(time=slice('2014-01-01','2018-03-14'))] = self.data.sel(time=slice('2014-01-01','2018-03-14')).where(lambda x: x > threshold)
    #     print(f"Border noise removed from the data.")

    def remove_bad_scenes_and_border_noise(self, threshold=0.001):
        cutoff_date = np.datetime64('2018-03-14')
        
        original_crs = self.data.rio.crs
        
        result = xr.where(
            self.data.time < cutoff_date,
            self.data.where(self.data > threshold),
            self.data.where(self.data > 0)
        )
        
        result.rio.write_crs(original_crs, inplace=True)
        
        self.data = result
        print(f"Falsely low scenes and border noise removed from the data.")

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
    A class to handle Harmonized Landsat Sentinel (HLS) satellite data.

    This class provides functionality to search, retrieve, and process HLS data, which combines
    data from Landsat and Sentinel-2 satellites. It supports various data operations including
    masking, scaling, and metadata retrieval.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    start_date : str, optional
        The start date for the data in the format 'YYYY-MM-DD'. Default is '2014-01-01'.
    end_date : str, optional
        The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
    bands : list, optional
        The bands to be used. Default is all bands.
    resolution : str, optional
        The resolution of the data. Defaults to native resolution.
    crs : str, optional
        The coordinate reference system. Default is 'utm'.
    remove_nodata : bool, optional
        Whether to remove no data values. Default is True.
    scale_data : bool, optional
        Whether to scale the data. Default is True.
    add_metadata : bool, optional
        Whether to add metadata to the data. Default is True.
    add_platform : bool, optional
        Whether to add platform information to the data. Default is True.
    groupby : str, optional
        The groupby parameter for the data. Default is "solar_day".

    Attributes
    ----------
    data : xarray.Dataset
        The loaded HLS data.
    metadata : geopandas.GeoDataFrame
        Metadata for the retrieved HLS scenes.
    rgb : xarray.DataArray
        RGB composite of the HLS data.
    ndvi : xarray.DataArray
        Normalized Difference Vegetation Index (NDVI) calculated from the data.

    Methods
    -------
    search_data()
        Searches for HLS data based on the specified parameters.
    get_data()
        Retrieves the HLS data based on the search results.
    get_metadata()
        Retrieves metadata for the HLS scenes.
    get_combined_metadata()
        Retrieves and combines metadata for both Landsat and Sentinel-2 scenes.
    remove_nodata_inplace()
        Removes no data values from the data.
    mask_data()
        Masks the data based on the Fmask quality layer.
    scale_data_inplace()
        Scales the data to reflectance values.
    add_platform_inplace()
        Adds platform information to the data as coordinates.
    get_rgb()
        Retrieves the RGB composite of the data.
    get_ndvi()
        Calculates the Normalized Difference Vegetation Index (NDVI).
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
        self.band_info = { # https://github.com/stac-extensions/eo#common-band-names
            "coastal": {
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
            "rededge071": {
                "landsat_band": "-",
                "sentinel_band": "B05",
                "description": "690-710 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "rededge075": {
                "landsat_band": "-",
                "sentinel_band": "B06",
                "description": "730-750 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "rededge078": {
                "landsat_band": "-",
                "sentinel_band": "B07",
                "description": "770-790 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "nir": {
                "landsat_band": "-",
                "sentinel_band": "B08",
                "description": "780-880 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "nir08": {
                "landsat_band": "B05",
                "sentinel_band": "B8A",
                "description": "850-880 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "swir16": {
                "landsat_band": "B06",
                "sentinel_band": "B11",
                "description": "1570-1650 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "swir22": {
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
            "lwir11": {
                "landsat_band": "B10",
                "sentinel_band": "-",
                "description": "10600-11190 nm",
                "data_type": "int16",
                "nodata": "-9999",
                "scale": "0.0001",
            },
            "lwir12": {
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
            collections=["HLSL30_2.0"],
            bbox=self.bbox_gdf.total_bounds,
            datetime=(self.start_date, self.end_date),
        )
        sentinel_search = catalog.search(
            collections=["HLSS30_2.0"],
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
            "stac_cfg": get_stac_cfg(sensor="HLSL30_2.0"),
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
            "stac_cfg": get_stac_cfg(sensor="HLSS30_2.0"),
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
        data_removed=False
        for band in self.data.data_vars:
            nodata_value = self.data[band].attrs.get("nodata")
            if nodata_value is not None:
                self.data[band] = self.data[band].where(self.data[band] != nodata_value)
                data_removed=True
        if data_removed:
            print(f"Nodata values removed from the data. In doing so, all bands converted to float32. To turn this behavior off, set remove_nodata=False.")
        else:
            print(f"Tried to remove nodata values and set them to nans, but no nodata values found in the data.")


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
        HLS_metadata = HLS_metadata.drop(
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
        self.data = self.data.apply(scale_var, keep_attrs=True)
        print(
            f"Data scaled to reflectance. Access with the .data attribute. To turn this behavior off, set scale_data=False."
        )

    # def get_rgb(self):
    #     """
    #     The method to get the RGB data.

    #     Returns:
    #         xarray.DataArray: The RGB data.
    #     """
    #     # Convert the red, green, and blue bands to an RGB DataArray
    #     rgb_da = self.data[["red", "green", "blue"]].to_dataarray(dim="band")
    #     self.rgb = rgb_da

    #     print(f"RGB data retrieved. Access with the .rgb attribute.")


    def get_rgb(self, percentile_kwargs={'lower': 2, 'upper': 98}, clahe_kwargs={'clip_limit': 0.03, 'nbins': 256, 'kernel_size': None}):
        """
        Retrieve RGB data with optional percentile-based contrast stretching and CLAHE enhancement.

        This method calculates and stores three versions of RGB data: raw, percentile-stretched, and CLAHE-enhanced.

        Parameters
        ----------
        percentile_kwargs : dict, optional
            Parameters for percentile-based contrast stretching. Keys are:
            - 'lower': Lower percentile for contrast stretching (default: 2)
            - 'upper': Upper percentile for contrast stretching (default: 98)
        clahe_kwargs : dict, optional
            Parameters for CLAHE enhancement. Keys are:
            - 'clip_limit': Clipping limit for CLAHE (default: 0.03)
            - 'nbins': Number of bins for CLAHE histogram (default: 256)
            - 'kernel_size': Size of kernel for CLAHE (default: None)

        Returns
        -------
        None
            The method stores results in instance attributes.

        Notes
        -----
        Results are stored in the following attributes:
        - .rgb: Raw RGB data
        - .rgb_percentile: Percentile-stretched RGB data
        - .rgb_clahe: CLAHE-enhanced RGB data
        """

        rgba_da = self.data.odc.to_rgba(bands=('red','green','blue'),vmin=-0.30, vmax=1.35)
        self.rgba = rgba_da

        rgb_da = rgba_da.isel(band=slice(0, 3)) # .where(self.data.scl>=0, other=255) if we want to make no data white
        self.rgb = rgb_da

        self.rgb_percentile = self.get_rgb_percentile(**percentile_kwargs)
        self.rgb_clahe = self.get_rgb_clahe(**clahe_kwargs)

        print(f"RGB data retrieved.\nAccess with the following attributes:\n.rgb for raw RGB,\n.rgba for RGBA,\n.rgb_percentile for percentile RGB,\n.rgb_clahe for CLAHE RGB.\nYou can pass in percentile_kwargs and clahe_kwargs to adjust RGB calculations, check documentation for options.")

    def get_rgb_percentile(self, **percentile_kwargs):
        """
        Apply percentile-based contrast stretching to the RGB bands of the Sentinel-2 data.

        This function creates a new DataArray with the contrast-stretched RGB bands.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for percentile calculation. Supported keys:
            - 'lower': Lower percentile for contrast stretching (default: 2)
            - 'upper': Upper percentile for contrast stretching (default: 98)

        Returns
        -------
        xarray.DataArray
            RGB data with percentile-based contrast stretching applied.

        Notes
        -----
        The function clips values to the range [0, 1] and masks areas where SCL < 0.
        """
        lower_percentile = percentile_kwargs.get('lower', 2)
        upper_percentile = percentile_kwargs.get('upper', 98)

        def stretch_percentile(da):
            p_low, p_high = np.nanpercentile(da.values, [lower_percentile, upper_percentile])
            return (da - p_low) / (p_high - p_low)

        rgb_da = self.rgb.where(self.rgba.isel(band=-1)==255)
        
        template = xr.zeros_like(rgb_da)
        rgb_percentile_da = xr.map_blocks(stretch_percentile, rgb_da, template=template)
        rgb_percentile_da = rgb_percentile_da.clip(0, 1)#.where(self.data.scl>=0)

        return rgb_percentile_da
    
    def get_rgb_clahe(self, **kwargs):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to RGB bands.

        This function creates a new DataArray with CLAHE applied to the RGB bands.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for CLAHE. Supported keys:
            - 'clip_limit': Clipping limit for CLAHE (default: 0.03)
            - 'nbins': Number of bins for CLAHE histogram (default: 256)
            - 'kernel_size': Size of kernel for CLAHE (default: None)

        Returns
        -------
        xarray.DataArray
            RGB data with CLAHE enhancement applied.

        Notes
        -----
        The function applies CLAHE to each band separately and masks areas where SCL < 0.
        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
        """

        # Custom wrapper to preserve xarray metadata
        def equalize_adapthist_da(da, **kwargs):
            # Apply the CLAHE function from skimage
            result = skimage.exposure.equalize_adapthist(da.values, **kwargs)
            #new_coords = {k: v for k, v in da.coords.items() if k != 'band' or len(v) == 3}

            # Convert the result back to a DataArray, preserving the original metadata
            return xr.DataArray(result, dims=da.dims, coords=da.coords, attrs=da.attrs)
        
        rgb_da = self.rgb
        
        #template = rgb_da.copy(data=np.empty_like(rgb_da).data)
        template = xr.zeros_like(rgb_da)
        rgb_clahe_da = xr.map_blocks(equalize_adapthist_da, rgb_da, template=template, kwargs=kwargs)
        rgb_clahe_da = rgb_clahe_da.where(self.rgba.isel(band=-1)==255)#.where(self.data.scl>=0)

        return rgb_clahe_da

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

    This class provides functionality to search, retrieve, and process MODIS snow cover data.
    It supports various MODIS snow products and allows for spatial and temporal subsetting.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry, optional
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    clip_to_bbox : bool, optional
        Whether to clip the data to the bounding box. Default is True.
    start_date : str, optional
        The start date for the data in the format 'YYYY-MM-DD'. Default is '2000-01-01'.
    end_date : str, optional
        The end date for the data in the format 'YYYY-MM-DD'. Default is today's date.
    data_product : str, optional
        The MODIS data product to retrieve. Can choose between 'MOD10A1F', 'MOD10A1', or 'MOD10A2'. Default is 'MOD10A2'.
    bands : list, optional
        The bands to be used. Default is all bands.
    resolution : str, optional
        The resolution of the data. Defaults to native resolution.
    crs : str, optional
        The coordinate reference system. Default is None.
    vertical_tile : int, optional
        The vertical tile number for MODIS data. Default is None.
    horizontal_tile : int, optional
        The horizontal tile number for MODIS data. Default is None.
    mute : bool, optional
        Whether to mute print outputs. Default is False.

    Attributes
    ----------
    data : xarray.Dataset
        The loaded MODIS snow data.
    binary_snow : xarray.DataArray
        Binary snow cover map derived from the data (only for MOD10A2 product).

    Methods
    -------
    search_data()
        Searches for MODIS snow data based on the specified parameters.
    get_data()
        Retrieves the MODIS snow data based on the search results.
    get_binary_snow()
        Calculates a binary snow cover map from the data (only for MOD10A2 product).

    Notes
    -----
    Available data products:
    MOD10A1: Daily snow cover, 500m resolution
    MOD10A2: 8-day maximum snow cover, 500m resolution
    MOD10A1F: Daily cloud-free snow cover (gap-filled), 500m resolution

    Data citations:
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


            if self.clip_to_bbox:
                modis_snow = xr.concat(
                    [
                        rxr.open_rasterio(
                            file, variable="CGF_NDSI_Snow_Cover", chunks={}
                        )["CGF_NDSI_Snow_Cover"]
                        .squeeze()
                        .rio.clip_box(*self.bbox_gdf.total_bounds,crs=self.bbox_gdf.crs)
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
            self.binary_snow = xr.where(self.data["Maximum_Snow_Extent"] == 200, 1, 0).rio.write_crs(self.data.rio.crs)
            print("Binary snow map calculated. Access with the .binary_snow attribute.")
        else:
            print("This method is only available for the MOD10A2 product.")






# palsar2
# ic = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR').filterDate('2020-10-05', '2021-03-31')
# ds = xarray.open_dataset(ic, geometry=bbox_ee,engine='ee')