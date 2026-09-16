"""Access remote sensing datasets for snow science applications.

Includes Sentinel-1, Sentinel-2, HLS, MODIS snow products, land cover,
snow classifications, and more.
"""

from __future__ import annotations

import logging

import ee
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import xarray as xr

from easysnowdata import auth, providers, temporal
from easysnowdata._deprecation import deprecated
from easysnowdata.utils import (
    _EARTHACCESS_SETUP_MSG,
    CredentialError,
    _has_earthaccess_credentials,
    convert_bbox_to_geodataframe,
    get_ee_grid_params,
    initialize_earthengine,
    requires_earthengine,
    suppress_stdout,
)

# No import-time side effects (design contract §2.9): GDAL options are applied
# inside the providers' context managers, xarray options are never set
# globally, and "today" is computed when a class is instantiated.

__all__ = [
    "authenticate_all",
    "get_forest_cover_fraction",
    "get_seasonal_snow_classification",
    "get_seasonal_mountain_snow_mask",
    "get_esa_worldcover",
    "get_nlcd_landcover",
    "Sentinel2",
    "Sentinel1",
    "HLS",
    "MODIS_snow",
]

_logger = logging.getLogger(__name__)


def __getattr__(name: str) -> str:
    # ``remote_sensing.today`` used to be a module constant frozen at import;
    # it is now computed on access so long-running sessions get the real date.
    if name == "today":
        return temporal.today()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def authenticate_all():
    """Interactively authenticate with all credentialed data providers.

    Runs the one-time credential setup for NASA EarthData and Google Earth
    Engine, then saves credentials locally so subsequent sessions require no
    further authentication.

    Providers that require no credentials (Planetary Computer, anonymous GCS)
    are skipped.

    Notes
    -----
    Call this function once before using any function that requires
    ``requires_earthengine`` or ``requires_earthaccess``.  Credentials are
    stored in ``~/.netrc`` (EarthData) and
    ``~/.config/earthengine/credentials`` (Earth Engine).
    """
    _logger.info("Starting interactive credential setup for all providers.")

    auth.login("earthdata", persist=True)
    _logger.info("NASA EarthData: done.")

    auth.login("earthengine")
    _logger.info("Google Earth Engine: done.")


def get_forest_cover_fraction(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    mask_nodata: bool = False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments passed to ``rioxarray.open_rasterio`` (e.g.
        ``chunks={"x": 1024, "y": 1024}``). These take precedence over the
        defaults used here (``chunks=True``, ``mask_and_scale=mask_nodata``).

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

        cmap = matplotlib.colormaps.get_cmap("Greens").copy()
        cmap.set_over("white")  # Set values over 100 (i.e., 255) to white

        im = self.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=100, add_colorbar=False)

        cbar = plt.colorbar(im, ax=ax, extend="max")
        cbar.set_label("Forest Cover Fraction (%)")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            "Copernicus Global Land Service Forest Cover Fraction\nLand Cover 100m: collection 3: epoch 2019"
        )
        f.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
        f.dpi = 300

        return f, ax

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    open_params = {"chunks": True, "mask_and_scale": mask_nodata, **kwargs}
    fcf_da = providers.raster_http.open(
        "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        squeeze=False,
        **open_params,
    )

    fcf_da = fcf_da.rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs).squeeze()

    fcf_da.attrs["example_plot"] = plot_forest_cover
    fcf_da.attrs["data_citation"] = (
        "Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050"
    )

    return fcf_da


def get_seasonal_snow_classification(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    mask_nodata: bool = False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments passed to ``rioxarray.open_rasterio`` (e.g.
        ``chunks={"x": 1024, "y": 1024}``). These take precedence over the
        defaults used here (``chunks=True``, ``mask_and_scale=mask_nodata``).

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
            (class_values[i] + class_values[i + 1]) / 2
            for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(
            ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False
        )
        # ax.set_aspect("equal")

        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black"
                )
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

    open_params = {"chunks": True, "mask_and_scale": mask_nodata, **kwargs}
    snow_classification_da = providers.raster_http.open(
        "https://uwcryo.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif",
        squeeze=False,
        **open_params,
    )
    snow_classification_da = snow_classification_da.rio.clip_box(
        *bbox_gdf.total_bounds, crs=bbox_gdf.crs
    ).squeeze()

    if mask_nodata:
        snow_classification_da.rio.write_nodata(9, encoded=True, inplace=True)
    else:
        snow_classification_da.rio.set_nodata(9, inplace=True)

    snow_classification_da.attrs["class_info"] = get_class_info()
    snow_classification_da.attrs["cmap"] = get_class_cmap(
        snow_classification_da.attrs["class_info"]
    )
    snow_classification_da.attrs["data_citation"] = (
        "Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0. Date Accessed 03-06-2024."
    )

    snow_classification_da.attrs["example_plot"] = plot_classes

    return snow_classification_da


def get_seasonal_mountain_snow_mask(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    data_product: str = "mountain_snow",
    mask_nodata: bool = False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments passed to ``rioxarray.open_rasterio`` (e.g.
        ``chunks={"x": 1024, "y": 1024}``). These take precedence over the
        defaults used here (``chunks=True``, ``mask_and_scale=mask_nodata``).

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
            raise ValueError(
                'Invalid data_product. Choose from "snow" or "mountain_snow".'
            )
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
            (class_values[i] + class_values[i + 1]) / 2
            for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(
            ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False
        )
        # ax.set_aspect("equal")

        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black"
                )
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
        ax.set_title(
            f"Global seasonal {'mountain ' if data_product == 'mountain_snow' else ''}snow mask\nfrom Wrzesien et al 2019"
        )
        f.tight_layout(pad=5.5, w_pad=5.5, h_pad=1.5)
        f.dpi = 300

        return f, ax

    print("This function takes a moment, getting zipped file from zenodo...")
    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    url = f"zip+https://zenodo.org/records/2626737/files/MODIS_{'mtnsnow' if data_product == 'mountain_snow' else 'snow'}_classes.zip!/MODIS_{'mtnsnow' if data_product == 'mountain_snow' else 'snow'}_classes.tif"

    open_params = {"chunks": True, "mask_and_scale": mask_nodata, **kwargs}
    mountain_snow_da = (
        providers.raster_http.open(url, squeeze=False, **open_params)
        .rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs)
        .squeeze()
    )

    # looks like the creators accidently set no data to 256 and 265 instead of 255, therefore unmasked the data is of type uint32 :(
    # attempt to fix this by setting all invalid values to 255, then converting types
    mask = mountain_snow_da > 3
    mountain_snow_da = mountain_snow_da.where(~mask, 255)

    if mask_nodata:
        mountain_snow_da = mountain_snow_da.astype("float32").rio.write_nodata(
            255, encoded=True
        )
    else:
        mountain_snow_da = mountain_snow_da.astype("uint8").rio.set_nodata(255)

    mountain_snow_da.attrs["class_info"] = get_class_info(data_product)
    mountain_snow_da.attrs["cmap"] = get_class_cmap(
        mountain_snow_da.attrs["class_info"]
    )
    mountain_snow_da.attrs["data_citation"] = (
        "Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019). Global Seasonal Mountain Snow Mask from MODIS MOD10A2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2626737"
    )

    mountain_snow_da.attrs["example_plot"] = plot_classes

    return mountain_snow_da


def get_esa_worldcover(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    version: str = "v200",
    mask_nodata: bool = False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments passed to ``odc.stac.load`` (e.g.
        ``chunks={"x": 1024, "y": 1024}``). These take precedence over the
        defaults used here (``bands="map"``, ``chunks={}``).

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
            (class_values[i] + class_values[i + 1]) / 2
            for i in range(len(class_values) - 1)
        ]
        bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot.imshow(
            ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False
        )
        # ax.set_aspect("equal")

        legend_handles = []
        class_names = []
        for class_value, class_info in self.attrs["class_info"].items():
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black"
                )
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

    catalog = providers.stac.open_catalog("planetary-computer")
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox_gdf.total_bounds)
    load_params = {
        "bbox": bbox_gdf.total_bounds,
        "bands": "map",
        "chunks": {},
        **kwargs,
    }
    worldcover_da = (
        providers.stac.odc_load(
            search.items(), catalog="planetary-computer", **load_params
        )["map"]
        .sel(time=version_year)
        .squeeze()
    )

    if mask_nodata:
        worldcover_da = worldcover_da.where(worldcover_da > 0)
        worldcover_da.rio.write_nodata(0, encoded=True, inplace=True)

    worldcover_da.attrs["class_info"] = get_class_info()
    worldcover_da.attrs["cmap"] = get_class_cmap(worldcover_da.attrs["class_info"])
    worldcover_da.attrs["data_citation"] = (
        "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936."
    )

    worldcover_da.attrs["example_plot"] = plot_classes

    return worldcover_da


@requires_earthengine
def get_nlcd_landcover(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    layer: str = "landcover",
    initialize_ee: bool = True,
    **kwargs,
) -> xr.DataArray:
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
    **kwargs
        Additional keyword arguments passed to ``xarray.open_dataset`` with
        ``engine="ee"`` (e.g. ``chunks={"time": 1, "x": 512, "y": 512}``).
        These take precedence over the defaults used here (``chunks={}``).

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
    Requires Google Earth Engine authentication. Run ``ee.Authenticate()`` and
    ``ee.Initialize()`` once, or call ``easysnowdata.authenticate_all()``.

    - NLCD data is only available for the contiguous United States
    - The latest version (2021) includes data from 2001-2021
    - Resolution is 30 meters

    Data citation:
    Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, doi:10.5066/P9JZ7AO3
    """
    # Initialize Earth Engine with high-volume endpoint
    if initialize_ee:
        initialize_earthengine()
    else:
        _logger.info(
            "Earth Engine initialization skipped. Ensure EE is already initialized."
        )

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    image_collection = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
    image = image_collection.first()

    # Match NLCD's native 30 m Albers grid, cropped to the bbox
    grid = get_ee_grid_params(image, bbox_gdf)

    open_params = {"engine": "ee", "chunks": {}, **grid, **kwargs}
    ds = (
        providers.gee.open_dataset(image_collection, grid={}, **open_params)
        .squeeze()
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(open_params["crs"])
        .astype("uint8")
    )

    nlcd_da = ds[layer]

    # would be nice for them to come in as ints
    # https://github.com/google/Xee/issues/86
    # https://github.com/google/Xee/issues/146

    def get_class_info():
        info = image.getInfo()["properties"]

        if layer == "landcover":
            return {
                value: {"name": name.split(":")[0], "color": f"#{palette}"}
                for value, name, palette in zip(
                    info["landcover_class_values"],
                    info["landcover_class_names"],
                    info["landcover_class_palette"],
                )
            }
        elif layer == "impervious":
            return None
        elif layer == "impervious_descriptor":
            return {
                value: {"name": name.split(".")[0], "color": f"#{palette}"}
                for value, name, palette in zip(
                    info["impervious_descriptor_class_values"],
                    info["impervious_descriptor_class_names"],
                    info["impervious_descriptor_class_palette"],
                )
            }
        elif layer.startswith("science_products"):
            return {
                value: {"name": name, "color": f"#{palette}"}
                for value, name, palette in zip(
                    info[f"{layer}_class_values"],
                    info[f"{layer}_class_names"],
                    info[f"{layer}_class_palette"],
                )
            }

    def get_class_cmap(classes):
        if classes is None:
            return plt.cm.YlOrRd
        return plt.cm.colors.ListedColormap(
            [classes[key]["color"] for key in classes.keys()]
        )

    def plot_classes(self, ax=None, figsize=(8, 10), legend_kwargs=None):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        if self.name != "impervious":
            class_values = sorted(list(self.attrs["class_info"].keys()))
            bounds = [
                (class_values[i] + class_values[i + 1]) / 2
                for i in range(len(class_values) - 1)
            ]
            bounds = [class_values[0] - 0.5] + bounds + [class_values[-1] + 0.5]
            norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

            im = self.plot.imshow(
                ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False
            )

            legend_handles = []
            class_names = []
            for class_value, class_info in self.attrs["class_info"].items():
                legend_handles.append(
                    plt.Rectangle(
                        (0, 0), 1, 1, facecolor=class_info["color"], edgecolor="black"
                    )
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
            f.colorbar(im, ax=ax, label="Percent impervious surface [%]")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.axis('equal')
        ax.set_title(f"NLCD {self.name.title()} (2021)")
        # f.tight_layout(pad=0, w_pad=0, h_pad=0)
        f.dpi = 300

        return f, ax

    class_info = get_class_info()
    nlcd_da.attrs["class_info"] = class_info
    nlcd_da.attrs["cmap"] = get_class_cmap(class_info)
    nlcd_da.attrs["example_plot"] = plot_classes

    nlcd_da.attrs["data_citation"] = (
        "Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, doi:10.5066/P9JZ7AO3"
    )

    return nlcd_da


@deprecated(
    "easysnowdata.optical.sentinel2.load",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.remote_sensing.Sentinel2",
    extra=(
        "The new loader is a function returning the Dataset; search results come from "
        "optical.sentinel2.search(), masking from mask=, and the SCL legend from "
        "easysnowdata.plotting.categorical()."
    ),
)
def Sentinel2(  # noqa: N802 — this was a class
    bbox_input,
    start_date="2014-01-01",
    end_date=None,
    catalog_choice="planetarycomputer",
    collection="sentinel-2-l2a",
    bands=None,
    resolution=None,
    crs=None,
    remove_nodata=True,
    harmonize_to_old=None,
    scale_data=True,
    groupby="solar_day",
    **kwargs,
):
    """Deprecated factory for :func:`easysnowdata.optical.sentinel2.load`.

    Returns the loaded ``xarray.Dataset`` instead of a ``Sentinel2`` object.
    Every old keyword still works for one release:

    * ``catalog_choice="planetarycomputer"``/``"earthsearch"`` → ``source=``
    * ``remove_nodata`` → ``mask_nodata=``
    * ``harmonize_to_old`` → ``harmonize=`` (``None`` now means "decide from
      the item metadata": Earth Search's ``sentinel-2-l2a`` pixels already
      carry the correction, Collection-1 and Planetary Computer do not)
    * ``scale_data`` → ``scale=``

    The ``.data`` / ``.metadata`` / ``.search`` attributes, the ``get_*``
    methods and the ``class_info`` / ``cmap`` / ``example_plot`` attrs are
    gone: use :func:`easysnowdata.optical.sentinel2.search`,
    :mod:`easysnowdata.processing` (``ndsi``, ``ndvi``, ``rgb``,
    ``stretch_percentile`` …) and :mod:`easysnowdata.plotting`.
    """
    from easysnowdata.optical import sentinel2 as _sentinel2  # noqa: PLC0415

    catalogs = {
        "planetarycomputer": "planetary-computer",
        "earthsearch": "earth-search",
    }
    if catalog_choice not in catalogs:
        raise ValueError(
            "Invalid catalog_choice. Choose either 'planetarycomputer' or 'earthsearch'."
        )
    return _sentinel2.load(
        bbox_input,
        (start_date, end_date if end_date is not None else temporal.today()),
        bands=list(bands) if bands else list(_sentinel2.ALL_BANDS),
        source=catalogs[catalog_choice],
        collection=collection,
        resolution=resolution,
        crs=crs if crs is not None else "utm",
        groupby=groupby,
        harmonize=True if harmonize_to_old is None else bool(harmonize_to_old),
        scale=bool(scale_data),
        mask_nodata=bool(remove_nodata),
        **kwargs,
    )


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
    **kwargs
        Additional keyword arguments passed to ``odc.stac.load``. These take
        precedence over the defaults used by ``get_data()``.

    Attributes
    ----------
    data : xarray.Dataset
        The loaded Sentinel-1 data.
    metadata : geopandas.GeoDataFrame
        Metadata for the retrieved Sentinel-1 scenes.
    local_incidence_angle_data : xarray.DataArray
        Local incidence angle values calculated from Sentinel-1 data and Copernicus DEM.
        functionality requires google earth engine initialization, and is based on https://gis.stackexchange.com/a/352658

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
    get_local_incidence_angle()
        Calculates and retrieves the local incidence angle for the area of interest.
    """

    def __init__(
        self,
        bbox_input,
        start_date="2014-01-01",
        end_date=None,
        catalog_choice="planetarycomputer",
        bands=None,
        units="dB",  # linear power or dB
        resolution=None,
        crs=None,
        groupby="sat:absolute_orbit",
        chunks={},  # {"x": 512, "y": 512} or # {"x": 512, "y": 512, "time": -1}
        remove_border_noise=True,
        **kwargs,
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
            **kwargs: Additional keyword arguments passed to odc.stac.load. These take precedence over the defaults used by get_data().
        """
        # Initialize the attributes
        self.bbox_input = bbox_input
        self.start_date = start_date
        self.end_date = end_date if end_date is not None else temporal.today()
        self.catalog_choice = catalog_choice
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.chunks = chunks
        self.groupby = groupby
        self.remove_border_noise = remove_border_noise
        self.load_kwargs = kwargs

        # if not self.geobox:
        self.bbox_gdf = convert_bbox_to_geodataframe(self.bbox_input)

        if self.crs is None:
            self.crs = self.bbox_gdf.estimate_utm_crs()

        # if resolution == None:
        #     self.resolution = 10

        self.search = None
        self.data = None
        self.metadata = None
        self._local_incidence_angle_data = None

        self.search_data()
        self.get_data()
        self.get_metadata()
        if self.remove_border_noise:
            self.remove_bad_scenes_and_border_noise()
        self.add_orbit_info()
        if units == "dB":
            self.linear_to_db()
        else:
            print(
                "Units remain in linear power. Convert to dB using the .linear_to_db() method."
            )

    def search_data(self):
        """
        The method to search the data.
        """

        # Choose the catalog URL based on catalog_choice
        if self.catalog_choice == "planetarycomputer":
            catalog = providers.stac.open_catalog("planetary-computer")
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
        print("Data searched. Access the returned seach with the .search attribute.")

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
        # User-supplied kwargs take precedence over the defaults above
        load_params.update(self.load_kwargs)

        # Load the data lazily using odc.stac
        self.data = providers.stac.odc_load(
            load_params.pop("items"), catalog="planetary-computer", **load_params
        ).sortby(
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
        print("Metadata retrieved. Access with the .metadata attribute.")

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
        cutoff_date = np.datetime64("2018-03-14")

        original_crs = self.data.rio.crs

        result = xr.where(
            self.data.time < cutoff_date,
            self.data.where(self.data > threshold),
            self.data.where(self.data > 0),
        )

        result.rio.write_crs(original_crs, inplace=True)

        self.data = result
        print("Falsely low scenes and border noise removed from the data.")

    def linear_to_db(self):
        """
        The method to convert the linear power data to dB.
        """
        self.data = 10 * np.log10(self.data)
        self.data.attrs["units"] = "dB"
        print(
            "Linear power units converted to dB. Convert back to linear power units using the .db_to_linear() method."
        )

    def db_to_linear(self):
        """
        The method to convert the dB data to linear power.
        """
        self.data = 10 ** (self.data / 10)
        self.data.attrs["units"] = "linear power"
        print(
            "dB converted to linear power units. Convert back to dB using the .linear_to_db() method."
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
        print("Added relative orbit number and orbit state as coordinates to the data.")

    @property
    def local_incidence_angle_data(self):
        """
        Property to access local incidence angle data.

        Returns
        -------
        xarray.DataArray
            DataArray containing local incidence angle values aligned to the same grid as the primary data.

        Notes
        -----
        On first access, this property calculates local incidence angles using Sentinel-1 data and
        Copernicus 30m DEM. Results are cached for subsequent accesses.
        """
        if self._local_incidence_angle_data is None:
            self.get_local_incidence_angle()
        return self._local_incidence_angle_data

    @requires_earthengine
    def get_local_incidence_angle(self, resolution=None, initialize_ee=True):
        """
        Calculate local incidence angle for Sentinel-1 data within a bounding box.

        Parameters
        ----------
        resolution : int or float, optional
            Desired resolution in meters for the calculation. Defaults to self.resolution if set, or 30m.
        initialize_ee : bool, optional
            Whether to initialize Earth Engine. Default is True.

        Returns
        -------
        xarray.DataArray
            DataArray containing local incidence angle values with dimensions (sat:relative_orbit, y, x),
            aligned to the same grid as the primary data.

        Notes
        -----
        Requires Google Earth Engine authentication. Run ``ee.Authenticate()`` and
        ``ee.Initialize()`` once, or call ``easysnowdata.authenticate_all()``.

        This method calculates the local incidence angle using the Copernicus 30m DEM and
        stores the results in the local_incidence_angle_data attribute.
        """
        import math
        from collections import Counter

        from xee import helpers as xee_helpers

        # Use object's resolution if none provided
        calc_resolution = resolution or self.resolution or 30

        # Initialize Earth Engine with high-volume endpoint
        if initialize_ee:
            initialize_earthengine()
        else:
            _logger.info(
                "Earth Engine initialization skipped. Ensure EE is already initialized."
            )

        # Convert bbox to Earth Engine geometry
        bbox = tuple(self.bbox_gdf.total_bounds)
        ee_bbox = ee.Geometry.Rectangle(bbox)

        # Filter Sentinel-1 collection
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(ee_bbox)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )

        # Get distinct orbit numbers
        distinct_orbits = collection.aggregate_array(
            "relativeOrbitNumber_start"
        ).distinct()
        orbit_list = distinct_orbits.getInfo()

        if not orbit_list:
            raise ValueError("No Sentinel-1 data found for the specified bounding box.")

        print(f"Found {len(orbit_list)} unique relative orbits: {orbit_list}")

        # Find the most common projection among the orbits
        orbit_projections = {}
        print("Analyzing orbit projections to find the most common one...")

        for orbit in orbit_list:
            orbit_image = collection.filter(
                ee.Filter.eq("relativeOrbitNumber_start", orbit)
            ).first()

            if orbit_image:
                # Get projection info
                proj_info = orbit_image.select(0).projection().getInfo()
                crs = proj_info["crs"]
                orbit_projections[orbit] = {
                    "crs": crs,
                    "transform": proj_info["transform"],
                }
                print(f"Orbit {orbit} uses {crs}")

        # Count CRS frequencies
        crs_counts = Counter([info["crs"] for info in orbit_projections.values()])
        most_common_crs = crs_counts.most_common(1)[0][0]

        print(
            f"Most common CRS: {most_common_crs} (used by {crs_counts[most_common_crs]} of {len(orbit_list)} orbits)"
        )

        # Function to calculate local incidence angle using Copernicus 30m DEM
        def calculate_local_incidence_angle(image):
            img_geom = image.geometry()

            # Use Copernicus 30m DEM with proper reprojection
            dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")
            dem = dem_collection.select("DEM").mosaic().clip(img_geom)

            # Reproject DEM to the most common CRS with the specified resolution
            projection = ee.Projection(most_common_crs).atScale(calc_resolution)
            dem = dem.reproject(projection)

            # 2.1.1 Radar geometry
            theta_i = image.select("angle")
            phi_i = (
                ee.Terrain.aspect(theta_i)
                .reduceRegion(ee.Reducer.mean(), theta_i.get("system:footprint"), 1000)
                .get("aspect")
            )

            # 2.1.2 Terrain geometry
            alpha_s = ee.Terrain.slope(dem).select("slope")
            phi_s = ee.Terrain.aspect(dem).select("aspect")

            # 2.1.3 Model geometry
            # reduce to 3 angle
            phi_r = ee.Image.constant(phi_i).subtract(phi_s)

            # convert all to radians
            phi_rRad = phi_r.multiply(math.pi / 180)
            alpha_sRad = alpha_s.multiply(math.pi / 180)
            theta_iRad = theta_i.multiply(math.pi / 180)
            ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

            # slope steepness in range (eq. 2)
            alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

            # slope steepness in azimuth (eq 3)
            alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

            # local incidence angle (eq. 4)
            cos_theta_lia = alpha_az.cos().multiply(
                (theta_iRad.subtract(alpha_r)).cos()
            )

            # Ensure valid range for acos
            cos_theta_lia = cos_theta_lia.clamp(-1, 1)

            theta_lia = cos_theta_lia.acos()
            theta_liaDeg = theta_lia.multiply(180 / math.pi)

            return image.addBands(theta_liaDeg.rename("local_incidence_angle"))

        # Create list to store DataArrays for each orbit
        orbit_arrays = []

        # Output grid for xee (>= 0.1 requires an explicit pixel grid): the most
        # common CRS at the requested resolution, covering the bbox
        grid = xee_helpers.fit_geometry(
            shapely.geometry.box(*self.bbox_gdf.total_bounds),
            geometry_crs=str(self.bbox_gdf.crs or "EPSG:4326"),
            grid_crs=most_common_crs,
            grid_scale=(calc_resolution, -calc_resolution),
        )

        # Process each orbit
        for orbit in orbit_list:
            # Get images for this orbit
            orbit_images = (
                collection.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
                .sort("system:time_start", True)
                .limit(3)
            )

            if orbit_images.size().getInfo() > 0:
                # Calculate LIA for each image
                lia_images = orbit_images.map(calculate_local_incidence_angle)

                # Calculate median LIA
                median_lia = lia_images.select("local_incidence_angle").median()

                # Set properties on the median image
                timestamp = orbit_images.first().get("system:time_start")
                median_lia = median_lia.set(
                    {"relativeOrbitNumber_start": orbit, "system:time_start": timestamp}
                )

                # Create a single-image collection for xee
                orbit_collection = ee.ImageCollection([median_lia])

                # Use xee to convert to xarray
                try:
                    ds = providers.gee.open_dataset(
                        orbit_collection, grid={}, engine="ee", chunks={}, **grid
                    )

                    # Extract the DataArray
                    da = ds["local_incidence_angle"]

                    # Remove the time dimension if present
                    if "time" in da.dims:
                        da = da.isel(time=0, drop=True)

                    # Add orbit as a coordinate
                    da = da.assign_coords({"sat:relative_orbit": orbit})

                    # Check for NaN values
                    nan_percentage = np.isnan(da.values).mean() * 100
                    print(
                        f"Orbit {orbit} - Shape: {da.shape}, NaN percentage: {nan_percentage:.1f}%"
                    )

                    if nan_percentage < 100:  # Only keep arrays with some valid data
                        # Store in list
                        orbit_arrays.append(da)
                        print(f"Successfully processed orbit {orbit}")
                    else:
                        print(f"Skipping orbit {orbit} - all values are NaN")

                except Exception as e:
                    print(f"Error processing orbit {orbit}: {e}")

        if orbit_arrays:
            # Ensure all arrays have the same shape before concatenating
            shapes = [da.shape for da in orbit_arrays]
            if len(set(shapes)) > 1:
                print(f"Warning: Arrays have different shapes: {shapes}")

                # Take the shape with the most non-NaN values as template
                best_da_idx = np.argmax(
                    [~np.isnan(da.values).sum() for da in orbit_arrays]
                )
                template_da = orbit_arrays[best_da_idx]

                for i in range(len(orbit_arrays)):
                    if i != best_da_idx and orbit_arrays[i].shape != template_da.shape:
                        orbit_num = orbit_arrays[i].sat_relative_orbit.values[0]
                        print(
                            f"Resampling orbit {orbit_num} to match template shape {template_da.shape}"
                        )
                        orbit_arrays[i] = orbit_arrays[i].reindex_like(template_da)

            # Combine all orbits into a single DataArray
            lia_da = xr.concat(orbit_arrays, dim="sat:relative_orbit")

            # Add attributes
            lia_da.attrs.update(
                {
                    "long_name": "Sentinel-1 Local Incidence Angle",
                    "units": "degrees",
                    "description": "Local incidence angle calculated from Sentinel-1 data and Copernicus 30m DEM",
                    "source": "Sentinel-1 GRD",
                }
            )

            lia_da = (
                lia_da.transpose("sat:relative_orbit", "y", "x")
                .rio.set_spatial_dims(x_dim="x", y_dim="y")
                .rio.write_crs(grid["crs"])
            )
            lia_da = lia_da.sortby("sat:relative_orbit")
            # should be in range from 0 to 90
            # lia_da = lia_da.where(lambda x: (x >= 0) & (x <= 90))

            # Reproject to match data grid exactly using bilinear interpolation
            if self.data is not None:
                # Get reference grid from first data variable
                ref_da = self.data[list(self.data.data_vars)[0]].isel(time=0)

                # Ensure lia_da has CRS information
                if not lia_da.rio.crs and ref_da.rio.crs:
                    lia_da.rio.write_crs(ref_da.rio.crs, inplace=True)

                # Reproject to match data grid using bilinear interpolation, careful with nodata
                lia_da = lia_da.rio.reproject_match(
                    ref_da,
                    resampling=rio.warp.Resampling.bilinear,
                    nodata=np.nan,
                )

                print(
                    "Local incidence angle data reprojected to match main data grid using bilinear resampling."
                )

            self._local_incidence_angle_data = lia_da
            print(
                "Local incidence angle calculation complete. Access via the .local_incidence_angle_data attribute."
            )

            # return self._local_incidence_angle_data
        else:
            raise ValueError(
                "No valid Sentinel-1 data found for the specified bounding box."
            )


@deprecated(
    "easysnowdata.optical.hls.load",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.remote_sensing.HLS",
    extra=(
        "The new loader reads the scene metadata from STAC properties instead of "
        "fetching one XML file per granule, and adds a credential-free Planetary "
        "Computer route (source='planetary-computer')."
    ),
)
def HLS(  # noqa: N802 — this was a class
    bbox_input,
    start_date="2014-01-01",
    end_date=None,
    bands=None,
    resolution=None,
    crs="utm",
    remove_nodata=True,
    scale_data=True,
    add_metadata=True,
    add_platform=True,
    groupby="solar_day",
    **kwargs,
):
    """Deprecated factory for :func:`easysnowdata.optical.hls.load`.

    Returns the loaded ``xarray.Dataset`` (both products stacked on ``time``
    with ``product`` and ``platform`` coordinates) instead of an ``HLS``
    object. ``remove_nodata`` → ``mask_nodata=``, ``scale_data`` → ``scale=``.
    ``add_metadata`` and ``add_platform`` are accepted and ignored: the
    platform is always a coordinate now, and the per-scene metadata comes from
    :func:`easysnowdata.optical.hls.search` instead of the XML scrape.
    """
    from easysnowdata.optical import hls as _hls  # noqa: PLC0415

    return _hls.load(
        bbox_input,
        (start_date, end_date if end_date is not None else temporal.today()),
        bands=list(bands) if bands else list(_hls.ALL_BANDS),
        resolution=resolution if resolution is not None else 30,
        crs=crs if crs is not None else "utm",
        groupby=groupby,
        scale=bool(scale_data),
        mask_nodata=bool(remove_nodata),
        **kwargs,
    )


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
    **kwargs
        Additional keyword arguments passed to the underlying loader:
        ``odc.stac.load`` for ``MOD10A1`` / ``MOD10A2``, or
        ``rioxarray.open_rasterio`` for ``MOD10A1F``. These take precedence
        over the defaults used by ``get_data()`` (e.g.
        ``chunks={"time": 1, "x": 512, "y": 512}``).

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
    The ``MOD10A1F`` product requires NASA EarthData credentials
    (``EARTHDATA_TOKEN``, or ``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD``,
    or a ``~/.netrc`` entry written by ``earthaccess.login(persist=True)``);
    the class logs in through ``earthaccess`` itself and downloads the HDF4
    granules into the easysnowdata cache directory. Reading them needs a GDAL
    build with the HDF4 driver (``libgdal-hdf4`` on conda-forge). The
    ``MOD10A1`` and ``MOD10A2`` products use Microsoft Planetary Computer and
    require no credentials.

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
        end_date=None,
        data_product="MOD10A2",
        bands=None,
        resolution=None,
        crs=None,
        vertical_tile=None,
        horizontal_tile=None,
        mute=False,
        **kwargs,
    ):

        if data_product == "MOD10A1F" and not _has_earthaccess_credentials():
            raise CredentialError(
                f"`MODIS_snow` with data_product='MOD10A1F' requires NASA EarthData credentials.\n\n{_EARTHACCESS_SETUP_MSG}"
            )

        self.bbox_input = bbox_input
        self.bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
        self.clip_to_bbox = clip_to_bbox
        self.start_date = start_date
        self.end_date = end_date if end_date is not None else temporal.today()
        self.data_product = data_product
        self.bands = bands
        self.resolution = resolution
        self.crs = crs
        self.vertical_tile = vertical_tile
        self.horizontal_tile = horizontal_tile
        self.load_kwargs = kwargs

        if mute:
            with suppress_stdout():
                self.search_data()
                self.get_data()
        else:
            self.search_data()
            self.get_data()

    def search_data(self):

        if self.data_product == "MOD10A1" or self.data_product == "MOD10A2":
            catalog = providers.stac.open_catalog("planetary-computer")

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
            # MOD10A1F v61 is cloud-hosted at NSIDC (provider NSIDC_CPRD); the
            # provider logs in explicitly (earthaccess >= 0.16) before searching.
            search = providers.earthdata.search(
                "MOD10A1F",
                cloud_hosted=True,
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
            # User-supplied kwargs take precedence over the defaults above
            load_params.update(self.load_kwargs)

            modis_snow = providers.stac.odc_load(
                load_params.pop("items"), catalog="planetary-computer", **load_params
            )

        elif self.data_product == "MOD10A1F":
            # The granules are HDF-EOS2 (HDF4) files. Check for the driver
            # before downloading anything: rasterio's PyPI wheels ship GDAL
            # without HDF4 (verified for rasterio 1.5.1 / GDAL 3.12.4), while
            # conda-forge provides it as the separate libgdal-hdf4 package.
            providers.earthdata.require_hdf4("MOD10A1F")
            # HDF4 cannot be read through fsspec file objects either, so
            # download the granules once into the easysnowdata cache directory
            # (~/.cache/easysnowdata/MOD10A1F on Linux; override with
            # EASYSNOWDATA_CACHE_DIR). earthaccess skips files already present.
            files = providers.earthdata.download(self.search, "MOD10A1F")

            # User-supplied kwargs take precedence over the defaults here
            open_params = {
                "variable": "CGF_NDSI_Snow_Cover",
                "chunks": {},
                **self.load_kwargs,
            }

            if self.clip_to_bbox:
                modis_snow = xr.concat(
                    [
                        providers.raster_http.open(file, squeeze=False, **open_params)[
                            "CGF_NDSI_Snow_Cover"
                        ]
                        .squeeze()
                        .rio.clip_box(
                            *self.bbox_gdf.total_bounds, crs=self.bbox_gdf.crs
                        )
                        .assign_coords(
                            time=pd.to_datetime(
                                providers.raster_http.open(
                                    file, squeeze=False, **open_params
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
                        providers.raster_http.open(file, squeeze=False, **open_params)[
                            "CGF_NDSI_Snow_Cover"
                        ]
                        .squeeze()
                        .assign_coords(
                            time=pd.to_datetime(
                                providers.raster_http.open(
                                    file, squeeze=False, **open_params
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
            self.binary_snow = xr.where(
                self.data["Maximum_Snow_Extent"] == 200, 1, 0
            ).rio.write_crs(self.data.rio.crs)
            print("Binary snow map calculated. Access with the .binary_snow attribute.")
        else:
            print("This method is only available for the MOD10A2 product.")


# palsar2
# ic = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR').filterDate('2020-10-05', '2021-03-31')
# ds = xarray.open_dataset(ic, geometry=bbox_ee,engine='ee')
