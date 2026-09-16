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
import shapely
import xarray as xr

from easysnowdata import auth, providers, temporal
from easysnowdata._deprecation import deprecated
from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    get_ee_grid_params,
    initialize_earthengine,
    requires_earthengine,
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


@deprecated(
    "easysnowdata.sar.sentinel1.load",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.remote_sensing.Sentinel1",
    extra=(
        "The new loader adds the OPERA RTC-S1 route (source='opera-rtc-s1') and "
        "easysnowdata.sar.sentinel1.local_incidence_angle(), which reads the OPERA "
        "static layer or computes the angle from a DEM without Earth Engine."
    ),
)
def Sentinel1(  # noqa: N802 — this was a class
    bbox_input,
    start_date="2014-01-01",
    end_date=None,
    catalog_choice="planetarycomputer",
    bands=None,
    units="dB",
    resolution=None,
    crs=None,
    groupby="sat:absolute_orbit",
    chunks=None,
    remove_border_noise=True,
    **kwargs,
):
    """Deprecated factory for :func:`easysnowdata.sar.sentinel1.load`.

    Returns the loaded ``xarray.Dataset`` (with ``sat:orbit_state`` and
    ``sat:relative_orbit`` coordinates) instead of a ``Sentinel1`` object.
    The ``.local_incidence_angle_data`` property is replaced by
    :func:`easysnowdata.sar.sentinel1.local_incidence_angle`, which no longer
    needs Earth Engine.
    """
    from easysnowdata.sar import sentinel1 as _sentinel1  # noqa: PLC0415

    if catalog_choice != "planetarycomputer":
        raise ValueError(
            "Invalid catalog_choice. Choose either 'planetarycomputer' or <unimplemented>."
        )
    return _sentinel1.load(
        bbox_input,
        (start_date, end_date if end_date is not None else temporal.today()),
        bands=bands,
        source="planetary-computer",
        units=units,
        resolution=resolution,
        crs=crs if crs is not None else "utm",
        groupby=groupby,
        border_noise=bool(remove_border_noise),
        chunks=chunks if chunks else None,
        **kwargs,
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


@deprecated(
    "easysnowdata.snow.modis.load",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.remote_sensing.MODIS_snow",
    extra=(
        "The new loader defaults to NSIDC (the archive of record, and the only route "
        "with the cloud-gap-filled and Aqua products) and keeps Planetary Computer as "
        "source='planetary-computer'; get_binary_snow() is now "
        "easysnowdata.processing.binary_snow()."
    ),
)
def MODIS_snow(  # noqa: N802 — this was a class
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
    """Deprecated factory for :func:`easysnowdata.snow.modis.load`.

    Returns the loaded ``xarray.Dataset`` instead of a ``MODIS_snow`` object.
    ``data_product`` → ``product=``, ``bands`` → ``variables=``,
    ``clip_to_bbox`` → the AOI's own ``clip`` flag. ``mute`` is accepted and
    ignored (the loaders log instead of printing), and ``vertical_tile`` /
    ``horizontal_tile`` are no longer needed: pass an AOI, or ``clip=False``
    on it to keep whole tiles.
    """
    from easysnowdata.aoi import parse_aoi  # noqa: PLC0415
    from easysnowdata.snow import modis as _modis  # noqa: PLC0415

    product = str(data_product).upper()
    source = "planetary-computer" if product in _modis.PC_COLLECTIONS else "nsidc"
    aoi = (
        parse_aoi(bbox_input, clip=bool(clip_to_bbox))
        if bbox_input is not None
        else None
    )
    return _modis.load(
        aoi,
        (start_date, end_date if end_date is not None else temporal.today()),
        product=product,
        variables=bands,
        source=source,
        resolution=resolution,
        crs=crs,
        **kwargs,
    )


# palsar2
# ic = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR').filterDate('2020-10-05', '2021-03-31')
# ds = xarray.open_dataset(ic, geometry=bbox_ee,engine='ee')
