"""Access remote sensing datasets for snow science applications.

Includes Sentinel-1, Sentinel-2, HLS, MODIS snow products, land cover,
snow classifications, and more.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import shapely
import xarray as xr

from easysnowdata import auth, temporal
from easysnowdata._deprecation import deprecated
from easysnowdata.land import forest_cover, landcover, nlcd
from easysnowdata.snow import mountain_snow_mask, snow_classification

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


@deprecated(
    "easysnowdata.land.forest_cover.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.remote_sensing.get_forest_cover_fraction",
    extra="The example_plot attr is gone; the data carries units and a long_name.",
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.land.forest_cover.load`, which masks the 255
        sentinel by default and adds the Earth Engine epochs
        (``source="gee"``, ``time=``).

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=255, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=255)
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.land.forest_cover.load` (and on to
        ``rioxarray.open_rasterio``).

    Returns
    -------
    xarray.DataArray
        Forest cover fraction DataArray, in percent.

    Notes
    -----
    Data citation:
    Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020).
    Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050
    """
    return forest_cover.load(bbox_input, mask=mask_nodata, **kwargs)


@deprecated(
    "easysnowdata.snow.snow_classification.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.remote_sensing.get_seasonal_snow_classification",
    extra=(
        "The new loader defaults to NSIDC-0768 (Earthdata Login); this shim "
        "keeps the hosted COG. The class table is CF flag attrs now."
    ),
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.snow.snow_classification.load`. Its default
        source is the authoritative NSIDC-0768 archive (Earthdata Login, and
        the coarser grids); this shim keeps reading the hosted COG, whose
        location is likely to change.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=9, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=9)
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.snow.snow_classification.load` (and on to
        ``rioxarray.open_rasterio``).

    Returns
    -------
    xarray.DataArray
        Seasonal snow class DataArray with CF flag attributes.

    Notes
    -----
    Data citation:
    Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set].
    Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0.
    """
    return snow_classification.load(
        bbox_input, source="hosted-cog", mask=mask_nodata, **kwargs
    )


@deprecated(
    "easysnowdata.snow.mountain_snow_mask.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.remote_sensing.get_seasonal_mountain_snow_mask",
    extra=(
        "The archive is cached after the first call, the clouds layer is "
        "reachable with layer='clouds', and the class table is CF flag attrs."
    ),
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.snow.mountain_snow_mask.load`, whose ``layer=``
        also reaches the clouds layer and which caches the Zenodo archive
        instead of re-reading it on every call.

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
        Additional keyword arguments passed to
        :func:`easysnowdata.snow.mountain_snow_mask.load` (and on to
        ``rioxarray.open_rasterio``).

    Returns
    -------
    xarray.DataArray
        Mountain snow DataArray with CF flag attributes.

    Notes
    -----
    Data citation:
    Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019).
    Global Seasonal Mountain Snow Mask from MODIS MOD10A2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2626737
    """
    return mountain_snow_mask.load(
        bbox_input, layer=data_product, mask=mask_nodata, **kwargs
    )


@deprecated(
    "easysnowdata.land.landcover.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.remote_sensing.get_esa_worldcover",
    extra=(
        "The class table is now CF flag attrs (esd.plotting.categorical draws "
        "them); the class_info, cmap and example_plot attrs are gone."
    ),
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.land.landcover.load`, which adds the unsigned
        AWS bucket (``source="aws-open-data"``) and carries the class table as
        CF flag attributes.

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
        Additional keyword arguments passed to
        :func:`easysnowdata.land.landcover.load` (and on to ``odc.stac.load``).

    Returns
    -------
    xarray.DataArray
        WorldCover DataArray with CF flag attributes.

    Notes
    -----
    Data citation:
    Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A.,
    Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin,
    Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936.
    """
    return landcover.load(bbox_input, version=version, mask=mask_nodata, **kwargs)


@deprecated(
    "easysnowdata.land.nlcd.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.remote_sensing.get_nlcd_landcover",
    extra=(
        "The new loader defaults to Annual NLCD (1985-2024); this shim keeps "
        "the official 2021 release. Class tables are CF flag attrs now."
    ),
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.land.nlcd.load`. Its default source is the
        Annual NLCD collection (1985-2024, ``time=`` selects years); pass
        ``source="gee"`` for the official 2021 release this shim uses.

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
        Ignored: Earth Engine is initialised on first use (§2.8).
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.land.nlcd.load` (and on to
        ``xarray.open_dataset(engine="ee")``).

    Returns
    -------
    xarray.DataArray
        NLCD DataArray for the specified region and layer.

    Notes
    -----
    Requires Google Earth Engine authentication; see ``esd.auth.status()``.

    - NLCD data is only available for the contiguous United States
    - Resolution is 30 meters

    Data citation:
    Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, doi:10.5066/P9JZ7AO3
    """
    return nlcd.load(bbox_input, source="gee", layer=layer, **kwargs)


@deprecated(
    "easysnowdata.optical.sentinel2.load",
    since="0.1.0",
    remove_in="0.3.0",
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
    remove_in="0.3.0",
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
    remove_in="0.3.0",
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
    remove_in="0.3.0",
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
