"""Access hydroclimatology datasets: ERA5, SNODAS, UCLA reanalysis, basin geometries, and more."""

from __future__ import annotations

import json
import logging
import re

import ee
import geopandas as gpd
import pandas as pd
import shapely
import xarray as xr

from easysnowdata import providers, temporal
from easysnowdata._deprecation import deprecated
from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    initialize_earthengine,
    requires_earthaccess,
    requires_earthengine,
)

__all__ = [
    "get_huc_geometries",
    "get_hydroBASINS",
    "get_grdc_major_river_basins_of_the_world",
    "get_grdc_wmo_basins",
    "get_era5",
    "get_snodas",
    "get_ucla_snow_reanalysis",
    "get_koppen_geiger_classes",
]

_logger = logging.getLogger(__name__)

# Position of each ensemble statistic along the ``Stats`` dimension of the
# WUS_UCLA_SR files: mean, standard deviation, median, 25th and 75th percentile.
_UCLA_SR_STATS_INDEX = {"mean": 0, "std": 1, "median": 2, "25pct": 3, "75pct": 4}


@requires_earthengine
def get_huc_geometries(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
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
    Requires Google Earth Engine authentication. Run ``ee.Authenticate()`` and
    ``ee.Initialize()`` once, or call ``easysnowdata.authenticate_all()``.

    Data citation:
    Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022,
    Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.):
    U.S. Geological Survey Techniques and Methods 11-A3, 54 p.,
    https://doi.org/10.3133/tm11A3
    """

    initialize_earthengine()

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
            f"huc{huc_level.lstrip('0')}",
            "areasqkm",
            "states",
            "tnmid",
            "geometry",
        ]
    ]

    huc_gdf.attrs = {
        "Data citation": "Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): U.S. Geological Survey Techniques and Methods 11-A3, 54 p., https://doi.org/10.3133/tm11A3"
    }

    return huc_gdf


def get_hydroBASINS(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    level: int = 5,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Retrieves HydroATLAS sub-basin boundaries at specified hierarchical level.

    This function downloads and loads vectorized polygon layers depicting sub-basin boundaries
    from the HydroATLAS database via figshare. It provides consistently sized and hierarchically
    nested sub-basins at different scales, supported by Pfafstetter coding for catchment topology analysis.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    level : int, optional
        The hierarchical level (1-12) of sub-basin delineation. Higher levels represent
        finer subdivisions. Default is 5.
    **kwargs
        Additional keyword arguments passed to ``geopandas.read_file`` (e.g.
        ``columns=[...]``, ``rows=...``, ``engine="pyogrio"``). These take
        precedence over the defaults used here (``layer`` and, when a bbox is
        given, ``mask``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the HydroATLAS sub-basin boundaries with associated attributes.

    Examples
    --------
    Get level 5 sub-basins for all regions...

    >>> basins = get_hydroBASINS()
    >>> basins.plot()

    Get level 6 sub-basins for a specific region...

    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> regional_basins = get_hydroBASINS(bbox_input=bbox, level=6)
    >>> regional_basins.plot()

    Notes
    -----
    This function uses the HydroATLAS dataset which provides global coverage in a single file,
    making it more efficient than downloading individual regional HydroBASINS files.

    Data citation:
    Linke, S., Lehner, B., Ouellet Dallaire, C., Ariwi, J., Grill, G., Anand, M., Beames, P.,
    Burchard-Levine, V., Maxwell, S., Moidu, H., Tan, F., Thieme, M. (2019). Global hydro-
    environmental sub-basin and river reach characteristics at high spatial resolution.
    Scientific Data 6: 283. doi: 10.1038/s41597-019-0300-6
    """

    # Validate level parameter
    if level < 1 or level > 12:
        raise ValueError(f"Level must be between 1 and 12, got {level}")

    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = (
        convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    )

    # Construct URL and layer name. Use the ndownloader.figshare.com host: it
    # answers with a plain 302 to the signed S3 object, whereas
    # figshare.com/ndownloader serves a bot-challenge page (HTTP 202) to
    # non-browser clients such as GDAL.
    url = "https://ndownloader.figshare.com/files/20082137/BasinATLAS_Data_v10.gdb.zip"
    layer_name = f"BasinATLAS_v10_lev{level:02d}"

    _logger.info("Loading HydroATLAS level {level} basins...")

    # Load the data with optional spatial masking
    read_params = {"layer": layer_name}
    if bbox_gdf is not None:
        read_params["mask"] = bbox_gdf
    else:
        _logger.info("Loading global dataset (this may take a while)...")
    # User-supplied kwargs take precedence over the defaults above
    read_params.update(kwargs)
    basins_gdf = providers.vector_http.read("zip+" + url, **read_params)

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = (
        "Linke, S., Lehner, B., Ouellet Dallaire, C., Ariwi, J., Grill, G., Anand, M., "
        "Beames, P., Burchard-Levine, V., Maxwell, S., Moidu, H., Tan, F., Thieme, M. (2019). "
        "Global hydro-environmental sub-basin and river reach characteristics at high spatial "
        "resolution. Scientific Data 6: 283. doi: 10.1038/s41597-019-0300-6"
    )

    return basins_gdf


def get_grdc_major_river_basins_of_the_world(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Retrieves GRDC Major River Basins of the World dataset.

    This function downloads and loads the Global Runoff Data Centre's (GRDC) Major River Basins
    dataset, which contains 520 river/lake basins considered major in size or hydro-political
    importance. The basins include both exorheic drainage (flowing to oceans) and endorheic
    drainage (inland sinks/lakes) systems.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    **kwargs
        Additional keyword arguments passed to ``geopandas.read_file`` (e.g.
        ``columns=[...]``, ``rows=...``, ``engine="pyogrio"``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the GRDC major river basins with associated attributes.

    Examples
    --------
    Get all major river basins...

    >>> basins = get_grdc_basins()
    >>> basins.plot()

    Get basins for a specific region...

    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> regional_basins = get_grdc_basins(bbox_input=bbox)
    >>> regional_basins.plot()

    Notes
    -----
    This dataset incorporates data from HydroSHEDS database which is © World Wildlife Fund, Inc.
    (2006-2013) and has been used under license.

    Data citation:
    GRDC (2020): GRDC Major River Basins. Global Runoff Data Centre. 2nd, rev. ed.
    Koblenz: Federal Institute of Hydrology (BfG).
    """

    url = "https://datacatalogfiles.worldbank.org/ddh-published/0041426/DR0051689/major_basins_of_the_world_0_0_0.zip"

    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = (
        convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    )

    # Load the data
    basins_gdf = providers.vector_http.read("zip+" + url, **kwargs)

    # Clip to bbox if provided
    if bbox_gdf is not None:
        basins_gdf = basins_gdf.clip(bbox_gdf)
    else:
        _logger.info("No spatial subsetting because bbox_input was not provided.")

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = (
        "GRDC (2020): GRDC Major River Basins. Global Runoff Data Centre. 2nd, rev. ed. Koblenz: Federal Institute of Hydrology (BfG)."
    )

    return basins_gdf


def get_grdc_wmo_basins(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Retrieves WMO Basins and Sub-Basins dataset.

    This function downloads and loads the Global Runoff Data Centre's (GRDC) WMO Basins
    and Sub-Basins dataset. It contains 515 WMO Basins representing hydrographic regions
    including river/lake basins with both exorheic drainage (flowing to oceans) and
    endorheic drainage (inland sinks/lakes).

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    **kwargs
        Additional keyword arguments passed to ``geopandas.read_file`` (e.g.
        ``columns=[...]``, ``rows=...``, ``engine="pyogrio"``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the WMO Basins and Sub-Basins with associated attributes.

    Examples
    --------
    Get all WMO basins...

    >>> basins = get_wmo_basins_and_subbasins()
    >>> basins.plot()

    Get basins for a specific region...

    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> regional_basins = get_wmo_basins_and_subbasins(bbox_input=bbox)
    >>> regional_basins.plot()

    Notes
    -----
    The GRDC archive (about 380 MB) is downloaded once into the easysnowdata
    cache directory (``~/.cache/easysnowdata/grdc`` on Linux; override with
    ``EASYSNOWDATA_CACHE_DIR``) and re-used on later calls.

    This dataset incorporates data from the HydroSHEDS database which is © World Wildlife Fund, Inc.
    (2006-2013) and has been used under license.

    WMO basins and sub-basins are attributed with:
    - WMOBB: identifier of hydrographic region
    - WMOBB_NAME: name of hydrographic region
    - WMOBB_BASIN: name of river/lake basin, coastal region or island
    - WMOBB_SUBBASIN: name of river/lake basin forming a separate sub-basin
    - WMOBB_DESCRIPTION: description of hydrographic region
    - REGNUM: number of the WMO Region (Regional Association)
    - REGNAME: name of the WMO Region (Regional Association)
    - WMO306_MoC_NUM: reference to Manual on Codes, 2-digit basin code
    - WMO306_MoC_REFERENCE: reference to Manual on Codes, name of basin/sub-basin
    - SUMSUBAREA: approximate of drainage area (in square km)

    Data citation:
    GRDC (2020): WMO Basins and Sub-Basins / Global Runoff Data Centre, GRDC. 3rd, rev. ext. ed.
    Koblenz, Germany: Federal Institute of Hydrology (BfG).
    """

    url = "https://grdc.bafg.de/downloads/wmobb_json.zip"

    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = (
        convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    )

    # The GRDC server answers HTTP 400 to HEAD requests, which GDAL's /vsicurl
    # sends before any range read, so a remote "zip+https://" read fails even
    # though the file is there. Fetch the archive once with a plain GET into
    # the user cache directory (~380 MB) and read the basins layer locally.
    zip_path = providers.raster_http.fetch(url, "wmobb_json.zip", subdir="grdc")
    basins_gdf = providers.vector_http.read(
        f"zip://{zip_path}!wmobb_basins.json", **kwargs
    )

    # Clip to bbox if provided
    if bbox_gdf is not None:
        basins_gdf = basins_gdf.clip(bbox_gdf)
    else:
        _logger.info("No spatial subsetting because bbox_input was not provided.")

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = (
        "GRDC (2020): WMO Basins and Sub-Basins / Global Runoff Data Centre, GRDC. 3rd, rev. ext. ed. Koblenz, Germany: Federal Institute of Hydrology (BfG)."
    )

    return basins_gdf


@deprecated(
    "easysnowdata.climate.era5.load",
    since="0.1.0",
    remove_in="0.2.0",
    extra=(
        "The new loader takes aoi= and time= (any AOI/time form), serves hourly ERA5 "
        "from ARCO-ERA5 without Earth Engine credentials, and returns the standard attrs."
    ),
)
def get_era5(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    version: str = "ERA5",
    cadence: str = "HOURLY",
    source: str = "auto",  # "auto", "GEE", or "GCS"
    start_date: str | None = None,
    end_date: str | None = None,
    variables: str | list | None = None,
    initialize_ee: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Deprecated alias of :func:`easysnowdata.climate.era5.load`.

    Every old keyword still works for one release. ``initialize_ee`` is
    accepted and ignored (Earth Engine is initialised on first use through
    ``easysnowdata.auth``), and ``source`` still takes ``"auto"``, ``"GEE"``
    or ``"GCS"``.
    """
    from easysnowdata.climate import era5 as _era5  # noqa: PLC0415

    source_map = {"AUTO": None, "GEE": "gee", "GCS": "arco-era5-gcs"}
    resolved = source_map.get(str(source).upper(), source)
    time = None if (start_date is None and end_date is None) else (start_date, end_date)
    return _era5.load(
        bbox_input,
        time,
        variables=variables,
        source=resolved,
        version=version,
        cadence=cadence,
        **kwargs,
    )


@deprecated(
    "easysnowdata.snow.snodas.load",
    since="0.1.0",
    remove_in="0.2.0",
    extra=(
        "The new loader defaults to the authoritative NSIDC G02158 archive, which "
        "needs no Earth Engine account; pass source='gee-climate-engine' for the "
        "mirror this function used."
    ),
)
def get_snodas(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    start_date: str = "2003-10-01",
    end_date: str = None,
    variables: str | list | None = None,
    initialize_ee: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Deprecated alias of :func:`easysnowdata.snow.snodas.load`.

    Keeps the Earth Engine route this function has always used, so the data
    and the credentials are unchanged; ``initialize_ee`` is accepted and
    ignored. The new default route (``source="nsidc"``) needs no account.
    """
    from easysnowdata.snow import snodas as _snodas  # noqa: PLC0415

    return _snodas.load(
        bbox_input,
        (start_date, end_date if end_date is not None else temporal.today()),
        variables=variables,
        source="gee-climate-engine",
        **kwargs,
    )


@requires_earthaccess
def get_ucla_snow_reanalysis(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    variable: str = "SWE_Post",
    stats: str = "mean",
    start_date: str = "1984-10-01",
    end_date: str = "2021-09-30",
    **kwargs,
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
    **kwargs
        Additional keyword arguments passed to ``xarray.open_mfdataset`` (e.g.
        ``chunks={"Day": 30}`` or ``parallel=True``).

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
    Requires NASA EarthData credentials: ``EARTHDATA_TOKEN``, or
    ``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD``, or a ``~/.netrc`` entry
    (``earthaccess.login(persist=True)`` writes one). The function logs in
    through ``earthaccess`` itself before opening any file.

    Data citation:

    Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2
    """

    if stats not in _UCLA_SR_STATS_INDEX:
        raise ValueError(
            f"stats must be one of {list(_UCLA_SR_STATS_INDEX)}, got {stats!r}."
        )
    stats_index = _UCLA_SR_STATS_INDEX[stats]

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    # The earthdata provider logs in explicitly (earthaccess >= 0.16) first.
    search = providers.earthdata.search(
        "WUS_UCLA_SR",
        cloud_hosted=True,
        bounding_box=tuple(bbox_gdf.total_bounds),
        temporal=(start_date, end_date),
    )

    files = providers.earthdata.open(
        search
    )  # cant disable progress bar yet https://github.com/nsidc/earthaccess/issues/612
    snow_reanalysis_ds = xr.open_mfdataset(files, **kwargs).transpose()

    # Each file holds one water year of daily data starting 1 October. Take the
    # year from the file name (..._WY1999_...); fall back to a date embedded in
    # the archive path (.../1999/10/01/... or the older 1999.10.01 form).
    url = files[0].path
    if match := re.search(r"_WY(\d{4})_", url):
        WY_start_date = pd.Timestamp(year=int(match.group(1)), month=10, day=1)
    elif match := re.search(r"(\d{4})[./](\d{2})[./](\d{2})", url):
        WY_start_date = pd.Timestamp(*(int(g) for g in match.groups()))
    else:
        raise ValueError(
            f"Could not determine the water-year start date from file path: {url}"
        )

    snow_reanalysis_ds.coords["time"] = (
        "Day",
        pd.date_range(WY_start_date, periods=snow_reanalysis_ds.sizes["Day"]),
    )
    snow_reanalysis_ds = snow_reanalysis_ds.swap_dims({"Day": "time"})

    snow_reanalysis_ds = snow_reanalysis_ds.sel(time=slice(start_date, end_date))

    snow_reanalysis_da = snow_reanalysis_ds[variable].sel(Stats=stats_index)
    snow_reanalysis_da = snow_reanalysis_da.rio.set_spatial_dims(
        x_dim="Longitude", y_dim="Latitude"
    )
    snow_reanalysis_da = snow_reanalysis_da.rio.write_crs(bbox_gdf.crs)
    snow_reanalysis_da = snow_reanalysis_da.rio.clip_box(
        *bbox_gdf.total_bounds, crs=bbox_gdf.crs
    )

    snow_reanalysis_da.attrs["data_citation"] = (
        "Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2"
    )

    return snow_reanalysis_da


@deprecated(
    "easysnowdata.climate.koppen_geiger.load",
    since="0.1.0",
    remove_in="0.2.0",
    extra=(
        "The new loader adds period= and scenario= (the archive's other 30-year periods "
        "and CMIP6 projections) and returns CF flag attrs instead of class_info/cmap/"
        "example_plot; plot it with easysnowdata.plotting.categorical()."
    ),
)
def get_koppen_geiger_classes(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    resolution: str = "0.1 degree",
    **kwargs,
) -> xr.DataArray:
    """Deprecated alias of :func:`easysnowdata.climate.koppen_geiger.load`.

    Returns the 1991-2020 classes at *resolution*, as before. The
    ``class_info``, ``cmap`` and ``example_plot`` attrs are gone (design
    contract §2.5: nothing in ``.attrs`` is a Python object); the classes are
    now CF ``flag_values`` / ``flag_meanings`` / ``flag_colors`` and
    ``easysnowdata.plotting.categorical`` draws the legend.
    """
    from easysnowdata.climate import koppen_geiger as _koppen  # noqa: PLC0415

    return _koppen.load(bbox_input, resolution=resolution, **{"chunks": None, **kwargs})


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
