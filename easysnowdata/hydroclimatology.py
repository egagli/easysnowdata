"""Access hydroclimatology datasets: ERA5, SNODAS, UCLA reanalysis, basin geometries, and more.

The basin loaders moved to :mod:`easysnowdata.hydro.basins` (§3.4): the names
here keep working for one minor release and warn on first use.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import shapely
import xarray as xr

from easysnowdata import temporal
from easysnowdata._deprecation import deprecated
from easysnowdata.hydro import basins
from easysnowdata.snow import ucla_sr as _ucla_sr_stats

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
# WUS_UCLA_SR files. The table now lives in easysnowdata.snow.ucla_sr.STATS;
# this module-level alias keeps the old private constant working for one
# release (Phase 0 fixed it: "median" and "25pct" used to share index 2).
_UCLA_SR_STATS_INDEX = dict(_ucla_sr_stats.STATS)

# Position of each ensemble statistic along the ``Stats`` dimension of the
# WUS_UCLA_SR files: mean, standard deviation, median, 25th and 75th percentile.
# The table now lives in easysnowdata.snow.ucla_sr.STATS; this alias keeps the
# old module constant working for one release.


@deprecated(
    "easysnowdata.hydro.basins.huc",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.hydroclimatology.get_huc_geometries",
    extra=(
        "The new loader defaults to the public USGS WBD REST service, so HUC "
        "boundaries no longer need Earth Engine, and the citation is in the "
        "data_citation attr."
    ),
)
def get_huc_geometries(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    huc_level: str = "02",
) -> gpd.GeoDataFrame:
    """
    Retrieves Hydrologic Unit Code (HUC) geometries within a specified bounding box and HUC level.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.hydro.basins.huc`. Its default source is the
        USGS Watershed Boundary Dataset REST service (no credentials, newer
        than the 2017 Earth Engine snapshot); pass ``source="gee"`` for the
        Earth Engine route this function used.

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

    Notes
    -----
    Data citation:
    Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022,
    Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.):
    U.S. Geological Survey Techniques and Methods 11-A3, 54 p.,
    https://doi.org/10.3133/tm11A3
    """
    return basins.huc(bbox_input, level=huc_level)


@deprecated(
    "easysnowdata.hydro.basins.hydrobasins",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.hydroclimatology.get_hydroBASINS",
)
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

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.hydro.basins.hydrobasins`, which adds the
        lighter per-region HydroSHEDS zips (``source="hydrosheds"``) and the
        Earth Engine route.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    level : int, optional
        The hierarchical level (1-12) of sub-basin delineation. Higher levels represent
        finer subdivisions. Default is 5.
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.hydro.basins.hydrobasins` (and on to
        ``geopandas.read_file``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the HydroATLAS sub-basin boundaries with associated attributes.

    Notes
    -----
    Data citation:
    Linke, S., Lehner, B., Ouellet Dallaire, C., Ariwi, J., Grill, G., Anand, M., Beames, P.,
    Burchard-Levine, V., Maxwell, S., Moidu, H., Tan, F., Thieme, M. (2019). Global hydro-
    environmental sub-basin and river reach characteristics at high spatial resolution.
    Scientific Data 6: 283. doi: 10.1038/s41597-019-0300-6
    """
    return basins.hydrobasins(bbox_input, level=level, **kwargs)


@deprecated(
    "easysnowdata.hydro.basins.grdc_major",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.hydroclimatology.get_grdc_major_river_basins_of_the_world",
    extra="Basins that intersect the AOI are returned whole, not cut at its edge.",
)
def get_grdc_major_river_basins_of_the_world(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Retrieves GRDC Major River Basins of the World dataset.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.hydro.basins.grdc_major`. It returns the
        intersecting basins whole instead of clipping them to the AOI.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.hydro.basins.grdc_major` (and on to
        ``geopandas.read_file``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the GRDC major river basins with associated attributes.

    Notes
    -----
    Data citation:
    GRDC (2020): GRDC Major River Basins. Global Runoff Data Centre. 2nd, rev. ed.
    Koblenz: Federal Institute of Hydrology (BfG).
    """
    return basins.grdc_major(bbox_input, **kwargs)


@deprecated(
    "easysnowdata.hydro.basins.grdc_wmo",
    since="0.1.0",
    remove_in="0.2.0",
    name="easysnowdata.hydroclimatology.get_grdc_wmo_basins",
    extra="Basins that intersect the AOI are returned whole, not cut at its edge.",
)
def get_grdc_wmo_basins(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Retrieves WMO Basins and Sub-Basins dataset.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.hydro.basins.grdc_wmo`. It returns the
        intersecting basins whole instead of clipping them to the AOI.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.hydro.basins.grdc_wmo` (and on to
        ``geopandas.read_file``).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the WMO Basins and Sub-Basins with associated attributes.

    Notes
    -----
    The GRDC archive is downloaded once into the easysnowdata cache directory
    (``~/.cache/easysnowdata/grdc`` on Linux; override with
    ``EASYSNOWDATA_CACHE_DIR``) and re-used on later calls.

    Data citation:
    GRDC (2020): WMO Basins and Sub-Basins / Global Runoff Data Centre, GRDC. 3rd, rev. ext. ed.
    Koblenz, Germany: Federal Institute of Hydrology (BfG).
    """
    return basins.grdc_wmo(bbox_input, **kwargs)


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


@deprecated(
    "easysnowdata.snow.ucla_sr.load",
    since="0.1.0",
    remove_in="0.2.0",
    extra=(
        "The new loader adds region='hma' (High Mountain Asia), virtualize='auto' for "
        "long series and access='auto' for in-region S3 reads."
    ),
)
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
    """Deprecated alias of :func:`easysnowdata.snow.ucla_sr.load`.

    Every old keyword still works for one release. The ensemble-statistic
    mapping (``_UCLA_SR_STATS_INDEX``) now lives in
    ``easysnowdata.snow.ucla_sr.STATS``.
    """
    from easysnowdata.snow import ucla_sr as _ucla_sr  # noqa: PLC0415

    return _ucla_sr.load(
        bbox_input,
        (start_date, end_date),
        variable=variable,
        stats=stats,
        **kwargs,
    )


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
