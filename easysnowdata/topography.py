"""Deprecated shims for the terrain products (§3.4, §11).

The loaders moved to :mod:`easysnowdata.terrain`:

* ``get_copernicus_dem`` → :func:`easysnowdata.terrain.dem.load`
* ``get_chili`` → :func:`easysnowdata.terrain.chili.load`

Every name here keeps working for one minor release and emits an
:class:`~easysnowdata._deprecation.EasysnowdataDeprecationWarning` on first use.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the ``.rio`` accessor used below)
import shapely
import xarray as xr

from easysnowdata._deprecation import deprecated
from easysnowdata.terrain import chili, dem

__all__ = ["get_copernicus_dem", "get_chili"]

_logger = logging.getLogger(__name__)


@deprecated(
    "easysnowdata.terrain.dem.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.topography.get_copernicus_dem",
)
def get_copernicus_dem(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    resolution: int = 30,
    **kwargs,
) -> xr.DataArray:
    """Fetch the Copernicus Global DEM for a bounding box.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.terrain.dem.load`, which takes any AOI form,
        offers the unsigned AWS route (``source="earth-search"``) and masks the
        -32767 sentinel to NaN by default.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.geometry, optional
        Spatial extent. Tuple should be ``(xmin, ymin, xmax, ymax)`` in
        EPSG:4326. Defaults to global extent if ``None``.
    resolution : int, optional
        DEM resolution in metres. Either ``30`` or ``90``. Default is ``30``.
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.terrain.dem.load` (and on to ``odc.stac.load``).

    Returns
    -------
    xarray.DataArray
        Elevation DataArray in metres (EPSG:4326).

    Raises
    ------
    ValueError
        If *resolution* is not ``30`` or ``90``.

    Notes
    -----
    Data citation:
        European Space Agency, Sinergise (2021). Copernicus Global Digital
        Elevation Model. Distributed by OpenTopography.
        https://doi.org/10.5069/G9028PQB
    """
    return dem.load(bbox_input, resolution=resolution, **kwargs)


@deprecated(
    "easysnowdata.terrain.chili.load",
    since="0.1.0",
    remove_in="0.3.0",
    name="easysnowdata.topography.get_chili",
    extra=(
        "The new loader returns native values (normalize='minmax' keeps this "
        "AOI-relative rescaling) and names its dims latitude/longitude."
    ),
)
def get_chili(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    initialize_ee: bool = True,
    **kwargs,
) -> xr.DataArray:
    """Fetch CHILI (Continuous Heat-Insolation Load Index) for a bounding box.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.terrain.chili.load`. It returns the native
        values by default; pass ``normalize="minmax"`` for the AOI-relative
        rescaling this function applied, or ``normalize="index"`` for the 0-1
        index.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.geometry, optional
        Spatial extent. Defaults to global extent if ``None``.
    initialize_ee : bool, optional
        Ignored: Earth Engine is initialised on first use (§2.8).
    **kwargs
        Additional keyword arguments passed to
        :func:`easysnowdata.terrain.chili.load` (and on to
        ``xarray.open_dataset(engine="ee")``).

    Returns
    -------
    xarray.DataArray
        CHILI DataArray, min-max normalised to [0, 1] within the AOI.

    Notes
    -----
    Requires Google Earth Engine authentication; see ``esd.auth.status()``.

    Data are only available between 70°N and 70°S.

    Data citation:
        Theobald, D.M., Harrison-Atlas, D., Monahan, W.B., Albano, C.M. (2015).
        Ecologically-Relevant Maps of Landforms and Physiographic Diversity for
        Climate Adaptation Planning. PLoS ONE 10(12): e0143619.
        https://doi.org/10.1371/journal.pone.0143619
    """
    return chili.load(bbox_input, normalize="minmax", **kwargs)
