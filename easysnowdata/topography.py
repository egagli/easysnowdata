"""Access digital elevation models and topographic indices.

Currently supported datasets:

* **Copernicus DEM** (30 m / 90 m) via Microsoft Planetary Computer
* **CHILI** — Continuous Heat-Insolation Load Index via Google Earth Engine
"""

from __future__ import annotations

import logging

import ee
import geopandas as gpd
import odc.stac
import planetary_computer
import pystac_client
import rioxarray as rxr
import shapely
import xarray as xr

odc.stac.configure_rio(cloud_defaults=True)

from easysnowdata.utils import convert_bbox_to_geodataframe, get_stac_cfg, requires_earthengine

__all__ = ["get_copernicus_dem", "get_chili"]

_logger = logging.getLogger(__name__)


def get_copernicus_dem(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    resolution: int = 30,
) -> xr.DataArray:
    """Fetch the Copernicus Global DEM for a bounding box.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.geometry, optional
        Spatial extent. Tuple should be ``(xmin, ymin, xmax, ymax)`` in
        EPSG:4326. Defaults to global extent if ``None``.
    resolution : int, optional
        DEM resolution in metres. Either ``30`` or ``90``. Default is ``30``.

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
    The Copernicus DEM is a Digital Surface Model (DSM) derived from the
    WorldDEM with additional editing applied to water bodies and coastlines.

    Data citation:
        European Space Agency, Sinergise (2021). Copernicus Global Digital
        Elevation Model. Distributed by OpenTopography.
        https://doi.org/10.5069/G9028PQB
    """
    if resolution not in (30, 90):
        raise ValueError(
            f"Copernicus DEM is available at 30 m and 90 m only, got {resolution} m."
        )

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[f"cop-dem-glo-{resolution}"], bbox=bbox_gdf.total_bounds
    )
    cop_dem_da = odc.stac.load(
        search.items(), bbox=bbox_gdf.total_bounds, chunks={}
    )["data"].squeeze()
    cop_dem_da = cop_dem_da.rio.write_nodata(-32767, encoded=True)

    cop_dem_da.attrs["data_citation"] = (
        "European Space Agency, Sinergise (2021). Copernicus Global Digital "
        "Elevation Model. Distributed by OpenTopography. "
        "https://doi.org/10.5069/G9028PQB. Accessed: 2024-03-18"
    )
    return cop_dem_da


@requires_earthengine
def get_chili(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    initialize_ee: bool = True,
) -> xr.DataArray:
    """Fetch CHILI (Continuous Heat-Insolation Load Index) for a bounding box.

    CHILI is a topographic index quantifying the combined effect of solar
    radiation and surface temperature, derived from ALOS World 3D-30m (AW3D30).
    Values range 0–1: warm (> 0.767), neutral (0.448–0.767), cool (< 0.448).

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.geometry, optional
        Spatial extent. Defaults to global extent if ``None``.
    initialize_ee : bool, optional
        Initialise Earth Engine before fetching. Default ``True``. Set to
        ``False`` if EE is already initialised in the calling script.

    Returns
    -------
    xarray.DataArray
        CHILI DataArray, min–max normalised to [0, 1].

    Notes
    -----
    Requires Google Earth Engine authentication. Run ``ee.Authenticate()`` and
    ``ee.Initialize()`` once, or call ``easysnowdata.authenticate_all()``.

    Data are only available between 70°N and 70°S.

    Data citation:
        Theobald, D.M., Harrison-Atlas, D., Monahan, W.B., Albano, C.M. (2015).
        Ecologically-Relevant Maps of Landforms and Physiographic Diversity for
        Climate Adaptation Planning. PLoS ONE 10(12): e0143619.
        https://doi.org/10.1371/journal.pone.0143619
    """
    if initialize_ee:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    image = ee.Image("CSP/ERGo/1_0/Global/ALOS_CHILI")
    crs = image.projection().getInfo()["crs"]
    transform = image.projection().getInfo()["transform"]

    chili_da = (
        xr.open_dataset(
            ee.ImageCollection(image),
            engine="ee",
            geometry=tuple(bbox_gdf.total_bounds),
            projection=ee.Projection(crs=crs, transform=transform),
        )
        .drop_vars("time")
        .squeeze()["constant"]
        .squeeze()
        .transpose()
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    )
    chili_da = chili_da.rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs)

    if chili_da.isnull().all().item():
        _logger.warning(
            "No CHILI data for this location. CHILI is only available 70°N–70°S."
        )

    chili_da = (chili_da - chili_da.min()) / (chili_da.max() - chili_da.min())
    chili_da.attrs["data_citation"] = (
        "Theobald, D.M., Harrison-Atlas, D., Monahan, W.B., Albano, C.M. (2015). "
        "Ecologically-Relevant Maps of Landforms and Physiographic Diversity for "
        "Climate Adaptation Planning. PLoS ONE 10(12): e0143619. "
        "https://doi.org/10.1371/journal.pone.0143619"
    )
    return chili_da
