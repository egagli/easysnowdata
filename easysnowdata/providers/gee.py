"""Google Earth Engine via ``xee``: open at the native grid, features → GeoDataFrame.

``import ee`` and ``import xee`` are deferred so a broken Google auth stack
cannot break ``import easysnowdata``. Earth Engine is initialised once through
:mod:`easysnowdata.auth`; Dask workers re-initialise themselves through
``ee_init_if_necessary`` / ``ee_init_kwargs``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import geopandas as gpd

from easysnowdata import auth
from easysnowdata.aoi import parse_aoi

__all__ = [
    "ensure",
    "ee",
    "grid_params",
    "open_dataset",
    "features_to_geodataframe",
    "geometry",
]

_logger = logging.getLogger(__name__)


def ensure(**kwargs: Any) -> None:
    """Initialise Earth Engine once (see the ``earthengine`` auth provider)."""
    auth.get("earthengine").ensure(**kwargs)


def ee() -> Any:
    """Return the initialised ``ee`` module."""
    ensure()
    import ee as _ee  # noqa: PLC0415

    return _ee


def geometry(aoi: Any) -> Any:
    """An ``ee.Geometry`` for the AOI footprint (EPSG:4326, geodesic=False)."""
    parsed = parse_aoi(aoi)
    _ee = ee()
    return _ee.Geometry(parsed.stac_intersects, "EPSG:4326", False)


def grid_params(ee_obj: Any, aoi: Any = None) -> dict[str, Any]:
    """Pixel-grid kwargs for ``xarray.open_dataset(engine="ee")``.

    xee ≥ 0.1 needs ``crs``, ``crs_transform`` and ``shape_2d``. They come from
    the object's *native* grid (first band of the first image for a
    collection), cropped to the smallest block of native pixels that covers
    *aoi* so the returned values are exact native pixels, not a resample.
    """
    from xee import helpers as xee_helpers  # noqa: PLC0415

    native = xee_helpers.extract_grid_params(ee_obj)
    if aoi is None:
        return dict(native)
    parsed = parse_aoi(aoi)
    if parsed.is_global:
        return dict(native)

    a, b, c, d, e, f = native["crs_transform"][:6]
    if b or d:
        raise ValueError("Rotated Earth Engine grids are not supported.")

    # Densify the outline so curved edges survive reprojection to projected CRSs.
    footprint = gpd.GeoSeries([parsed.unwrapped_geometry], crs="EPSG:4326")
    xmin0, ymin0, xmax0, ymax0 = footprint.total_bounds
    seg = max(xmax0 - xmin0, ymax0 - ymin0) / 100 or 1.0
    x_min, y_min, x_max, y_max = (
        footprint.segmentize(seg).to_crs(native["crs"]).total_bounds
    )

    eps = 1e-9  # tolerate float noise when an edge sits exactly on a pixel boundary
    cols = ((x_min - c) / a, (x_max - c) / a)
    rows = ((y_min - f) / e, (y_max - f) / e)
    col0 = math.floor(min(cols) + eps)
    col1 = math.ceil(max(cols) - eps)
    row0 = math.floor(min(rows) + eps)
    row1 = math.ceil(max(rows) - eps)
    # Pixels outside the asset footprint come back as NaN, so the grid is not
    # clamped to the native extent; just guarantee >= 1 pixel.
    col1 = max(col1, col0 + 1)
    row1 = max(row1, row0 + 1)
    return {
        "crs": native["crs"],
        "crs_transform": (a, 0.0, c + col0 * a, 0.0, e, f + row0 * e),
        "shape_2d": (col1 - col0, row1 - row0),
    }


def open_dataset(
    collection: Any,
    aoi: Any = None,
    *,
    grid: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """``xarray.open_dataset(engine="ee")`` at the native grid cropped to *aoi*.

    *collection* is an ``ee.ImageCollection`` (an ``ee.Image`` is wrapped).
    Extra keyword arguments go to ``open_dataset`` and take precedence; the
    Earth Engine init kwargs are attached for Dask workers.
    """
    import xarray as xr  # noqa: PLC0415

    _ee = ee()
    if isinstance(collection, _ee.Image):
        collection = _ee.ImageCollection(collection)
    if grid is None:
        grid = grid_params(
            collection.first() if hasattr(collection, "first") else collection, aoi
        )
    params: dict[str, Any] = {
        "engine": "ee",
        "ee_init_if_necessary": True,
        "ee_init_kwargs": auth.get("earthengine").xee_init_kwargs(),
        **grid,
    }
    params.update(kwargs)
    return xr.open_dataset(collection, **params)


def features_to_geodataframe(
    collection: Any, aoi: Any = None, *, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """An ``ee.FeatureCollection`` (or asset id) → GeoDataFrame, filtered to *aoi*."""
    _ee = ee()
    fc = (
        _ee.FeatureCollection(collection) if isinstance(collection, str) else collection
    )
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            fc = fc.filterBounds(geometry(parsed))
    info = fc.getInfo()
    gdf = gpd.GeoDataFrame.from_features(info.get("features", []), crs=crs)
    return gdf
