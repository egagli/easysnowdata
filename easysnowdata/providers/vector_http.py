"""Remote vectors (GDB, GeoJSON, zipped shapefiles, GeoParquet) with pyogrio pushdown."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

from easysnowdata import auth
from easysnowdata._gdal import gdal_env
from easysnowdata.aoi import parse_aoi

__all__ = ["read", "read_parquet", "zip_path"]

_logger = logging.getLogger(__name__)


def zip_path(url_or_path: str | Path, member: str | None = None) -> str:
    """GDAL/fiona-style zip path: ``zip+https://…!member`` or ``zip:///local.zip!member``."""
    text = str(url_or_path)
    prefix = "zip+" if "://" in text else "zip://"
    base = f"{prefix}{text}"
    return f"{base}!{member.lstrip('/')}" if member else base


def read(
    path: str | Path,
    aoi: Any = None,
    *,
    layer: str | None = None,
    columns: list[str] | None = None,
    engine: str = "pyogrio",
    use_arrow: bool | None = None,
    requires: tuple[str, ...] = (),
    gdal: dict[str, Any] | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """``geopandas.read_file`` with AOI pushdown inside the read environment.

    With an AOI and ``clip=True`` the footprint becomes ``mask=`` (exact) —
    or ``bbox=`` when the footprint is a plain rectangle — so only
    intersecting features are read. Extra keyword arguments (``rows=``,
    ``where=``, ``bbox=``, ``mask=``…) go to ``read_file`` and take precedence.
    GeoParquet paths are routed to :func:`read_parquet`.
    """
    text = str(path)
    if text.lower().endswith((".parquet", ".geoparquet", ".pq")):
        return read_parquet(text, aoi, columns=columns, **kwargs)
    params: dict[str, Any] = {"engine": engine}
    if layer is not None:
        params["layer"] = layer
    if columns is not None:
        params["columns"] = columns
    if use_arrow is not None:
        params["use_arrow"] = use_arrow
    if aoi is not None and "mask" not in kwargs and "bbox" not in kwargs:
        parsed = parse_aoi(aoi)
        if parsed.clip and not parsed.is_global:
            geom = parsed.geometry
            if geom.geom_type == "Polygon" and geom.equals(geom.envelope):
                params["bbox"] = tuple(geom.bounds)
            else:
                params["mask"] = parsed.footprint
    params.update(kwargs)
    with auth.env(*requires), gdal_env(**(gdal or {})):
        return gpd.read_file(text, **params)


def _primary_geometry_column(path: str | Path) -> str:
    """The GeoParquet primary geometry column (from the ``geo`` metadata), default ``geometry``."""
    try:
        import json  # noqa: PLC0415

        import pyarrow.parquet as pq  # noqa: PLC0415

        meta = pq.read_schema(str(path)).metadata or {}
        return json.loads(meta[b"geo"])["primary_column"]
    except Exception:  # noqa: BLE001 — remote or non-GeoParquet: fall back
        return "geometry"


def read_parquet(
    path: str | Path,
    aoi: Any = None,
    *,
    columns: list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """``geopandas.read_parquet`` with ``bbox`` pushdown from the AOI (GeoParquet 1.1 bbox column)."""
    if columns is not None:
        geometry_column = _primary_geometry_column(path)
        if geometry_column not in columns:
            columns = [*columns, geometry_column]
    params: dict[str, Any] = {
        "columns": columns,
        "storage_options": storage_options,
        **kwargs,
    }
    if aoi is not None and "bbox" not in kwargs:
        parsed = parse_aoi(aoi)
        if parsed.clip and not parsed.is_global:
            params["bbox"] = tuple(parsed.footprint.total_bounds)
    params = {k: v for k, v in params.items() if v is not None}
    gdf = gpd.read_parquet(str(path), **params)
    if aoi is not None and "bbox" in params:
        parsed = parse_aoi(aoi)
        if parsed.geometry.geom_type != "Polygon" or not parsed.geometry.equals(
            parsed.geometry.envelope
        ):
            gdf = gdf[gdf.intersects(parsed.geometry)]
    return gdf
