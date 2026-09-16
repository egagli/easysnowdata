"""Area of interest (AOI) parsing — the one spatial input of every loader.

Design contract §2.1: a loader accepts ``aoi`` as a ``(west, south, east,
north)`` tuple in EPSG:4326, a shapely geometry (EPSG:4326), a GeoJSON-like
mapping, a :class:`geopandas.GeoDataFrame` / :class:`geopandas.GeoSeries` in
any CRS, an :class:`odc.geo.geobox.GeoBox`, or ``None`` for the whole globe.
:func:`parse_aoi` turns each of these into an :class:`AOI`: a footprint
(single-row GeoDataFrame in EPSG:4326) plus, on request, a target grid
(:class:`~odc.geo.geobox.GeoBox`) in a UTM zone or any other CRS.

Antimeridian
------------
A bounding box whose ``west`` is greater than ``east`` (``(170, -20, -170,
-10)``) crosses the antimeridian. The footprint is stored as a MultiPolygon
split at ±180° (what STAC ``intersects`` and GDAL expect); :attr:`AOI.bounds`
keeps the STAC convention (``west > east``); grids are built on the
*unwrapped* geometry (longitudes 170 → 190) so a UTM or geographic GeoBox is
one contiguous raster.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
import shapely.ops
from odc.geo import Resolution, res_, resxy_
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pyproj import CRS

__all__ = [
    "AOI",
    "parse_aoi",
    "estimate_utm_crs",
    "split_antimeridian",
    "WGS84",
    "WORLD_BOUNDS",
]

_logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
_ANTIMERIDIAN = shapely.LineString([(180.0, -90.0), (180.0, 90.0)])
# Polar stereographic fallbacks for AOIs outside UTM coverage (84°N / 80°S)
_NORTH_POLAR = "EPSG:3413"
_SOUTH_POLAR = "EPSG:3031"


# ── helpers ───────────────────────────────────────────────────────────────────


def _as_crs(crs: Any) -> CRS:
    return crs if isinstance(crs, CRS) else CRS.from_user_input(crs)


def _unwrap(geom: shapely.Geometry) -> shapely.Geometry:
    """Shift the western hemisphere by +360° so a split AOI is contiguous."""
    return shapely.transform(
        geom,
        lambda xy: np.column_stack(
            [np.where(xy[:, 0] < 0, xy[:, 0] + 360, xy[:, 0]), xy[:, 1]]
        ),
    )


def split_antimeridian(geom: shapely.Geometry) -> shapely.Geometry:
    """Split a geometry given in *unwrapped* longitudes (0…360) at 180°.

    Parts east of the antimeridian are shifted back into −180…0. The result
    is a MultiPolygon when the geometry straddles 180°, otherwise the input
    with longitudes normalised to −180…180.
    """
    pieces = shapely.get_parts(shapely.ops.split(geom, _ANTIMERIDIAN))
    polygons: list[shapely.Polygon] = []
    for piece in pieces:
        if piece.is_empty:
            continue
        if piece.bounds[0] >= 180.0 - 1e-9:  # east of the antimeridian → back to −180…0
            piece = shapely.affinity.translate(piece, xoff=-360.0)
        polygons.extend(p for p in shapely.get_parts(piece) if p.geom_type == "Polygon")
    if len(polygons) == 1:
        return polygons[0]
    return shapely.MultiPolygon(polygons)


def _wrapped_footprint_from_projected(geom_4326: shapely.Geometry) -> shapely.Geometry:
    """Repair a footprint that came out of a projected CRS across ±180°.

    After ``to_crs("EPSG:4326")`` such a polygon appears to span the whole
    globe (longitudes jump from 179 to −179). Unwrapping and re-splitting gives
    the intended two-part footprint.
    """
    minx, _, maxx, _ = geom_4326.bounds
    if maxx - minx <= 180.0:
        return geom_4326
    return split_antimeridian(_unwrap(geom_4326))


def estimate_utm_crs(geometry: shapely.Geometry) -> CRS:
    """Return the UTM CRS for the centroid of a geometry in EPSG:4326.

    Works across the antimeridian (the geometry may be a MultiPolygon split at
    180°) and falls back to polar stereographic projections (EPSG:3413 north
    of 84°N, EPSG:3031 south of 80°S) where UTM is not defined.
    """
    centroid = (
        _unwrap(geometry).centroid
        if geometry.geom_type == "MultiPolygon"
        else geometry.centroid
    )
    lon, lat = centroid.x, centroid.y
    if lon > 180.0:
        lon -= 360.0
    if lat > 84.0:
        return CRS.from_user_input(_NORTH_POLAR)
    if lat < -80.0:
        return CRS.from_user_input(_SOUTH_POLAR)
    try:
        return gpd.GeoSeries([shapely.Point(lon, lat)], crs=WGS84).estimate_utm_crs()
    except RuntimeError:  # pragma: no cover — pyproj database quirks
        return CRS.from_user_input(_NORTH_POLAR if lat >= 0 else _SOUTH_POLAR)


def _wants_utm(crs: Any) -> bool:
    return crs is None or (isinstance(crs, str) and crs.lower() == "utm")


def _resolution(res: Any) -> Resolution:
    if isinstance(res, Resolution):
        return res
    if isinstance(res, (tuple, list)) and len(res) == 2:
        return resxy_(float(res[0]), float(res[1]))
    return res_(float(res))


def _bbox_geometry(
    bounds: tuple[float, float, float, float],
) -> tuple[shapely.Geometry, bool]:
    west, south, east, north = (float(v) for v in bounds)
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValueError(
            f"Latitudes must lie in [-90, 90]; got south={south}, north={north}."
        )
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError(
            f"Longitudes must lie in [-180, 180]; got west={west}, east={east}."
        )
    if south >= north:
        raise ValueError(f"south ({south}) must be less than north ({north}).")
    if west > east:  # STAC convention for boxes crossing the antimeridian
        return shapely.MultiPolygon(
            [
                shapely.box(west, south, 180.0, north),
                shapely.box(-180.0, south, east, north),
            ]
        ), True
    return shapely.box(west, south, east, north), False


# ── the AOI ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AOI:
    """A parsed area of interest: footprint in EPSG:4326 plus an optional grid.

    Build one with :func:`parse_aoi`; loaders accept either form.

    Attributes
    ----------
    footprint
        Single-row GeoDataFrame in EPSG:4326. A MultiPolygon when the AOI
        crosses the antimeridian.
    clip
        Whether loaders should clip rasters/vectors to the footprint (``True``)
        or return the covering tiles/granules (``False``).
    geobox
        The native grid when the AOI was built from a GeoBox, else ``None``.
        :meth:`to_geobox` returns it when no other grid is requested.
    source_crs
        CRS of the input (EPSG:4326 for tuples and shapely geometries).
    crosses_antimeridian
        ``True`` when the footprint straddles ±180°.
    """

    footprint: gpd.GeoDataFrame
    clip: bool = True
    geobox: GeoBox | None = None
    source_crs: CRS = field(default_factory=lambda: CRS.from_user_input(WGS84))
    crosses_antimeridian: bool = False

    # -- geometry views --------------------------------------------------------

    @property
    def geometry(self) -> shapely.Geometry:
        """The footprint as one shapely geometry in EPSG:4326."""
        return self.footprint.geometry.iloc[0]

    @property
    def unwrapped_geometry(self) -> shapely.Geometry:
        """The footprint with longitudes in 0…360 when it crosses ±180°."""
        if not self.crosses_antimeridian:
            return self.geometry
        return shapely.union_all(shapely.get_parts(_unwrap(self.geometry)))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in EPSG:4326 (STAC convention: ``west > east`` across ±180°)."""
        if self.crosses_antimeridian:
            west, south, east, north = self.unwrapped_geometry.bounds
            return (west, south, east - 360.0, north)
        return tuple(float(v) for v in self.geometry.bounds)  # type: ignore[return-value]

    @property
    def is_global(self) -> bool:
        """``True`` when the footprint covers the whole globe."""
        return bool(np.allclose(self.geometry.bounds, WORLD_BOUNDS))

    @property
    def stac_bbox(self) -> list[float]:
        """The bounds as a STAC ``bbox`` list."""
        return list(self.bounds)

    @property
    def stac_intersects(self) -> dict[str, Any]:
        """The footprint as a GeoJSON geometry mapping for STAC ``intersects``."""
        return shapely.geometry.mapping(self.geometry)

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        return self.stac_intersects

    # -- CRS and grids ---------------------------------------------------------

    @property
    def utm_crs(self) -> CRS:
        """The UTM zone (or polar stereographic fallback) at the AOI centroid."""
        return estimate_utm_crs(self.geometry)

    def to_crs(self, crs: Any) -> gpd.GeoDataFrame:
        """Reproject the footprint, unwrapping it across ±180° for projected CRSs."""
        target = _as_crs(crs)
        if self.crosses_antimeridian and target.is_projected:
            return gpd.GeoDataFrame(
                geometry=[self.unwrapped_geometry], crs=WGS84
            ).to_crs(target)
        return self.footprint.to_crs(target)

    def total_bounds(self, crs: Any = WGS84) -> tuple[float, float, float, float]:
        """Rectangular bounds of the footprint in *crs* (unwrapped across ±180°)."""
        target = _as_crs(crs)
        if target.is_geographic:
            west, south, east, north = self.unwrapped_geometry.bounds
            return (float(west), float(south), float(east), float(north))
        return tuple(float(v) for v in self.to_crs(target).total_bounds)  # type: ignore[return-value]

    def to_geobox(
        self,
        resolution: float | tuple[float, float] | Resolution | None = None,
        crs: Any = None,
        *,
        shape: int | tuple[int, int] | None = None,
        anchor: Any = "default",
    ) -> GeoBox:
        """Return a grid covering the footprint.

        Parameters
        ----------
        resolution
            Pixel size in CRS units (metres for UTM, degrees for EPSG:4326).
            A scalar gives square pixels with a negative y step.
        crs
            Target CRS. ``None`` or ``"utm"`` selects :attr:`utm_crs`.
        shape
            Alternative to *resolution*: number of pixels (``(ny, nx)`` or one
            value for the longer side).
        anchor
            Passed to :meth:`odc.geo.geobox.GeoBox.from_geopolygon`.

        With no arguments the native GeoBox is returned when the AOI was built
        from one; otherwise *resolution* or *shape* is required.
        """
        if resolution is None and shape is None:
            if self.geobox is not None and self._geobox_matches(crs):
                return self.geobox
            raise ValueError(
                "to_geobox() needs resolution= or shape= (no matching native grid on this AOI)."
            )
        target = self.utm_crs if _wants_utm(crs) else _as_crs(crs)
        polygon = Geometry(self.unwrapped_geometry, crs=WGS84)
        kwargs: dict[str, Any] = {"crs": str(target), "anchor": anchor}
        if resolution is not None:
            kwargs["resolution"] = _resolution(resolution)
        if shape is not None:
            kwargs["shape"] = shape
        return GeoBox.from_geopolygon(polygon, **kwargs)

    def _geobox_matches(self, crs: Any) -> bool:
        """Whether the native GeoBox satisfies a requested *crs* (None = any)."""
        assert self.geobox is not None
        native = CRS.from_user_input(str(self.geobox.crs))
        if crs is None:
            return True
        if _wants_utm(crs):
            return native.is_projected
        return _as_crs(crs) == native

    # -- derived AOIs ----------------------------------------------------------

    def buffer(self, distance: float) -> AOI:
        """Return a new AOI buffered by *distance* metres (computed in UTM)."""
        utm = self.utm_crs
        buffered = self.to_crs(utm).buffer(distance).to_crs(WGS84).geometry.iloc[0]
        if self.crosses_antimeridian or buffered.bounds[2] - buffered.bounds[0] > 180.0:
            buffered = _wrapped_footprint_from_projected(buffered)
        return replace(
            self,
            footprint=gpd.GeoDataFrame(geometry=[buffered], crs=WGS84),
            geobox=None,
            crosses_antimeridian=buffered.geom_type == "MultiPolygon",
        )

    def with_clip(self, clip: bool) -> AOI:
        """Return a copy with the clip flag set to *clip*."""
        return replace(self, clip=bool(clip))

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        w, s, e, n = self.bounds
        grid = f", geobox={self.geobox.shape}" if self.geobox is not None else ""
        return (
            f"AOI(bounds=({w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}), clip={self.clip}{grid})"
        )


# ── parsing ───────────────────────────────────────────────────────────────────


def parse_aoi(
    aoi: Any,
    *,
    clip: bool = True,
    crs: Any = None,
    resolution: float | tuple[float, float] | Resolution | None = None,
) -> AOI:
    """Parse any supported spatial input into an :class:`AOI`.

    Parameters
    ----------
    aoi
        One of: ``(west, south, east, north)`` in EPSG:4326 (``west > east``
        crosses the antimeridian); a shapely geometry or GeoJSON-like mapping
        in EPSG:4326; a GeoDataFrame or GeoSeries in any CRS (rows are
        unioned; a missing CRS is assumed to be EPSG:4326 with a warning); an
        ``odc.geo.geobox.GeoBox``; an existing :class:`AOI`; or ``None`` for
        the whole globe.
    clip
        Stored on the AOI for loaders (§2.1: ``clip=False`` returns covering
        tiles or granules instead of a clipped result).
    crs, resolution
        When both are given, a target GeoBox is attached (see
        :meth:`AOI.to_geobox`). ``crs="utm"`` picks the UTM zone.

    Raises
    ------
    TypeError
        For unsupported input types.
    ValueError
        For malformed bounds or empty geometries.
    """
    if isinstance(aoi, AOI):
        result = aoi.with_clip(clip)
    elif aoi is None:
        result = AOI(
            footprint=gpd.GeoDataFrame(
                geometry=[shapely.box(*WORLD_BOUNDS)], crs=WGS84
            ),
            clip=clip,
        )
    elif isinstance(aoi, GeoBox):
        footprint = aoi.footprint(WGS84, wrapdateline=True).geom
        crosses = (
            footprint.geom_type == "MultiPolygon"
            and footprint.bounds[2] - footprint.bounds[0] > 180.0
        )
        result = AOI(
            footprint=gpd.GeoDataFrame(geometry=[footprint], crs=WGS84),
            clip=clip,
            geobox=aoi,
            source_crs=CRS.from_user_input(str(aoi.crs)),
            crosses_antimeridian=crosses,
        )
    elif isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        series = aoi.geometry if isinstance(aoi, gpd.GeoDataFrame) else aoi
        if series.empty or series.is_empty.all():
            raise ValueError("The AOI GeoDataFrame/GeoSeries has no geometries.")
        source_crs = series.crs
        if source_crs is None:
            _logger.warning("AOI has no CRS; assuming EPSG:4326.")
            series = series.set_crs(WGS84)
            source_crs = series.crs
        geom = series.to_crs(WGS84).union_all()
        crosses = False
        if _as_crs(source_crs).is_projected:
            fixed = _wrapped_footprint_from_projected(geom)
            crosses = fixed is not geom and fixed.geom_type == "MultiPolygon"
            geom = fixed
        result = AOI(
            footprint=gpd.GeoDataFrame(geometry=[geom], crs=WGS84),
            clip=clip,
            source_crs=_as_crs(source_crs),
            crosses_antimeridian=crosses,
        )
    elif (
        isinstance(aoi, (tuple, list))
        and len(aoi) == 4
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in aoi)
    ):
        geom, crosses = _bbox_geometry(tuple(aoi))  # type: ignore[arg-type]
        result = AOI(
            footprint=gpd.GeoDataFrame(geometry=[geom], crs=WGS84),
            clip=clip,
            crosses_antimeridian=crosses,
        )
    elif (
        isinstance(aoi, shapely.Geometry)
        or isinstance(aoi, Mapping)
        or hasattr(aoi, "__geo_interface__")
    ):
        geom = aoi if isinstance(aoi, shapely.Geometry) else shapely.geometry.shape(aoi)
        if geom.is_empty:
            raise ValueError("The AOI geometry is empty.")
        result = AOI(footprint=gpd.GeoDataFrame(geometry=[geom], crs=WGS84), clip=clip)
    else:
        raise TypeError(
            f"Unsupported AOI type {type(aoi).__name__}. Expected a (west, south, east, north) "
            "tuple, a shapely geometry, a GeoJSON mapping, a GeoDataFrame/GeoSeries, an "
            "odc.geo GeoBox, an AOI, or None."
        )

    if crs is not None or resolution is not None:
        if resolution is None:
            raise ValueError("resolution= is required when crs= is given.")
        result = replace(result, geobox=result.to_geobox(resolution, crs))
    return result
