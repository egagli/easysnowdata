"""Unit tests for easysnowdata.aoi — pure geometry, no network."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from odc.geo import res_
from odc.geo.geobox import GeoBox
from odc.geo.geom import box as odc_box
from pyproj import CRS

from easysnowdata.aoi import (
    AOI,
    WORLD_BOUNDS,
    estimate_utm_crs,
    parse_aoi,
    split_antimeridian,
)

RAINIER = (-121.94, 46.72, -121.54, 46.99)
DATELINE = (170.0, -20.0, -170.0, -10.0)  # west > east: crosses the antimeridian


class TestParseInputs:
    def test_tuple(self):
        aoi = parse_aoi(RAINIER)
        assert isinstance(aoi, AOI)
        assert aoi.footprint.crs.to_epsg() == 4326
        assert aoi.bounds == pytest.approx(RAINIER)
        assert aoi.clip is True
        assert not aoi.crosses_antimeridian
        assert aoi.geobox is None

    def test_list_and_numpy_scalars(self):
        aoi = parse_aoi([np.float64(v) for v in RAINIER])
        assert aoi.bounds == pytest.approx(RAINIER)

    def test_shapely_geometry(self):
        aoi = parse_aoi(shapely.box(*RAINIER))
        assert aoi.geometry.equals(shapely.box(*RAINIER))
        assert aoi.source_crs.to_epsg() == 4326

    def test_geojson_mapping(self):
        aoi = parse_aoi(shapely.geometry.mapping(shapely.box(*RAINIER)))
        assert aoi.bounds == pytest.approx(RAINIER)

    def test_geo_interface_object(self):
        class Thing:
            __geo_interface__ = shapely.geometry.mapping(shapely.box(*RAINIER))

        assert parse_aoi(Thing()).bounds == pytest.approx(RAINIER)

    def test_geodataframe_in_utm_is_reprojected(self):
        gdf = gpd.GeoDataFrame(
            geometry=[shapely.box(*RAINIER)], crs="EPSG:4326"
        ).to_crs("EPSG:32610")
        aoi = parse_aoi(gdf)
        assert aoi.footprint.crs.to_epsg() == 4326
        assert aoi.source_crs.to_epsg() == 32610
        assert aoi.bounds == pytest.approx(RAINIER, abs=1e-3)

    def test_geoseries_rows_are_unioned(self):
        series = gpd.GeoSeries(
            [shapely.box(-122, 46, -121.7, 47), shapely.box(-121.8, 46, -121.5, 47)],
            crs="EPSG:4326",
        )
        aoi = parse_aoi(series)
        assert aoi.geometry.geom_type == "Polygon"
        assert aoi.bounds == pytest.approx((-122, 46, -121.5, 47))

    def test_geodataframe_without_crs_assumes_wgs84(self, caplog):
        gdf = gpd.GeoDataFrame(geometry=[shapely.box(*RAINIER)])
        with caplog.at_level("WARNING", logger="easysnowdata.aoi"):
            aoi = parse_aoi(gdf)
        assert "assuming EPSG:4326" in caplog.text
        assert aoi.bounds == pytest.approx(RAINIER)

    def test_empty_geodataframe_raises(self):
        with pytest.raises(ValueError, match="no geometries"):
            parse_aoi(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))

    def test_geobox_input_keeps_native_grid(self):
        gb = GeoBox.from_bbox(RAINIER, crs="EPSG:4326", resolution=res_(0.01))
        aoi = parse_aoi(gb)
        assert aoi.geobox is gb
        assert aoi.to_geobox() is gb
        assert aoi.to_geobox(crs="EPSG:4326") is gb
        assert aoi.bounds == pytest.approx(gb.boundingbox.bbox, abs=0.02)

    def test_projected_geobox_footprint(self):
        gb = GeoBox.from_geopolygon(
            odc_box(*RAINIER, "EPSG:4326"), resolution=res_(100), crs="EPSG:32610"
        )
        aoi = parse_aoi(gb)
        assert aoi.source_crs.to_epsg() == 32610
        assert aoi.to_geobox(crs="utm") is gb
        w, s, e, n = aoi.bounds
        assert w < RAINIER[0] + 1e-6 and e > RAINIER[2] - 1e-6

    def test_none_is_global(self):
        aoi = parse_aoi(None)
        assert aoi.is_global
        assert aoi.bounds == WORLD_BOUNDS

    def test_aoi_passthrough_updates_clip(self):
        aoi = parse_aoi(RAINIER)
        again = parse_aoi(aoi, clip=False)
        assert again.clip is False
        assert again.footprint is aoi.footprint

    def test_crs_and_resolution_attach_geobox(self):
        aoi = parse_aoi(RAINIER, crs="utm", resolution=30)
        assert aoi.geobox is not None
        assert aoi.geobox.crs.epsg == 32610
        assert aoi.geobox.resolution.x == 30

    def test_crs_without_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution="):
            parse_aoi(RAINIER, crs="EPSG:4326")

    @pytest.mark.parametrize(
        "bad",
        ["not an aoi", 42, (1, 2, 3), (-121.94, 46.72, -121.54)],
    )
    def test_unsupported_type_raises(self, bad):
        with pytest.raises(TypeError, match="Unsupported AOI type"):
            parse_aoi(bad)

    @pytest.mark.parametrize(
        "bad, match",
        [
            ((-121.9, 95, -121.5, 96), "Latitudes"),
            ((-200, 46, -121.5, 47), "Longitudes"),
            ((-121.9, 47, -121.5, 46), "south"),
        ],
    )
    def test_malformed_bounds_raise(self, bad, match):
        with pytest.raises(ValueError, match=match):
            parse_aoi(bad)

    def test_empty_geometry_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_aoi(shapely.Polygon())


class TestAntimeridian:
    def test_tuple_is_split(self):
        aoi = parse_aoi(DATELINE)
        assert aoi.crosses_antimeridian
        assert aoi.geometry.geom_type == "MultiPolygon"
        assert len(aoi.geometry.geoms) == 2
        assert aoi.bounds == pytest.approx(DATELINE)
        assert aoi.stac_bbox == pytest.approx(list(DATELINE))
        assert aoi.stac_intersects["type"] == "MultiPolygon"

    def test_unwrapped_geometry_is_contiguous(self):
        aoi = parse_aoi(DATELINE)
        assert aoi.unwrapped_geometry.bounds == pytest.approx((170, -20, 190, -10))
        assert aoi.total_bounds() == pytest.approx((170, -20, 190, -10))

    def test_utm_zone_at_the_dateline(self):
        crs = parse_aoi(DATELINE).utm_crs
        assert crs.is_projected
        assert crs.to_epsg() in (32701, 32760, 32601, 32660)

    def test_geobox_is_one_contiguous_raster(self):
        aoi = parse_aoi(DATELINE)
        gb = aoi.to_geobox(resolution=0.1, crs="EPSG:4326")
        assert gb.shape == (100, 200)
        assert gb.boundingbox.left == pytest.approx(170)
        assert gb.boundingbox.right == pytest.approx(190)
        utm = aoi.to_geobox(resolution=5000)
        assert utm.crs.projected
        assert all(n > 10 for n in utm.shape)

    def test_projected_gdf_across_dateline_is_repaired(self):
        gdf = gpd.GeoDataFrame(
            geometry=[shapely.box(179.5, -16, 180.5, -15)], crs="EPSG:4326"
        )
        utm = gdf.to_crs("EPSG:32760")
        aoi = parse_aoi(utm)
        assert aoi.crosses_antimeridian
        assert aoi.geometry.geom_type == "MultiPolygon"
        w, s, e, n = aoi.bounds
        assert w == pytest.approx(179.5, abs=1e-3)
        assert e == pytest.approx(-179.5, abs=1e-3)

    def test_split_antimeridian_helper(self):
        geom = shapely.box(175, 0, 185, 5)
        split = split_antimeridian(geom)
        assert split.geom_type == "MultiPolygon"
        assert split.bounds == pytest.approx((-180, 0, 180, 5))
        untouched = split_antimeridian(shapely.box(10, 0, 20, 5))
        assert untouched.geom_type == "Polygon"

    def test_to_crs_projected_uses_unwrapped_footprint(self):
        aoi = parse_aoi(DATELINE)
        projected = aoi.to_crs("EPSG:32601")
        assert projected.geometry.iloc[0].geom_type == "Polygon"
        assert aoi.to_crs("EPSG:4326").geometry.iloc[0].geom_type == "MultiPolygon"


class TestGrids:
    def test_utm_estimate(self):
        assert parse_aoi(RAINIER).utm_crs.to_epsg() == 32610
        assert estimate_utm_crs(shapely.Point(10, 87)).to_epsg() == 3413
        assert estimate_utm_crs(shapely.Point(10, -85)).to_epsg() == 3031

    def test_to_geobox_default_crs_is_utm(self):
        gb = parse_aoi(RAINIER).to_geobox(resolution=30)
        assert gb.crs.epsg == 32610
        assert gb.resolution.x == 30 and gb.resolution.y == -30
        assert gb.shape[0] > 900 and gb.shape[1] > 900

    def test_to_geobox_geographic_and_tuple_resolution(self):
        gb = parse_aoi(RAINIER).to_geobox(resolution=(0.01, -0.01), crs="EPSG:4326")
        assert gb.crs.epsg == 4326
        assert gb.shape == (27, 40)

    def test_to_geobox_by_shape(self):
        gb = parse_aoi(RAINIER).to_geobox(shape=(100, 200), crs="EPSG:4326")
        assert gb.shape == (100, 200)

    def test_world_grid(self):
        gb = parse_aoi(None).to_geobox(resolution=1, crs="EPSG:4326")
        assert gb.shape == (180, 360)

    def test_to_geobox_requires_resolution(self):
        with pytest.raises(ValueError, match="resolution="):
            parse_aoi(RAINIER).to_geobox()

    def test_native_geobox_not_reused_for_other_crs(self):
        gb = GeoBox.from_bbox(RAINIER, crs="EPSG:4326", resolution=res_(0.01))
        aoi = parse_aoi(gb)
        with pytest.raises(ValueError):
            aoi.to_geobox(crs="EPSG:32610")
        assert aoi.to_geobox(resolution=100, crs="utm").crs.epsg == 32610

    def test_total_bounds_in_projected_crs(self):
        b = parse_aoi(RAINIER).total_bounds("EPSG:32610")
        assert 500_000 < b[0] < 620_000 and 5_170_000 < b[1] < 5_210_000

    def test_buffer_grows_footprint(self):
        aoi = parse_aoi(RAINIER)
        bigger = aoi.buffer(5000)
        assert bigger.geometry.contains(aoi.geometry)
        assert bigger.geobox is None
        assert bigger.clip is aoi.clip

    def test_with_clip_and_geo_interface(self):
        aoi = parse_aoi(RAINIER)
        assert aoi.with_clip(False).clip is False
        assert aoi.__geo_interface__["type"] == "Polygon"
        assert CRS.from_user_input(aoi.footprint.crs).to_epsg() == 4326
