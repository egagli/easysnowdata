"""Tests for easysnowdata.utils — all pure functions, no credentials required."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
import shapely

from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    datetime_to_DOWY,
    datetime_to_WY,
    get_stac_cfg,
    get_water_year_start,
    suppress_stdout,
)


class TestSuppressStdout:
    def test_context_manager_runs(self, capsys):
        with suppress_stdout():
            print("this should be suppressed")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestConvertBboxToGeoDataFrame:
    def test_tuple_input(self):
        bbox = (-121.94, 46.72, -121.54, 46.99)
        result = convert_bbox_to_geodataframe(bbox)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326
        assert len(result) == 1

    def test_geodataframe_passthrough(self):
        gdf = gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(-121.94, 46.72, -121.54, 46.99)],
            crs="EPSG:4326",
        )
        result = convert_bbox_to_geodataframe(gdf)
        assert result is gdf

    def test_shapely_geometry_input(self):
        geom = shapely.geometry.box(-121.94, 46.72, -121.54, 46.99)
        result = convert_bbox_to_geodataframe(geom)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326

    def test_none_returns_world_extent(self):
        result = convert_bbox_to_geodataframe(None)
        assert isinstance(result, gpd.GeoDataFrame)
        bounds = result.total_bounds
        assert bounds[0] == -180
        assert bounds[2] == 180

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            convert_bbox_to_geodataframe("not a valid input")  # type: ignore[arg-type]


class TestWaterYear:
    def test_northern_oct_starts_new_wy(self):
        assert datetime_to_WY(pd.Timestamp("2020-10-01")) == 2021

    def test_northern_sep_still_previous_wy(self):
        assert datetime_to_WY(pd.Timestamp("2020-09-30")) == 2020

    def test_northern_jan_mid_wy(self):
        assert datetime_to_WY(pd.Timestamp("2021-01-15")) == 2021

    def test_dowy_oct1_is_day1(self):
        assert datetime_to_DOWY(pd.Timestamp("2020-10-01")) == 1

    def test_dowy_oct2_is_day2(self):
        assert datetime_to_DOWY(pd.Timestamp("2020-10-02")) == 2

    def test_dowy_nov1_is_day32(self):
        assert datetime_to_DOWY(pd.Timestamp("2020-11-01")) == 32

    def test_water_year_start_northern(self):
        start = get_water_year_start(pd.Timestamp("2021-03-15"), "northern")
        assert start == pd.Timestamp("2020-10-01")

    def test_water_year_start_northern_oct(self):
        start = get_water_year_start(pd.Timestamp("2021-10-15"), "northern")
        assert start == pd.Timestamp("2021-10-01")

    def test_water_year_start_southern(self):
        start = get_water_year_start(pd.Timestamp("2021-05-15"), "southern")
        assert start == pd.Timestamp("2021-04-01")

    def test_string_date_input(self):
        assert isinstance(datetime_to_WY("2020-06-01"), (int, float))

    def test_invalid_date_returns_nan(self):
        result = datetime_to_DOWY("not-a-date")
        assert np.isnan(result)


class TestGetStacCfg:
    def test_sentinel2_returns_dict(self):
        cfg = get_stac_cfg("sentinel-2-l2a")
        assert isinstance(cfg, dict)
        assert "sentinel-2-l2a" in cfg

    def test_hls_l30_returns_dict(self):
        cfg = get_stac_cfg("HLSL30_2.0")
        assert isinstance(cfg, dict)
        assert "HLSL30_2.0" in cfg

    def test_hls_s30_returns_dict(self):
        cfg = get_stac_cfg("HLSS30_2.0")
        assert isinstance(cfg, dict)
        assert "HLSS30_2.0" in cfg

    def test_unknown_sensor_raises(self):
        with pytest.raises(ValueError, match="Unknown sensor"):
            get_stac_cfg("unknown-sensor")
