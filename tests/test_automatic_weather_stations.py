"""Tests for easysnowdata.automatic_weather_stations.

These tests make real HTTP requests to GitHub-hosted CSV/GeoJSON files.
No API credentials required.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
import pytest
import xarray as xr

from easysnowdata.automatic_weather_stations import StationCollection


@pytest.fixture(scope="module")
def sc() -> StationCollection:
    """Shared StationCollection instance for the test module."""
    return StationCollection()


class TestGetAllStations:
    def test_returns_geodataframe(self, sc):
        assert isinstance(sc.all_stations, gpd.GeoDataFrame)

    def test_has_many_stations(self, sc):
        assert len(sc.all_stations) > 900

    def test_has_valid_crs(self, sc):
        assert sc.all_stations.crs is not None
        assert sc.all_stations.crs.to_epsg() == 4326

    def test_has_geometry_column(self, sc):
        assert sc.all_stations.geometry is not None

    def test_station_index_is_code(self, sc):
        # Codes look like "679_WA_SNTL"
        assert sc.all_stations.index.name == "code"
        assert "_" in sc.all_stations.index[0]


class TestChooseStations:
    def test_choose_by_string(self, sc):
        sc.choose_stations("679_WA_SNTL")
        assert len(sc.stations) == 1
        assert sc.stations.index[0] == "679_WA_SNTL"

    def test_choose_by_list(self, sc):
        sc.choose_stations(["679_WA_SNTL", "680_WA_SNTL"])
        assert len(sc.stations) == 2

    def test_choose_by_geodataframe(self, sc):
        gdf = sc.all_stations.iloc[:3]
        sc.choose_stations(gdf)
        assert len(sc.stations) == 3


class TestGetData:
    def test_single_station_returns_dataframe(self, sc):
        sc.get_data(
            stations="679_WA_SNTL",
            variables=["WTEQ"],
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        assert isinstance(sc.data, pd.DataFrame)
        assert "WTEQ" in sc.data.columns

    def test_single_station_date_range(self, sc):
        sc.get_data(
            stations="679_WA_SNTL",
            variables=["SNWD"],
            start_date="2021-01-01",
            end_date="2021-01-31",
        )
        assert sc.data.index.min() >= pd.Timestamp("2021-01-01")
        assert sc.data.index.max() <= pd.Timestamp("2021-01-31")

    def test_multiple_stations_returns_dataset(self, sc):
        sc.get_data(
            stations=["679_WA_SNTL", "680_WA_SNTL"],
            variables=["WTEQ"],
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        assert isinstance(sc.data, xr.Dataset)
        assert "WTEQ" in sc.data.data_vars

    def test_multiple_stations_has_wy_coord(self, sc):
        sc.get_data(
            stations=["679_WA_SNTL", "680_WA_SNTL"],
            variables=["WTEQ"],
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        assert "WY" in sc.data.coords

    def test_multiple_stations_has_dowy_coord(self, sc):
        sc.get_data(
            stations=["679_WA_SNTL", "680_WA_SNTL"],
            variables=["WTEQ"],
            start_date="2020-01-01",
            end_date="2020-03-31",
        )
        assert "DOWY" in sc.data.coords
