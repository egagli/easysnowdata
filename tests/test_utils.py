"""Tests for easysnowdata.utils — all pure functions, no credentials required."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import easysnowdata
from easysnowdata.utils import (
    CredentialError,
    _earthaccess_login,
    _has_earthaccess_credentials,
    _has_earthengine_credentials,
    convert_bbox_to_geodataframe,
    datetime_to_DOWY,
    datetime_to_WY,
    get_stac_cfg,
    get_water_year_start,
    requires_earthaccess,
    requires_earthengine,
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


class TestCredentialError:
    def test_credential_error_is_exception(self):
        with pytest.raises(CredentialError):
            raise CredentialError("test")

    def test_top_level_export(self):
        assert easysnowdata.CredentialError is CredentialError

    def test_authenticate_all_is_callable(self):
        assert callable(easysnowdata.authenticate_all)

    def test_requires_earthengine_decorator_blocks_without_creds(self, monkeypatch):
        monkeypatch.setattr(
            "easysnowdata.utils._has_earthengine_credentials", lambda: False
        )

        @requires_earthengine
        def fake_gee_func():
            return "ok"

        with pytest.raises(CredentialError, match="Google Earth Engine"):
            fake_gee_func()

    def test_requires_earthengine_passes_with_creds(self, monkeypatch):
        monkeypatch.setattr(
            "easysnowdata.utils._has_earthengine_credentials", lambda: True
        )

        @requires_earthengine
        def fake_gee_func():
            return "ok"

        assert fake_gee_func() == "ok"

    def test_requires_earthaccess_decorator_blocks_without_creds(self, monkeypatch):
        monkeypatch.setattr(
            "easysnowdata.utils._has_earthaccess_credentials", lambda: False
        )

        @requires_earthaccess
        def fake_ea_func():
            return "ok"

        with pytest.raises(CredentialError, match="NASA EarthData"):
            fake_ea_func()

    def test_requires_earthaccess_passes_with_creds(self, monkeypatch):
        monkeypatch.setattr(
            "easysnowdata.utils._has_earthaccess_credentials", lambda: True
        )

        @requires_earthaccess
        def fake_ea_func():
            return "ok"

        assert fake_ea_func() == "ok"

    def test_has_earthengine_credentials_returns_bool(self):
        result = _has_earthengine_credentials()
        assert isinstance(result, bool)

    def test_has_earthaccess_credentials_returns_bool(self):
        result = _has_earthaccess_credentials()
        assert isinstance(result, bool)


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


class TestEarthaccessLogin:
    """Offline checks of the explicit-login helper (earthaccess is monkeypatched)."""

    @pytest.fixture
    def fake_earthaccess(self, monkeypatch):
        import types

        import earthaccess

        # earthaccess exposes __auth__/__store__ through a module __getattr__
        # that returns its private _auth/_store globals; patch those.
        auth = types.SimpleNamespace(authenticated=False)
        state = {"auth": auth, "calls": 0}

        def login(strategy):
            state["calls"] += 1
            auth.authenticated = True
            earthaccess._store = object()
            return auth

        monkeypatch.setattr(earthaccess, "_auth", auth)
        monkeypatch.setattr(earthaccess, "_store", None)
        monkeypatch.setattr(earthaccess, "login", login)
        monkeypatch.setattr("easysnowdata.utils.time.sleep", lambda s: None)
        return state

    def test_no_credentials_raises_before_calling_earthaccess(
        self, monkeypatch, fake_earthaccess
    ):
        for var in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "easysnowdata.utils._has_earthaccess_credentials", lambda: False
        )
        with pytest.raises(CredentialError, match="credentials are required"):
            _earthaccess_login()
        assert fake_earthaccess["calls"] == 0

    def test_token_uses_environment_strategy_once(self, monkeypatch, fake_earthaccess):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        _earthaccess_login()
        _earthaccess_login()  # already authenticated with a store: no second call
        assert fake_earthaccess["calls"] == 1

    def test_authenticated_without_store_logs_in_again(
        self, monkeypatch, fake_earthaccess
    ):
        # State left behind when Store creation failed after a token login.
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        fake_earthaccess["auth"].authenticated = True
        _earthaccess_login()
        assert fake_earthaccess["calls"] == 1

    def test_connection_error_is_retried_once_then_wrapped(
        self, monkeypatch, fake_earthaccess
    ):
        import earthaccess

        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")

        def failing_login(strategy):
            fake_earthaccess["calls"] += 1
            raise ConnectionError("Network is unreachable")

        monkeypatch.setattr(earthaccess, "login", failing_login)
        with pytest.raises(CredentialError, match="Network is unreachable"):
            _earthaccess_login()
        assert fake_earthaccess["calls"] == 2
