"""Tests for easysnowdata.hydroclimatology.

No-credential tests use public HTTP endpoints (figshare, World Bank, GRDC).
Tests marked requires_earthengine / requires_earthaccess are skipped when
the corresponding environment variables are absent.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
import xarray as xr

TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)


# ---------------------------------------------------------------------------
# HydroATLAS (figshare — no credentials required)
# ---------------------------------------------------------------------------
class TestGetHydroBasins:
    @pytest.mark.live
    def test_returns_geodataframe(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        result = get_hydroBASINS(bbox_input=TEST_BBOX, level=5)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    @pytest.mark.live
    def test_has_data_citation(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        result = get_hydroBASINS(bbox_input=TEST_BBOX, level=5)
        assert "data_citation" in result.attrs
        assert result.attrs["product_id"] == "hydrobasins"

    def test_invalid_level_raises(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        with pytest.raises(ValueError):
            get_hydroBASINS(bbox_input=TEST_BBOX, level=15)

    @pytest.mark.live
    def test_kwargs_forwarded_to_read_file(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        # rows= is a geopandas.read_file argument; it must reach the reader
        result = get_hydroBASINS(bbox_input=TEST_BBOX, level=5, rows=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Köppen-Geiger (figshare — no credentials required)
# ---------------------------------------------------------------------------
class TestKoppenGeiger:
    pytestmark = pytest.mark.live

    def test_returns_dataarray(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert isinstance(result, xr.DataArray)

    def test_has_cf_flags_instead_of_class_info(self):
        """The class table is CF flag attrs now, not a dict of Python objects.

        ``class_info``, ``cmap`` and ``example_plot`` are gone by design
        (REVAMP_PLAN §2.5: nothing in ``.attrs`` is a Python object);
        ``easysnowdata.plotting.categorical`` draws the legend from the flags.
        """
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert len(result.attrs["flag_values"]) == 30
        assert result.attrs["flag_meanings"].split()[0] == "Af"
        assert len(result.attrs["flag_colors"].split()) == 30
        assert "class_info" not in result.attrs
        assert "cmap" not in result.attrs
        assert "example_plot" not in result.attrs
        assert all(not callable(value) for value in result.attrs.values())

    def test_kwargs_forwarded_to_open_rasterio(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        # Default load is not dask-backed; chunks= must make it so
        result = get_koppen_geiger_classes(
            bbox_input=TEST_BBOX, resolution="1 degree", chunks={"x": 90, "y": 90}
        )
        assert result.chunks is not None


# ---------------------------------------------------------------------------
# GRDC basins (World Bank / GRDC — no credentials required)
# ---------------------------------------------------------------------------
class TestGrdcMajorRiverBasins:
    pytestmark = pytest.mark.live

    def test_returns_geodataframe(self):
        from easysnowdata.hydroclimatology import (
            get_grdc_major_river_basins_of_the_world,
        )

        result = get_grdc_major_river_basins_of_the_world(bbox_input=TEST_BBOX)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_kwargs_forwarded_to_read_file(self):
        from easysnowdata.hydroclimatology import (
            get_grdc_major_river_basins_of_the_world,
        )

        # rows= is a geopandas.read_file argument; it must reach the reader
        result = get_grdc_major_river_basins_of_the_world(bbox_input=None, rows=5)
        assert len(result) == 5


class TestGrdcWmoBasins:
    pytestmark = pytest.mark.live

    def test_kwargs_forwarded_to_read_file(self):
        from easysnowdata.hydroclimatology import get_grdc_wmo_basins

        result = get_grdc_wmo_basins(bbox_input=None, rows=5)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# ERA5 via GCS (no credentials required for anonymous Zarr access)
# ---------------------------------------------------------------------------
class TestEra5Gcs:
    @pytest.mark.live
    def test_gcs_returns_dataset(self):
        from easysnowdata.hydroclimatology import get_era5

        result = get_era5(
            bbox_input=TEST_BBOX,
            source="GCS",
            start_date="2020-01-01",
            end_date="2020-01-02",
        )
        assert isinstance(result, xr.Dataset)

    def test_invalid_source_raises(self):
        from easysnowdata.hydroclimatology import get_era5

        with pytest.raises(ValueError):
            get_era5(bbox_input=TEST_BBOX, source="INVALID")

    def test_gcs_wrong_version_raises(self):
        from easysnowdata.hydroclimatology import get_era5

        with pytest.raises(ValueError):
            get_era5(bbox_input=TEST_BBOX, source="GCS", version="ERA5_LAND")

    @pytest.mark.live
    def test_kwargs_forwarded_to_open_zarr(self):
        from easysnowdata.hydroclimatology import get_era5

        # Default is chunks=None (lazy, not dask); chunks= must make it dask-backed
        result = get_era5(
            bbox_input=TEST_BBOX,
            source="GCS",
            start_date="2020-01-01",
            end_date="2020-01-02",
            chunks={"time": 24},
        )
        assert result["2m_temperature"].chunks is not None


# ---------------------------------------------------------------------------
# GEE-backed functions (EARTHENGINE_TOKEN required)
# ---------------------------------------------------------------------------
class TestHucGeometries:
    pytestmark = pytest.mark.live

    def test_returns_geodataframe(self):
        # No longer needs Earth Engine: the default route is the USGS WBD
        # REST service (§12 Q7).
        from easysnowdata.hydroclimatology import get_huc_geometries

        result = get_huc_geometries(bbox_input=TEST_BBOX, huc_level="08")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert list(result.columns)[:2] == ["name", "huc8"]


class TestSnodas:
    pytestmark = pytest.mark.live

    @pytest.mark.requires_earthengine
    def test_returns_dataset_with_swe(self):
        from easysnowdata.hydroclimatology import get_snodas

        result = get_snodas(
            bbox_input=TEST_BBOX,
            start_date="2020-01-01",
            end_date="2020-01-03",
        )
        assert isinstance(result, xr.Dataset)
        assert "SWE" in result.data_vars or "Snow_Depth" in result.data_vars
        assert result["SWE"].dims == ("time", "latitude", "longitude")
        assert result.rio.crs is not None

    @pytest.mark.requires_earthengine
    def test_invalid_variable_raises(self):
        from easysnowdata.hydroclimatology import get_snodas

        with pytest.raises(ValueError, match="Invalid variables"):
            get_snodas(
                bbox_input=TEST_BBOX,
                start_date="2020-01-01",
                end_date="2020-01-03",
                variables=["NotAVariable"],
            )

    @pytest.mark.requires_earthengine
    def test_kwargs_forwarded_to_open_dataset(self):
        from easysnowdata.hydroclimatology import get_snodas

        # Default is chunks=None (lazy, not dask); chunks= must make it dask-backed
        result = get_snodas(
            bbox_input=TEST_BBOX,
            start_date="2020-01-01",
            end_date="2020-01-03",
            variables="SWE",
            chunks={"time": 1, "x": 64, "y": 64},
        )
        swe = result["SWE"]
        assert swe.chunks is not None
        assert max(swe.chunks[swe.get_axis_num("longitude")]) <= 64
        assert max(swe.chunks[swe.get_axis_num("latitude")]) <= 64


class TestEra5Gee:
    pytestmark = pytest.mark.live

    @pytest.mark.requires_earthengine
    def test_gee_returns_dataset_time_lat_lon(self):
        from easysnowdata.hydroclimatology import get_era5

        result = get_era5(
            bbox_input=TEST_BBOX,
            source="GEE",
            version="ERA5_LAND",
            cadence="DAILY",
            start_date="2020-01-01",
            end_date="2020-01-03",
            variables=["temperature_2m"],
        )
        assert isinstance(result, xr.Dataset)
        assert result["temperature_2m"].dims == ("time", "latitude", "longitude")
        assert result.sizes["time"] == 3
        assert result.rio.crs is not None


# ---------------------------------------------------------------------------
# UCLA snow reanalysis — offline checks of the ensemble-statistic mapping
# ---------------------------------------------------------------------------
class TestUclaSnowReanalysisStatsMapping:
    def test_stats_indices_are_distinct_and_in_file_order(self):
        from easysnowdata.hydroclimatology import _UCLA_SR_STATS_INDEX

        # Regression: "median" and "25pct" used to share index 2.
        assert _UCLA_SR_STATS_INDEX == {
            "mean": 0,
            "std": 1,
            "median": 2,
            "25pct": 3,
            "75pct": 4,
        }
        assert len(set(_UCLA_SR_STATS_INDEX.values())) == len(_UCLA_SR_STATS_INDEX)

    def test_invalid_stats_raises_before_any_network_call(self, monkeypatch):
        from easysnowdata import hydroclimatology, providers

        monkeypatch.setattr(
            "easysnowdata.utils._has_earthaccess_credentials", lambda: True
        )
        # The loader now searches through the earthdata provider (which logs
        # in first); neither may be reached for an invalid `stats`. Patch the
        # provider itself: hydroclimatology is a shim module now and no longer
        # imports it.
        monkeypatch.setattr(
            providers.earthdata,
            "search",
            lambda *args, **kwargs: pytest.fail("network"),
        )
        monkeypatch.setattr(
            providers.earthdata,
            "ensure",
            lambda: pytest.fail("login"),
        )
        with pytest.raises(ValueError, match="stats must be one of"):
            hydroclimatology.get_ucla_snow_reanalysis(
                bbox_input=TEST_BBOX, stats="mode"
            )


# ---------------------------------------------------------------------------
# earthaccess-backed functions (EARTHDATA credentials required)
# ---------------------------------------------------------------------------
class TestUclaSnowReanalysis:
    pytestmark = pytest.mark.live

    @pytest.mark.requires_earthaccess
    def test_returns_dataarray(self):
        from easysnowdata.hydroclimatology import get_ucla_snow_reanalysis

        result = get_ucla_snow_reanalysis(
            bbox_input=TEST_BBOX,
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        assert isinstance(result, xr.DataArray)
        assert "data_citation" in result.attrs

    @pytest.mark.requires_earthaccess
    def test_percentiles_bracket_the_median(self):
        from easysnowdata.hydroclimatology import get_ucla_snow_reanalysis

        kwargs = dict(
            bbox_input=TEST_BBOX, start_date="2020-03-01", end_date="2020-03-02"
        )
        q25 = get_ucla_snow_reanalysis(stats="25pct", **kwargs).compute()
        med = get_ucla_snow_reanalysis(stats="median", **kwargs).compute()
        q75 = get_ucla_snow_reanalysis(stats="75pct", **kwargs).compute()
        # Distinct Stats slices: the percentiles must differ from the median
        # somewhere and be ordered where they do.
        assert not med.equals(q25)
        assert bool((q25 <= med).all()) and bool((med <= q75).all())
