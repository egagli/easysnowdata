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
    def test_returns_geodataframe(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        result = get_hydroBASINS(bbox_input=TEST_BBOX, level=5)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_has_data_citation(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        result = get_hydroBASINS(bbox_input=TEST_BBOX, level=5)
        assert "data_citation" in result.attrs

    def test_invalid_level_raises(self):
        from easysnowdata.hydroclimatology import get_hydroBASINS

        with pytest.raises(ValueError):
            get_hydroBASINS(bbox_input=TEST_BBOX, level=15)


# ---------------------------------------------------------------------------
# Köppen-Geiger (figshare — no credentials required)
# ---------------------------------------------------------------------------
class TestKoppenGeiger:
    def test_returns_dataarray(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert isinstance(result, xr.DataArray)

    def test_has_class_info(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert "class_info" in result.attrs
        assert len(result.attrs["class_info"]) == 30

    def test_has_cmap(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert "cmap" in result.attrs

    def test_has_example_plot_callable(self):
        from easysnowdata.hydroclimatology import get_koppen_geiger_classes

        result = get_koppen_geiger_classes(bbox_input=TEST_BBOX, resolution="1 degree")
        assert callable(result.attrs.get("example_plot"))


# ---------------------------------------------------------------------------
# GRDC basins (World Bank / GRDC — no credentials required)
# ---------------------------------------------------------------------------
class TestGrdcMajorRiverBasins:
    def test_returns_geodataframe(self):
        from easysnowdata.hydroclimatology import get_grdc_major_river_basins_of_the_world

        result = get_grdc_major_river_basins_of_the_world(bbox_input=TEST_BBOX)
        assert isinstance(result, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# ERA5 via GCS (no credentials required for anonymous Zarr access)
# ---------------------------------------------------------------------------
class TestEra5Gcs:
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


# ---------------------------------------------------------------------------
# GEE-backed functions (EARTHENGINE_TOKEN required)
# ---------------------------------------------------------------------------
class TestHucGeometries:
    @pytest.mark.requires_earthengine
    def test_returns_geodataframe(self):
        from easysnowdata.hydroclimatology import get_huc_geometries

        result = get_huc_geometries(bbox_input=TEST_BBOX, huc_level="08")
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0


class TestSnodas:
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


# ---------------------------------------------------------------------------
# earthaccess-backed functions (EARTHDATA credentials required)
# ---------------------------------------------------------------------------
class TestUclaSnowReanalysis:
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
