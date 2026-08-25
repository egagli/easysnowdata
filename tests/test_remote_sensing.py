"""Tests for easysnowdata.remote_sensing.

No-credential tests use public HTTP endpoints (Zenodo, Azure Blob).
Planetary Computer provides anonymous access to its STAC catalog.
"""

from __future__ import annotations

import pytest
import xarray as xr

TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)


# ---------------------------------------------------------------------------
# Seasonal snow classification (Azure Blob — no credentials required)
# ---------------------------------------------------------------------------
class TestSeasonalSnowClassification:
    def test_returns_dataarray(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(bbox_input=TEST_BBOX)
        assert isinstance(result, xr.DataArray)

    def test_has_class_info(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(bbox_input=TEST_BBOX)
        assert "class_info" in result.attrs
        assert len(result.attrs["class_info"]) == 9

    def test_has_cmap(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(bbox_input=TEST_BBOX)
        assert "cmap" in result.attrs

    def test_has_example_plot(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(bbox_input=TEST_BBOX)
        assert callable(result.attrs.get("example_plot"))

    def test_has_data_citation(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(bbox_input=TEST_BBOX)
        assert "data_citation" in result.attrs

    def test_kwargs_forwarded_to_open_rasterio(self):
        from easysnowdata.remote_sensing import get_seasonal_snow_classification

        result = get_seasonal_snow_classification(
            bbox_input=TEST_BBOX, chunks={"x": 64, "y": 64}
        )
        assert result.chunks is not None
        assert max(result.chunks[result.get_axis_num("x")]) <= 64
        assert max(result.chunks[result.get_axis_num("y")]) <= 64


# ---------------------------------------------------------------------------
# Forest cover fraction (Zenodo — no credentials required)
# ---------------------------------------------------------------------------
class TestForestCoverFraction:
    def test_returns_dataarray(self):
        from easysnowdata.remote_sensing import get_forest_cover_fraction

        result = get_forest_cover_fraction(bbox_input=TEST_BBOX)
        assert isinstance(result, xr.DataArray)

    def test_has_data_citation(self):
        from easysnowdata.remote_sensing import get_forest_cover_fraction

        result = get_forest_cover_fraction(bbox_input=TEST_BBOX)
        assert "data_citation" in result.attrs

    def test_values_non_negative(self):
        import numpy as np

        from easysnowdata.remote_sensing import get_forest_cover_fraction

        result = get_forest_cover_fraction(bbox_input=TEST_BBOX)
        valid = result.values[~np.isnan(result.values.astype(float))]
        assert valid.min() >= 0

    def test_kwargs_forwarded_to_open_rasterio(self):
        from easysnowdata.remote_sensing import get_forest_cover_fraction

        result = get_forest_cover_fraction(
            bbox_input=TEST_BBOX, chunks={"x": 128, "y": 128}
        )
        assert result.chunks is not None
        assert max(result.chunks[result.get_axis_num("x")]) <= 128
        assert max(result.chunks[result.get_axis_num("y")]) <= 128


# ---------------------------------------------------------------------------
# Seasonal mountain snow mask (Zenodo — no credentials required)
# ---------------------------------------------------------------------------
class TestSeasonalMountainSnowMask:
    def test_returns_dataarray(self):
        from easysnowdata.remote_sensing import get_seasonal_mountain_snow_mask

        result = get_seasonal_mountain_snow_mask(bbox_input=TEST_BBOX)
        assert isinstance(result, xr.DataArray)

    def test_invalid_product_raises(self):
        from easysnowdata.remote_sensing import get_seasonal_mountain_snow_mask

        with pytest.raises(ValueError):
            get_seasonal_mountain_snow_mask(
                bbox_input=TEST_BBOX, data_product="invalid"
            )


# ---------------------------------------------------------------------------
# ESA WorldCover (Planetary Computer — anonymous access)
# ---------------------------------------------------------------------------
class TestEsaWorldcover:
    def test_returns_dataarray(self):
        from easysnowdata.remote_sensing import get_esa_worldcover

        result = get_esa_worldcover(bbox_input=TEST_BBOX)
        assert isinstance(result, xr.DataArray)

    def test_invalid_version_raises(self):
        from easysnowdata.remote_sensing import get_esa_worldcover

        with pytest.raises(ValueError):
            get_esa_worldcover(bbox_input=TEST_BBOX, version="v999")

    def test_kwargs_forwarded_to_odc_stac_load(self):
        from easysnowdata.remote_sensing import get_esa_worldcover

        result = get_esa_worldcover(bbox_input=TEST_BBOX, chunks={"x": 512, "y": 512})
        assert result.chunks is not None
        # odc names dims latitude/longitude in EPSG:4326, so check every dim
        assert all(max(sizes) <= 512 for sizes in result.chunks)


# ---------------------------------------------------------------------------
# NLCD land cover (Google Earth Engine — EARTHENGINE_TOKEN required)
# ---------------------------------------------------------------------------
class TestNlcdLandcover:
    @pytest.mark.requires_earthengine
    def test_returns_uint8_dataarray_y_x(self):
        from easysnowdata.remote_sensing import get_nlcd_landcover

        result = get_nlcd_landcover(bbox_input=TEST_BBOX, layer="landcover")
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("y", "x")
        assert result.dtype == "uint8"
        assert "class_info" in result.attrs
        assert result.rio.crs is not None

    @pytest.mark.requires_earthengine
    def test_kwargs_forwarded_to_open_dataset(self):
        from easysnowdata.remote_sensing import get_nlcd_landcover

        result = get_nlcd_landcover(
            bbox_input=TEST_BBOX, layer="landcover", chunks={"x": 256, "y": 256}
        )
        assert result.chunks is not None
        assert max(result.chunks[result.get_axis_num("x")]) <= 256
        assert max(result.chunks[result.get_axis_num("y")]) <= 256


# ---------------------------------------------------------------------------
# Sentinel-2 (Planetary Computer — anonymous access; load is lazy)
# ---------------------------------------------------------------------------
class TestSentinel2:
    def test_kwargs_forwarded_to_odc_stac_load(self):
        from easysnowdata.remote_sensing import Sentinel2

        chunks = {"time": 1, "x": 256, "y": 256}
        s2 = Sentinel2(
            TEST_BBOX,
            start_date="2023-08-01",
            end_date="2023-08-10",
            bands=["red", "scl"],
            remove_nodata=False,
            harmonize_to_old=False,
            scale_data=False,
            chunks=chunks,
        )
        assert s2.load_kwargs == {"chunks": chunks}
        red = s2.data["red"]
        assert red.chunks is not None
        assert max(red.chunks[red.get_axis_num("x")]) <= 256
        assert max(red.chunks[red.get_axis_num("y")]) <= 256


# ---------------------------------------------------------------------------
# MODIS snow (MOD10A2 via Planetary Computer — anonymous access; load is lazy)
# ---------------------------------------------------------------------------
class TestModisSnow:
    def test_kwargs_forwarded_to_odc_stac_load(self):
        from easysnowdata.remote_sensing import MODIS_snow

        modis = MODIS_snow(
            TEST_BBOX,
            start_date="2023-01-01",
            end_date="2023-01-20",
            data_product="MOD10A2",
            mute=True,
            chunks={"time": 1, "x": 64, "y": 64},
        )
        da = modis.data["Maximum_Snow_Extent"]
        assert da.chunks is not None
        assert max(da.chunks[da.get_axis_num("x")]) <= 64
        assert max(da.chunks[da.get_axis_num("y")]) <= 64
