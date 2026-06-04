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
