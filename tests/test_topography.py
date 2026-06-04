"""Tests for easysnowdata.topography.

Copernicus DEM uses Planetary Computer (anonymous access, no credentials required).
CHILI uses Google Earth Engine (requires EARTHENGINE_TOKEN).
"""

from __future__ import annotations

import pytest
import xarray as xr

TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)


class TestCopernicusDem:
    def test_30m_returns_dataarray(self):
        from easysnowdata.topography import get_copernicus_dem

        result = get_copernicus_dem(bbox_input=TEST_BBOX, resolution=30)
        assert isinstance(result, xr.DataArray)

    def test_90m_returns_dataarray(self):
        from easysnowdata.topography import get_copernicus_dem

        result = get_copernicus_dem(bbox_input=TEST_BBOX, resolution=90)
        assert isinstance(result, xr.DataArray)

    def test_has_data_citation(self):
        from easysnowdata.topography import get_copernicus_dem

        result = get_copernicus_dem(bbox_input=TEST_BBOX, resolution=30)
        assert "data_citation" in result.attrs

    def test_invalid_resolution_raises(self):
        from easysnowdata.topography import get_copernicus_dem

        with pytest.raises(ValueError, match="30 m and 90 m"):
            get_copernicus_dem(bbox_input=TEST_BBOX, resolution=15)

    def test_values_are_elevation(self):
        from easysnowdata.topography import get_copernicus_dem

        result = get_copernicus_dem(bbox_input=TEST_BBOX, resolution=90)
        # Mount Rainier area — should include values > 1000 m
        assert float(result.max()) > 1000


class TestChili:
    @pytest.mark.requires_earthengine
    def test_returns_dataarray(self):
        from easysnowdata.topography import get_chili

        result = get_chili(bbox_input=TEST_BBOX)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.requires_earthengine
    def test_values_normalised_0_1(self):
        from easysnowdata.topography import get_chili

        result = get_chili(bbox_input=TEST_BBOX)
        import numpy as np
        valid = result.values[~np.isnan(result.values)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    @pytest.mark.requires_earthengine
    def test_has_data_citation(self):
        from easysnowdata.topography import get_chili

        result = get_chili(bbox_input=TEST_BBOX)
        assert "data_citation" in result.attrs
