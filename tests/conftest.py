"""Shared fixtures and skip-markers for easysnowdata tests."""

from __future__ import annotations

import os

import geopandas as gpd
import pytest
import shapely

# Small bbox for testing — Mount Rainier, WA (covers SNOTEL, snow products, etc.)
TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_earthengine: marks tests that need EARTHENGINE_TOKEN set",
    )
    config.addinivalue_line(
        "markers",
        "requires_earthaccess: marks tests that need EARTHDATA_USERNAME and EARTHDATA_PASSWORD set",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks slow integration tests that hit live external APIs",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    for _ in item.iter_markers("requires_earthengine"):
        if not os.getenv("EARTHENGINE_TOKEN"):
            pytest.skip("Skipping: EARTHENGINE_TOKEN environment variable is not set.")
    for _ in item.iter_markers("requires_earthaccess"):
        if not (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")):
            pytest.skip("Skipping: EARTHDATA_USERNAME / EARTHDATA_PASSWORD not set.")


@pytest.fixture(scope="session")
def test_bbox() -> tuple:
    """Small bounding box around Mount Rainier, WA."""
    return TEST_BBOX


@pytest.fixture(scope="session")
def test_bbox_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame wrapping the test bbox."""
    return gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(*TEST_BBOX)], crs="EPSG:4326"
    )
