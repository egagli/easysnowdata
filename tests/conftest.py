"""Shared fixtures and skip-markers for easysnowdata tests."""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely

from easysnowdata.utils import (
    _has_earthaccess_credentials,
    _has_earthengine_credentials,
)

# Small bbox for testing — Mount Rainier, WA (covers SNOTEL, snow products, etc.)
TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: marks tests that make network requests to a data provider; "
        "deselected in the offline tier (pytest -m 'not live')",
    )
    config.addinivalue_line(
        "markers",
        "requires_earthengine: marks tests that need Google Earth Engine "
        "credentials (EARTHENGINE_TOKEN or ~/.config/earthengine/credentials)",
    )
    config.addinivalue_line(
        "markers",
        "requires_earthaccess: marks tests that need NASA Earthdata credentials "
        "(EARTHDATA_TOKEN, EARTHDATA_USERNAME + EARTHDATA_PASSWORD, or a "
        "~/.netrc entry for urs.earthdata.nasa.gov)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks slow integration tests that hit live external APIs",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Skip only when *no* credential source is available. The detection is the
    # same one the package uses, so a CI runner with EARTHDATA_TOKEN (or a
    # developer with ~/.netrc) no longer skips the Earthdata tests.
    for _ in item.iter_markers("requires_earthengine"):
        if not _has_earthengine_credentials():
            pytest.skip(
                "Skipping: no Google Earth Engine credentials "
                "(EARTHENGINE_TOKEN or ~/.config/earthengine/credentials)."
            )
    for _ in item.iter_markers("requires_earthaccess"):
        if not _has_earthaccess_credentials():
            pytest.skip(
                "Skipping: no NASA Earthdata credentials (EARTHDATA_TOKEN, "
                "EARTHDATA_USERNAME + EARTHDATA_PASSWORD, or ~/.netrc)."
            )


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
