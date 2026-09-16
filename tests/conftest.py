"""Shared fixtures and skip-markers for easysnowdata tests."""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely

from easysnowdata import auth

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
        "requires_planet: marks tests that need Planet credentials (PL_API_KEY, "
        "PL_AUTH_* or a `planet auth login` session)",
    )
    config.addinivalue_line(
        "markers",
        "requires_nve: marks tests that need an NVE HydAPI key (NVE_API_KEY)",
    )
    config.addinivalue_line(
        "markers",
        "recorded: marks tests replayed from pytest-recording cassettes or run "
        "against the tiny local fixtures (offline tier, no sockets)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks slow integration tests that hit live external APIs",
    )


_REQUIRES = {
    "requires_earthengine": "earthengine",
    "requires_earthaccess": "earthdata",
    "requires_planet": "planet",
    "requires_nve": "nve",
}


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Skip only when *no* credential source is available. The detection is the
    # package's own (easysnowdata.auth), so a CI runner with EARTHDATA_TOKEN or
    # a developer with ~/.netrc runs the Earthdata tests.
    for marker, provider_name in _REQUIRES.items():
        if item.get_closest_marker(marker) is None:
            continue
        provider = auth.get(provider_name)
        if not provider.detect():
            pytest.skip(
                f"Skipping: no {provider.title} credentials "
                f"({', '.join(provider.env_vars + provider.files)})."
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
