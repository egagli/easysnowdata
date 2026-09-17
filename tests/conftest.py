"""Shared fixtures and skip-markers for easysnowdata tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def _scrub_response(response: dict) -> dict:
    """Drop cookies and rate-limit tokens from recorded responses."""
    headers = response.get("headers", {})
    for name in list(headers):
        if name.lower() in {"set-cookie", "x-ms-request-id", "x-azure-ref"}:
            headers.pop(name)
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """pytest-recording defaults; the record mode comes from --record-mode (default none)."""
    return {
        "before_record_response": _scrub_response,
        "filter_headers": ["authorization", "cookie", "set-cookie", "x-api-key"],
        "filter_query_parameters": ["token", "st", "se", "sig"],
        "decode_compressed_response": True,
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Tiny local COG / Zarr / GeoParquet / GeoJSON fixtures for the offline tier.

    Generated once per session by ``tests/fixtures/make_fixtures.py`` in a
    subprocess (see the note in that module about import order).
    """
    target = tmp_path_factory.mktemp("fixtures")
    script = Path(__file__).parent / "fixtures" / "make_fixtures.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
    subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        env=env,
    )
    return target


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
