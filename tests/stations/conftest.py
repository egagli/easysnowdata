"""Fixtures and well-known stations for the Phase 3 (stations) tests.

Three kinds of test live here:

* the **vendored client tests** (``test_*_client.py``), carried over from
  ``global_snow_networks`` with their imports rewritten — all ``live``, since
  every one of them talks to a real network API;
* ``test_clients_offline.py``, the client-contract unit tests, which touch no
  socket;
* the **adapter, archive and shim tests**, which replay pytest-recording
  cassettes under ``tests/stations/cassettes/`` or read the tiny synthetic
  archive written by ``tests/stations/fixtures/make_fixtures.py``.

Cassettes are recorded unsigned; refresh with::

    pixi run -e dev pytest tests/stations -m recorded --record-mode=rewrite
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent

RAINIER = (-121.94, 46.72, -121.54, 46.99)

# ── Well-known test stations (from global_snow_networks/tests/conftest.py) ────

# AWDB triplets
AWDB_SNOTEL_CO = ["303:CO:SNTL", "457:CO:SNTL"]  # Bear Lake, Grand Mesa
AWDB_SNOTEL_WA = ["335:WA:SNTL"]  # Fish Lake

# CDEC station IDs
CDEC_PILLOWS = ["QUA", "BLC"]  # Quartz Basin, Blue Canyon
CDEC_COURSE = ["HNT"]  # Huntington Lake snow course

# DataBC location IDs
DATABC_ASWS = ["1A01P", "1E08P"]  # Tupper, Yellowhead
DATABC_MSS = ["1A06A", "1A10"]  # Tupper Creek MSS sites

# NVE station IDs — snow pillows with daily SWE (param 2003) and snow
# depth (param 2002).  NB: 2.11.0 / 12.228.0 are river gauges, not snow.
NVE_SWE_STATIONS = ["12.142.0", "121.2.0"]  # Bakko, Maurhaugen-Oppdal
NVE_SNWD_STATION = ["12.142.0"]  # Bakko — has snow depth too

# Yukon AquaCache location codes.
# Automated snow-weather stations (snow-pillow SWE + snow depth).
YUKON_AWS = ["09AA-M1", "09BA-M7"]  # Tagish, Twin Creeks North (SWE since 1988)
# Manual snow courses (periodic Feb/Mar/Apr/May surveys, no daily CSVs).
YUKON_COURSES = ["08AA-SC01", "09CB-SC01"]  # Canyon Lake, Beaver Creek (1975–)
# ECCC climate station mirrored into AquaCache — snow depth only.
YUKON_ECCC = ["10MD-MET-00004"]  # Herschel Island, 69.57°N
# A course the Yukon Snow Survey operates outside Yukon, so `state` != "YT".
YUKON_NON_YT_COURSE = "09AA-SC04"  # Atlin (B.C.) Snow Course

# Date window for data tests (known-good winter period)
TEST_BEGIN = "2024-01-01"
TEST_END = "2024-01-15"

# Bounding boxes
BBOX_COLORADO = (-109.1, 36.9, -102.0, 41.1)
BBOX_NORTHERN_CA = (-122.5, 38.5, -119.5, 41.5)
BBOX_BC_INTERIOR = (-120.5, 49.5, -119.0, 51.0)
BBOX_NORWAY = (4.5, 57.5, 31.5, 71.5)
# Wide enough to include the Yukon Snow Survey's BC and Alaska courses
# (Eaglecrest at 58.28°N, Boundary at 141.45°W).
BBOX_YUKON = (-142.0, 58.0, -123.0, 70.0)

# Required record keys for standardized get_data() output
RECORD_KEYS = {"station_id", "date", "variable", "type", "value", "units", "interval"}

# Two Mount Rainier SNOTEL stations, in the two spellings that matter: the
# global code the inventory and the archive use, and the AWDB triplet the
# client speaks.
PARADISE_CODE = "679_WA_SNTL"
PARADISE_TRIPLET = "679:WA:SNTL"
MORSE_LAKE_CODE = "642_WA_SNTL"


# ── tier machinery ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vcr_cassette_dir(request) -> str:
    """Keep this directory's cassettes next to its tests."""
    return str(_HERE / "cassettes" / request.module.__name__.rsplit(".", 1)[-1])


@pytest.fixture(scope="session")
def station_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Synthetic station inventory + CSV archive (paths keyed by name).

    Written once per session by ``fixtures/make_fixtures.py`` in a subprocess,
    for the same reason the other tiers do it: writing a GeoJSON through
    geopandas after it has been imported alongside rasterio is what crashes
    GDAL in this environment.
    """
    target = tmp_path_factory.mktemp("station-fixtures")
    script = _HERE / "fixtures" / "make_fixtures.py"
    env = {**os.environ, "PYTHONPATH": str(_HERE.parents[1])}
    result = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    paths: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        name, _, path = line.partition(": ")
        if path:
            paths[name] = Path(path)
    return paths


@pytest.fixture
def no_network(monkeypatch):
    """Make any accidental client call fail loudly."""
    from easysnowdata.stations import clients

    def boom(*args, **kwargs):  # pragma: no cover — only runs on a bug
        raise AssertionError("network call attempted in an offline test")

    for name in (
        "AWDBClient",
        "CDECClient",
        "DataBCClient",
        "NVEClient",
        "YukonClient",
    ):
        monkeypatch.setattr(getattr(clients, name), "get_all_stations", boom)
        monkeypatch.setattr(getattr(clients, name), "get_data", boom)
    return boom
