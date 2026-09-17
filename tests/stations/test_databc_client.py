"""
Live-API tests for DataBCClient.

These tests hit the real BC government data endpoints (WFS + CSV files).
An internet connection is required.
"""

import pytest

from easysnowdata.stations.clients.databc import DataBCClient
from easysnowdata.stations.clients.databc.databc_client import (
    _TYPE_TO_DATABC_VARS,
    VARIABLES,
)
from tests.stations.conftest import (
    BBOX_BC_INTERIOR,
    DATABC_ASWS,
    DATABC_MSS,
    RECORD_KEYS,
    TEST_BEGIN,
    TEST_END,
)

pytestmark = pytest.mark.live  # hits real network endpoints


@pytest.fixture(scope="module")
def client():
    return DataBCClient()


# ── VARIABLES dict compliance ─────────────────────────────────────────────────


def test_variables_required_fields():
    required = {"name", "type", "units", "description", "notes", "source"}
    for key, info in VARIABLES.items():
        missing = required - set(info.keys())
        assert not missing, f"VARIABLES[{key!r}] missing keys: {missing}"


def test_variables_types_are_valid():
    valid_types = {
        "swe",
        "snwd",
        "temp",
        "temp_max",
        "temp_min",
        "precip",
        "rh",
        "wind_spd",
        "wind_gust",
        "wind_dir",
        "wind_run",
        "solar",
        "baro",
        "density",
        "snow_line",
        "other",
    }
    for key, info in VARIABLES.items():
        assert info["type"] in valid_types, (
            f"VARIABLES[{key!r}] has unknown type {info['type']!r}"
        )


def test_type_to_databc_vars_has_swe_snwd():
    assert "swe" in _TYPE_TO_DATABC_VARS
    assert "snwd" in _TYPE_TO_DATABC_VARS
    assert "swe_mm" in _TYPE_TO_DATABC_VARS["swe"]
    assert "snwd_cm" in _TYPE_TO_DATABC_VARS["snwd"]


# ── get_asws_stations ─────────────────────────────────────────────────────────


def test_get_asws_stations_returns_nonempty(client):
    stations = client.get_asws_stations()
    assert isinstance(stations, list)
    assert len(stations) > 100, "Expected >100 ASWS stations"


def test_get_asws_stations_ids_nonempty(client):
    stations = client.get_asws_stations()
    for sta in stations:
        assert sta["location_id"], "ASWS location_id should not be empty"


def test_get_asws_stations_required_fields(client):
    stations = client.get_asws_stations()
    required = {
        "location_id",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "status",
        "station_type",
        "station_url",
    }
    for sta in stations[:20]:
        missing = required - set(sta.keys())
        assert not missing, f"ASWS station {sta.get('location_id')} missing: {missing}"


def test_get_asws_stations_station_type(client):
    stations = client.get_asws_stations()
    for sta in stations[:20]:
        assert sta["station_type"] == "ASWS"


def test_get_asws_stations_elevation_m_metric(client):
    stations = client.get_asws_stations()
    for sta in stations[:10]:
        elev = sta.get("elevation_m")
        assert elev is not None, f"{sta['location_id']} missing elevation_m"
        assert 0 < elev < 3500, f"Implausible elevation_m={elev}"


def test_get_asws_stations_station_url_format(client):
    stations = client.get_asws_stations()
    for sta in stations[:10]:
        url = sta.get("station_url", "")
        assert url.startswith("https://"), (
            f"Bad station_url for {sta['location_id']}: {url!r}"
        )


# ── get_mss_stations ──────────────────────────────────────────────────────────


def test_get_mss_stations_returns_nonempty(client):
    stations = client.get_mss_stations()
    assert isinstance(stations, list)
    assert len(stations) > 200, "Expected >200 MSS sites"


def test_get_mss_stations_ids_no_trailing_p(client):
    stations = client.get_mss_stations()
    for sta in stations[:20]:
        # MSS IDs should NOT end in P
        assert not sta["location_id"].upper().endswith("P"), (
            f"MSS ID should not end in P: {sta['location_id']!r}"
        )


def test_get_mss_stations_required_fields(client):
    stations = client.get_mss_stations()
    required = {
        "location_id",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "status",
        "station_type",
        "station_url",
    }
    for sta in stations[:20]:
        missing = required - set(sta.keys())
        assert not missing, f"MSS site {sta.get('location_id')} missing: {missing}"


def test_get_mss_stations_station_type(client):
    stations = client.get_mss_stations()
    for sta in stations[:20]:
        assert sta["station_type"] == "MSS"


# ── get_all_stations ──────────────────────────────────────────────────────────


def test_get_all_stations_combines_asws_and_mss(client):
    asws = client.get_asws_stations()
    mss = client.get_mss_stations()
    all_sta = client.get_all_stations()
    assert len(all_sta) == len(asws) + len(mss), (
        "get_all_stations() should be ASWS + MSS"
    )


def test_get_all_stations_active_only_fewer(client):
    all_sta = client.get_all_stations(active_only=False)
    active = client.get_all_stations(active_only=True)
    assert len(active) < len(all_sta), "active_only=True should return fewer stations"


def test_get_all_stations_bbox(client):
    stations = client.get_all_stations(bbox=BBOX_BC_INTERIOR)
    assert len(stations) > 5, "Expected stations in BC Interior bbox"
    for sta in stations:
        lat = float(sta["latitude"])
        lon = float(sta["longitude"])
        assert BBOX_BC_INTERIOR[1] <= lat <= BBOX_BC_INTERIOR[3]
        assert BBOX_BC_INTERIOR[0] <= lon <= BBOX_BC_INTERIOR[2]


# ── get_data — ASWS daily ─────────────────────────────────────────────────────


def test_get_data_asws_flat_schema(client):
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0, "Expected records for ASWS station"
    for r in records:
        missing = RECORD_KEYS - set(r.keys())
        assert not missing, f"Record missing keys: {missing}\n  record={r}"


def test_get_data_swe_converted_to_cm(client):
    """swe_mm should be converted to cm (÷ 10) by get_data()."""
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0
    for r in swe:
        assert r["units"] == "cm", f"SWE units should be cm, got {r['units']!r}"
        assert r["variable"] == "swe_mm", (
            f"Native variable should be 'swe_mm', got {r['variable']!r}"
        )
        if r["value"] is not None:
            assert 0 <= r["value"] <= 300, f"Implausible SWE (cm): {r['value']}"


def test_get_data_snwd(client):
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["snwd"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    snwd = [r for r in records if r["type"] == "snwd"]
    assert len(snwd) > 0
    for r in snwd:
        assert r["units"] == "cm"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 800


def test_get_data_station_id_matches_input(client):
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["station_id"] == DATABC_ASWS[0]


def test_get_data_multiple_asws_stations(client):
    records = client.get_data(
        station_ids=DATABC_ASWS,
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    found = {r["station_id"] for r in records}
    for lid in DATABC_ASWS:
        assert lid in found, f"{lid} has no records"


def test_get_data_include_flags(client):
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
        include_flags=True,
    )
    assert len(records) > 0
    for r in records:
        assert "flag" in r


def test_get_data_native_variable_key(client):
    """Passing 'swe_mm' (native key) works like type 'swe'."""
    by_type = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    by_native = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe_mm"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    dates_type = {r["date"] for r in by_type}
    dates_native = {r["date"] for r in by_native}
    assert dates_type == dates_native


def test_get_data_interval_field_is_daily(client):
    records = client.get_data(
        station_ids=DATABC_ASWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["interval"] == "daily"


def test_get_data_bbox(client):
    records = client.get_data(
        bbox=BBOX_BC_INTERIOR,
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    station_ids = {r["station_id"] for r in records}
    assert len(station_ids) >= 1


def test_get_data_no_ids_no_bbox_raises(client):
    with pytest.raises(ValueError, match="station_ids or bbox"):
        client.get_data(variables=["swe"])


# ── get_data — MSS periodic ───────────────────────────────────────────────────


def test_get_data_mss_periodic(client):
    records = client.get_data(
        station_ids=DATABC_MSS[:1],
        variables=["swe"],
        interval="periodic",
    )
    # MSS sites don't always have data every year — just check schema
    for r in records:
        missing = RECORD_KEYS - set(r.keys())
        assert not missing, f"MSS record missing keys: {missing}"
        assert r["interval"] == "periodic"
        assert r["type"] == "swe"
        if r["value"] is not None:
            assert r["units"] == "cm"


# ── _get_asws_daily_data (private DataFrame layer) ────────────────────────────


def test_get_asws_daily_data_returns_dataframe(client):
    import pandas as pd

    df = client._get_asws_daily_data(
        location_ids=DATABC_ASWS,
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
        archive=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert "location_id" in df.columns
    assert "swe_mm" in df.columns or "date" in df.columns
    found_ids = set(df["location_id"].unique())
    for lid in DATABC_ASWS:
        assert lid in found_ids, f"{lid} not in asws_daily_data result"
