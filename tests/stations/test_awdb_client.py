"""
Live-API tests for AWDBClient.

These tests hit the real USDA NRCS AWDB REST API — an internet connection
is required.  Station-level assertions use well-known SNOTEL sites that
have been operating for decades and have complete data.
"""

import pytest

from easysnowdata.stations.clients.awdb import AWDBClient, AWDBError
from easysnowdata.stations.clients.awdb.awdb_client import (
    _AWDB_DURATION_TO_INTERVAL,
    _INTERVAL_TO_AWDB_DURATION,
    _TYPE_TO_ELEMENTS,
    VARIABLES,
)
from tests.stations.conftest import (
    AWDB_SNOTEL_CO,
    AWDB_SNOTEL_WA,
    BBOX_COLORADO,
    RECORD_KEYS,
    TEST_BEGIN,
    TEST_END,
)

pytestmark = pytest.mark.live  # hits real network endpoints


@pytest.fixture(scope="module")
def client():
    return AWDBClient()


# ── VARIABLES dict compliance ─────────────────────────────────────────────────


def test_variables_required_fields():
    """Every entry in VARIABLES has all required keys."""
    required = {"name", "type", "units", "description", "notes", "source"}
    for code, info in VARIABLES.items():
        missing = required - set(info.keys())
        assert not missing, f"VARIABLES[{code!r}] missing keys: {missing}"


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
    for code, info in VARIABLES.items():
        assert info["type"] in valid_types, (
            f"VARIABLES[{code!r}] has unknown type {info['type']!r}"
        )


def test_type_to_elements_covers_all_swe_snwd():
    assert "swe" in _TYPE_TO_ELEMENTS
    assert "snwd" in _TYPE_TO_ELEMENTS
    assert "WTEQ" in _TYPE_TO_ELEMENTS["swe"]
    assert "SNWD" in _TYPE_TO_ELEMENTS["snwd"]


def test_interval_dicts_roundtrip():
    for std, awdb in _INTERVAL_TO_AWDB_DURATION.items():
        assert awdb in _AWDB_DURATION_TO_INTERVAL, (
            f"{awdb!r} not in _AWDB_DURATION_TO_INTERVAL"
        )


# ── get_all_stations ──────────────────────────────────────────────────────────


def test_get_all_stations_returns_nonempty_list(client):
    stations = client.get_all_stations()
    assert isinstance(stations, list)
    assert len(stations) > 500, "Expected >500 AWDB stations total"


def test_get_all_stations_required_fields(client):
    stations = client.get_all_stations()
    required = {
        "stationTriplet",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "status",
        "station_url",
    }
    for sta in stations[:20]:  # spot-check first 20
        missing = required - set(sta.keys())
        assert not missing, f"Station {sta.get('stationTriplet')} missing: {missing}"


def test_get_all_stations_elevation_m_is_metric(client):
    stations = client.get_all_stations()
    snotel = [s for s in stations if s.get("networkCode") == "SNTL"][:10]
    for sta in snotel:
        elev_m = sta.get("elevation_m")
        assert elev_m is not None, f"{sta['stationTriplet']} missing elevation_m"
        assert 0 < elev_m < 5000, f"Implausible elevation_m={elev_m}"


def test_get_all_stations_station_url_for_snotel(client):
    stations = client.get_all_stations()
    snotel = [s for s in stations if s.get("networkCode") == "SNTL"][:5]
    for sta in snotel:
        url = sta.get("station_url", "")
        assert url.startswith("https://"), (
            f"{sta['stationTriplet']} has bad station_url: {url!r}"
        )
        assert str(sta.get("stationId")) in url


def test_get_all_stations_status_field(client):
    stations = client.get_all_stations()
    for sta in stations[:50]:
        assert sta.get("status") in {"Active", "Inactive"}, (
            f"Unexpected status: {sta.get('status')!r}"
        )


def test_get_all_stations_active_only_fewer(client):
    all_sta = client.get_all_stations(active_only=False)
    active = client.get_all_stations(active_only=True)
    assert len(active) < len(all_sta), "active_only=True should return fewer stations"


def test_get_all_stations_bbox_colorado(client):
    stations = client.get_all_stations(bbox=BBOX_COLORADO)
    assert len(stations) > 50, "Expected >50 stations in Colorado bbox"
    for sta in stations:
        lat = float(sta["latitude"])
        lon = float(sta["longitude"])
        assert BBOX_COLORADO[1] <= lat <= BBOX_COLORADO[3], f"lat={lat} outside bbox"
        assert BBOX_COLORADO[0] <= lon <= BBOX_COLORADO[2], f"lon={lon} outside bbox"


# ── get_stations ──────────────────────────────────────────────────────────────


def test_get_stations_by_network_and_state(client):
    stations = client.get_stations(networks=["SNTL"], states=["CO"])
    assert len(stations) > 100
    for sta in stations:
        assert sta["networkCode"] == "SNTL"
        assert sta["stateCode"] == "CO"


def test_get_stations_by_triplet(client):
    stations = client.get_stations(station_triplets=AWDB_SNOTEL_CO)
    triplets = {s["stationTriplet"] for s in stations}
    for t in AWDB_SNOTEL_CO:
        assert t in triplets, f"{t} not found in response"


# ── get_metadata ──────────────────────────────────────────────────────────────


def test_get_metadata_returns_station_elements(client):
    meta = client.get_metadata(AWDB_SNOTEL_CO, elements=["WTEQ", "SNWD"])
    assert len(meta) == len(AWDB_SNOTEL_CO)
    for sta in meta:
        assert "stationElements" in sta, "stationElements missing"
        elem_codes = {e["elementCode"] for e in sta["stationElements"]}
        assert "WTEQ" in elem_codes or "SNWD" in elem_codes


def test_get_metadata_all_elements(client):
    meta = client.get_metadata(AWDB_SNOTEL_CO[:1], elements="*")
    assert len(meta) == 1
    assert len(meta[0]["stationElements"]) > 2, (
        "Expected multiple elements for full metadata request"
    )


# ── get_data ──────────────────────────────────────────────────────────────────


def test_get_data_flat_schema(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0, "Expected at least one record"
    for r in records:
        missing = RECORD_KEYS - set(r.keys())
        assert not missing, f"Record missing keys: {missing}\n  record={r}"


def test_get_data_swe_type_and_cm_units(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0
    for r in swe:
        assert r["units"] == "cm", f"SWE units should be cm, got {r['units']!r}"
        assert r["variable"] == "WTEQ"
        assert r["interval"] == "daily"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 500, f"Implausible SWE value: {r['value']}"


def test_get_data_snwd_type(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["snwd"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    snwd = [r for r in records if r["type"] == "snwd"]
    assert len(snwd) > 0
    for r in snwd:
        assert r["units"] == "cm"
        assert r["variable"] == "SNWD"


def test_get_data_native_element_code(client):
    """Passing a native code ('WTEQ') works the same as type 'swe'."""
    by_type = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    by_code = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["WTEQ"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    dates_type = {r["date"] for r in by_type}
    dates_code = {r["date"] for r in by_code}
    assert dates_type == dates_code, (
        "Type-resolved and native-code results should match"
    )


def test_get_data_multiple_stations(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO,
        variables=["swe", "snwd"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    station_ids_found = {r["station_id"] for r in records}
    for t in AWDB_SNOTEL_CO:
        assert t in station_ids_found, f"{t} has no records"


def test_get_data_station_id_is_triplet(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["station_id"] == AWDB_SNOTEL_CO[0]


def test_get_data_include_flags(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_CO[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
        include_flags=True,
    )
    assert len(records) > 0
    for r in records:
        assert "flag" in r, "include_flags=True should add 'flag' key"


def test_get_data_bbox(client):
    records = client.get_data(
        bbox=BBOX_COLORADO,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 100, "Expected many records for Colorado bbox"
    station_ids = {r["station_id"] for r in records}
    assert len(station_ids) > 20


def test_get_data_no_ids_no_bbox_raises(client):
    with pytest.raises(ValueError, match="station_ids or bbox"):
        client.get_data(variables=["swe"])


def test_get_data_hourly_interval(client):
    records = client.get_data(
        station_ids=AWDB_SNOTEL_WA,
        variables=["swe"],
        interval="hourly",
        begin_date=TEST_BEGIN,
        end_date="2024-01-02",
    )
    hourly = [r for r in records if r["interval"] == "hourly"]
    # Not all stations have hourly — just check schema if any returned
    for r in hourly:
        assert RECORD_KEYS.issubset(r.keys())


def test_get_normals_returns_records(client):
    data = client.get_normals(
        triplets=AWDB_SNOTEL_CO[:1],
        elements=["WTEQ"],
        normal_period="1991-2020",
    )
    assert len(data) == 1
    assert "data" in data[0]
    values = data[0]["data"][0]["values"]
    assert len(values) > 100, "Normals should cover a full water year"


def test_get_data_by_water_year(client):
    data = client.get_data_by_water_year(
        triplets=AWDB_SNOTEL_CO[:1],
        elements=["WTEQ"],
        water_year=2023,
    )
    assert len(data) == 1
    values = data[0]["data"][0]["values"]
    assert len(values) == 365 or len(values) == 366


# ── Error handling ────────────────────────────────────────────────────────────


def test_awdb_error_on_bad_triplet(client):
    """A completely invalid triplet should either return empty or raise AWDBError."""
    try:
        result = client.get_data(
            station_ids=["XXXXX:ZZ:SNTL"],
            variables=["swe"],
            begin_date=TEST_BEGIN,
            end_date=TEST_END,
        )
        # Empty result is also acceptable
        assert isinstance(result, list)
    except AWDBError:
        pass  # also acceptable
