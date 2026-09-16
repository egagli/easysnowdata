"""
Live-API tests for CDECClient.

These tests hit the real CDEC (California Data Exchange Center) endpoints.
An internet connection is required.
"""

import pytest

from easysnowdata.stations.clients.cdec import CDECClient
from easysnowdata.stations.clients.cdec.cdec_client import (
    _CDEC_DURATION_TO_INTERVAL,
    _INTERVAL_TO_CDEC_DURATION,
    _TYPE_TO_SENSORS,
    SENSORS,
)
from tests.stations.conftest import (
    BBOX_NORTHERN_CA,
    CDEC_PILLOWS,
    RECORD_KEYS,
    TEST_BEGIN,
    TEST_END,
)

pytestmark = pytest.mark.live  # hits real network endpoints


@pytest.fixture(scope="module")
def client():
    return CDECClient()


# ── SENSORS dict compliance ───────────────────────────────────────────────────


def test_sensors_required_fields():
    required = {"name", "short_name", "type", "units", "description", "notes", "source"}
    for num, info in SENSORS.items():
        missing = required - set(info.keys())
        assert not missing, f"SENSORS[{num}] missing keys: {missing}"


def test_sensors_types_are_valid():
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
    for num, info in SENSORS.items():
        assert info["type"] in valid_types, (
            f"SENSORS[{num}] has unknown type {info['type']!r}"
        )


def test_type_to_sensors_has_swe_snwd():
    assert "swe" in _TYPE_TO_SENSORS
    assert "snwd" in _TYPE_TO_SENSORS
    assert 82 in _TYPE_TO_SENSORS["swe"], "Sensor 82 should be primary SWE"
    assert 18 in _TYPE_TO_SENSORS["snwd"]


def test_swe_priority_order():
    """Sensor 82 should appear before sensor 3 for type 'swe'."""
    swe_sensors = _TYPE_TO_SENSORS["swe"]
    assert swe_sensors.index(82) < swe_sensors.index(3), (
        "Sensor 82 should take priority over sensor 3"
    )


def test_interval_dicts_roundtrip():
    for std, cdec in _INTERVAL_TO_CDEC_DURATION.items():
        assert cdec in _CDEC_DURATION_TO_INTERVAL, (
            f"{cdec!r} not in _CDEC_DURATION_TO_INTERVAL"
        )


# ── get_snow_courses / get_snow_pillows ───────────────────────────────────────


def test_get_snow_courses_returns_nonempty(client):
    courses = client.get_snow_courses()
    assert isinstance(courses, list)
    assert len(courses) > 100, "Expected >100 CCSS snow courses"


def test_get_snow_courses_fields(client):
    courses = client.get_snow_courses()
    required = {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "elevation_ft",
        "is_snow_course",
        "station_url",
    }
    for c in courses[:10]:
        missing = required - set(c.keys())
        assert not missing, f"Snow course {c.get('station_id')} missing: {missing}"
        assert c["is_snow_course"] is True
        assert c["is_snow_pillow"] is False


def test_get_snow_courses_station_id_format(client):
    courses = client.get_snow_courses()
    import re

    pattern = re.compile(r"^[A-Z0-9]{2,5}$")
    for c in courses[:20]:
        assert pattern.match(c["station_id"]), (
            f"Bad station_id format: {c['station_id']!r}"
        )


def test_get_snow_pillows_returns_nonempty(client):
    pillows = client.get_snow_pillows()
    assert isinstance(pillows, list)
    assert len(pillows) > 50, "Expected >50 CCSS snow pillow stations"


def test_get_snow_pillows_fields(client):
    pillows = client.get_snow_pillows()
    for p in pillows[:10]:
        assert p.get("is_snow_pillow") is True
        assert p.get("has_daily_swe") is True
        assert p.get("station_url", "").startswith("https://")


# ── get_stations ──────────────────────────────────────────────────────────────


def test_get_stations_returns_nonempty(client):
    stations = client.get_stations(sensors=(3, 18, 82))
    assert isinstance(stations, list)
    assert len(stations) > 200, "Expected >200 CDEC snow stations"


def test_get_stations_required_fields(client):
    stations = client.get_stations(sensors=(3, 18, 82))
    required = {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "elevation_ft",
        "elevation_m",
        "status",
        "has_daily_swe",
        "has_daily_snwd",
        "is_snow_course",
        "is_snow_pillow",
        "sensors",
        "station_url",
    }
    for sta in stations[:20]:
        missing = required - set(sta.keys())
        assert not missing, f"Station {sta.get('station_id')} missing: {missing}"


def test_get_stations_elevation_m_reasonable(client):
    stations = client.get_stations(sensors=(82,))
    for sta in stations[:10]:
        elev_m = sta.get("elevation_m")
        assert elev_m is not None
        assert 0 < elev_m < 5000, f"Implausible elevation_m={elev_m}"


def test_get_stations_station_url_format(client):
    stations = client.get_stations(sensors=(82,))
    for sta in stations[:10]:
        url = sta.get("station_url", "")
        assert "cdec.water.ca.gov" in url, f"Unexpected station_url: {url!r}"


def test_get_stations_sensors_is_sorted_list(client):
    stations = client.get_stations(sensors=(3, 18, 82))
    for sta in stations[:20]:
        sensors = sta.get("sensors", [])
        assert isinstance(sensors, list)
        assert sensors == sorted(sensors), "sensors should be sorted"


# ── get_all_stations ──────────────────────────────────────────────────────────


def test_get_all_stations_same_as_get_stations(client):
    all_sta = client.get_all_stations()
    direct = client.get_stations()
    assert len(all_sta) == len(direct)


def test_get_all_stations_bbox(client):
    stations = client.get_all_stations(bbox=BBOX_NORTHERN_CA)
    assert len(stations) > 10, "Expected stations in Northern CA bbox"
    for sta in stations:
        lat = float(sta["latitude"])
        lon = float(sta["longitude"])
        assert BBOX_NORTHERN_CA[1] <= lat <= BBOX_NORTHERN_CA[3]
        assert BBOX_NORTHERN_CA[0] <= lon <= BBOX_NORTHERN_CA[2]


# ── get_metadata ──────────────────────────────────────────────────────────────


def test_get_metadata_returns_sensor_inventory(client):
    meta = client.get_metadata(CDEC_PILLOWS[0])
    assert "sensor_inventory" in meta or "sensors" in meta, (
        "Metadata should include sensor inventory"
    )
    assert meta.get("station_id", "").upper() == CDEC_PILLOWS[0]


def test_get_metadata_fields(client):
    meta = client.get_metadata(CDEC_PILLOWS[0])
    for key in ("station_id", "name", "latitude", "longitude", "station_url"):
        assert key in meta, f"Metadata missing field: {key!r}"


# ── get_data ──────────────────────────────────────────────────────────────────


def test_get_data_flat_schema(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    for r in records:
        missing = RECORD_KEYS - set(r.keys())
        assert not missing, f"Record missing keys: {missing}\n  record={r}"


def test_get_data_swe_in_cm(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0
    for r in swe:
        assert r["units"] == "cm", f"SWE units should be cm, got {r['units']!r}"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 500, f"Implausible SWE value: {r['value']}"


def test_get_data_snwd_in_cm(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["snwd"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    snwd = [r for r in records if r["type"] == "snwd"]
    assert len(snwd) > 0
    for r in snwd:
        assert r["units"] == "cm"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 1000, f"Implausible snow depth: {r['value']}"


def test_get_data_swe_priority_no_sensor3_dates(client):
    """When sensor 82 is present, sensor 3 should not appear for same dates."""
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    # Group by date — should have at most one SWE record per date
    from collections import Counter

    date_counts = Counter(r["date"] for r in records if r["type"] == "swe")
    duplicates = {d: n for d, n in date_counts.items() if n > 1}
    assert not duplicates, (
        f"Multiple SWE records per date (priority not applied): {duplicates}"
    )


def test_get_data_station_id_is_uppercase(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["station_id"] == r["station_id"].upper()


def test_get_data_both_stations_represented(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    found = {r["station_id"] for r in records}
    for sid in CDEC_PILLOWS:
        assert sid in found, f"{sid} has no records in response"


def test_get_data_include_flags(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
        include_flags=True,
    )
    assert len(records) > 0
    for r in records:
        assert "flag" in r, "include_flags=True should add 'flag' key"


def test_get_data_variable_field_is_short_name(client):
    """The 'variable' field should be the sensor short name (e.g. 'SNO ADJ')."""
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0
    for r in swe:
        assert r["variable"] in {"SNO ADJ", "SNOW WC"}, (
            f"Unexpected variable name: {r['variable']!r}"
        )


def test_get_data_interval_field(client):
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["interval"] == "daily"


def test_get_data_bbox(client):
    records = client.get_data(
        bbox=BBOX_NORTHERN_CA,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    station_ids = {r["station_id"] for r in records}
    assert len(station_ids) > 3, "Expected multiple stations in Northern CA bbox"


def test_get_data_no_ids_no_bbox_raises(client):
    with pytest.raises(ValueError, match="station_ids or bbox"):
        client.get_data(variables=["swe"])


def test_get_data_native_sensor_name(client):
    """Passing 'SNO ADJ' (native short name) works like type 'swe'."""
    records = client.get_data(
        station_ids=CDEC_PILLOWS[:1],
        variables=["SNO ADJ"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    for r in records:
        assert r["type"] == "swe"
