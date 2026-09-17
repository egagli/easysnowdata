"""
Live-API tests for NVEClient.

These tests hit the real NVE HydAPI (https://hydapi.nve.no/) — an internet
connection is required.  Station-level assertions use well-known Norwegian
snow monitoring sites that have been operating for multiple seasons.
"""

import pytest

from easysnowdata.stations.clients.nve import NVEClient, NVEError
from easysnowdata.stations.clients.nve.nve_client import (
    _INTERVAL_TO_RESOLUTION,
    _MAX_POINTS_PER_REQUEST,
    _PARAM_TO_VAR,
    _RESOLUTION_TO_INTERVAL,
    _TYPE_TO_NVE_VARS,
    _VAR_TO_PARAM,
    VARIABLES,
    _enrich_station,
    _reference_windows,
)
from tests.stations.conftest import (
    BBOX_NORWAY,
    NVE_SNWD_STATION,
    NVE_SWE_STATIONS,
    RECORD_KEYS,
    TEST_BEGIN,
    TEST_END,
)

# Every NVE endpoint needs the API key, so these skip without one
# (easysnowdata.auth's `nve` provider does the detection).
pytestmark = [pytest.mark.live, pytest.mark.requires_nve]


@pytest.fixture(scope="module")
def client():
    return NVEClient()


# ── VARIABLES dict compliance ─────────────────────────────────────────────────


def test_variables_required_fields():
    """Every entry in VARIABLES has all required keys."""
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


def test_type_to_nve_vars_has_swe_snwd():
    assert "swe" in _TYPE_TO_NVE_VARS
    assert "snwd" in _TYPE_TO_NVE_VARS
    assert "swe_m" in _TYPE_TO_NVE_VARS["swe"]
    assert "snwd_cm" in _TYPE_TO_NVE_VARS["snwd"]


def test_param_to_var_roundtrip():
    """Every entry in _PARAM_TO_VAR should have a reverse mapping."""
    for param_id, var_key in _PARAM_TO_VAR.items():
        assert _VAR_TO_PARAM.get(var_key) == param_id, (
            f"_VAR_TO_PARAM[{var_key!r}] does not round-trip to {param_id}"
        )


def test_interval_dicts_roundtrip():
    for std, res in _INTERVAL_TO_RESOLUTION.items():
        assert res in _RESOLUTION_TO_INTERVAL, f"{res} not in _RESOLUTION_TO_INTERVAL"


def test_swe_units_is_cm():
    assert VARIABLES["swe_m"]["units"] == "cm", (
        "swe_m should report cm (converted from metres)"
    )


def test_snwd_units_is_cm():
    assert VARIABLES["snwd_cm"]["units"] == "cm"


# ── _enrich_station (offline) ─────────────────────────────────────────────────


def _raw_station(series_list):
    return {
        "stationId": "2.11.0",
        "stationName": "Filefjell",
        "latitude": 61.18,
        "longitude": 8.11,
        "elevation": 955,
        "active": True,
        "seriesList": series_list,
    }


def test_enrich_station_reads_parameter_key():
    """/Stations seriesList items use the key ``parameter``, not ``parameterId``."""
    sta = _enrich_station(
        _raw_station(
            [
                {"parameter": 2003, "resolutionList": [{"resTime": 1440}]},
                {"parameter": 2002, "resolutionList": [{"resTime": 0}]},
            ]
        )
    )
    assert sta["parameters"] == [2002, 2003]


def test_enrich_station_daily_parameters_from_resolution_list():
    sta = _enrich_station(
        _raw_station(
            [
                {
                    "parameter": 2003,
                    "resolutionList": [{"resTime": 0}, {"resTime": 1440}],
                },
                {
                    "parameter": 2002,
                    "resolutionList": [{"resTime": 0}, {"resTime": 60}],
                },
            ]
        )
    )
    assert sta["daily_parameters"] == [2003], (
        "only parameters with a 1440-minute resolution are daily"
    )


def test_enrich_station_empty_series_list():
    sta = _enrich_station(_raw_station([]))
    assert sta["parameters"] == []
    assert sta["daily_parameters"] == []


def test_enrich_station_coordinate_override_applied():
    """Known-wrong HydAPI coords (Nepal stations) fall back to overrides."""
    raw = _raw_station([])
    raw.update(stationId="1977.1.4", latitude=28.15441, longitude=25.5625)
    sta = _enrich_station(raw)
    assert sta["coordinates_overridden"] is True
    assert sta["longitude"] == 85.5625, "longitude must be corrected +60°"
    assert sta["latitude"] == 28.15441


def test_enrich_station_coordinate_override_skipped_when_upstream_fixed():
    """If HydAPI starts reporting the correct position, keep upstream."""
    raw = _raw_station([])
    raw.update(stationId="1977.1.4", latitude=28.15441, longitude=85.5625)
    sta = _enrich_station(raw)
    assert sta["coordinates_overridden"] is False
    assert sta["longitude"] == 85.5625


def test_enrich_station_no_override_for_normal_station():
    sta = _enrich_station(_raw_station([]))
    assert sta["coordinates_overridden"] is False
    assert sta["latitude"] == 61.18


# ── _reference_windows (offline) ──────────────────────────────────────────────


def test_reference_windows_no_dates_means_latest_only():
    assert _reference_windows(None, None, "2010-01-01", 1440) == [(None, None)]


def test_reference_windows_clips_begin_to_data_from():
    wins = _reference_windows("1950-01-01", "2024-06-01", "2010-01-01", 1440)
    assert wins == [("2010-01-01", "2024-06-01")]


def test_reference_windows_open_end_is_today_not_data_to():
    """dataToTime can lag the newest observations — an omitted end_date
    must extend to today, never be clipped by series metadata."""
    from datetime import date

    wins = _reference_windows("2024-01-01", None, "2010-01-01", 1440)
    assert wins[-1][1] == date.today().isoformat()


def test_reference_windows_begin_after_end_returns_empty():
    assert _reference_windows("2024-06-01", "2024-01-01", "", 1440) == []


def test_reference_windows_chunks_long_daily_ranges():
    wins = _reference_windows("1950-01-01", "2026-07-01", "", 1440)
    assert len(wins) > 1, "76 years of daily data must be split"
    # Windows are contiguous, ordered, and within the requested period
    assert wins[0][0] == "1950-01-01"
    assert wins[-1][1] == "2026-07-01"
    for (_, prev_end), (next_begin, _) in zip(wins, wins[1:]):
        from datetime import date, timedelta

        assert date.fromisoformat(next_begin) == (
            date.fromisoformat(prev_end) + timedelta(days=1)
        )
    # Each window stays under the per-request point budget
    from datetime import date

    for begin, end in wins:
        n_days = (date.fromisoformat(end) - date.fromisoformat(begin)).days + 1
        assert n_days <= _MAX_POINTS_PER_REQUEST


def test_reference_windows_hourly_budget():
    wins = _reference_windows("2020-01-01", "2024-01-01", "", 60)
    from datetime import date

    for begin, end in wins:
        n_days = (date.fromisoformat(end) - date.fromisoformat(begin)).days + 1
        assert n_days * 24 <= _MAX_POINTS_PER_REQUEST + 24


def test_reference_windows_open_start():
    assert _reference_windows(None, "2024-06-01", "", 1440) == [(None, "2024-06-01")]


# ── get_all_stations ──────────────────────────────────────────────────────────


def test_get_all_stations_returns_nonempty_list(client):
    stations = client.get_all_stations()
    assert isinstance(stations, list)
    assert len(stations) > 20, "Expected >20 NVE snow stations"


def test_get_all_stations_required_fields(client):
    stations = client.get_all_stations()
    required = {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "status",
        "station_url",
    }
    for sta in stations[:20]:
        missing = required - set(sta.keys())
        assert not missing, f"Station {sta.get('station_id')} missing fields: {missing}"


def test_get_all_stations_station_url_format(client):
    stations = client.get_all_stations()
    for sta in stations[:10]:
        url = sta.get("station_url", "")
        assert url.startswith("https://"), (
            f"Station {sta.get('station_id')} has bad station_url: {url!r}"
        )
        assert sta["station_id"] in url, (
            f"station_id {sta['station_id']!r} not found in URL {url!r}"
        )


def test_get_all_stations_latitude_in_norway(client):
    stations = client.get_all_stations()
    for sta in stations[:20]:
        lat = sta.get("latitude")
        if lat is not None:
            assert 55 <= lat <= 72, f"Latitude {lat} outside Norway bounds"


def test_get_all_stations_status_field(client):
    stations = client.get_all_stations()
    for sta in stations[:30]:
        assert sta.get("status") in {"Active", "Inactive"}, (
            f"Unexpected status: {sta.get('status')!r}"
        )


def test_get_all_stations_active_only_fewer(client):
    all_sta = client.get_all_stations(active_only=False)
    active = client.get_all_stations(active_only=True)
    assert len(active) <= len(all_sta), (
        "active_only=True should return same or fewer stations"
    )


def test_get_all_stations_bbox_norway(client):
    stations = client.get_all_stations(bbox=BBOX_NORWAY)
    assert len(stations) > 10, "Expected stations in Norway bbox"
    for sta in stations:
        lat = sta.get("latitude")
        lon = sta.get("longitude")
        if lat is not None and lon is not None:
            assert BBOX_NORWAY[1] <= lat <= BBOX_NORWAY[3], (
                f"lat={lat} outside Norway bbox"
            )
            assert BBOX_NORWAY[0] <= lon <= BBOX_NORWAY[2], (
                f"lon={lon} outside Norway bbox"
            )


# ── get_stations ──────────────────────────────────────────────────────────────


def test_get_stations_by_swe_parameter(client):
    stations = client.get_stations(parameter_ids=2003)
    assert isinstance(stations, list)
    assert len(stations) > 10, "Expected >10 SWE stations"


def test_get_stations_by_snwd_parameter(client):
    stations = client.get_stations(parameter_ids=2002)
    assert isinstance(stations, list)
    assert len(stations) > 10, "Expected >10 snow depth stations"


def test_get_stations_by_multiple_parameters(client):
    single = client.get_stations(parameter_ids=2003)
    multi = client.get_stations(parameter_ids=[2002, 2003])
    assert len(multi) >= len(single), (
        "Multiple parameter filter should return >= single parameter results"
    )


# ── get_metadata ──────────────────────────────────────────────────────────────


def test_get_metadata_returns_station(client):
    meta = client.get_metadata(NVE_SWE_STATIONS[0])
    assert meta["station_id"] == NVE_SWE_STATIONS[0]


def test_get_metadata_required_fields(client):
    meta = client.get_metadata(NVE_SWE_STATIONS[0])
    required = {"station_id", "name", "latitude", "longitude", "station_url"}
    for key in required:
        assert key in meta, f"Metadata missing field: {key!r}"


def test_get_metadata_invalid_station_raises(client):
    with pytest.raises(NVEError):
        client.get_metadata("XXXXX_INVALID_STATION_99999")


# ── get_data ──────────────────────────────────────────────────────────────────


def test_get_data_flat_schema(client):
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
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
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0, "Expected at least one SWE record"
    for r in swe:
        assert r["units"] == "cm", f"SWE units should be cm, got {r['units']!r}"
        assert r["variable"] == "swe_m"
        assert r["interval"] == "daily"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 500, f"Implausible SWE value: {r['value']} cm"


def test_get_data_snwd_type(client):
    records = client.get_data(
        station_ids=NVE_SNWD_STATION,
        variables=["snwd"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    snwd = [r for r in records if r["type"] == "snwd"]
    assert len(snwd) > 0, "Expected at least one snow depth record"
    for r in snwd:
        assert r["units"] == "cm"
        assert r["variable"] == "snwd_cm"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 1000, f"Implausible snow depth: {r['value']} cm"


def test_get_data_native_variable_key(client):
    """Passing 'swe_m' (native key) works the same as type 'swe'."""
    by_type = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    by_key = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe_m"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    dates_type = {r["date"] for r in by_type}
    dates_key = {r["date"] for r in by_key}
    assert dates_type == dates_key, (
        "Type-resolved and native-key results should have the same dates"
    )


def test_get_data_multiple_stations(client):
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    station_ids_found = {r["station_id"] for r in records}
    for sid in NVE_SWE_STATIONS:
        assert sid in station_ids_found, f"{sid} has no records"


def test_get_data_station_id_field(client):
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["station_id"] == NVE_SWE_STATIONS[0]


def test_get_data_include_flags(client):
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
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
        bbox=BBOX_NORWAY,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0, "Expected records for Norway bbox"
    station_ids = {r["station_id"] for r in records}
    assert len(station_ids) > 3, "Expected multiple stations in Norway bbox"


def test_get_data_no_ids_no_bbox_raises(client):
    with pytest.raises(ValueError, match="station_ids or bbox"):
        client.get_data(variables=["swe"])


def test_get_data_interval_field_daily(client):
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["interval"] == "daily"


def test_get_data_date_format(client):
    """Date field should be YYYY-MM-DD format."""
    import re

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    records = client.get_data(
        station_ids=NVE_SWE_STATIONS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    for r in records:
        assert date_re.match(r["date"]), f"Unexpected date format: {r['date']!r}"


# ── Error handling ────────────────────────────────────────────────────────────


def test_nve_error_on_bad_station(client):
    """An invalid station ID should produce empty results or raise NVEError."""
    try:
        result = client.get_data(
            station_ids=["XXXXX_INVALID_99999"],
            variables=["swe"],
            begin_date=TEST_BEGIN,
            end_date=TEST_END,
        )
        # Empty result is acceptable
        assert isinstance(result, list)
    except NVEError:
        pass  # also acceptable
