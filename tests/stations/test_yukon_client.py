"""
Live-API tests for YukonClient.

These tests hit the real Yukon Water Data (AquaCache) API
(https://service.yukon.ca/water-data/api/v1) — an internet connection is
required.  No API key is needed; the service is fully open.

Station-level assertions use long-running Yukon Snow Survey sites: the
Tagish and Twin Creeks North automated snow-weather stations (snow-pillow
SWE since 1988), the Canyon Lake and Rose Creek snow courses (surveyed
since 1975), and the Herschel Island ECCC climate station (69.57°N).
"""

import re

import pytest

from easysnowdata.stations.clients.yukon import YukonClient, YukonError
from easysnowdata.stations.clients.yukon.yukon_client import (
    _TYPE_TO_YUKON_VARS,
    APPROVAL_FLAGS,
    DATA_FLAGS,
    ECCC_NETWORK,
    GRADE_FLAGS,
    NETWORK_CODES,
    QUALIFIER_FLAGS,
    SNOW_SURVEY_FLAGS,
    SNOW_SURVEY_NETWORK,
    SNOW_VARIABLES,
    VARIABLES,
    _parse_csv,
    _parse_pg_array,
    _resolve_variables,
    _series_interval,
    _series_variable_key,
    _station_state,
    _strip_api_comments,
    _survey_period,
)
from tests.stations.conftest import (
    BBOX_YUKON,
    RECORD_KEYS,
    TEST_BEGIN,
    TEST_END,
    YUKON_AWS,
    YUKON_COURSES,
    YUKON_ECCC,
    YUKON_NON_YT_COURSE,
)

pytestmark = pytest.mark.live  # hits real network endpoints


@pytest.fixture(scope="module")
def client():
    return YukonClient()


# ── VARIABLES dict compliance ─────────────────────────────────────────────────


def test_variables_required_fields():
    """Every entry in VARIABLES has all required keys."""
    required = {"name", "type", "units", "description", "notes", "source"}
    for key, info in VARIABLES.items():
        missing = required - set(info.keys())
        assert not missing, f"VARIABLES[{key!r}] missing keys: {missing}"


def test_variables_have_output_units():
    """This client documents both native and emitted units."""
    for key, info in VARIABLES.items():
        assert info.get("output_units"), f"VARIABLES[{key!r}] missing output_units"


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


def test_type_to_yukon_vars_has_swe_snwd():
    assert "swe_mm" in _TYPE_TO_YUKON_VARS["swe"]
    assert "snwd_cm" in _TYPE_TO_YUKON_VARS["snwd"]


def test_type_to_yukon_vars_keys_exist():
    """Every variable key referenced by a type must exist in VARIABLES."""
    for std_type, keys in _TYPE_TO_YUKON_VARS.items():
        for key in keys:
            assert key in VARIABLES, (
                f"_TYPE_TO_YUKON_VARS[{std_type!r}] references unknown variable {key!r}"
            )


def test_swe_output_units_is_cm():
    assert VARIABLES["swe_mm"]["units"] == "mm", "native unit is mm"
    assert VARIABLES["swe_mm"]["output_units"] == "cm", "emitted in cm"


def test_snwd_units_is_cm():
    assert VARIABLES["snwd_cm"]["units"] == "cm"
    assert VARIABLES["snwd_cm"]["output_units"] == "cm"


def test_snow_variables_are_swe_and_snwd():
    types = {VARIABLES[k]["type"] for k in SNOW_VARIABLES}
    assert types == {"swe", "snwd"}


def test_data_flags_union_covers_vocabularies():
    """DATA_FLAGS must contain every source vocabulary it claims to."""
    for code in SNOW_SURVEY_FLAGS:
        assert code in DATA_FLAGS
    for code in GRADE_FLAGS:
        assert f"grade:{code}" in DATA_FLAGS
    for code in APPROVAL_FLAGS:
        assert f"approval:{code}" in DATA_FLAGS
    for code in QUALIFIER_FLAGS:
        assert f"qualifier:{code}" in DATA_FLAGS


def test_network_codes_cover_station_types():
    assert set(NETWORK_CODES) == {"SC", "AWS", "ECCC"}
    assert NETWORK_CODES["SC"] == NETWORK_CODES["AWS"] == "YSS"
    assert NETWORK_CODES["ECCC"] == "YKEC"


# ── Pure helpers (offline) ────────────────────────────────────────────────────


def test_parse_pg_array_multiple_elements():
    assert _parse_pg_array(
        '{"Yukon Snow Survey Network","Yukon Small Stream Network"}'
    ) == ["Yukon Snow Survey Network", "Yukon Small Stream Network"]


def test_parse_pg_array_null_and_empty_forms():
    for raw in ("{NULL}", "[]", "{}", "", None, "NULL"):
        assert _parse_pg_array(raw) == [], f"{raw!r} should parse to []"


def test_parse_pg_array_drops_null_elements():
    assert _parse_pg_array('{"Avalanche Canada",NULL}') == ["Avalanche Canada"]


def test_strip_api_comments_handles_quoted_header_terminator():
    """The comment block ends with a line containing exactly ``""``.

    Filtering on ``line.strip()`` alone treats that line as content and
    makes it the CSV header — the regression this guards against.
    """
    raw = (
        '"# Description: Snow survey measurements."\n'
        '"# Generated at : 2026-07-27 15:14 MST"\n'
        '"# "\n'
        '""\n'
        "location_code,parameter,result\n"
        "08AA-SC01,snow water equivalent,97\n"
    )
    assert _strip_api_comments(raw).splitlines()[0] == (
        "location_code,parameter,result"
    )
    rows = _parse_csv(raw)
    assert rows == [
        {
            "location_code": "08AA-SC01",
            "parameter": "snow water equivalent",
            "result": "97",
        }
    ]


def test_parse_csv_suppresses_status_envelope():
    """A no-match query returns ``status,message``, not an empty CSV."""
    raw = (
        "status,message\n"
        'info,"No daily measurements found for the specified timeseries '
        'and date range."\n'
    )
    assert _parse_csv(raw) == []


def test_parse_csv_empty_body():
    assert _parse_csv("") == []


def test_series_variable_key_maps_snow_parameters():
    assert _series_variable_key("snow water equivalent", "instantaneous") == "swe_mm"
    assert _series_variable_key("snow depth", "instantaneous") == "snwd_cm"


def test_series_variable_key_disambiguates_air_temp_by_aggregation():
    """ECCC stores daily air temperature as three separate series."""
    assert _series_variable_key("temperature, air", "maximum") == "air_temp_max_degc"
    assert _series_variable_key("temperature, air", "minimum") == "air_temp_min_degc"
    assert _series_variable_key("temperature, air", "(min+max)/2") == "air_temp_degc"
    assert _series_variable_key("temperature, air", "instantaneous") == "air_temp_degc"


def test_series_variable_key_ignores_hydrometric_parameters():
    """Water flow, water level and water quality are out of scope."""
    for param in (
        "water flow",
        "water level",
        "pH",
        "turbidity",
        "water level below ground surface",
    ):
        assert _series_variable_key(param, "instantaneous") is None, param


def test_series_interval_maps_recording_rates():
    assert _series_interval("1 day") == "daily"
    assert _series_interval("01:00:00") == "hourly"
    assert _series_interval("03:00:00") == "sub_daily"
    assert _series_interval("00:05:00") == "sub_daily"


def test_survey_period_handles_non_integer_month():
    """``month`` is 5.5 for the May 15 survey, so it must parse as a float."""
    assert _survey_period(2) == "01-Feb"
    assert _survey_period(4) == "01-Apr"
    assert _survey_period("5.5") == "15-May"
    assert _survey_period("") == ""


def test_station_state_prefers_name_declared_jurisdiction():
    assert (
        _station_state("09AA-SC04", "Atlin (B.C.) Snow Course", 59.59, -133.71) == "BC"
    )
    assert (
        _station_state("09EC-SC02", "Boundary (Alaska) Snow Course", 64.08, -141.45)
        == "AK"
    )


def test_station_state_defaults_to_yt_inside_yukon():
    assert (
        _station_state("08AA-SC01", "Canyon Lake Snow Course", 61.12, -136.99) == "YT"
    )


def test_station_state_uses_override_outside_yukon():
    assert _station_state("08AK-SC01", "Eaglecrest Snow Course", 58.28, -134.53) == "AK"


def test_resolve_variables_accepts_types_and_native_keys():
    assert _resolve_variables("swe") == ["swe_mm"]
    assert _resolve_variables("swe_mm") == ["swe_mm"]
    assert _resolve_variables(None) == list(SNOW_VARIABLES)


def test_resolve_variables_skips_unknown_and_falls_back():
    assert _resolve_variables(["not_a_variable"]) == list(SNOW_VARIABLES)


# ── get_locations / get_timeseries ────────────────────────────────────────────


def test_get_locations_returns_many(client):
    locations = client.get_locations()
    assert len(locations) > 300, f"Expected 300+ locations, got {len(locations)}"


def test_get_locations_required_fields(client):
    required = {
        "location_id",
        "location_code",
        "name",
        "location_type",
        "latitude",
        "longitude",
        "elevation_m",
        "networks",
    }
    for loc in client.get_locations()[:20]:
        missing = required - set(loc.keys())
        assert not missing, f"Location missing fields: {missing}"


def test_get_locations_filter_by_type(client):
    snowpack = client.get_locations(location_types="snowpack")
    assert len(snowpack) > 80, "Expected 80+ snow courses"
    for loc in snowpack:
        assert loc["location_type"] == "snowpack"


def test_get_locations_filter_by_network(client):
    stations = client.get_locations(networks=SNOW_SURVEY_NETWORK)
    assert len(stations) > 80
    for loc in stations:
        assert SNOW_SURVEY_NETWORK in loc["networks"]


def test_get_locations_networks_are_parsed_lists(client):
    """``networks`` is a Postgres array literal in the raw CSV."""
    with_nets = [loc for loc in client.get_locations() if loc["networks"]]
    assert with_nets, "Expected some locations to declare a network"
    for loc in with_nets[:20]:
        assert isinstance(loc["networks"], list)
        for net in loc["networks"]:
            assert not net.startswith("{"), f"Unparsed array literal: {net!r}"


def test_get_timeseries_only_exposes_known_variables(client):
    series = client.get_timeseries()
    assert len(series) > 100
    for ser in series:
        assert ser["variable"] in VARIABLES


def test_get_timeseries_filter_by_variable(client):
    swe_series = client.get_timeseries(variables=["swe"])
    assert len(swe_series) >= 8, "Expected 8+ snow-pillow SWE series"
    for ser in swe_series:
        assert ser["variable"] == "swe_mm"


def test_get_timeseries_publicly_visible_by_default(client):
    for ser in client.get_timeseries():
        assert ser["publicly_visible"] is True


# ── Station endpoints ─────────────────────────────────────────────────────────


def test_get_snow_course_stations_covers_all_snowpack_locations(client):
    """
    Courses are built from ``/locations``, not ``/snow-survey/metadata``.

    Composite records such as 09DC-SC01 (Mayo Airport) appear only in
    ``/locations``, so building from the metadata endpoint alone would
    silently drop them.
    """
    snowpack_codes = {
        loc["location_code"] for loc in client.get_locations(location_types="snowpack")
    }
    course_codes = {s["station_id"] for s in client.get_snow_course_stations()}
    assert course_codes == snowpack_codes


def test_get_snow_course_stations_fields(client):
    courses = client.get_snow_course_stations()
    assert len(courses) > 80
    required = {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "elevation_m",
        "state",
        "station_type",
        "status",
        "station_url",
        "variables",
        "first_survey",
        "last_survey",
        "survey_counts",
    }
    for sta in courses:
        missing = required - set(sta.keys())
        assert not missing, f"Course missing fields: {missing}"
        assert sta["station_type"] == "SC"
        assert sta["network_code"] == "YSS"


def test_get_snow_course_stations_known_course(client):
    courses = {s["station_id"]: s for s in client.get_snow_course_stations()}
    canyon = courses[YUKON_COURSES[0]]
    assert "Canyon Lake" in canyon["name"]
    assert canyon["first_survey"].startswith("197")
    assert canyon["elevation_m"] > 1000


def test_get_automated_stations_split_by_network(client):
    stations = client.get_automated_stations()
    assert len(stations) >= 15, f"Expected 15+, got {len(stations)}"
    by_type = {}
    for sta in stations:
        by_type.setdefault(sta["station_type"], []).append(sta)
    assert len(by_type.get("AWS", [])) >= 8, "Expected 8+ Yukon AWS stations"
    assert len(by_type.get("ECCC", [])) >= 6, "Expected 6+ ECCC stations"
    for sta in by_type.get("ECCC", []):
        assert sta["network"] == ECCC_NETWORK
        assert sta["network_code"] == "YKEC"
        assert "Environment and Climate Change Canada" in sta["operator"]


def test_get_automated_stations_have_snow_series(client):
    for sta in client.get_automated_stations():
        snow = [s for s in sta["series"] if s["variable"] in SNOW_VARIABLES]
        assert snow, f"{sta['station_id']} has no snow series"


def test_automated_stations_expose_met_variables(client):
    """The full met suite is available, not just snow."""
    stations = {s["station_id"]: s for s in client.get_automated_stations()}
    non_snow = {
        s["variable"]
        for sta in stations.values()
        for s in sta["series"]
        if s["variable"] not in SNOW_VARIABLES
    }
    assert "air_temp_degc" in non_snow
    assert "precip_total_mm" in non_snow


def test_get_all_stations_combines_and_sorts(client):
    stations = client.get_all_stations()
    courses = client.get_snow_course_stations()
    automated = client.get_automated_stations()
    assert len(stations) == len(courses) + len(automated)
    codes = [s["station_id"] for s in stations]
    assert codes == sorted(codes)


def test_get_all_stations_state_is_not_blanket_yt(client):
    """The Yukon Snow Survey also runs courses in BC and Alaska."""
    by_code = {s["station_id"]: s for s in client.get_all_stations()}
    assert by_code[YUKON_NON_YT_COURSE]["state"] == "BC"
    states = {s["state"] for s in by_code.values()}
    assert "YT" in states
    assert states & {"BC", "AK"}, "Expected some non-Yukon stations"


def test_get_all_stations_bbox(client):
    inside = client.get_all_stations(bbox=BBOX_YUKON)
    assert len(inside) > 80
    min_lon, min_lat, max_lon, max_lat = BBOX_YUKON
    for sta in inside:
        assert min_lat <= sta["latitude"] <= max_lat
        assert min_lon <= sta["longitude"] <= max_lon


def test_get_all_stations_active_only_is_subset(client):
    assert len(client.get_all_stations(active_only=True)) < len(
        client.get_all_stations()
    )


def test_get_metadata_returns_station(client):
    meta = client.get_metadata(YUKON_AWS[0])
    assert meta["station_id"] == YUKON_AWS[0]
    assert meta["series"], "Automated station should list its series"


def test_get_metadata_unknown_station_returns_empty(client):
    assert client.get_metadata("XXXXX_INVALID_STATION_99999") == {}


def test_get_station_image_url_is_none(client):
    """This source publishes no station imagery."""
    assert client.get_station_image_url(YUKON_AWS[0]) is None


# ── get_data — continuous (daily) ─────────────────────────────────────────────


def test_get_data_flat_schema(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
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
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    swe = [r for r in records if r["type"] == "swe"]
    assert len(swe) > 0, "Expected at least one SWE record"
    for r in swe:
        assert r["units"] == "cm", f"SWE units should be cm, got {r['units']!r}"
        assert r["variable"] == "swe_mm"
        assert r["interval"] == "daily"
        if r["value"] is not None:
            assert 0 <= r["value"] <= 500, f"Implausible SWE value: {r['value']} cm"


def test_get_data_swe_is_converted_from_mm(client):
    """A mid-winter Yukon snowpack is tens of cm, not hundreds (i.e. not mm)."""
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date="2024-03-01",
        end_date="2024-03-31",
    )
    values = [r["value"] for r in records if r["value"] is not None]
    assert values, "Expected March SWE values"
    assert max(values) < 200, (
        f"Peak SWE {max(values)} looks like mm, not cm — check the conversion"
    )


def test_get_data_snwd_type(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
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
    """Passing 'swe_mm' (native key) works the same as type 'swe'."""
    by_type = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    by_key = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe_mm"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert {r["date"] for r in by_type} == {r["date"] for r in by_key}


def test_get_data_multiple_stations(client):
    records = client.get_data(
        station_ids=YUKON_AWS,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    found = {r["station_id"] for r in records}
    for sid in YUKON_AWS:
        assert sid in found, f"{sid} has no records"


def test_get_data_station_id_field(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["station_id"] == YUKON_AWS[0]


def test_get_data_carries_series_provenance(client):
    """``aggregation`` and ``timeseries_id`` disambiguate same-parameter series."""
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert records
    for r in records:
        assert r["aggregation"]
        assert r["timeseries_id"].isdigit()


def test_get_data_include_flags(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
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
        bbox=BBOX_YUKON,
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0, "Expected records for the Yukon bbox"
    assert len({r["station_id"] for r in records}) > 3


def test_get_data_no_ids_no_bbox_raises(client):
    with pytest.raises(ValueError, match="station_ids or bbox"):
        client.get_data(variables=["swe"])


def test_get_data_interval_field_daily(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        interval="daily",
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    for r in records:
        assert r["interval"] == "daily"


def test_get_data_date_format(client):
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert len(records) > 0
    for r in records:
        assert date_re.match(r["date"]), f"Unexpected date format: {r['date']!r}"


def test_get_data_hourly_adds_datetime(client):
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        interval="hourly",
        begin_date="2024-03-01 00:00",
        end_date="2024-03-02 00:00",
    )
    assert len(records) > 0, "Expected sub-daily records"
    for r in records:
        assert "datetime" in r, "Sub-daily records should carry 'datetime'"
        assert r["interval"] != "daily"


def test_get_data_empty_range_returns_empty_list(client):
    """A no-match window must not leak the API's status envelope as data."""
    records = client.get_data(
        station_ids=YUKON_AWS[:1],
        variables=["swe"],
        begin_date="1900-01-01",
        end_date="1900-01-31",
    )
    assert records == []


def test_get_data_eccc_snow_depth(client):
    """ECCC snow depth is sparse, so query a whole year."""
    records = client.get_data(
        station_ids=YUKON_ECCC,
        variables=["snwd"],
        begin_date="2010-01-01",
        end_date="2010-12-31",
    )
    assert len(records) > 0, "Expected Herschel Island snow depth for 2010"
    for r in records:
        assert r["type"] == "snwd"
        assert r["units"] == "cm"


def test_get_data_courses_have_no_continuous_series(client):
    """Snow courses are periodic — a daily request yields nothing."""
    assert (
        client.get_data(
            station_ids=YUKON_COURSES,
            variables=["swe"],
            interval="daily",
            begin_date=TEST_BEGIN,
            end_date=TEST_END,
        )
        == []
    )


# ── get_data — periodic (snow courses) ────────────────────────────────────────


def test_get_data_periodic_flat_schema(client):
    records = client.get_data(station_ids=YUKON_COURSES[:1], interval="periodic")
    assert len(records) > 0, "Expected snow course surveys"
    for r in records:
        missing = RECORD_KEYS - set(r.keys())
        assert not missing, f"Record missing keys: {missing}\n  record={r}"
        assert r["interval"] == "periodic"


def test_get_snow_survey_data_units_and_conversion(client):
    records = client.get_snow_survey_data(station_ids=YUKON_COURSES[:1])
    swe = [r for r in records if r["type"] == "swe" and r["value"] is not None]
    snwd = [r for r in records if r["type"] == "snwd" and r["value"] is not None]
    assert swe and snwd
    for r in swe:
        assert r["units"] == "cm"
        assert 0 <= r["value"] <= 200, f"Implausible course SWE: {r['value']} cm"
    for r in snwd:
        # /snow-survey/data leaves the units field empty for snow depth;
        # the client must supply cm rather than trusting the response.
        assert r["units"] == "cm"
        assert 0 <= r["value"] <= 1000


def test_get_snow_survey_data_survey_periods(client):
    records = client.get_snow_survey_data(station_ids=YUKON_COURSES[:1])
    periods = {r["survey_period"] for r in records}
    assert "01-Apr" in periods, "Expected April 1 surveys"
    assert periods <= {"01-Feb", "01-Mar", "01-Apr", "01-May", "15-May", ""}


def test_get_snow_survey_data_target_and_sample_dates(client):
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for r in client.get_snow_survey_data(station_ids=YUKON_COURSES[:1])[:50]:
        assert date_re.match(r["date"]), r["date"]
        assert date_re.match(r["target_date"]), r["target_date"]


def test_get_snow_survey_data_include_flags(client):
    records = client.get_snow_survey_data(
        station_ids=YUKON_COURSES[:1], include_flags=True
    )
    assert records
    flags = {r["flag"] for r in records}
    assert flags <= set(SNOW_SURVEY_FLAGS) | {""}, f"Unexpected flags: {flags}"


def test_get_snow_survey_data_date_filter(client):
    records = client.get_snow_survey_data(
        station_ids=YUKON_COURSES[:1],
        begin_date="2000-01-01",
        end_date="2009-12-31",
    )
    assert records
    for r in records:
        assert "2000-01-01" <= r["date"] <= "2009-12-31"


def test_get_snow_survey_data_long_record(client):
    """The archive reaches back to the 1960s-70s."""
    records = client.get_snow_survey_data()
    assert len(records) > 15_000, f"Expected a large archive, got {len(records)}"
    assert min(r["date"] for r in records) < "1970-01-01"


def test_get_snow_survey_stats(client):
    stats = client.get_snow_survey_stats()
    assert len(stats) > 80
    for row in stats[:5]:
        assert row["location_code"]
        assert row["max_SWE_mm"]


def test_get_snow_survey_trends(client):
    trends = client.get_snow_survey_trends()
    assert len(trends) > 80
    for row in trends[:5]:
        assert row["location_code"]
        assert "sens.slope_SWE_max" in row


# ── Error handling ────────────────────────────────────────────────────────────


def test_get_data_unknown_station_returns_empty(client):
    records = client.get_data(
        station_ids=["XXXXX_INVALID_99999"],
        variables=["swe"],
        begin_date=TEST_BEGIN,
        end_date=TEST_END,
    )
    assert records == []


def test_yukon_error_on_bad_endpoint(client):
    with pytest.raises(YukonError):
        client._get_csv("no-such-endpoint", {})


def test_yukon_error_is_exception():
    assert issubclass(YukonError, Exception)
