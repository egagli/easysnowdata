"""
Tests for FMIClient.

Offline tests feed trimmed copies of real FMI WFS responses (station list,
daily and instantaneous time-value pairs, the HTTP 400 exception reports)
and of the public station table's Nuxt payload through a stub session.
``live`` tests hit opendata.fmi.fi (no credentials).
"""

import json
from datetime import date

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.fmi import FMIClient, FMIError
from easysnowdata.stations.clients.fmi.fmi_client import (
    _MAX_IDS,
    DAILY_QUERY,
    DATA_FLAGS,
    INSTANT_QUERY,
    SNOW_NETWORKS,
    STATIONS_PAGE_URL,
    STATIONS_QUERY,
    VARIABLES,
    _convert,
    _parse_series,
    _parse_station_table,
    _parse_stations,
    _resolve_variables,
    _windows,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

BASE = "https://example.test/wfs"

_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" '
    'xmlns:gml="http://www.opengis.net/gml/3.2" '
    'xmlns:ef="http://inspire.ec.europa.eu/schemas/ef/4.0" '
    'xmlns:om="http://www.opengis.net/om/2.0" '
    'xmlns:wml2="http://www.opengis.net/waterml/2.0" '
    'xmlns:target="http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1" '
    'xmlns:xlink="http://www.w3.org/1999/xlink">'
)


def _facility(fmisid, name, pos, periods, networks, wmo=None):
    per = "".join(
        "<ef:operationalActivityPeriod><ef:OperationalActivityPeriod>"
        "<ef:activityTime><gml:TimePeriod>"
        f"<gml:beginPosition>{b}T00:00:00Z</gml:beginPosition>"
        + (
            '<gml:endPosition indeterminatePosition="now"/>'
            if e == "now"
            else f"<gml:endPosition>{e}T00:00:00Z</gml:endPosition>"
        )
        + "</gml:TimePeriod></ef:activityTime>"
        "</ef:OperationalActivityPeriod></ef:operationalActivityPeriod>"
        for b, e in periods
    )
    nets = "".join(
        f'<ef:belongsTo xlink:title="{t}" xlink:href="https://opendata.fmi.fi/wfs/eng'
        f'?request=getFeature&amp;storedquery_id=fmi::ef::networks&amp;networkid={n}&amp;"/>'
        for n, t in networks
    )
    wmo_el = (
        f'<gml:name codeSpace="http://xml.fmi.fi/namespace/locationcode/wmo">{wmo}</gml:name>'
        if wmo
        else ""
    )
    return (
        "<wfs:member><ef:EnvironmentalMonitoringFacility>"
        f'<gml:identifier codeSpace="http://xml.fmi.fi/namespace/stationcode/fmisid">{fmisid}</gml:identifier>'
        f'<gml:name codeSpace="http://xml.fmi.fi/namespace/locationcode/name">{name}</gml:name>'
        f"{wmo_el}"
        '<gml:name codeSpace="http://xml.fmi.fi/namespace/location/region">Sodankylä</gml:name>'
        f"<ef:name>{name}</ef:name>"
        f"<ef:representativePoint><gml:Point><gml:pos>{pos}</gml:pos></gml:Point></ef:representativePoint>"
        f"{per}{nets}"
        "</ef:EnvironmentalMonitoringFacility></wfs:member>"
    )


STATIONS_XML = (
    _HEAD
    + _facility(
        "101932",
        "Sodankylä Tähtelä",
        "67.36663 26.62901",
        [("1891-01-01", "now"), ("1908-01-01", "now")],
        [(121, "Automaattinen sääasema"), (124, "Sadeasema")],
        wmo="02836",
    )
    + _facility(
        "100694",
        "Kitee Närsäkkälä",
        "61.915285 30.141083",
        [("1925-05-01", "1958-12-31")],
        [(124, "Sadeasema")],
    )
    + _facility(
        "100662",
        "Helsinki Kallio 2",
        "60.187390 24.950600",
        [("1999-01-01", "now")],
        [(151, "Kolmannen osapuolen ilmanlaadun havaintoasema")],
    )
    + "</wfs:FeatureCollection>"
).encode()


def _series(fmisid, param, pairs):
    tvps = "".join(
        f"<wml2:point><wml2:MeasurementTVP><wml2:time>{t}</wml2:time>"
        f"<wml2:value>{v}</wml2:value></wml2:MeasurementTVP></wml2:point>"
        for t, v in pairs
    )
    return (
        "<wfs:member><omso:PointTimeSeriesObservation "
        'xmlns:omso="http://inspire.ec.europa.eu/schemas/omso/3.0">'
        '<om:observedProperty xlink:href="https://opendata.fmi.fi/meta?'
        f'observableProperty=observation&amp;param={param}&amp;language=eng"/>'
        "<om:featureOfInterest><target:Location>"
        f'<gml:identifier codeSpace="http://xml.fmi.fi/namespace/stationcode/fmisid">{fmisid}</gml:identifier>'
        "</target:Location></om:featureOfInterest>"
        f'<om:result><wml2:MeasurementTimeseries gml:id="obs-{fmisid}-{param}">{tvps}'
        "</wml2:MeasurementTimeseries></om:result>"
        "</omso:PointTimeSeriesObservation></wfs:member>"
    )


def _collection(*members):
    return (_HEAD + "".join(members) + "</wfs:FeatureCollection>").encode()


def _exception(text):
    # Real layout: the ExceptionText starts ~360 characters in.
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ExceptionReport xmlns="http://www.opengis.net/ows/1.1"\n'
        '\t\t xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '  xsi:schemaLocation="http://www.opengis.net/ows/1.1 '
        'http://schemas.opengis.net/ows/1.1.0/owsExceptionReport.xsd"\n'
        '  version="2.0.0" xml:lang="eng">\n\n\n'
        '  <Exception exceptionCode="OperationParsingFailed">\n'
        f"    <ExceptionText>{text}</ExceptionText>\n"
        "  </Exception>\n</ExceptionReport>\n"
    )


#: A Nuxt devalue payload with the shape of FMI's station table.
TABLE_PAYLOAD = [
    {"data": 1},
    [2, 6],
    {"fmisid": 3, "z": 4, "name": 5, "groups": 9},
    101932,
    179,
    "Sodankylä Tähtelä",
    {"fmisid": 7, "name": 8, "groups": 10, "ended": -1},
    100694,
    "Kitee Närsäkkälä",
    "weather, sounding, air quality (FMI)",
    "precipitation",
    {
        "fmisid": 12,
        "name": 13,
        "y": 14,
        "x": 15,
        "z": 16,
        "started": 17,
        "ended": 18,
        "groups": 19,
    },
    101839,
    "Ylitornio Portimojärvi",
    66.4,
    23.9,
    79,
    1935,
    2002,
    "weather",
]
TABLE_HTML = (
    '<html><body><script type="application/json" data-nuxt-data="nuxt-app" '
    f'id="__NUXT_DATA__">{json.dumps(TABLE_PAYLOAD)}</script></body></html>'
)

UNKNOWN_IDS = {"100694"}


class _Resp:
    def __init__(self, status, content):
        self.status_code = status
        self.ok = status == 200
        self.content = content if isinstance(content, bytes) else content.encode()
        self.text = self.content.decode()
        self.headers = {}


class _StubSession:
    """Routes by stored query; fails a request naming an unknown fmisid."""

    def __init__(self, table=TABLE_HTML):
        self.headers = {}
        self.calls = []
        self.table = table

    def request(self, method, url, params=None, timeout=None):
        if url == STATIONS_PAGE_URL:
            if self.table is None:
                return _Resp(503, "unavailable")
            return _Resp(200, self.table)
        p = list(params or [])
        self.calls.append(p)
        d = dict(p)
        ids = [v for k, v in p if k == "fmisid"]
        query = d["storedquery_id"]
        if query == STATIONS_QUERY:
            return _Resp(200, STATIONS_XML)
        if "bbox" in d:
            return _Resp(
                200,
                _collection(
                    _series("101932", "snow", [(d["starttime"], "57.0")]),
                    _series("101933", "snow", [(d["starttime"], "NaN")]),
                ),
            )
        if UNKNOWN_IDS & set(ids):
            return _Resp(400, _exception("Unknown &apos;fmisid&apos; value!"))
        start = d["starttime"]
        members = []
        if query == DAILY_QUERY and "2024-01-01" <= start[:10] <= "2024-12-31":
            for i in ids:
                members.append(
                    _series(
                        i,
                        "snow",
                        [
                            ("2024-01-01T00:00:00Z", "57.0"),
                            ("2024-01-02T00:00:00Z", "NaN"),
                            ("2024-07-01T00:00:00Z", "-1.0"),
                        ],
                    )
                )
        if query == INSTANT_QUERY:
            for i in ids:
                members.append(
                    _series(
                        i,
                        "ws_10min",
                        [
                            ("2024-03-25T06:00:00Z", "10.0"),
                            ("2024-03-25T07:00:00Z", "NaN"),
                        ],
                    )
                )
        return _Resp(200, _collection(*members))


@pytest.fixture
def client():
    return FMIClient(
        base_url=BASE, session=_StubSession(), max_retries=1, min_interval=0
    )


# ── registries ───────────────────────────────────────────────────────────────


def test_registries_use_shared_vocab():
    for code, info in VARIABLES.items():
        assert info["type"] in TYPES, code
        assert set(info["interval"]) <= INTERVALS, code
        for key in ("units", "output_units", "description", "notes", "source"):
            assert key in info, (code, key)
    assert set(DATA_FLAGS) == {"snow_free", "no_precipitation"}


def test_output_units_are_canonical():
    for code, info in VARIABLES.items():
        assert info["output_units"] == canonical_units(info["type"]), code


def test_snow_networks_cover_the_reporting_networks():
    assert set(SNOW_NETWORKS) == {121, 122, 123, 124, 224}


def test_resolve_variables():
    assert _resolve_variables(None, "daily") == ["snow"]
    assert _resolve_variables(None, "hourly") == ["snow_aws"]
    assert _resolve_variables(["snwd", "temp"], "daily") == ["snow", "tday"]
    assert _resolve_variables(["snwd", "temp"], "hourly") == ["snow_aws", "t2m"]
    assert _resolve_variables("SNOW", "daily") == ["snow"]
    with pytest.raises(FMIError, match="Unknown FMI variable"):
        _resolve_variables("swe", "daily")
    # snow_aws is rejected by the daily query with HTTP 400 upstream.
    with pytest.raises(FMIError, match="not 'daily'"):
        _resolve_variables("snow_aws", "daily")
    with pytest.raises(FMIError, match="no 'precip' variable at interval 'sub_hourly'"):
        _resolve_variables("precip", "sub_hourly")


def test_convert_codes_units_and_clamp():
    assert _convert("snow", "57.0") == (57.0, None)
    assert _convert("snow", "-1.0") == (0.0, "snow_free")
    assert _convert("snow", "0.0") == (0.0, None)
    assert _convert("snow", "NaN") == (None, None)
    assert _convert("snow", "1600") == (None, None)  # 16 m
    assert _convert("rrday", "-1.0") == (0.0, "no_precipitation")
    assert _convert("tday", "-1.0") == (-1.0, None)  # a real temperature
    assert _convert("ws_10min", "10.0") == (36.0, None)  # m/s → km/h


def test_windows_respect_the_service_limits():
    daily = _windows(date(2020, 1, 1), date(2023, 6, 30), "daily")
    assert daily[0] == ("2020-01-01T00:00:00Z", "2020-12-31T00:00:00Z")
    assert daily[1][0] == "2021-01-01T00:00:00Z"
    assert daily[-1][1] == "2023-06-30T00:00:00Z"
    hourly = _windows(date(2024, 1, 1), date(2024, 1, 10), "hourly")
    assert hourly[0] == ("2024-01-01T00:00:00Z", "2024-01-08T00:00:00Z")  # 168 h
    assert hourly[1][0] == "2024-01-08T01:00:00Z"
    assert hourly[-1][1] == "2024-01-10T23:59:00Z"


# ── parsers ──────────────────────────────────────────────────────────────────


def test_parse_stations():
    raw = _parse_stations(STATIONS_XML)
    assert [r["fmisid"] for r in raw] == ["101932", "100694", "100662"]
    sod = raw[0]
    assert sod["latitude"] == pytest.approx(67.36663)
    assert sod["wmo"] == "02836"
    assert sod["periods"] == [("1891-01-01", "now"), ("1908-01-01", "now")]
    assert [n for n, _ in sod["networks"]] == [121, 124]
    assert raw[1]["periods"] == [("1925-05-01", "1958-12-31")]


def test_parse_station_table_reads_devalue_indices():
    table = _parse_station_table(TABLE_HTML)
    assert table["101932"]["elevation_m"] == 179.0
    assert table["101932"]["groups"] == "weather, sounding, air quality (FMI)"
    assert table["101839"]["latitude"] == 66.4
    assert table["101839"]["end_year"] == 2002
    assert table["100694"]["elevation_m"] is None
    with pytest.raises(FMIError, match="__NUXT_DATA__"):
        _parse_station_table("<html></html>")


def test_parse_series():
    content = _collection(_series("101932", "snow", [("2024-01-01T00:00:00Z", "57.0")]))
    assert _parse_series(content) == [
        ("101932", "snow", [("2024-01-01T00:00:00Z", "57.0")])
    ]


# ── stations ─────────────────────────────────────────────────────────────────


def test_get_all_stations(client):
    stas = client.get_all_stations()
    # The air-quality station (network 151) is not a snow-depth station;
    # 101839 is only in the station table.
    assert [s["station_id"] for s in stas] == ["101932", "100694", "101839"]
    sod, kitee, ylitornio = stas
    assert ylitornio["station_source"] == "station_table"
    assert ylitornio["coordinate_precision_deg"] == 0.1
    assert ylitornio["network_code"] == "WS" and ylitornio["status"] == "Inactive"
    assert (ylitornio["begin_year"], ylitornio["end_year"]) == (1935, 2002)
    assert client._record_span(["101839"]) == (date(1935, 1, 1), date(2002, 12, 31))
    assert sod["station_source"] == "wfs"
    assert sod["status"] == "Active" and sod["end_date"] is None
    assert sod["begin_date"] == "1891-01-01"
    assert sod["elevation_m"] == 179.0
    assert sod["network_code"] == "AWS"
    assert kitee["status"] == "Inactive" and kitee["end_date"] == "1958-12-31"
    assert kitee["network_code"] == "PRCP"
    assert kitee["elevation_m"] is None
    for s in stas:
        assert {
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "status",
        } <= set(s)
    assert [s["station_id"] for s in client.get_all_stations(active_only=True)] == [
        "101932"
    ]
    assert client.get_all_stations(bbox=(29, 61, 31, 62))[0]["station_id"] == "100694"
    assert len(client.get_all_stations(active_only=True)) == 1


def test_station_table_failure_raises_unless_partial():
    c = FMIClient(
        base_url=BASE, session=_StubSession(table=None), max_retries=1, min_interval=0
    )
    with pytest.raises(FMIError, match="station table"):
        c.get_all_stations()
    c = FMIClient(
        base_url=BASE,
        session=_StubSession(table=None),
        max_retries=1,
        min_interval=0,
        allow_partial=True,
    )
    assert c.get_all_stations()[0]["elevation_m"] is None


def test_get_metadata(client):
    md = client.get_metadata("101932")
    names = {(v["name"], v["interval"]) for v in md["data_variables"]}
    assert ("snow", "daily") in names and ("snow_aws", "hourly") in names
    # A manual precipitation station gets daily variables only.
    md = client.get_metadata("100694")
    assert {v["interval"] for v in md["data_variables"]} == {"daily"}
    with pytest.raises(FMIError, match="not in FMI's snow-depth networks"):
        client.get_metadata("100662")


# ── data ─────────────────────────────────────────────────────────────────────


def test_get_data_schema_flags_and_unknown_station(client, caplog):
    recs = client.get_data(
        ["101932", "100694"],
        begin_date="2024-01-01",
        end_date="2024-12-31",
        include_flags=True,
    )
    # 100694 is unknown to the observation service: skipped, not fatal.
    assert {r["station_id"] for r in recs} == {"101932"}
    assert "unknown to the observation service" in caplog.text
    assert [(r["date"], r["value"], r["flag"]) for r in recs] == [
        ("2024-01-01", 57.0, None),
        ("2024-07-01", 0.0, "snow_free"),
    ]
    for r in recs:
        assert set(r) == RECORD_KEYS | {"flag"}
        assert r["units"] == "cm" and r["interval"] == "daily"


def test_get_data_batches_ids_and_windows(client):
    ids = [str(101000 + i) for i in range(_MAX_IDS + 5)]
    client.get_data(ids, begin_date="2023-01-01", end_date="2024-03-01")
    data_calls = [
        c
        for c in client._session.calls
        if dict(c)["storedquery_id"] == DAILY_QUERY
        and dict(c)["starttime"] != dict(c)["endtime"]
    ]
    # two id batches × two windows (2023-01-01..2024-01-01, 2024-01-02..03-01)
    assert len(data_calls) == 4
    assert max(sum(1 for k, _ in c if k == "fmisid") for c in data_calls) == _MAX_IDS


def test_screen_bisects_to_the_unknown_id(client):
    known, unknown = client.screen_station_ids(["101932", "100694", "101933"])
    assert known == ["101932", "101933"] and unknown == ["100694"]
    n = len(client._session.calls)
    client.screen_station_ids(["101932", "100694"])  # cached
    assert len(client._session.calls) == n


def test_get_data_subdaily_has_datetime_and_kmh(client):
    recs = client.get_data(
        "101932",
        variables="wind_spd",
        interval="hourly",
        begin_date="2024-03-25",
        end_date="2024-03-25",
    )
    assert recs == [
        {
            "station_id": "101932",
            "date": "2024-03-25",
            "datetime": "2024-03-25T06:00:00Z",
            "variable": "ws_10min",
            "type": "wind_spd",
            "value": 36.0,
            "units": "km/h",
            "interval": "hourly",
        }
    ]


def test_get_data_errors(client):
    with pytest.raises(ValueError):
        client.get_data()
    with pytest.raises(FMIError, match="Unsupported interval"):
        client.get_data("101932", interval="monthly")
    with pytest.raises(FMIError, match="integers"):
        client.get_data("Sodankylä")
    with pytest.raises(FMIError, match="Unknown FMI variable"):
        client.get_data("101932", variables="swe")


def test_get_data_bbox_resolves_stations(client):
    recs = client.get_data(
        bbox=(26, 67, 27, 68), begin_date="2024-01-01", end_date="2024-01-31"
    )
    assert {r["station_id"] for r in recs} == {"101932"}


def test_other_400_still_raises():
    class _Bad(_StubSession):
        def request(self, method, url, params=None, timeout=None):
            if dict(params or []).get("storedquery_id") == STATIONS_QUERY:
                return super().request(method, url, params, timeout)
            return _Resp(400, _exception("Too long time interval requested!"))

    c = FMIClient(base_url=BASE, session=_Bad(), max_retries=1, min_interval=0)
    with pytest.raises(FMIError, match="Too long"):
        c.get_data("101932", begin_date="2024-01-01", end_date="2024-01-02")


# ── live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_stations():
    c = FMIClient()
    stas = c.get_all_stations()
    assert len(stas) > 2000  # 2 244 in November 2026, open and closed
    active = [s for s in stas if s["status"] == "Active"]
    assert len(active) > 150
    sod = next(s for s in stas if s["station_id"] == "101932")
    assert sod["name"] == "Sodankylä Tähtelä"
    assert 67 < sod["latitude"] < 68 and 26 < sod["longitude"] < 27
    assert sod["elevation_m"] == pytest.approx(179, abs=5)


@pytest.mark.live
def test_live_daily_snow_depth():
    """Daily snow depth, Sodankylä Tähtelä, Jan–Apr 2024: one value a day."""
    c = FMIClient()
    recs = c.get_data("101932", begin_date="2024-01-01", end_date="2024-04-30")
    assert len(recs) == 121
    assert all(r["type"] == "snwd" and 0 <= r["value"] < 200 for r in recs)


@pytest.mark.live
def test_live_daily_snow_is_the_06utc_reading():
    c = FMIClient()
    daily = {
        r["date"]: r["value"]
        for r in c.get_data("101932", begin_date="2024-04-22", end_date="2024-04-25")
    }
    hourly = {
        r["datetime"]: r["value"]
        for r in c.get_data(
            "101932", interval="hourly", begin_date="2024-04-22", end_date="2024-04-25"
        )
    }
    for day, value in daily.items():
        assert hourly[f"{day}T06:00:00Z"] == value


@pytest.mark.live
def test_live_multi_year_request_is_chunked():
    c = FMIClient()
    recs = c.get_data(
        ["101932", "100971"],
        variables=["snwd", "temp"],
        begin_date="2021-10-01",
        end_date="2023-05-31",
    )
    days = {(r["station_id"], r["variable"], r["date"]) for r in recs}
    assert len(days) == len(recs) > 2 * 2 * 500


def test_default_span_follows_the_activity_period(client):
    # Kitee Närsäkkälä closed in 1958: without dates, nothing after it.
    assert client._record_span(["100694"]) == (date(1925, 5, 1), date(1958, 12, 31))
    # An active station extends the span to today.
    begin, end = client._record_span(["100694", "101932"])
    assert begin == date(1900, 1, 1) and end == date.today()  # 1891 → floor 1900
    # A station not in the list: the whole range.
    assert client._record_span(["999999"]) == (date(1900, 1, 1), date.today())


def test_reporting_station_ids_keeps_stations_with_a_value(client):
    assert client.reporting_station_ids("2024-01-15") == {"101932"}
    assert set(client.reporting_stations("2024-01-15")) == {"101932"}
    call = dict(client._session.calls[-1])
    assert call["bbox"] == "19.0,59.0,33.0,71.0"
    assert call["starttime"] == call["endtime"] == "2024-01-15T00:00:00Z"
    with pytest.raises(FMIError, match="not 'daily'"):
        client.reporting_station_ids("2024-01-15", variable="snow_aws")
