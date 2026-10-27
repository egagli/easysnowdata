"""
Tests for AlbertaClient.

The offline tests serve synthetic payloads in the real Alberta River Basins
formats (ArcGIS MapServer query JSON through ``proxy.ashx`` and the WISKI
``Download?...&json=true`` reply) through a stub session, so they touch no
socket.  The ``live`` tests hit rivers.alberta.ca (no credentials needed).
"""

import json
import urllib.parse
from datetime import UTC, datetime, timedelta

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.alberta import AlbertaClient, AlbertaError
from easysnowdata.stations.clients.alberta.alberta_client import (
    DATA_FLAGS,
    VARIABLES,
    _convert,
    _daily_means,
    _epoch_ms_to_mst_wall,
    _parse_download_json,
    _resolve_variables,
    _split_name,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS


def _wall_ms(text: str) -> int:
    """MST wall clock → the epoch-ms encoding the catalog uses."""
    d = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    return int(d.timestamp() * 1000)


_NOW_WALL = datetime.now(UTC) - timedelta(hours=7)
_RECENT = int(_NOW_WALL.replace(tzinfo=UTC).timestamp() * 1000)


def _ts_row(sid, par, ts_id, lowres, ts_name, unit, current, first, last, site="AB"):
    return {
        "OBJECTID": 1,
        "SiteNumber": site,
        "station_number": sid,
        "ParNumber": par,
        "ParName": par,
        "TsName": ts_name,
        "tsID": ts_id,
        "TsUnitSymbol": unit,
        "TsMinDateUtc": first + 6 * 3600 * 1000,
        "TsMaxDateUtc": last + 6 * 3600 * 1000,
        "TsMinDataAsCurrentTime": first,
        "TsMaxDataAsCurrentTime": last,
        "CurrentData": current,
        "data_type": "Meteorological",
        "TsIdLowres": lowres,
        "station_type": "envdata",
        "station_status": "Active" if site == "AB" else "Unknown",
    }


SNOW_ROWS = [
    _ts_row(
        "05BB803", "SW", "101", "102", "C.Corrected-Seasonal", "mm", "current",
        _wall_ms("1984-03-12 11:00"), _RECENT,
    ),
    _ts_row(
        "05BB803", "SD", "103", "104", "C.Corrected-Seasonal", "cm", "current",
        _wall_ms("2016-09-15 12:00"), _RECENT,
    ),
    _ts_row(
        "05BB803", "SW-SS", "105", "105", "P.Survey", "mm", "not_current",
        _wall_ms("1970-02-25 12:00"), _wall_ms("2026-05-28 12:00"),
    ),
    # BC mirror missing from the station layer (no coordinates).
    _ts_row(
        "4A37P", "SW", "111", "112", "C.Corrected-Seasonal", "mm", "current",
        _wall_ms("2019-10-29 00:00"), _RECENT, site="BC",
    ),
    # Old survey-only course.
    _ts_row(
        "07AA001", "SW-SS", "121", "121", "P.Survey", "mm", "not_current",
        _wall_ms("1965-03-01 12:00"), _wall_ms("1999-04-01 12:00"),
    ),
]  # fmt: skip

EXTRA_ROWS = [
    _ts_row(
        "05BB803", "TA", "131", "132", "C.Corrected-Sensor", "°C", "current",
        _wall_ms("1984-03-12 11:00"), _RECENT,
    ),
    _ts_row(
        "05BB803", "PC", "141", "142", "C.Merged", "mm", "current",
        _wall_ms("1984-03-12 11:00"), _RECENT,
    ),
]  # fmt: skip

POINT_ROWS = [
    {
        "station_number": "05BB803",
        "station_name": "Sunshine Village - EPA",
        "station_latitude": 51.077453,
        "station_longitude": -115.781592,
        "station_status": "Active",
        "data_type": "Meteorological",
    },
    {
        "station_number": "07AA001",
        "station_name": "Old Course",
        "station_latitude": 53.0,
        "station_longitude": -118.0,
        "station_status": "Discontinued",
        "data_type": "Meteorological",
    },
]


def _wiski(sid, par, ts_name, unit, data):
    return [
        {
            "station_no": sid,
            "station_name": "x",
            "stationparameter_name": par,
            "ts_name": ts_name,
            "ts_unitsymbol": unit,
            "rows": str(len(data)),
            "columns": "Timestamp,Value",
            "data": data,
        }
    ]


_TA_DAY = [[f"2026-03-01 {h:02d}:00:00", float(h)] for h in range(24)]

DOWNLOADS = {
    "101": _wiski(
        "05BB803",
        "SW",
        "C.Corrected-Seasonal",
        "mm",
        [
            ["2026-03-01 00:00:00", 565.0],
            ["2026-03-01 00:15:00", 565.0],
            ["2026-03-01 01:00:00", None],
            ["2026-03-01 02:00:00", -3.0],
            ["2026-03-02 00:00:00", 566.0],
        ],
    ),
    "102": _wiski(
        "05BB803",
        "SW",
        "A.DayMean",
        "mm",
        [
            ["2026-03-01 00:00:00", 565.0],
            ["2026-03-02 00:00:00", 564.640625],
            ["2026-03-03 00:00:00", None],
        ],
    ),
    "105": _wiski(
        "05BB803",
        "SW-SS",
        "P.Survey",
        "mm",
        [["2026-03-04 11:00:00", 553.0], ["2026-03-27 14:00:00", 732.0]],
    ),
    "131": _wiski("05BB803", "TA", "C.Corrected-Sensor", "°C", _TA_DAY),
    # The "daily" TA series belongs to a different sensor → ignored.
    "132": _wiski("05BB803", "TA2", "A.DayMean", "°C", [["2026-03-01 00:00:00", None]]),
    "142": _wiski("05BB803", "PC", "A.DayTotal", "mm", [["2026-03-01 00:00:00", 1.4]]),
}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status == 200
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._payload


class _StubSession:
    """Answers proxy.ashx MapServer queries and WISKI downloads."""

    def __init__(self):
        self.headers = {}
        self.urls = []

    def request(self, method, url, params=None, timeout=None):
        self.urls.append(url)
        decoded = urllib.parse.unquote_plus(url)
        if "proxy.ashx" in url:
            if "/39/query" in decoded:
                if "'SW-SS'" in decoded:
                    return _Resp({"features": [{"attributes": r} for r in SNOW_ROWS]})
                return _Resp({"features": [{"attributes": r} for r in EXTRA_ROWS]})
            if "/11/query" in decoded:
                return _Resp({"features": [{"attributes": r} for r in POINT_ROWS]})
        if "WiskiLiveDataService/Download" in url:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            ts_id = qs["tsId"][0]
            if ts_id in DOWNLOADS:
                return _Resp(DOWNLOADS[ts_id])
            return _Resp({"error": "boom"}, status=404)
        return _Resp({}, status=404)


@pytest.fixture
def stub():
    return _StubSession()


@pytest.fixture
def client(stub):
    return AlbertaClient(session=stub, max_retries=1)


# ── Registry compliance ──────────────────────────────────────────────────────


def test_variables_use_shared_vocabulary():
    required = {"name", "type", "units", "output_units", "description", "notes"}
    for key, info in VARIABLES.items():
        assert required | {"source"} <= set(info), key
        assert info["type"] in TYPES, key
        assert set(info["intervals"]) <= INTERVALS, key


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES,):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_snow_variables_emit_cm():
    for key, info in VARIABLES.items():
        if info["type"] in {"swe", "snwd"}:
            assert info["output_units"] == "cm", key


def test_data_flags_empty():
    assert DATA_FLAGS == {}


# ── Parsers / helpers ────────────────────────────────────────────────────────


def test_parse_download_json():
    header, rows = _parse_download_json(DOWNLOADS["101"])
    assert header["ts_name"] == "C.Corrected-Seasonal"
    assert rows[0] == (datetime(2026, 3, 1, 0, 0), 565.0)
    assert rows[2][1] is None
    assert _parse_download_json([]) == ({}, [])
    with pytest.raises(AlbertaError):
        _parse_download_json({"nope": 1})


def test_convert_units_and_clamp():
    assert _convert("SW", 565.0) == pytest.approx(56.5)  # mm → cm
    assert _convert("SD", 142.9) == pytest.approx(142.9)
    assert _convert("SW", -3.0) is None  # negative pillow drift
    assert _convert("SD", 20000) is None  # > 15 m
    assert _convert("TA", -12.5) == pytest.approx(-12.5)
    assert _convert("SW", None) is None


def test_epoch_wall_clock_decode():
    assert _epoch_ms_to_mst_wall(_wall_ms("1984-03-12 11:00")) == datetime(
        1984, 3, 12, 11, 0
    )
    assert _epoch_ms_to_mst_wall(None) is None


def test_split_name():
    assert _split_name("Sunshine Village - EPA") == ("Sunshine Village", "EPA")
    assert _split_name("Many Glacier - Snotel") == ("Many Glacier", "Snotel")
    assert _split_name("Wildcat Cr. Mistaya Lodge") == (
        "Wildcat Cr. Mistaya Lodge",
        None,
    )


def test_daily_means_require_coverage():
    rows = [(datetime(2026, 3, 1, h), h) for h in range(24)]
    rows += [(datetime(2026, 3, 2, h), 1.0) for h in range(5)]
    out = _daily_means(rows, "TA")
    assert out == [(datetime(2026, 3, 1), pytest.approx(11.5))]


# ── Variable / interval resolution ───────────────────────────────────────────


def test_resolve_defaults_per_interval():
    assert _resolve_variables(None, "daily") == ["SW", "SD"]
    assert _resolve_variables(None, "periodic") == ["SW-SS", "SD-SS"]
    assert _resolve_variables([], "daily") == ["SW", "SD"]  # empty = default


def test_resolve_types_follow_interval():
    assert _resolve_variables("swe", "daily") == ["SW"]
    assert _resolve_variables("swe", "periodic") == ["SW-SS"]
    assert _resolve_variables(["temp", "sw"], "hourly") == ["TA", "SW"]


def test_resolve_errors():
    with pytest.raises(AlbertaError, match="Unknown variable"):
        _resolve_variables("WTEQ", "daily")
    with pytest.raises(AlbertaError, match="not available"):
        _resolve_variables("SW-SS", "daily")
    with pytest.raises(AlbertaError, match="No Alberta variable"):
        _resolve_variables("precip", "hourly")


def test_unsupported_interval_raises(client):
    with pytest.raises(AlbertaError, match="Unsupported interval"):
        client.get_data("05BB803", interval="monthly")


def test_get_data_requires_station_or_bbox(client):
    with pytest.raises(ValueError):
        client.get_data()


# ── Client behaviour against the stub ───────────────────────────────────────


def test_session_sends_referer(client, stub):
    assert stub.headers["Referer"] == "https://rivers.alberta.ca/"
    assert stub.headers["User-Agent"] == "easysnowdata"


def test_get_all_stations(client):
    by_id = {s["station_id"]: s for s in client.get_all_stations()}
    assert set(by_id) == {"05BB803", "4A37P", "07AA001"}
    sun = by_id["05BB803"]
    assert sun["name"] == "Sunshine Village"
    assert sun["operator"] == "Alberta Environment and Protected Areas"
    assert sun["latitude"] == pytest.approx(51.077453)
    assert sun["elevation_m"] is None
    assert sun["status"] == "Active"
    assert sun["has_pillow_swe"] and sun["has_survey"] and sun["has_snwd"]
    assert sun["parameters"] == ["PC", "SD", "SW", "SW-SS", "TA"]
    assert sun["begin_date"] == "1970-02-25"
    assert by_id["4A37P"]["latitude"] is None
    assert by_id["4A37P"]["state"] == "BC"
    assert by_id["07AA001"]["status"] == "Inactive"
    assert by_id["07AA001"]["source_status"] == "Discontinued"


def test_active_only_and_bbox(client):
    assert {s["station_id"] for s in client.get_all_stations(active_only=True)} == {
        "05BB803",
        "4A37P",
    }
    kept = client.get_all_stations(bbox=(-117, 50, -115, 52))
    assert [s["station_id"] for s in kept] == ["05BB803"]


def test_get_metadata(client):
    meta = client.get_metadata("05BB803")
    combos = {(d["name"], d["interval"]) for d in meta["data_variables"]}
    assert ("SW", "daily") in combos and ("SW-SS", "periodic") in combos
    assert ("PC", "daily") in combos and ("PC", "hourly") not in combos
    assert all(d["interval"] in INTERVALS for d in meta["data_variables"])
    sw = next(d for d in meta["data_variables"] if d["name"] == "SW")
    assert sw["begin_date"] == "1984-03-12"
    with pytest.raises(AlbertaError, match="not found"):
        client.get_metadata("NOPE")


def test_get_data_daily_uses_source_aggregate(client):
    recs = client.get_data(
        "05BB803", variables="SW", begin_date="2026-03-01", end_date="2026-03-03"
    )
    assert all(set(RECORD_KEYS) <= set(r) for r in recs)
    assert [r["date"] for r in recs] == ["2026-03-01", "2026-03-02", "2026-03-03"]
    assert recs[1]["value"] == pytest.approx(56.4641, abs=1e-3)
    assert recs[2]["value"] is None
    assert all(r["units"] == "cm" and r["interval"] == "daily" for r in recs)
    assert "datetime" not in recs[0]


def test_get_data_hourly_and_sub_hourly(client):
    sub = client.get_data(
        "05BB803", variables="SW", interval="sub_hourly", end_date="2026-03-01"
    )
    assert len(sub) == 4
    assert sub[1]["datetime"] == "2026-03-01T00:15:00-07:00"
    hourly = client.get_data(
        "05BB803", variables="SW", interval="hourly", include_flags=True
    )
    assert [r["datetime"][11:16] for r in hourly] == [
        "00:00",
        "01:00",
        "02:00",
        "00:00",
    ]
    assert hourly[1]["value"] is None and hourly[2]["value"] is None
    assert all(r["flag"] is None for r in hourly)


def test_get_data_periodic_survey(client):
    recs = client.get_data("05BB803", interval="periodic")
    assert [(r["variable"], r["date"], r["value"]) for r in recs] == [
        ("SW-SS", "2026-03-04", pytest.approx(55.3)),
        ("SW-SS", "2026-03-27", pytest.approx(73.2)),
    ]
    assert all(r["interval"] == "periodic" for r in recs)


def test_ta_daily_is_computed_when_aggregate_is_other_sensor(client):
    recs = client.get_data("05BB803", variables="TA", interval="daily")
    assert [(r["date"], r["value"]) for r in recs] == [
        ("2026-03-01", pytest.approx(11.5))
    ]
    assert recs[0]["units"] == "°C"


def test_pc_daily_total(client):
    recs = client.get_data("05BB803", variables="precip")
    assert [(r["variable"], r["value"], r["units"]) for r in recs] == [
        ("PC", pytest.approx(1.4), "mm")
    ]


def test_failed_download_raises_unless_partial_is_allowed(client):
    # 4A37P's series ids are not served by the stub.
    with pytest.raises(AlbertaError):
        client.get_data("4A37P", variables="SW", interval="sub_hourly")
    client.allow_partial = True
    assert client.get_data("4A37P", variables="SW", interval="sub_hourly") == []


def test_unknown_station_is_skipped(client):
    assert client.get_data("UNKNOWN") == []


def test_bbox_get_data(client):
    recs = client.get_data(bbox=(-117, 50, -115, 52), interval="periodic")
    assert {r["station_id"] for r in recs} == {"05BB803"}


def test_query_layer_pages(stub):
    pages = [
        {"features": [{"attributes": {"a": 1}}], "exceededTransferLimit": True},
        {"features": [{"attributes": {"a": 2}}]},
    ]

    def request(method, url, params=None, timeout=None):
        stub.urls.append(url)
        return _Resp(pages[len(stub.urls) - 1])

    stub.request = request
    rows = AlbertaClient(session=stub, max_retries=1)._query_layer(39, "1=1")
    assert rows == [{"a": 1}, {"a": 2}]
    assert "resultOffset=1" in stub.urls[1]


def test_query_layer_error_payload(stub):
    stub.request = lambda *a, **k: _Resp({"error": {"code": 499}})
    with pytest.raises(AlbertaError, match="query failed"):
        AlbertaClient(session=stub, max_retries=1)._query_layer(39, "1=1")


# ── Live tests ───────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_catalog():
    stations = AlbertaClient().get_all_stations()
    pillows = [s for s in stations if s["has_pillow_swe"]]
    assert len(stations) > 200
    assert len(pillows) >= 40
    assert any(s["station_id"] == "05BB803" for s in pillows)


@pytest.mark.live
def test_live_sunshine_village_values():
    """Sunshine Village pillow (05BB803), early March 2026."""
    c = AlbertaClient()
    daily = c.get_data(
        "05BB803", variables="SW", begin_date="2026-03-01", end_date="2026-03-01"
    )
    assert daily[0]["value"] == pytest.approx(56.5)
    surveys = c.get_data(
        "05BB803",
        interval="periodic",
        variables="SW-SS",
        begin_date="2026-03-01",
        end_date="2026-03-31",
    )
    assert [r["value"] for r in surveys] == [pytest.approx(55.3), pytest.approx(73.2)]
