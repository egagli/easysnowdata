"""
Tests for HydroQuebecClient.

The offline tests feed synthetic Opendatasoft ``exports/json`` payloads
(shaped like the real ``donnees-hydrometeorologiques`` rows, including the
``date = null`` junk rows and ``-999.00`` sentinels) through a stub
session.  The ``live`` tests hit the real portal (no credentials).
"""

import json

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.hydroquebec import (
    HydroQuebecClient,
    HydroQuebecError,
)
from easysnowdata.stations.clients.hydroquebec.hydroquebec_client import (
    DATA_FLAGS,
    MIN_HOURS_DAILY,
    VARIABLES,
    _aggregate,
    _build_stations,
    _convert,
    _parse_utc,
    _resolve_variables,
    _site_date,
    _station_code,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

SWE = "Équivalent en eau de la neige"
SD = "Épaisseur de neige"
TX = "Température Maximum"
PR = "Précipitation"

_SITE = {
    "1-10613": {
        "nom": "Cooper Ouvert (CM3V)",
        "xcoord": -71.3,
        "ycoord": 48.8492,
        "zcoord": 506.0,
        "regionqc": "Saguenay--Lac-Saint-Jean",
        "coderegionqc": "02",
        "date_debut": "2016-09-01T00:00:00+00:00",
        "date_fin": None,
    },
    "1-2948": {
        "nom": "Baie-des-Rapides",
        "xcoord": -76.73,
        "ycoord": 47.23,
        "zcoord": None,
        "regionqc": "Outaouais",
        "coderegionqc": "07",
        "date_debut": "1800-01-01T00:00:00+00:00",
        "date_fin": None,
    },
    "1-15884": {
        "nom": "Bernard (ULaval)",
        "xcoord": -70.0,
        "ycoord": 50.0,
        "zcoord": None,
        "regionqc": "Côte-Nord",
        "coderegionqc": "09",
        "date_debut": "2023-01-01T00:00:00+00:00",
        "date_fin": None,
    },
}

#: group_by over every row (incl. junk rows) per station × variable.
ALL_GROUPS = [
    {"identifiant": "1-10613", **_SITE["1-10613"], "variable": SWE, "n_rows": 10},
    {"identifiant": "1-10613", **_SITE["1-10613"], "variable": SD, "n_rows": 10},
    {"identifiant": "1-10613", **_SITE["1-10613"], "variable": TX, "n_rows": 10},
    {"identifiant": "1-2948", **_SITE["1-2948"], "variable": "Niveau", "n_rows": 10},
    {"identifiant": "1-15884", **_SITE["1-15884"], "variable": SWE, "n_rows": 4},
]

#: group_by over valid rows only (date not null, valeur != -999.00).
VALID_GROUPS = [
    {
        "identifiant": "1-10613",
        **_SITE["1-10613"],
        "variable": v,
        "n_obs": 6,
        "first": "2026-10-04T08:00:00+00:00",
        "last": "2026-10-14T11:00:00+00:00",
    }
    for v in (SWE, SD, TX)
] + [
    {
        "identifiant": "1-2948",
        **_SITE["1-2948"],
        "variable": "Niveau",
        "n_obs": 6,
        "first": "2026-10-04T07:00:00+00:00",
        "last": "2026-10-14T10:00:00+00:00",
    }
]


def _row(var, date, valeur, kind="Instantanée", sid="1-10613"):
    return {
        "identifiant": sid,
        "composition_depil_type_point_donnee": var,
        "composition_depil_type_mesure": kind,
        "composition_depil_nom_unite_mesure": "mm",
        "date": date,
        "valeur": valeur,
    }


# 2026-10-07 local (EDT, UTC-4) runs 04:00Z on the 7th → 04:00Z on the 8th.
DATA_ROWS = [
    _row(SWE, "2026-10-07T03:00:00+00:00", "15.00"),  # local 10-06
    _row(SWE, "2026-10-07T07:00:00+00:00", "18.00"),
    _row(SWE, "2026-10-07T19:00:00+00:00", "19.00"),
    _row(SWE, "2026-10-08T01:00:00+00:00", "20.00"),  # last of local 10-07
    _row(SWE, "2026-10-08T07:00:00+00:00", "-999.00"),  # local 10-08, missing
    _row(SD, "2026-10-07T07:00:00+00:00", "-0.33"),  # negative noise → None
    _row(SD, "2026-10-07T08:00:00+00:00", "2.50"),
    _row(TX, "2026-10-07T12:00:00+00:00", "5.10", kind="Maximum"),
] + [
    _row(PR, f"2026-10-07T{h:02d}:00:00+00:00", "0.50", kind="Somme")
    for h in range(4, 24)
]


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.text = json.dumps(payload)
        self.status_code = status
        self.ok = status == 200
        self.headers = {}

    def json(self):
        return self._payload


class _StubSession:
    """Dispatches on the ODSQL ``select`` the client sends."""

    def __init__(self, data_rows=None):
        self.headers = {}
        self.calls = []
        self.data_rows = DATA_ROWS if data_rows is None else data_rows

    def request(self, method, url, params=None, timeout=None):
        self.calls.append(params or {})
        if not url.endswith("/donnees-hydrometeorologiques/exports/json"):
            return _Resp({"error": "not found"}, 404)
        select = (params or {}).get("select", "")
        if "n_rows" in select:
            return _Resp(ALL_GROUPS)
        if "n_obs" in select:
            return _Resp(VALID_GROUPS)
        return _Resp(self.data_rows)


@pytest.fixture
def stub():
    return _StubSession()


@pytest.fixture
def client(stub):
    return HydroQuebecClient(session=stub, max_retries=1)


# ── Registry compliance ──────────────────────────────────────────────────────


def test_variables_use_shared_vocabulary():
    required = {"name", "type", "units", "output_units", "description", "notes"}
    for key, info in VARIABLES.items():
        assert required | {"source", "daily"} <= set(info), key
        assert info["type"] in TYPES, key


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES,):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_snow_variables_emit_cm():
    assert VARIABLES[SWE]["output_units"] == "cm"
    assert VARIABLES[SD]["output_units"] == "cm"


def test_data_flags_empty():
    assert DATA_FLAGS == {}


# ── Parsers / conversion ─────────────────────────────────────────────────────


def test_convert_units_sentinels_and_clamps():
    assert _convert(SWE, "75.00") == pytest.approx(7.5)  # mm → cm
    assert _convert(SD, "38.27") == pytest.approx(38.27)  # cm stays cm
    assert _convert(SWE, "-999.00") is None
    assert _convert(SD, "-0.33") is None  # negative noise
    assert _convert(SD, "1600") is None  # > 15 m
    assert _convert(PR, "-1642.71") is None
    assert _convert(TX, "-12.5") == pytest.approx(-12.5)
    assert _convert(SWE, "Horaire") is None  # junk row label


def test_parse_utc_and_site_date():
    dt = _parse_utc("2026-10-04T08:00:00+00:00")
    assert dt.hour == 8 and dt.utcoffset().total_seconds() == 0
    assert _parse_utc(None) is None
    assert _site_date("1800-01-01T00:00:00+00:00") is None
    assert _site_date("2016-09-01T00:00:00+00:00") == "2016-09-01"


def test_station_code_from_name():
    assert _station_code("Cooper Ouvert (CM3V)") == "CM3V"
    assert _station_code("Baie-des-Rapides") is None


def test_aggregate_methods():
    assert _aggregate("last", [1.0, 2.0]) == 2.0
    assert _aggregate("last", []) is None
    hours = [1.0] * MIN_HOURS_DAILY
    assert _aggregate("sum", hours) == pytest.approx(MIN_HOURS_DAILY)
    assert _aggregate("sum", hours[:-1]) is None  # too few hours
    assert _aggregate("max", [1.0] * 19 + [5.0]) == 5.0
    assert _aggregate("circmean", [350.0] * 10 + [10.0] * 10) == 0.0


def test_circmean_wraps_to_zero_not_360():
    # atan2 gives a tiny negative angle here; it must wrap to 0.0, not 360.0.
    assert _aggregate("circmean", [350.0, 10.0] * 12) == 0.0
    assert _aggregate("circmean", [90.0] * 24) == 90.0


def test_build_stations_status_and_flags():
    st = _build_stations(ALL_GROUPS, VALID_GROUPS)
    assert set(st) == {"1-10613", "1-2948", "1-15884"}
    cooper = st["1-10613"]
    assert cooper["status"] == "Active"
    assert cooper["has_swe"] and cooper["has_snwd"]
    assert cooper["hq_code"] == "CM3V"
    assert cooper["latitude"] == pytest.approx(48.8492)
    assert cooper["latest_observation"] == "2026-10-14T11:00:00+00:00"
    assert "GMON" in cooper["measurement_method"]
    assert st["1-15884"]["status"] == "Inactive"  # only junk rows
    assert st["1-2948"]["commissioned"] is None  # 1800 placeholder


# ── Variable / interval resolution ───────────────────────────────────────────


def test_resolve_defaults_to_snow():
    assert _resolve_variables(None) == [SWE, SD]


def test_resolve_types_and_labels():
    assert _resolve_variables(["swe", TX]) == [SWE, TX]
    assert _resolve_variables("other") == [
        "Niveau",
        "Débit",
        "Température de l'eau",
    ]


def test_unknown_variable_raises():
    with pytest.raises(HydroQuebecError, match="Unknown variable"):
        _resolve_variables("WTEQ")


def test_unsupported_interval_raises(client):
    with pytest.raises(HydroQuebecError, match="Unsupported interval"):
        client.get_data("1-10613", interval="monthly")


def test_get_data_requires_station_or_bbox(client):
    with pytest.raises(ValueError):
        client.get_data()


# ── Client against the stub ──────────────────────────────────────────────────


def test_get_all_stations_is_snow_only(client):
    ids = {s["station_id"] for s in client.get_all_stations()}
    assert ids == {"1-10613", "1-15884"}
    active = client.get_all_stations(active_only=True)
    assert [s["station_id"] for s in active] == ["1-10613"]
    every = {s["station_id"] for s in client.get_stations(variables="all")}
    assert "1-2948" in every


def test_get_stations_bbox(client):
    kept = client.get_all_stations(bbox=(-72, 48, -70, 49))
    assert [s["station_id"] for s in kept] == ["1-10613"]


def test_get_metadata(client):
    meta = client.get_metadata("1-10613")
    names = {d["name"] for d in meta["data_variables"]}
    assert names == {SWE, SD, TX}
    assert all(d["interval"] in INTERVALS for d in meta["data_variables"])
    with pytest.raises(HydroQuebecError, match="not found"):
        client.get_metadata("1-0")


def test_get_data_hourly_records(client, stub):
    recs = client.get_data("1-10613", include_flags=True)
    assert all(set(RECORD_KEYS) <= set(r) for r in recs)
    assert {r["variable"] for r in recs} == {SWE, SD}
    by_dt = {(r["variable"], r["datetime"]): r for r in recs}
    assert by_dt[(SWE, "2026-10-07T07:00:00Z")]["value"] == pytest.approx(1.8)
    assert by_dt[(SWE, "2026-10-08T07:00:00Z")]["value"] is None
    assert by_dt[(SD, "2026-10-07T07:00:00Z")]["value"] is None
    assert by_dt[(SD, "2026-10-07T08:00:00Z")]["flag"] == "Instantanée"
    assert all(r["interval"] == "hourly" and r["units"] == "cm" for r in recs)
    where = stub.calls[-1]["where"]
    assert "date is not null" in where and '"1-10613"' in where


def test_get_data_hourly_date_filter_is_utc(client):
    recs = client.get_data("1-10613", variables="swe", begin_date="2026-10-08")
    assert {r["date"] for r in recs} == {"2026-10-08"}


def test_get_data_daily_local_day_aggregation(client):
    recs = client.get_data(
        "1-10613",
        variables=["swe", "snwd", "precip", "temp_max"],
        interval="daily",
        begin_date="2026-10-06",
        end_date="2026-10-08",
    )
    got = {(r["variable"], r["date"]): r["value"] for r in recs}
    assert got[(SWE, "2026-10-06")] == pytest.approx(1.5)  # 03Z = 23:00 EDT
    assert got[(SWE, "2026-10-07")] == pytest.approx(2.0)  # last of local day
    assert got[(SWE, "2026-10-08")] is None  # only a -999 value
    assert got[(SD, "2026-10-07")] == pytest.approx(2.5)
    assert got[(PR, "2026-10-07")] == pytest.approx(10.0)  # 20 h × 0.5 mm
    assert got[(TX, "2026-10-07")] is None  # 1 h < MIN_HOURS_DAILY
    assert all("datetime" not in r and r["interval"] == "daily" for r in recs)


def test_get_data_bbox_uses_station_list(client):
    recs = client.get_data(bbox=(-72, 48, -70, 49))
    assert {r["station_id"] for r in recs} == {"1-10613"}


def test_portal_error_envelope_raises():
    class _ErrSession(_StubSession):
        def request(self, method, url, params=None, timeout=None):
            return _Resp({"error_code": "ODSQLSyntaxError", "message": "bad"})

    c = HydroQuebecClient(session=_ErrSession(), max_retries=1)
    with pytest.raises(HydroQuebecError, match="bad"):
        c.get_data("1-10613")


# ── Live tests ───────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_snow_station_inventory():
    stations = HydroQuebecClient().get_all_stations()
    swe = [s for s in stations if s["has_swe"]]
    assert len(swe) >= 80
    assert all(45 < s["latitude"] < 63 and -80 < s["longitude"] < -57 for s in swe)


@pytest.mark.live
def test_live_hourly_and_daily_swe():
    client = HydroQuebecClient()
    sid = next(
        s["station_id"]
        for s in client.get_all_stations(active_only=True)
        if s["has_swe"]
    )
    hourly = client.get_data(sid, variables="swe")
    assert hourly and all(r["datetime"].endswith("Z") for r in hourly)
    daily = client.get_data(sid, variables="swe", interval="daily")
    assert 5 <= len({r["date"] for r in daily}) <= 13
