"""
Tests for SNIHClient.

Offline tests feed payloads in the real layout of the SNIH page methods
(``{"d": ...}`` wrappers, .NET ``/Date(ms)/`` timestamps, the real
station rows of Toscas, an RMET pillow and a river gauge, trimmed) through
a stub session that routes by endpoint and request body.  ``live`` tests
hit the real portal (no credentials).
"""

from datetime import date, datetime

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.snih import SNIHClient, SNIHError
from easysnowdata.stations.clients.snih import snih_client as sc
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

BASE = "https://snih.test"
TODAY = date(2026, 10, 15)


def ms(local: str) -> str:
    """Local Argentina time ``YYYY-MM-DDTHH:MM`` → ``/Date(ms)/``."""
    ts = datetime.fromisoformat(local).replace(tzinfo=sc.ART)
    return f"/Date({int(ts.timestamp() * 1000)})/"


# Real LeerEstaciones rows (October 2026), trimmed of a few long fields.
TOSCAS = {
    "__type": "Compartido.Contenedor.CEstacionesCN",
    "Codigo": 19145,
    "Descripcion": "Toscas",
    "Regional": 4,
    "Sistema": 6,
    "Cuenca": 55,
    "Red": 1,
    "Provincia": 8,
    "Rio": "Mendoza",
    "Lugar": "Toscas",
    "Cota": 2950,
    "Latitud": 33.1641666666667,
    "Longitud": 69.8886111111111,
    "Habilitada": True,
    "Tipo": "N",
    "Transmision": "T",
    "Actual": "S",
}
DURAZNO = {
    **TOSCAS,
    "Codigo": 281523,
    "Descripcion": (
        "Cuenca del Río Tunuyán-Nacientes del Río Colorado Nacientes del "
        "Arroyo El Duraz"
    ),
    "Regional": -1,
    "Cuenca": 57,
    "Red": 28,
    "Rio": "NA",
    "Lugar": "Nacientes del Arroyo el Durazno",
    "Cota": -999,
    "Latitud": 34.0177777777778,
    "Longitud": 69.7374166666667,
}
VALLE_DE_UCO = {
    **TOSCAS,
    "Codigo": 11419,
    "Descripcion": "Valle de Uco",
    "Cuenca": 57,
    "Rio": "Tunuyán",
    "Lugar": "Valle de Uco",
    "Cota": 1199,
    "Latitud": 33.7764722222222,
    "Longitud": 69.2726277777778,
    "Tipo": "HMA",
}
# A non-snow station, listed twice as the real list does for two codes.
PERGAMINO = {
    **TOSCAS,
    "Codigo": 111872,
    "Descripcion": "25 de Mayo - EEA Pergamino",
    "Red": 11,
    "Tipo": "M",
}

STATIONS = [TOSCAS, DURAZNO, VALLE_DE_UCO, PERGAMINO, PERGAMINO]
NETWORKS = [
    {
        "__type": "Compartido.Contenedor.CAgrupacionesCN",
        "Clase": 2,
        "Codigo": 1,
        "Descripcion": "Red Hidrológica Nacional",
    },
    {
        "__type": "Compartido.Contenedor.CAgrupacionesCN",
        "Clase": 2,
        "Codigo": 28,
        "Descripcion": "Red Meteorológica RMET",
    },
]


def _assoc(sid, code, since):
    return {
        "__type": "Compartido.Contenedor.CAsociacionesCN",
        "Estacion": sid,
        "Codigo": code,
        "Desde": ms(since),
        "Hasta": "/Date(7289578800000)/",
        "Minimo": 0,
        "Maximo": 1500,
        "TipoValidacion": 11,
        "Actual": "S",
    }


ASSOCIATIONS = [
    *(_assoc(19145, c, "1951-08-01T00:00") for c in (70, 71, 72, 74)),
    _assoc(19145, 73, "1989-04-01T00:00"),
    _assoc(281523, 73, "2025-04-09T00:00"),
    _assoc(11419, 70, "2014-12-19T00:00"),
    _assoc(151492, 73, "2006-10-01T00:00"),  # not in the station list
    _assoc(111872, 20, "2019-12-12T00:00"),  # precipitation, not snow
]


def _hist(local, value, cal="", validated=True):
    return {
        "FechaHora": ms(local),
        "Medicion": value,
        "Calificador": cal,
        "Validado": validated,
    }


#: LeerDatosHistoricos series by (station, code).
HISTORICAL = {
    ("19145", "73"): [
        _hist("2005-05-01T06:00", 0),
        _hist("2005-05-01T08:00", 4),  # second value of the same day
        _hist("2015-07-01T12:00", 20),
        _hist("2015-07-02T12:00", 30, " "),
        _hist("2015-07-03T12:00", -999),
        _hist("2015-07-04T12:00", 35, "D"),
        _hist("2015-07-05T12:00", 40, "", False),
        _hist("2015-07-06T12:00", 16000),  # 16 m of water
        _hist("2023-07-02T00:00", 210),
    ],
    ("19145", "74"): [_hist("1951-08-24T12:00", 133), _hist("2008-09-12T10:25", 699)],
    ("19145", "70"): [_hist("1951-08-24T12:00", 40), _hist("2008-09-12T10:25", 192.6)],
    ("19145", "72"): [_hist("1951-08-24T12:00", 0.34)],
}


def _block(rows):
    return [
        {
            "ExtensionData": {},
            "FechaHora": ms(t),
            "Mediciones": [{"Codigo": 73, "Valor": v}],
        }
        for t, v in rows
    ]


#: Raw telemetry in two logger blocks, as at Toscas (6-hourly, then 4-hourly).
RAW_BLOCKS = {
    "19145": [
        [("2025-09-26T21:00", 0.0), ("2025-09-27T03:00", 0.0)],
        [
            ("2026-10-13T20:00", 557.0),
            ("2026-10-14T00:00", 562.99997568130493),
            ("2026-10-14T16:00", 567.0),
            ("2026-10-15T16:00", 647.0),
        ],
    ],
    "281523": [[("2026-10-15T12:00", 1197)]],
}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.text = str(payload)
        self.headers = {}

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not JSON")
        return self._payload


class _StubSession:
    """Routes page-method POSTs by endpoint and JSON body."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, params=None, timeout=None, json=None):
        self.calls.append((method, url.rsplit("/", 1)[-1], json))
        name = url.rsplit("/", 1)[-1]
        if name == "LeerEstaciones":
            return _Resp({"d": STATIONS})
        if name == "leerListaRedes":
            return _Resp({"d": NETWORKS})
        if name == "LeerListaAsociaciones":
            return _Resp({"d": ASSOCIATIONS})
        if name == "LeerDatosHistoricos":
            rows = HISTORICAL.get((json["estacion"], json["codigo"]), [])
            lo = datetime.fromisoformat(json["fechaDesde"]).replace(tzinfo=sc.ART)
            hi = datetime.fromisoformat(json["fechaHasta"]).replace(tzinfo=sc.ART)
            keep = [r for r in rows if lo <= sc.parse_ms_date(r["FechaHora"]) <= hi]
            return _Resp(
                {
                    "d": {
                        "__type": "WebSSRHv2_1.RespuestaMediciones",
                        "RespuestaOK": "True",
                        "MsgErr": "",
                        "Mediciones": keep,
                    }
                }
            )
        if name == "LeerUltimosRegistros":
            lo = date.fromisoformat(json["fechaDesde"])
            hi = date.fromisoformat(json["fechaHasta"])
            # Like the portal: only the first block holding data from
            # fechaDesde on, and up to one day past fechaHasta.
            out = []
            for block in RAW_BLOCKS.get(json["estacion"], []):
                rows = [
                    (t, v)
                    for t, v in block
                    if lo
                    <= date.fromisoformat(t[:10])
                    <= date.fromordinal(hi.toordinal() + 1)
                ]
                if rows:
                    out = rows
                    break
            return _Resp(
                {
                    "d": {
                        "__type": "WebSSRHv2_1.WCFIntegracion.CRespuestaMedicionesBlock",
                        "Mediciones": _block(out),
                        "ExtensionData": {},
                        "MsgErr": "",
                        "RespuestaOK": True,
                    }
                }
            )
        raise AssertionError(f"unexpected endpoint {url}")


class _NoNetwork:
    headers = {}

    def request(self, *args, **kwargs):
        raise AssertionError("network call made")


@pytest.fixture
def stub():
    return _StubSession()


@pytest.fixture
def client(stub, monkeypatch):
    monkeypatch.setattr(sc, "_today", lambda: TODAY)
    return SNIHClient(base_url=BASE, session=stub, max_retries=1, backoff=0)


# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_follows_the_contract():
    assert set(sc.VARIABLES) == {
        "SWE_PILLOW",
        "SWE_COURSE",
        "SNWD",
        "DENSITY",
        "CORE_LENGTH",
    }
    for info in sc.VARIABLES.values():
        assert info["type"] in TYPES
        if info["type"] != "other":
            assert info["output_units"] == canonical_units(info["type"])
        assert {"units", "description", "notes", "source", "code"} <= set(info)
        assert set(info["intervals"]) <= INTERVALS
    assert sc.VARIABLES["SWE_PILLOW"]["code"] == 73
    assert sc.VARIABLES["DENSITY"]["scale"] == 100.0
    assert {"D", "raw", "unvalidated"} <= set(sc.DATA_FLAGS)


def test_resolve_variables():
    assert sc._resolve_variables(None, "daily") == ["SWE_PILLOW"]
    assert sc._resolve_variables(None, "sub_daily") == ["SWE_PILLOW"]
    assert sc._resolve_variables(None, "periodic") == ["SWE_COURSE", "SNWD"]
    assert sc._resolve_variables("swe", "periodic") == ["SWE_COURSE"]
    assert sc._resolve_variables(["73", "h_spilow", "swe_pillow"], "daily") == [
        "SWE_PILLOW"
    ]
    assert sc._resolve_variables(["density", "EspNieve"], "periodic") == [
        "DENSITY",
        "SNWD",
    ]
    with pytest.raises(SNIHError, match="Unknown"):
        sc._resolve_variables("bogus", "daily")
    with pytest.raises(SNIHError, match="not offered"):
        sc._resolve_variables("SNWD", "daily")
    with pytest.raises(SNIHError, match="No SNIH 'other'"):
        sc._resolve_variables("other", "sub_daily")


def test_parse_ms_date_lands_on_the_argentine_day():
    # 2015-07-01 15:00 UTC is noon in Argentina.
    assert sc.parse_ms_date("/Date(1435762800000)/").isoformat() == (
        "2015-07-01T12:00:00-03:00"
    )
    # Midnight local is 03:00 UTC; it must stay on its own day.
    assert sc.parse_ms_date("/Date(1688266800000)/").date() == date(2023, 7, 2)
    assert sc.parse_ms_date("/Date(-579258000000)/").date() == date(1951, 8, 24)
    assert sc.parse_ms_date("/Date(1435762800000-0300)/").hour == 12
    assert sc.parse_ms_date("2015-07-01") is None


def test_convert_units_and_clamps():
    assert sc._convert(358, "SWE_PILLOW") == 35.8
    assert sc._convert(562.99997568130493, "SWE_PILLOW") == 56.3
    assert sc._convert(0.34, "DENSITY") == 34.0
    assert sc._convert(192.6, "SNWD") == 192.6
    assert sc._convert(-999, "SWE_PILLOW") is None
    assert sc._convert(-999.5, "SWE_PILLOW") is None
    assert sc._convert(16000, "SWE_PILLOW") is None  # > 15 m
    assert sc._convert(-5, "SWE_COURSE") is None
    assert sc._convert(1.2, "DENSITY") is None  # > 100 %
    assert sc._convert(None, "SNWD") is None


def test_parse_station():
    s = sc.parse_station(
        TOSCAS, {1: "Red Hidrológica Nacional"}, {73: "1989-04-01", 74: "1951-08-01"}
    )
    assert (s["station_id"], s["name"]) == ("19145", "Toscas")
    assert (s["latitude"], s["longitude"], s["elevation_m"]) == (
        -33.164167,
        -69.888611,
        2950.0,
    )
    assert s["province"] == "Mendoza" and s["network"] == "Red Hidrológica Nacional"
    assert s["operator"] == s["network"]
    assert s["transmission"] == "telemetric"
    assert s["station_type_description"] == "Nivométrico"
    assert s["variables"] == ["SWE_PILLOW", "SWE_COURSE"]
    assert s["has_swe"] is True and s["has_snwd"] is False
    assert s["measurement_method"].startswith("Snow pillow")
    assert sc.parse_station(DURAZNO, {}, {73: None})["elevation_m"] is None
    assert (
        sc.parse_station({**TOSCAS, "Cota": 0}, {}, {73: None})["elevation_m"] is None
    )


# ── Stations ────────────────────────────────────────────────────────────────


def test_get_all_stations(client, stub):
    stations = {s["station_id"]: s for s in client.get_all_stations()}
    # Snow scope: the duplicated non-snow station and the unlisted 151492 drop out.
    assert list(stations) == ["19145", "281523", "11419"]
    for s in stations.values():
        assert {
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "status",
            "operator",
            "measurement_method",
            "has_swe",
            "has_snwd",
            "station_url",
        } <= set(s)
    assert stations["19145"]["status"] == "Active"
    assert stations["19145"]["last_raw_obs"] == "2026-10-15T16:00:00-03:00"
    assert stations["281523"]["network"] == "Red Meteorológica RMET"
    # A river gauge with only a snow-depth association, and no pillow.
    uco = stations["11419"]
    assert uco["status"] == "Inactive" and uco["last_raw_obs"] is None
    assert uco["has_snwd"] is True and uco["has_swe"] is False
    assert stations["19145"]["variables_since"]["SWE_PILLOW"] == "1989-04-01"
    # Only the two pillow stations were checked for status, once each.
    raw = [c for c in stub.calls if c[1] == "LeerUltimosRegistros"]
    assert sorted(c[2]["estacion"] for c in raw) == ["19145", "281523"]
    client.get_all_stations(active_only=True)
    assert len([c for c in stub.calls if c[1] == "LeerEstaciones"]) == 1


def test_get_all_stations_filters(client):
    assert [s["station_id"] for s in client.get_all_stations(active_only=True)] == [
        "19145",
        "281523",
    ]
    south = client.get_all_stations(bbox=(-71, -34.5, -69, -33.9))
    assert [s["station_id"] for s in south] == ["281523"]


def test_get_metadata(client):
    meta = client.get_metadata("19145")
    by = {(d["name"], d["interval"]): d for d in meta["data_variables"]}
    assert set(by) == {
        ("SNWD", "periodic"),
        ("CORE_LENGTH", "periodic"),
        ("DENSITY", "periodic"),
        ("SWE_PILLOW", "daily"),
        ("SWE_PILLOW", "sub_daily"),
        ("SWE_COURSE", "periodic"),
    }
    daily = by["SWE_PILLOW", "daily"]
    assert (daily["begin_date"], daily["end_date"], daily["n_obs"]) == (
        "2005-05-01",
        "2023-07-02",
        8,
    )
    assert by["SWE_PILLOW", "sub_daily"]["end_date"] == "2026-10-15"
    assert by["SWE_COURSE", "periodic"]["associated_since"] == "1951-08-01"
    with pytest.raises(SNIHError, match="not a SNIH snow station"):
        client.get_metadata("111872")


# ── Data ────────────────────────────────────────────────────────────────────


def test_get_data_daily(client, stub):
    recs = client.get_data(
        "19145", begin_date="2005-05-01", end_date="2015-07-06", include_flags=True
    )
    for r in recs:
        assert RECORD_KEYS | {"flag"} == set(r)
        assert (r["variable"], r["type"], r["units"], r["interval"]) == (
            "SWE_PILLOW",
            "swe",
            "cm",
            "daily",
        )
    got = [(r["date"], r["value"], r["flag"]) for r in recs]
    assert got == [
        ("2005-05-01", 0.0, None),  # the 08:00 duplicate is dropped
        ("2015-07-01", 2.0, None),
        ("2015-07-02", 3.0, None),  # blank qualifier " "
        ("2015-07-03", None, None),  # -999
        ("2015-07-04", 3.5, "D"),
        ("2015-07-05", 4.0, "unvalidated"),
        ("2015-07-06", None, None),  # 16 m
    ]
    # fechaHasta is exclusive of the day's values: one more day is asked.
    (body,) = [c[2] for c in stub.calls if c[1] == "LeerDatosHistoricos"]
    assert body == {
        "fechaDesde": "2005-05-01",
        "fechaHasta": "2015-07-07",
        "estacion": "19145",
        "codigo": "73",
        "validados": "true",
    }
    assert stub.calls[-1][0] == "POST"


def test_get_data_inclusive_end_day(client):
    recs = client.get_data("19145", begin_date="2023-07-02", end_date="2023-07-02")
    assert [(r["date"], r["value"]) for r in recs] == [("2023-07-02", 21.0)]


def test_get_data_periodic(client):
    recs = client.get_data("19145", interval="periodic")
    assert [(r["variable"], r["date"], r["value"]) for r in recs] == [
        ("SWE_COURSE", "1951-08-24", 13.3),
        ("SWE_COURSE", "2008-09-12", 69.9),
        ("SNWD", "1951-08-24", 40.0),
        ("SNWD", "2008-09-12", 192.6),
    ]
    (dens,) = client.get_data("19145", variables="density", interval="periodic")
    assert (dens["value"], dens["units"], dens["type"]) == (34.0, "%", "density")


def test_get_data_sub_daily_reads_every_block(client):
    recs = client.get_data(
        "19145",
        interval="sub_daily",
        begin_date="2025-09-01",
        end_date="2026-10-14",
        include_flags=True,
    )
    assert [r["datetime"] for r in recs] == [
        "2025-09-26T21:00:00-03:00",
        "2025-09-27T03:00:00-03:00",
        "2026-10-13T20:00:00-03:00",
        "2026-10-14T00:00:00-03:00",
        "2026-10-14T16:00:00-03:00",  # 10-15 lies past end_date
    ]
    assert recs[3]["value"] == 56.3 and recs[3]["date"] == "2026-10-14"
    for r in recs:
        assert RECORD_KEYS | {"datetime", "flag"} == set(r)
        assert (r["interval"], r["flag"], r["units"]) == ("sub_daily", "raw", "cm")


def test_get_data_bbox_and_unknown_station(client):
    recs = client.get_data(
        bbox=(-71, -34.5, -69, -33.9),
        interval="sub_daily",
        begin_date="2026-10-15",
        end_date="2026-10-15",
    )
    assert [(r["station_id"], r["value"]) for r in recs] == [("281523", 119.7)]
    assert client.get_data("999999") == []


def test_get_data_validates_before_any_request():
    client = SNIHClient(session=_NoNetwork())
    with pytest.raises(ValueError):
        client.get_data()
    with pytest.raises(SNIHError, match="Unsupported interval"):
        client.get_data("19145", interval="hourly")
    with pytest.raises(SNIHError, match="Unknown"):
        client.get_data("19145", variables=["bogus"])
    with pytest.raises(SNIHError, match="not offered"):
        client.get_data("19145", variables="SNWD", interval="daily")
    with pytest.raises(SNIHError, match="integers"):
        client.get_data("Toscas")


def test_portal_errors_raise(stub, monkeypatch):
    client = SNIHClient(base_url=BASE, session=stub, max_retries=1, backoff=0)

    def failed(method, url, params=None, timeout=None, json=None):
        return _Resp(
            {"d": {"RespuestaOK": "False", "MsgErr": "boom", "Mediciones": None}}
        )

    monkeypatch.setattr(stub, "request", failed)
    with pytest.raises(SNIHError, match="boom"):
        client.get_data("19145")

    def http500(method, url, params=None, timeout=None, json=None):
        return _Resp({"Message": "There was an error processing the request."}, 500)

    monkeypatch.setattr(stub, "request", http500)
    with pytest.raises(SNIHError, match="500"):
        client.get_data("19145")

    def html(method, url, params=None, timeout=None, json=None):
        return _Resp("<!DOCTYPE html>")

    monkeypatch.setattr(stub, "request", html)
    with pytest.raises(SNIHError, match="Non-JSON"):
        client.get_all_stations()


# ── Live ────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_stations():
    stations = {s["station_id"]: s for s in SNIHClient().get_all_stations()}
    assert len(stations) >= 45
    toscas = stations["19145"]
    assert toscas["name"] == "Toscas" and toscas["latitude"] == pytest.approx(
        -33.16, abs=0.01
    )
    assert toscas["longitude"] == pytest.approx(-69.89, abs=0.01)
    assert sum(s["has_swe"] for s in stations.values()) >= 30
    assert any(s["status"] == "Active" for s in stations.values())
    assert {s["network"] for s in stations.values()} >= {"Red Hidrológica Nacional"}


@pytest.mark.live
def test_live_toscas_daily_and_periodic():
    client = SNIHClient()
    recs = client.get_data("19145", begin_date="2015-06-01", end_date="2015-09-30")
    assert len(recs) == 122
    assert max(r["value"] for r in recs) == pytest.approx(35.8)
    course = client.get_data(
        "19145", interval="periodic", begin_date="1951-01-01", end_date="1951-12-31"
    )
    assert {r["variable"] for r in course} == {"SWE_COURSE", "SNWD"}
    assert min(r["value"] for r in course if r["variable"] == "SWE_COURSE") == 13.3


@pytest.mark.live
def test_live_sub_daily():
    recs = SNIHClient().get_data(
        "19145", interval="sub_daily", begin_date="2026-08-01", end_date="2026-08-05"
    )
    assert len(recs) >= 20
    assert all(r["datetime"].endswith("-03:00") for r in recs)
    assert {r["date"] for r in recs} <= {f"2026-08-0{d}" for d in range(1, 6)}
