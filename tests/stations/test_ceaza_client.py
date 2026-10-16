"""
Tests for CEAZAClient.

Offline tests replay trimmed copies of real CEAZA-Met responses (the CSV
station and sensor lists with their ``#`` metadata lines and trailing
commas, and ``GetSerieSensor`` JSON, captured in October 2026) through a
stub session.  ``live`` tests hit the real web service (no credentials).
"""

import json
from datetime import date

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.ceaza import CEAZAClient, CEAZAError
from easysnowdata.stations.clients.ceaza import ceaza_client as cz
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

_HEAD = (
    "#Datos provistos por CEAZA a traves del sistema CEAZA-Met, generados el:"
    "2026-10-15 19:14:35\n"
    "#Para consultas, contactar con Cristian Orrego al mail cristian.orrego@ceaza.cl.\n"
    "#Revisar condiciones de uso en www.ceazamet.cl\n"
    "#Params , start_date: 2026-10-08 00:00:00, end_date: 2026-10-15 19:14:37, "
    "gmt: gmt-4\n"
)

STATIONS_CSV = _HEAD + (
    "#e_cod,e_nombre,e_lat,e_lon,e_altitud,e_primera_lectura,e_ultima_lectura,"
    "e_cod_provincia,e_owner,\n"
    "HDC,Guandacol,-30.543625,-70.33290088,2983,2016-10-29 20:00:00,"
    "2026-10-15 22:00:00,043,ceaza,\n"
    "8,Pisco Elqui,-30.129028,-70.494712,1240,2004-02-01 00:02:00,"
    "2026-10-15 18:50:00,041,ceaza,\n"
    "LCP,La Coipa,-30.96609,-70.329083,3381,2025-04-02 21:00:00,"
    "2026-10-15 17:00:00,,,\n"
    "TASC,Tascadero,-31.262951,-70.539993,3427,2012-11-27 06:00:00,"
    "2026-10-15 10:00:00,043,ceaza,\n"
    "TPF,El Tapado,-30.1583,-69.908179,4306,2013-11-21 20:00:00,"
    "2025-11-26 15:00:00,041,ceaza,\n"
    "HUAL,Hualtatas,-31.707322,-70.525002,3632,2014-04-28 00:00:00,"
    "2019-10-23 11:00:00,042,ceaza,\n"
)

SENSORS_CSV = _HEAD + (
    "#e_cod,s_cod,tf_nombre,um_notacion,s_altura,s_primera_lectura,"
    "s_ultima_lectura,tm_cod,\n"
    "8,59,Temperatura del Aire,°C,1.5,2004-02-01 00:02:00,2026-10-15 18:50:00,ta_c,\n"
    "8,55,Precipitación,mm,1.5,2004-01-31 23:02:00,2026-10-15 18:50:00,pp_mm,\n"
    "TPF,TPFTA,Temperatura del Aire,°C,3,2013-11-21 20:00:00,2023-10-25 13:00:00,ta_c,\n"
    "TPF,TPFPP1,Precipitación Acumulada,mm,1.5,2013-11-21 20:00:00,"
    "2025-10-07 21:00:00,ppa_mm,\n"
    "TPF,TPFSWE_TL,Agua equivalente,mm,3.1,2016-04-12 22:00:00,"
    "2025-11-26 15:00:00,ae_mm,\n"
    "TPF,TPFSWE_K,Agua equivalente,mm,3.1,2016-04-12 22:00:00,"
    "2025-11-26 15:00:00,ae_mm,\n"
    "TPF,TPFSNOWD,Altura de Nieve,cm,0,2013-11-21 20:00:00,2025-10-08 01:00:00,ani_cm,\n"
    "HDC,HDCTA,Temperatura del Aire,°C,2,2016-10-29 20:00:00,2026-10-15 21:00:00,ta_c,\n"
    "HDC,HDCVV,Velocidad de Viento,m/s,3,2016-10-29 20:00:00,2026-10-15 21:00:00,vv_ms,\n"
    "HDC,HDCAN,Altura de Nieve,cm,2.85,2016-10-29 20:00:00,2026-10-15 21:00:00,ani_cm,\n"
    "HDC,HDCPN,Peso de la Nieve,kg/m²,0,2016-10-29 20:00:00,2026-10-15 21:00:00,pnkg,\n"
    "LCP,LCPAN,Altura de Nieve,cm,4.5,2025-03-13 14:00:00,2026-10-15 17:00:00,ani_cm,\n"
    "TASC,TASCTA,Temperatura del Aire,°C,1.5,2012-11-27 06:00:00,"
    "2026-10-15 10:00:00,ta_c,\n"
    "TASC,TASCPP,Precipitación,mm,1.5,2012-11-27 06:00:00,2026-10-15 10:00:00,pp_mm,\n"
    "TASC,TASCPN,Agua equivalente,mm,0,2012-11-27 06:00:00,2026-10-15 10:00:00,ae_mm,\n"
    "TASC,TASCAN,Altura de Nieve,cm,2.65,2020-02-19 15:00:00,"
    "2026-10-15 10:00:00,ani_cm,\n"
    "HUAL,HUALPN,Agua equivalente,cm,0,2014-04-28 00:00:00,2019-10-23 11:00:00,ae_cm,\n"
    "HUAL,HUALAN,Altura de Nieve,cm,5,2014-04-30 12:00:00,2019-10-23 11:00:00,ani_cm,\n"
    "MET2BG,MET2BGTA,Temperatura del Aire,°C,2,,,ta_c,\n"
)


def _row(fecha, lo, mean, hi, pc, first=None, last=None):
    return {
        "fecha": fecha,
        "min": lo,
        "prom": mean,
        "max": hi,
        "data_pc": pc,
        "fecha_min": first,
        "fecha_max": last,
    }


def _series(s_cod, rows, e_cod="TASC", name="Altura de Nieve", units="cm"):
    return {
        "e_cod": e_cod,
        "e_nombre": "Tascadero",
        "s_cod": s_cod,
        "s_altura": "2.65",
        "tf_nombre": name,
        "um_notacion": units,
        "fecha_inicio": "2025-06-12 00:00:00",
        "fecha_fin": "2025-06-15 23:59:59",
        "intervalo": None,
        "file_date": "2026-10-15 19:21:26",
        "serie": rows,
    }


# Real rows (Tascadero, June 2025), plus an empty interval and a negative
# bare-ground depth.
SERIES = {
    ("TASCAN", "dia"): _series(
        "TASCAN",
        [
            _row("2025-06-12 00:00:00", 0.2, 4.575, 13.5, 83, "2025-06-12 00:00:00"),
            _row("2025-06-13 00:00:00", 11.1, 15.72, 17.4, 41, "2025-06-13 14:00:00"),
            _row("2025-06-14 00:00:00", None, None, None, 0),
            _row("2025-06-15 00:00:00", -0.4, -0.025, 0.3, 100),
        ],
    ),
    ("TASCPN", "dia"): _series(
        "TASCPN",
        [_row("2025-06-12 00:00:00", 15.96, 17.186, 17.95, 100)],
        name="Agua equivalente",
        units="mm",
    ),
    ("TASCPP", "dia"): _series(
        "TASCPP",
        [_row("2025-06-12 00:00:00", 28.2, 28.2, 28.2, 83)],
        name="Precipitación",
        units="mm",
    ),
    ("TASCPP", "hora"): _series(
        "TASCPP",
        [
            _row("2025-06-12 06:00:00", 1.3, 1.3, 1.3, 100),
            _row("2025-06-12 07:00:00", 2.6, 2.6, 2.6, 100),
        ],
        name="Precipitación",
        units="mm",
    ),
    ("TASCTA", "hora"): _series(
        "TASCTA",
        [_row("2025-06-12 00:00:00", -2.309, -1.675, -0.953, 100)],
        name="Temperatura del Aire",
        units="°C",
    ),
    ("HDCPN", "dia"): _series(
        "HDCPN",
        [_row("2026-08-14 00:00:00", 318.0, 319.6, 321.0, 100)],
        e_cod="HDC",
        name="Peso de la Nieve",
        units="kg/m²",
    ),
    ("HUALPN", "dia"): _series(
        "HUALPN",
        [_row("2016-07-01 00:00:00", 14.47, 14.632, 15.12, 100)],
        e_cod="HUAL",
        name="Agua equivalente",
        units="cm",
    ),
}

# What the service sends for an unsupported ``interv``: HTML, then JSON
# without ``serie``.
ERROR_TEXT = (
    "\n<p>ERROR en lib_datos.GetDatosCSVquery, la query viene vacia.</p>"
    '{"e_cod":"TASC","e_nombre":"Tascadero","s_cod":"TASCAN"}'
)
# ... and for an unknown sensor.
UNKNOWN_SENSOR = _series(
    "NOPE", [_row("2025-07-01 00:00:00", None, None, None, 0)], e_cod=None
)


class _Resp:
    def __init__(self, text):
        self.text = text
        self.ok = True
        self.status_code = 200
        self.encoding = "utf-8"


class _StubSession:
    """Routes requests by ``fn`` and ``s_cod``; records every call."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        fn = params.get("fn")
        if fn == "GetListaEstaciones":
            return _Resp(STATIONS_CSV)
        if fn == "GetListaSensores":
            return _Resp(SENSORS_CSV)
        key = (params.get("s_cod"), params.get("interv"))
        if key in SERIES:
            return _Resp(json.dumps(SERIES[key], ensure_ascii=False))
        if params.get("s_cod") == "NOPE":
            return _Resp(json.dumps(UNKNOWN_SENSOR))
        return _Resp(json.dumps(_series(params.get("s_cod"), [])))

    def series_calls(self):
        return [c for c in self.calls if c.get("fn") == "GetSerieSensor"]


class _Today(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 15)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(cz, "date", _Today)
    return _StubSession()


@pytest.fixture
def client(session):
    return CEAZAClient(session=session)


# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_follows_the_contract():
    assert set(cz.VARIABLES) == {"ani_cm", "ae_mm", "ae_cm", "pnkg", "ta_c", "pp_mm"}
    for code, info in cz.VARIABLES.items():
        assert info["type"] in TYPES, code
        assert info["output_units"] == canonical_units(info["type"]), code
        assert {"name", "units", "description", "notes", "source"} <= set(info)
    assert set(cz.DATA_FLAGS) == {"complete", "partial"}
    assert {"daily", "hourly"} <= INTERVALS
    assert set(cz.SNOW_VARIABLES) <= set(cz.VARIABLES)


def test_resolve_variables():
    assert cz._resolve_variables(None) == ["ani_cm", "ae_mm", "ae_cm", "pnkg"]
    assert cz._resolve_variables("swe") == ["ae_mm", "ae_cm", "pnkg"]
    assert cz._resolve_variables(["TA_C", "temp", "precip"]) == ["ta_c", "pp_mm"]
    for bad in ("ppa_mm", "an_mm", "bogus"):
        with pytest.raises(CEAZAError, match="Unknown"):
            cz._resolve_variables(bad)


def test_convert_scales_and_bounds():
    assert cz._convert("ae_mm", 426.2) == pytest.approx(42.62)
    assert cz._convert("pnkg", "319.6") == pytest.approx(31.96)  # kg/m² = mm
    assert cz._convert("ae_cm", 14.632) == pytest.approx(14.632)
    assert cz._convert("ani_cm", -0.025) is None  # bare-ground offset
    assert cz._convert("ani_cm", 4021.4) is None  # > 15 m spike
    assert cz._convert("ta_c", -13.1) == -13.1
    assert cz._convert("ta_c", -82.8) is None
    assert cz._convert("pp_mm", 28.2) == 28.2
    assert cz._convert("pp_mm", None) is None
    assert cz._convert("ani_cm", "") is None


# ── Parsing ─────────────────────────────────────────────────────────────────


def test_parse_csv_uses_the_last_comment_line_as_header():
    rows = cz.parse_csv(STATIONS_CSV)
    assert len(rows) == 6
    assert list(rows[0]) == list(cz.STATION_COLUMNS)
    assert rows[2]["e_owner"] == "" and rows[2]["e_cod"] == "LCP"
    with pytest.raises(CEAZAError, match="no header"):
        cz.parse_csv("HDC,Guandacol\n")


def test_parse_series_errors():
    payload = cz.parse_series(json.dumps(SERIES[("TASCAN", "dia")]))
    assert len(payload["serie"]) == 4
    with pytest.raises(CEAZAError, match="ERROR en lib_datos"):
        cz.parse_series(ERROR_TEXT)
    with pytest.raises(CEAZAError, match="Unknown CEAZA-Met sensor 'NOPE'"):
        cz.parse_series(json.dumps(UNKNOWN_SENSOR))
    with pytest.raises(CEAZAError, match="no 'serie'"):
        cz.parse_series('{"e_cod":"TASC","s_cod":"TASCAN"}')
    with pytest.raises(CEAZAError, match="non-JSON"):
        cz.parse_series("{not json")


def test_windows():
    assert cz._windows("2025-01-01", "2025-01-01", 365) == [
        ("2025-01-01", "2025-01-01")
    ]
    wins = cz._windows("2023-01-01", "2025-06-30", 365)
    assert wins[0] == ("2023-01-01", "2023-12-31")
    assert wins[-1][1] == "2025-06-30" and len(wins) == 3


# ── Stations ────────────────────────────────────────────────────────────────


def test_constructor_is_network_free(session):
    CEAZAClient(session=session)
    assert session.calls == []


def test_get_all_stations(client, session):
    stations = {s["station_id"]: s for s in client.get_all_stations()}
    # Pisco Elqui has no snow sensor.
    assert list(stations) == ["HDC", "LCP", "TASC", "TPF", "HUAL"]
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
    hdc = stations["HDC"]
    assert (hdc["latitude"], hdc["longitude"], hdc["elevation_m"]) == (
        -30.543625,
        -70.33290088,
        2983.0,
    )
    assert hdc["operator"].startswith("Centro de Estudios Avanzados")
    assert hdc["snow_variables"] == ["ani_cm", "pnkg"] and hdc["has_swe"]
    assert hdc["served_sensors"] == ["HDCTA", "HDCAN", "HDCPN"]  # not HDCVV
    assert hdc["station_url"].endswith("e_cod=HDC&p_cod=ceazamet")
    assert stations["LCP"]["operator"] is None and not stations["LCP"]["has_swe"]
    assert stations["TPF"]["snow_sensors"] == ["TPFSWE_TL", "TPFSWE_K", "TPFSNOWD"]
    assert stations["TASC"]["has_precip"] and stations["TASC"]["has_temp"]
    # Active = latest reading at most 90 days before 2026-10-15.
    assert {s: v["status"] for s, v in stations.items()} == {
        "HDC": "Active",
        "LCP": "Active",
        "TASC": "Active",
        "TPF": "Inactive",
        "HUAL": "Inactive",
    }
    # Two requests, both with the documented explicit column list.
    assert [c["fn"] for c in session.calls] == [
        "GetListaEstaciones",
        "GetListaSensores",
    ]
    assert session.calls[0]["c8"] == "e_owner" and session.calls[1]["c7"] == "tm_cod"
    assert all(c["p_cod"] == "ceazamet" and "user" not in c for c in session.calls)
    client.get_all_stations()
    assert len(session.calls) == 2  # cached


def test_get_all_stations_filters(client):
    assert len(client.get_all_stations(snow_only=False)) == 6
    assert {s["station_id"] for s in client.get_all_stations(active_only=True)} == {
        "HDC",
        "LCP",
        "TASC",
    }
    south = client.get_all_stations(bbox=(-71, -32, -70, -31.5))
    assert [s["station_id"] for s in south] == ["HUAL"]


def test_get_metadata(client):
    meta = client.get_metadata("TPF")
    names = [(d["name"], d["sensor"]) for d in meta["data_variables"]]
    # TPFPP1 (ppa_mm, a cumulative gauge level) is not served.
    assert names == [
        ("ta_c", "TPFTA"),
        ("ae_mm", "TPFSWE_TL"),
        ("ae_mm", "TPFSWE_K"),
        ("ani_cm", "TPFSNOWD"),
    ]
    swe = meta["data_variables"][1]
    assert (swe["units"], swe["output_units"], swe["height_m"]) == ("mm", "cm", 3.1)
    assert swe["end_date"] == "2025-11-26" and swe["intervals"] == ["daily", "hourly"]
    assert len(meta["sensors"]) == 5
    assert client.get_metadata("8")["data_variables"][1]["name"] == "pp_mm"
    with pytest.raises(CEAZAError, match="not in"):
        client.get_metadata("NOPE")


# ── Data ────────────────────────────────────────────────────────────────────


def test_get_data_daily_records(client, session):
    recs = client.get_data(
        "TASC", begin_date="2025-06-12", end_date="2025-06-15", include_flags=True
    )
    for r in recs:
        assert RECORD_KEYS <= set(r) and "datetime" not in r
        assert r["interval"] == "daily" and r["units"] == "cm"
    depth = [r for r in recs if r["sensor"] == "TASCAN"]
    # The empty 2025-06-14 row is dropped; the negative depth is nulled.
    assert [(r["date"], r["value"], r["flag"]) for r in depth] == [
        ("2025-06-12", 4.575, "partial"),
        ("2025-06-13", 15.72, "partial"),
        ("2025-06-15", None, "complete"),
    ]
    assert (depth[0]["value_min"], depth[0]["value_max"], depth[0]["data_pct"]) == (
        0.2,
        13.5,
        83,
    )
    (swe,) = [r for r in recs if r["sensor"] == "TASCPN"]
    assert (swe["variable"], swe["type"], swe["value"]) == ("ae_mm", "swe", 1.719)
    calls = session.series_calls()
    assert {c["s_cod"] for c in calls} == {"TASCAN", "TASCPN"}
    assert all(
        (c["fecha_inicio"], c["fecha_fin"], c["interv"], c["tipo_resp"])
        == ("2025-06-12", "2025-06-15", "dia", "json")
        for c in calls
    )


def test_get_data_snow_weight_and_cm_swe(client):
    (hdc,) = client.get_data(
        "HDC", variables="swe", begin_date="2026-08-14", end_date="2026-08-14"
    )
    assert (hdc["variable"], hdc["value"], hdc["units"]) == ("pnkg", 31.96, "cm")
    (hual,) = client.get_data(
        "HUAL", variables="ae_cm", begin_date="2016-07-01", end_date="2016-07-01"
    )
    assert hual["value"] == pytest.approx(14.632)


def test_get_data_precip_is_a_total_without_min_max(client):
    (rec,) = client.get_data(
        "TASC", variables="precip", begin_date="2025-06-12", end_date="2025-06-12"
    )
    assert (rec["value"], rec["units"], rec["value_min"], rec["value_max"]) == (
        28.2,
        "mm",
        None,
        None,
    )


def test_get_data_hourly_has_datetime(client, session):
    recs = client.get_data(
        "TASC",
        variables=["temp", "precip"],
        begin_date="2025-06-12",
        end_date="2025-06-12",
        interval="hourly",
    )
    assert [r["datetime"] for r in recs] == [
        "2025-06-12T00:00:00-04:00",
        "2025-06-12T06:00:00-04:00",
        "2025-06-12T07:00:00-04:00",
    ]
    assert all(r["interval"] == "hourly" for r in recs)
    assert {c["interv"] for c in session.series_calls()} == {"hora"}


def test_get_data_clips_to_sensor_dates_and_splits_windows(client, session):
    # No dates: each sensor from its first reading to its last (HUAL ended
    # 2019), in ≤ 10-year windows.
    client.get_data("HUAL", variables="snwd")
    (call,) = session.series_calls()
    assert (call["s_cod"], call["fecha_inicio"], call["fecha_fin"]) == (
        "HUALAN",
        "2014-04-30",
        "2019-10-23",
    )
    session.calls.clear()
    client.get_data("TASC", variables="ae_mm", end_date="2025-12-31")
    spans = [(c["fecha_inicio"], c["fecha_fin"]) for c in session.series_calls()]
    assert spans == [("2012-11-27", "2022-11-24"), ("2022-11-25", "2025-12-31")]
    session.calls.clear()
    # Outside a sensor's life: no request at all.
    assert client.get_data("HUAL", begin_date="2024-01-01") == []
    assert session.series_calls() == []


def test_get_data_sensor_filter_and_bbox(client, session):
    client.get_data("TPF", variables="swe", sensors="TPFSWE_K", begin_date="2025-07-01")
    assert [c["s_cod"] for c in session.series_calls()] == ["TPFSWE_K"]
    with pytest.raises(CEAZAError, match="not at station"):
        client.get_data("TPF", sensors="TASCAN")
    session.calls.clear()
    client.get_data(bbox=(-71, -32, -70, -31.5), begin_date="2016-07-01")
    assert {c["s_cod"] for c in session.series_calls()} == {"HUALPN", "HUALAN"}


def test_get_data_errors_raise_before_any_request(session):
    client = CEAZAClient(session=session)
    with pytest.raises(ValueError):
        client.get_data()
    with pytest.raises(CEAZAError, match="Unsupported interval"):
        client.get_data("TASC", interval="monthly")
    with pytest.raises(CEAZAError, match="Unknown"):
        client.get_data("TASC", variables=["bogus"])
    assert session.calls == []
    with pytest.raises(CEAZAError, match="not in"):
        client.get_data("NOPE")


def test_service_errors_raise(client):
    with pytest.raises(CEAZAError, match="Unknown CEAZA-Met sensor"):
        client.get_series("NOPE", "2025-07-01", "2025-07-01")
    with pytest.raises(CEAZAError, match="Unsupported interval"):
        client.get_series("TASCAN", "2025-07-01", "2025-07-01", interval="weekly")


def test_user_is_sent_when_given(session):
    CEAZAClient(user="someone@example.org", session=session).get_all_stations()
    assert all(c["user"] == "someone@example.org" for c in session.calls)


# ── Live ────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_stations_and_metadata():
    client = CEAZAClient()
    stations = {s["station_id"]: s for s in client.get_all_stations()}
    assert len(stations) >= 15
    assert {"TPF", "TASC", "HDC", "LLL"} <= set(stations)
    assert all(s["has_snwd"] or s["has_swe"] for s in stations.values())
    assert len(client.get_all_stations(snow_only=False)) > 80
    meta = client.get_metadata("TPF")
    sensors = {d["sensor"]: d for d in meta["data_variables"]}
    assert {"TPFSWE_K", "TPFSWE_TL", "TPFANSHM31"} <= set(sensors)
    assert sensors["TPFSWE_K"]["units"] == "mm"


@pytest.mark.live
def test_live_daily_and_hourly():
    client = CEAZAClient()
    recs = client.get_data(
        "TASC", begin_date="2025-08-01", end_date="2025-08-31", include_flags=True
    )
    depth = [r for r in recs if r["sensor"] == "TASCAN"]
    assert len(depth) == 31 and max(r["value"] for r in depth) > 30
    assert {r["flag"] for r in depth} <= {"complete", "partial"}
    # El Tapado SWE: winter 2022 peaked above 40 cm.
    swe = client.get_data(
        "TPF",
        variables="swe",
        sensors="TPFSWE_K",
        begin_date="2022-07-01",
        end_date="2022-08-31",
    )
    assert len(swe) > 50 and max(r["value"] or 0 for r in swe) > 40
    hourly = client.get_data(
        "TASC",
        variables="temp",
        begin_date="2025-06-12",
        end_date="2025-06-12",
        interval="hourly",
    )
    # 20 of 24 hours (the station's daily data_pc that day is 83 %).
    assert len(hourly) == 20 and hourly[0]["datetime"].endswith("-04:00")
