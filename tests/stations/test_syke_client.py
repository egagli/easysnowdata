"""
Tests for SYKEClient.

Offline tests feed synthetic OData v3 JSON payloads (same layout as the real
Hydrologiarajapinta responses, including ``odata.nextLink`` paging) through
a stub session.  ``live`` tests hit the real Syke API (no credentials).
"""

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.syke import SYKEClient, SYKEError
from easysnowdata.stations.clients.syke.syke_client import (
    AREAL_VARIABLES,
    DATA_FLAGS,
    VARIABLES,
    _build_filter,
    _convert,
    _dms_to_deg,
    _parse_site,
    _resolve_variables,
    _site_coords,
    _tm35fin_to_wgs84,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

BASE = "https://example.test/odata"

# Real Paikka rows (trimmed), as served by the API in Oct 2026.
SITE_ACTIVE = {
    "Paikka_Id": 1479,
    "Suure_Id": 8,
    "KuntaNimi": "Tohmajärvi",
    "Nro": "1020101",
    "Nimi": "Tohmajärvi,Kemie ",
    "KoordLat": "621405",
    "KoordLong": "302105",
    "KoordTarkkuus": 0,
    "Omistaja_Id": 16,
    "Tila_Id": 1,
    "Laitteisto_Id": 14,
    "VesalNimi": "Lahdenjoen va",
    "PaaVesalNimi": "Tohmajoki",
    "KoordErTmPohj": 6904874,
    "KoordErTmIta": 673976,
}
SITE_CLOSED = {
    **SITE_ACTIVE,
    "Paikka_Id": 1483,
    "Nro": "0400101",
    "Nimi": "Leppävirta, Paukarlahti",
    "Tila_Id": 2,
    "KoordErTmPohj": None,
    "KoordErTmIta": None,
    "KoordLat": "693000",
    "KoordLong": "270000",
}
AREA = {
    **SITE_ACTIVE,
    "Paikka_Id": 1,
    "Suure_Id": 7,
    "Nimi": "Jänisjoki,Ruskeakoski ",
    "Laitteisto_Id": 12,
    "KoordErTmPohj": 6926501,
    "KoordErTmIta": 677699,
}


def _odata(table, rows, next_link=None, count=None):
    d = {"odata.metadata": f"{BASE}/$metadata#{table}", "value": rows}
    if next_link:
        d["odata.nextLink"] = next_link
    if count is not None:
        d["odata.count"] = str(count)
    return d


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = str(payload)

    def json(self):
        return self._payload


class _StubSession:
    """Routes requests by table name and filter content."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, params=None, timeout=None):
        self.calls.append((url, params))
        flt = (params or {}).get("$filter", "")
        table = url.split("/odata/")[1].split("?")[0]
        if table == "Omistaja":
            return _Resp(
                _odata(table, [{"Omistaja_Id": 16, "Nimi": "Ympäristöhallinto"}])
            )
        if table == "Paikka":
            if "Suure_Id eq 8" in flt:
                return _Resp(_odata(table, [SITE_ACTIVE, SITE_CLOSED]))
            return _Resp(_odata(table, [AREA]))
        if table == "LumiPiste":
            if "skip=1" in url:
                return _Resp(
                    _odata(
                        table,
                        [
                            {
                                "Paikka_Id": 1479,
                                "Aika": "2026-03-21T00:00:00",
                                "Arvo": "14.55",
                                "Lippu_Id": 117,
                            }
                        ],
                    )
                )
            if params and params.get("$top") == "1":
                order = params["$orderby"]
                day = "1960-02-15" if "asc" in order else "2026-03-21"
                return _Resp(
                    _odata(
                        table,
                        [
                            {
                                "Paikka_Id": 1479,
                                "Aika": f"{day}T00:00:00",
                                "Arvo": "1",
                                "Lippu_Id": 96,
                            }
                        ],
                        count=161 if "asc" in order else None,
                    )
                )
            return _Resp(
                _odata(
                    table,
                    [
                        {
                            "Paikka_Id": 1479,
                            "Aika": "2026-02-20T00:00:00",
                            "Arvo": "26.04",
                            "Lippu_Id": 96,
                        }
                    ],
                    next_link=f"{BASE}/LumiPiste?$filter=x&$skip=1",
                )
            )
        if table == "LumiAlue":
            return _Resp(
                _odata(
                    table,
                    [
                        {
                            "Paikka_Id": 1,
                            "Aika": "2026-03-01T00:00:00",
                            "Arvo": "131.00",
                        },
                        {"Paikka_Id": 1, "Aika": "2026-03-16T00:00:00", "Arvo": "0.00"},
                    ],
                )
            )
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def client():
    return SYKEClient(base_url=BASE, session=_StubSession(), max_retries=1)


# ── registries ───────────────────────────────────────────────────────────────


def test_registries_use_shared_vocab():
    for reg in (VARIABLES, AREAL_VARIABLES):
        for info in reg.values():
            assert info["type"] in TYPES
            for k in (
                "name",
                "units",
                "output_units",
                "description",
                "notes",
                "source",
            ):
                assert k in info
    assert "96" in DATA_FLAGS


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES, AREAL_VARIABLES):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_resolve_variables():
    assert _resolve_variables(None) == ["S"]
    assert _resolve_variables([]) == ["S"]  # empty = default
    assert _resolve_variables("swe") == ["S"]
    assert _resolve_variables(["S", "s"]) == ["S"]
    with pytest.raises(SYKEError):
        _resolve_variables("snwd")
    with pytest.raises(SYKEError):
        _resolve_variables("SA")


# ── parsers ──────────────────────────────────────────────────────────────────


def test_convert_mm_to_cm_and_clamp():
    assert _convert("26.04") == pytest.approx(2.604)
    assert _convert("0.00") == 0.0
    assert _convert("-5") is None
    assert _convert("20000") is None
    assert _convert(None) is None


def test_coordinates_tm35fin_and_fallbacks():
    lat, lon = _tm35fin_to_wgs84(6904874, 673976)
    assert lat == pytest.approx(62.2351, abs=1e-3)
    assert lon == pytest.approx(30.3485, abs=1e-3)
    # KKJ uniform coordinates (EPSG:2393) of the same site agree to metres.
    ykj = _site_coords({"KoordYkPohj": 6907768, "KoordYkIta": 3674217})
    assert ykj == pytest.approx((lat, lon), abs=1e-4)
    # DDMMSS text is on the KKJ datum; transformed, it lands within ~300 m.
    dms = _site_coords({"KoordLat": "621405", "KoordLong": "302105"})
    assert dms == pytest.approx((lat, lon), abs=5e-3)
    assert _dms_to_deg("621405") == pytest.approx(62.234722, abs=1e-6)
    assert _tm35fin_to_wgs84(None, None) == (None, None)
    assert _site_coords({}) == (None, None)
    assert _dms_to_deg("619999") is None
    assert _dms_to_deg("") is None


def test_parse_site_status_and_fallback_coords():
    a = _parse_site(SITE_ACTIVE, {16: "Ympäristöhallinto"})
    assert a["station_id"] == "1479"
    assert a["name"] == "Tohmajärvi,Kemie"
    assert a["status"] == "Active" and a["status_native"] == "Jatkuva"
    assert a["elevation_m"] is None
    assert a["operator"] == "Ympäristöhallinto"
    c = _parse_site(SITE_CLOSED)
    assert c["status"] == "Inactive"
    # no projected coords in this synthetic row -> KKJ-datum DDMMSS fallback
    assert c["latitude"] == pytest.approx(69.5, abs=0.01)
    assert c["longitude"] == pytest.approx(27.0, abs=0.01)


def test_build_filter():
    f = _build_filter(["1", "2"], "2020-01-01", "2020-12-31")
    assert f == (
        "(Paikka_Id eq 1 or Paikka_Id eq 2) and Aika ge datetime'2020-01-01T00:00:00'"
        " and Aika le datetime'2020-12-31T00:00:00'"
    )
    assert _build_filter(None, None, None) is None


# ── client ───────────────────────────────────────────────────────────────────


def test_get_all_stations(client):
    stas = client.get_all_stations()
    assert {s["station_id"] for s in stas} == {"1479", "1483"}
    for s in stas:
        for k in (
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "status",
        ):
            assert k in s
    assert [s["station_id"] for s in client.get_all_stations(active_only=True)] == [
        "1479"
    ]
    assert [
        s["station_id"] for s in client.get_all_stations(bbox=(30, 62, 31, 63))
    ] == ["1479"]


def test_get_data_paging_flags_and_schema(client):
    recs = client.get_data("1479", begin_date="2026-01-01", include_flags=True)
    assert len(recs) == 2  # followed odata.nextLink
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["interval"] in INTERVALS and r["interval"] == "periodic"
        assert r["units"] == "cm" and r["type"] == "swe"
    assert recs[0]["value"] == pytest.approx(2.604)
    assert recs[0]["flag"] == "96"
    # Lippu 117 = snow-model value: nulled by default
    assert recs[1]["value"] is None and recs[1]["flag"] == "117"
    flt = client._session.calls[-2][1]["$filter"]
    assert "Paikka_Id eq 1479" in flt and "datetime'2026-01-01T00:00:00'" in flt


def test_get_data_keep_modelled():
    c = SYKEClient(base_url=BASE, session=_StubSession(), keep_modelled=True)
    recs = c.get_data(["1479"])
    assert recs[1]["value"] == pytest.approx(1.455)
    assert "flag" not in recs[0]


def test_get_data_errors(client):
    with pytest.raises(ValueError):
        client.get_data()
    with pytest.raises(SYKEError):
        client.get_data("1479", interval="daily")
    with pytest.raises(SYKEError):
        client.get_data("1479", variables="density")
    with pytest.raises(SYKEError):
        client.get_data("abc")


def test_get_data_chunks_ids_under_node_limit(client):
    client.get_data([str(i) for i in range(1, 21)])
    first_pages = [
        p for u, p in client._session.calls if u.endswith("/LumiPiste") and p
    ]
    assert len(first_pages) == 2  # 20 ids -> chunks of 15 + 5
    assert first_pages[0]["$filter"].count("Paikka_Id eq") == 15


def test_get_data_bbox(client):
    recs = client.get_data(bbox=(30, 62, 31, 63))
    assert {r["station_id"] for r in recs} == {"1479"}


def test_get_metadata(client):
    md = client.get_metadata("1479")
    assert md["kind"] == "snow_course"
    (dv,) = md["data_variables"]
    assert dv["name"] == "S" and dv["interval"] == "periodic" and dv["units"] == "cm"
    assert dv["begin_date"] == "1960-02-15" and dv["end_date"] == "2026-03-21"
    assert dv["n_obs"] == 161
    with pytest.raises(SYKEError):
        client.get_metadata("999999")


def test_get_areal_swe(client):
    units = client.get_areal_units()
    assert units[0]["station_id"] == "1"
    recs = client.get_areal_swe("1", begin_date="2026-03-01")
    assert [r["value"] for r in recs] == [pytest.approx(13.1), 0.0]
    assert all(r["interval"] == "semi_monthly" and r["variable"] == "SA" for r in recs)


# ── live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_stations_and_data():
    c = SYKEClient()
    stas = c.get_all_stations()
    assert len(stas) > 300
    active = [s for s in stas if s["status"] == "Active"]
    assert len(active) > 100
    tohma = next(s for s in stas if s["station_id"] == "1479")
    assert 62 < tohma["latitude"] < 62.5 and 30 < tohma["longitude"] < 31
    recs = c.get_data("1479", begin_date="2023-11-01", end_date="2024-05-31")
    assert len(recs) >= 5
    assert all(r["value"] is None or 0 <= r["value"] < 100 for r in recs)


@pytest.mark.live
def test_live_paging_and_metadata():
    c = SYKEClient()
    # >500 rows → exercises odata.nextLink paging.
    ids = [s["station_id"] for s in c.get_all_stations(active_only=True)][:40]
    recs = c.get_data(ids, begin_date="2015-01-01", end_date="2024-12-31")
    assert len(recs) > 500
    assert len({(r["station_id"], r["date"]) for r in recs}) == len(recs)
    md = c.get_metadata("1479")
    assert md["data_variables"][0]["n_obs"] > 100


@pytest.mark.live
def test_live_areal_swe():
    c = SYKEClient()
    recs = c.get_areal_swe("1", begin_date="2024-01-01", end_date="2024-04-30")
    assert len(recs) >= 7
    assert {r["date"][8:] for r in recs} <= {"01", "16"}
