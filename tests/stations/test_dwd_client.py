"""
Tests for DWDClient.

The offline tests feed synthetic payloads in the real DWD CDC layouts
(fixed-width latin-1 station lists, Apache directory indexes, per-station
zips holding a semicolon ``produkt_*.txt``) through a stub session, so
they touch no socket.  The ``live`` tests read the real opendata.dwd.de
files (no credentials needed).
"""

import io
import zipfile

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.dwd import DWDClient, DWDError
from easysnowdata.stations.clients.dwd.dwd_client import (
    DATA_FLAGS,
    VARIABLES,
    _convert,
    _merge,
    _pad_id,
    _parse_listing,
    _parse_produkt,
    _parse_station_list,
    _resolve_variables,
)
from tests.stations.conftest import RECORD_KEYS

# Real-layout station lists.  The water_equiv list pads the coordinates
# differently from the more_precip one; both must parse.
_WA_HEADER = (
    "Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge "
    "Stationsname Bundesland Abgabe\r\n"
    "----------- --------- --------- ------------- --------- --------- "
    "----------------------------------------- ---------- ------\r\n"
)
WA_STATIONS = (
    _WA_HEADER
    + "00020 20040128 20260127            432     48.9219    9.9129 "
    + "Abtsgmünd-Untergröningen".ljust(41)
    + "Baden-Württemberg".ljust(41)
    + "Frei   \r\n"
    + "00001 19370130 19580125            478     47.8413    8.8493 "
    + "Aach".ljust(41)
    + "Baden-Württemberg".ljust(41)
    + "Frei   \r\n"
)
RR_STATIONS = (
    _WA_HEADER
    + "00020 19410101 20261013            432   48.9219      9.9129 "
    + "Abtsgmünd-Untergröningen".ljust(41)
    + "Baden-Württemberg".ljust(41)
    + "Frei   \r\n"
    + "05792 19010101 20261013           2956   47.4210     10.9848 "
    + "Zugspitze".ljust(41)
    + "Bayern".ljust(41)
    + "       \r\n"
)


def _listing(names):
    rows = "".join(
        f'<a href="{n}">{n}</a>    14-Oct-2026 08:40:09   10729\r\n' for n in names
    )
    return f"<html><body><pre>{rows}</pre></body></html>"


def _zip(member, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Metadaten_Geographie_00020.txt", "x")
        zf.writestr(member, text.encode("latin-1"))
    return buf.getvalue()


WA_HIST = (
    "STATIONS_ID;MESS_DATUM;QN_6;ASH_6;SH_TAG;WASH_6;WAAS_6;eor\r\n"
    "         20;20251230;    9;  12;  12;   20.5;   20.5;eor\r\n"
    "         20;20251231;    9;-999;   8;   -999;-999;eor\r\n"
)
WA_RECENT = (
    "STATIONS_ID;MESS_DATUM;QN_6;ASH_6;SH_TAG;WASH_6;WAAS_6;eor\r\n"
    "         20;20251231;    3;  99;  99;   99.0;   99.0;eor\r\n"
    "         20;20260126;    3;  26;  26;   33.8;   33.8;eor\r\n"
    "         20;20260127;    3;  10;  10;99999.0;   15.0;eor\r\n"
)
RR_RECENT = (
    "STATIONS_ID;MESS_DATUM;QN_6;  RS; RSF;SH_TAG;NSH_TAG;eor\r\n"
    "          20;20260126;    3;   4.2;   7;  27;   5;eor\r\n"
    "          20;20260127;    1;   0.0;   0;  11;   0;eor\r\n"
)
RR_ZUG = (
    "STATIONS_ID;MESS_DATUM;QN_6;  RS; RSF;SH_TAG;NSH_TAG;eor\r\n"
    "        5792;20260201;    1;   1.0;   7; 122;-999;eor\r\n"
)

WA = "water_equiv"
RR = "more_precip"


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.content = body
        self.text = body.decode("latin-1")
        self.ok = status == 200
        self.status_code = status


class _StubSession:
    """Serves canned DWD files by URL suffix and records requests."""

    def __init__(self):
        self.headers = {}
        self.calls = []
        self.files = {
            f"{WA}/recent/Wa_Tageswerte_Beschreibung_Stationen.txt": WA_STATIONS.encode(
                "latin-1"
            ),
            f"{RR}/recent/RR_Tageswerte_Beschreibung_Stationen.txt": RR_STATIONS.encode(
                "latin-1"
            ),
            f"{WA}/historical/": _listing(
                [
                    "tageswerte_Wa_00001_19370130_19580125_hist.zip",
                    "tageswerte_Wa_00020_20040128_20251231_hist.zip",
                ]
            ).encode(),
            f"{WA}/recent/": _listing(["tageswerte_Wa_00020_akt.zip"]).encode(),
            f"{RR}/historical/": _listing([]).encode(),
            f"{RR}/recent/": _listing(
                ["tageswerte_RR_00020_akt.zip", "tageswerte_RR_05792_akt.zip"]
            ).encode(),
            f"{WA}/historical/tageswerte_Wa_00020_20040128_20251231_hist.zip": _zip(
                "produkt_waequi_tag_20040128_20251231_00020.txt", WA_HIST
            ),
            f"{WA}/recent/tageswerte_Wa_00020_akt.zip": _zip(
                "produkt_waequi_tag_20250412_20260127_00020.txt", WA_RECENT
            ),
            f"{RR}/recent/tageswerte_RR_00020_akt.zip": _zip(
                "produkt_nieder_tag_20250412_20261013_00020.txt", RR_RECENT
            ),
            f"{RR}/recent/tageswerte_RR_05792_akt.zip": _zip(
                "produkt_nieder_tag_20250412_20261013_05792.txt", RR_ZUG
            ),
        }

    def request(self, method, url, params=None, timeout=None):
        self.calls.append(url)
        for suffix, body in self.files.items():
            if url.endswith(suffix):
                return _Resp(body)
        return _Resp(b"not found", 404)


@pytest.fixture
def stub(tmp_path):
    session = _StubSession()
    return DWDClient(session=session, max_retries=1, cache_path=tmp_path), session


# ── Registry compliance ──────────────────────────────────────────────────────


def test_variables_use_shared_vocabulary():
    required = {"name", "type", "units", "output_units", "description", "notes"}
    for key, info in VARIABLES.items():
        assert required | {"source"} <= set(info), key
        assert info["type"] in TYPES, key


def test_snow_variables_emit_cm():
    for key, info in VARIABLES.items():
        if info["type"] in {"swe", "snwd", "snowfall"}:
            assert info["output_units"] == "cm", key


def test_data_flags_are_qn_levels():
    assert set(DATA_FLAGS) == {"1", "2", "3", "5", "7", "8", "9", "10"}


# ── Parsers ──────────────────────────────────────────────────────────────────


def test_parse_station_list_both_layouts():
    wa = _parse_station_list(WA_STATIONS)
    rr = _parse_station_list(RR_STATIONS)
    s = wa["00020"]
    assert s["name"] == "Abtsgmünd-Untergröningen"
    assert s["state"] == "Baden-Württemberg"
    assert s["latitude"] == pytest.approx(48.9219)
    assert s["longitude"] == pytest.approx(9.9129)
    assert s["elevation_m"] == 432.0
    assert (s["begin_date"], s["end_date"]) == ("2004-01-28", "2026-01-27")
    assert s["release"] == "Frei"
    z = rr["05792"]
    assert z["name"] == "Zugspitze" and z["elevation_m"] == 2956.0
    assert z["release"] is None
    assert len(wa) == 2 and len(rr) == 2  # header lines skipped


def test_parse_listing_hist_and_recent():
    h = _parse_listing(_listing(["tageswerte_Wa_00020_20040128_20251231_hist.zip"]))
    assert h["00020"] == {
        "file": "tageswerte_Wa_00020_20040128_20251231_hist.zip",
        "begin": "2004-01-28",
        "end": "2025-12-31",
    }
    r = _parse_listing(_listing(["tageswerte_RR_05792_akt.zip"]))
    assert r["05792"]["end"] is None


def test_parse_produkt_missing_and_qn():
    cols, rows = _parse_produkt(_zip("produkt_x.txt", WA_HIST))
    assert cols == ["ASH_6", "SH_TAG", "WASH_6", "WAAS_6"]
    assert rows["2025-12-30"]["WASH_6"] == 20.5
    assert rows["2025-12-31"]["WASH_6"] is None  # -999
    assert rows["2025-12-31"]["QN_6"] == "9"


def test_parse_produkt_bad_zip():
    with pytest.raises(DWDError, match="zip"):
        _parse_produkt(b"not a zip")


def test_convert_and_clamp():
    assert _convert("WASH_6", 33.8) == pytest.approx(3.38)  # mm → cm
    assert _convert("SH_TAG", 26.0) == 26.0
    assert _convert("RS", 4.2) == 4.2
    assert _convert("WASH_6", 99999.0) is None  # > 15 m
    assert _convert("SH_TAG", -5.0) is None
    assert _convert("SH_TAG", None) is None


def test_merge_prefers_historical():
    m = _merge({"2025-12-31": {"v": 1}}, {"2025-12-31": {"v": 2}, "2026-01-01": {}})
    assert m["2025-12-31"] == {"v": 1} and "2026-01-01" in m


def test_pad_id():
    assert _pad_id(20) == "00020" and _pad_id("5792") == "05792"
    with pytest.raises(DWDError):
        _pad_id("abc")


# ── Variable / interval resolution ───────────────────────────────────────────


def test_resolve_defaults_to_snow():
    assert _resolve_variables(None) == ["WASH_6", "SH_TAG"]


def test_resolve_types_and_codes():
    assert _resolve_variables(["swe", "NSH_TAG"]) == ["WASH_6", "NSH_TAG"]
    assert _resolve_variables("snowfall") == ["NSH_TAG"]


def test_unknown_variable_raises():
    with pytest.raises(DWDError, match="Unknown variable"):
        _resolve_variables(["WTEQ"])
    with pytest.raises(DWDError, match="Unknown variable"):
        _resolve_variables("other")


def test_unknown_interval_raises(stub):
    client, _ = stub
    with pytest.raises(DWDError, match="Unsupported interval"):
        client.get_data("20", interval="hourly")


def test_get_data_requires_station_or_bbox(stub):
    with pytest.raises(ValueError):
        stub[0].get_data()


# ── Client behaviour against the stub ───────────────────────────────────────


def test_get_all_stations_merges_products(stub):
    by_id = {s["station_id"]: s for s in stub[0].get_all_stations()}
    assert set(by_id) == {"00001", "00020", "05792"}
    assert by_id["00020"]["has_swe"] is True
    assert set(by_id["00020"]["products"]) == {WA, RR}
    assert by_id["05792"]["has_swe"] is False
    assert by_id["00001"]["status"] == "Inactive"
    assert by_id["00020"]["end_date"] == "2026-10-13"


def test_get_stations_product_and_bbox(stub):
    client = stub[0]
    assert {s["station_id"] for s in client.get_stations(products=WA)} == {
        "00001",
        "00020",
    }
    kept = client.get_all_stations(bbox=(10.5, 47.0, 11.5, 48.0))
    assert [s["station_id"] for s in kept] == ["05792"]
    with pytest.raises(DWDError, match="Unknown DWD product"):
        client.get_stations(products="kl")


def test_get_metadata_lists_variables(stub):
    meta = stub[0].get_metadata(20)
    names = [d["name"] for d in meta["data_variables"]]
    assert names == ["WASH_6", "WAAS_6", "ASH_6", "SH_TAG", "NSH_TAG", "RS"]
    assert all(d["interval"] in INTERVALS for d in meta["data_variables"])
    zug = stub[0].get_metadata("05792")
    assert "WASH_6" not in [d["name"] for d in zug["data_variables"]]
    with pytest.raises(DWDError, match="not found"):
        stub[0].get_metadata("99999")


def test_get_data_merges_hist_recent_and_flags(stub):
    client, _ = stub
    recs = client.get_data("20", variables="swe", include_flags=True)
    assert all(RECORD_KEYS <= set(r) for r in recs)
    by_date = {r["date"]: r for r in recs}
    # Historical wins on the overlapping day (20.5 mm, QN 9) over recent (99)
    assert by_date["2025-12-30"]["value"] == pytest.approx(2.05)
    assert "2025-12-31" not in by_date  # missing in historical → dropped
    assert by_date["2026-01-26"]["value"] == pytest.approx(3.38)
    assert by_date["2026-01-26"]["flag"] == "3"
    assert by_date["2026-01-27"]["value"] is None  # clamped
    assert all(r["units"] == "cm" and r["interval"] == "daily" for r in recs)


def test_snow_depth_comes_from_more_precip(stub):
    recs = stub[0].get_data("00020", variables=["snwd", "snowfall", "precip"])
    got = {(r["date"], r["variable"]): r["value"] for r in recs}
    assert got[("2026-01-26", "SH_TAG")] == 27.0  # more_precip, not 26
    assert got[("2026-01-26", "NSH_TAG")] == 5.0
    assert got[("2026-01-26", "RS")] == 4.2
    assert ("2025-12-30", "SH_TAG") not in got  # not in more_precip


def test_station_without_swe_yields_nothing_for_swe(stub):
    client, _ = stub
    assert client.get_data("5792", variables="swe") == []
    recs = client.get_data("5792")
    assert [(r["variable"], r["value"]) for r in recs] == [("SH_TAG", 122.0)]


def test_date_window_skips_recent_and_caches_hist(stub, tmp_path):
    client, session = stub
    recs = client.get_data(
        "20", variables="WASH_6", begin_date="2025-12-01", end_date="2025-12-31"
    )
    assert [r["date"] for r in recs] == ["2025-12-30"]
    assert not any(u.endswith("tageswerte_Wa_00020_akt.zip") for u in session.calls)
    assert (tmp_path / WA / "tageswerte_Wa_00020_20040128_20251231_hist.zip").exists()
    n = len(session.calls)
    client.get_data("20", variables="WASH_6", end_date="2025-12-31")
    assert len(session.calls) == n  # served from the disk cache


def test_bbox_get_data(stub):
    recs = stub[0].get_data(bbox=(10.5, 47.0, 11.5, 48.0))
    assert {r["station_id"] for r in recs} == {"05792"}


# ── Live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_station_lists():
    client = DWDClient()
    wa = client.get_stations(products=WA)
    assert len(wa) > 1000
    assert sum(s["status"] == "Active" for s in wa) > 300
    assert any(s["station_id"] == "05792" for s in client.get_all_stations())


@pytest.mark.live
def test_live_swe_balderschwang(tmp_path):
    client = DWDClient(cache_path=tmp_path)
    recs = client.get_data(
        "00275", begin_date="2025-01-01", end_date="2025-03-31", include_flags=True
    )
    swe = [r for r in recs if r["variable"] == "WASH_6" and r["value"]]
    assert swe and all(0 < r["value"] < 150 for r in swe)
    assert {r["variable"] for r in recs} == {"WASH_6", "SH_TAG"}
