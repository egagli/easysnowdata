"""
Tests for RIHMIClient.

Offline tests serve synthetic NSIDC G01170 files (``station.loc`` and a
``syn.tar`` built in memory, both in the real column layout) through a stub
session, and parse synthetic АИСОРИ-M exports written from the documented
RIHMI record layouts.  The ``live`` tests read the real NSIDC archive (no
credentials); RIHMI's own АИСОРИ-M service needs a login and is not tested.
"""

import io
import tarfile

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.rihmi import RIHMIClient, RIHMIError
from easysnowdata.stations.clients.rihmi.rihmi_client import (
    _ROUTE_CATALOGUE,
    _SNOW_COVER_CATALOGUE,
    DATA_FLAGS,
    VARIABLES,
    _convert,
    _ddmm,
    _parse_station_loc,
    _read_catalogue,
    _resolve_variables,
    parse_aisori_text,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

# Real-layout lines from G01170 station.loc (Murmansk is in both RIHMI
# catalogues; 99001 is a made-up NSIDC-only station).
STATION_LOC = (
    "22113   MURMANSK                        6858     3303       46\n"
    "21982   OSTROV VRANGELJA                7059    18131\n"
    "99001   TEST NSIDC ONLY                 5000     4000      100\n"
)

# Real-layout .syn lines (17 columns): field + forest for Murmansk, a
# duplicated field line, and a line with the undocumented -1 code.
SYN_19860110 = (
    "22113 6858 03303 1 10 10 -1 045 070 020  0.22 -02 -02 099 03 04 00\n"
    "22113 6858 03303 2 10 10 -2 060 080 040  0.20 -02 -02 120 03 04 05\n"
    "22113 6858 03303 1 10 10 -2 999 999 999  0.99 -02 -02 999 03 04 00\n"
    "99001 5000 04000 1 10 10 -2 002 004 000 -1.00 -02 -02 -01 01 01 00\n"
)
SYN_19860120 = "22113 6858 03303 1 10 10 00 050 075 025  0.24 002 -02 120 03 04 01\n"
SYN_19920110 = "22113 6858 03303 1 10 10 00 030 040 020  0.20 -02 -02 060 03 04 00\n"


def _tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# АИСОРИ-M route-survey export, 19 fields, ";"-separated with blanks (the
# layout is an assumption — see the module docstring).
ROUTE_EXPORT = (
    "Маршрутные снегомерные съемки\n"
    "22113;1986;01;1;10; 10; 10;  0;  50;  80;  20;0.25;   ;   ;   ; 125; 125; 3; 4\n"
    "22113;1986;01;2;10; 10; 10;  0;  70;  90;  50;0.21;   ;   ;   ; 147;9999; 3; 4;\n"
    "22113;1986;02;1;31; 10; 10;  0;  50;  80;  20;0.25;   ;   ;   ; 125; 125; 3; 4\n"
)

# Daily snow cover in the documented IIIII.dat fixed-width layout.
SNOW_COVER_DAT = (
    "22113 1986 01 10   45 10 0 0 0\n"
    "22113 1986 01 11 9999 99 9 0 0\n"
    "22113 1986 01 12   47 10 0 1 0\n"
    "22113 1955 01 12   30  1 0 0 0\n"
)


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.content = body
        self.text = body.decode("latin-1")
        self.ok = status == 200
        self.status_code = status
        self.headers = {}


class _StubSession:
    def __init__(self):
        self.headers = {}
        self.calls: list[str] = []
        self.files = {
            "station.loc": STATION_LOC.encode(),
            "product/syn.tar": _tar(
                {"./19860110.syn": SYN_19860110, "./19860120.syn": SYN_19860120}
            ),
            "update04012002/syn.tar": _tar({"19920110.syn": SYN_19920110}),
            "update02172003/syn.tar": _tar({}),
            "update05152003/syn.tar": _tar({}),
            "update04052004/syn.tar": _tar({}),
        }

    def request(self, method, url, params=None, timeout=None):
        self.calls.append(url)
        for suffix, body in self.files.items():
            if url.endswith(suffix):
                return _Resp(body)
        return _Resp(b"not found", 404)


@pytest.fixture
def stub_client(tmp_path):
    return RIHMIClient(
        cache_dir=tmp_path / "cache", session=_StubSession(), max_retries=1
    )


@pytest.fixture
def export_dir(tmp_path):
    d = tmp_path / "aisori"
    d.mkdir()
    (d / "routes.txt").write_bytes(ROUTE_EXPORT.encode("cp1251"))
    (d / "22113.dat").write_text(SNOW_COVER_DAT)
    return d


# ── Registry compliance ──────────────────────────────────────────────────────


def test_variables_use_shared_vocabulary():
    required = {
        "name",
        "type",
        "units",
        "output_units",
        "description",
        "notes",
        "source",
        "interval",
    }
    for key, info in VARIABLES.items():
        assert required <= set(info), key
        assert info["type"] in TYPES, key
        assert info["interval"] in INTERVALS, key
        if info["type"] in {"swe", "snwd"}:
            assert info["output_units"] == "cm", key
    assert VARIABLES["DENSITY"]["output_units"] == "%"


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES,):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_data_flags_documented():
    assert "NSIDC:5" in DATA_FLAGS and "Q1:9" in DATA_FLAGS


# ── Catalogues and parsers ───────────────────────────────────────────────────


def test_packaged_catalogues():
    routes = _read_catalogue(_ROUTE_CATALOGUE)
    cover = _read_catalogue(_SNOW_COVER_CATALOGUE)
    assert len(routes) == 517
    assert len(cover) == 620
    murmansk = routes["22113"]
    assert murmansk["latitude"] == pytest.approx(68 + 58 / 60, abs=1e-3)
    assert murmansk["longitude"] == pytest.approx(33 + 3 / 60, abs=1e-3)
    assert murmansk["elevation_m"] == 57
    # Wrangel Island, printed 181°31′E → west of the antimeridian.
    assert cover["21982"]["longitude"] == pytest.approx(-178.483, abs=1e-3)
    assert all(-180 <= s["longitude"] <= 180 for s in cover.values())


def test_ddmm_and_station_loc():
    assert _ddmm("5545") == pytest.approx(55.75)
    assert _ddmm("03734") == pytest.approx(37 + 34 / 60, abs=1e-4)
    assert _ddmm("18131") == pytest.approx(-178.4833, abs=1e-3)
    st = _parse_station_loc(STATION_LOC)
    assert st["22113"]["name"] == "MURMANSK"
    assert st["22113"]["elevation_m"] == 46
    assert st["21982"]["elevation_m"] is None


def test_parse_route_export_semicolon():
    rows = parse_aisori_text(ROUTE_EXPORT)
    assert len(rows) == 2  # the 31 February line is skipped
    field = rows[0]
    assert field["kind"] == "route_survey"
    assert (field["date"], field["route_type"]) == ("1986-01-10", "field")
    assert field["values"]["SWE"] == 125
    assert field["values"]["ICE_CRUST_THICKNESS"] is None  # blank
    assert rows[1]["route_type"] == "forest"
    assert rows[1]["values"]["WATER_TOTAL"] is None  # 9999 + trailing ';'


def test_parse_route_export_whitespace():
    line = "22113 1986 1 1 10 10 10 0 50 80 20 0.25 0 0 0 125 125 3 4\n"
    (row,) = parse_aisori_text(line)
    assert row["values"]["HS_MEAN"] == 50 and row["values"]["DENSITY"] == 0.25


def test_parse_snow_cover_dat():
    rows = parse_aisori_text(SNOW_COVER_DAT)
    assert [r["kind"] for r in rows] == ["snow_cover"] * 4
    assert rows[0]["values"] == {"H": 45, "S": 10}
    assert rows[1]["values"] == {"H": None, "S": None}  # 9999 / 99 / Q1=9
    assert rows[2]["flag"] == "Q1:0,Q2:1,Q3:0"
    assert rows[3]["values"]["S"] is None  # 0/1 code before Aug 1959


def test_parse_skips_bad_lines_and_kind_override():
    text = "header\n22113;1986;13;1;10;10;10;0;5;8;2;.2;;;;12;12;3;4\n"
    assert parse_aisori_text(text) == []  # month 13
    with pytest.raises(RIHMIError):
        parse_aisori_text("", kind="bogus")
    # Forcing the wrong kind drops lines whose field count does not fit.
    assert parse_aisori_text(SNOW_COVER_DAT, kind="route_survey") == []


def test_convert_scales_and_clamps():
    assert _convert("SWE", 125) == pytest.approx(12.5)  # mm → cm
    assert _convert("DENSITY", 0.25) == pytest.approx(25.0)  # g/cm3 → %
    assert _convert("COVER_ROUTE", 7) == pytest.approx(70.0)  # tenths → %
    assert _convert("HS_MEAN", 2000) is None
    assert _convert("DENSITY", 1.5) is None
    assert _convert("COVER_ROUTE", 99) is None
    assert _convert("SWE", None) is None


# ── Variable / interval resolution ───────────────────────────────────────────


def test_resolve_defaults_and_types():
    assert _resolve_variables(None, "periodic") == ["SWE", "HS_MEAN"]
    assert _resolve_variables(None, "daily") == ["H"]
    assert _resolve_variables(["density", "SWE"], "periodic") == ["DENSITY", "SWE"]
    assert _resolve_variables("snwd", "daily") == ["H"]


def test_unknown_or_wrong_interval_variable_raises():
    with pytest.raises(RIHMIError, match="Unknown variable"):
        _resolve_variables(["WTEQ"], "periodic")
    with pytest.raises(RIHMIError, match="belongs to interval"):
        _resolve_variables(["H"], "periodic")


def test_unknown_interval_route_and_source(stub_client):
    with pytest.raises(RIHMIError, match="Unsupported interval"):
        stub_client.get_data("22113", interval="hourly")
    with pytest.raises(RIHMIError, match="route type"):
        stub_client.get_data("22113", route_types="meadow")
    with pytest.raises(RIHMIError, match="source"):
        stub_client.get_data("22113", source="aisori")


def test_get_data_requires_station_or_bbox(stub_client):
    with pytest.raises(ValueError):
        stub_client.get_data()


# ── Client behaviour against the stub ───────────────────────────────────────


def test_get_all_stations_union_and_status(stub_client):
    stations = stub_client.get_all_stations()
    by_id = {s["station_id"]: s for s in stations}
    assert len(stations) == 517 + 620 - 231 + 1  # + the NSIDC-only test site
    m = by_id["22113"]
    assert m["name"] == "Мурманск" and m["name_latin"] == "MURMANSK"
    assert m["status"] == "Active" and m["in_nsidc_g01170"]
    assert by_id["99001"]["status"] == "Inactive"
    assert by_id["20289"]["status"] == "Inactive"  # «Закрыта в 1999 г.»
    for s in stations:
        assert {"station_id", "name", "latitude", "longitude", "elevation_m"} <= set(s)
        assert s["status"] in {"Active", "Inactive"}


def test_get_all_stations_bbox_and_active(stub_client):
    kola = stub_client.get_all_stations(bbox=(32.5, 68.5, 33.5, 69.5))
    assert "22113" in {s["station_id"] for s in kola}
    active = stub_client.get_all_stations(active_only=True)
    assert all(s["status"] == "Active" for s in active)


def test_get_metadata(stub_client):
    meta = stub_client.get_metadata("22113")
    names = {d["name"] for d in meta["data_variables"]}
    assert {"SWE", "HS_MEAN", "H", "S"} <= names
    assert all(d["interval"] in INTERVALS for d in meta["data_variables"])
    only = stub_client.get_metadata("99001")
    names = {d["name"] for d in only["data_variables"]}
    assert "WATER_TOTAL" not in names and "H" not in names
    with pytest.raises(RIHMIError, match="not found"):
        stub_client.get_metadata("00001")


def test_get_data_nsidc_records(stub_client):
    recs = stub_client.get_data(
        "22113",
        variables=["SWE", "HS_MEAN", "ICE_CRUST_COVER"],
        begin_date="1986-01-01",
        end_date="1986-01-31",
        include_flags=True,
    )
    assert all(RECORD_KEYS <= set(r) for r in recs)
    assert all(r["interval"] == "periodic" for r in recs)
    key = {(r["date"], r["route_type"], r["variable"]): r for r in recs}
    assert key[("1986-01-10", "field", "SWE")]["value"] == pytest.approx(9.9)
    assert key[("1986-01-10", "field", "HS_MEAN")]["value"] == 45
    # -1 (undocumented) → None; duplicate field line ignored (first wins).
    assert key[("1986-01-10", "field", "ICE_CRUST_COVER")]["value"] is None
    assert key[("1986-01-10", "forest", "SWE")]["flag"] == "NSIDC:5"
    assert key[("1986-01-20", "field", "SWE")]["source"] == "nsidc_g01170"
    # Field before forest on the same date.
    first = [r for r in recs if r["date"] == "1986-01-10"]
    assert first[0]["route_type"] == "field"
    # The 1991–96 update tar was not needed for a 1986 window.
    assert not any("update" in u for u in stub_client._session.calls)


def test_get_data_route_filter_and_updates(stub_client):
    recs = stub_client.get_data("22113", route_types="forest", end_date="1986-12-31")
    assert {r["route_type"] for r in recs} == {"forest"}
    late = stub_client.get_data("22113", variables="swe", begin_date="1991-01-01")
    assert [(r["date"], r["value"]) for r in late] == [("1992-01-10", 6.0)]


def test_local_export_wins_over_nsidc(tmp_path, export_dir):
    client = RIHMIClient(
        data_dir=export_dir,
        cache_dir=tmp_path / "cache",
        session=_StubSession(),
        max_retries=1,
    )
    recs = client.get_data("22113", variables="SWE", end_date="1986-12-31")
    by = {(r["date"], r["route_type"]): r for r in recs}
    assert by[("1986-01-10", "field")]["source"] == "aisori_export"
    assert by[("1986-01-10", "field")]["value"] == pytest.approx(12.5)
    assert by[("1986-01-20", "field")]["source"] == "nsidc_g01170"
    # 31 February in the export is not a date: that line is skipped.
    assert not any(d.startswith("1986-02") for d, _ in by)
    files_only = client.get_data("22113", variables="SWE", source="files")
    assert {r["source"] for r in files_only} == {"aisori_export"}


def test_daily_from_export(tmp_path, export_dir):
    client = RIHMIClient(data_dir=export_dir, use_nsidc=False)
    recs = client.get_data("22113", interval="daily", include_flags=True)
    vals = {r["date"]: r["value"] for r in recs}
    assert vals["1986-01-10"] == 45 and vals["1986-01-11"] is None
    assert all(r["type"] == "snwd" and r["interval"] == "daily" for r in recs)
    assert {r["flag"] for r in recs} >= {"Q1:9,Q2:0,Q3:0"}
    cov = client.get_data(
        "22113",
        interval="daily",
        variables="S",
        begin_date="1986-01-10",
        end_date="1986-01-10",
    )
    assert cov[0]["value"] == pytest.approx(100.0) and cov[0]["units"] == "%"


def test_daily_without_export_raises(stub_client):
    with pytest.raises(RIHMIError, match="АИСОРИ"):
        stub_client.get_data("22113", interval="daily")
    with pytest.raises(RIHMIError, match="no daily"):
        stub_client.get_data("22113", interval="daily", source="nsidc")


def test_download_error_raises(tmp_path):
    session = _StubSession()
    session.files.pop("product/syn.tar")
    client = RIHMIClient(cache_dir=tmp_path, session=session, max_retries=1)
    with pytest.raises(RIHMIError, match="404"):
        client.get_data("22113", end_date="1980-01-01")


def test_base_url_override(tmp_path):
    sess = _StubSession()
    client = RIHMIClient(
        base_url="https://mirror.example/G01170/",
        cache_dir=tmp_path / "cache",
        session=sess,
        max_retries=1,
    )
    assert client.get_all_stations()
    assert sess.calls == ["https://mirror.example/G01170/station.loc"]


# ── Live tests (NSIDC G01170, no credentials) ───────────────────────────────


@pytest.mark.live
def test_live_nsidc_station_list():
    stations = RIHMIClient().get_all_stations()
    nsidc = [s for s in stations if s["in_nsidc_g01170"]]
    assert len(nsidc) == 1345
    assert len(stations) == 1552


@pytest.mark.live
def test_live_nsidc_murmansk_winter():
    recs = RIHMIClient().get_data(
        "22113", begin_date="1986-01-01", end_date="1986-04-30", route_types="field"
    )
    swe = [r for r in recs if r["variable"] == "SWE" and r["value"] is not None]
    assert swe and all(r["units"] == "cm" for r in swe)
    assert max(r["value"] for r in swe) > 5
