"""
Tests for GHCNDClient.

The offline tests feed synthetic fixed-width and CSV payloads through a stub
session, so they touch no socket.  The ``live`` tests read the real NCEI
GHCN-Daily files (no credentials needed).
"""

import gzip
import io

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.ghcnd import GHCNDClient, GHCNDError
from easysnowdata.stations.clients.ghcnd.ghcnd_client import (
    DATA_FLAGS,
    VARIABLES,
    _convert,
    _parse_by_station_csv,
    _parse_inventory_txt,
    _parse_stations_txt,
    _resolve_variables,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

# Two real-layout lines from ghcnd-stations.txt (column positions matter).
STATIONS_TXT = (
    "USS0021D08S  47.0000 -121.5800 1478.3 WA CORRAL PASS"
    "                                     \n"
    "RSM00027612  55.8330   37.6170  156.0    MOSCOW"
    "                         GSN     27612\n"
    "XXM00099999  10.0000   10.0000 -999.9    NO ELEVATION"
    "                                    \n"
)

INVENTORY_TXT = (
    "USS0021D08S  47.0000 -121.5800 WESD 1980 2026\n"
    "USS0021D08S  47.0000 -121.5800 SNWD 1980 2026\n"
    "USS0021D08S  47.0000 -121.5800 TMAX 1980 2026\n"
    "RSM00027612  55.8330   37.6170 SNWD 1973 2022\n"
    "RSM00027612  55.8330   37.6170 TMAX 1937 2022\n"
    "XXM00099999  10.0000   10.0000 TMAX 1990 2026\n"
)

STATION_CSV = (
    "USS0021D08S,20240101,SNWD,584,,,T,\n"
    "USS0021D08S,20240101,WESD,1753,,,T,\n"
    "USS0021D08S,20240101,TMAX,-55,,D,T,\n"
    "USS0021D08S,20240102,WESD,1753,,X,T,\n"
    "USS0021D08S,20240103,SNWD,99999,,,T,\n"
    "USS0021D08S,20240201,SNWD,600,,,T,\n"
)


YEAR_CSV = STATION_CSV + (
    "RSM00027612,20240101,SNWD,250,,,E,\nRSM00027612,20240101,TMAX,-80,,,E,\n"
)


class _Resp:
    def __init__(self, body: bytes):
        self.content = body
        self.text = body.decode("ascii", errors="replace")
        self.ok = True
        self.status_code = 200


class _StubSession:
    """Serves canned GHCN-D files by URL suffix."""

    def __init__(self):
        self.headers = {}
        self.files = {
            "ghcnd-stations.txt": STATIONS_TXT.encode(),
            "ghcnd-inventory.txt": INVENTORY_TXT.encode(),
            "by_station/USS0021D08S.csv.gz": gzip.compress(STATION_CSV.encode()),
            "by_year/2024.csv.gz": gzip.compress(YEAR_CSV.encode()),
        }

    def request(self, method, url, params=None, timeout=None):
        for suffix, body in self.files.items():
            if url.endswith(suffix):
                return _Resp(body)
        resp = _Resp(b"not found")
        resp.ok, resp.status_code = False, 404
        return resp

    def get(self, url, stream=False, timeout=None):
        resp = self.request("GET", url)
        resp.raw = io.BytesIO(resp.content)
        return _Ctx(resp)


class _Ctx:
    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self.resp

    def __exit__(self, *exc):
        return False


@pytest.fixture
def stub_client():
    return GHCNDClient(session=_StubSession(), max_retries=1)


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
    }
    for key, info in VARIABLES.items():
        assert required <= set(info), key
        assert info["type"] in TYPES, key


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES,):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_snow_variables_emit_cm():
    for key, info in VARIABLES.items():
        if info["type"] in {"swe", "snwd", "snowfall"}:
            assert info["output_units"] == "cm", key


def test_data_flags_are_single_letters():
    assert DATA_FLAGS and all(len(k) == 1 for k in DATA_FLAGS)


# ── Parsers ──────────────────────────────────────────────────────────────────


def test_parse_stations_fixed_width():
    st = _parse_stations_txt(STATIONS_TXT)
    corral = st["USS0021D08S"]
    assert corral["latitude"] == pytest.approx(47.0)
    assert corral["longitude"] == pytest.approx(-121.58)
    assert corral["elevation_m"] == pytest.approx(1478.3)
    assert corral["state"] == "WA"
    assert corral["name"] == "CORRAL PASS"
    assert corral["ghcn_network"] == "S"
    moscow = st["RSM00027612"]
    assert moscow["wmo_id"] == "27612"
    assert moscow["gsn"] is True
    assert moscow["state"] is None
    assert st["XXM00099999"]["elevation_m"] is None  # -999.9 sentinel


def test_parse_inventory_filters_elements():
    inv = _parse_inventory_txt(INVENTORY_TXT, {"SNWD", "WESD"})
    assert inv["USS0021D08S"] == {"WESD": (1980, 2026), "SNWD": (1980, 2026)}
    assert "XXM00099999" not in inv


def test_parse_by_station_csv_filters_dates_and_elements():
    raw = gzip.compress(STATION_CSV.encode())
    rows = _parse_by_station_csv(raw, {"SNWD"}, "2024-01-01", "2024-01-31")
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("2024-01-01", "SNWD", 584),
        ("2024-01-03", "SNWD", 99999),
    ]


def test_parse_by_station_csv_rejects_non_gzip():
    with pytest.raises(GHCNDError, match="decompress"):
        _parse_by_station_csv(b"not gzip", {"SNWD"}, None, None)


def test_convert_scales_and_clamps():
    assert _convert("SNWD", 584) == pytest.approx(58.4)  # mm → cm
    assert _convert("WESD", 1753) == pytest.approx(17.53)  # tenths mm → cm
    assert _convert("TMAX", -55) == pytest.approx(-5.5)  # negative temps survive
    assert _convert("PRCP", 123) == pytest.approx(12.3)  # tenths mm → mm
    assert _convert("SNWD", 99999) is None  # 100 m of snow: implausible
    assert _convert("SNWD", -10) is None
    assert _convert("SNWD", -9999) is None


# ── Variable / interval resolution ───────────────────────────────────────────


def test_resolve_defaults_to_snow():
    assert _resolve_variables(None) == ["WESD", "SNWD"]


def test_resolve_types_and_codes():
    assert _resolve_variables(["swe", "SNOW"]) == ["WESD", "SNOW"]
    assert _resolve_variables("temp_max") == ["TMAX"]


def test_unknown_variable_raises():
    with pytest.raises(GHCNDError, match="Unknown variable"):
        _resolve_variables(["WTEQ"])


def test_unknown_interval_raises(stub_client):
    with pytest.raises(GHCNDError, match="Unsupported interval"):
        stub_client.get_data("USS0021D08S", interval="hourly")


def test_get_data_requires_station_or_bbox(stub_client):
    with pytest.raises(ValueError):
        stub_client.get_data()


# ── Client behaviour against the stub ───────────────────────────────────────


def test_get_all_stations_keeps_snow_stations_only(stub_client):
    ids = {s["station_id"] for s in stub_client.get_all_stations()}
    assert ids == {"USS0021D08S", "RSM00027612"}


def test_get_stations_status_and_network_filter(stub_client):
    by_id = {s["station_id"]: s for s in stub_client.get_all_stations()}
    assert by_id["RSM00027612"]["status"] == "Inactive"  # last SNWD 2022
    assert by_id["USS0021D08S"]["has_swe"] is True
    kept = stub_client.get_stations(exclude_networks="S")
    assert [s["station_id"] for s in kept] == ["RSM00027612"]


def test_get_stations_bbox(stub_client):
    kept = stub_client.get_stations(bbox=(-125, 45, -120, 50))
    assert [s["station_id"] for s in kept] == ["USS0021D08S"]


def test_get_metadata_lists_variables(stub_client):
    meta = stub_client.get_metadata("USS0021D08S")
    names = {d["name"] for d in meta["data_variables"]}
    assert names == {"WESD", "SNWD", "TMAX"}
    assert all(d["interval"] in INTERVALS for d in meta["data_variables"])


def test_get_metadata_unknown_station(stub_client):
    with pytest.raises(GHCNDError, match="not found"):
        stub_client.get_metadata("ZZZ00000000")


def test_get_data_records_and_qc_nulling(stub_client):
    recs = stub_client.get_data(
        "USS0021D08S",
        begin_date="2024-01-01",
        end_date="2024-01-31",
        include_flags=True,
    )
    assert all(set(RECORD_KEYS) <= set(r) for r in recs)
    by_key = {(r["date"], r["variable"]): r for r in recs}
    assert by_key[("2024-01-01", "SNWD")]["value"] == pytest.approx(58.4)
    assert by_key[("2024-01-01", "WESD")]["value"] == pytest.approx(17.53)
    # QFLAG X → nulled by default, flag still carried
    assert by_key[("2024-01-02", "WESD")]["value"] is None
    assert by_key[("2024-01-02", "WESD")]["flag"] == ",X,T"
    assert ("2024-02-01", "SNWD") not in by_key  # outside the window
    assert all(r["units"] == "cm" and r["interval"] == "daily" for r in recs)


def test_keep_qc_failed_returns_stored_value():
    client = GHCNDClient(session=_StubSession(), max_retries=1, keep_qc_failed=True)
    recs = client.get_data("USS0021D08S", variables="swe", end_date="2024-01-31")
    assert [r["value"] for r in recs] == [pytest.approx(17.53), pytest.approx(17.53)]


def test_get_data_skips_missing_station(stub_client):
    assert stub_client.get_data(["ZZZ00000000"]) == []


def test_get_year_data_streams_and_filters(stub_client):
    recs = list(stub_client.get_year_data(2024, variables="snwd"))
    assert {r["station_id"] for r in recs} == {"USS0021D08S", "RSM00027612"}
    assert all(r["variable"] == "SNWD" for r in recs)
    moscow = [r for r in recs if r["station_id"] == "RSM00027612"]
    assert moscow[0]["value"] == pytest.approx(25.0)
    only = list(stub_client.get_year_data(2024, station_ids="RSM00027612"))
    assert [r["variable"] for r in only] == ["SNWD"]


def test_get_year_data_missing_year_raises(stub_client):
    with pytest.raises(GHCNDError, match="HTTP 404"):
        list(stub_client.get_year_data(1700))


# ── Live tests ───────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_snotel_mirror_matches_known_values():
    """Corral Pass SNOTEL via GHCN-D, early January 2024."""
    recs = GHCNDClient().get_data(
        "USS0021D08S", begin_date="2024-01-01", end_date="2024-01-01"
    )
    by_var = {r["variable"]: r["value"] for r in recs}
    assert by_var["SNWD"] == pytest.approx(58.4)
    assert by_var["WESD"] == pytest.approx(17.53)


@pytest.mark.live
def test_live_inventory_is_global():
    stations = GHCNDClient().get_all_stations(active_only=True)
    countries = {s["country_code"] for s in stations}
    assert {"US", "CA", "RS"} <= countries
    assert len(stations) > 5000
