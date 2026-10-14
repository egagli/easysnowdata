"""
Tests for SLFClient.

The offline tests serve synthetic payloads that copy the real layouts
(EnviDat CKAN JSON, the ``;``-separated GCOS CSV, the GCOS metadata.txt, the
wide new-snow tables inside a zip, and measurement-data.slf.ch CSVs) through
a stub session, so they touch no socket.  The ``live`` tests read the real
EnviDat and SLF file services (no credentials needed).
"""

import io
import json
import zipfile

import pytest

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.slf import SLFClient, SLFError
from easysnowdata.stations.clients.slf.slf_client import (
    DATA_FLAGS,
    DATASETS,
    VARIABLES,
    _convert,
    _parse_gcos_csv,
    _parse_gcos_metadata,
    _parse_new_snow_table,
    _resolve_variables,
    lv03_to_wgs84,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

# ── Synthetic payloads in the real formats ───────────────────────────────────

GCOS_CKAN = {
    "success": True,
    "result": {
        "name": "gcos-swe-data",
        "license_id": "odc-odbl",
        "resources": [
            {
                "name": "STATION METADATA",
                "url": "https://s3.example/envidat-doi/10.16904_15/data/GCOS/metadata.txt",
            },
            {
                "name": "DATA ACCESS",
                "url": (
                    "https://envicloud.wsl.ch/#/?bucket=https://s3.example/"
                    "envidat-doi/&prefix=10.16904_15/data/GCOS"
                ),
            },
        ],
    },
}

GCOS_META = (
    "STAT_ABK    LABEL    WGS84_LONGITUDE    WGS84_LATITUDE START_YEAR\n"
    "1MR    Mürren (1MR) 1650 m    7.890192955   46.55818213\t1951\n"
    "5WJ    Weissfluhjoch (5WJ) 2540 m    9.8092511    46.82938238\t1947\n"
)

GCOS_CSV = (
    "STAT_ABK;DATUM;HS(cm);SWE(mm)\n"
    "1MR;1949-02-01;65;162\n"
    "1MR;1949-02-15;63;166\n"
    "5WJ;2025-03-01;180;612\n"
    "5WJ;2025-03-15;;\n"
    "5WJ;2025-04-01;99999;700\n"
)

NEW_SNOW_STATIONS = (
    "Station,Easting (m),Northing (m),Altitude (m.a.s.l.)\n"
    "SLF.5WJ,780845,189230,2540\n"
    "SLF.1AD,609920,149820,1350\n"
)


def _wide(desc, units, rows):
    head = (
        "GLOBAL FILE ATTRIBUTES\n\n"
        f"Description:                  {desc}\n"
        f"Units:                        {units}\n"
        "Date created:                 2024-11-11T08:49:57\n\n"
        "TABLE WITH DATA\n\n"
        "Time,SLF.1AD,SLF.5WJ\n"
    )
    return head + "".join(rows)


NEW_SNOW_FILES = {
    "STATION-LIST.txt": NEW_SNOW_STATIONS,
    "SWE/OBS-SWE-PROFILE.txt": _wide(
        "snow water equivalent, observed at manual profile",
        "mm",
        [
            "2017-01-02 06:00:00,-9999,-9999\n",
            "2017-01-03 06:00:00,-9999,101.5\n",
        ],
    ),
    "HS/OBS-HS-STAKE.txt": _wide(
        "snow depth, observed at fixed stake",
        "cm",
        ["2017-01-02 06:00:00,30,40\n", "2017-01-03 06:00:00,31,41\n"],
    ),
    "HS/OBS-HS-PROFILE.txt": _wide(
        "snow depth, observed at manual profile",
        "cm",
        ["2017-01-02 06:00:00,-9999,-9999\n", "2017-01-03 06:00:00,-9999,42\n"],
    ),
    "HN/OBS-HN.txt": _wide(
        "new snow depth, observed",
        "cm",
        ["2017-01-02 06:00:00,0,5\n", "2017-01-03 06:00:00,0,2\n"],
    ),
    "HNW/OBS-HNW.txt": _wide(
        "new snow water equivalent, observed",
        "mm",
        ["2017-01-02 06:00:00,0,6\n", "2017-01-03 06:00:00,-9999,3\n"],
    ),
}


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in NEW_SNOW_FILES.items():
            zf.writestr(f"new_snow_dataset_v1/{name}", text)
    return buf.getvalue()


NEW_SNOW_CKAN = {
    "success": True,
    "result": {
        "name": "dataset-on-new-snow-water-equivalent",
        "resources": [
            {
                "name": "new_snow_dataset_v1.zip",
                "url": "https://www.envidat.ch/dataset/x/resource/y/download/new_snow_dataset_v1.zip",
            }
        ],
    },
}

SP_STATIONS = (
    "network,station_code,label,active,lon,lat,elevation,station_type\n"
    "BEOB,5WJ0,Weissfluhjoch,True,9.809,46.829,2536,\n"
    "BEOB,1XX0,Old Plot,False,7.5,46.5,1200,\n"
)

SP_5WJ0 = (
    "station_code,measure_date,hyear,HS,HN_1D,HNW_1D\n"
    "5WJ0,2026-10-09 06:00:00+00:00,2027,1.0,2.0,8.0\n"
    "5WJ0,2026-10-10 06:00:00+00:00,2027,0.0,0.0,\n"
    "5WJ0,2026-10-11 06:00:00+00:00,2027,,1.0,3.0\n"
)

IMIS_STATIONS = (
    "network,station_code,label,active,lon,lat,elevation,station_type\n"
    "IMIS,WFJ2,Weissfluhjoch,True,9.809289,46.829638,2536,SNOW_FLAT\n"
    "IMIS,WFJ1,Weissfluhjoch Wind,True,9.81,46.83,2690,WIND\n"
)

IMIS_DAILY = (
    "station_code,measure_date,hyear,HS,HN_1D\n"
    "WFJ2,1992-10-01 00:00:00+00:00,1993,1.0,\n"
    "WFJ2,1992-10-02 00:00:00+00:00,1993,-1.0,\n"
    "WFJ2,1992-10-03 00:00:00+00:00,1993,12.0,10.0\n"
)

IMIS_30MIN = (
    "station_code,measure_date,hyear,TA_30MIN_MEAN,VW_30MIN_MEAN,HS\n"
    "WFJ2,2026-10-10 00:00:00+00:00,2027,-2.6,0.0,0.0\n"
    "WFJ2,2026-10-10 00:30:00+00:00,2027,-2.2,0.0,\n"
)


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.content = body
        self.text = body.decode("utf-8", errors="replace")
        self.status_code = status
        self.ok = status < 400
        self.headers = {}

    def json(self):
        return json.loads(self.text)


class _StubSession:
    """Serves canned files by URL suffix and CKAN records by ``id``."""

    def __init__(self):
        self.headers = {}
        self.calls = []
        self.files = {
            "GCOS/metadata.txt": GCOS_META.encode(),
            "GCOS/SWE_GCOS.csv": GCOS_CSV.encode(),
            "download/new_snow_dataset_v1.zip": _zip_bytes(),
            "study-plot/stations.csv": SP_STATIONS.encode(),
            "study-plot/data/by_station/5WJ0.csv": SP_5WJ0.encode(),
            "imis/stations.csv": IMIS_STATIONS.encode(),
            "daily_snow_by_station/daily_snow_WFJ2.csv": IMIS_DAILY.encode(),
            "imis/data/by_station/WFJ2.csv": IMIS_30MIN.encode(),
        }
        self.ckan = {
            "gcos-swe-data": GCOS_CKAN,
            "dataset-on-new-snow-water-equivalent": NEW_SNOW_CKAN,
        }

    def request(self, method, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("package_show"):
            rec = self.ckan.get((params or {}).get("id"))
            if rec is None:
                return _Resp(b'{"success": false}', 404)
            return _Resp(json.dumps(rec).encode())
        for suffix, body in self.files.items():
            if url.endswith(suffix):
                return _Resp(body)
        return _Resp(b"not found", 404)


def _client(dataset, tmp_path):
    return SLFClient(dataset, cache_dir=tmp_path, session=_StubSession(), max_retries=1)


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
    for name, info in VARIABLES.items():
        assert required <= set(info), name
        assert info["type"] in TYPES, name
    assert DATA_FLAGS == {}


def test_output_units_are_canonical():
    """Typed variables emit exactly the networks.TYPES unit (DESIGN.md §3.5)."""
    for reg in (VARIABLES,):
        for key, info in reg.items():
            if info["type"] != "other":
                assert info["output_units"] == canonical_units(info["type"]), key


def test_datasets_reference_known_variables_and_intervals():
    for ds, spec in DATASETS.items():
        for var, vspec in spec["variables"].items():
            assert var in VARIABLES, (ds, var)
            assert set(vspec["intervals"]) <= INTERVALS, (ds, var)
        assert set(spec["default_variables"]) <= set(spec["variables"])


# ── Parsers / units ──────────────────────────────────────────────────────────


def test_lv03_conversion_matches_weissfluhjoch():
    lat, lon = lv03_to_wgs84(780849, 189231)  # WFJ2 SMET header
    assert lat == pytest.approx(46.829636, abs=1e-4)
    assert lon == pytest.approx(9.809290, abs=1e-4)


def test_parse_gcos_metadata():
    meta = _parse_gcos_metadata(GCOS_META)
    assert set(meta) == {"1MR", "5WJ"}
    assert meta["1MR"]["name"] == "Mürren"
    assert meta["1MR"]["elevation_m"] == 1650.0
    assert meta["5WJ"]["latitude"] == pytest.approx(46.82938238)
    assert meta["5WJ"]["start_year"] == 1947


def test_parse_gcos_csv_and_new_snow_table():
    df = _parse_gcos_csv(GCOS_CSV)
    assert list(df["SWE"][:2]) == ["162", "166"]
    long = _parse_new_snow_table(NEW_SNOW_FILES["HN/OBS-HN.txt"])
    assert set(long["station_code"]) == {"1AD", "5WJ"}
    assert len(long) == 4


def test_convert_scales_and_clamps():
    assert _convert("SWE", "612") == 61.2
    assert _convert("HS", "-1") is None
    assert _convert("HS", "99999") is None
    assert _convert("HS", "-9999") is None
    assert _convert("HN_1D", "600") is None
    assert _convert("HNW_1D", "8") == 8.0
    assert _convert("TA_30MIN_MEAN", "-2.6") == -2.6


def test_resolve_variables_defaults_and_types():
    assert _resolve_variables("gcos", None, None) == [
        ("SWE", "periodic"),
        ("HS", "periodic"),
    ]
    assert _resolve_variables("new_snow", None, None) == [
        ("SWE", "periodic"),
        ("HS", "daily"),
    ]
    assert _resolve_variables("new_snow", None, "periodic") == [
        ("SWE", "periodic"),
        ("HS_PROFILE", "periodic"),
    ]
    assert _resolve_variables("new_snow", "snwd", "daily") == [("HS", "daily")]
    assert _resolve_variables("imis", ["HS", "temp"], "sub_hourly") == [
        ("HS", "sub_hourly"),
        ("TA_30MIN_MEAN", "sub_hourly"),
    ]


def test_unknown_variable_and_interval_raise():
    with pytest.raises(SLFError):
        _resolve_variables("gcos", "bogus", None)
    with pytest.raises(SLFError):
        _resolve_variables("gcos", "HN_1D", None)  # exists, but not in gcos
    with pytest.raises(SLFError):
        _resolve_variables("gcos", None, "daily")
    with pytest.raises(SLFError):
        _resolve_variables("new_snow", "SWE", "daily")
    with pytest.raises(SLFError):
        _resolve_variables("study_plot", None, "hourly")
    with pytest.raises(SLFError):
        SLFClient("nope")


def test_get_data_requires_station_or_bbox(tmp_path):
    with pytest.raises(ValueError):
        _client("gcos", tmp_path).get_data()


# ── GCOS ─────────────────────────────────────────────────────────────────────


def test_gcos_resolves_s3_folder_from_ckan(tmp_path):
    c = _client("gcos", tmp_path)
    assert c._gcos_base() == "https://s3.example/envidat-doi/10.16904_15/data/GCOS"


def test_gcos_stations_and_status(tmp_path):
    c = _client("gcos", tmp_path)
    sta = {s["station_id"]: s for s in c.get_all_stations()}
    assert set(sta) == {"1MR", "5WJ"}
    assert sta["1MR"]["status"] == "Inactive"  # last value 1949
    assert sta["5WJ"]["period_of_record"] == "2025-03-01/2025-04-01"
    for s in sta.values():
        assert {
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "status",
        } <= set(s)
    assert [s["station_id"] for s in c.get_all_stations(bbox=(9, 46, 10, 47))] == [
        "5WJ"
    ]


def test_gcos_records(tmp_path):
    c = _client("gcos", tmp_path)
    recs = c.get_data("5WJ", include_flags=True)
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["interval"] == "periodic"
        assert "flag" in r
    by = {(r["date"], r["variable"]): r["value"] for r in recs}
    assert by[("2025-03-01", "SWE")] == 61.2
    assert by[("2025-03-01", "HS")] == 180.0
    assert ("2025-03-15", "SWE") not in by  # blank = not observed
    assert by[("2025-04-01", "HS")] is None  # clamped
    assert by[("2025-04-01", "SWE")] == 70.0
    only = c.get_data("5WJ", variables="swe", end_date="2025-03-01")
    assert [(r["date"], r["variable"]) for r in only] == [("2025-03-01", "SWE")]


def test_gcos_metadata(tmp_path):
    c = _client("gcos", tmp_path)
    md = c.get_metadata("1MR")
    dv = {d["name"]: d for d in md["data_variables"]}
    assert dv["SWE"]["begin_date"] == "1949-02-01"
    assert dv["SWE"]["n_obs"] == 2
    assert dv["SWE"]["units"] == "cm"
    with pytest.raises(SLFError):
        c.get_metadata("XXX")


def test_cache_is_reused(tmp_path):
    c = _client("gcos", tmp_path)
    c.get_data("5WJ")
    c2 = SLFClient("gcos", cache_dir=tmp_path, session=_StubSession(), max_retries=1)
    c2._gcos_base = lambda: "unused"
    assert c2.get_data("5WJ")  # served from cache_dir/gcos/SWE_GCOS.csv
    assert (tmp_path / "gcos" / "SWE_GCOS.csv").exists()


# ── new_snow ─────────────────────────────────────────────────────────────────


def test_new_snow_stations_and_records(tmp_path):
    c = _client("new_snow", tmp_path)
    sta = {s["station_id"]: s for s in c.get_all_stations()}
    assert sta["5WJ"]["latitude"] == pytest.approx(46.8296, abs=1e-3)
    assert sta["5WJ"]["has_swe"] and not sta["1AD"]["has_swe"]
    assert sta["5WJ"]["status"] == "Inactive"
    recs = c.get_data("5WJ")
    got = {(r["date"], r["variable"], r["interval"]): r["value"] for r in recs}
    assert got == {
        ("2017-01-03", "SWE", "periodic"): 10.15,
        ("2017-01-02", "HS", "daily"): 40.0,
        ("2017-01-03", "HS", "daily"): 41.0,
    }
    hnw = c.get_data("1AD", variables="HNW_1D")
    assert [(r["date"], r["value"], r["type"]) for r in hnw] == [
        ("2017-01-02", 0.0, "other")
    ]


# ── study_plot / imis ────────────────────────────────────────────────────────


def test_study_plot(tmp_path):
    c = _client("study_plot", tmp_path)
    assert len(c.get_all_stations()) == 2
    assert [s["station_id"] for s in c.get_all_stations(active_only=True)] == ["5WJ0"]
    recs = c.get_data("5WJ0", variables=["HS", "HN_1D", "HNW_1D"])
    got = {(r["date"], r["variable"]): r["value"] for r in recs}
    assert got[("2026-10-09", "HNW_1D")] == 8.0
    assert ("2026-10-10", "HNW_1D") not in got
    assert ("2026-10-11", "HS") not in got
    assert all(r["interval"] == "daily" for r in recs)
    md = c.get_metadata("5WJ0")
    assert md["period_of_record"] == "2026-10-09/2026-10-11"
    assert {d["name"] for d in md["data_variables"]} == {"HS", "HN_1D", "HNW_1D"}


def test_imis_daily_and_sub_hourly(tmp_path):
    c = _client("imis", tmp_path)
    assert [s["station_id"] for s in c.get_all_stations()] == ["WFJ2"]  # WIND dropped
    daily = c.get_data("WFJ2")
    assert [(r["date"], r["value"]) for r in daily] == [
        ("1992-10-01", 1.0),
        ("1992-10-02", None),  # negative sensor noise nulled
        ("1992-10-03", 12.0),
    ]
    sub = c.get_data("WFJ2", variables=["HS", "temp"], interval="sub_hourly")
    assert {r["datetime"] for r in sub} == {
        "2026-10-10T00:00:00Z",
        "2026-10-10T00:30:00Z",
    }
    assert all(RECORD_KEYS <= set(r) and r["interval"] == "sub_hourly" for r in sub)
    assert len([r for r in sub if r["variable"] == "HS"]) == 1
    with pytest.raises(SLFError):
        c.get_data("WFJ2", variables="HN_1D", interval="sub_hourly")


def test_base_url_override(tmp_path):
    sess = _StubSession()
    client = SLFClient(
        "study_plot",
        cache_dir=tmp_path,
        base_url="https://mirror.example/slf/",
        session=sess,
        max_retries=1,
    )
    assert client.get_all_stations()
    assert sess.calls == ["https://mirror.example/slf/study-plot/stations.csv"]


# ── Live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_gcos_weissfluhjoch(tmp_path):
    c = SLFClient("gcos", cache_dir=tmp_path)
    stations = c.get_all_stations()
    assert len(stations) == 11
    recs = c.get_data(
        "5WJ", variables="SWE", begin_date="1949-02-01", end_date="1949-03-31"
    )
    assert recs and all(r["units"] == "cm" for r in recs)


@pytest.mark.live
def test_live_new_snow_zip(tmp_path):
    c = SLFClient("new_snow", cache_dir=tmp_path)
    assert len(c.get_all_stations()) == 41
    recs = c.get_data("5WJ", variables="SWE", end_date="2017-03-01")
    assert recs and recs[0]["interval"] == "periodic"


@pytest.mark.live
def test_live_study_plot_recent(tmp_path):
    c = SLFClient("study_plot", cache_dir=tmp_path)
    assert len(c.get_all_stations(active_only=True)) >= 80
    recs = c.get_data("5WJ0", begin_date="1936-09-01", end_date="1936-12-31")
    assert recs and recs[0]["variable"] == "HS"
