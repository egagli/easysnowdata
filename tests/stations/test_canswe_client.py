"""
Tests for CanSWEClient (and the shared CanSWE-format reader).

Offline tests write a tiny synthetic netCDF with the real CanSWE layout
(``station_id`` × ``time``, S1 flag arrays, WMO ``type_mes``) and read it
back through the client; the Zenodo lookup goes through a stub session.
The ``live`` tests resolve and download the real latest CanSWE (14 MB).
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.canswe import CanSWEClient, CanSWEError
from easysnowdata.stations.clients.canswe._format import (
    CanSWEFormat,
    convert,
    max_window_count,
    season_of,
)
from easysnowdata.stations.clients.canswe.canswe_client import (
    DATA_FLAGS,
    METHODS,
    VARIABLES,
)
from tests.stations.conftest import RECORD_KEYS

IDS = ["BCE-1A01P", "BCE-1A01", "HQ-GMON1", "SCD-AL001"]


def write_canswe_like(path, ids=IDS, start="2024-08-01", days=300, extra=None):
    """A CanSWE-layout file: one pillow, one survey, one GMON, one old survey."""
    times = pd.date_range(start, periods=days, freq="D")
    n, t = len(ids), len(times)
    snw = np.full((n, t), np.nan, dtype="float32")
    snd = np.full((n, t), np.nan, dtype="float32")
    flag_snw = np.full((n, t), b"", dtype="S1")
    qc_snw = np.full((n, t), b"", dtype="S1")
    flag_snd = np.full((n, t), b"", dtype="S1")
    qc_snd = np.full((n, t), b"", dtype="S1")
    # Pillow: daily SWE rising 0..299 mm, depth 0..2.99 m.
    snw[0] = np.arange(t, dtype="float32")
    snd[0] = np.arange(t, dtype="float32") / 100.0
    snw[0, 5] = np.nan
    qc_snw[0, 5] = b"V"  # outlier removed by QC
    snw[0, 6] = 99999.0  # implausible (> 15 m after conversion)
    # Survey: first of each month Jan–Jun 2025.
    for d in ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]:
        i = times.get_loc(pd.Timestamp(d))
        snw[1, i] = 250.0
        snd[1, i] = 0.8
    flag_snw[1, times.get_loc(pd.Timestamp("2025-03-01"))] = b"B"
    # GMON: SWE only, positive 40 days in a row.
    snw[2, 100:140] = 120.0
    # Old survey: nothing in this window (all NaN) — has no data.
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.where(snd > 0, snw / snd, np.nan).astype("float32")
    ds = xr.Dataset(
        {
            "lat": ("station_id", np.array([49.0, 49.1, 50.0, 51.0], "float32")),
            "lon": ("station_id", np.array([-117.0, -117.1, -70.0, -113.0], "float32")),
            "elevation": (
                "station_id",
                np.array([1950.0, 1900.0, np.nan, 1200.0], "float32"),
            ),
            "source": (
                "station_id",
                [
                    "British Columbia Environment",
                    "British Columbia Environment",
                    "Hydro-Quebec",
                    "Meteorological Service of Canada",
                ],
            ),
            "station_name": (
                "station_id",
                ["MISSION RIDGE", "MISSION", "GMON 1", "OLD"],
            ),
            "station_name_sec": ("station_id", ["MISSION RIDGE P", "", "", ""]),
            "station_name_ter": ("station_id", ["", "", "", ""]),
            "station_id_sec": ("station_id", ["SCD-BC001", "", "", ""]),
            "station_id_ter": ("station_id", ["", "", "", ""]),
            "type_mes": ("station_id", np.array([2, 0, 3, 0], "int64")),
            "snw": (("station_id", "time"), snw),
            "snd": (("station_id", "time"), snd),
            "den": (("station_id", "time"), den),
            "data_flag_snw": (("station_id", "time"), flag_snw),
            "data_flag_snd": (("station_id", "time"), flag_snd),
            "qc_flag_snw": (("station_id", "time"), qc_snw),
            "qc_flag_snd": (("station_id", "time"), qc_snd),
            **(extra or {}),
        },
        coords={"station_id": ids, "time": times},
        attrs={"title": "synthetic CanSWE"},
    )
    enc = {v: {"chunksizes": (2, 100), "zlib": True} for v in ("snw", "snd", "den")}
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)
    return path


@pytest.fixture
def nc_path(tmp_path):
    return write_canswe_like(tmp_path / "CanSWE-CanEEN_test.nc")


@pytest.fixture
def client(nc_path):
    return CanSWEClient(path=nc_path)


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


def test_snow_variables_emit_cm_and_density_percent():
    assert VARIABLES["snw"]["output_units"] == "cm"
    assert VARIABLES["snd"]["output_units"] == "cm"
    assert VARIABLES["den"]["output_units"] == "%"


def test_methods_map_to_shared_intervals():
    for code, info in METHODS.items():
        assert info["interval"] in INTERVALS, code
    assert METHODS[0]["interval"] == "periodic"
    assert METHODS[2]["interval"] == "daily"
    assert METHODS[3]["interval"] == "daily"


def test_data_flags_cover_agency_and_qc():
    assert DATA_FLAGS["agency:R"] == "Revised data"
    assert "qc:W" in DATA_FLAGS and "qc:V" in DATA_FLAGS


# ── Pure helpers ─────────────────────────────────────────────────────────────


def test_convert_units_and_clamp():
    assert convert("snw", 250.0) == pytest.approx(25.0)
    assert convert("snd", np.float32(0.58)) == pytest.approx(58.0)
    assert convert("den", 312.5) == pytest.approx(31.25)
    assert convert("snw", 99999.0) is None  # > 15 m
    assert convert("snd", -0.1) is None
    assert convert("snw", float("nan")) is None


def test_season_of_and_window_count():
    assert season_of(np.datetime64("2024-08-01")) == 2025
    assert season_of(np.datetime64("2025-07-31")) == 2025
    mask = np.zeros((2, 200), bool)
    mask[0, 10:50] = True
    mask[1, ::7] = True
    assert list(max_window_count(mask, 90)) == [40, 13]


# ── Reader / summary ─────────────────────────────────────────────────────────


def test_summary_is_cached_next_to_file(nc_path):
    reader = CanSWEFormat(nc_path)
    summ = reader.summary()
    assert summ["BCE-1A01P"]["daily_seasons"] == [2025]
    assert summ["BCE-1A01"]["snw"]["n"] == 4
    assert "snw" not in summ["SCD-AL001"]
    cache = nc_path.with_name(nc_path.name + ".summary.json")
    assert json.loads(cache.read_text())["stations"]["HQ-GMON1"]["max90_swe_pos"] == 40
    # A second reader uses the cache.
    assert CanSWEFormat(nc_path).summary() == summ


# ── Stations ─────────────────────────────────────────────────────────────────


def test_get_all_stations_schema(client):
    stations = client.get_all_stations()
    assert [s["station_id"] for s in stations] == IDS
    pillow = stations[0]
    for key in ("station_id", "name", "latitude", "longitude", "elevation_m", "status"):
        assert key in pillow
    assert pillow["status"] in ("Active", "Inactive")
    assert pillow["method_code"] == 2 and pillow["interval"] == "daily"
    assert pillow["has_daily_swe"] is True
    assert pillow["source"] == "British Columbia Environment"
    assert pillow["agency"] == "British Columbia Ministry of Environment"
    assert pillow["alt_ids"] == ["SCD-BC001"]
    survey = stations[1]
    assert survey["interval"] == "periodic" and survey["has_daily_swe"] is False
    assert survey["begin_date"] == "2025-01-01" and survey["end_date"] == "2025-04-01"
    gmon = stations[2]
    assert gmon["elevation_m"] is None and gmon["has_snwd"] is False
    assert stations[3]["has_swe"] is False


def test_station_filters(client):
    assert {s["station_id"] for s in client.get_all_stations(sources="hydro")} == {
        "HQ-GMON1"
    }
    assert {s["station_id"] for s in client.get_all_stations(methods="automated")} == {
        "BCE-1A01P",
        "HQ-GMON1",
    }
    assert [s["station_id"] for s in client.get_all_stations(methods=[0])] == [
        "BCE-1A01",
        "SCD-AL001",
    ]
    assert [
        s["station_id"] for s in client.get_all_stations(methods="passive_gamma")
    ] == ["HQ-GMON1"]
    assert len(client.get_all_stations(daily_only=True)) == 2
    assert [
        s["station_id"] for s in client.get_all_stations(bbox=(-118, 48, -116, 50))
    ] == ["BCE-1A01P", "BCE-1A01"]
    with pytest.raises(CanSWEError):
        client.get_all_stations(methods="dowsing")


def test_get_metadata(client):
    meta = client.get_metadata("BCE-1A01P")
    names = {v["name"] for v in meta["data_variables"]}
    assert names == {"snw", "snd", "den"}
    snw = next(v for v in meta["data_variables"] if v["name"] == "snw")
    assert snw["interval"] == "daily" and snw["units"] == "cm"
    assert snw["n_obs"] == 299  # one day QC-removed (NaN); 99999 is stored
    with pytest.raises(CanSWEError):
        client.get_metadata("NOPE")


# ── Data ─────────────────────────────────────────────────────────────────────


def test_get_data_records(client):
    recs = client.get_data("BCE-1A01P", begin_date="2024-08-03", end_date="2024-08-08")
    assert recs and all(set(r) == RECORD_KEYS for r in recs)
    swe = {r["date"]: r["value"] for r in recs if r["variable"] == "snw"}
    assert swe["2024-08-03"] == pytest.approx(0.2)  # 2 mm → 0.2 cm
    assert "2024-08-06" not in swe  # QC'd to NaN, not returned without flags
    assert swe["2024-08-07"] is None  # implausible → None
    depth = [r for r in recs if r["type"] == "snwd"]
    assert depth[0]["units"] == "cm" and depth[0]["interval"] == "daily"


def test_get_data_flags_include_qc_removed_days(client):
    recs = client.get_data(
        "BCE-1A01P",
        variables="swe",
        begin_date="2024-08-06",
        end_date="2024-08-06",
        include_flags=True,
    )
    assert recs == [
        {
            "station_id": "BCE-1A01P",
            "date": "2024-08-06",
            "variable": "snw",
            "type": "swe",
            "value": None,
            "units": "cm",
            "interval": "daily",
            "flag": ",V",
        }
    ]
    survey = client.get_data("BCE-1A01", variables="snw", include_flags=True)
    assert [r["flag"] for r in survey] == [",", ",", "B,", ","]
    assert {r["interval"] for r in survey} == {"periodic"}


def test_get_data_interval_filter(client):
    both = client.get_data(["BCE-1A01P", "BCE-1A01"], variables="swe")
    assert {r["station_id"] for r in both} == {"BCE-1A01P", "BCE-1A01"}
    daily = client.get_data(
        ["BCE-1A01P", "BCE-1A01"], variables="swe", interval="daily"
    )
    assert {r["station_id"] for r in daily} == {"BCE-1A01P"}
    periodic = client.get_data(["BCE-1A01P", "BCE-1A01"], interval="periodic")
    assert {r["station_id"] for r in periodic} == {"BCE-1A01"}
    with pytest.raises(CanSWEError):
        client.get_data("BCE-1A01P", interval="hourly")


def test_get_data_variables_and_bbox(client):
    recs = client.get_data(bbox=(-71, 49, -69, 51), variables=["density", "swe"])
    assert {r["station_id"] for r in recs} == {"HQ-GMON1"}
    assert {r["variable"] for r in recs} == {"snw"}  # no depth → no density
    with pytest.raises(CanSWEError):
        client.get_data("HQ-GMON1", variables="temp")
    with pytest.raises(ValueError):
        client.get_data()
    assert client.get_data("UNKNOWN-1") == []


# ── Zenodo resolution ────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, payload=None, status=200):
        self.headers = {}
        self.payload, self.status = payload, status
        self.urls = []

    def request(self, method, url, params=None, timeout=None):
        self.urls.append(url)
        return _Resp(self.payload or {}, self.status)


def test_resolves_latest_version_and_uses_cached_file(tmp_path, nc_path):
    cached = tmp_path / "cache" / "CanSWE-CanEEN_1928-2099_v99.nc"
    cached.parent.mkdir()
    cached.write_bytes(nc_path.read_bytes())
    record = {
        "id": 99,
        "files": [
            {"key": "CanSWE-CanEEN_1928-2099_v99.zip", "size": 1},
            {
                "key": cached.name,
                "size": cached.stat().st_size,
                "checksum": "md5:x",
                "links": {"self": "https://example.invalid/f"},
            },
        ],
    }
    session = _StubSession(record)
    c = CanSWEClient(cache_dir=tmp_path / "cache", session=session, max_retries=1)
    assert c.dataset_path() == cached
    assert session.urls == ["https://zenodo.org/api/records/4734371/versions/latest"]
    assert len(c.get_all_stations()) == len(IDS)


def test_pinned_record_and_offline_fallback(tmp_path, nc_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "old.nc").write_bytes(nc_path.read_bytes())
    session = _StubSession({"message": "down"}, status=404)
    c = CanSWEClient(
        record_id="19075529", cache_dir=cache, session=session, max_retries=1
    )
    assert c.dataset_path() == cache / "old.nc"
    assert session.urls == ["https://zenodo.org/api/records/19075529"]
    empty = CanSWEClient(cache_dir=tmp_path / "empty", session=session, max_retries=1)
    with pytest.raises(CanSWEError):
        empty.dataset_path()


# ── Live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_latest_canswe(tmp_path_factory):
    c = CanSWEClient()
    stations = c.get_all_stations()
    assert len(stations) > 2900
    pillows = c.get_all_stations(sources="British Columbia", methods="snow_pillow")
    assert len(pillows) > 100
    recs = c.get_data(
        "BCE-1A01P", variables="swe", begin_date="2024-03-01", end_date="2024-03-31"
    )
    assert len(recs) >= 28
    assert all(r["interval"] == "daily" and r["units"] == "cm" for r in recs)
