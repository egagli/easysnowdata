"""
Tests for SouthernAndesClient.

Offline tests read small synthetic versions of the two Zenodo CSVs (the
real header layout and BOM, the real metadata rows of four stations,
made-up daily values, and the record's quoted junk cells) from a temporary
folder, and replay the Zenodo record lookup and
downloads with stubs.  ``live`` tests read the real Zenodo record (no
credentials).
"""

import pytest

from easysnowdata.stations.clients import SouthernAndesClient, SouthernAndesError
from easysnowdata.stations.clients._common import INTERVALS, TYPES
from easysnowdata.stations.clients.southern_andes import (
    southern_andes_client as sac,
)
from easysnowdata.stations.networks import canonical_units
from tests.stations.conftest import RECORD_KEYS

DATA_NAME = "Southern_Andes_Snow_Depth_Dataset_v4.2.csv"
META_NAME = "Station_metadata_for_snow_depth_observations.csv"

# The real files start with a UTF-8 byte-order mark.
DATA_CSV = (
    "﻿date,2000002,5703009,Capayan,1150003\n"
    '2019-10-31,",",,0.00,\n'
    '2019-12-04,,,",0.45",\n'
    "2022-07-01,,45.10,12.50,3.00\n"
    "2022-07-02,,46.25,,1700.00\n"
    "2023-07-01,10.00,120.00,30.00,\n"
    "2024-12-31,0.00,0.00,,\n"
)
META_CSV = (
    "﻿code_internal,name,lat,lon,elevation,basin,source,Snow Depth Sensor,"
    "start date,end date,potential days with data,effective days with data\n"
    "2000002,Volcán Auncanquilcha,-21.1906,-68.4528,5087,"
    "Fronterizas Salar Michincha - R. Loa,DGA,"
    "Campbell/SR50A was replaced by Lufft/SHM31,2021/12/01,2024/12/31,1127,1062 (94%)\n"
    "5703009,Laguna Negra,-33.6659,-70.1077,2785,Río Maipo,DGA,"
    "Campbell/SR50A was replaced by Lufft/SHM31,2013/04/10,2024/12/31,4284,3841 (90%)\n"
    "Capayan,Capayán,-30.1734,-69.8038,4754,Río Jáchal,IANIGLA,Campbell/SR50A,"
    "2016/12/20,2024/12/31,2934,2889 (98%)\n"
    "1150003,Steffen,-47.5555,-73.6958,30,Río Baker,CIEP,Campbell/SR50A,"
    "2022/01/01,2022/10/09,282,240 (85%)\n"
)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    (d / DATA_NAME).write_text(DATA_CSV, encoding="utf-8")
    (d / META_NAME).write_text(META_CSV, encoding="utf-8")
    return d


@pytest.fixture
def client(folder):
    return SouthernAndesClient(path=folder)


def _record(record_id="21577171"):
    def entry(key, text):
        return {
            "key": key,
            "size": len(text.encode()),
            "checksum": None,
            "links": {"self": f"https://zenodo.test/{record_id}/{key}"},
        }

    return {
        "id": record_id,
        "files": [
            entry(DATA_NAME, DATA_CSV),
            entry(META_NAME, META_CSV),
            {"key": "Figure_1.png", "size": 1, "links": {}},
        ],
    }


@pytest.fixture
def zenodo(monkeypatch):
    """Stub the record lookup and the downloads; count the downloads."""
    calls = {"resolve": 0, "download": []}
    texts = {DATA_NAME: DATA_CSV, META_NAME: META_CSV}

    def resolve(self):
        calls["resolve"] += 1
        self.record = _record(self.record_id or "21577171")
        return self.record

    def download(session, url, dest, **kwargs):
        calls["download"].append(url)
        dest.write_text(texts[dest.name], encoding="utf-8")
        return dest

    monkeypatch.setattr(SouthernAndesClient, "resolve_record", resolve)
    monkeypatch.setattr(sac, "download_file", download)
    return calls


# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_follows_the_contract():
    assert set(sac.VARIABLES) == {"SD"}
    for info in sac.VARIABLES.values():
        assert info["type"] in TYPES
        assert info["output_units"] == canonical_units(info["type"]) == "cm"
        assert {"units", "description", "notes", "source"} <= set(info)
    assert sac.DATA_FLAGS == {}
    assert set(sac.NETWORKS) == {"DGA", "IANIGLA", "CEAZA", "CIEP"}
    assert "daily" in INTERVALS


def test_resolve_variables():
    assert sac._resolve_variables(None) == ["SD"]
    assert sac._resolve_variables("snwd") == ["SD"]
    assert sac._resolve_variables(["sd", "SD"]) == ["SD"]
    with pytest.raises(SouthernAndesError, match="no SWE"):
        sac._resolve_variables("swe")


def test_convert_nulls_junk_and_implausible():
    assert sac._convert("45.10") == pytest.approx(45.1)
    assert sac._convert("0.00") == 0.0
    assert sac._convert("") is None
    assert sac._convert(",0.45") is None
    assert sac._convert("1700.00") is None  # > 15 m
    assert sac._convert("-1") is None


# ── Parsing ─────────────────────────────────────────────────────────────────


def test_parse_data_skips_quoted_junk_cells():
    ids, cols = sac.parse_data(DATA_CSV)
    assert ids == ["2000002", "5703009", "Capayan", "1150003"]
    # 2019-10-31 holds only '","' for 2000002: not an observation.
    assert cols["2000002"] == [("2023-07-01", "10.00"), ("2024-12-31", "0.00")]
    assert ("2019-12-04", ",0.45") not in cols["Capayan"]
    assert cols["Capayan"][0] == ("2019-10-31", "0.00")


def test_parse_data_rejects_other_layouts():
    with pytest.raises(SouthernAndesError, match="date"):
        sac.parse_data("station,2000002\n")
    with pytest.raises(SouthernAndesError, match="empty"):
        sac.parse_data("")


def test_parse_metadata_fields():
    rows = {r["station_id"]: r for r in sac.parse_metadata(META_CSV)}
    r = rows["2000002"]
    assert r["name"] == "Volcán Auncanquilcha"
    assert (r["latitude"], r["longitude"], r["elevation_m"]) == (
        -21.1906,
        -68.4528,
        5087.0,
    )
    assert r["begin_date"] == "2021-12-01" and r["end_date"] == "2024-12-31"
    assert (r["potential_days"], r["effective_days"]) == (1127, 1062)
    assert r["network"] == "DGA" and r["operator"].startswith("Dirección General")
    assert r["country"] == "CL" and rows["Capayan"]["country"] == "AR"
    assert r["station_url"] == sac.NETWORK_URLS["DGA"]
    assert r["has_swe"] is False and r["has_snwd"] is True


# ── Stations ────────────────────────────────────────────────────────────────


def test_get_all_stations(client):
    stations = {s["station_id"]: s for s in client.get_all_stations()}
    assert list(stations) == ["2000002", "5703009", "Capayan", "1150003"]
    for s in stations.values():
        assert {
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation_m",
            "status",
        } <= set(s)
    # Active = reported in the file's final year (2024).
    assert stations["2000002"]["status"] == "Active"
    assert stations["Capayan"]["status"] == "Inactive"  # last value 2023
    assert stations["2000002"]["first_obs"] == "2023-07-01"
    assert stations["5703009"]["n_snwd"] == 4
    assert stations["1150003"]["n_snwd"] == 2  # the 17 m value is still a cell


def test_get_all_stations_filters(client):
    assert [s["station_id"] for s in client.get_all_stations(networks="ianigla")] == [
        "Capayan"
    ]
    assert {s["station_id"] for s in client.get_all_stations(active_only=True)} == {
        "2000002",
        "5703009",
    }
    south = client.get_all_stations(bbox=(-75, -50, -65, -40))
    assert [s["station_id"] for s in south] == ["1150003"]
    with pytest.raises(SouthernAndesError, match="Unknown network"):
        client.get_all_stations(networks="UdeChile")


def test_status_goes_inactive_when_the_dataset_is_old(client, monkeypatch):
    class _Today:
        @staticmethod
        def today():
            import datetime

            return datetime.date(2030, 1, 1)

    monkeypatch.setattr(sac, "date", _Today)
    assert {s["status"] for s in client.get_all_stations()} == {"Inactive"}


def test_get_metadata(client):
    meta = client.get_metadata("5703009")
    (dv,) = meta["data_variables"]
    assert dv["name"] == "SD" and dv["interval"] == "daily" and dv["units"] == "cm"
    assert (dv["begin_date"], dv["end_date"], dv["n_obs"]) == (
        "2022-07-01",
        "2024-12-31",
        4,
    )
    with pytest.raises(SouthernAndesError, match="not in"):
        client.get_metadata("9999999")


# ── Data ────────────────────────────────────────────────────────────────────


def test_get_data_records(client):
    recs = client.get_data("5703009", include_flags=True)
    assert [r["date"] for r in recs] == [
        "2022-07-01",
        "2022-07-02",
        "2023-07-01",
        "2024-12-31",
    ]
    for r in recs:
        assert RECORD_KEYS | {"flag"} == set(r)
        assert (r["variable"], r["type"], r["units"], r["interval"]) == (
            "SD",
            "snwd",
            "cm",
            "daily",
        )
        assert r["flag"] is None
    assert recs[1]["value"] == pytest.approx(46.25)


def test_get_data_bounds_bbox_and_clamp(client):
    recs = client.get_data(
        ["5703009", "1150003"], begin_date="2022-07-02", end_date="2022-07-02"
    )
    assert {(r["station_id"], r["value"]) for r in recs} == {
        ("5703009", 46.25),
        ("1150003", None),  # 17 m is implausible
    }
    north = client.get_data(bbox=(-70, -25, -65, -20))
    assert {r["station_id"] for r in north} == {"2000002"}


def test_get_data_errors(client):
    with pytest.raises(ValueError):
        client.get_data()
    with pytest.raises(SouthernAndesError, match="Unsupported interval"):
        client.get_data("5703009", interval="hourly")
    with pytest.raises(SouthernAndesError, match="Unknown"):
        client.get_data("5703009", variables="SWE")
    with pytest.raises(SouthernAndesError, match="not in"):
        client.get_data("nope")


def test_bad_path_raises(tmp_path):
    with pytest.raises(SouthernAndesError, match="folder holding"):
        SouthernAndesClient(path=tmp_path).get_all_stations()


# ── Zenodo resolution and cache ─────────────────────────────────────────────


def test_downloads_once_into_a_record_folder(tmp_path, zenodo):
    client = SouthernAndesClient(cache_dir=tmp_path)
    assert len(client.get_all_stations()) == 4
    assert len(zenodo["download"]) == 2
    assert (tmp_path / "21577171" / DATA_NAME).exists()
    # A new client with the same cache re-resolves but does not re-download.
    SouthernAndesClient(cache_dir=tmp_path).get_all_stations()
    assert zenodo["resolve"] == 2 and len(zenodo["download"]) == 2


def test_pinned_record_uses_its_own_folder(tmp_path, zenodo):
    SouthernAndesClient(cache_dir=tmp_path, record_id="19819613").get_all_stations()
    assert (tmp_path / "19819613" / META_NAME).exists()


def test_falls_back_to_the_cache_when_zenodo_is_down(tmp_path, zenodo, monkeypatch):
    SouthernAndesClient(cache_dir=tmp_path).get_all_stations()

    def down(self):
        raise SouthernAndesError("HTTP 503")

    monkeypatch.setattr(SouthernAndesClient, "resolve_record", down)
    assert len(SouthernAndesClient(cache_dir=tmp_path).get_all_stations()) == 4
    with pytest.raises(SouthernAndesError, match="503"):
        SouthernAndesClient(cache_dir=tmp_path / "empty").get_all_stations()


def test_record_without_the_csvs_raises(tmp_path, monkeypatch):
    def resolve(self):
        return {"id": "1", "files": [{"key": "Figure_1.png"}]}

    monkeypatch.setattr(SouthernAndesClient, "resolve_record", resolve)
    with pytest.raises(SouthernAndesError, match="lacks the expected CSVs"):
        SouthernAndesClient(cache_dir=tmp_path).get_all_stations()


# ── Live ────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_latest_record(tmp_path):
    client = SouthernAndesClient(cache_dir=tmp_path)
    stations = client.get_all_stations()
    assert len(stations) >= 69
    assert {s["network"] for s in stations} >= {"DGA", "IANIGLA", "CEAZA", "CIEP"}
    # The data file agrees with the metadata file's own day counts.
    assert all(s["n_snwd"] == s["effective_days"] for s in stations)
    recs = client.get_data("5703009", begin_date="2023-06-01", end_date="2023-09-30")
    assert len(recs) > 100 and max(r["value"] for r in recs) > 50
