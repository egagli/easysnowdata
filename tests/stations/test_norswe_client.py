"""
Tests for NorSWEClient.

Offline tests write a tiny synthetic netCDF with the NorSWE layout (the
CanSWE layout without the secondary name/id fields, plus ``mmask``) and
read it back; the zip download path is exercised with a local zip.  The
``live`` test resolves the latest record on Zenodo (metadata only — the
data file is 2.4 GB).
"""

import zipfile

import numpy as np
import pytest
import xarray as xr

from easysnowdata.stations.clients._common import TYPES
from easysnowdata.stations.clients.norswe import NorSWEClient, NorSWEError
from easysnowdata.stations.clients.norswe.norswe_client import (
    DATA_FLAGS,
    ID_PREFIXES,
    VARIABLES,
)
from tests.stations.conftest import RECORD_KEYS
from tests.stations.test_canswe_client import write_canswe_like

IDS = ["SNOTEL-679", "SYKE-1201", "RIHMI-27612_2", "NOHRSC-AB101"]
FLAG_VARS = ["data_flag_snw", "data_flag_snd", "qc_flag_snw", "qc_flag_snd"]


def write_norswe_like(path):
    """The real NorSWE layout: time × station data, vlen-string flags,
    ``mmask``, type_mes 64 for airborne gamma, no secondary names/ids."""
    write_canswe_like(path, ids=IDS)
    ds = xr.load_dataset(path, engine="h5netcdf")
    ds = ds.drop_vars(
        ["station_name_sec", "station_name_ter", "station_id_sec", "station_id_ter"]
    )
    ds["source"] = (
        "station_id",
        ["Snotel US network", "Finland SYKE", "RIHMI-WDC", "NOAA-NOHRSC"],
    )
    ds["lat"] = ("station_id", np.array([46.8, 63.0, 55.8, 48.5], "float32"))
    ds["lon"] = ("station_id", np.array([-121.7, 27.0, 37.6, -97.0], "float32"))
    ds["mmask"] = ("station_id", np.array([1, 0, 0, 0], "uint8"))
    ds["type_mes"] = ("station_id", np.array([2, 0, 0, 64], "uint8"))
    for v in FLAG_VARS:
        ds[v] = ds[v].astype(str).astype(object)
    ds["data_flag_snw"][3, 150] = "AM-avg2"
    ds["snw"][3, 150] = 85.0
    ds = ds.transpose("time", "station_id")
    path.unlink()
    enc = {v: {"dtype": str} for v in FLAG_VARS}
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)
    return path


@pytest.fixture
def nc_path(tmp_path):
    return write_norswe_like(tmp_path / "NorSWE-NorEEN_test.nc")


@pytest.fixture
def client(nc_path):
    return NorSWEClient(path=nc_path)


def test_registries():
    for key, info in VARIABLES.items():
        assert info["type"] in TYPES, key
        assert "NorSWE" in info["source"]
    assert "agency:AI" in DATA_FLAGS
    assert ID_PREFIXES["RIHMI"][1] == "RU" and ID_PREFIXES["SYKE"][1] == "FI"


def test_station_extras_and_filters(client):
    stations = {s["station_id"]: s for s in client.get_all_stations()}
    assert stations["SNOTEL-679"]["mmask"] == 1
    assert stations["SNOTEL-679"]["country"] == "US"
    assert stations["SNOTEL-679"]["interval"] == "daily"
    assert stations["RIHMI-27612_2"]["rihmi_landscape"] == "forest"
    assert stations["RIHMI-27612_2"]["country"] == "RU"
    gamma = stations["NOHRSC-AB101"]
    assert gamma["airborne_gamma"] is True and gamma["method_code"] == 64
    assert gamma["interval"] == "periodic"
    assert [s["station_id"] for s in client.get_all_stations(countries="ru")] == [
        "RIHMI-27612_2"
    ]
    assert [s["station_id"] for s in client.get_all_stations(prefixes="NOHRSC")] == [
        "NOHRSC-AB101"
    ]
    assert [s["station_id"] for s in client.get_all_stations(mountain_only=True)] == [
        "SNOTEL-679"
    ]
    assert [
        s["station_id"] for s in client.get_all_stations(methods="airborne_gamma")
    ] == ["NOHRSC-AB101"]
    assert [s["station_id"] for s in client.get_all_stations(sources="SYKE")] == [
        "SYKE-1201"
    ]


def test_get_data_and_metadata(client):
    recs = client.get_data("SNOTEL-679", begin_date="2024-08-03", end_date="2024-08-04")
    assert recs and all(set(r) == RECORD_KEYS for r in recs)
    assert {r["interval"] for r in recs} == {"daily"}
    swe = [r for r in recs if r["type"] == "swe"]
    assert [r["value"] for r in swe] == [pytest.approx(0.2), pytest.approx(0.3)]
    assert client.get_metadata("SYKE-1201")["data_variables"][0]["interval"] == (
        "periodic"
    )
    with pytest.raises(NorSWEError):
        client.get_data("SYKE-1201", interval="monthly")


def test_vlen_string_flags_on_time_by_station_layout(client):
    recs = client.get_data(
        "NOHRSC-AB101",
        variables="swe",
        begin_date="2024-12-29",
        end_date="2024-12-29",
        include_flags=True,
    )
    assert recs[0]["flag"] == "AM-avg2,"
    assert recs[0]["value"] == pytest.approx(8.5)
    qc = client.get_data(
        "SNOTEL-679",
        variables="snw",
        begin_date="2024-08-06",
        end_date="2024-08-06",
        include_flags=True,
    )
    assert qc[0]["value"] is None and qc[0]["flag"] == ",V"
    with pytest.raises(NorSWEError):
        client.get_data("SYKE-1201", variables="precip")


class _Resp:
    def __init__(self, payload):
        self._payload, self.ok, self.status_code, self.text = payload, True, 200, ""

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.headers, self.payload = {}, payload

    def request(self, method, url, params=None, timeout=None):
        return _Resp(self.payload)


def test_zip_is_downloaded_extracted_and_reused(tmp_path, nc_path, monkeypatch):
    cache = tmp_path / "cache"
    zpath = tmp_path / "src.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(nc_path, "NorSWE-NorEEN_1979-2021_v9.nc")
    record = {
        "id": 1,
        "files": [
            {"key": "Readme.pdf", "size": 1},
            {"key": "NorSWE-NorEEN_1979-2021_v9.zip", "size": zpath.stat().st_size},
        ],
    }
    calls = []

    def fake_retrieve(url, known_hash, fname, path, downloader=None):
        calls.append(fname)
        dest = path / fname
        dest.write_bytes(zpath.read_bytes())
        return str(dest)

    import pooch

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    c = NorSWEClient(cache_dir=cache, session=_Session(record), max_retries=1)
    path = c.dataset_path()
    assert (
        path == cache / "NorSWE-NorEEN_1979-2021_v9" / "NorSWE-NorEEN_1979-2021_v9.nc"
    )
    assert not (cache / "NorSWE-NorEEN_1979-2021_v9.zip").exists()  # zip removed
    assert len(c.get_all_stations()) == 4
    # A second client finds the extracted file and does not download again.
    again = NorSWEClient(cache_dir=cache, session=_Session(record), max_retries=1)
    assert again.dataset_path() == path
    assert calls == ["NorSWE-NorEEN_1979-2021_v9.zip"]


@pytest.mark.live
def test_live_latest_record_resolves():
    c = NorSWEClient()
    record = c.resolve_record()
    info = c._pick_file(record)
    assert info["key"].startswith("NorSWE") and info["key"].endswith(".zip")
    assert int(info["size"]) > 1_000_000_000
