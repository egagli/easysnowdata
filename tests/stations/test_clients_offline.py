"""
Offline unit tests for client-contract behaviour — no network access.

Covers the DESIGN.md §3 contract points fixed in the July 2026
unification: input validation (no silent fallbacks), metric conversion,
timestamp preservation on sub-daily records, and CDEC SWE sensor
priority.  The second half applies the same contract checks to the
clients added in October 2026 (GHCN-D, Hydro-Québec, Alberta, DWD, Syke,
SLF, RIHMI) through their private resolve/convert helpers, or through
clients whose network methods are stubbed.  CanSWE and NorSWE are left
out while their storage layer is rewritten.  Live-API integration tests
live in the per-client test files (marked ``live``).
"""

import io
import zipfile
from datetime import datetime

import pytest

from easysnowdata.stations.clients.alberta import AlbertaClient, AlbertaError
from easysnowdata.stations.clients.alberta import alberta_client as alberta
from easysnowdata.stations.clients.awdb import AWDBClient, AWDBError
from easysnowdata.stations.clients.awdb.awdb_client import (
    _METRIC_CONVERSIONS,
    _resolve_variables_to_awdb,
)
from easysnowdata.stations.clients.awdb.awdb_client import (
    DATA_FLAGS as AWDB_DATA_FLAGS,
)
from easysnowdata.stations.clients.cdec import CDECClient, CDECError
from easysnowdata.stations.clients.cdec.cdec_client import (
    _normalise_cdec_date,
    _resolve_variables_to_cdec_sensors,
)
from easysnowdata.stations.clients.ceaza import CEAZAClient, CEAZAError
from easysnowdata.stations.clients.ceaza import ceaza_client as ceaza
from easysnowdata.stations.clients.databc import DataBCClient, DataBCError
from easysnowdata.stations.clients.dwd import DWDClient, DWDError
from easysnowdata.stations.clients.dwd import dwd_client as dwd
from easysnowdata.stations.clients.ghcnd import GHCNDClient, GHCNDError
from easysnowdata.stations.clients.ghcnd import ghcnd_client as ghcnd
from easysnowdata.stations.clients.hydroquebec import (
    HydroQuebecClient,
    HydroQuebecError,
)
from easysnowdata.stations.clients.hydroquebec import hydroquebec_client as hq
from easysnowdata.stations.clients.nve import NVEError
from easysnowdata.stations.clients.nve.nve_client import (
    _resolve_variables as nve_resolve,
)
from easysnowdata.stations.clients.rihmi import RIHMIClient, RIHMIError
from easysnowdata.stations.clients.rihmi import rihmi_client as rihmi
from easysnowdata.stations.clients.slf import SLFClient, SLFError
from easysnowdata.stations.clients.slf import slf_client as slf
from easysnowdata.stations.clients.snih import SNIHClient, SNIHError
from easysnowdata.stations.clients.snih import snih_client as snih
from easysnowdata.stations.clients.southern_andes import (
    SouthernAndesClient,
    SouthernAndesError,
)
from easysnowdata.stations.clients.southern_andes import (
    southern_andes_client as southern_andes,
)
from easysnowdata.stations.clients.syke import SYKEClient, SYKEError
from easysnowdata.stations.clients.syke import syke_client as syke
from easysnowdata.stations.clients.yukon import YukonClient, YukonError
from easysnowdata.stations.clients.yukon.yukon_client import (
    _resolve_variables as yukon_resolve,
)

# ── Variable resolution: unknown names raise, never fall back ────────────────


def test_awdb_resolves_types_and_codes():
    assert _resolve_variables_to_awdb(["swe"]) == ["WTEQ"]
    # TAVG leads: the daily mean wins over the instantaneous reading when a
    # consumer flattens both into one `temp` series.
    assert _resolve_variables_to_awdb(["temp"]) == ["TAVG", "TOBS", "TMAX", "TMIN"]
    assert _resolve_variables_to_awdb(None)  # all variables


def test_awdb_unknown_variable_raises():
    with pytest.raises(AWDBError, match="Unknown variable"):
        _resolve_variables_to_awdb(["SMS"])


def test_cdec_resolves_swe_priority_order():
    assert _resolve_variables_to_cdec_sensors(["swe"]) == [82, 3]
    assert _resolve_variables_to_cdec_sensors(None) == [3, 18, 82]


def test_cdec_unknown_variable_raises():
    with pytest.raises(CDECError, match="Unknown variable"):
        _resolve_variables_to_cdec_sensors(["bogus"])


def test_nve_unknown_variable_raises():
    with pytest.raises(NVEError, match="Unknown variable"):
        nve_resolve(["swe_mm"])  # the stale docs' name — must not resolve


def test_nve_resolves_swe():
    jobs = nve_resolve(["swe"])
    assert [(j[0], j[1]) for j in jobs] == [("swe_m", 2003)]
    assert jobs[0][2](1.5) == pytest.approx(150.0)  # m → cm


def test_yukon_unknown_variable_raises():
    with pytest.raises(YukonError, match="Unknown variable"):
        yukon_resolve(["bogus"])


# ── Interval validation: unsupported intervals raise ─────────────────────────


def test_awdb_unsupported_interval_raises():
    client = AWDBClient()
    with pytest.raises(AWDBError, match="Unsupported interval"):
        client.get_data(station_ids=["303:CO:SNTL"], interval="weekly")


def test_cdec_unsupported_interval_raises():
    client = CDECClient()
    with pytest.raises(CDECError, match="Unsupported interval"):
        client.get_data(station_ids=["QUA"], interval="annual")


def test_databc_unsupported_interval_raises():
    client = DataBCClient()
    with pytest.raises(DataBCError, match="Unsupported interval"):
        client.get_data(station_ids=["1A01P"], interval="monthly")


def test_databc_asws_archive_load_failure_raises(monkeypatch):
    client = DataBCClient()

    def load(url, **kwargs):
        if url == "archive.csv":
            raise DataBCError("HTTP 503")
        import pandas as pd

        return pd.DataFrame(
            {"date": ["2026-10-22"], "location_id": ["1A01P"], "snwd_cm": [10.0]}
        )

    monkeypatch.setattr(client, "_load_asws_wide_csv", load)
    with pytest.raises(DataBCError, match="HTTP 503"):
        client._get_asws_var_data(
            "current.csv", "archive.csv", "snwd_cm", True, None, None, None
        )


@pytest.mark.parametrize("url", ["current.csv", "archive.csv"])
def test_databc_asws_file_load_failure_raises(monkeypatch, url):
    client = DataBCClient()

    def load(requested_url, **kwargs):
        if requested_url == url:
            raise DataBCError("HTTP 503")
        import pandas as pd

        return pd.DataFrame(
            {"date": ["2026-10-22"], "location_id": ["1A01P"], "snwd_cm": [10.0]}
        )

    monkeypatch.setattr(client, "_load_asws_wide_csv", load)
    with pytest.raises(DataBCError, match="HTTP 503"):
        client._get_asws_var_data(
            "current.csv", "archive.csv", "snwd_cm", True, None, None, None
        )


def test_databc_snowall_file_load_failure_raises(monkeypatch):
    client = DataBCClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda url, **kwargs: (_ for _ in ()).throw(DataBCError("HTTP 503")),
    )
    with pytest.raises(DataBCError, match="HTTP 503"):
        client._get_asws_combined_data("1A01P")


@pytest.mark.parametrize("url", ["allmss_current.csv", "allmss_archive.csv"])
def test_databc_manual_survey_file_load_failure_raises(monkeypatch, url):
    client = DataBCClient()

    def load(requested_url):
        if requested_url.endswith(url):
            raise DataBCError("HTTP 503")
        import pandas as pd

        return pd.DataFrame({"location_id": ["1A01P"], "date": ["2026-10-22"]})

    monkeypatch.setattr(client, "_load_mss_csv", load)
    with pytest.raises(DataBCError, match="HTTP 503"):
        client._get_mss_survey_data()


def test_yukon_unsupported_interval_raises():
    client = YukonClient()
    with pytest.raises(YukonError, match="Unsupported interval"):
        client.get_data(station_ids=["09AA-M1"], interval="monthly")


def test_get_data_requires_ids_or_bbox():
    for client, err in (
        (AWDBClient(), ValueError),
        (CDECClient(), ValueError),
        (DataBCClient(), ValueError),
        (YukonClient(), ValueError),
    ):
        with pytest.raises(err):
            client.get_data()


# ── AWDB metric conversion (DESIGN.md §3.5) ──────────────────────────────────


def test_awdb_metric_transforms():
    transform, unit = _METRIC_CONVERSIONS["TOBS"]
    assert transform(32.0) == pytest.approx(0.0)
    assert unit == "°C"
    transform, unit = _METRIC_CONVERSIONS["PREC"]
    assert transform(1.0) == pytest.approx(25.4)
    assert unit == "mm"
    transform, unit = _METRIC_CONVERSIONS["WSPDV"]
    assert transform(10.0) == pytest.approx(16.09344)
    assert unit == "km/h"
    transform, unit = _METRIC_CONVERSIONS["WTEQ"]
    assert transform(10.0) == pytest.approx(25.4)
    assert unit == "cm"


def test_awdb_get_data_emits_metric_and_hourly_datetime():
    client = AWDBClient()
    raw = [
        {
            "stationTriplet": "303:CO:SNTL",
            "data": [
                {
                    "stationElement": {
                        "elementCode": "TOBS",
                        "durationName": "HOURLY",
                        "originalUnitCode": "degF",
                    },
                    "values": [{"date": "2024-01-01 05:00", "value": 32.0}],
                },
                {
                    "stationElement": {
                        "elementCode": "WTEQ",
                        "durationName": "HOURLY",
                        "originalUnitCode": "in",
                    },
                    "values": [{"date": "2024-01-01 05:00", "value": 10.0}],
                },
            ],
        }
    ]

    def fake_fetch(ids, elements, duration, begin, end):
        import copy

        blocks = copy.deepcopy(raw)
        client._convert_data_response_to_metric(blocks)
        return blocks

    client._get_data_awdb = fake_fetch
    records = client.get_data(
        station_ids=["303:CO:SNTL"],
        variables=["temp", "swe"],
        interval="hourly",
    )
    by_var = {r["variable"]: r for r in records}
    assert by_var["TOBS"]["value"] == pytest.approx(0.0)
    assert by_var["TOBS"]["units"] == "°C"
    assert by_var["WTEQ"]["value"] == pytest.approx(25.4)
    assert by_var["WTEQ"]["units"] == "cm"
    assert by_var["TOBS"]["datetime"] == "2024-01-01 05:00"


def test_awdb_data_flags_importable_and_empty():
    # The documented import pattern must work for every client; AWDB has
    # no per-value flags so its registry is deliberately empty.
    assert AWDB_DATA_FLAGS == {}


# ── CDEC timestamp preservation and SWE priority ─────────────────────────────


def test_cdec_normalise_date_preserves_time():
    assert _normalise_cdec_date("2023-1-1 16:00") == "2023-01-01 16:00"
    assert _normalise_cdec_date("2023-1-1") == "2023-01-01"


def _cdec_client_with(raw):
    client = CDECClient()
    client._get_data_cdec = lambda *a, **k: raw
    return client


def test_cdec_sensor_82_beats_sensor_3():
    raw = [
        {
            "stationId": "QUA",
            "data": [
                {
                    "stationElement": {"sensorNum": 3, "durationCode": "D"},
                    "values": [{"date": "2024-01-01 00:00", "value": 10.0}],
                },
                {
                    "stationElement": {"sensorNum": 82, "durationCode": "D"},
                    "values": [{"date": "2024-01-01 00:00", "value": 11.0}],
                },
            ],
        }
    ]
    records = _cdec_client_with(raw).get_data(
        station_ids=["QUA"],
        variables=["swe"],
        interval="daily",
    )
    assert len(records) == 1
    assert records[0]["value"] == 11.0
    assert records[0]["variable"] == "SNO ADJ"
    assert records[0]["date"] == "2024-01-01"
    assert "datetime" not in records[0]


def test_cdec_hourly_keeps_all_timestamps():
    raw = [
        {
            "stationId": "QUA",
            "data": [
                {
                    "stationElement": {"sensorNum": 3, "durationCode": "H"},
                    "values": [
                        {"date": f"2024-01-01 {h:02d}:00", "value": float(h)}
                        for h in range(24)
                    ],
                },
            ],
        }
    ]
    records = _cdec_client_with(raw).get_data(
        station_ids=["QUA"],
        variables=["swe"],
        interval="hourly",
    )
    # Before the fix, 24 hourly SWE readings collapsed to the last one.
    assert len(records) == 24
    assert records[5]["datetime"] == "2024-01-01 05:00"
    assert records[5]["interval"] == "hourly"
    assert records[5]["date"] == "2024-01-01"


# ── DataBC negative-value filter scoping ─────────────────────────────────────


def test_databc_negative_filter_spares_air_temperature():
    import pandas as pd

    client = DataBCClient()

    class FakeResp:
        text = (
            "DATE(UTC),1A01P Yellowhead Lake\n"
            "2024-01-01 16:00,-12.5\n"
            "2024-01-02 16:00,-99999\n"
            "2024-01-03 16:00,3.0\n"
        )

    client._request = lambda url, **k: FakeResp()
    df = client._load_asws_wide_csv("fake://ta.csv", value_col="air_temp_degc")
    vals = df["air_temp_degc"].tolist()
    assert vals[0] == pytest.approx(-12.5)  # sub-zero temp is real data
    assert pd.isna(vals[1])  # sentinel nulled
    assert vals[2] == pytest.approx(3.0)

    df_swe = client._load_asws_wide_csv("fake://sw.csv", value_col="swe_mm")
    assert pd.isna(df_swe["swe_mm"].tolist()[0])  # negative SWE nulled


# ── the shared type vocabulary (global_snow_networks' DESIGN.md §3.2) ────────

#: Every client module checked by the vocabulary test.  CanSWE and NorSWE
#: are left out while their storage layer is rewritten.
_ALL_CLIENT_MODULES = (
    "awdb",
    "cdec",
    "databc",
    "nve",
    "yukon",
    "ghcnd",
    "hydroquebec",
    "alberta",
    "dwd",
    "syke",
    "slf",
    "rihmi",
    "southern_andes",
    "snih",
    "ceaza",
)


def test_every_client_uses_only_the_shared_type_vocabulary():
    """No client may invent a standardized type of its own.

    The vocabulary used to be restated inside two test files, which is how
    `soil_moisture` came to be missing from one of them while four CDEC
    sensors used it. It lives in `_common.py` now, like INTERVALS.
    """
    import importlib

    from easysnowdata.stations.clients._common import TYPES

    offenders = {}
    for name in _ALL_CLIENT_MODULES:
        module = importlib.import_module(
            f"easysnowdata.stations.clients.{name}.{name}_client"
        )
        registry = dict(
            getattr(module, "VARIABLES", None) or getattr(module, "SENSORS", {})
        )
        registry.update(getattr(module, "AREAL_VARIABLES", {}))  # Syke
        assert registry, name
        bad = {k: v["type"] for k, v in registry.items() if v["type"] not in TYPES}
        if bad:
            offenders[name] = bad
    assert offenders == {}, offenders


def test_snowfall_is_not_filed_under_precipitation():
    """A depth of snow is not a depth of water (DESIGN.md §3.2).

    Yukon's `precip_snow_cm` is the only snowfall series any client carries;
    typing it `precip` invited consumers to rescale 5 cm of snow into 50 mm
    of water.
    """
    from easysnowdata.stations.clients.yukon.yukon_client import (
        _TYPE_TO_YUKON_VARS,
        VARIABLES,
    )

    assert VARIABLES["precip_snow_cm"]["type"] == "snowfall"
    assert VARIABLES["precip_snow_cm"]["output_units"] == "cm"
    assert "precip_snow_cm" not in _TYPE_TO_YUKON_VARS["precip"]
    assert _TYPE_TO_YUKON_VARS["snowfall"] == ["precip_snow_cm"]
    # every `precip` series really is a depth of water, in mm
    for key in _TYPE_TO_YUKON_VARS["precip"]:
        assert VARIABLES[key]["output_units"] == "mm"


# ── AWDB: a rejected triplet must not poison its batch ───────────────────────


def test_awdb_get_metadata_bisects_a_rejected_batch_down_to_the_triplet(
    monkeypatch, caplog
):
    """AWDB answers HTTP 400 for a whole 150-station batch when one triplet
    is malformed (two USGS gauges with hyphenated ids do this). The client
    bisects to the offender, skips it with a warning, and returns the rest."""
    client = AWDBClient()
    bad = "NV-16:CA:USGS"
    requested: list[list[str]] = []

    def fake_get(endpoint, params=None, **kwargs):
        batch = params["stationTriplets"].split(",")
        requested.append(batch)
        if bad in batch:
            raise AWDBError("HTTP 400 Bad Request: .../stations (params=...)")
        return [
            {"stationTriplet": t, "stationElements": [{"elementCode": "WTEQ"}]}
            for t in batch
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    triplets = [f"{n}:CO:SNTL" for n in range(300, 310)] + [bad] + ["713:CO:SNTL"]
    result = client.get_metadata(triplets, elements=["WTEQ"], durations=["DAILY"])

    assert [m["stationTriplet"] for m in result] == [t for t in triplets if t != bad]
    assert [bad] in requested  # bisected all the way down to the offender
    assert f"rejects station triplet {bad}" in caplog.text


def test_awdb_get_metadata_for_one_rejected_triplet_still_raises(monkeypatch):
    """Skipping is for batches; a caller who asked for that one station gets
    the error."""
    client = AWDBClient()

    def fake_get(endpoint, params=None, **kwargs):
        raise AWDBError("HTTP 400 Bad Request")

    monkeypatch.setattr(client, "_get", fake_get)
    with pytest.raises(AWDBError, match="400"):
        client.get_metadata("NV-16:CA:USGS")


# ══ Clients added in October 2026 ════════════════════════════════════════════
#
# Same contract points as above, for GHCN-D, Hydro-Québec, Alberta, DWD,
# Syke, SLF, RIHMI, the Southern Andes dataset, SNIH and CEAZA.  Every constructor here is network-free; methods
# that would download are stubbed on the instance.


_NEW_CLIENTS = (
    (GHCNDClient, GHCNDError, "USS0021D08S", "hourly"),
    (HydroQuebecClient, HydroQuebecError, "1-10613", "periodic"),
    (AlbertaClient, AlbertaError, "05BB803", "monthly"),
    (DWDClient, DWDError, "00020", "hourly"),
    (SYKEClient, SYKEError, "1", "daily"),
    (RIHMIClient, RIHMIError, "27612", "hourly"),
    (SouthernAndesClient, SouthernAndesError, "5703009", "periodic"),
    (SNIHClient, SNIHError, "19145", "hourly"),
    (CEAZAClient, CEAZAError, "TPF", "periodic"),
)


@pytest.mark.parametrize(
    "cls,err,sid,bad_interval", _NEW_CLIENTS, ids=lambda x: getattr(x, "__name__", "")
)
def test_new_clients_unsupported_interval_raises(cls, err, sid, bad_interval):
    with pytest.raises(err, match="(?i)unsupported interval"):
        cls().get_data(station_ids=[sid], interval=bad_interval)


@pytest.mark.parametrize(
    "cls,err,sid,bad_interval", _NEW_CLIENTS, ids=lambda x: getattr(x, "__name__", "")
)
def test_new_clients_unknown_variable_raises(cls, err, sid, bad_interval):
    with pytest.raises(err, match="(?i)unknown"):
        cls().get_data(station_ids=[sid], variables=["bogus"])


@pytest.mark.parametrize("cls", [c[0] for c in _NEW_CLIENTS], ids=lambda c: c.__name__)
def test_new_clients_get_data_requires_ids_or_bbox(cls):
    with pytest.raises(ValueError):
        cls().get_data()


def test_slf_unsupported_interval_and_ids_or_bbox(tmp_path):
    client = SLFClient(dataset="gcos", cache_dir=tmp_path)
    with pytest.raises(SLFError, match="Unsupported interval"):
        client.get_data(station_ids=["5WJ"], interval="hourly")
    with pytest.raises(SLFError, match="Unknown variable"):
        client.get_data(station_ids=["5WJ"], variables=["bogus"])
    with pytest.raises(ValueError):
        client.get_data()


def test_new_clients_data_flags_importable():
    # The documented `from <module> import VARIABLES, DATA_FLAGS` pattern.
    assert hq.DATA_FLAGS == {}
    assert alberta.DATA_FLAGS == {}
    assert slf.DATA_FLAGS == {}
    assert southern_andes.DATA_FLAGS == {}
    assert "W" in ghcnd.DATA_FLAGS  # QFLAG "too warm for snow"
    assert dwd.DATA_FLAGS["10"].startswith("Quality control finished")
    assert set(syke.DATA_FLAGS) == {"96", "110", "117"}
    assert "NSIDC:3" in rihmi.DATA_FLAGS and "Q1:9" in rihmi.DATA_FLAGS


def test_new_clients_snow_output_units_are_cm():
    """DESIGN.md §3.5: SWE and snow depth are emitted in cm everywhere."""
    registries = {
        "ghcnd": ghcnd.VARIABLES,
        "hydroquebec": hq.VARIABLES,
        "alberta": alberta.VARIABLES,
        "dwd": dwd.VARIABLES,
        "syke": {**syke.VARIABLES, **syke.AREAL_VARIABLES},
        "slf": slf.VARIABLES,
        "rihmi": rihmi.VARIABLES,
        "southern_andes": southern_andes.VARIABLES,
        "snih": snih.VARIABLES,
        "ceaza": ceaza.VARIABLES,
    }
    bad = {
        (name, key): info["output_units"]
        for name, reg in registries.items()
        for key, info in reg.items()
        if info["type"] in ("swe", "snwd") and info["output_units"] != "cm"
    }
    assert bad == {}


# ── GHCN-D ───────────────────────────────────────────────────────────────────


def test_ghcnd_resolves_types_and_defaults():
    assert ghcnd._resolve_variables(None) == ["WESD", "SNWD"]
    assert ghcnd._resolve_variables(["swe"]) == ["WESD"]
    assert ghcnd._resolve_variables("snowfall") == ["SNOW"]
    # WESF is a snowfall water equivalent, never total precipitation.
    assert ghcnd._resolve_variables("precip") == ["PRCP"]


def test_ghcnd_metric_conversion_and_clamps():
    assert ghcnd._convert("WESD", 1000) == pytest.approx(10.0)  # 0.1 mm → cm
    assert ghcnd._convert("SNWD", 250) == pytest.approx(25.0)  # mm → cm
    assert ghcnd._convert("TMAX", -125) == pytest.approx(-12.5)  # 0.1 °C → °C
    assert ghcnd._convert("PRCP", 37) == pytest.approx(3.7)  # 0.1 mm → mm
    assert ghcnd._convert("SNWD", -9999) is None
    assert ghcnd._convert("SNWD", 16000) is None  # 16 m of snow
    assert ghcnd._convert("SNWD", -10) is None


def test_ghcnd_qc_failed_values_are_nulled_unless_kept():
    lines = [
        "USS0021D08S,20240101,WESD,2540,,,T,",
        "USS0021D08S,20240102,WESD,2600,,X,T,",
        "USS0021D08S,20240103,TMAX,10,,,T,",
        "USS0021D08S,20231231,WESD,2500,,,T,",
    ]
    rows = list(ghcnd._iter_rows(lines, {"WESD"}, "2024-01-01", None))
    assert [r[1] for r in rows] == ["2024-01-01", "2024-01-02"]

    strict = GHCNDClient()
    recs = [strict._record(*r, include_flags=True) for r in rows]
    assert recs[0]["value"] == pytest.approx(25.4)
    assert recs[0]["units"] == "cm"
    assert recs[0]["flag"] == ",,T"
    assert recs[1]["value"] is None  # QFLAG X
    assert recs[1]["flag"] == ",X,T"

    lenient = GHCNDClient(keep_qc_failed=True)
    assert lenient._record(*rows[1], include_flags=False)["value"] == pytest.approx(
        26.0
    )


# ── Hydro-Québec ─────────────────────────────────────────────────────────────


def test_hydroquebec_resolves_types_and_defaults():
    assert hq._resolve_variables(None) == [hq._SWE, hq._SNWD]
    assert hq._resolve_variables("swe") == [hq._SWE]
    assert hq._resolve_variables(["temp_max"]) == ["Température Maximum"]
    with pytest.raises(HydroQuebecError, match="Unknown variable"):
        hq._resolve_variables(["swe_mm"])


def test_hydroquebec_conversion_and_sentinels():
    assert hq._convert(hq._SWE, "123.0") == pytest.approx(12.3)  # mm → cm
    assert hq._convert(hq._SNWD, "45") == pytest.approx(45.0)
    assert hq._convert(hq._SWE, "-999.00") is None  # missing sentinel
    assert hq._convert(hq._SNWD, "-0.3") is None  # negative noise
    assert hq._convert("Précipitation", "-1") is None
    assert hq._convert("Humidité relative.2 mètres", "104") is None


def _hq_rows(var, values, start="2026-01-10T05:00:00+00:00", measure="Instantanée"):
    t0 = datetime.fromisoformat(start)
    return [
        {
            "identifiant": "1-10613",
            hq._F_VAR: var,
            hq._F_KIND: measure,
            "date": t0.replace(hour=(t0.hour + i) % 24).isoformat()
            if t0.hour + i < 24
            else t0.replace(day=t0.day + 1, hour=(t0.hour + i) % 24).isoformat(),
            "valeur": v,
        }
        for i, v in enumerate(values)
    ]


def test_hydroquebec_hourly_records_carry_utc_datetime():
    rows = _hq_rows(hq._SWE, ["100", "110"])
    recs = hq._hourly_records(rows, None, None, include_flags=True)
    assert [r["datetime"] for r in recs] == [
        "2026-01-10T05:00:00Z",
        "2026-01-10T06:00:00Z",
    ]
    assert recs[1]["value"] == pytest.approx(11.0)
    assert recs[1]["units"] == "cm"
    assert recs[1]["interval"] == "hourly"
    assert recs[1]["flag"] == "Instantanée"


def test_hydroquebec_daily_aggregation_on_local_day():
    # 05:00 UTC .. 04:00 UTC next day = 00:00 .. 23:00 EST on 2026-01-10.
    swe = _hq_rows(hq._SWE, [str(100 + i) for i in range(24)])
    rain = _hq_rows("Précipitation", ["0.5"] * 24, measure="Somme")
    short = _hq_rows("Précipitation", ["1"] * (hq.MIN_HOURS_DAILY - 1))
    for r in short:
        r["identifiant"] = "1-99999"
    recs = hq._daily_records(swe + rain + short, None, None, include_flags=True)
    by = {(r["station_id"], r["variable"]): r for r in recs}
    assert all(r["date"] == "2026-01-10" for r in recs)
    assert by[("1-10613", hq._SWE)]["value"] == pytest.approx(12.3)  # last value
    assert by[("1-10613", "Précipitation")]["value"] == pytest.approx(12.0)  # sum
    assert by[("1-99999", "Précipitation")]["value"] is None  # too few hours
    assert by[("1-10613", hq._SWE)]["flag"] is None
    assert "datetime" not in by[("1-10613", hq._SWE)]


def test_hydroquebec_circular_mean_wind_direction():
    # 350° and 20° average to 5°, not to the arithmetic 185°.
    assert hq._aggregate("circmean", [350.0, 20.0] * 12) == pytest.approx(5.0)


# ── Alberta ──────────────────────────────────────────────────────────────────


def test_alberta_resolves_types_per_interval():
    assert alberta._resolve_variables(None, "daily") == ["SW", "SD"]
    assert alberta._resolve_variables(None, "periodic") == ["SW-SS", "SD-SS"]
    assert alberta._resolve_variables("swe", "daily") == ["SW"]
    assert alberta._resolve_variables("swe", "periodic") == ["SW-SS"]
    with pytest.raises(AlbertaError, match="not available at interval"):
        alberta._resolve_variables("PC", "hourly")  # cumulative gauge
    with pytest.raises(AlbertaError, match="No Alberta variable of type"):
        alberta._resolve_variables("precip", "periodic")
    with pytest.raises(AlbertaError, match="Unknown variable"):
        alberta._resolve_variables("bogus", "daily")


def test_alberta_conversion_and_clamps():
    assert alberta._convert("SW", 250.0) == pytest.approx(25.0)  # mm → cm
    assert alberta._convert("SD", 80.0) == pytest.approx(80.0)
    assert alberta._convert("SW", -3.0) is None  # pillow noise
    assert alberta._convert("TA", -12.5) == pytest.approx(-12.5)


def test_alberta_hourly_keeps_on_the_hour_samples_in_mst():
    client = AlbertaClient()
    client._series_cache = {
        "05BB803": {"SW": {"tsID": "123", "TsMinDataAsCurrentTime": None}}
    }
    client.download_series = lambda ts, b, e: (
        {},
        [
            (datetime(2024, 1, 1, 5, 0), 100.0),
            (datetime(2024, 1, 1, 5, 15), 101.0),
            (datetime(2024, 1, 1, 6, 0), 102.0),
        ],
    )
    recs = client.get_data(
        station_ids="05BB803",
        variables="swe",
        interval="hourly",
        begin_date="2024-01-01",
        end_date="2024-01-01",
    )
    assert [r["datetime"] for r in recs] == [
        "2024-01-01T05:00:00-07:00",
        "2024-01-01T06:00:00-07:00",
    ]
    assert recs[0]["value"] == pytest.approx(10.0)
    assert recs[0]["units"] == "cm"
    assert recs[0]["interval"] == "hourly"


def test_alberta_client_daily_mean_needs_twenty_hours():
    rows = [(datetime(2024, 1, 1, h), 1.0) for h in range(19)]
    rows += [(datetime(2024, 1, 2, h), float(h)) for h in range(24)]
    out = alberta._daily_means(rows, "TA")
    assert [d.date().isoformat() for d, _ in out] == ["2024-01-02"]
    assert out[0][1] == pytest.approx(11.5)


# ── DWD ──────────────────────────────────────────────────────────────────────


def test_dwd_resolves_types_and_defaults():
    assert dwd._resolve_variables(None) == ["WASH_6", "SH_TAG"]
    assert dwd._resolve_variables("swe") == ["WASH_6"]  # not WAAS_6
    assert dwd._resolve_variables(["snowfall", "precip"]) == ["NSH_TAG", "RS"]
    with pytest.raises(DWDError, match="Unknown variable"):
        dwd._resolve_variables("other")  # too broad to be a request


def test_dwd_station_id_padding():
    assert dwd._pad_id(20) == "00020"
    assert dwd._pad_id(" 1048 ") == "01048"
    with pytest.raises(DWDError, match="Invalid DWD station id"):
        dwd._pad_id("Zugspitze")


def test_dwd_produkt_parse_convert_and_merge():
    text = (
        "STATIONS_ID;MESS_DATUM;QN_6;ASH_6;SH_TAG;WASH_6;WAAS_6;eor\n"
        "   20;20250101;    5;   12;   12;  34.0;  34.0;eor\n"
        "   20;20250102;    5;-999;   10;-999;-999;eor\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("produkt_wa_tag_20250101_20250102_00020.txt", text)
    cols, rows = dwd._parse_produkt(buf.getvalue())
    assert "WASH_6" in cols
    assert rows["2025-01-01"]["WASH_6"] == pytest.approx(34.0)
    assert rows["2025-01-02"]["WASH_6"] is None  # -999
    assert rows["2025-01-01"]["QN_6"] == "5"

    assert dwd._convert("WASH_6", 34.0) == pytest.approx(3.4)  # mm → cm
    assert dwd._convert("SH_TAG", 1600.0) is None  # 16 m

    recs = DWDClient._records("00020", "WASH_6", rows, include_flags=True)
    assert len(recs) == 1  # the missing day is not emitted
    assert recs[0]["value"] == pytest.approx(3.4)
    assert recs[0]["units"] == "cm"
    assert recs[0]["flag"] == "5"

    merged = dwd._merge({"2025-01-01": {"v": "hist"}}, {"2025-01-01": {"v": "recent"}})
    assert merged["2025-01-01"]["v"] == "hist"  # QC-finished file wins


def test_dwd_unknown_product_raises():
    with pytest.raises(DWDError, match="Unknown DWD product"):
        DWDClient().get_stations(products="hourly")


# ── Syke ─────────────────────────────────────────────────────────────────────


def test_syke_resolves_swe_only():
    assert syke._resolve_variables(None) == ["S"]
    assert syke._resolve_variables(["swe", "s"]) == ["S"]
    with pytest.raises(SYKEError, match="Snow depth and density are not published"):
        syke._resolve_variables("snwd")


def test_syke_station_ids_must_be_integers():
    with pytest.raises(SYKEError, match="integers"):
        SYKEClient().get_data(station_ids=["Kilpisjärvi"])


def test_syke_conversion_and_modelled_flag():
    assert syke._convert("123") == pytest.approx(12.3)  # mm → cm
    assert syke._convert("16000") is None
    row = {"Paikka_Id": 42, "Aika": "2024-03-15T00:00:00", "Arvo": 150, "Lippu_Id": 117}
    rec = SYKEClient()._record(row, "S", "periodic", include_flags=True)
    assert rec["value"] is None  # modelled, nulled by default
    assert rec["flag"] == "117"
    assert rec["date"] == "2024-03-15"
    kept = SYKEClient(keep_modelled=True)._record(row, "S", "periodic", False)
    assert kept["value"] == pytest.approx(15.0)
    assert "flag" not in kept


def test_syke_odata_filter_chunks_stay_under_node_limit():
    flt = syke._build_filter(["1", "2"], "2024-01-01", None)
    assert flt == (
        "(Paikka_Id eq 1 or Paikka_Id eq 2) and Aika ge datetime'2024-01-01T00:00:00'"
    )
    assert syke._ID_CHUNK <= 18


# ── SLF ──────────────────────────────────────────────────────────────────────


def test_slf_unknown_dataset_raises(tmp_path):
    with pytest.raises(SLFError, match="Unknown SLF dataset"):
        SLFClient(dataset="osshd", cache_dir=tmp_path)


def test_slf_resolves_per_dataset_and_interval():
    assert slf._resolve_variables("gcos", None, None) == [
        ("SWE", "periodic"),
        ("HS", "periodic"),
    ]
    assert slf._resolve_variables("new_snow", "snwd", None) == [
        ("HS", "daily"),
        ("HS_PROFILE", "periodic"),
    ]
    assert slf._resolve_variables("imis", "HS", "sub_hourly") == [("HS", "sub_hourly")]
    with pytest.raises(SLFError, match="Unsupported interval"):
        slf._resolve_variables("gcos", None, "daily")
    with pytest.raises(SLFError, match="not part of SLF dataset"):
        slf._resolve_variables("imis", "SWE", None)  # IMIS measures no SWE
    with pytest.raises(SLFError, match="not available at interval"):
        slf._resolve_variables("imis", "HN_1D", "sub_hourly")


def test_slf_conversion_and_lv03():
    assert slf._convert("SWE", "250") == pytest.approx(25.0)  # mm → cm
    assert slf._convert("HS", "-2") is None  # IMIS bare-ground noise
    assert slf._convert("HN_1D", "600") is None  # > 5 m of new snow
    assert slf._convert("HNW_1D", "12.5") == pytest.approx(12.5)  # stays mm
    assert slf._convert("SWE", "-9999") is None
    lat, lon = slf.lv03_to_wgs84(600000.0, 200000.0)  # Bern origin
    assert lat == pytest.approx(46.95108, abs=1e-4)
    assert lon == pytest.approx(7.43864, abs=1e-4)


# ── RIHMI ────────────────────────────────────────────────────────────────────


def test_rihmi_resolves_per_interval():
    assert rihmi._resolve_variables(None, "periodic") == ["SWE", "HS_MEAN"]
    assert rihmi._resolve_variables(None, "daily") == ["H"]
    assert rihmi._resolve_variables("density", "periodic") == ["DENSITY"]
    with pytest.raises(RIHMIError, match="belongs to interval"):
        rihmi._resolve_variables("H", "periodic")
    with pytest.raises(RIHMIError, match="Unknown route type"):
        rihmi._resolve_routes(["tundra"])


def test_rihmi_conversion_and_coordinates():
    assert rihmi._convert("SWE", 120.0) == pytest.approx(12.0)  # mm → cm
    assert rihmi._convert("DENSITY", 0.25) == pytest.approx(25.0)  # g/cm3 → %
    assert rihmi._convert("COVER_ROUTE", 7.0) == pytest.approx(70.0)  # tenths → %
    assert rihmi._convert("HS_MEAN", 1600.0) is None
    assert rihmi._num("9999") is None
    assert rihmi._nsidc_num("-1") is None  # undocumented NSIDC missing code
    assert rihmi._ddmm("5545") == pytest.approx(55.75)  # Moscow, DDMM
    assert rihmi._ddmm("3734") == pytest.approx(37.5667, abs=1e-4)


def test_rihmi_parses_both_export_layouts():
    route = "27612;2002;3;1;10;10;10;0;45;60;30;0.25;0;0;0;112;112;1;4"
    cover = "27612 2002 03 10   45 10 0 0 0"
    rows = rihmi.parse_aisori_text("header line\n" + route + "\n" + cover + "\n")
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["route_survey"]["route_type"] == "field"
    assert by_kind["route_survey"]["date"] == "2002-03-10"
    assert by_kind["route_survey"]["values"]["SWE"] == pytest.approx(112.0)
    assert by_kind["snow_cover"]["values"]["H"] == pytest.approx(45.0)
    assert by_kind["snow_cover"]["flag"] == "Q1:0,Q2:0,Q3:0"


def test_rihmi_daily_without_exports_raises():
    client = RIHMIClient(use_nsidc=False)
    with pytest.raises(RIHMIError, match="only available from"):
        client.get_data(station_ids="27612", interval="daily")


def test_rihmi_local_export_records(tmp_path):
    export = tmp_path / "routes.txt"
    export.write_text(
        "27612;2002;3;2;10;10;10;0;45;60;30;0.25;0;0;0;112;112;1;4\n"
        "27612;2002;3;1;10;10;10;0;40;55;28;0.24;0;0;0;96;96;1;4\n",
        encoding="utf-8",
    )
    client = RIHMIClient(data_dir=tmp_path, use_nsidc=False)
    recs = client.get_data(station_ids=27612, variables="swe", include_flags=True)
    # Field route first, so a first-value pivot prefers it.
    assert [r["route_type"] for r in recs] == ["field", "forest"]
    assert recs[0]["value"] == pytest.approx(9.6)
    assert recs[0]["units"] == "cm"
    assert recs[0]["source"] == "aisori_export"
    assert recs[0]["flag"] is None
    assert [
        r["route_type"]
        for r in client.get_data(
            station_ids="27612", variables="swe", route_types="forest"
        )
    ] == ["forest"]


def test_temperature_output_units_are_degrees_celsius():
    """One spelling of the unit across clients, so consumers can group on it."""
    import importlib

    bad = {}
    for name in _ALL_CLIENT_MODULES:
        module = importlib.import_module(
            f"easysnowdata.stations.clients.{name}.{name}_client"
        )
        registry = getattr(module, "VARIABLES", None) or getattr(module, "SENSORS", {})
        for key, info in registry.items():
            if info["type"] in ("temp", "temp_max", "temp_min"):
                unit = info.get("output_units", info.get("units"))
                if unit != "°C":
                    bad[(name, key)] = unit
    assert bad == {}, bad
