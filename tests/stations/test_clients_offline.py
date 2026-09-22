"""
Offline unit tests for client-contract behaviour — no network access.

Covers the DESIGN.md §3 contract points fixed in the July 2026
unification: input validation (no silent fallbacks), metric conversion,
timestamp preservation on sub-daily records, and CDEC SWE sensor
priority.  Live-API integration tests live in the per-client test files
(marked ``live``).
"""

import pytest

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
from easysnowdata.stations.clients.databc import DataBCClient, DataBCError
from easysnowdata.stations.clients.nve import NVEError
from easysnowdata.stations.clients.nve.nve_client import (
    _resolve_variables as nve_resolve,
)
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


def test_every_client_uses_only_the_shared_type_vocabulary():
    """No client may invent a standardized type of its own.

    The vocabulary used to be restated inside two test files, which is how
    `soil_moisture` came to be missing from one of them while four CDEC
    sensors used it. It lives in `_common.py` now, like INTERVALS.
    """
    import importlib

    from easysnowdata.stations.clients._common import TYPES

    offenders = {}
    for name in ("awdb", "cdec", "databc", "nve", "yukon"):
        module = importlib.import_module(
            f"easysnowdata.stations.clients.{name}.{name}_client"
        )
        registry = getattr(module, "VARIABLES", None) or getattr(module, "SENSORS", {})
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
