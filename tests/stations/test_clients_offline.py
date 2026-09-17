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
