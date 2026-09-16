"""The adapter's orchestration, with the network clients stubbed out.

``test_adapter.py`` covers the pure conversions and the live smoke tests.
This file covers what sits between them — which network each station is sent
to, what each client is asked for, and how the answers are stitched back
together — by replacing the five client classes with stubs that record their
calls. No sockets, so the offline tier exercises the fan-out that otherwise
only the live tier would reach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import easysnowdata as esd
from easysnowdata.stations import _adapter, networks

from .conftest import PARADISE_CODE, PARADISE_TRIPLET, RAINIER

STATION_LISTS = {
    "awdb": [
        {
            "stationTriplet": PARADISE_TRIPLET,
            "stationId": "679",
            "name": "Paradise",
            "latitude": 46.78,
            "longitude": -121.75,
            "elevation_m": 1563.6,
            "stateCode": "WA",
            "networkCode": "SNTL",
            "status": "Active",
        },
        {
            "stationTriplet": "642:WA:SNTL",
            "stationId": "642",
            "name": "Morse Lake",
            "latitude": 46.93,
            "longitude": -121.39,
            "elevation_m": 1646.0,
            "stateCode": "WA",
            "networkCode": "SNTL",
            "status": "Inactive",
        },
    ],
    "cdec": [
        {
            "station_id": "QUA",
            "name": "QUARTZ BASIN",
            "latitude": 38.4,
            "longitude": -119.8,
            "elevation_m": 2400.0,
            "status": "Active",
            "has_daily_swe": True,
            "has_daily_snwd": False,
        },
        {
            "station_id": "HNT",
            "name": "HUNTINGTON LAKE",
            "latitude": 37.2,
            "longitude": -119.2,
            "elevation_m": 2134.0,
            "status": "Active",
            "has_daily_swe": False,
            "has_daily_snwd": False,
        },
    ],
    "databc": [
        {
            "location_id": "1A01P",
            "name": "Tupper",
            "latitude": 55.0,
            "longitude": -120.0,
            "elevation_m": 1200.0,
            "status": "Active",
            "station_type": "ASWS",
        },
        {
            "location_id": "1A06A",
            "name": "Tupper Creek",
            "latitude": 55.1,
            "longitude": -120.1,
            "elevation_m": 1100.0,
            "status": "Active",
            "station_type": "MSS",
        },
    ],
    "nve": [
        {
            "station_id": "12.142.0",
            "name": "Bakko",
            "latitude": 59.8,
            "longitude": 7.5,
            "elevation_m": 1100.0,
            "status": "Active",
            "daily_parameters": [2002, 2003],
        },
        {
            "station_id": "1.15.0",
            "name": "Elvegard",
            "latitude": 59.0,
            "longitude": 8.0,
            "elevation_m": 300.0,
            "status": "Inactive",
            "daily_parameters": [1001],
        },
    ],
    "yukon": [
        {
            "station_id": "09AA-M1",
            "name": "Tagish",
            "latitude": 60.3,
            "longitude": -134.3,
            "elevation_m": 700.0,
            "status": "Active",
            "series": [{"interval": "daily"}],
        },
        {
            "station_id": "08AA-SC01",
            "name": "Canyon Lake",
            "latitude": 60.6,
            "longitude": -135.0,
            "elevation_m": 900.0,
            "status": "Active",
            "series": [{"interval": "periodic"}],
        },
    ],
}


class StubClient:
    """Stands in for one network client, recording what it was asked."""

    network = "awdb"
    calls: list[dict]

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    def get_all_stations(self, active_only=False, bbox=None):
        type(self).calls.append(
            {"method": "get_all_stations", "active_only": active_only, "bbox": bbox}
        )
        stations = list(STATION_LISTS[type(self).network])
        if active_only:
            stations = [s for s in stations if s.get("status") == "Active"]
        if bbox is not None:
            west, south, east, north = bbox
            stations = [
                s
                for s in stations
                if west <= s["longitude"] <= east and south <= s["latitude"] <= north
            ]
        return stations

    def get_data(
        self,
        station_ids=None,
        variables=None,
        begin_date=None,
        end_date=None,
        interval="daily",
        include_flags=False,
        **kwargs,
    ):
        type(self).calls.append(
            {
                "method": "get_data",
                "station_ids": list(station_ids or []),
                "variables": list(variables or []),
                "begin_date": begin_date,
                "end_date": end_date,
                "interval": interval,
                "include_flags": include_flags,
            }
        )
        days = pd.date_range(begin_date or "2024-01-01", periods=3)
        return [
            {
                "station_id": station_id,
                "date": day.strftime("%Y-%m-%d"),
                "variable": "WTEQ",
                "type": "swe",
                "value": 10.0 + i,
                "units": "cm",
                "interval": interval,
            }
            for station_id in station_ids or []
            for i, day in enumerate(days)
        ]

    def get_metadata(self, station_id):
        type(self).calls.append({"method": "get_metadata", "station_id": station_id})
        return {"station_id": station_id, "series": []}


@pytest.fixture
def stub_clients(monkeypatch):
    """Replace every network's client class with a recording stub."""
    from easysnowdata.stations import clients

    made: dict[str, type[StubClient]] = {}
    for name, network in networks.NETWORKS.items():
        stub = type(f"Stub{name}", (StubClient,), {"network": name, "calls": []})
        made[name] = stub
        # Network is a frozen dataclass, so patch what client_class() looks up
        monkeypatch.setattr(clients, network.client, stub)
    monkeypatch.setenv("NVE_API_KEY", "stub-key")
    esd.auth.reset()
    yield made
    esd.auth.reset()


# ── inventory over the live clients ──────────────────────────────────────────


def test_inventory_queries_every_network_and_concatenates(stub_clients):
    inv = esd.stations.inventory(source="clients")
    assert len(inv) == 10  # two per network
    assert set(inv["network"]) == set(networks.NETWORKS)
    assert inv.crs.to_epsg() == 4326
    assert inv.index.name == "code"
    assert PARADISE_CODE in inv.index
    assert inv.attrs["source"] == "clients"
    assert "advertised" in inv.attrs["daily_flag"]


def test_inventory_pushes_the_aoi_down_as_a_bbox(stub_clients):
    esd.stations.inventory(RAINIER, networks="awdb", source="clients")
    call = stub_clients["awdb"].calls[-1]
    assert call["bbox"] == pytest.approx(RAINIER)


def test_inventory_passes_active_only_through(stub_clients):
    inv = esd.stations.inventory(networks="awdb", active_only=True, source="clients")
    assert stub_clients["awdb"].calls[-1]["active_only"] is True
    assert list(inv.index) == [PARADISE_CODE]


def test_advertised_daily_reads_each_networks_own_signal(stub_clients):
    inv = esd.stations.inventory(source="clients")
    # AWDB's station list carries no element inventory at all
    assert inv.loc[PARADISE_CODE, "daily"] is None
    assert bool(inv.loc["QUA", "daily"]) is True
    assert bool(inv.loc["HNT", "daily"]) is False
    assert bool(inv.loc["1A01P", "daily"]) is True  # ASWS
    assert bool(inv.loc["1A06A", "daily"]) is False  # manual survey course
    assert bool(inv.loc["12.142.0", "daily"]) is True  # params 2002/2003
    assert bool(inv.loc["1.15.0", "daily"]) is False
    assert bool(inv.loc["09AA-M1", "daily"]) is True
    assert bool(inv.loc["08AA-SC01", "daily"]) is False


def test_daily_only_drops_the_unknowns_on_the_live_route(stub_clients):
    inv = esd.stations.inventory(daily_only=True, source="clients")
    # AWDB reports nothing, so it cannot be kept by an advertised filter
    assert set(inv["network"]) == {"cdec", "databc", "nve", "yukon"}
    assert set(inv.index) == {"QUA", "1A01P", "12.142.0", "09AA-M1"}


def test_unknown_network_names_the_five(stub_clients):
    with pytest.raises(ValueError, match="Unknown station network"):
        esd.stations.inventory(networks="snotel", source="clients")


# ── load's fan-out ───────────────────────────────────────────────────────────


def test_load_sends_each_station_to_its_own_network(stub_clients):
    ds = esd.stations.load(
        [PARADISE_CODE, "12.142.0", "09AA-M1"],
        variables="swe",
        time="2024-03-01/2024-03-03",
    )
    assert stub_clients["awdb"].calls[-1]["station_ids"] == [PARADISE_TRIPLET]
    assert stub_clients["nve"].calls[-1]["station_ids"] == ["12.142.0"]
    assert stub_clients["yukon"].calls[-1]["station_ids"] == ["09AA-M1"]
    assert stub_clients["cdec"].calls == []
    assert sorted(ds["station"].values.tolist()) == sorted(
        [PARADISE_CODE, "12.142.0", "09AA-M1"]
    )


def test_load_passes_the_window_variables_and_interval_to_the_client(stub_clients):
    esd.stations.load(
        PARADISE_CODE,
        variables=["swe", "snwd"],
        time="2024-03",
        interval="hourly",
        include_flags=True,
    )
    call = stub_clients["awdb"].calls[-1]
    assert call["variables"] == ["swe", "snwd"]
    assert (call["begin_date"], call["end_date"]) == ("2024-03-01", "2024-03-31")
    assert call["interval"] == "hourly"
    assert call["include_flags"] is True


def test_time_none_asks_for_the_whole_record(stub_clients):
    esd.stations.load(PARADISE_CODE, variables="swe")
    call = stub_clients["awdb"].calls[-1]
    assert call["begin_date"] is None and call["end_date"] is None


def test_default_variables_are_swe_and_snow_depth(stub_clients):
    esd.stations.load(PARADISE_CODE, time="2024-03")
    assert stub_clients["awdb"].calls[-1]["variables"] == ["swe", "snwd"]


def test_one_network_gets_the_full_provenance_attrs(stub_clients):
    ds = esd.stations.load([PARADISE_CODE], variables="swe", time="2024-03")
    assert ds.attrs["source"] == "awdb"
    assert ds.attrs["source_id"] == "awdb"
    assert ds.attrs["product_id"] == "awdb-stations"
    assert ds.attrs["interval"] == "daily"
    assert ds.attrs["license"]
    assert ds.attrs["data_citation"]
    assert ds.attrs["easysnowdata_version"]
    # `source` must hand straight back to a load() keyword
    assert ds.attrs["source"] in networks.NETWORKS


def test_several_networks_join_their_ids_and_keep_the_per_station_truth(stub_clients):
    ds = esd.stations.load([PARADISE_CODE, "12.142.0"], variables="swe", time="2024-03")
    assert set(ds.attrs["source"].split()) == {"awdb", "nve"}
    assert set(ds.attrs["product_id"].split()) == {"awdb-stations", "nve-stations"}
    assert ds.attrs["title"] == "Snow station observations"
    # the exact answer per station is the coordinate, not the joined attr
    per_station = dict(zip(ds["station"].values, ds["network"].values, strict=True))
    assert per_station == {PARADISE_CODE: "awdb", "12.142.0": "nve"}
    assert all(isinstance(v, (str, int, float)) for v in ds.attrs.values())


def test_networks_keyword_pins_every_station(stub_clients):
    # "QUA" is ambiguous by shape, and naming the network settles it
    esd.stations.load(["QUA"], variables="swe", time="2024-03", networks="cdec")
    assert stub_clients["cdec"].calls[-1]["station_ids"] == ["QUA"]


def test_networks_keyword_can_also_narrow_a_mixed_request(stub_clients):
    ds = esd.stations.load(
        [PARADISE_CODE, "12.142.0"],
        variables="swe",
        time="2024-03",
        networks=["awdb", "cdec"],
    )
    assert stub_clients["nve"].calls == []
    assert list(ds["station"].values) == [PARADISE_CODE]


@pytest.fixture
def aoi_from_clients(monkeypatch):
    """Make load(aoi=...) pick its stations from the live clients.

    It normally asks the *archive* inventory, which is one HTTP request and
    carries the probe-verified daily flag; this pins the station list to the
    stubs so the fan-out can be tested without a socket.
    """
    import functools

    monkeypatch.setattr(
        _adapter, "inventory", functools.partial(_adapter.inventory, source="clients")
    )


def test_load_from_an_aoi_takes_its_stations_from_the_inventory(
    stub_clients, aoi_from_clients
):
    ds = esd.stations.load(
        aoi=RAINIER, networks="awdb", time="2024-03", daily_only=False
    )
    # Morse Lake sits east of the Rainier box, so the bbox pushdown drops it
    assert list(ds["station"].values) == [PARADISE_CODE]
    assert np.isfinite(ds["swe"].values).any()


def test_an_aoi_with_no_stations_says_so(stub_clients, aoi_from_clients):
    with pytest.raises(ValueError, match="No stations found"):
        esd.stations.load(aoi=(0.0, 0.0, 0.1, 0.1), networks="awdb", time="2024-03")


def test_an_inventory_frame_supplies_the_metadata_coordinates(stub_clients):
    inv = esd.stations.inventory(networks="awdb", source="clients")
    ds = esd.stations.load(inv, variables="swe", time="2024-03")
    assert ds["name"].sel(station=PARADISE_CODE).item() == "Paradise"
    assert ds["elevation_m"].sel(station=PARADISE_CODE).item() == pytest.approx(1563.6)


def test_metadata_passes_through_to_each_client(stub_clients):
    out = esd.stations.metadata([PARADISE_CODE, "12.142.0"])
    assert set(out) == {PARADISE_CODE, "12.142.0"}
    assert stub_clients["awdb"].calls[-1] == {
        "method": "get_metadata",
        "station_id": PARADISE_TRIPLET,
    }


def test_the_bbox_helper_ignores_a_global_aoi():
    assert _adapter._bbox(None) is None
    assert _adapter._bbox((-180.0, -90.0, 180.0, 90.0)) is None
    assert _adapter._bbox(RAINIER) == pytest.approx(RAINIER)


def test_metadata_unwraps_the_awdb_clients_one_element_list():
    """DESIGN.md §3.4 says get_metadata returns a dict; AWDB returns a list."""
    assert _adapter._one_metadata([{"a": 1}], "awdb", PARADISE_TRIPLET) == {"a": 1}
    assert _adapter._one_metadata({"a": 1}, "cdec", "QUA") == {"a": 1}
    # more than one record is not something to silently pick from
    two = [{"a": 1}, {"a": 2}]
    assert _adapter._one_metadata(two, "awdb", PARADISE_TRIPLET) == two
