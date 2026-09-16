"""easysnowdata.stations — the adapter over the vendored network clients.

The offline half works on the clients' record dicts directly, which is the
contract the adapter is written against, so no cassette is needed to pin the
shape of what comes back. The live half is one smoke test per network.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.processing import wateryear
from easysnowdata.stations import _frames, networks

from .conftest import (
    BBOX_NORWAY,
    MORSE_LAKE_CODE,
    PARADISE_CODE,
    PARADISE_TRIPLET,
    RAINIER,
)

# ── catalog entries ──────────────────────────────────────────────────────────


def test_every_network_has_a_catalog_entry_registered_from_this_package():
    listed = catalog.list(theme="stations")
    for network, product_id in esd.stations.PRODUCT_IDS.items():
        product = catalog.get(product_id)
        assert product_id in listed.index
        assert product.loader == "easysnowdata.stations.load"
        assert product.resolve_loader() is esd.stations.load
        assert [s.id for s in product.sources] == [network]
        assert product.variables, f"{product_id} lists no variables"
        assert any(s.health for s in product.sources)
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_only_nve_needs_a_credential():
    needs = {
        network: catalog.get(pid).requires
        for network, pid in esd.stations.PRODUCT_IDS.items()
    }
    assert needs == {
        "awdb": (),
        "cdec": (),
        "databc": (),
        "nve": ("nve",),
        "yukon": (),
    }
    assert networks.get("nve").requires == ("nve",)


def test_catalog_variables_come_from_the_client_registry():
    # swe and snwd everywhere; only the richer networks carry wind and temp
    for product_id in esd.stations.PRODUCT_IDS.values():
        names = {v.name for v in catalog.get(product_id).variables}
        assert {"swe", "snwd"} <= names
    assert {"swe", "snwd"} == {v.name for v in catalog.get("nve-stations").variables}
    assert "wind_spd" in {v.name for v in catalog.get("awdb-stations").variables}
    # units follow DESIGN.md §3.5
    swe = catalog.get("awdb-stations").variable("swe")
    assert swe.units == "cm" and swe.long_name == "snow water equivalent"


# ── station identity ─────────────────────────────────────────────────────────


def test_code_and_station_id_are_two_spellings_of_one_station():
    assert networks.to_station_id(PARADISE_CODE, "awdb") == PARADISE_TRIPLET
    assert networks.to_code(PARADISE_TRIPLET, "awdb") == PARADISE_CODE
    # the other four networks use the same string for both
    assert networks.to_station_id("QUA", "cdec") == "QUA"
    assert networks.to_code("12.142.0", "nve") == "12.142.0"


@pytest.mark.parametrize(
    ("station", "expected"),
    [
        (PARADISE_CODE, "awdb"),
        (PARADISE_TRIPLET, "awdb"),
        ("12.142.0", "nve"),
        ("09AA-M1", "yukon"),
        ("08AA-SC01", "yukon"),
        ("10MD-MET-00004", "yukon"),
        ("QUA", None),  # CDEC and DataBC codes have the same shape
        ("1A01P", None),
    ],
)
def test_network_is_guessed_from_the_code_shape_where_it_can_be(station, expected):
    assert networks.guess_network(station) == expected


def test_ambiguous_codes_raise_rather_than_guessing():
    with pytest.raises(ValueError, match="Cannot tell which network"):
        networks.split_by_network(["QUA", "1A01P"])
    # naming the network resolves it
    assert networks.split_by_network(["QUA"], network="cdec") == {"cdec": ["QUA"]}


def test_split_by_network_groups_a_mixed_list():
    grouped = networks.split_by_network([PARADISE_CODE, "12.142.0", "09AA-M1"])
    assert grouped == {
        "awdb": [PARADISE_TRIPLET],
        "nve": ["12.142.0"],
        "yukon": ["09AA-M1"],
    }


def test_split_by_network_trusts_an_inventory_frame_over_the_code_shape():
    frame = _frames.stations_to_geodataframe(
        [
            {
                "station_id": "QUA",
                "name": "Quartz",
                "latitude": 38.0,
                "longitude": -120.0,
            },
        ],
        "cdec",
    )
    assert networks.split_by_network(frame) == {"cdec": ["QUA"]}


def test_unknown_network_names_itself_in_the_error():
    with pytest.raises(ValueError, match="Unknown station network 'snotel'"):
        networks.get("snotel")


# ── records -> GeoDataFrame ──────────────────────────────────────────────────


def test_station_dicts_become_a_geodataframe_indexed_by_code():
    gdf = _frames.stations_to_geodataframe(
        [
            {
                "stationTriplet": PARADISE_TRIPLET,
                "name": "Paradise",
                "latitude": 46.78,
                "longitude": -121.75,
                "elevation_m": 1563.6,
                "stateCode": "WA",
                "networkCode": "SNTL",
                "status": "Active",
                "huc": "171100140103",
            }
        ],
        "awdb",
    )
    assert list(gdf.index) == [PARADISE_CODE]
    assert gdf.index.name == "code"
    assert gdf.crs.to_epsg() == 4326
    row = gdf.loc[PARADISE_CODE]
    assert row["station_id"] == PARADISE_TRIPLET
    assert row["network"] == "awdb" and row["network_code"] == "SNTL"
    assert row["state"] == "WA" and bool(row["is_active"]) is True
    assert row["data_provider"].startswith("USDA NRCS")
    assert row["huc"] == "171100140103"  # source-specific extras ride along
    assert row.geometry.x == pytest.approx(-121.75)


def test_stations_without_coordinates_or_an_id_are_dropped_not_guessed():
    gdf = _frames.stations_to_geodataframe(
        [{"name": "nameless"}, {"station_id": "AAA", "name": "fine"}], "cdec"
    )
    assert list(gdf.index) == ["AAA"]


# ── records -> Dataset ───────────────────────────────────────────────────────


def _records(station_id, variable, type_name, units, values, start="2024-01-01"):
    import pandas as pd

    days = pd.date_range(start, periods=len(values))
    return [
        {
            "station_id": station_id,
            "date": day.strftime("%Y-%m-%d"),
            "variable": variable,
            "type": type_name,
            "value": value,
            "units": units,
            "interval": "daily",
        }
        for day, value in zip(days, values, strict=True)
    ]


def test_records_become_a_station_time_dataset():
    records = _records(PARADISE_TRIPLET, "WTEQ", "swe", "cm", [10.0, 20.0, 30.0])
    records += _records(PARADISE_TRIPLET, "SNWD", "snwd", "cm", [50.0, 60.0, 70.0])
    ds = _frames.records_to_dataset(records, "awdb")
    assert isinstance(ds, xr.Dataset)
    assert ds["swe"].dims == ("station", "time")
    assert ds.sizes == {"station": 1, "time": 3}
    assert list(ds["station"].values) == [PARADISE_CODE]
    np.testing.assert_allclose(ds["swe"].values[0], [10.0, 20.0, 30.0])
    assert ds["swe"].attrs["units"] == "cm"
    assert ds["swe"].attrs["long_name"] == "snow water equivalent"
    assert ds["swe"].attrs["native_variables"] == "WTEQ"
    assert ds["swe"].attrs["interval"] == "daily"
    assert ds["swe"].dtype == np.float64


def test_missing_observations_are_nan_not_dropped():
    records = _records(PARADISE_TRIPLET, "WTEQ", "swe", "cm", [10.0, None, 30.0])
    ds = _frames.records_to_dataset(records, "awdb")
    assert ds.sizes["time"] == 3
    assert np.isnan(ds["swe"].values[0][1])


def test_stations_that_returned_nothing_still_appear_on_the_station_axis():
    records = _records(PARADISE_TRIPLET, "WTEQ", "swe", "cm", [10.0, 20.0])
    ds = _frames.records_to_dataset(
        records, "awdb", stations=[PARADISE_CODE, MORSE_LAKE_CODE]
    )
    assert list(ds["station"].values) == [PARADISE_CODE, MORSE_LAKE_CODE]
    assert np.isnan(ds["swe"].values[1]).all()


def test_several_networks_merge_onto_one_station_axis():
    ds = _frames.records_to_dataset(
        {
            "awdb": _records(PARADISE_TRIPLET, "WTEQ", "swe", "cm", [10.0, 20.0]),
            "cdec": _records("QUA", "SNO ADJ", "swe", "cm", [1.0, 2.0]),
        }
    )
    assert sorted(ds["station"].values.tolist()) == sorted([PARADISE_CODE, "QUA"])
    assert set(ds["network"].values.tolist()) == {"awdb", "cdec"}
    assert ds["swe"].attrs["native_variables"] == "SNO ADJ WTEQ"


def test_a_plain_record_list_without_a_network_name_raises():
    with pytest.raises(ValueError, match="needs network="):
        _frames.records_to_dataset([{"station_id": "QUA"}])


def test_water_year_coordinates_are_attached_to_time():
    records = _records(
        PARADISE_TRIPLET, "WTEQ", "swe", "cm", [1.0, 2.0, 3.0], start="2023-09-30"
    )
    ds = _frames.records_to_dataset(records, "awdb")
    assert list(ds["water_year"].values) == [2023, 2024, 2024]
    assert list(ds["dowy"].values) == [365, 1, 2]


def test_station_metadata_becomes_non_dimension_coordinates():
    meta = _frames.stations_to_geodataframe(
        [
            {
                "stationTriplet": PARADISE_TRIPLET,
                "name": "Paradise",
                "latitude": 46.78,
                "longitude": -121.75,
                "elevation_m": 1563.6,
                "status": "Active",
            }
        ],
        "awdb",
    )
    records = _records(PARADISE_TRIPLET, "WTEQ", "swe", "cm", [10.0])
    ds = _frames.records_to_dataset(records, "awdb", metadata=meta)
    assert ds["name"].dims == ("station",)
    assert ds["name"].values[0] == "Paradise"
    assert ds["elevation_m"].values[0] == pytest.approx(1563.6)
    # nothing in a coordinate is a Python object we could not serialize
    for coord in ds.coords.values():
        assert coord.dtype.kind in "USfiMb", coord.name


def test_a_type_served_in_two_units_is_brought_onto_one():
    # the vendored Yukon client emits one precip series in cm, against
    # DESIGN.md §3.5, so the adapter converts rather than mixing units
    records = _records("09AA-M1", "precip_mm", "precip", "mm", [10.0])
    records += _records("09BA-M7", "precip_cm", "precip", "cm", [1.0])
    ds = _frames.records_to_dataset(records, "yukon")
    assert ds["precip"].attrs["units"] == "mm"
    values = dict(zip(ds["station"].values, ds["precip"].values[:, 0], strict=True))
    assert values["09AA-M1"] == pytest.approx(10.0)
    assert values["09BA-M7"] == pytest.approx(10.0)  # 1 cm became 10 mm


def test_an_unconvertible_unit_clash_raises_instead_of_mixing():
    records = _records("09AA-M1", "a", "swe", "cm", [1.0])
    records += _records("09BA-M7", "b", "swe", "furlongs", [1.0])
    with pytest.raises(ValueError, match="No conversion from"):
        _frames.records_to_dataset(records, "yukon")


def test_sub_daily_records_keep_their_timestamp():
    records = [
        {
            "station_id": "1A01P",
            "date": "2024-01-01",
            "datetime": "2024-01-01T06:00:00",
            "variable": "SW",
            "type": "swe",
            "value": 5.0,
            "units": "cm",
            "interval": "hourly",
        },
        {
            "station_id": "1A01P",
            "date": "2024-01-01",
            "datetime": "2024-01-01T12:00:00",
            "variable": "SW",
            "type": "swe",
            "value": 6.0,
            "units": "cm",
            "interval": "hourly",
        },
    ]
    ds = _frames.records_to_dataset(records, "databc")
    assert ds.sizes["time"] == 2
    assert str(ds["time"].values[0])[:16] == "2024-01-01T06:00"


# ── argument handling ────────────────────────────────────────────────────────


def test_load_without_stations_or_aoi_says_so(no_network):
    with pytest.raises(ValueError, match="needs either stations="):
        esd.stations.load()


def test_inventory_rejects_an_unknown_source(no_network):
    with pytest.raises(ValueError, match="Unknown inventory source"):
        esd.stations.inventory(RAINIER, source="the-internet")


# ── the reconciled water-year helpers (§12 Q16) ──────────────────────────────


def test_water_year_helpers_match_the_global_snow_networks_originals():
    # the values utils/utils.py documents in its own docstrings
    assert wateryear.water_year("2023-10-01") == 2024
    assert wateryear.water_year("2024-09-30") == 2024
    assert wateryear.water_year("2024-10-01") == 2025
    assert wateryear.day_of_water_year("2023-10-01") == 1
    assert wateryear.day_of_water_year("2024-09-30") == 366
    assert wateryear.day_of_water_year("2024-01-01") == 93
    start, end = wateryear.water_year_bounds(2024)
    assert (str(start.date()), str(end.date())) == ("2023-10-01", "2024-09-30")
    assert wateryear.water_year_length(2024) == 366
    assert wateryear.water_year_length(2023) == 365
    assert len(wateryear.water_year_range(2024)) == 366


def test_water_year_helpers_take_a_hemisphere_the_originals_did_not():
    start, end = wateryear.water_year_bounds(2024, "southern")
    assert (str(start.date()), str(end.date())) == ("2024-04-01", "2025-03-31")
    assert wateryear.water_year("2024-04-01", "southern") == 2024


# ── live smoke tests ─────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_inventory_and_load_over_snotel():
    inv = esd.stations.inventory(RAINIER, networks="awdb", source="clients")
    assert inv.crs.to_epsg() == 4326
    assert inv.index.name == "code"
    assert PARADISE_CODE in inv.index
    assert inv.loc[PARADISE_CODE, "station_id"] == PARADISE_TRIPLET

    ds = esd.stations.load(
        [PARADISE_CODE, MORSE_LAKE_CODE],
        variables=["swe", "snwd"],
        time="2024-03-01/2024-03-07",
    )
    assert ds["swe"].dims == ("station", "time")
    assert ds.sizes == {"station": 2, "time": 7}
    assert ds["swe"].dtype == np.float64
    assert ds["swe"].attrs["units"] == "cm"
    assert ds["swe"].attrs["native_variables"] == "WTEQ"
    assert {"water_year", "dowy", "network"} <= set(ds.coords)
    assert list(ds["water_year"].values) == [2024] * 7
    # Paradise in March carries metres of snow; the units are centimetres
    paradise = ds["swe"].sel(station=PARADISE_CODE).values
    assert 50.0 < float(np.nanmedian(paradise)) < 500.0
    assert ds.attrs["source"] == "awdb"
    assert ds.attrs["product_id"] == "awdb-stations"
    assert ds.attrs["easysnowdata_version"]
    assert all(isinstance(v, (str, int, float)) for v in ds.attrs.values())


@pytest.mark.live
def test_live_load_accepts_native_station_ids_too():
    ds = esd.stations.load(
        PARADISE_TRIPLET, variables="swe", time="2024-03-01/2024-03-03"
    )
    assert list(ds["station"].values) == [PARADISE_CODE]


@pytest.mark.live
def test_live_cdec_smoke():
    ds = esd.stations.load(
        ["QUA"], variables=["swe"], time="2024-03-01/2024-03-05", networks="cdec"
    )
    assert ds.sizes == {"station": 1, "time": 5}
    assert ds.attrs["product_id"] == "cdec-stations"


@pytest.mark.live
def test_live_databc_smoke():
    ds = esd.stations.load(
        ["1A01P"], variables=["swe"], time="2024-03-01/2024-03-05", networks="databc"
    )
    assert ds.sizes["station"] == 1
    assert ds.attrs["product_id"] == "databc-stations"


@pytest.mark.live
def test_live_yukon_smoke():
    ds = esd.stations.load(["09AA-M1"], variables=["swe"], time="2024-03-01/2024-03-05")
    assert ds.sizes["station"] == 1
    assert ds.attrs["product_id"] == "yukon-stations"


@pytest.mark.live
@pytest.mark.requires_nve
def test_live_nve_smoke():
    inv = esd.stations.inventory(BBOX_NORWAY, networks="nve", source="clients")
    assert len(inv) > 100
    ds = esd.stations.load(
        ["12.142.0"], variables=["swe"], time="2024-03-01/2024-03-05"
    )
    assert ds.sizes["station"] == 1
    assert ds.attrs["product_id"] == "nve-stations"


@pytest.mark.live
def test_live_load_from_an_aoi_picks_its_own_stations():
    ds = esd.stations.load(
        aoi=RAINIER,
        networks="awdb",
        time="2024-03-01/2024-03-03",
        daily_only=False,
    )
    assert ds.sizes["station"] >= 1
    assert "swe" in ds.data_vars


@pytest.mark.live
def test_live_metadata_carries_the_variable_inventory():
    meta = esd.stations.metadata(PARADISE_CODE)
    assert PARADISE_CODE in meta
    # The AWDB client answers with a one-element list rather than the dict
    # DESIGN.md §3.4 specifies; the adapter unwraps it so callers see one shape.
    assert isinstance(meta[PARADISE_CODE], dict)
    assert meta[PARADISE_CODE]["stationTriplet"] == PARADISE_TRIPLET


# ── credentials (§9 step 5) ──────────────────────────────────────────────────


def test_only_the_nve_client_reads_a_credential():
    """The other four networks are open APIs; nothing else reads a key.

    DataBC fetches an anti-forgery token from a public disclaimer page, which
    is a session mechanic rather than a credential — it needs no account.
    """
    import re
    from pathlib import Path

    root = Path(esd.stations.clients.__file__).parent
    offenders = {}
    for path in sorted(root.glob("*/*_client.py")):
        hits = re.findall(r"os\.environ|os\.getenv|getpass|netrc", path.read_text())
        if hits:
            offenders[path.parent.name] = hits
    assert offenders == {}, offenders


def test_nve_client_is_constructible_without_a_key():
    """Constructing must not need the key; only a request does."""
    client = networks.get("nve").client_class()()
    assert "X-API-Key" not in client._session.headers


def test_nve_raises_before_the_request_when_the_key_is_missing(monkeypatch):
    from easysnowdata.utils import CredentialError

    monkeypatch.delenv("NVE_API_KEY", raising=False)
    esd.auth.reset()
    client = networks.get("nve").client_class()()
    with pytest.raises(CredentialError) as excinfo:
        client._authorize()
    assert excinfo.value.provider == "nve"
    assert "NVE_API_KEY" in str(excinfo.value)
    esd.auth.reset()


def test_the_nve_key_comes_from_the_auth_provider(monkeypatch):
    monkeypatch.setenv("NVE_API_KEY", "a-test-key")
    esd.auth.reset()
    client = networks.get("nve").client_class()()
    client._authorize()
    assert client._session.headers["X-API-Key"] == "a-test-key"
    # an explicitly passed key still wins over the environment
    explicit = networks.get("nve").client_class()(api_key="explicit")
    explicit._authorize()
    assert explicit._session.headers["X-API-Key"] == "explicit"
    esd.auth.reset()


def test_loading_from_nve_without_a_key_names_the_provider(monkeypatch):
    from easysnowdata.utils import CredentialError

    monkeypatch.delenv("NVE_API_KEY", raising=False)
    esd.auth.reset()
    with pytest.raises(CredentialError) as excinfo:
        esd.stations.load(["12.142.0"], variables="swe", time="2024-03")
    assert excinfo.value.provider == "nve"
    esd.auth.reset()


@pytest.mark.parametrize("network", ["awdb", "cdec", "databc", "yukon"])
def test_the_other_four_networks_need_no_provider(network, monkeypatch):
    monkeypatch.delenv("NVE_API_KEY", raising=False)
    esd.auth.reset()
    # resolving the product and its credentials must not raise for these
    from easysnowdata.catalog._access import ensure_source, resolve_source

    product = catalog.get(esd.stations.PRODUCT_IDS[network])
    assert ensure_source(product, resolve_source(product, network)) == {}
    esd.auth.reset()
