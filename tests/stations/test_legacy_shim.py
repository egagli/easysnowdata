"""easysnowdata.automatic_weather_stations — the StationCollection shim.

The class's own historical tests still live in
``tests/test_automatic_weather_stations.py`` and still pass unchanged; this
file covers what the shim does *differently* now that the frozen CSV route is
gone, and does it offline against the fixture archive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata import automatic_weather_stations as aws
from easysnowdata._deprecation import EasysnowdataDeprecationWarning, reset_warnings
from easysnowdata.automatic_weather_stations import StationCollection
from easysnowdata.stations import archive

from .conftest import PARADISE_CODE, PARADISE_TRIPLET


@pytest.fixture
def local_archive(station_fixtures, monkeypatch):
    """Serve both the inventory and the data from the local fixtures."""
    monkeypatch.setattr(
        archive, "INVENTORY_URL", str(station_fixtures["inventory"]), raising=False
    )
    monkeypatch.setattr(
        archive.providers.raster_http,
        "fetch",
        lambda *a, **k: station_fixtures["archive"],
    )
    monkeypatch.setattr(archive, "_INVENTORY_CACHE", {})
    return station_fixtures


@pytest.fixture
def offline_collection(local_archive, monkeypatch):
    """A StationCollection whose data comes from the fixture archive."""

    def fake_load(stations, *, variables, time, **kwargs):
        ds = archive.load(
            stations, variables=[v for v in variables if v in ("swe", "snwd")]
        )
        for missing in variables:
            if missing not in ds.data_vars:
                ds[missing] = xr.full_like(ds["swe"], np.nan)
        return ds

    monkeypatch.setattr(aws._stations, "load", fake_load)
    reset_warnings()
    with pytest.warns(EasysnowdataDeprecationWarning, match="StationCollection"):
        return StationCollection()


# ── the shim still looks like the old class ──────────────────────────────────


def test_constructing_it_warns_once(local_archive):
    import warnings

    reset_warnings()
    with pytest.warns(EasysnowdataDeprecationWarning, match="removed in 0.2.0"):
        first = StationCollection()
    assert first.all_stations is not None
    # the message is emitted once per process, not once per instance
    with warnings.catch_warnings():
        warnings.simplefilter("error", EasysnowdataDeprecationWarning)
        StationCollection()


def test_all_stations_keeps_the_old_columns_and_index(offline_collection):
    gdf = offline_collection.all_stations
    assert gdf.index.name == "code"
    assert gdf.crs.to_epsg() == 4326
    # the names the frozen GeoJSON used
    for column in ("name", "network", "elevation_m", "latitude", "longitude"):
        assert column in gdf.columns
    assert "csvData" in gdf.columns
    assert set(gdf["network"]) <= {"SNOTEL", "CCSS", "SNOW", "MSNT", "SNTLT", "SCAN"}
    assert gdf.loc[PARADISE_CODE, "network"] == "SNOTEL"


def test_snotel_sorts_before_ccss_as_the_frozen_file_did(offline_collection):
    networks = list(offline_collection.all_stations["network"])
    assert networks.index("SNOTEL") < networks.index("CCSS")
    assert "_" in offline_collection.all_stations.index[0]


def test_data_available_false_keeps_the_periodic_stations(local_archive):
    reset_warnings()
    everything = StationCollection(data_available=False)
    daily = StationCollection()
    assert len(everything.all_stations) > len(daily.all_stations)


def test_choose_stations_still_takes_all_three_forms(offline_collection):
    sc = offline_collection
    sc.choose_stations(PARADISE_CODE)
    assert list(sc.stations.index) == [PARADISE_CODE]
    sc.choose_stations([PARADISE_CODE])
    assert list(sc.stations.index) == [PARADISE_CODE]
    sc.choose_stations(sc.all_stations.iloc[:2])
    assert len(sc.stations) == 2


# ── what the shim now does differently ───────────────────────────────────────


def test_codes_map_to_the_network_station_ids(offline_collection):
    from easysnowdata.stations import networks

    assert networks.to_station_id(PARADISE_CODE, "awdb") == PARADISE_TRIPLET
    assert offline_collection.all_stations.loc[PARADISE_CODE, "station_id"] == (
        PARADISE_TRIPLET
    )


def test_the_six_old_names_map_onto_the_design_vocabulary():
    assert aws.VARIABLE_TYPES == {
        "WTEQ": "swe",
        "SNWD": "snwd",
        "PRCPSA": "precip",
        "TAVG": "temp",
        "TMIN": "temp_min",
        "TMAX": "temp_max",
    }


def test_values_are_converted_back_to_the_frozen_archives_units():
    """The old CSVs held metres; the networks report centimetres."""
    ds = xr.Dataset(
        {
            "swe": ("time", [100.0]),  # cm
            "snwd": ("time", [250.0]),  # cm
            "precip": ("time", [10.0]),  # mm
            "temp": ("time", [-3.0]),  # °C
        },
        coords={"time": pd.to_datetime(["2024-01-01"])},
    )
    out = aws._to_legacy_units(ds, ["WTEQ", "SNWD", "PRCPSA", "TAVG"])
    assert float(out["WTEQ"][0]) == pytest.approx(1.0)  # m
    assert float(out["SNWD"][0]) == pytest.approx(2.5)  # m
    assert float(out["PRCPSA"][0]) == pytest.approx(0.01)  # m
    assert float(out["TAVG"][0]) == pytest.approx(-3.0)  # °C, untouched
    assert out["WTEQ"].attrs["units"] == "m"
    assert out["TAVG"].attrs["units"] == "degC"


def test_asking_for_tavg_warns_that_it_is_a_different_element():
    aws._TAVG_WARNED = False
    with pytest.warns(UserWarning, match="TAVG no longer comes from"):
        aws._variables(["TAVG"], default=["WTEQ"])
    # once per process
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        aws._variables(["TAVG"], default=["WTEQ"])


def test_an_unknown_variable_names_the_six_it_serves():
    with pytest.raises(ValueError, match="this class serves"):
        aws._variables(["SNOWDEPTH"], default=["WTEQ"])


def test_single_station_returns_a_dataframe_in_metres(offline_collection):
    sc = offline_collection
    sc.get_data(stations=PARADISE_CODE, variables=["WTEQ", "SNWD"])
    assert isinstance(sc.data, pd.DataFrame)
    assert list(sc.data.columns) == ["WTEQ", "SNWD"]
    assert sc.data.index.name == "datetime"
    # the fixture's peak is 12 cm, so every value is well under a metre
    assert float(sc.data["WTEQ"].max()) < 1.0


def test_a_variable_the_station_does_not_serve_is_an_empty_column(offline_collection):
    sc = offline_collection
    sc.get_data(stations="QUA", variables=["WTEQ", "SNWD"])
    assert sc.data["SNWD"].isna().all()
    assert sc.data["WTEQ"].notna().any()


def test_multiple_stations_return_a_dataset_with_wy_and_dowy(offline_collection):
    sc = offline_collection
    sc.get_data(stations=[PARADISE_CODE, "QUA"], variables="WTEQ")
    assert isinstance(sc.data, xr.Dataset)
    assert "WTEQ" in sc.data.data_vars
    assert {"WY", "DOWY"} <= set(sc.data.coords)
    assert sc.data["WTEQ"].dims == ("station", "time")
    # the per-variable attribute the old class set
    assert isinstance(sc.WTEQ, pd.DataFrame)


def test_dtype_is_still_honoured_although_there_is_no_read_csv(offline_collection):
    sc = offline_collection
    sc.get_data(stations=PARADISE_CODE, variables=["WTEQ"], dtype={"WTEQ": "float32"})
    assert sc.data["WTEQ"].dtype == "float32"
    sc.get_data(
        stations=[PARADISE_CODE, "QUA"], variables=["WTEQ"], dtype={"WTEQ": "float32"}
    )
    assert sc.data["WTEQ"].dtype == "float32"


def test_other_read_csv_arguments_are_ignored_with_a_warning(
    offline_collection, caplog
):
    import logging

    sc = offline_collection
    with caplog.at_level(logging.WARNING, logger="easysnowdata"):
        sc.get_data(stations=PARADISE_CODE, variables=["WTEQ"], sep=";")
    assert "no longer comes from a CSV" in caplog.text


@pytest.mark.recorded
def test_the_entire_archive_comes_from_the_new_fast_path(offline_collection):
    ds = offline_collection.get_entire_data_archive()
    assert set(ds.data_vars) == {"WTEQ", "SNWD"}
    assert {"WY", "DOWY"} <= set(ds.coords)
    assert ds is offline_collection.entire_data_archive
    assert float(np.nanmax(ds["WTEQ"])) < 1.0  # metres, not centimetres


# ── live ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_shim_matches_the_numbers_the_frozen_archive_served():
    """WTEQ and SNWD for Paradise, winter 2021, are what they always were.

    Measured against the frozen egagli/snotel_ccss_stations CSV before it was
    retired: WTEQ mean 1.6134 m, SNWD mean 3.8552 m over 2021-01-01..2021-04-01.
    """
    reset_warnings()
    sc = StationCollection()
    sc.get_data(
        stations=PARADISE_CODE,
        variables=["WTEQ", "SNWD"],
        start_date="2021-01-01",
        end_date="2021-04-01",
    )
    assert len(sc.data) == 91
    assert float(sc.data["WTEQ"].mean()) == pytest.approx(1.6134, abs=1e-3)
    assert float(sc.data["SNWD"].mean()) == pytest.approx(3.8552, abs=1e-3)
