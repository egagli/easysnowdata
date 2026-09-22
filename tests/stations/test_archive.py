"""easysnowdata.stations.archive — the daily fast path.

Offline, the three published artefacts are stood in for by the tiny fixtures
``tests/stations/fixtures/make_fixtures.py`` writes, which are laid out
exactly like the real ones. Live, one smoke test per route plus a check that
the routes agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.stations import archive

from .conftest import MORSE_LAKE_CODE, PARADISE_CODE, RAINIER


@pytest.fixture
def local_archive(station_fixtures, monkeypatch):
    """Point the archive module at the fixtures instead of GitHub."""
    monkeypatch.setattr(
        archive, "INVENTORY_URL", str(station_fixtures["inventory"]), raising=False
    )
    monkeypatch.setattr(
        archive.providers.raster_http,
        "fetch",
        lambda *a, **k: station_fixtures["archive"],
    )
    monkeypatch.setattr(
        archive,
        "ZARR_URLS",
        {
            "by_time": str(station_fixtures["zarr_by_time"]),
            "by_station": str(station_fixtures["zarr_by_station"]),
        },
    )
    monkeypatch.setattr(archive, "_INVENTORY_CACHE", {})
    return station_fixtures


@pytest.fixture
def store_unreachable(local_archive, monkeypatch):
    """The Pages store 404s (a deploy that has not happened, or an outage)."""
    monkeypatch.setattr(
        archive,
        "ZARR_URLS",
        {k: f"{v}.does-not-exist" for k, v in archive.ZARR_URLS.items()},
    )
    return local_archive


# ── catalog entry ────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("snow-station-archive")
    assert product is archive.PRODUCT
    assert product.loader == "easysnowdata.stations.archive.load"
    assert product.resolve_loader() is archive.load
    assert [s.id for s in product.sources] == [
        "github-pages-zarr",
        "github-tarball",
        "github-csv",
    ]
    assert product.default_source.id == "github-pages-zarr"
    assert product.requires == ()  # the archive needs no credential at all
    assert {v.name for v in product.variables} == {"swe", "snwd"}
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_urls_point_at_the_published_artefacts():
    assert archive.INVENTORY_URL.endswith("/all_snow_stations.geojson")
    assert archive.ARCHIVE_URL.endswith("/data/all_station_csvs.tar.xz")
    assert archive.csv_url(PARADISE_CODE).endswith(f"/stations/{PARADISE_CODE}.csv")
    assert archive.PAGES_BASE == "https://egagli.github.io/global_snow_networks/archive"
    assert archive.ZARR_URLS["by_time"].endswith("/archive/by_time.zarr")
    assert archive.ZARR_URLS["by_station"].endswith("/archive/by_station.zarr")
    assert archive.MANIFEST_URL.endswith("/archive/archive.json")


def test_the_store_layout_follows_how_many_stations_are_named():
    assert archive._layout_for(None) == "by_time"
    assert archive._layout_for([PARADISE_CODE]) == "by_station"
    assert archive._layout_for([str(i) for i in range(64)]) == "by_station"
    assert archive._layout_for([str(i) for i in range(65)]) == "by_time"


def test_the_archive_only_holds_swe_and_snow_depth():
    with pytest.raises(ValueError, match="The archive holds only"):
        archive._types(["temp"])
    assert archive._types(None) == ["snwd", "swe"]
    assert archive._types("swe") == ["swe"]


# ── inventory ────────────────────────────────────────────────────────────────


@pytest.mark.recorded
def test_inventory_is_indexed_by_code_and_carries_the_probe_verdict(local_archive):
    inv = archive.inventory()
    assert inv.index.name == "code"
    assert inv.crs.to_epsg() == 4326
    assert set(inv.index) == {PARADISE_CODE, "QUA", "HNT", "12.142.0", "08AA-SC01"}
    assert inv.loc[PARADISE_CODE, "network"] == "awdb"
    assert inv.loc[PARADISE_CODE, "station_id"] == "679:WA:SNTL"
    assert bool(inv.loc[PARADISE_CODE, "daily_or_better"]) is True
    assert bool(inv.loc["08AA-SC01", "daily_or_better"]) is False
    assert inv.attrs["product_id"] == "snow-station-archive"


@pytest.mark.recorded
def test_the_upstream_network_property_does_not_shadow_the_access_path(local_archive):
    """The published inventory has its own `network`, meaning something else.

    Upstream it is a Yukon-only display name and null for the other four
    clients; here `network` is the access path. The upstream value is kept
    under `network_name`.
    """
    inv = archive.inventory(columns="all")
    assert inv.loc["08AA-SC01", "network"] == "yukon"
    assert inv.loc["08AA-SC01", "network_name"] == "Yukon Snow Survey Network"
    assert inv["network"].notna().all()


@pytest.mark.recorded
def test_inventory_filters(local_archive):
    assert set(archive.inventory(daily_only=True).index) == {
        PARADISE_CODE,
        "QUA",
        "12.142.0",
    }
    assert set(archive.inventory(networks="nve").index) == {"12.142.0"}
    assert set(archive.inventory(networks=["awdb", "cdec"]).index) == {
        PARADISE_CODE,
        "QUA",
        "HNT",
    }
    assert set(archive.inventory(RAINIER).index) == {PARADISE_CODE}


@pytest.mark.recorded
def test_network_of_resolves_the_codes_whose_shape_is_ambiguous(local_archive):
    assert archive.network_of(["QUA"]) == {"QUA": "cdec"}
    assert archive.network_of(["nope"]) == {}


# ── load ─────────────────────────────────────────────────────────────────────


@pytest.mark.recorded
def test_the_default_route_is_the_pages_store(local_archive):
    ds = archive.load()
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"swe", "snwd"}
    assert ds["swe"].dims == ("station", "time")
    assert ds.sizes == {"station": 3, "time": 7}
    assert ds["swe"].attrs["units"] == "cm"
    assert ds["swe"].attrs["native_variables"] == "wteq_cm"
    assert ds["swe"].dtype == np.float64
    assert ds.attrs["source"] == "github-pages-zarr"
    assert ds.attrs["source_url"] == archive.ZARR_URLS["by_time"]
    assert ds.attrs["product_id"] == "snow-station-archive"
    assert ds.attrs["interval"] == "daily"


@pytest.mark.recorded
def test_tarball_route_builds_the_station_time_dataset(local_archive):
    ds = archive.load(source="github-tarball")
    assert set(ds.data_vars) == {"swe", "snwd"}
    assert ds.sizes == {"station": 3, "time": 7}
    assert ds["swe"].dtype == np.float64
    assert ds.attrs["source"] == "github-tarball"
    assert ds.attrs["source_url"] == archive.ARCHIVE_URL


@pytest.mark.recorded
def test_the_store_and_the_tarball_agree(local_archive):
    """Same observations, two published forms — coordinates and all.

    The store holds float32 (that repo's DESIGN.md §6.5) and the CSVs parse
    as float64, so values agree to float32 precision, not bit for bit.
    """
    from_store = archive.load()
    from_bundle = archive.load(source="github-tarball")
    from_store.attrs.clear()
    from_bundle.attrs.clear()
    xr.testing.assert_identical(
        from_store.drop_vars(list(from_store.data_vars)),
        from_bundle.drop_vars(list(from_bundle.data_vars)),
    )
    for name in ("swe", "snwd"):
        assert from_store[name].attrs == from_bundle[name].attrs
        np.testing.assert_allclose(
            from_store[name].values, from_bundle[name].values, rtol=1e-6, equal_nan=True
        )


@pytest.mark.recorded
def test_a_few_named_stations_read_from_the_by_station_store(local_archive):
    ds = archive.load([PARADISE_CODE, "QUA"])
    assert ds.attrs["source_url"] == archive.ZARR_URLS["by_station"]
    assert list(ds["station"].values) == [PARADISE_CODE, "QUA"]
    assert ds["name"].sel(station=PARADISE_CODE).item() == "Paradise"


@pytest.mark.recorded
def test_the_store_route_gives_a_station_without_data_an_all_nan_row(local_archive):
    ds = archive.load([PARADISE_CODE, "08AA-SC01"])
    assert list(ds["station"].values) == [PARADISE_CODE, "08AA-SC01"]
    assert np.isnan(ds["swe"].sel(station="08AA-SC01")).all()
    assert np.isfinite(ds["swe"].sel(station=PARADISE_CODE)).any()


@pytest.mark.recorded
def test_the_store_route_narrows_time_and_variables(local_archive):
    ds = archive.load(variables="swe", time="2023-10-01/2023-10-03")
    assert set(ds.data_vars) == {"swe"}
    assert ds.sizes["time"] == 3
    assert str(ds["time"].values[0])[:10] == "2023-10-01"
    assert list(ds["water_year"].values) == [2024, 2024, 2024]


@pytest.mark.recorded
def test_an_unreachable_store_falls_back_to_the_tarball_with_a_warning(
    store_unreachable, caplog
):
    ds = archive.load()
    assert ds.attrs["source"] == "github-tarball"
    assert ds.sizes == {"station": 3, "time": 7}
    assert "reading the bundled CSVs instead" in caplog.text


@pytest.mark.recorded
def test_an_unreachable_store_asked_for_by_name_fails(store_unreachable):
    with pytest.raises(Exception, match="does-not-exist|No such|not found|Unable"):
        archive.load(source="github-pages-zarr")


def test_manifest_is_the_pages_archive_json(monkeypatch):
    seen = {}

    def fake_fetch(url):
        seen["url"] = url
        return {"built_from_commit": "abc123", "stores": {}}

    monkeypatch.setattr(archive, "_fetch_json", fake_fetch)
    assert archive.manifest()["built_from_commit"] == "abc123"
    assert seen["url"] == archive.MANIFEST_URL


@pytest.mark.recorded
def test_empty_csv_fields_become_nan_not_the_string_nan(local_archive):
    ds = archive.load()
    # QUA has no snow-depth sensor at all, and every series has one gap
    assert np.isnan(ds["snwd"].sel(station="QUA")).all()
    assert np.isnan(ds["swe"].sel(station=PARADISE_CODE).values[1])
    assert np.isfinite(ds["swe"].sel(station=PARADISE_CODE).values[0])


@pytest.mark.recorded
def test_water_year_coordinates_span_the_boundary(local_archive):
    ds = archive.load()
    assert list(ds["water_year"].values) == [2023, 2023, 2024, 2024, 2024, 2024, 2024]
    assert list(ds["dowy"].values) == [364, 365, 1, 2, 3, 4, 5]


@pytest.mark.recorded
def test_station_metadata_rides_along_as_coordinates(local_archive):
    ds = archive.load()
    assert ds["name"].sel(station=PARADISE_CODE).item() == "Paradise"
    assert ds["network"].sel(station=PARADISE_CODE).item() == "awdb"
    assert ds["elevation_m"].sel(station=PARADISE_CODE).item() == pytest.approx(1569.7)
    for coord in ds.coords.values():
        assert coord.dtype.kind in "USfiMb", coord.name


@pytest.mark.recorded
def test_time_narrows_the_grid_as_it_is_built(local_archive):
    ds = archive.load(time="2023-10-01/2023-10-03")
    assert ds.sizes["time"] == 3
    assert str(ds["time"].values[0])[:10] == "2023-10-01"


@pytest.mark.recorded
def test_a_named_subset_keeps_the_station_axis_the_caller_asked_for(local_archive):
    ds = archive.load([PARADISE_CODE, "QUA"])
    assert list(ds["station"].values) == [PARADISE_CODE, "QUA"]


@pytest.mark.recorded
def test_a_station_without_a_csv_is_all_nan_rather_than_missing(local_archive):
    ds = archive.load([PARADISE_CODE, "08AA-SC01"])
    assert list(ds["station"].values) == [PARADISE_CODE, "08AA-SC01"]
    assert np.isnan(ds["swe"].sel(station="08AA-SC01")).all()


@pytest.mark.recorded
def test_an_aoi_picks_its_own_stations(local_archive):
    ds = archive.load(aoi=RAINIER)
    assert list(ds["station"].values) == [PARADISE_CODE]


@pytest.mark.recorded
def test_one_variable_only(local_archive):
    ds = archive.load(variables="swe")
    assert set(ds.data_vars) == {"swe"}


@pytest.mark.recorded
def test_the_csv_route_needs_to_know_which_stations(local_archive):
    with pytest.raises(ValueError, match="needs stations= or aoi="):
        archive.load(source="github-csv")


# ── live smoke tests ─────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_archive_inventory():
    inv = archive.inventory(daily_only=True)
    assert len(inv) > 1000
    assert inv.crs.to_epsg() == 4326
    assert inv.index.name == "code"
    assert PARADISE_CODE in inv.index
    assert set(inv["network"]) <= set(esd.stations.networks.NETWORKS)
    assert inv["network"].notna().all()


@pytest.mark.live
def test_live_archive_csv_route():
    ds = archive.load(
        [PARADISE_CODE, MORSE_LAKE_CODE],
        source="github-csv",
        time="2024-01-01/2024-03-31",
    )
    assert ds["swe"].dims == ("station", "time")
    assert ds.sizes == {"station": 2, "time": 91}
    assert ds["swe"].dtype == np.float64
    assert ds["swe"].attrs["units"] == "cm"
    # Paradise carries metres of snow in late winter, reported in centimetres
    paradise = ds["swe"].sel(station=PARADISE_CODE)
    assert 50.0 < float(np.nanmedian(paradise)) < 500.0
    assert {"water_year", "dowy", "network", "name"} <= set(ds.coords)
    assert ds.attrs["source"] == "github-csv"
    assert all(isinstance(v, (str, int, float)) for v in ds.attrs.values())


@pytest.mark.live
def test_live_archive_store_route():
    ds = archive.load(
        [PARADISE_CODE], source="github-pages-zarr", time="2024-01/2024-03"
    )
    assert ds.attrs["source"] == "github-pages-zarr"
    assert ds.attrs["source_url"] == archive.ZARR_URLS["by_station"]
    assert ds.sizes["station"] == 1
    assert ds.sizes["time"] == 91  # the store's time axis is complete
    assert 50.0 < float(np.nanmedian(ds["swe"])) < 500.0
    assert archive.manifest()["built_from_commit"]


@pytest.mark.live
def test_live_archive_routes_agree():
    when = "2024-01-01/2024-03-31"
    bundled = archive.load([PARADISE_CODE], source="github-tarball", time=when)
    per_station = archive.load([PARADISE_CODE], source="github-csv", time=when)
    chunked = archive.load([PARADISE_CODE], source="github-pages-zarr", time=when)
    assert bundled.attrs["source"] == "github-tarball"
    np.testing.assert_allclose(
        bundled["swe"].values, per_station["swe"].values, equal_nan=True
    )
    # Same days in this window, so the complete axis and the observed axis agree
    np.testing.assert_allclose(
        chunked["swe"].values, bundled["swe"].values, equal_nan=True
    )


@pytest.mark.live
def test_live_archive_over_an_aoi():
    ds = archive.load(aoi=RAINIER, time="2024-03-01/2024-03-07")
    assert ds.sizes["station"] >= 1
    assert ds.sizes["time"] == 7
    assert float(np.nanmax(ds["swe"])) > 0.0
