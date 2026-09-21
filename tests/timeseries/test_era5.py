"""easysnowdata.climate.era5 — offline tier on the synthetic ARCO-like store,
plus the live smoke tests (ARCO on GCS, ERA5-Land on Earth Engine)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.climate import era5

RAINIER = (-121.94, 46.72, -121.54, 46.99)
DATELINE = (179.5, -18.0, -179.5, -17.0)


@pytest.fixture
def arco(monkeypatch, ts_fixtures):
    """Point the ARCO route at the local synthetic store."""
    monkeypatch.setattr(era5, "ARCO_URL", str(ts_fixtures["reanalysis"]))
    return ts_fixtures["reanalysis"]


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("era5")
    assert product is era5.PRODUCT
    assert product.loader == "easysnowdata.climate.era5.load"
    assert product.resolve_loader() is era5.load
    assert [s.id for s in product.sources] == ["arco-era5-gcs", "gee"]
    assert product.default_source.requires == ()
    assert product.source("gee").requires == ("earthengine",)
    labels = [p.label for s in product.sources for p in s.health if p.kind == "health"]
    assert labels == ["ARCO-ERA5 (GCS anonymous)", "ERA5 (Google Earth Engine)"]
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_source_selection_rules():
    assert era5._pick_source(None, "ERA5", "hourly").id == "arco-era5-gcs"
    assert era5._pick_source("auto", "ERA5", "daily").id == "gee"
    assert era5._pick_source("auto", "ERA5_LAND", "hourly").id == "gee"
    assert era5._pick_source("gee", "ERA5", "hourly").id == "gee"
    with pytest.raises(ValueError, match="only serves hourly ERA5"):
        era5._pick_source("arco-era5-gcs", "ERA5_LAND", "hourly")
    with pytest.raises(ValueError, match="version must be one of"):
        era5._normalise("ERA6", "hourly")
    with pytest.raises(ValueError, match="cadence must be one of"):
        era5._normalise("ERA5", "yearly")
    assert era5._normalise("era5-land", "HOURLY") == ("ERA5_LAND", "hourly")
    with pytest.raises(ValueError, match="no source"):
        era5._pick_source("nope", "ERA5", "hourly")


# ── the ARCO route, offline ───────────────────────────────────────────────────


@pytest.mark.recorded
def test_load_arco_output_contract(arco):
    ds = era5.load(RAINIER, "2020-01-01", variables="2m_temperature")
    assert isinstance(ds, xr.Dataset) and list(ds.data_vars) == ["2m_temperature"]
    assert ds["2m_temperature"].dims == ("time", "latitude", "longitude")
    assert ds["2m_temperature"].chunks is not None  # lazy by default
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    # the AOI subset, in −180…180 longitudes
    assert float(ds.longitude.min()) < 0 and float(ds.longitude.max()) < 0
    assert -122.0 <= float(ds.longitude.min()) <= -121.5
    assert 46.5 <= float(ds.latitude.min()) <= 47.0
    assert ds.sizes["time"] == 6  # the whole synthetic day
    for key in ("source", "source_url", "product_id", "data_citation", "license"):
        assert ds.attrs[key]
    assert ds.attrs["product_id"] == "era5" and ds.attrs["source_id"] == "arco-era5-gcs"
    assert ds.attrs["easysnowdata_version"] == esd.__version__
    assert ds.attrs["version"] == "ERA5" and ds.attrs["cadence"] == "hourly"
    assert ds.attrs["valid_time_start"] == "2020-01-01"
    assert ds["2m_temperature"].attrs["units"] == "K"
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_load_arco_time_aoi_and_chunks(arco):
    full = era5.load(None, None)
    assert set(full.data_vars) == {"2m_temperature", "snow_depth"}
    assert full.sizes["longitude"] == 1440 and full.sizes["latitude"] == 721
    assert float(full.longitude.min()) == -180.0 and float(full.longitude.max()) < 180.0
    window = era5.load(RAINIER, ("2020-01-01T02:00", "2020-01-01T03:00"))
    assert window.sizes["time"] == 2
    chunked = era5.load(RAINIER, "2020-01-01", chunks={"time": 3})
    assert max(chunked["2m_temperature"].chunks[0]) == 3
    # a time range beyond the store's validity is clipped to it
    clipped = era5.load(RAINIER, "2020-01-01/2026-01-01")
    assert clipped.sizes["time"] == 6
    with pytest.raises(KeyError, match="not in ARCO-ERA5"):
        era5.load(RAINIER, "2020-01-01", variables=["nope"])


@pytest.mark.recorded
def test_load_arco_across_the_antimeridian_keeps_0_360(arco):
    ds = era5.load(DATELINE, "2020-01-01")
    assert ds.attrs["longitude_convention"] == "0-360"
    assert float(ds.longitude.min()) > 179 and float(ds.longitude.max()) > 179
    assert ds.sizes["longitude"] > 1


@pytest.mark.recorded
def test_search_arco_lists_variables(arco):
    frame = era5.search(RAINIER)
    assert list(frame.index) == ["2m_temperature", "snow_depth"]
    assert frame.loc["2m_temperature", "units"] == "K"
    assert frame.attrs["time_start"] == "2020-01-01"
    assert frame.attrs["source"] == "arco-era5-gcs"


# ── the Earth Engine route, mocked at the providers.gee boundary ──────────────


@pytest.fixture
def fake_gee(monkeypatch, fake_credentials):
    """Mock providers.gee so the GEE route can be exercised offline."""
    calls: dict[str, object] = {}

    class Collection:
        def __init__(self, asset):
            self.asset = asset
            self.dates = None
            self.bands = None

        def filterDate(self, start, end):
            self.dates = (start, end)
            return self

        def select(self, names):
            self.bands = names
            return self

        def first(self):
            return self

        def getInfo(self):
            return {"bands": [{"id": "temperature_2m"}, {"id": "snow_depth"}]}

    class FakeEE:
        ImageCollection = Collection

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = (collection, aoi, kwargs)
        lat = np.linspace(47.0, 46.7, 4)
        lon = np.linspace(-121.9, -121.5, 5)
        time = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"
        )
        data = np.zeros((time.size, lat.size, lon.size), dtype="float32")
        ds = xr.Dataset(
            {"temperature_2m": (("time", "y", "x"), data)},
            coords={"time": time, "y": lat, "x": lon},
        )
        return ds.chunk({"time": 1})

    monkeypatch.setattr(era5.providers.gee, "ee", lambda: FakeEE)
    monkeypatch.setattr(era5.providers.gee, "open_dataset", open_dataset)
    monkeypatch.setattr(era5.auth, "ensure", lambda *names, **kw: {})
    return calls


@pytest.mark.recorded
def test_load_gee_era5_land(fake_gee):
    ds = era5.load(
        RAINIER,
        ("2020-01-01", "2020-01-03"),
        version="ERA5_LAND",
        cadence="daily",
        variables="temperature_2m",
    )
    collection, aoi, kwargs = fake_gee["open"]
    assert collection.asset == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert collection.bands == ["temperature_2m"]
    assert collection.dates[0].startswith("2020-01-01")
    assert aoi.bounds == pytest.approx(RAINIER)
    assert kwargs["chunks"] == {}
    assert ds["temperature_2m"].dims == ("time", "latitude", "longitude")
    assert ds["temperature_2m"].chunks is not None
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    assert ds.attrs["source_id"] == "gee" and ds.attrs["version"] == "ERA5_LAND"
    assert ds.attrs["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"


@pytest.mark.recorded
def test_search_gee_lists_bands(fake_gee):
    frame = era5.search(version="ERA5_LAND", cadence="hourly")
    assert list(frame.index) == ["temperature_2m", "snow_depth"]
    assert frame.attrs["collection"] == "ECMWF/ERA5_LAND/HOURLY"


@pytest.mark.recorded
def test_gee_route_without_credentials_names_the_free_alternative(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        era5.load(RAINIER, "2020-01", version="ERA5_LAND")
    message = str(excinfo.value)
    assert excinfo.value.provider == "earthengine"
    assert 'source="arco-era5-gcs"' in message
    assert "era5" in message


# ── the deprecation shim ──────────────────────────────────────────────────────


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_arco_era5():
    ds = era5.load(
        RAINIER,
        ("2020-01-01", "2020-01-02"),
        variables=["2m_temperature", "snow_depth"],
    )
    assert ds["2m_temperature"].dims == ("time", "latitude", "longitude")
    assert ds["2m_temperature"].dtype == np.float32
    assert ds["2m_temperature"].chunks is not None
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    assert ds.attrs["product_id"] == "era5" and ds.attrs["license"]
    assert ds.attrs["valid_time_start"] == "1940-01-01"
    values = ds["2m_temperature"].isel(time=0).compute()
    assert 200 < float(values.mean()) < 330  # kelvin


@pytest.mark.live
def test_live_arco_search():
    frame = era5.search()
    assert "2m_temperature" in frame.index and len(frame) > 100
    assert frame.attrs["time_stop"] >= frame.attrs["time_stop_final"]


@pytest.mark.live
@pytest.mark.requires_earthengine
def test_live_gee_era5_land_daily():
    ds = era5.load(
        RAINIER,
        ("2020-01-01", "2020-01-03"),
        version="ERA5_LAND",
        cadence="daily",
        variables=["temperature_2m"],
    )
    assert ds["temperature_2m"].dims == ("time", "latitude", "longitude")
    assert ds.sizes["time"] == 3
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    assert ds.attrs["source_id"] == "gee"
