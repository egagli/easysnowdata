"""easysnowdata.snow.snodas — the NSIDC flat-binary reader on a synthetic day
that carries the real NOHRSC header layout, the Earth Engine mirror on stubs,
and the live smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.processing import snow as snow_processing
from easysnowdata.snow import snodas

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("snodas")
    assert product is snodas.PRODUCT
    assert product.loader == "easysnowdata.snow.snodas.load"
    assert product.resolve_loader() is snodas.load
    # NSIDC is now the credential-free default, the GEE mirror stays second
    assert [s.id for s in product.sources] == ["nsidc", "gee-climate-engine"]
    assert product.requires == ()
    assert product.source("gee-climate-engine").requires == ("earthengine",)
    labels = [p.label for s in product.sources for p in s.health]
    # the label the weekly health check has been recording is unchanged
    assert "SNODAS (GEE/Climate Engine)" in labels
    assert labels[0] == "SNODAS (NSIDC G02158)"
    assert "SNODAS latency (NSIDC G02158)" in labels
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_urls_variables_and_day_lists():
    assert snodas.tar_url("2024-03-15") == (
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2024/03_Mar/"
        "SNODAS_20240315.tar"
    )
    assert "SNODAS_unmasked_20240315.tar" in snodas.tar_url(
        "2024-03-15", region="unmasked"
    )
    with pytest.raises(ValueError, match="region must be one of"):
        snodas.tar_url("2024-03-15", region="global")
    assert snodas._variables(None) == ["SWE", "snow_depth"]
    assert snodas._variables("snow_melt") == ["snow_melt"]
    with pytest.raises(ValueError, match="Invalid variables"):
        snodas._variables(["NotAVariable"])
    days = snodas._days(("2024-03-01", "2024-03-03"))
    assert len(days) == 3 and days[0] == pd.Timestamp("2024-03-01")
    with pytest.raises(ValueError, match="needs a start date"):
        snodas._days(None)
    assert snodas._variable_of("us_ssmv11034tS__T0001TTNATS2024031505HP001") == "SWE"
    assert snodas._variable_of("us_ssmv11036tS__T0001") == "snow_depth"
    assert snodas._variable_of("us_ssmv99999") is None


def test_search_lists_the_days_and_their_urls():
    frame = snodas.search(RAINIER, ("2024-03-01", "2024-03-03"))
    assert list(frame["date"]) == list(pd.date_range("2024-03-01", "2024-03-03"))
    assert frame["url"].iloc[0].endswith("SNODAS_20240301.tar")
    assert frame.attrs == {"source": "nsidc", "region": "masked"}


# ── the NOHRSC reader (pure processing) ──────────────────────────────────────


def test_parse_snodas_header_and_array(ts_fixtures):
    members = snow_processing.snodas_members(ts_fixtures["snodas_tar"].read_bytes())
    assert len(members) == 2
    stem = next(s for s in members if "11034" in s)
    header = snow_processing.parse_snodas_header(members[stem]["txt"].decode())
    assert header["number_of_rows"] == 40 and header["number_of_columns"] == 60
    assert header["no_data_value"] == -9999
    assert header["scale_factor"] == pytest.approx(1 / 1000)
    assert header["units"] == "m"
    assert header["description"].startswith("Modeled snow water equivalent")

    da = snow_processing.snodas_array(members[stem]["dat"], header, name="SWE")
    assert da.dims == ("latitude", "longitude")
    assert da.rio.crs.to_epsg() == 4326
    assert np.isnan(da.values[0, 0])  # the fixture's nodata corner
    assert 0.0 <= float(np.nanmax(da)) <= 2.0  # millimetres scaled to metres
    with pytest.raises(ValueError, match="expected"):
        snow_processing.snodas_array(members[stem]["dat"][:100], header)


# ── the NSIDC route ───────────────────────────────────────────────────────────


@pytest.fixture
def fake_nsidc(monkeypatch, ts_fixtures):
    """Serve the synthetic tar for every requested day."""
    calls: list[str] = []

    def fetch(url, fname=None, *, subdir=None, **kwargs):
        calls.append(url)
        if "20240316" in url:  # one missing day, to exercise the skip
            raise RuntimeError("HTTP 404")
        return ts_fixtures["snodas_tar"]

    monkeypatch.setattr(snodas.providers.raster_http, "fetch", fetch)
    return calls


@pytest.mark.recorded
def test_load_nsidc_output_contract(fake_nsidc):
    ds = snodas.load(RAINIER, ("2024-03-15", "2024-03-15"))
    assert fake_nsidc[0].endswith("SNODAS_20240315.tar")
    assert set(ds.data_vars) == {"SWE", "snow_depth"}
    assert ds["SWE"].dims == ("time", "latitude", "longitude")
    assert ds.sizes["time"] == 1
    assert ds["SWE"].dtype == np.float32
    assert ds["SWE"].chunks is not None  # lazy by default
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    assert ds["SWE"].attrs["units"] == "m"
    assert ds["snow_depth"].attrs["long_name"].startswith("snow layer thickness")
    assert ds.attrs["product_id"] == "snodas" and ds.attrs["source_id"] == "nsidc"
    assert ds.attrs["region"] == "masked"
    assert ds.attrs["license"].startswith("Public domain")
    assert all(not callable(v) for v in ds.attrs.values())
    # the AOI subset is smaller than the fixture's 60×40 grid
    assert ds.sizes["longitude"] < 60 and ds.sizes["latitude"] < 40


@pytest.mark.recorded
def test_load_nsidc_skips_missing_days_and_stacks_the_rest(fake_nsidc, caplog):
    with caplog.at_level("WARNING"):
        ds = snodas.load(RAINIER, ("2024-03-15", "2024-03-17"), variables="SWE")
    assert ds.sizes["time"] == 2  # 2024-03-16 is missing in the stub
    assert "not available" in caplog.text
    assert list(ds.data_vars) == ["SWE"]
    assert ds["time"].to_index().is_monotonic_increasing


@pytest.mark.recorded
def test_load_nsidc_when_no_day_is_available(monkeypatch):
    def fetch(url, fname=None, **kwargs):
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr(snodas.providers.raster_http, "fetch", fetch)
    with pytest.raises(ValueError, match="No SNODAS days available"):
        snodas.load(RAINIER, ("2024-03-15", "2024-03-16"))


@pytest.mark.recorded
def test_load_nsidc_unclipped_and_chunked(fake_nsidc):
    whole = snodas.load(
        esd.parse_aoi(RAINIER, clip=False), ("2024-03-15", "2024-03-15")
    )
    assert whole.sizes["longitude"] == 60 and whole.sizes["latitude"] == 40
    chunked = snodas.load(RAINIER, ("2024-03-15", "2024-03-15"), chunks={"latitude": 5})
    assert max(chunked["SWE"].chunks[1]) == 5


# ── the Earth Engine mirror ───────────────────────────────────────────────────


@pytest.fixture
def fake_gee(monkeypatch, fake_credentials):
    calls: dict[str, object] = {}

    class Collection:
        def __init__(self, asset):
            self.asset = asset
            self.bands = None

        def filterDate(self, start, end):
            calls["dates"] = (start, end)
            return self

        def select(self, bands):
            self.bands = bands
            calls["bands"] = bands
            return self

    class FakeEE:
        ImageCollection = Collection

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = kwargs
        time = pd.to_datetime(["2020-01-01", "2020-01-02"]).values
        shape = (2, 3, 4)
        return xr.Dataset(
            {
                "SWE": (("time", "y", "x"), np.full(shape, 0.5, dtype="float32")),
                "Snow_Depth": (
                    ("time", "y", "x"),
                    np.full(shape, 1.5, dtype="float32"),
                ),
            },
            coords={
                "time": time,
                "y": np.linspace(47.0, 46.7, 3),
                "x": np.linspace(-121.9, -121.5, 4),
            },
        ).chunk({"time": 1})

    monkeypatch.setattr(snodas.providers.gee, "ee", lambda: FakeEE)
    monkeypatch.setattr(snodas.providers.gee, "open_dataset", open_dataset)
    return calls


@pytest.mark.recorded
def test_load_gee_mirror(fake_gee):
    ds = snodas.load(RAINIER, ("2020-01-01", "2020-01-02"), source="gee-climate-engine")
    assert fake_gee["bands"] == ["SWE", "Snow_Depth"]
    assert set(ds.data_vars) == {"SWE", "snow_depth"}  # renamed to our names
    assert ds["SWE"].dims == ("time", "latitude", "longitude")
    assert ds["SWE"].chunks is not None
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    assert ds["snow_depth"].attrs["units"] == "m"
    assert ds.attrs["source_id"] == "gee-climate-engine"
    assert ds.attrs["collection"] == snodas.GEE_COLLECTION


@pytest.mark.recorded
def test_gee_mirror_refuses_the_variables_it_lacks(fake_gee):
    with pytest.raises(ValueError, match="carries only"):
        snodas.load(
            RAINIER,
            ("2020-01-01", "2020-01-02"),
            variables="snow_melt",
            source="gee-climate-engine",
        )


@pytest.mark.recorded
def test_gee_route_without_credentials_names_the_free_alternative(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        snodas.load(RAINIER, ("2020-01-01", "2020-01-02"), source="gee-climate-engine")
    assert excinfo.value.provider == "earthengine"
    assert 'source="nsidc"' in str(excinfo.value)


# ── the deprecation shim ──────────────────────────────────────────────────────


@pytest.mark.recorded
def test_old_get_snodas_keeps_the_earth_engine_route(fake_gee):
    from easysnowdata import _deprecation, hydroclimatology

    _deprecation.reset_warnings()
    with pytest.warns(
        _deprecation.EasysnowdataDeprecationWarning, match="snow.snodas.load"
    ):
        ds = hydroclimatology.get_snodas(
            bbox_input=RAINIER,
            start_date="2020-01-01",
            end_date="2020-01-02",
            initialize_ee=False,
        )
    assert ds["SWE"].dims == ("time", "latitude", "longitude")
    assert ds.attrs["source_id"] == "gee-climate-engine"


def test_old_get_snodas_keeps_its_variable_validation():
    from easysnowdata import hydroclimatology

    with pytest.raises(ValueError, match="Invalid variables"):
        hydroclimatology.get_snodas(
            bbox_input=RAINIER,
            start_date="2020-01-01",
            end_date="2020-01-03",
            variables=["NotAVariable"],
        )


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_snodas_nsidc_needs_no_account():
    ds = snodas.load(RAINIER, ("2024-03-15", "2024-03-16"))
    assert set(ds.data_vars) == {"SWE", "snow_depth"}
    assert ds["SWE"].dims == ("time", "latitude", "longitude")
    assert ds.sizes["time"] == 2
    assert ds.rio.crs.to_epsg() == 4326 and ds.odc.crs.epsg == 4326
    swe = ds["SWE"].isel(time=0).compute()
    assert float(np.nanmedian(swe)) > 0.0  # mid-March on Rainier has snow
    assert 0.0 < float(np.nanmedian(swe)) < 5.0  # metres of water equivalent
    # SNODAS never melts glaciers out, so SWE saturates the 16-bit field at
    # 32.767 m over Rainier's ice cap; those values are the model's own.
    assert float(np.nanmax(swe)) == pytest.approx(32.767, abs=1e-3)
    assert ds.attrs["source_id"] == "nsidc" and ds.attrs["doi"] == "10.7265/N5TB14TC"


@pytest.mark.live
@pytest.mark.requires_earthengine
def test_live_snodas_sources_agree():
    """The authoritative archive and the Earth Engine mirror should match."""
    when = ("2024-03-15", "2024-03-15")
    nsidc = snodas.load(RAINIER, when, variables="SWE")
    mirror = snodas.load(RAINIER, when, variables="SWE", source="gee-climate-engine")
    # compare medians: the glacier pixels that saturate the 16-bit field would
    # dominate a mean, and the two grids resample them differently
    nsidc_median = float(np.nanmedian(nsidc["SWE"].isel(time=0).compute()))
    mirror_median = float(np.nanmedian(mirror["SWE"].isel(time=0).compute()))
    assert abs(nsidc_median - mirror_median) < 0.1  # metres of water equivalent
