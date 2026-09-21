"""easysnowdata.snow.ucla_sr — the reanalysis reader on synthetic NetCDF water
years, the virtualization switch, and the live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.snow import ucla_sr

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("ucla-snow-reanalysis")
    assert product is ucla_sr.PRODUCT
    assert product.loader == "easysnowdata.snow.ucla_sr.load"
    assert product.resolve_loader() is ucla_sr.load
    assert [s.id for s in product.sources] == ["nsidc", "nsidc-hma"]
    assert product.requires == ("earthdata",)
    # the labels the weekly health check has been recording are unchanged
    labels = [p.label for s in product.sources for p in s.health if p.kind == "health"]
    assert labels == [
        "UCLA Snow Reanalysis (NASA NSIDC)",
        "HMA Snow Reanalysis (NASA NSIDC)",
    ]
    # ...and the DMR++ probe rides alongside without needing credentials (§8)
    virtualization = [
        p for s in product.sources for p in s.health if p.kind == "virtualization"
    ]
    assert [p.requires for p in virtualization] == [()]
    assert {v.name for v in product.variables} == {"SWE_Post", "SCA_Post", "SD_Post"}
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_regions_stats_and_water_years():
    assert ucla_sr._region("HMA") == "hma"
    with pytest.raises(ValueError, match="region must be one of"):
        ucla_sr._region("alps")
    # Phase 0's fix stays: every statistic has its own index
    assert ucla_sr.STATS == {"mean": 0, "std": 1, "median": 2, "25pct": 3, "75pct": 4}
    assert len(set(ucla_sr.STATS.values())) == len(ucla_sr.STATS)
    with pytest.raises(ValueError, match="stats must be one of"):
        ucla_sr._stats_index("mode")
    assert ucla_sr.water_year_start(
        "WUS_UCLA_SR_v01_N46_0W122_0_agg_16_WY2019_20_SWE_SCA_POST.nc"
    ) == pd.Timestamp("2019-10-01")
    assert ucla_sr.water_year_start(
        "s3://bucket/HMA/HMA_SR_D/1/1999/10/01/granule.nc"
    ) == pd.Timestamp("1999-10-01")
    with pytest.raises(ValueError, match="Could not determine"):
        ucla_sr.water_year_start("mystery.nc")


def test_virtualize_switch():
    assert ucla_sr._should_virtualize("auto", 2) is False
    assert ucla_sr._should_virtualize("auto", ucla_sr.VIRTUALIZE_THRESHOLD + 1) is True
    assert ucla_sr._should_virtualize(True, 1) is True
    assert ucla_sr._should_virtualize(False, 100) is False
    with pytest.raises(ValueError, match="virtualize must be"):
        ucla_sr._should_virtualize("sometimes", 10)


def test_access_mode_follows_the_compute_region(monkeypatch):
    monkeypatch.setenv("EASYSNOWDATA_REGION", "us-west-2")
    esd.config.region(refresh=True)
    assert ucla_sr._resolve_access("auto") == "direct"
    monkeypatch.setenv("EASYSNOWDATA_REGION", "local")
    esd.config.region(refresh=True)
    assert ucla_sr._resolve_access("auto") == "indirect"
    assert ucla_sr._resolve_access("direct") == "direct"
    with pytest.raises(ValueError, match="access must be"):
        ucla_sr._resolve_access("s3")
    esd.config.region(refresh=True)


# ── loading, on synthetic water years ────────────────────────────────────────


@pytest.fixture
def fake_nsidc(monkeypatch, ts_fixtures, fake_credentials):
    """earthaccess stand-in: the granules are the synthetic NetCDF water years."""
    calls: dict[str, object] = {}
    paths = sorted(ts_fixtures["ucla_dir"].glob("*.nc"))

    class Granule(dict):
        def __init__(self, path):
            super().__init__(id=path.name)
            self.path = str(path)

    def search(short_name, aoi=None, time=None, **kwargs):
        calls["search"] = (short_name, kwargs)
        return [Granule(p) for p in paths]

    def open_(granules, **kwargs):
        calls["open"] = len(granules)
        return [g.path for g in granules]

    monkeypatch.setattr(ucla_sr.providers.earthdata, "search", search)
    monkeypatch.setattr(ucla_sr.providers.earthdata, "open", open_)
    return calls


@pytest.mark.recorded
def test_load_output_contract(fake_nsidc):
    da = ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-10"))
    assert fake_nsidc["search"][0] == "WUS_UCLA_SR"
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "latitude", "longitude")
    assert da.dtype == np.float32
    assert da.chunks is not None
    assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
    assert str(da["time"].values[0])[:10] == "2019-10-01"
    assert da.attrs["product_id"] == "ucla-snow-reanalysis"
    assert da.attrs["region"] == "wus" and da.attrs["short_name"] == "WUS_UCLA_SR"
    assert da.attrs["statistic"] == "mean" and da.attrs["variable"] == "SWE_Post"
    assert da.attrs["units"] == "m"
    assert da.attrs["virtualized"] == "False"  # two granules is below the threshold
    assert da.attrs["access"] in {"direct", "indirect"}
    assert da.attrs["data_citation"].startswith("Fang")
    assert all(not callable(v) for v in da.attrs.values())


@pytest.mark.recorded
def test_statistics_are_distinct_slices(fake_nsidc):
    common = dict(aoi=RAINIER, time=("2019-10-01", "2019-10-05"))
    median = ucla_sr.load(**common, stats="median").compute()
    q25 = ucla_sr.load(**common, stats="25pct").compute()
    q75 = ucla_sr.load(**common, stats="75pct").compute()
    # the fixture builds the percentiles around the median, as the real files do
    assert not np.allclose(median.values, q25.values)
    assert bool((q25 <= median).all()) and bool((median <= q75).all())


@pytest.mark.recorded
def test_variables_regions_and_validation(fake_nsidc):
    sca = ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-03"), variable="SCA_Post")
    assert sca.attrs["variable"] == "SCA_Post" and sca.attrs["units"] == "1"

    ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-03"), region="hma")
    assert fake_nsidc["search"][0] == "HMA_SR_D"

    with pytest.raises(ValueError, match="variable must be one of"):
        ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-03"), variable="SWE")


@pytest.mark.recorded
def test_invalid_stats_raises_before_any_network_call(monkeypatch, fake_credentials):
    monkeypatch.setattr(
        ucla_sr.providers.earthdata,
        "search",
        lambda *a, **k: pytest.fail("no search must run"),
    )
    monkeypatch.setattr(
        ucla_sr.providers.earthdata,
        "open",
        lambda *a, **k: pytest.fail("no open must run"),
    )
    with pytest.raises(ValueError, match="stats must be one of"):
        ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-03"), stats="mode")


@pytest.mark.recorded
def test_no_granules_names_the_coverage(monkeypatch, fake_credentials):
    monkeypatch.setattr(ucla_sr.providers.earthdata, "search", lambda *a, **k: [])
    with pytest.raises(ValueError, match="1984-10/2021-09"):
        ucla_sr.load(RAINIER, ("1970-10-01", "1970-10-03"))


@pytest.mark.recorded
def test_virtualization_falls_back_when_it_fails(fake_nsidc, monkeypatch, caplog):
    import earthaccess

    def boom(*args, **kwargs):
        raise RuntimeError("no DMR++ sidecars and the HDF5 parser gave up")

    monkeypatch.setattr(earthaccess, "virtualize", boom)
    with caplog.at_level("WARNING"):
        da = ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-05"), virtualize=True)
    assert "Virtualization failed" in caplog.text
    assert da.dims == ("time", "latitude", "longitude")  # the plain path took over
    assert da.attrs["virtualized"] == "True"  # what was asked for is recorded


@pytest.mark.recorded
def test_virtualization_uses_the_reference_cache(fake_nsidc, monkeypatch, tmp_path):
    import earthaccess

    seen: dict[str, object] = {}

    def fake_virtualize(granules, **kwargs):
        seen.update(kwargs)
        seen["granules"] = len(granules)
        return xr.Dataset(
            {
                "SWE_Post": (
                    ("Day", "Stats", "Latitude", "Longitude"),
                    np.zeros((12, 5, 4, 4), dtype="float32"),
                )
            },
            coords={
                "Day": np.arange(12),
                "Latitude": np.linspace(46.99, 46.72, 4),
                "Longitude": np.linspace(-121.94, -121.54, 4),
            },
        )

    monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(earthaccess, "virtualize", fake_virtualize)
    da = ucla_sr.load(RAINIER, ("2019-10-01", "2019-10-05"), virtualize=True)
    assert seen["load"] is True and seen["concat_dim"] == "Day"
    assert seen["access"] in {"direct", "indirect"}
    assert str(tmp_path) in seen["reference_dir"]
    assert da.attrs["virtualized"] == "True"
    assert da.dims == ("time", "latitude", "longitude")


# ── the deprecation shim ──────────────────────────────────────────────────────


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_ucla_sr_search():
    gdf = ucla_sr.search(RAINIER, ("2020-01-01", "2020-01-31"))
    assert len(gdf) >= 1
    assert gdf.attrs["short_name"] == "WUS_UCLA_SR"


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_ucla_sr_load_one_water_year():
    da = ucla_sr.load(RAINIER, ("2020-01-01", "2020-01-31"))
    assert da.dims == ("time", "latitude", "longitude")
    assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
    assert da.attrs["units"] == "m" and da.attrs["statistic"] == "mean"
    assert da.attrs["virtualized"] == "False"  # one water year needs no references
    values = da.isel(time=0).compute()
    # Not a maximum: the reanalysis accumulates unbounded SWE over perennial
    # ice, exactly as SNODAS does, so the Rainier summit ice cap reaches ~106 m
    # while the field around it is seasonal. Assert on the bulk of the
    # distribution and on the glacier tail being a tail (module docstring).
    assert float(np.nanmin(values)) >= 0.0
    assert 0.0 < float(np.nanmedian(values)) < 5.0
    assert float(np.nanpercentile(values, 90)) < 10.0
    seasonal = values.where(values < 10)
    assert float(seasonal.isnull().mean()) < 0.1  # the ice cap is a few % of pixels


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_ucla_sr_percentiles_bracket_the_median():
    common = dict(aoi=RAINIER, time=("2020-03-01", "2020-03-02"))
    q25 = ucla_sr.load(**common, stats="25pct").compute()
    median = ucla_sr.load(**common, stats="median").compute()
    q75 = ucla_sr.load(**common, stats="75pct").compute()
    assert not median.equals(q25)
    assert bool((q25 <= median).all()) and bool((median <= q75).all())


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_hma_sibling_is_reachable():
    gdf = ucla_sr.search(
        (80.0, 30.0, 81.0, 31.0), ("2000-03-01", "2000-03-07"), region="hma"
    )
    assert len(gdf) >= 1
    assert gdf.attrs["short_name"] == "HMA_SR_D"
    assert isinstance(gdf, gpd.GeoDataFrame)


@pytest.mark.recorded
def test_granule_path_handles_every_granule_shape():
    class WithPath:
        path = "/cache/WY2019_20.nc"

    class WithLinks:
        def data_links(self):
            return ["https://data.nsidc.example/WY2019_20.nc"]

    class Broken:
        def data_links(self):
            raise RuntimeError("no links")

    assert ucla_sr._granule_path(WithPath()) == "/cache/WY2019_20.nc"
    assert ucla_sr._granule_path(WithLinks()).endswith("WY2019_20.nc")
    assert ucla_sr._granule_path({"id": "granule-id"}) == "granule-id"
    assert "Broken" in ucla_sr._granule_path(Broken())
