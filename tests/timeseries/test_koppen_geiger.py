"""easysnowdata.climate.koppen_geiger — offline tier on the synthetic figshare
archive, plus the live smoke test against the real figshare file."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.climate import koppen_geiger as kg

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def archive(monkeypatch, ts_fixtures):
    """Point the figshare route at the local synthetic zip."""
    url = ts_fixtures["koppen_zip"].as_uri()
    monkeypatch.setattr(kg, "ZIP_URL", url)
    return url


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("koppen-geiger")
    assert product is kg.PRODUCT
    assert product.loader == "easysnowdata.climate.koppen_geiger.load"
    assert product.resolve_loader() is kg.load
    assert [s.id for s in product.sources] == ["figshare"]
    assert product.default_source.requires == ()
    assert [p.label for p in product.default_source.health] == [
        "Köppen-Geiger classification (figshare)"
    ]
    assert "61012822" in product.default_source.location
    variable = product.variables[0]
    assert len(variable.flag_values) == 30 and variable.flag_meanings[0] == "Af"
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_member_paths_and_resolution_tokens():
    assert kg._member("1991_2020", None, "1p0") == "1991_2020/koppen_geiger_1p0.tif"
    assert kg._member("1991-2020", None, "0p1") == "1991_2020/koppen_geiger_0p1.tif"
    assert (
        kg._member("2071_2099", "SSP585", "0p00833333")
        == "2071_2099/ssp585/koppen_geiger_0p00833333.tif"
    )
    # a future period defaults to ssp245
    assert (
        kg._member("2041_2070", None, "1p0") == "2041_2070/ssp245/koppen_geiger_1p0.tif"
    )
    with pytest.raises(ValueError, match="takes no scenario"):
        kg._member("1991_2020", "ssp245", "1p0")
    with pytest.raises(ValueError, match="scenario must be one of"):
        kg._member("2071_2099", "rcp85", "1p0")
    with pytest.raises(ValueError, match="period must be one of"):
        kg._member("2100_2200", None, "1p0")
    assert kg._resolution_token("1 km") == "0p00833333"
    assert kg._resolution_token("0p5") == "0p5"
    with pytest.raises(ValueError, match="resolution must be one of"):
        kg._resolution_token("30 m")


def test_class_table_matches_the_flags():
    table = kg.class_table()
    assert len(table) == 30
    assert table.loc[29, "symbol"] == "ET" and table.loc[29, "color"] == "#b2b2b2"
    assert list(table.index) == list(kg.PRODUCT.variables[0].flag_values)


def test_search_lists_every_raster_in_the_archive():
    frame = kg.search()
    assert len(frame) == (4 + 2 * 7) * 4
    historical = frame[frame["period"] == "1991_2020"]
    assert historical["scenario"].isna().all() and len(historical) == 4
    future = frame[(frame["period"] == "2071_2099") & (frame["resolution"] == "1 km")]
    assert set(future["scenario"]) == set(kg.SCENARIOS)
    assert future["member"].str.endswith("koppen_geiger_0p00833333.tif").all()
    assert frame.attrs["source"] == "figshare"


# ── loading, offline ──────────────────────────────────────────────────────────


@pytest.mark.recorded
def test_load_output_contract(archive):
    da = kg.load(RAINIER, resolution="1 degree")
    assert isinstance(da, xr.DataArray) and da.name == "koppen_geiger_class"
    assert da.dtype == np.uint8  # categorical: the source values are kept
    assert da.dims == ("latitude", "longitude")
    assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
    assert da.rio.nodata == 0 and da.attrs["nodata"] == 0
    assert da.attrs["flag_values"][:3] == [1, 2, 3]
    assert da.attrs["flag_meanings"].split()[:2] == ["Af", "Am"]
    assert da.attrs["flag_colors"].split()[0] == "#0000ff"
    assert da.attrs["product_id"] == "koppen-geiger"
    assert da.attrs["period"] == "1991_2020"
    assert "scenario" not in da.attrs  # historical periods have none
    assert da.attrs["archive_member"] == "1991_2020/koppen_geiger_1p0.tif"
    assert da.attrs["easysnowdata_version"] == esd.__version__
    assert all(not callable(v) for v in da.attrs.values())
    # a sub-pixel AOI at 1° must not raise (allow_one_dimensional_raster)
    assert da.size >= 1
    assert da.chunks is not None


@pytest.mark.recorded
def test_load_period_scenario_and_global(archive):
    future = kg.load(
        RAINIER, period="2071_2099", scenario="ssp245", resolution="1 degree"
    )
    assert future.attrs["period"] == "2071_2099"
    assert future.attrs["scenario"] == "ssp245"
    assert future.attrs["archive_member"] == "2071_2099/ssp245/koppen_geiger_1p0.tif"
    present = kg.load(RAINIER, resolution="1 degree")
    assert not np.array_equal(future.values, present.values)

    whole = kg.load(None, resolution="1 degree")
    assert whole.sizes == {"latitude": 180, "longitude": 360}
    unclipped = kg.load(esd.parse_aoi(RAINIER, clip=False), resolution="1 degree")
    assert unclipped.sizes == whole.sizes

    with pytest.raises(ValueError, match="period must be one of"):
        kg.load(RAINIER, period="1800_1900")


@pytest.mark.recorded
def test_plotting_reads_the_flags(archive):
    import matplotlib

    matplotlib.use("Agg")
    da = kg.load(RAINIER, resolution="1 degree")
    cmap, norm, table = esd.plotting.colormap_from_flags(da)
    assert cmap.N == 30 and len(table) == 30
    assert norm.boundaries[0] == pytest.approx(0.5)
    handles, labels = esd.plotting.legend_handles(da)
    assert labels[:2] == ["Af", "Am"] and len(handles) == 30


# ── the deprecation shim ──────────────────────────────────────────────────────


@pytest.mark.recorded
def test_old_get_koppen_geiger_classes_still_works(archive):
    from easysnowdata import _deprecation, hydroclimatology

    _deprecation.reset_warnings()
    with pytest.warns(
        _deprecation.EasysnowdataDeprecationWarning, match="koppen_geiger.load"
    ):
        da = hydroclimatology.get_koppen_geiger_classes(
            bbox_input=RAINIER, resolution="1 degree"
        )
    assert da.dims == ("latitude", "longitude") and da.dtype == np.uint8
    assert da.chunks is None  # the old default was an eager read
    assert "class_info" not in da.attrs and "cmap" not in da.attrs
    assert "flag_values" in da.attrs


@pytest.mark.recorded
def test_old_loader_forwards_kwargs(archive):
    from easysnowdata import hydroclimatology

    da = hydroclimatology.get_koppen_geiger_classes(
        bbox_input=RAINIER, resolution="1 degree", chunks={"x": 90, "y": 90}
    )
    assert da.chunks is not None


# ── live smoke test ───────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_koppen_geiger_figshare():
    da = kg.load(RAINIER, resolution="1 km")
    assert da.dims == ("latitude", "longitude") and da.dtype == np.uint8
    assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
    assert da.rio.nodata == 0
    values = set(np.unique(da.compute().values.ravel()).tolist())
    assert values <= set(kg.CLASSES) | {0}
    assert values & {25, 26, 27, 29}  # Rainier is Dfa/Dfb/Dfc/ET country
    assert (
        da.attrs["product_id"] == "koppen-geiger" and da.attrs["license"] == "CC BY 4.0"
    )


@pytest.mark.live
def test_live_koppen_geiger_projection_period():
    da = kg.load(RAINIER, period="2071_2099", scenario="ssp585", resolution="1 degree")
    assert da.attrs["archive_member"] == "2071_2099/ssp585/koppen_geiger_1p0.tif"
    assert int(da.compute().max()) in kg.CLASSES
