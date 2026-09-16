"""easysnowdata.snow.viirs — the HDF-EOS5 reader on a synthetic granule that
carries the real ``StructMetadata.0`` layout, plus the live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.snow import viirs

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("viirs-snow")
    assert product is viirs.PRODUCT
    assert product.loader == "easysnowdata.snow.viirs.load"
    assert product.resolve_loader() is viirs.load
    assert [s.id for s in product.sources] == ["nsidc"]
    assert product.requires == ("earthdata",)
    assert product.credential_free_sources == ()
    assert [p.label for p in product.default_source.health] == [
        "VIIRS snow cover VNP10A1F (NASA NSIDC)"
    ]
    assert product.default_source.resolution_m == 375
    names = {v.name for v in product.variables}
    assert {"NDSI_Snow_Cover", "CGF_NDSI_Snow_Cover", "Cloud_Persistence"} <= names
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_products_and_granule_names():
    assert viirs._product("vnp10a1f") == "VNP10A1F"
    assert "VJ110A1F" in viirs.PRODUCTS  # the NOAA-20 sibling
    with pytest.raises(ValueError, match="product must be one of"):
        viirs._product("VNP09GA")
    assert viirs.DEFAULT_VARIABLES["VNP10A1"] == ("NDSI_Snow_Cover",)
    assert viirs.granule_date(
        "VNP10A1F.A2023060.h09v04.002.2023062.h5"
    ) == pd.Timestamp("2023-03-01")
    assert viirs.granule_date("no-date.h5") is None
    assert viirs.GRID_GROUP.endswith("VIIRS_Grid_IMG_2D/Data Fields")


# ── the HDF-EOS5 reader ───────────────────────────────────────────────────────


def test_read_struct_metadata_gives_the_sinusoidal_grid(ts_fixtures):
    grid = viirs.read_struct_metadata(ts_fixtures["viirs_h5"])
    assert grid["columns"] > 0 and grid["rows"] > 0
    assert grid["upper_left_x"] < grid["lower_right_x"]
    assert grid["upper_left_y"] > grid["lower_right_y"]
    # the AOI is in the northern hemisphere, west of the prime meridian
    assert grid["upper_left_x"] < 0 and grid["upper_left_y"] > 0


def test_open_granule_georeferences_the_phony_dims(ts_fixtures):
    ds = viirs._open_granule(
        ts_fixtures["viirs_h5"], ["CGF_NDSI_Snow_Cover"], True, None
    )
    assert set(ds.dims) == {"y", "x"}
    assert ds["CGF_NDSI_Snow_Cover"].dtype == np.uint8
    assert ds["CGF_NDSI_Snow_Cover"].chunks is not None
    assert ds.rio.crs is not None and ds.rio.crs.is_projected
    # 375 m pixels on the sinusoidal grid
    spacing = float(np.abs(np.diff(ds["x"].values[:2])[0]))
    assert 370 < spacing < 380
    with pytest.raises(KeyError, match="has no"):
        viirs._open_granule(ts_fixtures["viirs_h5"], ["Nope"], True, None)


@pytest.fixture
def fake_nsidc(monkeypatch, ts_fixtures, fake_credentials):
    calls: dict[str, object] = {}
    path = ts_fixtures["viirs_h5"]

    def search(short_name, aoi=None, time=None, **kwargs):
        calls["search"] = (short_name, kwargs)
        return [{"id": path.name, "path": path}]

    def download(granules, subdir, **kwargs):
        calls["download"] = subdir
        return [g["path"] for g in granules]

    monkeypatch.setattr(viirs.providers.earthdata, "search", search)
    monkeypatch.setattr(viirs.providers.earthdata, "download", download)
    return calls


@pytest.mark.recorded
def test_load_output_contract(fake_nsidc):
    ds = viirs.load(RAINIER, "2023-03-01/2023-03-02")
    assert fake_nsidc["search"][0] == "VNP10A1F"
    assert fake_nsidc["search"][1]["version"] == "2"
    assert fake_nsidc["download"] == "VNP10A1F"

    band = ds["CGF_NDSI_Snow_Cover"]
    assert band.dims == ("time", "y", "x") and ds.sizes["time"] == 1
    assert str(ds["time"].values[0])[:10] == "2023-03-01"
    assert band.dtype == np.uint8 and band.rio.nodata == 255
    assert band.attrs["units"] == "%"
    assert 250 in band.attrs["flag_values"]  # cloud stays meaningful
    assert ds.rio.crs is not None and ds.odc.crs is not None
    assert ds.attrs["product_id"] == "viirs-snow"
    assert ds.attrs["viirs_product"] == "VNP10A1F" and ds.attrs["version"] == "2"
    assert ds.attrs["easysnowdata_version"] == esd.__version__
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_load_variables_masking_and_errors(fake_nsidc, monkeypatch):
    several = viirs.load(
        RAINIER,
        "2023-03-01",
        variables=["CGF_NDSI_Snow_Cover", "Cloud_Persistence", "Basic_QA"],
    )
    assert set(several.data_vars) == {
        "CGF_NDSI_Snow_Cover",
        "Cloud_Persistence",
        "Basic_QA",
    }
    assert several["Basic_QA"].attrs["flag_meanings"].startswith("best good")

    masked = viirs.load(RAINIER, "2023-03-01", mask=True)
    assert masked["CGF_NDSI_Snow_Cover"].dtype == np.float32

    with pytest.raises(TypeError, match="not the GeoDataFrame"):
        viirs.load(RAINIER, "2023-03-01", granules=gpd.GeoDataFrame())
    monkeypatch.setattr(viirs.providers.earthdata, "search", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No VNP10A1F granules"):
        viirs.load(RAINIER, "1999-01")


@pytest.mark.recorded
def test_search_returns_a_granule_frame(fake_nsidc, monkeypatch):
    monkeypatch.setattr(
        viirs.providers.earthdata,
        "granules_to_geodataframe",
        lambda granules: gpd.GeoDataFrame(
            {"id": [g["id"] for g in granules]},
            geometry=[None] * len(granules),
            crs="EPSG:4326",
        ),
    )
    gdf = viirs.search(RAINIER, "2023-03", product="VNP10A1")
    assert len(gdf) == 1 and gdf["date"].iloc[0] == pd.Timestamp("2023-03-01")
    assert gdf.attrs == {"source": "nsidc", "product": "VNP10A1"}


@pytest.mark.recorded
def test_without_credentials_the_error_names_the_provider(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        viirs.load(RAINIER, "2023-03")
    assert excinfo.value.provider == "earthdata"
    # VIIRS has no credential-free route, so none is offered
    assert 'source="' not in str(excinfo.value)


@pytest.mark.recorded
def test_binary_snow_works_on_the_viirs_byte(fake_nsidc):
    ds = viirs.load(RAINIER, "2023-03-01")
    binary = esd.processing.binary_snow(
        ds["CGF_NDSI_Snow_Cover"], product="VNP10A1F", threshold=40
    ).compute()
    values = binary.values[~np.isnan(binary.values)]
    assert set(np.unique(values)) <= {0.0, 1.0}
    assert binary.attrs["source_product"] == "VNP10A1F"
    assert np.isnan(binary.values).any()  # the fixture's cloud and fill pixels


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_viirs_search():
    gdf = viirs.search(RAINIER, "2023-03-01/2023-03-03", product="VNP10A1F")
    assert len(gdf) >= 1
    assert gdf["id"].iloc[0].startswith("VNP10A1F")
    assert gdf["date"].notna().all()


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_viirs_load_cloud_gap_filled():
    ds = viirs.load(RAINIER, "2023-03-01/2023-03-02", product="VNP10A1F")
    band = ds["CGF_NDSI_Snow_Cover"]
    assert band.dims == ("time", "y", "x") and ds.sizes["time"] >= 1
    assert band.dtype == np.uint8
    assert ds.rio.crs is not None and ds.rio.crs.is_projected
    spacing = float(np.abs(np.diff(ds["x"].values[:2])[0]))
    assert 370 < spacing < 380  # 375 m VIIRS pixels
    assert ds.attrs["source_id"] == "nsidc"
    binary = esd.processing.binary_snow(band, product="VNP10A1F").isel(time=0).compute()
    assert float(np.nanmean(binary)) >= 0.0
