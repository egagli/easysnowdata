"""easysnowdata.snow.modis — the NSIDC route on synthetic HDF-EOS stand-ins
(two adjacent tiles, so the mosaic path runs), the Planetary Computer route on
stubs, and the live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.processing import snow as snow_processing
from easysnowdata.snow import modis

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("modis-snow")
    assert product is modis.PRODUCT
    assert product.loader == "easysnowdata.snow.modis.load"
    assert product.resolve_loader() is modis.load
    # NSIDC is now the default; Planetary Computer stays as the mirror
    assert [s.id for s in product.sources] == ["nsidc", "planetary-computer"]
    assert product.default_source.requires == ("earthdata",)
    assert product.source("planetary-computer").requires == ()
    # the labels the weekly health check has been recording are unchanged
    assert [
        p.label for s in product.sources for p in s.health if p.kind == "health"
    ] == [
        "MODIS snow cover MOD10A1F (NASA NSIDC)",
        "MODIS snow cover MOD10A1 (Planetary Computer)",
    ]
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_product_table_and_granule_names():
    assert modis._product("mod10a1f") == "MOD10A1F"
    with pytest.raises(ValueError, match="product must be one of"):
        modis._product("MOD09GA")
    # Every product, cloud-gap-filled included, stores its fields in the one
    # MOD_Grid_Snow_500m grid — verified against a v61 granule 2026-09-17.
    # This assertion used to pin MOD_CGF_NDSI_500m, which is why the wrong
    # name survived: GDAL answers a grid that does not exist with an empty
    # 1×0 dataset rather than an error.
    assert set(modis.GRIDS.values()) == {"MOD_Grid_Snow_500m"}
    assert modis.DEFAULT_VARIABLES["MOD10A2"] == ("Maximum_Snow_Extent",)
    assert modis.granule_date(
        "MOD10A1.A2023060.h09v04.061.2023062.hdf"
    ) == pd.Timestamp("2023-03-01")
    assert modis.granule_date("MOD10A1.A2020001.h09v04.hdf") == pd.Timestamp(
        "2020-01-01"
    )
    assert modis.granule_date("no-date-here.hdf") is None
    assert modis.subdataset("/x.hdf", "MOD10A2", "Maximum_Snow_Extent") == (
        'HDF4_EOS:EOS_GRID:"/x.hdf":MOD_Grid_Snow_500m:Maximum_Snow_Extent'
    )


# ── the NSIDC route, on synthetic tiles ──────────────────────────────────────


@pytest.fixture
def fake_nsidc(monkeypatch, ts_fixtures, fake_credentials):
    """Stand in for earthaccess: 'granules' are the synthetic tile GeoTIFFs."""
    calls: dict[str, object] = {}
    tiles = sorted(ts_fixtures["modis_dir"].glob("MOD10A1.*.tif"))

    def search(short_name, aoi=None, time=None, **kwargs):
        calls["search"] = (short_name, kwargs)
        return [{"id": p.name, "path": p} for p in tiles]

    def download(granules, subdir, **kwargs):
        calls["download"] = subdir
        return [g["path"] for g in granules]

    monkeypatch.setattr(modis.providers.earthdata, "search", search)
    monkeypatch.setattr(modis.providers.earthdata, "download", download)
    monkeypatch.setattr(modis.providers.earthdata, "require_hdf4", lambda name: None)
    # the fixtures are GeoTIFFs, so the subdataset path is just the file
    monkeypatch.setattr(modis, "subdataset", lambda path, product, variable: str(path))
    return calls


@pytest.mark.recorded
def test_load_nsidc_mosaics_tiles_and_keeps_the_flags(fake_nsidc, ts_fixtures):
    ds = modis.load(RAINIER, "2023-03-01/2023-03-02")
    assert fake_nsidc["search"][0] == "MOD10A1"
    assert fake_nsidc["search"][1]["version"] == "61"
    assert fake_nsidc["download"] == "MOD10A1"

    band = ds["NDSI_Snow_Cover"]
    assert band.dims == ("time", "y", "x")
    assert ds.sizes["time"] == 1  # both tiles share one acquisition date
    assert str(ds["time"].values[0])[:10] == "2023-03-01"
    assert band.dtype == np.uint8  # categorical: the raw byte is kept
    assert band.rio.nodata == 255
    assert band.attrs["flag_meanings"].split()[-1] == "fill"
    assert 200 in band.attrs["flag_values"]
    # the mosaic spans both tiles, so it is wider than either one of them
    tile = esd.providers.raster_http.open(
        sorted(ts_fixtures["modis_dir"].glob("MOD10A1.*.tif"))[0], None, chunks=None
    )
    assert band.sizes["x"] > tile.sizes["x"]
    assert ds.rio.crs is not None and ds.odc.crs is not None
    assert "sinu" in ds.rio.crs.to_proj4() or ds.rio.crs.is_projected
    assert ds.attrs["product_id"] == "modis-snow"
    assert ds.attrs["modis_product"] == "MOD10A1" and ds.attrs["version"] == "61"
    assert ds.attrs["source_id"] == "nsidc"
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_load_nsidc_masking_and_errors(fake_nsidc, monkeypatch):
    masked = modis.load(RAINIER, "2023-03-01", mask=True)
    band = masked["NDSI_Snow_Cover"]
    assert band.dtype == np.float32 and np.isnan(band.values).any()

    with pytest.raises(TypeError, match="not the GeoDataFrame"):
        modis.load(RAINIER, "2023-03-01", granules=gpd.GeoDataFrame())
    monkeypatch.setattr(modis.providers.earthdata, "search", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No MOD10A1 granules"):
        modis.load(RAINIER, "1999-01")


@pytest.mark.recorded
def test_search_nsidc_returns_a_granule_frame(fake_nsidc, monkeypatch):
    def granules_to_geodataframe(granules):
        return gpd.GeoDataFrame(
            {"id": [g["id"] for g in granules], "size_mb": [1.0] * len(granules)},
            geometry=[None] * len(granules),
            crs="EPSG:4326",
        )

    monkeypatch.setattr(
        modis.providers.earthdata,
        "granules_to_geodataframe",
        granules_to_geodataframe,
    )
    gdf = modis.search(RAINIER, "2023-03", product="MOD10A1")
    assert len(gdf) == 2 and "date" in gdf.columns
    assert gdf["date"].iloc[0] == pd.Timestamp("2023-03-01")
    assert gdf.attrs == {"source": "nsidc", "product": "MOD10A1"}


# ── the Planetary Computer route, on stubs ───────────────────────────────────


@pytest.fixture
def fake_stac(monkeypatch, fake_credentials):
    calls: dict[str, object] = {}

    def search(catalog_id, collection, aoi=None, time=None, **kwargs):
        calls["search"] = (catalog_id, collection)
        return [{"id": "granule", "collection": collection, "properties": {}}]

    def items_to_geodataframe(items):
        items = list(items)
        return gpd.GeoDataFrame(
            {"id": [i["id"] for i in items], "stac_item": items},
            geometry=[None] * len(items),
            crs="EPSG:4326",
        )

    def load(item_arg, aoi=None, **kwargs):
        calls["load"] = kwargs
        data = np.full((1, 2, 2), 200, dtype="uint8")
        return xr.Dataset(
            {"Maximum_Snow_Extent": (("time", "y", "x"), data)},
            coords={
                "time": pd.to_datetime(["2023-01-05"]).values,
                "y": [5180400.0, 5179900.0],
                "x": [580000.0, 580500.0],
            },
        ).rio.write_crs("EPSG:32610")

    monkeypatch.setattr(modis.providers.stac, "search", search)
    monkeypatch.setattr(
        modis.providers.stac, "items_to_geodataframe", items_to_geodataframe
    )
    monkeypatch.setattr(modis.providers.stac, "load", load)
    return calls


@pytest.mark.recorded
def test_load_planetary_computer_mirror(fake_stac):
    ds = modis.load(RAINIER, "2023-01", product="MOD10A2", source="planetary-computer")
    assert fake_stac["search"] == ("planetary-computer", "modis-10A2-061")
    assert fake_stac["load"]["bands"] == ["Maximum_Snow_Extent"]
    band = ds["Maximum_Snow_Extent"]
    assert band.dtype == np.uint8 and band.rio.nodata == 255
    assert band.attrs["flag_meanings"].split()[-3:] == [
        "snow",
        "detector_saturated",
        "fill",
    ]
    assert ds.attrs["source_id"] == "planetary-computer"


@pytest.mark.recorded
def test_planetary_computer_refuses_the_products_it_does_not_mirror(fake_stac):
    with pytest.raises(ValueError, match="mirrors only"):
        modis.load(RAINIER, "2023-01", product="MOD10A1F", source="planetary-computer")
    with pytest.raises(ValueError, match="mirrors only"):
        modis.search(
            RAINIER, "2023-01", product="MOD10A1F", source="planetary-computer"
        )


@pytest.mark.recorded
def test_nsidc_route_without_credentials_names_the_free_alternative(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        modis.load(RAINIER, "2023-03")
    assert excinfo.value.provider == "earthdata"
    assert 'source="planetary-computer"' in str(excinfo.value)


# ── binary snow (pure processing) ────────────────────────────────────────────


def test_binary_snow_thresholds_and_flags():
    values = xr.DataArray(
        np.array([0, 20, 40, 100, 200, 250, 255], dtype="uint8"), dims="x"
    )
    binary = snow_processing.binary_snow(values, product="MOD10A1")
    assert list(binary.values[:4]) == [0.0, 0.0, 1.0, 1.0]
    assert np.isnan(binary.values[4:]).all()  # sentinels say nothing
    assert binary.attrs["flag_meanings"] == "no_snow snow"
    assert binary.attrs["ndsi_threshold"] == 40

    lower = snow_processing.binary_snow(values, product="VNP10A1", threshold=20)
    assert list(lower.values[:3]) == [0.0, 1.0, 1.0]
    kept = snow_processing.binary_snow(
        values, product="MOD10A1", keep_flags_as_nan=False
    )
    assert list(kept.values[4:]) == [0.0, 0.0, 0.0]

    extent = xr.DataArray(np.array([25, 200, 50, 100], dtype="uint8"), dims="x")
    a2 = snow_processing.binary_snow(extent, product="MOD10A2")
    assert list(a2.values[:2]) == [0.0, 1.0]
    assert np.isnan(a2.values[2])  # cloud
    assert a2.values[3] == 0.0  # lake ice is not snow
    assert a2.attrs["ndsi_threshold"] == "n/a"


def test_ndsi_flag_attrs_are_plain_strings():
    attrs = snow_processing.ndsi_flag_attrs()
    assert attrs["units"] == "%" and attrs["valid_range"] == [0, 100]
    assert "cloud" in attrs["flag_meanings"]
    assert all(isinstance(v, (str, list)) for v in attrs.values())


# ── the deprecation shim ──────────────────────────────────────────────────────


@pytest.mark.recorded
def test_old_modis_snow_class_is_a_shim(fake_stac):
    from easysnowdata import _deprecation
    from easysnowdata.remote_sensing import MODIS_snow

    _deprecation.reset_warnings()
    with pytest.warns(
        _deprecation.EasysnowdataDeprecationWarning, match="snow.modis.load"
    ):
        ds = MODIS_snow(
            RAINIER,
            start_date="2023-01-01",
            end_date="2023-01-20",
            data_product="MOD10A2",
            mute=True,
        )
    assert isinstance(ds, xr.Dataset)
    assert "Maximum_Snow_Extent" in ds.data_vars
    # MOD10A2 still comes from Planetary Computer, as it did before
    assert ds.attrs["source_id"] == "planetary-computer"


@pytest.mark.recorded
def test_old_modis_snow_cgf_uses_nsidc(fake_nsidc):
    from easysnowdata.remote_sensing import MODIS_snow

    ds = MODIS_snow(
        RAINIER,
        start_date="2023-03-01",
        end_date="2023-03-02",
        data_product="MOD10A1F",
        mute=True,
    )
    assert ds.attrs["source_id"] == "nsidc"
    assert ds.attrs["modis_product"] == "MOD10A1F"


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_modis_search_planetary_computer():
    gdf = modis.search(
        RAINIER, "2023-01-01/2023-01-31", product="MOD10A1", source="planetary-computer"
    )
    assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326


@pytest.mark.live
def test_live_modis_load_planetary_computer():
    ds = modis.load(
        RAINIER, "2023-01-01/2023-01-10", product="MOD10A1", source="planetary-computer"
    )
    band = ds["NDSI_Snow_Cover"]
    assert band.dims == ("time", "y", "x") and ds.sizes["time"] >= 1
    assert band.dtype == np.uint8 and band.rio.nodata == 255
    assert ds.rio.crs is not None and ds.odc.crs is not None
    binary = esd.processing.binary_snow(band, product="MOD10A1").isel(time=0).compute()
    assert set(np.unique(binary.values[~np.isnan(binary.values)])) <= {0.0, 1.0}


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_modis_load_nsidc_cloud_gap_filled():
    import rasterio

    with rasterio.Env() as env:
        if "HDF4" not in env.drivers():
            pytest.skip("this GDAL build has no HDF4 driver (conda-forge libgdal-hdf4)")
    ds = modis.load(RAINIER, "2023-03-01/2023-03-02", product="MOD10A1F")
    band = ds["CGF_NDSI_Snow_Cover"]
    assert band.dims == ("time", "y", "x") and ds.sizes["time"] >= 1
    assert band.dtype == np.uint8 and ds.rio.crs is not None
    assert ds.attrs["source_id"] == "nsidc"


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_modis_a2_is_still_at_nsidc():
    """MOD10A2 left Planetary Computer in 2025; the granules are still at NSIDC."""
    gdf = modis.search(RAINIER, "2023-03-01/2023-03-10", product="MOD10A2")
    assert len(gdf) >= 1
    assert gdf["id"].iloc[0].startswith("MOD10A2")
