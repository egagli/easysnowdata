"""Forest cover fraction: catalog entry, both routes offline, live smoke."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata import auth, catalog, providers
from easysnowdata.land import forest_cover

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def local_geotiff(static_fixtures, monkeypatch):
    """Point the Zenodo route at the local fixture GeoTIFF."""
    monkeypatch.setattr(forest_cover, "ZENODO_URL", str(static_fixtures["forest_cog"]))
    return static_fixtures["forest_cog"]


@pytest.fixture
def fake_ee(monkeypatch):
    calls: dict = {}

    class ImageCollection:
        def __init__(self, asset):
            self.asset = asset
            calls["asset"] = asset

        def select(self, bands):
            calls["select"] = bands
            return self

        def filterDate(self, start, end):
            calls["filterDate"] = (start, end)
            return self

        def sort(self, key, ascending=True):
            calls["sort"] = (key, ascending)
            return self

        def limit(self, n):
            calls["limit"] = n
            return self

        def first(self):
            return object()

    monkeypatch.setattr(
        providers.gee,
        "ee",
        lambda: types.SimpleNamespace(ImageCollection=ImageCollection),
    )
    monkeypatch.setattr(
        providers.gee,
        "grid_params",
        lambda obj, aoi: {
            "crs": "EPSG:4326",
            "crs_transform": (0.00099, 0.0, -121.94, 0.0, -0.00099, 46.99),
            "shape_2d": (8, 6),
        },
    )

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = kwargs
        nt = calls.get("n_times", 1)
        values = np.resize(np.arange(0, 101, dtype="uint8"), (nt, 6, 8))
        return xr.Dataset(
            {forest_cover.GEE_BAND: (("time", "y", "x"), values)},
            coords={
                "time": pd.date_range("2019-01-01", periods=nt, freq="YS"),
                "y": np.linspace(46.99, 46.72, 6),
                "x": np.linspace(-121.94, -121.54, 8),
            },
        )

    monkeypatch.setattr(providers.gee, "open_dataset", open_dataset)
    return calls


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("forest-cover-fraction")
        assert product.resolve_loader() is forest_cover.load
        assert [s.id for s in product.sources] == ["zenodo", "gee"]
        assert product.requires == ()
        assert [p.label for s in product.sources for p in s.health] == [
            "Forest cover fraction (Zenodo)",
            "Forest cover fraction (GEE/CGLS-LC100)",
        ]
        assert catalog.validate_all() == []

    def test_missing_earthengine_credentials_raise(self, monkeypatch):
        provider = auth.get("earthengine")
        monkeypatch.setattr(provider, "_ensured", None)
        monkeypatch.setattr(provider, "detect", lambda: auth.Detection(False))
        with pytest.raises(auth.CredentialError):
            forest_cover.load(RAINIER, source="gee")


@pytest.mark.recorded
class TestZenodoRoute:
    def test_output_contract(self, local_geotiff):
        da = forest_cover.load(RAINIER)
        assert isinstance(da, xr.DataArray) and da.name == "tree_cover_fraction"
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.encoded_nodata == 255 and bool(da.isnull().any())
        assert float(da.max()) <= 100 and float(da.min()) >= 0
        assert da.attrs["units"] == "%" and da.attrs["epoch"] == "2019"
        assert da.attrs["product_id"] == "forest-cover-fraction"
        assert da.attrs["source"] == "zenodo"
        assert str(da.time.values)[:4] == "2019"

    def test_unmasked_keeps_uint8(self, local_geotiff):
        da = forest_cover.load(RAINIER, mask=False)
        assert da.dtype == "uint8" and da.rio.nodata == 255

    def test_chunks(self, local_geotiff):
        assert forest_cover.load(RAINIER, chunks=None).chunks is None
        da = forest_cover.load(RAINIER, chunks={"x": 16, "y": 16})
        assert max(max(sizes) for sizes in da.chunks) <= 16


@pytest.mark.recorded
class TestGeeRoute:
    def test_newest_epoch_by_default(self, fake_ee):
        da = forest_cover.load(RAINIER, source="gee")
        assert fake_ee["asset"] == forest_cover.GEE_ASSET
        assert fake_ee["select"] == [forest_cover.GEE_BAND]
        assert fake_ee["limit"] == 1 and "time" not in da.dims
        assert da.name == "tree_cover_fraction" and da.attrs["source"] == "gee"
        assert da.dims == ("latitude", "longitude") and "epoch" not in da.attrs

    def test_time_range_keeps_the_epochs(self, fake_ee):
        fake_ee["n_times"] = 3
        da = forest_cover.load(RAINIER, source="gee", time="2016/2018")
        assert fake_ee["filterDate"] == ("2016-01-01", "2019-01-01")
        assert da.dims == ("time", "latitude", "longitude") and da.sizes["time"] == 3

    def test_chunks_are_forwarded(self, fake_ee):
        forest_cover.load(RAINIER, source="gee", chunks={"x": 4})
        assert fake_ee["open"]["chunks"] == {"x": 4}


@pytest.mark.live
class TestLive:
    def test_smoke(self):
        da = forest_cover.load(RAINIER)
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.attrs["product_id"] == "forest-cover-fraction"
        assert da.attrs["source"] == "zenodo" and da.attrs["units"] == "%"
        valid = da.values[~np.isnan(da.values)]
        assert valid.min() >= 0 and valid.max() <= 100

    @pytest.mark.requires_earthengine
    def test_gee_smoke(self):
        da = forest_cover.load(RAINIER, source="gee")
        assert da.name == "tree_cover_fraction" and da.rio.crs is not None
        assert da.attrs["source"] == "gee"
