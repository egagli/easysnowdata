"""CHILI: catalog entry, the load path against a synthetic Earth Engine grid, live smoke."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata import auth, catalog, providers
from easysnowdata.terrain import chili

RAINIER = (-121.94, 46.72, -121.54, 46.99)
GRID = {
    "crs": "EPSG:4326",
    "crs_transform": (0.000833, 0.0, -121.94, 0.0, -0.000833, 46.99),
    "shape_2d": (480, 324),
}


@pytest.fixture
def fake_ee(monkeypatch):
    """Stand in for Earth Engine: xee-shaped output on the asset's native grid."""
    calls: dict = {}

    class Image:
        def __init__(self, asset):
            self.asset = asset
            calls["image"] = asset

    class ImageCollection:
        def __init__(self, image):
            self.image = image

    monkeypatch.setattr(
        providers.gee,
        "ee",
        lambda: types.SimpleNamespace(Image=Image, ImageCollection=ImageCollection),
    )
    monkeypatch.setattr(providers.gee, "grid_params", lambda obj, aoi: dict(GRID))

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = (collection, aoi, kwargs)
        ny, nx = 8, 6
        values = np.linspace(0, 255, ny * nx, dtype="float32").reshape(1, ny, nx)
        values[0, 0, 0] = np.nan  # xee returns NaN outside the asset
        return xr.Dataset(
            {"constant": (("time", "y", "x"), values)},
            coords={
                "time": pd.to_datetime(["2015-01-01"]),
                "y": np.linspace(46.99, 46.72, ny),
                "x": np.linspace(-121.94, -121.54, nx),
            },
        )

    monkeypatch.setattr(providers.gee, "open_dataset", open_dataset)
    return calls


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("chili")
        assert product.resolve_loader() is chili.load
        assert product.theme == "terrain" and product.requires == ("earthengine",)
        assert product.credential_free_sources == ()
        assert [p.label for s in product.sources for p in s.health] == [
            "CHILI (GEE/CSP ERGo)"
        ]
        assert catalog.validate_all() == []

    def test_missing_credentials_raise_before_any_request(self, monkeypatch):
        provider = auth.get("earthengine")
        monkeypatch.setattr(provider, "_ensured", None)
        monkeypatch.setattr(provider, "detect", lambda: auth.Detection(False))
        with pytest.raises(auth.CredentialError) as excinfo:
            chili.load(RAINIER)
        assert excinfo.value.provider == "earthengine"


@pytest.mark.recorded
class TestLoad:
    def test_output_contract(self, fake_ee):
        da = chili.load(RAINIER)
        assert da.name == "chili" and da.dims == ("latitude", "longitude")
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert float(da.max()) == pytest.approx(255.0)
        assert da.attrs["product_id"] == "chili" and da.attrs["source"] == "gee"
        assert da.attrs["units"].startswith("1 (0-255")
        assert da.attrs["data_citation"].startswith("Theobald")
        assert fake_ee["image"] == chili.ASSET_ID
        assert all(isinstance(v, (str, int, float)) for v in da.attrs.values())

    def test_index_normalization(self, fake_ee):
        da = chili.load(RAINIER, normalize="index")
        assert float(da.max()) == pytest.approx(1.0)
        assert "0-1 heat-insolation load index" in da.attrs["units"]

    def test_minmax_normalization_matches_the_old_behaviour(self, fake_ee):
        da = chili.load(RAINIER, normalize=True)
        assert float(da.min()) == pytest.approx(0.0)
        assert float(da.max()) == pytest.approx(1.0)
        assert "min-max" in da.attrs["units"]

    def test_invalid_normalize_raises(self, fake_ee):
        with pytest.raises(ValueError, match="normalize must be"):
            chili.load(RAINIER, normalize="zscore")

    def test_chunks_are_forwarded(self, fake_ee):
        chili.load(RAINIER, chunks={"x": 4})
        assert fake_ee["open"][2]["chunks"] == {"x": 4}
        chili.load(RAINIER)
        assert "chunks" not in fake_ee["open"][2]

    def test_chunks_none_loads_eagerly(self, fake_ee):
        da = chili.load(RAINIER, chunks=None)
        assert da.chunks is None


@pytest.mark.live
@pytest.mark.requires_earthengine
class TestLive:
    def test_smoke(self):
        da = chili.load(RAINIER)
        assert isinstance(da, xr.DataArray) and da.name == "chili"
        assert da.dims in (("latitude", "longitude"), ("y", "x"))
        assert np.issubdtype(da.dtype, np.floating) and da.rio.crs is not None
        assert da.odc.crs is not None and da.attrs["product_id"] == "chili"
        assert da.attrs["source"] == "gee" and da.attrs["units"].startswith("1")

    def test_index_values_are_between_0_and_1(self):
        da = chili.load(RAINIER, normalize="index", chunks=None)
        valid = da.values[~np.isnan(da.values)]
        assert valid.min() >= 0.0 and valid.max() <= 1.0
