"""NLCD: catalog entry, both Earth Engine assets against a synthetic grid, live smoke."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata import auth, catalog, providers
from easysnowdata.land import nlcd

RAINIER = (-121.94, 46.72, -121.54, 46.99)
GRID = {
    "crs": "EPSG:5070",  # NLCD's Albers equal-area grid
    "crs_transform": (30.0, 0.0, -2000000.0, 0.0, -30.0, 3000000.0),
    "shape_2d": (6, 8),
}
CLASS_TABLE = {
    "landcover_class_values": [11, 42, 90],
    "landcover_class_names": [
        "Open Water: areas of water",
        "Evergreen Forest",
        "Woody Wetlands",
    ],
    "landcover_class_palette": ["466b9f", "1c5f2c", "b8d9eb"],
}


@pytest.fixture
def fake_ee(monkeypatch):
    calls: dict = {}

    class Image:
        def __init__(self, properties=None):
            self.properties = (
                properties if properties is not None else dict(CLASS_TABLE)
            )
            self.selected = None

        def select(self, bands):
            self.selected = bands
            calls["selected"] = bands
            return self

        def getInfo(self):
            calls["getInfo"] = True
            return {"properties": self.properties}

    class ImageCollection:
        def __init__(self, asset, properties=None):
            self.asset = asset
            self.properties = properties
            calls.setdefault("assets", []).append(asset)

        def sort(self, key, ascending=True):
            calls["sort"] = (key, ascending)
            return self

        def limit(self, n):
            calls["limit"] = n
            return self

        def filterDate(self, start, end):
            calls["filterDate"] = (start, end)
            return self

        def select(self, bands):
            calls["collection_select"] = bands
            return self

        def first(self):
            return Image(self.properties)

    monkeypatch.setattr(
        providers.gee,
        "ee",
        lambda: types.SimpleNamespace(Image=Image, ImageCollection=ImageCollection),
    )
    monkeypatch.setattr(providers.gee, "grid_params", lambda obj, aoi: dict(GRID))

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = (collection, aoi, kwargs)
        ny, nx, nt = 6, 8, calls.get("n_times", 1)
        values = np.resize(np.array([11, 42, 90], dtype="uint8"), (nt, ny, nx))
        name = calls.get("band", "landcover")
        return xr.Dataset(
            {name: (("time", "y", "x"), values)},
            coords={
                "time": pd.date_range("2023-01-01", periods=nt, freq="YS"),
                "y": np.arange(3000000.0, 3000000.0 - 30 * ny, -30.0),
                "x": np.arange(-2000000.0, -2000000.0 + 30 * nx, 30.0),
            },
        )

    monkeypatch.setattr(providers.gee, "open_dataset", open_dataset)
    return calls


class TestCatalogEntry:
    def test_registered_with_annual_as_the_default(self):
        product = catalog.get("nlcd")
        assert product.resolve_loader() is nlcd.load
        assert [s.id for s in product.sources] == ["gee-annual", "gee"]
        assert product.requires == ("earthengine",)
        assert [p.label for s in product.sources for p in s.health] == [
            "Annual NLCD (GEE community asset)",
            "NLCD (GEE/USGS)",
        ]
        assert catalog.validate_all() == []

    def test_unknown_layers_raise_per_source(self):
        with pytest.raises(ValueError, match="Unknown Annual NLCD layer"):
            nlcd.load(RAINIER, layer="science_products_land_cover_change_index")
        with pytest.raises(ValueError, match="Unknown NLCD layer"):
            nlcd.load(RAINIER, source="gee", layer="landcover_confidence")

    def test_missing_credentials_raise_before_any_request(self, monkeypatch):
        provider = auth.get("earthengine")
        monkeypatch.setattr(provider, "_ensured", None)
        monkeypatch.setattr(provider, "detect", lambda: auth.Detection(False))
        with pytest.raises(auth.CredentialError) as excinfo:
            nlcd.load(RAINIER)
        assert excinfo.value.provider == "earthengine"


@pytest.mark.recorded
class TestLoad:
    def test_annual_default_is_the_newest_year(self, fake_ee):
        da = nlcd.load(RAINIER)
        assert fake_ee["assets"] == [nlcd.ANNUAL_ASSETS["landcover"]]
        assert fake_ee["sort"] == ("system:time_start", False) and fake_ee["limit"] == 1
        assert da.name == "landcover" and da.dims == ("y", "x")
        assert da.rio.crs.to_epsg() == 5070 and da.odc.crs.epsg == 5070
        assert "time" in da.coords and "time" not in da.dims
        assert da.attrs["asset"].endswith("ANNUAL_NLCD/LANDCOVER")
        assert da.attrs["source"] == "gee-annual" and da.attrs["extent"] == "CONUS"
        assert da.attrs["product_id"] == "nlcd"

    def test_class_table_comes_from_the_asset(self, fake_ee):
        da = nlcd.load(RAINIER)
        assert da.attrs["flag_values"] == [11, 42, 90]
        assert da.attrs["flag_meanings"] == "Open_Water Evergreen_Forest Woody_Wetlands"
        assert da.attrs["flag_colors"] == "#466b9f #1c5f2c #b8d9eb"
        assert da.attrs["long_name"] == "NLCD landcover"
        assert da.rio.nodata == 0  # 0 is not a class, so it is the sentinel

    def test_a_missing_class_table_is_not_fatal(self, fake_ee, monkeypatch, caplog):
        fake_ee["band"] = "b1"
        monkeypatch.setitem(fake_ee, "n_times", 1)
        ee = providers.gee.ee()
        original = ee.ImageCollection
        monkeypatch.setattr(
            providers.gee,
            "ee",
            lambda: types.SimpleNamespace(
                Image=ee.Image,
                ImageCollection=lambda asset: original(asset, properties={}),
            ),
        )
        da = nlcd.load(RAINIER)
        assert "flag_values" not in da.attrs and da.name == "landcover"

    def test_time_range_keeps_the_time_dim(self, fake_ee):
        fake_ee["n_times"] = 3
        da = nlcd.load(RAINIER, time="2021/2023")
        assert fake_ee["filterDate"] == ("2021-01-01", "2024-01-01")
        assert da.dims == ("time", "y", "x") and da.sizes["time"] == 3

    def test_release_source_selects_the_band(self, fake_ee):
        da = nlcd.load(RAINIER, source="gee", layer="landcover")
        assert fake_ee["assets"] == [nlcd.RELEASE_ASSET]
        assert fake_ee["collection_select"] == ["landcover"]
        assert da.attrs["source"] == "gee" and da.attrs["layer"] == "landcover"

    def test_unknown_band_raises(self, fake_ee, monkeypatch):
        monkeypatch.setattr(
            providers.gee,
            "open_dataset",
            lambda collection, aoi=None, **kw: xr.Dataset(
                {"a": ("x", [1]), "b": ("x", [2])}, coords={"x": [0]}
            ),
        )
        with pytest.raises(ValueError, match="no 'landcover' band"):
            nlcd.load(RAINIER)

    def test_chunks(self, fake_ee):
        nlcd.load(RAINIER, chunks={"x": 4})
        assert fake_ee["open"][2]["chunks"] == {"x": 4}
        assert nlcd.load(RAINIER, chunks=None).chunks is None

    def test_masked_form(self, fake_ee):
        da = nlcd.load(RAINIER, mask=True)
        assert da.dtype == "float32" and da.rio.encoded_nodata == 0

    def test_old_name_warns_and_uses_the_official_release(self, fake_ee):
        from easysnowdata import _deprecation
        from easysnowdata.remote_sensing import get_nlcd_landcover

        _deprecation.reset_warnings()
        with pytest.warns(
            _deprecation.EasysnowdataDeprecationWarning, match="nlcd.load"
        ):
            da = get_nlcd_landcover(RAINIER, layer="landcover", initialize_ee=False)
        assert fake_ee["assets"] == [nlcd.RELEASE_ASSET] and da.attrs["source"] == "gee"


@pytest.mark.live
@pytest.mark.requires_earthengine
class TestLive:
    def test_smoke(self):
        da = nlcd.load(RAINIER)
        assert isinstance(da, xr.DataArray) and da.dims == ("y", "x")
        assert np.issubdtype(da.dtype, np.integer) or np.issubdtype(
            da.dtype, np.floating
        )
        assert da.rio.crs is not None and da.odc.crs is not None
        assert da.attrs["product_id"] == "nlcd" and da.attrs["source"] == "gee-annual"
        assert "flag_values" in da.attrs

    def test_official_release_layer(self):
        da = nlcd.load(RAINIER, source="gee", layer="landcover")
        assert da.attrs["asset"] == nlcd.RELEASE_ASSET and "flag_values" in da.attrs

    def test_annual_time_series(self):
        da = nlcd.load(RAINIER, time="2020/2022")
        assert "time" in da.dims and da.sizes["time"] >= 2
