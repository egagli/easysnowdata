"""Wrzesien mountain snow mask: catalog entry, the three layers offline, live smoke."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.snow import mountain_snow_mask as msm

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def local_archives(static_fixtures, monkeypatch):
    """Serve the Zenodo zips from the local fixtures."""
    archives = {
        "MODIS_mtnsnow_classes.zip": static_fixtures["mountain_snow_zip"],
        "MODIS_snow_classes.zip": static_fixtures["snow_zip"],
        "MODIS_clouds.zip": static_fixtures["clouds_zip"],
    }
    calls: dict = {}

    def fetch(url, fname=None, **kwargs):
        calls["url"] = url
        calls["subdir"] = kwargs.get("subdir")
        return archives[fname]

    monkeypatch.setattr(providers.raster_http, "fetch", fetch)
    return calls


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("mountain-snow-mask")
        assert product.resolve_loader() is msm.load
        assert [s.id for s in product.sources] == ["zenodo"]
        assert product.requires == ()
        assert [p.label for s in product.sources for p in s.health] == [
            "Mountain snow mask (Zenodo)"
        ]
        assert [v.name for v in product.variables] == [
            "mountain_snow",
            "snow",
            "clouds",
        ]
        assert catalog.validate_all() == []

    def test_invalid_layer_raises(self):
        with pytest.raises(ValueError, match="Invalid layer"):
            msm.load(RAINIER, layer="invalid")

    def test_repair_fill_values(self):
        da = xr.DataArray(np.array([[0, 1, 3, 256, 265]], dtype="uint32"))
        fixed = msm.repair_fill_values(da)
        assert fixed.dtype == "uint8"
        assert list(fixed.values[0]) == [0, 1, 3, 255, 255]


@pytest.mark.recorded
class TestLayers:
    def test_mountain_snow_output_contract(self, local_archives):
        da = msm.load(RAINIER)
        assert isinstance(da, xr.DataArray) and da.name == "mountain_snow"
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 255 and da.rio.encoded_nodata is None
        assert da.attrs["product_id"] == "mountain-snow-mask"
        assert (
            da.attrs["period"] == "2000-2016" and da.attrs["layer"] == "mountain_snow"
        )
        assert da.attrs["flag_values"] == [0, 1, 2, 3, 255]
        assert (
            da.attrs["flag_meanings"].split()[0] == "Mountains_with_little-to-no_snow"
        )
        assert local_archives["subdir"] == "mountain_snow_mask"
        assert local_archives["url"].startswith(msm.ZENODO_FILES)

    def test_upstream_fill_is_repaired(self, local_archives):
        da = msm.load(RAINIER)
        assert set(np.unique(da.values)) <= {0, 1, 2, 3, 255}
        assert 255 in np.unique(da.values)  # the 256 corner came back as fill

    def test_snow_layer(self, local_archives):
        da = msm.load(RAINIER, layer="snow")
        assert da.name == "snow" and da.attrs["flag_values"] == [0, 1, 2, 3, 255]
        assert da.attrs["flag_meanings"].split()[0] == "Little-to-no_snow"
        assert local_archives["url"].endswith("MODIS_snow_classes.zip")

    def test_clouds_layer_is_not_categorical(self, local_archives):
        da = msm.load(RAINIER, layer="clouds")
        assert da.name == "clouds" and da.dtype == "uint8"
        assert "flag_values" not in da.attrs
        assert (
            da.attrs["units"] == "1" and "undefined upstream" in da.attrs["long_name"]
        )
        assert set(np.unique(da.values)) <= set(range(7)) | {255}
        assert local_archives["url"].endswith("MODIS_clouds.zip")

    def test_masked_form(self, local_archives):
        da = msm.load(RAINIER, mask=True)
        assert da.dtype == "float32" and da.rio.encoded_nodata == 255
        assert bool(da.isnull().any())

    def test_chunks(self, local_archives):
        assert msm.load(RAINIER, chunks=None).chunks is None
        da = msm.load(RAINIER, chunks={"x": 8, "y": 8})
        assert max(max(sizes) for sizes in da.chunks) <= 8

    def test_cache_false_reads_over_https(self, monkeypatch):
        seen = {}

        def fake_open(url, aoi=None, **kwargs):
            seen["url"] = url
            return xr.DataArray(
                np.ones((2, 2), dtype="uint8"),
                dims=("y", "x"),
                coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
            ).rio.write_crs("EPSG:4326")

        monkeypatch.setattr(providers.raster_http, "open", fake_open)
        msm.load(RAINIER, cache=False)
        assert seen["url"] == (
            f"zip+{msm.ZENODO_FILES}/MODIS_mtnsnow_classes.zip!/MODIS_mtnsnow_classes.tif"
        )

    def test_old_name_warns_and_forwards_data_product(self, local_archives):
        from easysnowdata import _deprecation
        from easysnowdata.remote_sensing import get_seasonal_mountain_snow_mask

        _deprecation.reset_warnings()
        with pytest.warns(
            _deprecation.EasysnowdataDeprecationWarning, match="mountain_snow_mask.load"
        ):
            da = get_seasonal_mountain_snow_mask(RAINIER, data_product="snow")
        assert da.name == "snow" and da.rio.nodata == 255
        with pytest.raises(ValueError):
            get_seasonal_mountain_snow_mask(RAINIER, data_product="invalid")


@pytest.mark.live
class TestLive:
    def test_smoke(self):
        da = msm.load(RAINIER)
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 255
        assert da.attrs["product_id"] == "mountain-snow-mask"
        assert (
            da.attrs["flag_meanings"].split()[0] == "Mountains_with_little-to-no_snow"
        )
        assert set(np.unique(da.values)) <= {0, 1, 2, 3, 255}

    def test_clouds_layer(self):
        da = msm.load(RAINIER, layer="clouds")
        assert da.name == "clouds" and set(np.unique(da.values)) <= set(range(7)) | {
            255
        }
