"""Natural Earth hillshade: catalog entry, styles and scales offline, live smoke."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.terrain import hillshade

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def local_archive(static_fixtures, monkeypatch):
    """Serve every Natural Earth zip from the one local fixture."""
    calls: dict = {}

    def fetch(url, fname=None, **kwargs):
        calls["url"] = url
        calls["fname"] = fname
        calls["subdir"] = kwargs.get("subdir")
        return static_fixtures["hillshade_zip"]

    monkeypatch.setattr(providers.raster_http, "fetch", fetch)
    return calls


@pytest.fixture
def fake_open(monkeypatch):
    """Record the path the loader opens and hand back a 2x2 raster."""
    seen: dict = {}

    def fake(url, aoi=None, **kwargs):
        seen["url"] = url
        return xr.DataArray(
            np.ones((2, 2), dtype="uint8"),
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        ).rio.write_crs("EPSG:4326")

    monkeypatch.setattr(providers.raster_http, "open", fake)
    return seen


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("hillshade")
        assert product.resolve_loader() is hillshade.load
        assert product.theme == "terrain"
        assert [s.id for s in product.sources] == ["natural-earth"]
        assert product.requires == ()
        assert [p.label for s in product.sources for p in s.health] == [
            "Natural Earth hillshade (S3)"
        ]
        assert [v.name for v in product.variables] == ["hillshade"]
        assert catalog.validate_all() == []

    def test_urls(self):
        assert hillshade.url() == (
            "https://naturalearth.s3.amazonaws.com/10m_raster/GRAY_HR_SR_OB_DR.zip"
        )
        assert hillshade.url("shaded-relief", "50m").endswith("50m_raster/SR_50M.zip")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"style": "relief"}, "Invalid style"),
            ({"scale": "110m"}, "Invalid scale"),
            ({"style": "gray-earth-ocean-drainages", "scale": "50m"}, "'10m' only"),
        ],
    )
    def test_invalid_choices_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            hillshade.load(RAINIER, **kwargs)


@pytest.mark.recorded
class TestLoad:
    def test_output_contract(self, local_archive):
        hillshade_da = hillshade.load(RAINIER)
        assert isinstance(hillshade_da, xr.DataArray)
        assert hillshade_da.name == "hillshade" and hillshade_da.dtype == "uint8"
        assert hillshade_da.dims == ("latitude", "longitude")
        assert hillshade_da.rio.crs.to_epsg() == 4326
        assert hillshade_da.odc.crs.epsg == 4326
        assert hillshade_da.rio.nodata is None
        assert hillshade_da.attrs["product_id"] == "hillshade"
        assert hillshade_da.attrs["style"] == "gray-earth-ocean-drainages"
        assert hillshade_da.attrs["scale"] == "1:10m"
        assert "TIFFTAG_SOFTWARE" not in hillshade_da.attrs
        assert local_archive["subdir"] == "natural_earth"
        assert local_archive["fname"] == "GRAY_HR_SR_OB_DR.zip"

    def test_values_are_untouched(self, local_archive):
        values = np.unique(hillshade.load(RAINIER, chunks=None).values)
        assert values.min() >= 40 and values.max() < 240

    def test_style_and_scale_pick_the_archive(self, local_archive, fake_open):
        hillshade.load(RAINIER, style="gray-earth", scale="50m")
        assert local_archive["url"].endswith("50m_raster/GRAY_50M_SR.zip")
        assert local_archive["fname"] == "GRAY_50M_SR.zip"
        assert fake_open["url"].endswith("GRAY_HR_SR_OB_DR.zip!GRAY_50M_SR.tif")

    def test_chunks(self, local_archive):
        assert hillshade.load(RAINIER, chunks=None).chunks is None
        hillshade_da = hillshade.load(RAINIER, chunks={"x": 8, "y": 8})
        assert max(max(sizes) for sizes in hillshade_da.chunks) <= 8

    def test_cache_false_reads_over_https(self, fake_open):
        hillshade.load(RAINIER, style="shaded-relief", cache=False)
        assert fake_open["url"] == (
            "zip+https://naturalearth.s3.amazonaws.com/10m_raster/SR_HR.zip!/SR_HR.tif"
        )


@pytest.mark.live
class TestLive:
    def test_smoke(self):
        hillshade_da = hillshade.load(RAINIER, scale="50m", style="shaded-relief")
        assert hillshade_da.dims == ("latitude", "longitude")
        assert hillshade_da.dtype == "uint8"
        assert hillshade_da.rio.crs.to_epsg() == 4326
        assert hillshade_da.attrs["product_id"] == "hillshade"
        assert (
            hillshade_da.sizes["latitude"] > 5 and hillshade_da.sizes["longitude"] > 10
        )
