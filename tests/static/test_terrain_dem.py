"""Copernicus DEM: catalog entry, offline load from a fixture, live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog
from easysnowdata.terrain import dem

RAINIER = (-121.94, 46.72, -121.54, 46.99)


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("copernicus-dem")
        assert product.loader == "easysnowdata.terrain.dem.load"
        assert product.resolve_loader() is dem.load
        assert [s.id for s in product.sources] == [
            "planetary-computer",
            "earth-search",
        ]
        assert product.requires == () and product.theme == "terrain"
        assert catalog.validate_all() == []

    def test_probe_labels_are_stable(self):
        labels = [
            p.label for s in catalog.get("copernicus-dem").sources for p in s.health
        ]
        assert labels == [
            "Copernicus DEM (Planetary Computer)",
            "Copernicus DEM (Earth Search)",
        ]

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="no source"):
            dem.load(RAINIER, source="nope")

    @pytest.mark.parametrize("resolution", [15, 0, "30m"])
    def test_invalid_resolution_raises(self, resolution):
        with pytest.raises(ValueError, match="30 m and 90 m"):
            dem.load(RAINIER, resolution=resolution)

    def test_empty_item_set_raises(self):
        with pytest.raises(ValueError, match="No cop-dem-glo-30 items"):
            dem.load(RAINIER, items=[])


@pytest.mark.recorded
class TestLoadFromFixture:
    """The full load path on a local COG — no network, no cassette."""

    def test_output_contract(self, dem_items):
        da = dem.load(RAINIER, items=dem_items)
        assert isinstance(da, xr.DataArray) and da.name == "elevation"
        assert da.dims == ("latitude", "longitude")
        assert da.dtype == "float32" and da.chunks is not None
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.encoded_nodata == -32767.0 and np.isnan(da.rio.nodata)
        assert bool(da.isnull().any())  # the sea corner became NaN
        assert da.attrs["product_id"] == "copernicus-dem"
        assert da.attrs["source"] == "planetary-computer"
        assert da.attrs["units"] == "m" and da.attrs["collection"] == "cop-dem-glo-30"
        assert da.attrs["data_citation"].startswith("European Space Agency")
        assert da.attrs["license"] and da.attrs["easysnowdata_version"]
        assert da.attrs["source_url"].endswith("/collections/cop-dem-glo-30")
        assert "time" in da.coords and "time" not in da.dims
        # rioxarray writes _FillValue as a numpy scalar; nothing is a Python object.
        assert all(
            isinstance(v, (str, int, float, list, np.generic))
            for v in da.attrs.values()
        )

    def test_unmasked_keeps_the_sentinel(self, dem_items):
        da = dem.load(RAINIER, items=dem_items, mask=False)
        assert da.rio.nodata == -32767.0 and da.rio.encoded_nodata is None
        assert float(da.min()) == -32767.0

    def test_chunks_none_loads_eagerly(self, dem_items):
        da = dem.load(RAINIER, items=dem_items, chunks=None)
        assert da.chunks is None

    def test_chunks_are_forwarded(self, dem_items):
        da = dem.load(
            RAINIER, items=dem_items, chunks={"latitude": 16, "longitude": 16}
        )
        assert max(max(sizes) for sizes in da.chunks) <= 16

    def test_reprojects_to_utm_with_y_x_dims(self, dem_items):
        da = dem.load(RAINIER, items=dem_items, crs="utm", grid_resolution=100)
        assert da.dims == ("y", "x")
        assert da.rio.crs.to_epsg() == 32610 and da.odc.crs.epsg == 32610


@pytest.mark.recorded
@pytest.mark.vcr
class TestSearchRecorded:
    """Searches are recorded unsigned, so no SAS tokens land in the cassettes."""

    def test_search_returns_tiles(self):
        gdf = dem.search(RAINIER, resolution=30, sign=False)
        assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) >= 1
        assert (gdf["collection"] == "cop-dem-glo-30").all()
        assert gdf.crs.to_epsg() == 4326 and "stac_item" in gdf.columns

    def test_search_on_earth_search(self):
        gdf = dem.search(RAINIER, source="earth-search", resolution=90, sign=False)
        assert len(gdf) >= 1 and (gdf["collection"] == "cop-dem-glo-90").all()


@pytest.mark.live
class TestLive:
    @pytest.mark.parametrize("source", ["planetary-computer", "earth-search"])
    def test_smoke(self, source):
        da = dem.load(RAINIER, source=source, resolution=90)
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.attrs["product_id"] == "copernicus-dem"
        assert da.attrs["source"] == source and da.attrs["units"] == "m"
        assert float(da.max()) > 1000  # Mount Rainier

    def test_search_then_load(self):
        gdf = dem.search(RAINIER)
        da = dem.load(RAINIER, items=gdf.head(1), crs="utm", grid_resolution=30)
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610
