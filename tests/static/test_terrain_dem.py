"""The DEM products: catalog entries, offline load from a fixture, live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog
from easysnowdata.terrain import dem

RAINIER = (-121.94, 46.72, -121.54, 46.99)


class TestCatalogEntries:
    def test_five_products_share_one_loader(self):
        assert set(dem.PRODUCTS) == {
            "copernicus-dem",
            "nasadem",
            "srtm",
            "3dep",
            "alos-dem",
        }
        for pid in dem.PRODUCTS:
            product = catalog.get(pid)
            assert product.loader == "easysnowdata.terrain.dem.load"
            assert product.resolve_loader() is dem.load
            assert product.theme == "terrain" and product.examples
            # every (product, source) pair has a route the loader knows
            for src in product.sources:
                assert (pid, src.id) in dem._ROUTES
        assert catalog.validate_all() == []

    def test_copernicus_stays_the_default_and_credential_free(self):
        product = catalog.get("copernicus-dem")
        assert dem.PRODUCT is product and dem.PRODUCT_ID == "copernicus-dem"
        assert [s.id for s in product.sources] == [
            "planetary-computer",
            "earth-search",
            "gee",
        ]
        assert product.requires == ()

    def test_only_srtm_needs_an_account(self):
        table = dem.compare()
        assert list(table.index) == list(dem.PRODUCTS)
        assert table["credential_free"].to_dict() == {
            "copernicus-dem": True,
            "nasadem": True,
            "srtm": False,
            "3dep": True,
            "alos-dem": True,
        }
        assert table.loc["3dep", "resolution_m"] == "10 / 30"
        assert table.loc["3dep", "vertical_datum"] == "NAVD88"

    def test_probe_labels_are_stable(self):
        labels = [
            p.label for s in catalog.get("copernicus-dem").sources for p in s.health
        ]
        assert labels == [
            "Copernicus DEM (Planetary Computer)",
            "Copernicus DEM (Earth Search)",
            "Copernicus DEM (Earth Engine)",
        ]

    @pytest.mark.parametrize(
        ("alias", "pid"),
        [("copernicus", "copernicus-dem"), ("alos", "alos-dem"), ("SRTM", "srtm")],
    )
    def test_aliases(self, alias, pid):
        assert dem._product_id(alias) == pid

    def test_unknown_product_raises(self):
        with pytest.raises(ValueError, match="Unknown DEM product"):
            dem.load(RAINIER, product="aster")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="no source"):
            dem.load(RAINIER, source="nope")

    @pytest.mark.parametrize("resolution", [15, 0, "30m"])
    def test_invalid_resolution_raises(self, resolution):
        with pytest.raises(ValueError, match="30 m and 90 m only"):
            dem.load(RAINIER, resolution=resolution)

    def test_resolution_is_checked_per_product(self):
        with pytest.raises(ValueError, match="10 m and 30 m only"):
            dem.load(RAINIER, product="3dep", resolution=90)
        with pytest.raises(ValueError, match="30 m only"):
            dem.load(RAINIER, product="nasadem", resolution=90)

    def test_empty_item_set_raises(self):
        with pytest.raises(ValueError, match="No cop-dem-glo-30 items"):
            dem.load(RAINIER, items=[])

    def test_search_refuses_an_earth_engine_route(self):
        with pytest.raises(ValueError, match="no tile search"):
            dem.search(RAINIER, product="srtm")

    def test_items_are_for_stac_routes(self):
        with pytest.raises(ValueError, match="items= is for the STAC routes"):
            dem.load(RAINIER, product="srtm", items=[])


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
        assert (
            da.attrs["vertical_datum"] == "EGM2008" and da.attrs["resolution_m"] == 30
        )
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

    @pytest.mark.parametrize(
        ("product", "kwargs"),
        [
            ("nasadem", {}),
            ("3dep", {"resolution": 10}),
            ("3dep", {"resolution": 30}),
            ("alos-dem", {}),
        ],
    )
    def test_other_stac_dems(self, product, kwargs):
        da = dem.load(RAINIER, product=product, **kwargs)
        values = da.compute()
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.attrs["product_id"] == product and da.attrs["units"] == "m"
        # Rainier's summit is 4392 m; every DEM must see most of it.
        assert 4200 < float(np.nanmax(values)) < 4500
        assert float(np.isnan(values).mean()) < 0.01

    @pytest.mark.requires_earthengine
    @pytest.mark.parametrize(
        "product", ["srtm", "nasadem", "3dep", "copernicus-dem", "alos-dem"]
    )
    def test_earth_engine_routes(self, product):
        da = dem.load(RAINIER, product=product, source="gee")
        values = da.compute()
        assert da.dtype == "float32" and da.attrs["source"] == "gee"
        assert 4200 < float(np.nanmax(values)) < 4500

    @pytest.mark.requires_earthengine
    def test_earth_engine_route_reprojects_like_the_stac_routes(self):
        da = dem.load(RAINIER, product="srtm", crs="utm", grid_resolution=90)
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610

    def test_search_then_load(self):
        gdf = dem.search(RAINIER)
        da = dem.load(RAINIER, items=gdf.head(1), crs="utm", grid_resolution=30)
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610
