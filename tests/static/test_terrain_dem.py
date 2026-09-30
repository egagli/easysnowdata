"""The DEM products: catalog entries, offline load from a fixture, live smoke tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.providers import opentopography
from easysnowdata.terrain import dem

RAINIER = (-121.94, 46.72, -121.54, 46.99)


class TestCatalogEntries:
    def test_six_products_share_one_loader(self):
        assert set(dem.PRODUCTS) == {
            "copernicus-dem",
            "nasadem",
            "srtm",
            "3dep",
            "alos-dem",
            "gedtm30",
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
            "opentopography",
        ]
        assert product.requires == ()

    def test_every_product_has_a_route_without_an_account(self):
        table = dem.compare()
        assert list(table.index) == list(dem.PRODUCTS)
        assert table["credential_free"].all()
        assert table.loc["3dep", "resolution_m"] == "10 / 30"
        assert table.loc["3dep", "vertical_datum"] == "NAVD88"
        assert table.loc["gedtm30", "vertical_datum"] == "EGM2008"
        assert table["ellipsoidal_copy"].to_dict() == {
            "copernicus-dem": False,
            "nasadem": False,
            "srtm": True,
            "3dep": False,
            "alos-dem": True,
            "gedtm30": False,
        }

    def test_srtm_and_gedtm30_default_to_opentopography(self):
        for pid in ("srtm", "gedtm30"):
            product = catalog.get(pid)
            assert product.default_source.id == "opentopography"
            assert product.default_source.provider == "opentopography"
            assert product.requires == ()
        assert [s.id for s in catalog.get("srtm").sources] == ["opentopography", "gee"]

    def test_opentopography_sources_point_at_their_items(self):
        for pid in ("copernicus-dem", "nasadem", "srtm", "alos-dem", "gedtm30"):
            src = catalog.get(pid).source("opentopography")
            route = dem._ROUTES[pid, "opentopography"]
            assert isinstance(route, dem._OpenTopoRoute)
            assert src.location == providers.opentopography.item_url(
                route.collections[30]
            )

    def test_probe_labels_are_stable(self):
        labels = [
            p.label for s in catalog.get("copernicus-dem").sources for p in s.health
        ]
        assert labels == [
            "Copernicus DEM (Planetary Computer)",
            "Copernicus DEM (Earth Search)",
            "Copernicus DEM (Earth Engine)",
            "Copernicus DEM (OpenTopography)",
        ]

    @pytest.mark.parametrize(
        ("alias", "pid"),
        [
            ("copernicus", "copernicus-dem"),
            ("alos", "alos-dem"),
            ("SRTM", "srtm"),
            ("gedtm", "gedtm30"),
        ],
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
            dem.search(RAINIER, product="srtm", source="gee")

    def test_items_are_for_stac_routes(self):
        with pytest.raises(ValueError, match="items= is for the STAC routes"):
            dem.load(RAINIER, product="srtm", source="gee", items=[])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"product": "nasadem", "source": "opentopography"},
            {"product": "srtm", "source": "gee"},
            {"product": "gedtm30"},
        ],
    )
    def test_ellipsoidal_only_where_a_copy_exists(self, kwargs):
        with pytest.raises(ValueError, match="no ellipsoidal copy"):
            dem.load(RAINIER, ellipsoidal=True, **kwargs)
        # an Earth Engine route is refused by search() for having no tiles first
        with pytest.raises(ValueError, match="no ellipsoidal copy|no tile search"):
            dem.search(RAINIER, ellipsoidal=True, **kwargs)

    def test_opentopography_search_takes_no_stac_arguments(self):
        with pytest.raises(ValueError, match="no search API"):
            dem.search(RAINIER, product="srtm", sign=False)


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


@pytest.fixture
def opentopography_catalog(static_fixtures, monkeypatch):
    """OpenTopography's item and collection JSON, served from memory.

    Every item id resolves to one tile, the local DEM fixture, with the bbox
    OpenTopography writes on its assets; a second tile far away checks that
    tiles outside the AOI are left out. The COG header is read for real.
    """
    import rasterio

    cog = static_fixtures["dem_cog"]
    with rasterio.open(cog) as src:
        bbox = list(src.bounds)
    fetched = []

    def item(item_id):
        fetched.append(item_id)
        return {
            "type": "Feature",
            "id": item_id,
            "assets": {
                "tile.tif": {
                    "href": str(cog),
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data"],
                    "bbox": bbox,
                },
                "far.tif": {
                    "href": "https://example.invalid/far.tif",
                    "type": "image/tiff",
                    "roles": ["data"],
                    "bbox": [10.0, 10.0, 11.0, 11.0],
                },
                f"{item_id}.vrt": {
                    "href": "https://example.invalid/mosaic.vrt",
                    "type": "application/xml",
                    "roles": ["metadata"],
                },
            },
        }

    def collection(name):
        return {
            "id": name,
            "extent": {
                "temporal": {
                    "interval": [["2000-02-11T00:00:00Z", "2000-02-21T00:00:00Z"]]
                }
            },
        }

    monkeypatch.setattr(opentopography, "item", item)
    monkeypatch.setattr(opentopography, "collection", collection)
    return fetched


class TestOpenTopographyProvider:
    def test_ids_and_urls(self):
        assert opentopography.collection_id("COP30_hh") == "COP30"
        assert opentopography.collection_id("SRTM_GL1_Ellip_srtm") == "SRTM_GL1_Ellip"
        assert opentopography.item_url("COP30_hh").endswith("/stac/items/COP30_hh.json")
        assert opentopography.collection_url("COP30").endswith(
            "/stac/COP30_collection.json"
        )

    @pytest.mark.recorded
    def test_tiles_are_picked_by_bbox(self, opentopography_catalog):
        assert [t["id"] for t in opentopography.tiles("X_be", RAINIER)] == ["tile"]
        # no AOI: every data asset, never the VRT
        assert [t["id"] for t in opentopography.tiles("X_be")] == ["tile", "far"]

    @pytest.mark.recorded
    def test_search_builds_loadable_items(self, opentopography_catalog):
        items = opentopography.search("SRTM_GL1_srtm", RAINIER)
        assert len(items) == 1
        item = items[0]
        assert item.collection_id == "SRTM_GL1"
        assert item.properties["proj:code"] == "EPSG:4326"
        assert item.properties["proj:shape"] == [48, 48]
        # dates from the collection's extent, not the item's processing time
        assert item.properties["start_datetime"].startswith("2000-02-11")
        assert item.datetime.year == 2000
        assert item.assets["data"].extra_fields["raster:bands"][0]["nodata"] == -32767.0


@pytest.mark.recorded
class TestOpenTopographyRoute:
    """The OpenTopography load path on the local COG, catalog JSON faked."""

    def test_output_contract(self, opentopography_catalog):
        da = dem.load(RAINIER, source="opentopography")
        assert opentopography_catalog == ["COP30_hh"]
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.attrs["source"] == "opentopography"
        assert da.attrs["collection"] == "COP30"
        assert da.attrs["source_url"] == opentopography.collection_url("COP30")
        assert da.attrs["vertical_datum"] == "EGM2008"
        assert da.attrs["long_name"] == "elevation above the EGM2008 geoid"
        # the fixture's fill is Copernicus's -32767: the sea corner became NaN
        assert da.rio.encoded_nodata == -32767.0 and bool(da.isnull().any())

    def test_srtm_defaults_to_opentopography(self, opentopography_catalog):
        da = dem.load(RAINIER, product="srtm")
        assert opentopography_catalog == ["SRTM_GL1_srtm"]
        assert da.attrs["source"] == "opentopography"
        assert da.attrs["collection"] == "SRTM_GL1"
        assert da.attrs["vertical_datum"] == "EGM96"

    @pytest.mark.parametrize(
        ("product", "item_id"),
        [("srtm", "SRTM_GL1_Ellip_srtm"), ("alos-dem", "AW3D30_E_global")],
    )
    def test_ellipsoidal_reads_the_other_item(
        self, opentopography_catalog, product, item_id
    ):
        da = dem.load(
            RAINIER, product=product, source="opentopography", ellipsoidal=True
        )
        assert opentopography_catalog == [item_id]
        assert da.attrs["vertical_datum"] == "WGS84 ellipsoid"
        assert da.attrs["long_name"] == "elevation above the WGS84 ellipsoid"

    def test_copernicus_90_m(self, opentopography_catalog):
        da = dem.load(RAINIER, source="opentopography", resolution=90)
        assert opentopography_catalog == ["COP90_hh"]
        assert da.attrs["collection"] == "COP90" and da.attrs["resolution_m"] == 90

    def test_search_then_load(self, opentopography_catalog):
        gdf = dem.search(RAINIER, product="srtm", ellipsoidal=True)
        assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) == 1
        assert (gdf["collection"] == "SRTM_GL1_Ellip").all()
        da = dem.load(RAINIER, product="srtm", items=gdf, ellipsoidal=True)
        assert da.attrs["vertical_datum"] == "WGS84 ellipsoid"
        # the other copy's tiles are refused rather than mislabelled
        with pytest.raises(ValueError, match="items= are from SRTM_GL1_Ellip"):
            dem.load(RAINIER, product="srtm", items=gdf)

    def test_gedtm30_reprojects_after_a_native_read(self, opentopography_catalog):
        da = dem.load(RAINIER, product="gedtm30", crs="utm", grid_resolution=100)
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610
        assert da.attrs["vertical_datum"] == "EGM2008"
        assert float(da.max()) > 500  # real values, not the fill


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

    @pytest.mark.parametrize(
        ("product", "kwargs"),
        [
            ("copernicus-dem", {}),
            ("copernicus-dem", {"resolution": 90}),
            ("nasadem", {}),
            ("srtm", {}),
            ("alos-dem", {}),
            ("gedtm30", {}),
        ],
    )
    def test_opentopography_routes(self, product, kwargs):
        da = dem.load(RAINIER, product=product, source="opentopography", **kwargs)
        values = da.compute()
        assert da.dims == ("latitude", "longitude") and da.dtype == "float32"
        assert da.attrs["source"] == "opentopography"
        assert 4200 < float(np.nanmax(values)) < 4500
        assert float(np.isnan(values).mean()) < 0.01

    @pytest.mark.parametrize("product", ["srtm", "alos-dem"])
    def test_ellipsoidal_copies_sit_below_the_geoid_heights(self, product):
        # The geoid is about 19 m below the WGS84 ellipsoid at Rainier, so the
        # ellipsoidal heights are that much lower everywhere.
        kwargs = {"product": product, "source": "opentopography", "chunks": None}
        geoid_da = dem.load(RAINIER, **kwargs)
        ellipsoid_da = dem.load(RAINIER, ellipsoidal=True, **kwargs)
        offset = float(np.nanmedian((ellipsoid_da - geoid_da).values))
        assert -22 < offset < -16
        assert ellipsoid_da.attrs["vertical_datum"] == "WGS84 ellipsoid"

    def test_gedtm30_on_a_utm_grid(self):
        da = dem.load(RAINIER, product="gedtm30", crs="utm", grid_resolution=30)
        values = da.compute()
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610
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
        da = dem.load(
            RAINIER, product="srtm", source="gee", crs="utm", grid_resolution=90
        )
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610

    def test_search_then_load(self):
        gdf = dem.search(RAINIER)
        da = dem.load(RAINIER, items=gdf.head(1), crs="utm", grid_resolution=30)
        assert da.dims == ("y", "x") and da.rio.crs.to_epsg() == 32610
