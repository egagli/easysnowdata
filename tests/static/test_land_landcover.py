"""ESA WorldCover: catalog entry, both routes offline on fixtures, live smoke."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from easysnowdata import catalog, plotting, providers
from easysnowdata.land import landcover

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def local_tiles(static_fixtures, monkeypatch) -> gpd.GeoDataFrame:
    """The bucket tile grid, pointed at the local fixture tiles."""
    monkeypatch.setattr(
        providers.raster_http,
        "fetch",
        lambda *a, **kw: static_fixtures["worldcover_grid"],
    )
    monkeypatch.setattr(
        landcover,
        "tile_url",
        lambda tile, version: str(
            static_fixtures[
                "worldcover_left" if tile == "N45W123" else "worldcover_right"
            ]
        ),
    )
    return landcover._tiles(RAINIER, "v200")


class TestCatalogEntry:
    def test_registered_from_the_theme_module(self):
        product = catalog.get("esa-worldcover")
        assert product.resolve_loader() is landcover.load
        assert [s.id for s in product.sources] == [
            "planetary-computer",
            "aws-open-data",
        ]
        assert product.requires == () and len(product.credential_free_sources) == 2
        assert [p.label for s in product.sources for p in s.health] == [
            "ESA WorldCover (Planetary Computer)",
            "ESA WorldCover (AWS bucket)",
        ]
        assert catalog.validate_all() == []

    def test_variable_carries_the_class_table(self):
        var = catalog.get("esa-worldcover").variables[0]
        assert var.categorical and len(var.flag_values) == 11
        assert var.flag_values[0] == 10 and var.flag_meanings[0] == "Tree cover"

    @pytest.mark.parametrize("version", ["v999", "2021", ""])
    def test_invalid_version_raises(self, version):
        with pytest.raises(ValueError, match="Incorrect version number"):
            landcover.load(RAINIER, version=version)

    def test_version_is_read_from_the_datetime_when_the_property_is_missing(self):
        assert (
            landcover._version_of({"properties": {"datetime": "2020-06-01"}}) == "v100"
        )
        assert (
            landcover._version_of({"properties": {"start_datetime": "2021-01-01"}})
            == "v200"
        )
        assert landcover._version_of({"properties": {"datetime": "1999-01-01"}}) is None
        assert landcover._version_of({"properties": {}}) is None

    def test_tile_url(self):
        url = landcover.tile_url("N45W123", "v100")
        assert url.endswith(
            "/v100/2020/map/ESA_WorldCover_10m_2020_v100_N45W123_Map.tif"
        )


@pytest.mark.recorded
class TestStacRoute:
    def test_output_contract(self, worldcover_items):
        da = landcover.load(RAINIER, items=worldcover_items)
        assert isinstance(da, xr.DataArray) and da.name == "landcover"
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 0 and da.rio.encoded_nodata is None
        assert da.attrs["product_id"] == "esa-worldcover"
        assert da.attrs["source"] == "planetary-computer"
        assert da.attrs["version"] == "v200" and da.attrs["year"] == "2021"
        assert da.attrs["flag_values"][:2] == [10, 20]
        assert da.attrs["flag_meanings"].split()[0] == "Tree_cover"
        assert da.attrs["flag_colors"].split()[0] == "#006400"
        assert all(
            isinstance(v, (str, int, float, list, np.generic))
            for v in da.attrs.values()
        )
        assert "time" in da.coords and "time" not in da.dims

    def test_version_filter_drops_other_years(self, worldcover_items):
        with pytest.raises(ValueError, match="No ESA WorldCover v100 tiles"):
            landcover.load(RAINIER, items=worldcover_items, version="v100")

    def test_masked_values(self, worldcover_items):
        da = landcover.load(RAINIER, items=worldcover_items, mask=True)
        assert da.dtype == "float32" and da.rio.encoded_nodata == 0
        assert bool(da.isnull().any())

    def test_plotting_reads_the_flags(self, worldcover_items):
        da = landcover.load(RAINIER, items=worldcover_items)
        cmap, norm, table = plotting.colormap_from_flags(da)
        assert list(table["value"])[:2] == [10, 20] and cmap.N == 11


@pytest.mark.recorded
class TestAwsRoute:
    def test_tiles_come_from_the_bucket_grid(self, local_tiles):
        assert list(local_tiles["tile"]) == ["N45W123", "N45W120"]
        assert len(local_tiles) == 2 and "url" in local_tiles

    def test_search_returns_the_tile_grid(self, local_tiles, static_fixtures):
        gdf = landcover.search(RAINIER, source="aws-open-data")
        assert list(gdf["tile"]) == ["N45W123", "N45W120"]
        assert gdf.crs.to_epsg() == 4326

    def test_tiles_are_merged_and_clipped(self, local_tiles):
        da = landcover.load(RAINIER, source="aws-open-data", items=local_tiles)
        assert da.name == "landcover" and da.dtype == "uint8"
        assert da.dims == ("latitude", "longitude")
        assert da.rio.crs.to_epsg() == 4326 and da.rio.nodata == 0
        assert da.sizes["longitude"] > 48  # both tiles, not one
        assert da.attrs["source"] == "aws-open-data"
        assert da.attrs["source_url"].endswith("/v200/2021/map")
        assert str(da.time.values)[:10] == "2021-01-01"
        assert set(np.unique(da.values)) <= set(da.attrs["flag_values"]) | {0}

    def test_single_tile(self, local_tiles):
        da = landcover.load(RAINIER, source="aws-open-data", items=local_tiles.head(1))
        assert da.sizes["longitude"] <= 49

    def test_tile_names_are_accepted(self, static_fixtures, local_tiles):
        da = landcover.load(RAINIER, source="aws-open-data", items=["N45W123"])
        assert da.name == "landcover"

    def test_empty_tile_list_raises(self, local_tiles):
        with pytest.raises(ValueError, match="No ESA WorldCover v200 tiles"):
            landcover.load(RAINIER, source="aws-open-data", items=[])

    def test_chunks_none_loads_eagerly(self, local_tiles):
        da = landcover.load(
            RAINIER, source="aws-open-data", items=local_tiles, chunks=None
        )
        assert da.chunks is None


@pytest.mark.recorded
@pytest.mark.vcr
class TestSearchRecorded:
    def test_search_filters_by_version(self):
        gdf = landcover.search(RAINIER, version="v100")
        assert len(gdf) == 1
        assert gdf["esa_worldcover:product_version"].iloc[0] == "1.0.0"


@pytest.mark.live
class TestLive:
    @pytest.mark.parametrize("source", ["planetary-computer", "aws-open-data"])
    def test_smoke(self, source):
        da = landcover.load(RAINIER, source=source)
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 0
        assert da.attrs["product_id"] == "esa-worldcover"
        assert da.attrs["source"] == source and da.attrs["version"] == "v200"
        assert da.attrs["flag_meanings"].split()[0] == "Tree_cover"

    def test_the_two_routes_agree(self):
        aoi = (-121.80, 46.84, -121.78, 46.86)
        pc = landcover.load(aoi).compute()
        aws = landcover.load(aoi, source="aws-open-data").compute()
        shared = (
            min(pc.sizes["latitude"], aws.sizes["latitude"]),
            min(pc.sizes["longitude"], aws.sizes["longitude"]),
        )
        left = pc.values[: shared[0], : shared[1]]
        right = aws.values[: shared[0], : shared[1]]
        assert (left == right).mean() > 0.99

    def test_v100_is_the_2020_map(self):
        da = landcover.load(RAINIER, version="v100")
        assert da.attrs["year"] == "2020" and str(da.time.values)[:4] == "2020"
