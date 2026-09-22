"""Unit tests for easysnowdata.providers — third-party clients monkeypatched,
loaders exercised against the tiny local fixtures. No network."""

from __future__ import annotations

import datetime as dt
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pytest
import shapely
import xarray as xr

from easysnowdata import auth, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.providers import (
    earthdata,
    gee,
    raster_http,
    stac,
    vector_http,
    zarr_cloud,
)

RAINIER = (-121.94, 46.72, -121.54, 46.99)
DATELINE = (170.0, -20.0, -170.0, -10.0)


def _item(i: int, collection: str = "cop-dem-glo-30") -> pystac.Item:
    geom = shapely.box(-122 + i, 46, -121 + i, 47)
    item = pystac.Item(
        id=f"item-{i}",
        geometry=shapely.geometry.mapping(geom),
        bbox=list(geom.bounds),
        datetime=dt.datetime(2021, 4, 22, tzinfo=dt.UTC),
        properties={"proj:epsg": 4326, "gsd": 30},
        collection=collection,
    )
    item.add_asset(
        "data",
        pystac.Asset(href=f"https://example.com/{i}.tif", media_type="image/tiff"),
    )
    return item


# ── stac ──────────────────────────────────────────────────────────────────────


class TestStac:
    def test_catalog_spec(self):
        assert stac._catalog_spec("planetary-computer")["sign"] is True
        assert (
            stac._catalog_spec(stac.EARTH_SEARCH_URL + "/")["gdal"][
                "AWS_NO_SIGN_REQUEST"
            ]
            == "YES"
        )
        assert stac._catalog_spec("https://other.example/stac") == {
            "url": "https://other.example/stac",
            "sign": False,
            "requires": (),
            "gdal": {},
        }
        assert stac.CATALOGS["cmr-lpcloud"]["requires"] == ("earthdata",)

    @pytest.fixture
    def fake_client(self, monkeypatch):
        import pystac_client

        seen = {}

        class Search:
            def __init__(self, params):
                seen["search"] = params

            def item_collection(self):
                return pystac.ItemCollection([_item(0), _item(1)])

        class Client:
            @staticmethod
            def open(url, modifier=None, **kw):
                seen["open"] = (url, modifier)
                return Client()

            def search(self, **params):
                return Search(params)

        monkeypatch.setattr(pystac_client, "Client", Client)
        return seen

    def test_open_catalog_signing(self, fake_client):
        import planetary_computer

        stac.open_catalog()
        assert fake_client["open"] == (
            stac.PLANETARY_COMPUTER_URL,
            planetary_computer.sign_inplace,
        )
        stac.open_catalog("earth-search")
        assert fake_client["open"][1] is None
        stac.open_catalog("planetary-computer", sign=False)
        assert fake_client["open"][1] is None
        stac.open_catalog("https://x.example/v1")
        assert fake_client["open"][0] == "https://x.example/v1"

    def test_search_builds_params(self, fake_client):
        items = stac.search(
            "earth-search",
            "sentinel-2-l2a",
            RAINIER,
            "2023-10",
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=5,
            limit=100,
        )
        assert isinstance(items, pystac.ItemCollection) and len(items) == 2
        params = fake_client["search"]
        assert params["collections"] == ["sentinel-2-l2a"] and params["max_items"] == 5
        assert params["intersects"]["type"] == "Polygon"
        assert params["datetime"] == "2023-10-01T00:00:00Z/2023-10-31T23:59:59Z"
        assert (
            params["query"] == {"eo:cloud_cover": {"lt": 20}} and params["limit"] == 100
        )
        stac.search("planetary-computer", ["a", "b"], None, None, filter={"op": "="})
        params = fake_client["search"]
        assert (
            "intersects" not in params
            and "datetime" not in params
            and params["filter"] == {"op": "="}
        )
        stac.search("planetary-computer", "a", parse_aoi(None))
        assert "intersects" not in fake_client["search"]
        stac.search("planetary-computer", "a", DATELINE)
        assert fake_client["search"]["intersects"]["type"] == "MultiPolygon"

    def test_items_geodataframe_round_trip(self):
        items = [_item(0), _item(1)]
        gdf = stac.items_to_geodataframe(pystac.ItemCollection(items))
        assert list(gdf["id"]) == ["item-0", "item-1"] and gdf.crs.to_epsg() == 4326
        assert (
            gdf["collection"].iloc[0] == "cop-dem-glo-30" and gdf["gsd"].iloc[0] == 30
        )
        assert pd.api.types.is_datetime64_any_dtype(gdf["datetime"])
        back = stac.geodataframe_to_items(gdf.iloc[1:])
        assert [i.id for i in back] == ["item-1"]
        assert back[0].assets["data"].href == "https://example.com/1.tif"
        assert stac.items_to_geodataframe([]).empty
        with pytest.raises(ValueError, match="stac_item"):
            stac.geodataframe_to_items(gpd.GeoDataFrame(geometry=[shapely.Point(0, 0)]))

    def test_iter_items_forms(self):
        item = _item(0)
        assert stac._iter_items(item) == [item]
        assert stac._iter_items([item]) == [item]
        assert len(stac._iter_items(pystac.ItemCollection([item]))) == 1
        search_like = types.SimpleNamespace(
            item_collection=lambda: pystac.ItemCollection([item])
        )
        assert len(stac._iter_items(search_like)) == 1
        assert stac._iter_items(stac.items_to_geodataframe([item]))[0].id == "item-0"

    def test_read_env(self):
        with stac.read_env("earth-search") as opts:
            assert (
                opts["AWS_NO_SIGN_REQUEST"] == "YES"
                and opts["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
            )
        with stac.read_env("planetary-computer", GDAL_HTTP_MAX_RETRY="9") as opts:
            assert opts["GDAL_HTTP_MAX_RETRY"] == "9"
        with stac.read_env(None) as opts:
            assert "AWS_NO_SIGN_REQUEST" not in opts

    @pytest.fixture
    def fake_odc(self, monkeypatch):
        import odc.stac

        seen = {}

        def load(items, **params):
            seen["items"] = list(items)
            seen["params"] = params
            return xr.Dataset()

        monkeypatch.setattr(odc.stac, "load", load)
        return seen

    def test_load_clip_and_grid(self, fake_odc):
        items = pystac.ItemCollection([_item(0)])
        stac.load(
            items,
            RAINIER,
            bands=["data"],
            resolution=30,
            crs="utm",
            catalog="planetary-computer",
        )
        p = fake_odc["params"]
        assert (
            p["crs"] == "EPSG:32610"
            and p["resolution"] == 30
            and p["bands"] == ["data"]
        )
        assert p["chunks"] == {} and p["groupby"] == "time" and "stac_cfg" not in p
        assert p["geopolygon"].crs == "EPSG:4326" and "intersects" not in p
        assert [i.id for i in fake_odc["items"]] == ["item-0"]

    def test_load_no_clip_returns_covering_tiles(self, fake_odc):
        stac.load(
            [_item(0)],
            parse_aoi(RAINIER, clip=False),
            crs="EPSG:4326",
            chunks={"x": 256},
            groupby=None,
            extra=1,
        )
        p = fake_odc["params"]
        assert p["intersects"]["type"] == "Polygon" and "geopolygon" not in p
        assert (
            p["chunks"] == {"x": 256}
            and p["groupby"] is None
            and p["extra"] == 1
            and p["crs"] == "EPSG:4326"
        )

    def test_load_global_and_geodataframe_input(self, fake_odc):
        gdf = stac.items_to_geodataframe([_item(0), _item(1)])
        stac.load(gdf.iloc[:1], None)
        assert "geopolygon" not in fake_odc["params"] and len(fake_odc["items"]) == 1
        stac.odc_load([_item(0)], catalog="earth-search", bbox=RAINIER)
        assert fake_odc["params"] == {"bbox": RAINIER}


# ── earthdata ─────────────────────────────────────────────────────────────────


class _Granule(dict):
    def data_links(self):
        return [f"https://data.example/{self['meta']['native-id']}.nc"]

    def size(self):
        return 12.5


def _granule(i: int, *, polygon: bool = True) -> _Granule:
    if polygon:
        geometry = {
            "GPolygons": [
                {
                    "Boundary": {
                        "Points": [
                            {"Longitude": -122, "Latitude": 46},
                            {"Longitude": -121, "Latitude": 46},
                            {"Longitude": -121, "Latitude": 47},
                            {"Longitude": -122, "Latitude": 46},
                        ]
                    }
                }
            ]
        }
    else:
        geometry = {
            "BoundingRectangles": [
                {
                    "WestBoundingCoordinate": -122,
                    "SouthBoundingCoordinate": 46,
                    "EastBoundingCoordinate": -121,
                    "NorthBoundingCoordinate": 47,
                }
            ]
        }
    return _Granule(
        meta={"native-id": f"G{i}"},
        umm={
            "GranuleUR": f"GRANULE_{i}",
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2020-01-01T00:00:00Z",
                    "EndingDateTime": "2020-01-02T00:00:00Z",
                }
            },
            "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": geometry}},
        },
    )


class TestEarthdata:
    @pytest.fixture
    def fake_earthaccess(self, monkeypatch, tmp_path):
        import earthaccess

        seen = {}
        monkeypatch.setattr(
            auth.get("earthdata"),
            "ensure",
            lambda **kw: (
                seen.setdefault("ensured", 0)
                or seen.__setitem__("ensured", seen["ensured"] + 1)
            ),
        )
        monkeypatch.setattr(
            earthaccess,
            "search_data",
            lambda **kw: seen.__setitem__("search", kw) or [_granule(0)],
        )
        monkeypatch.setattr(
            earthaccess,
            "open",
            lambda granules, **kw: seen.__setitem__("open", (granules, kw)) or ["file"],
        )
        monkeypatch.setattr(
            earthaccess,
            "download",
            lambda granules, path, **kw: (
                seen.__setitem__("download", (granules, path, kw))
                or [str(path / "a.hdf")]
            ),
        )
        monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path / "cache"))
        return seen

    def test_search_params(self, fake_earthaccess):
        earthdata.search("WUS_UCLA_SR", RAINIER, "2020-01", count=3)
        kw = fake_earthaccess["search"]
        assert (
            kw["short_name"] == "WUS_UCLA_SR"
            and kw["cloud_hosted"] is True
            and kw["count"] == 3
        )
        assert kw["bounding_box"] == pytest.approx(RAINIER) and kw["temporal"] == (
            "2020-01-01",
            "2020-01-31",
        )
        assert fake_earthaccess["ensured"] >= 1
        earthdata.search(
            "X",
            None,
            ("2020-01-01", None),
            cloud_hosted=False,
            bounding_box=(1, 2, 3, 4),
        )
        kw = fake_earthaccess["search"]
        assert (
            "cloud_hosted" not in kw
            and kw["bounding_box"] == (1, 2, 3, 4)
            and kw["temporal"][0] == "2020-01-01"
        )
        earthdata.search("X", DATELINE)
        west, south, east, north = fake_earthaccess["search"]["bounding_box"]
        assert west == 170 and east == 180 and (south, north) == (-20, -10)
        earthdata.search("X", parse_aoi(None), slice(None, "2020"))
        assert "bounding_box" not in fake_earthaccess["search"] and fake_earthaccess[
            "search"
        ]["temporal"] == (None, "2020-12-31")

    def test_granules_to_geodataframe(self):
        gdf = earthdata.granules_to_geodataframe(
            [_granule(0), _granule(1, polygon=False)]
        )
        assert list(gdf["id"]) == ["GRANULE_0", "GRANULE_1"]
        assert gdf.geometry.iloc[0].geom_type == "Polygon" and gdf.geometry.iloc[
            1
        ].bounds == (-122, 46, -121, 47)
        assert gdf["size_mb"].iloc[0] == 12.5 and gdf["data_links"].iloc[0][0].endswith(
            "G0.nc"
        )
        assert gdf["start"].dt.year.iloc[0] == 2020
        assert earthdata.granules_to_geodataframe([]).empty
        bare = earthdata.granules_to_geodataframe([{"meta": {"native-id": "x"}}])
        assert bare.geometry.iloc[0] is None and np.isnan(bare["size_mb"].iloc[0])

    def test_open_and_download(self, fake_earthaccess, tmp_path):
        assert earthdata.open([1], show_progress=False) == ["file"]
        assert fake_earthaccess["open"][1] == {"show_progress": False}
        paths = earthdata.download([1], "MOD10A1F", threads=2)
        granules, target, kw = fake_earthaccess["download"]
        assert (
            target == tmp_path / "cache" / "MOD10A1F"
            and target.is_dir()
            and kw == {"threads": 2}
        )
        assert paths[0].name == "a.hdf"
        with earthdata.env() as opts:
            assert "GDAL_HTTP_COOKIEFILE" in opts

    def test_hdf4(self, monkeypatch):
        assert isinstance(earthdata.hdf4_available(), bool)
        monkeypatch.setattr(earthdata, "hdf4_available", lambda: False)
        with pytest.raises(RuntimeError, match="no HDF4 driver"):
            earthdata.require_hdf4("MOD10A1F")
        monkeypatch.setattr(earthdata, "hdf4_available", lambda: True)
        earthdata.require_hdf4("MOD10A1F")


# ── gee ───────────────────────────────────────────────────────────────────────


class TestGee:
    NATIVE = {
        "crs": "EPSG:32610",
        "crs_transform": (30.0, 0.0, 500000.0, 0.0, -30.0, 5300000.0),
        "shape_2d": (10000, 10000),
    }

    @pytest.fixture
    def fake_xee(self, monkeypatch):
        from xee import helpers

        monkeypatch.setattr(
            helpers, "extract_grid_params", lambda obj: dict(self.NATIVE)
        )

    def test_grid_params_crop(self, fake_xee):
        grid = gee.grid_params(object(), RAINIER)
        a, b, c, d, e, f = grid["crs_transform"]
        assert (a, e) == (30.0, -30.0) and b == d == 0.0
        nx, ny = grid["shape_2d"]
        assert 1000 < nx < 1100 and 950 < ny < 1100  # ~31 km × 30 km at 30 m
        west, south, east, north = parse_aoi(RAINIER).total_bounds("EPSG:32610")
        assert c <= west and c + nx * a >= east and f >= north and f + ny * e <= south
        assert gee.grid_params(object()) == self.NATIVE
        assert gee.grid_params(object(), None) == self.NATIVE
        assert gee.grid_params(object(), parse_aoi(None)) == self.NATIVE

    def test_grid_params_rotated_raises(self, monkeypatch):
        from xee import helpers

        monkeypatch.setattr(
            helpers,
            "extract_grid_params",
            lambda obj: {**self.NATIVE, "crs_transform": (30, 1, 0, 0, -30, 0)},
        )
        with pytest.raises(ValueError, match="Rotated"):
            gee.grid_params(object(), RAINIER)

    @pytest.fixture
    def fake_ee(self, monkeypatch):
        import ee

        seen = {}
        monkeypatch.setattr(
            auth.get("earthengine"),
            "ensure",
            lambda **kw: seen.setdefault("ensure", []).append(kw),
        )
        monkeypatch.setattr(
            auth.get("earthengine"), "xee_init_kwargs", lambda: {"project": "p"}
        )

        class Image:
            def __init__(self, asset="img"):
                self.asset = asset

        class ImageCollection:
            def __init__(self, src):
                self.src = src

            def first(self):
                return Image()

        class Geometry:
            def __init__(self, geojson, crs, geodesic):
                self.geojson, self.crs, self.geodesic = geojson, crs, geodesic

        class FeatureCollection:
            def __init__(self, asset):
                self.asset = asset
                self.bounds = None

            def filterBounds(self, geom):
                self.bounds = geom
                return self

            def getInfo(self):
                seen["fc"] = self
                return {
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"name": "a"},
                            "geometry": shapely.geometry.mapping(
                                shapely.box(0, 0, 1, 1)
                            ),
                        }
                    ]
                }

        for name, cls in (
            ("Image", Image),
            ("ImageCollection", ImageCollection),
            ("Geometry", Geometry),
            ("FeatureCollection", FeatureCollection),
        ):
            monkeypatch.setattr(ee, name, cls)
        monkeypatch.setattr(
            xr,
            "open_dataset",
            lambda obj, **kw: seen.__setitem__("open", (obj, kw)) or xr.Dataset(),
        )
        return seen

    def test_open_dataset(self, fake_ee, fake_xee):
        import ee

        gee.open_dataset(ee.Image("x"), RAINIER, chunks={})
        obj, kw = fake_ee["open"]
        assert (
            isinstance(obj, ee.ImageCollection)
            and kw["engine"] == "ee"
            and kw["crs"] == "EPSG:32610"
        )
        assert (
            kw["ee_init_if_necessary"] is True
            and kw["ee_init_kwargs"] == {"project": "p"}
            and kw["chunks"] == {}
        )
        gee.open_dataset(
            ee.ImageCollection("y"),
            grid={},
            engine="ee",
            crs="EPSG:4326",
            crs_transform=(1, 0, 0, 0, -1, 0),
            shape_2d=(2, 2),
        )
        assert fake_ee["open"][1]["crs"] == "EPSG:4326"
        assert fake_ee["ensure"]

    def test_features_and_geometry(self, fake_ee):
        gdf = gee.features_to_geodataframe("USGS/WBD/2017/HUC08", RAINIER)
        assert list(gdf["name"]) == ["a"] and gdf.crs.to_epsg() == 4326
        assert (
            fake_ee["fc"].bounds.geojson["type"] == "Polygon"
            and fake_ee["fc"].bounds.geodesic is False
        )
        gee.features_to_geodataframe("X", None)
        assert fake_ee["fc"].bounds is None
        gee.features_to_geodataframe("X", parse_aoi(None))
        assert fake_ee["fc"].bounds is None
        assert gee.geometry(DATELINE).geojson["type"] == "MultiPolygon"


# ── raster_http ───────────────────────────────────────────────────────────────


@pytest.mark.recorded
class TestRasterHttp:
    def test_open_local_cog(self, fixtures_dir):
        path = fixtures_dir / "classes.tif"
        da = raster_http.open(path, chunks=None)
        assert (
            da.dims == ("y", "x")
            and da.shape == (64, 64)
            and da.rio.crs.to_epsg() == 32610
        )
        assert da.rio.nodata == 0
        full = raster_http.open(path, squeeze=False, chunks=None)
        assert full.dims == ("band", "y", "x")
        lazy = raster_http.open(path, chunks={"x": 32, "y": 32})
        assert lazy.chunks is not None and max(lazy.chunks[0]) == 32

    def test_open_clips_to_aoi(self, fixtures_dir):
        path = fixtures_dir / "classes.tif"
        small = (-121.80, 46.80, -121.70, 46.88)
        da = raster_http.open(path, small, chunks=None)
        assert da.shape[0] < 40 and da.shape[1] < 40
        whole = raster_http.open(path, parse_aoi(small, clip=False), chunks=None)
        assert whole.shape == (64, 64)
        world = raster_http.open(path, None, chunks=None)
        assert world.shape == (64, 64)

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # Remote: rasterio's own spelling, which it turns into
            # /vsizip/vsicurl/https://…
            ("https://h/a.zip", "zip+https://h/a.zip!/b.tif"),
            # Local: /vsizip/ directly. A POSIX absolute path keeps its leading
            # slash (/vsizip//tmp/… is absolute, /vsizip/tmp/… would not be);
            # a Windows drive letter must not gain one. Going through a
            # file:// URI instead gives /vsizip//C:/… , which GDAL cannot open
            # — the reason the offline tier failed on Windows in CI.
            ("file:///tmp/a.zip", "/vsizip//tmp/a.zip/b.tif"),
            ("/tmp/a.zip", "/vsizip//tmp/a.zip/b.tif"),
            ("file:///C:/Users/r/a.zip", "/vsizip/C:/Users/r/a.zip/b.tif"),
            ("C:/Users/r/a.zip", "/vsizip/C:/Users/r/a.zip/b.tif"),
        ],
    )
    def test_zip_url_spellings(self, url, expected):
        assert raster_http.zip_url(url, "/b.tif") == expected
        assert raster_http.zip_url(url, "b.tif") == expected

    def test_zip_url_and_fetch(self, monkeypatch, tmp_path):
        assert (
            raster_http.zip_url("https://h/a.zip", "/b.tif")
            == "zip+https://h/a.zip!/b.tif"
        )
        import pooch

        seen = {}
        monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            pooch,
            "retrieve",
            lambda url, known_hash, fname, path, **kw: (
                seen.update(url=url, fname=fname, path=path, hash=known_hash)
                or str(path / fname)
            ),
        )
        out = raster_http.fetch(
            "https://h/wmobb_json.zip", subdir="grdc", progressbar=False
        )
        assert (
            out == tmp_path / "grdc" / "wmobb_json.zip"
            and seen["fname"] == "wmobb_json.zip"
            and seen["hash"] is None
        )
        raster_http.fetch("https://h/x.zip", "renamed.zip", known_hash="md5:abc")
        assert (
            seen["fname"] == "renamed.zip"
            and seen["path"] == tmp_path
            and seen["hash"] == "md5:abc"
        )


# ── zarr_cloud ────────────────────────────────────────────────────────────────


# zarr 3 runs an asyncio loop whose self-pipe is a Unix socketpair on POSIX but a
# TCP loopback pair on Windows; allow loopback only (the network stays blocked).
@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@pytest.mark.recorded
class TestZarrCloud:
    def test_storage_options(self):
        assert zarr_cloud.storage_options_for("gs://b/s.zarr") == {"token": "anon"}
        assert zarr_cloud.storage_options_for("s3://b/s.zarr") == {"anon": True}
        assert zarr_cloud.storage_options_for("abfs://c/s.zarr") == {"anon": True}
        assert zarr_cloud.storage_options_for("/local/s.zarr") == {}
        assert zarr_cloud.storage_options_for("https://h/s.zarr") == {}

    def test_open_local(self, fixtures_dir):
        ds = zarr_cloud.open(str(fixtures_dir / "reanalysis.zarr"))
        assert "2m_temperature" in ds and ds["2m_temperature"].chunks is None
        lazy = zarr_cloud.open(
            str(fixtures_dir / "reanalysis.zarr"), chunks={}, storage_options={}
        )
        assert lazy["2m_temperature"].chunks is not None

    def test_select_0_360_grid(self, fixtures_dir):
        ds = zarr_cloud.open(str(fixtures_dir / "reanalysis.zarr"))
        sub = zarr_cloud.select(ds, RAINIER, "2020-01-01T00:00/2020-01-01T01:00")
        assert sub.sizes["time"] == 2
        lons = sub["longitude"].values
        assert lons.min() >= 238 and lons.max() <= 238.5  # -121.94 → 238.06
        lats = sub["latitude"].values
        assert lats.max() <= 47.0 and lats.min() >= 46.5 and lats.size >= 1
        big = zarr_cloud.select(ds, (-125.0, 40.0, -120.0, 50.0))
        assert big["latitude"].values[0] > big["latitude"].values[-1]  # descending kept
        assert big.sizes["latitude"] == 41 and big.sizes["longitude"] == 21
        assert zarr_cloud.select(ds, None) is ds
        assert zarr_cloud.select(ds, parse_aoi(None)).sizes == ds.sizes

    def test_select_across_dateline_and_negative_grid(self, fixtures_dir):
        ds = zarr_cloud.open(str(fixtures_dir / "reanalysis.zarr"))
        sub = zarr_cloud.select(ds, DATELINE)
        assert sub["longitude"].min() >= 170 and sub["longitude"].max() <= 190
        neg = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby(
            "longitude"
        )
        sub = zarr_cloud.select(neg, RAINIER)
        assert -122 <= sub["longitude"].min() and sub["longitude"].max() <= -121.5
        sub = zarr_cloud.select(neg, DATELINE)
        assert (
            sub.sizes["longitude"] > 0
            and (sub["longitude"] > 170).any()
            and (sub["longitude"] < -170).any()
        )
        # a 0..360 AOI that crosses 0° on the 0..360 grid
        sub = zarr_cloud.select(ds, (-5.0, 40.0, 5.0, 45.0))
        assert sub.sizes["longitude"] == 41


# ── vector_http ───────────────────────────────────────────────────────────────


@pytest.mark.recorded
class TestVectorHttp:
    def test_zip_path(self):
        assert (
            vector_http.zip_path("https://h/a.zip", "b.json")
            == "zip+https://h/a.zip!b.json"
        )
        assert vector_http.zip_path("/tmp/a.zip") == "zip:///tmp/a.zip"

    def test_read_geojson_pushdown(self, fixtures_dir):
        path = fixtures_dir / "basins.geojson"
        assert len(vector_http.read(path)) == 6
        hit = vector_http.read(path, RAINIER)
        assert set(hit["name"]) == {"basin-0", "basin-1"}
        triangle = shapely.Polygon([(-122.0, 46.6), (-121.7, 46.6), (-122.0, 46.9)])
        assert list(vector_http.read(path, triangle)["name"]) == ["basin-0"]
        assert len(vector_http.read(path, parse_aoi(RAINIER, clip=False))) == 6
        assert len(vector_http.read(path, RAINIER, rows=1)) == 1
        assert (
            len(vector_http.read(path, RAINIER, bbox=(170, -20, 176, -14))) == 1
        )  # explicit bbox wins
        assert list(vector_http.read(path, columns=["name"]).columns) == [
            "name",
            "geometry",
        ]

    def test_read_parquet_pushdown(self, fixtures_dir):
        path = fixtures_dir / "basins.parquet"
        assert len(vector_http.read(path)) == 6
        assert set(vector_http.read(path, RAINIER)["name"]) == {"basin-0", "basin-1"}
        triangle = shapely.Polygon([(-122.0, 46.6), (-121.7, 46.6), (-122.0, 46.9)])
        assert list(vector_http.read_parquet(path, triangle)["name"]) == ["basin-0"]
        assert (
            len(vector_http.read_parquet(path, RAINIER, columns=["name"]).columns) == 2
        )
        assert len(vector_http.read_parquet(path, parse_aoi(RAINIER, clip=False))) == 6


def test_package_exports():
    assert (
        providers.gdal_env is not None
        and providers.CLOUD_DEFAULTS["GDAL_HTTP_MAX_RETRY"] == "5"
    )
    assert providers.stac is stac and providers.zarr_cloud is zarr_cloud


class TestFetchFreshness:
    """raster_http.fetch(max_age=) re-downloads a stale cached file."""

    def test_stale_file_is_removed_before_retrieve(self, tmp_path, monkeypatch):
        import os
        import time

        import pooch

        from easysnowdata.providers import raster_http

        monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path))
        cached = tmp_path / "stations" / "bundle.tar.xz"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"old")
        old = time.time() - 3 * 24 * 3600
        os.utime(cached, (old, old))
        calls: list[dict] = []

        def retrieve(url, known_hash, fname, path, downloader):
            calls.append({"url": url, "fname": fname, "existed": cached.exists()})
            (tmp_path / "stations" / fname).write_bytes(b"new")
            return str(tmp_path / "stations" / fname)

        monkeypatch.setattr(pooch, "retrieve", retrieve)
        out = raster_http.fetch(
            "https://example.org/bundle.tar.xz",
            "bundle.tar.xz",
            subdir="stations",
            max_age=24 * 3600,
        )
        assert calls[0]["existed"] is False  # the stale copy was unlinked first
        assert out.read_bytes() == b"new"
        # a fresh copy is left alone
        raster_http.fetch(
            "https://example.org/bundle.tar.xz",
            "bundle.tar.xz",
            subdir="stations",
            max_age=24 * 3600,
        )
        assert calls[1]["existed"] is True
