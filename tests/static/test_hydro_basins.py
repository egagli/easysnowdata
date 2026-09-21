"""Basins: catalog entries, the four loaders offline on fixtures, live smoke."""

from __future__ import annotations

import json

import geopandas as gpd
import pytest

from easysnowdata import auth, catalog, providers
from easysnowdata.hydro import basins

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def fake_wbd(monkeypatch):
    """The WBD REST service, answering from a canned GeoJSON page."""
    state: dict = {"pages": None, "calls": []}

    class Response:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    def feature(huc, name):
        return {
            "type": "Feature",
            "properties": {
                "huc8": huc,
                "huc12": huc,
                "name": name,
                "areasqkm": 1994.5,
                "states": "WA",
                "tnmid": "{ABC}",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-121.9, 46.8],
                        [-121.6, 46.8],
                        [-121.6, 46.95],
                        [-121.9, 46.95],
                        [-121.9, 46.8],
                    ]
                ],
            },
        }

    def get(url, params=None, timeout=None):
        state["calls"].append((url, dict(params or {})))
        pages = state["pages"]
        if pages is None:
            return Response(
                {
                    "type": "FeatureCollection",
                    "features": [feature("17110015", "Nisqually")],
                }
            )
        index = len(state["calls"]) - 1
        return pages[min(index, len(pages) - 1)]

    monkeypatch.setattr("requests.get", get)
    state["Response"] = Response
    state["feature"] = feature
    return state


@pytest.fixture
def local_archives(static_fixtures, monkeypatch):
    """Serve the GRDC and HydroSHEDS archives from local fixtures."""
    files = {
        "wmobb_json.zip": static_fixtures["wmo_zip"],
        "hybas_na_lev01-12_v1c.zip": static_fixtures["hybas_zip"],
    }
    calls: dict = {}

    def fetch(url, fname=None, **kwargs):
        calls["url"] = url
        calls["subdir"] = kwargs.get("subdir")
        return files[fname]

    monkeypatch.setattr(providers.raster_http, "fetch", fetch)
    return calls


@pytest.fixture
def fake_vector_read(static_fixtures, monkeypatch):
    """Capture what the figshare route asks geopandas for."""
    calls: dict = {}

    def read(path, aoi=None, **kwargs):
        calls.update(path=path, aoi=aoi, **kwargs)
        return gpd.read_file(static_fixtures["wmo_zip"])

    monkeypatch.setattr(providers.vector_http, "read", read)
    return calls


class TestCatalogEntries:
    def test_all_four_registered_from_the_module(self):
        loaders = {
            "huc": basins.huc,
            "hydrobasins": basins.hydrobasins,
            "grdc-major-river-basins": basins.grdc_major,
            "grdc-wmo-basins": basins.grdc_wmo,
        }
        for product_id, loader in loaders.items():
            product = catalog.get(product_id)
            assert product.resolve_loader() is loader and product.theme == "hydro"
        assert catalog.validate_all() == []

    def test_huc_default_source_is_credential_free(self):
        product = catalog.get("huc")
        assert [s.id for s in product.sources] == ["usgs-wbd", "gee"]
        assert product.requires == ()
        assert [s.id for s in product.credential_free_sources] == ["usgs-wbd"]

    def test_probe_labels(self):
        labels = [
            p.label
            for pid in (
                "huc",
                "hydrobasins",
                "grdc-major-river-basins",
                "grdc-wmo-basins",
            )
            for s in catalog.get(pid).sources
            for p in s.health
        ]
        # The rows the weekly health check already records keep their labels.
        assert "HUC geometries (GEE/USGS WBD)" in labels
        assert "GRDC WMO basins" in labels
        assert "GRDC major river basins (World Bank)" in labels
        assert "HydroATLAS basins (figshare)" in labels
        assert "HUC geometries (USGS WBD REST)" in labels
        assert len(labels) == len(set(labels))

    def test_output_is_always_epsg_4326(self):
        import shapely

        product, src = basins._resolve("huc", None)
        web_mercator = gpd.GeoDataFrame(
            geometry=[shapely.box(-13_000_000, 5_900_000, -12_900_000, 6_000_000)],
            crs="EPSG:3857",
        )
        assert basins._finish(web_mercator, product, src).crs.to_epsg() == 4326
        no_crs = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1)])
        assert basins._finish(no_crs, product, src).crs.to_epsg() == 4326

    def test_load_dispatches_by_dataset(self, monkeypatch):
        monkeypatch.setattr(basins, "grdc_major", lambda aoi, **kw: ("major", aoi, kw))
        monkeypatch.setitem(
            basins._LOADERS, "grdc-major-river-basins", basins.grdc_major
        )
        assert basins.load(RAINIER, dataset="grdc-major-river-basins")[0] == "major"
        with pytest.raises(ValueError, match="Unknown basin dataset"):
            basins.load(RAINIER, dataset="nope")


@pytest.mark.recorded
class TestHuc:
    def test_rest_query_and_columns(self, fake_wbd):
        gdf = basins.huc(RAINIER, level=8)
        url, params = fake_wbd["calls"][0]
        assert url.endswith("/MapServer/4/query")
        assert params["f"] == "geojson" and params["outSR"] == "4326"
        assert params["geometryType"] == "esriGeometryEnvelope"
        assert params["geometry"].startswith("-121.940000,46.720000")
        assert params["resultRecordCount"] == 200 and params["resultOffset"] == 0
        assert list(gdf.columns) == [
            "name",
            "huc8",
            "areasqkm",
            "states",
            "tnmid",
            "geometry",
        ]
        assert gdf.crs.to_epsg() == 4326
        assert gdf.attrs["product_id"] == "huc" and gdf.attrs["source"] == "usgs-wbd"
        assert gdf.attrs["huc_level"] == 8
        assert gdf.attrs["data_citation"].startswith("Jones")

    def test_levels_map_to_layers(self, fake_wbd):
        basins.huc(RAINIER, level="12")
        assert fake_wbd["calls"][-1][0].endswith("/MapServer/6/query")
        with pytest.raises(ValueError, match="HUC level must be one of"):
            basins.huc(RAINIER, level=5)
        with pytest.raises(ValueError, match="Level must be between"):
            basins.huc(RAINIER, level=99)

    def test_all_columns_and_where(self, fake_wbd):
        gdf = basins.huc(RAINIER, columns="all", where="states LIKE '%WA%'")
        _, params = fake_wbd["calls"][0]
        assert params["outFields"] == "*" and params["where"] == "states LIKE '%WA%'"
        assert "tnmid" in gdf.columns

    def test_explicit_column_list(self, fake_wbd):
        gdf = basins.huc(RAINIER, columns=["huc8", "name"])
        assert list(gdf.columns) == ["huc8", "name", "geometry"]

    def test_paging_follows_exceeded_transfer_limit(self, fake_wbd):
        Response, feature = fake_wbd["Response"], fake_wbd["feature"]
        fake_wbd["pages"] = [
            Response(
                {
                    "type": "FeatureCollection",
                    "features": [feature("1", "a"), feature("2", "b")],
                    "exceededTransferLimit": True,
                }
            ),
            Response({"type": "FeatureCollection", "features": [feature("3", "c")]}),
        ]
        gdf = basins.huc(RAINIER, level=12, page_size=2)
        assert len(gdf) == 3
        assert [c[1]["resultOffset"] for c in fake_wbd["calls"]] == [0, 2]

    def test_a_failing_page_is_retried_then_reported(self, fake_wbd, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        fake_wbd["pages"] = [fake_wbd["Response"]({}, status=500)]
        with pytest.raises(RuntimeError, match="times out on large queries"):
            basins.huc(RAINIER)
        assert len(fake_wbd["calls"]) == 3  # two retries

    def test_service_error_payload(self, fake_wbd):
        fake_wbd["pages"] = [fake_wbd["Response"]({"error": {"code": 400}})]
        with pytest.raises(RuntimeError, match="USGS WBD service error"):
            basins.huc(RAINIER)

    def test_global_aoi_skips_the_geometry_filter(self, fake_wbd):
        basins.huc(None, level=2)
        assert "geometry" not in fake_wbd["calls"][0][1]

    def test_gee_route(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            providers.gee,
            "features_to_geodataframe",
            lambda asset, aoi: (
                seen.setdefault("asset", asset)
                and None
                or gpd.read_file(
                    json.dumps(
                        {
                            "type": "FeatureCollection",
                            "features": [
                                {
                                    "type": "Feature",
                                    "properties": {
                                        "name": "Nisqually",
                                        "huc8": "17110015",
                                    },
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [-121.8, 46.8],
                                    },
                                }
                            ],
                        }
                    )
                )
            ),
        )
        gdf = basins.huc(RAINIER, source="gee")
        assert seen["asset"] == "USGS/WBD/2017/HUC08"
        assert gdf.attrs["source"] == "gee" and gdf.crs.to_epsg() == 4326

    def test_gee_route_needs_credentials(self, monkeypatch):
        provider = auth.get("earthengine")
        monkeypatch.setattr(provider, "_ensured", None)
        monkeypatch.setattr(provider, "detect", lambda: auth.Detection(False))
        with pytest.raises(auth.CredentialError):
            basins.huc(RAINIER, source="gee")


@pytest.mark.recorded
class TestHydroBasins:
    def test_figshare_is_the_default(self, fake_vector_read):
        gdf = basins.hydrobasins(RAINIER, level=6, columns=["HYBAS_ID"], rows=2)
        assert fake_vector_read["path"] == f"zip+{basins.BASINATLAS_URL}"
        assert fake_vector_read["layer"] == "BasinATLAS_v10_lev06"
        assert (
            fake_vector_read["columns"] == ["HYBAS_ID"]
            and fake_vector_read["rows"] == 2
        )
        assert gdf.attrs["source"] == "figshare-basinatlas" and gdf.attrs["level"] == 6
        assert gdf.attrs["product_id"] == "hydrobasins"

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="Level must be between 1 and 12"):
            basins.hydrobasins(RAINIER, level=15)

    def test_hydrosheds_reads_the_regional_zip(self, local_archives):
        gdf = basins.hydrobasins(RAINIER, level=5, source="hydrosheds")
        assert local_archives["url"].endswith("hybas_na_lev01-12_v1c.zip")
        assert local_archives["subdir"] == "hydrosheds"
        assert len(gdf) == 2 and "HYBAS_ID" in gdf.columns  # only the intersecting ones
        assert gdf.attrs["source"] == "hydrosheds" and gdf.crs.to_epsg() == 4326

    def test_hydrosheds_region_inference(self):
        assert basins._hydrosheds_region(RAINIER) == "na"
        assert basins._hydrosheds_region((7.0, 45.5, 8.0, 46.5)) == "eu"
        assert basins._hydrosheds_region((86.0, 27.5, 87.0, 28.5)) == "as"
        assert basins._hydrosheds_region((-70.0, -33.5, -69.0, -32.5)) == "sa"
        with pytest.raises(ValueError, match="per region"):
            basins._hydrosheds_region(None)
        with pytest.raises(ValueError, match="No HydroSHEDS region"):
            basins._hydrosheds_region((-30.0, -60.0, -29.0, -59.0))

    def test_unknown_region(self, local_archives):
        with pytest.raises(ValueError, match="Unknown HydroSHEDS region"):
            basins.hydrobasins(RAINIER, source="hydrosheds", region="xx")

    def test_empty_region_warns(self, local_archives, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="easysnowdata.hydro.basins"):
            basins.hydrobasins(
                (-120.0, 40.0, -119.9, 40.1), level=5, source="hydrosheds"
            )
        assert "pass region=" in caplog.text

    def test_gee_route(self, monkeypatch, static_fixtures):
        seen = {}
        monkeypatch.setattr(
            providers.gee,
            "features_to_geodataframe",
            lambda asset, aoi: (
                seen.setdefault("asset", asset)
                and None
                or gpd.read_file(static_fixtures["wmo_zip"])
            ),
        )
        gdf = basins.hydrobasins(RAINIER, level=7, source="gee")
        assert seen["asset"] == "WWF/HydroATLAS/v1/Basins/level07"
        assert gdf.attrs["source"] == "gee"


@pytest.mark.recorded
class TestGrdc:
    def test_major_reads_the_world_bank_zip(self, fake_vector_read):
        gdf = basins.grdc_major(RAINIER, rows=5)
        assert fake_vector_read["path"] == f"zip+{basins.GRDC_MAJOR_URL}"
        assert fake_vector_read["rows"] == 5 and fake_vector_read["aoi"] == RAINIER
        assert gdf.attrs["product_id"] == "grdc-major-river-basins"
        assert gdf.attrs["source"] == "world-bank"

    def test_wmo_fetches_then_reads_locally(self, local_archives):
        gdf = basins.grdc_wmo(RAINIER)
        assert local_archives["url"] == basins.GRDC_WMO_URL
        assert local_archives["subdir"] == "grdc"
        assert len(gdf) == 2 and "WMOBB_NAME" in gdf.columns
        assert gdf.crs.to_epsg() == 4326 and gdf.attrs["source"] == "grdc"
        assert gdf.attrs["license"] and gdf.attrs["easysnowdata_version"]

    def test_wmo_without_an_aoi_returns_everything(self, local_archives):
        assert len(basins.grdc_wmo(None)) == 3


@pytest.mark.live
class TestLive:
    def test_huc_rest_smoke(self):
        gdf = basins.huc(RAINIER, level=12)
        assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) > 1
        assert gdf.crs.to_epsg() == 4326
        assert list(gdf.columns) == [
            "name",
            "huc12",
            "areasqkm",
            "states",
            "tnmid",
            "geometry",
        ]
        assert gdf.attrs["product_id"] == "huc" and gdf.attrs["source"] == "usgs-wbd"
        assert gdf.geometry.is_valid.all()

    def test_huc8_names_the_nisqually(self):
        gdf = basins.huc(RAINIER, level=8)
        assert "Nisqually" in set(gdf["name"])

    def test_hydrobasins_smoke(self):
        gdf = basins.hydrobasins(RAINIER, level=5)
        assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326
        assert gdf.attrs["source"] == "figshare-basinatlas"

    def test_grdc_major_smoke(self):
        gdf = basins.grdc_major(RAINIER)
        assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326
        assert gdf.attrs["product_id"] == "grdc-major-river-basins"

    def test_grdc_wmo_smoke(self):
        gdf = basins.grdc_wmo(RAINIER)
        assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326
        assert "WMOBB_NAME" in gdf.columns

    @pytest.mark.requires_earthengine
    def test_huc_gee_smoke(self):
        gdf = basins.huc(RAINIER, level=8, source="gee")
        assert len(gdf) >= 1 and gdf.attrs["source"] == "gee"
