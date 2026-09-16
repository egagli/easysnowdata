"""Unit tests for easysnowdata.catalog — registry, validation, health runner (offline)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import easysnowdata as esd
from easysnowdata import auth, catalog
from easysnowdata.catalog import (
    Probe,
    Product,
    Source,
    Variable,
    _registry,
    health,
    validate,
)

KNOWN_AUTH = tuple(auth.PROVIDERS)


def _product(**overrides) -> Product:
    base = dict(
        id="test-product",
        theme="snow",
        title="Test",
        description="A test product.",
        sources=(
            Source(
                id="src",
                provider="raster_http",
                location="https://example.com/x.tif",
                health=lambda: None,
            ),
        ),
        citation="Someone (2024).",
        license="CC0",
        loader="easysnowdata.aoi.parse_aoi",
    )
    base.update(overrides)
    return Product(**base)


class TestRegistryContents:
    def test_every_product_validates(self):
        assert catalog.validate_all(known_auth=KNOWN_AUTH) == []

    def test_expected_products_and_probe_labels(self):
        ids = set(catalog.products())
        assert {
            "copernicus-dem",
            "era5",
            "snodas",
            "sentinel-2-l2a",
            "sentinel-1-rtc",
            "hls",
            "modis-snow",
            "snotel-ccss-stations",
        } <= ids
        labels = [probe.label for _, _, probe in health.probes()]
        # The rows the weekly health check has been recording stay unchanged
        for label in (
            "SNOTEL/CCSS station list (GitHub)",
            "GRDC WMO basins",
            "Köppen-Geiger classification (figshare)",
            "ARCO-ERA5 (GCS anonymous)",
            "Copernicus DEM (Planetary Computer)",
            "HUC geometries (GEE/USGS WBD)",
            "UCLA Snow Reanalysis (NASA NSIDC)",
        ):
            assert label in labels
        assert len(labels) == len(set(labels)), "probe labels must be unique"

    def test_default_source_is_first_and_products_are_frozen(self):
        p = catalog.get("era5")
        assert p.default_source.id == "arco-era5-gcs"
        assert p.source("gee").requires == ("earthengine",)
        assert p.requires == ()
        assert [s.id for s in p.credential_free_sources] == ["arco-era5-gcs"]
        with pytest.raises(ValueError, match="no source"):
            p.source("nope")
        with pytest.raises(AttributeError):
            p.id = "x"  # type: ignore[misc]

    def test_loaders_resolve_to_the_old_api(self):
        assert (
            catalog.get("copernicus-dem").resolve_loader()
            is esd.topography.get_copernicus_dem
        )
        assert (
            catalog.get("sentinel-2-l2a").resolve_loader() is esd.optical.sentinel2.load
        )

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown product"):
            catalog.get("nope")

    def test_themes(self):
        assert set(catalog.themes()) == {
            "stations",
            "hydro",
            "climate",
            "snow",
            "land",
            "terrain",
            "optical",
            "sar",
        }

    def test_cf_attrs_are_plain(self):
        var = catalog.get("sentinel-2-l2a").variables[-1]
        attrs = var.cf_attrs()
        assert attrs["flag_values"] == list(range(12))
        assert attrs["flag_meanings"].split()[-1] == "Snow_or_ice"
        assert attrs["flag_colors"].split()[0] == "#000000"
        assert all(isinstance(v, (str, list)) for v in attrs.values())
        assert catalog.get("copernicus-dem").variables[0].cf_attrs() == {
            "long_name": "elevation",
            "units": "m",
        }


class TestQueries:
    def test_list_and_filters(self):
        table = catalog.list()
        assert isinstance(table, pd.DataFrame) and table.index.name == "id"
        assert set(catalog.list(theme="terrain").index) == {"copernicus-dem", "chili"}
        assert "hls" in catalog.list(provider="stac").index
        assert set(catalog.list(requires="earthengine").index) >= {
            "huc",
            "chili",
            "nlcd",
        }
        # SNODAS moved to the credential-free NSIDC archive in Phase 2b; Earth
        # Engine is still one of its sources, just no longer the default one
        assert "snodas" not in catalog.list(requires="earthengine").index
        assert "snodas" in catalog.list(provider="gee").index
        assert "snodas" in catalog.list(credential_free=True).index
        assert "chili" not in catalog.list(credential_free=True).index
        assert "era5" in catalog.list(credential_free=True).index
        assert catalog.list(theme="nope").empty

    def test_search(self):
        assert "snodas" in catalog.search("SWE").index
        assert "koppen-geiger" in catalog.search("climate classification").index
        assert catalog.search("zzz-nothing").empty

    def test_describe_markdown(self):
        text = catalog.describe("copernicus-dem")
        assert text.startswith("# Copernicus DEM")
        assert "| `planetary-computer` (default) | stac | none | 30 m |" in text
        assert "**DOI:** 10.5069/G9028PQB" in text
        assert "| `data` | m | float32 | -32767 | no |" in text
        cat = catalog.describe("sentinel-2-l2a")
        assert "yes (12 classes)" in cat
        hls = catalog.describe("hls")
        assert "| earthdata |" in hls

    def test_register_duplicate(self):
        with pytest.raises(ValueError, match="already registered"):
            catalog.register(catalog.get("era5"))
        with pytest.raises(TypeError):
            catalog.products()["x"] = 1  # type: ignore[index]

    def test_register_and_replace(self):
        p = _product(id="tmp-product")
        try:
            catalog.register(p)
            assert catalog.get("tmp-product") is p
            catalog.register(_product(id="tmp-product", title="Replaced"), replace=True)
            assert catalog.get("tmp-product").title == "Replaced"
        finally:
            _registry._PRODUCTS.pop("tmp-product", None)

    def test_auth_status_lists_needed_products(self):
        table = auth.status()
        assert "chili" in table.loc["earthengine", "needed_by"]
        assert "hls" in table.loc["earthdata", "needed_by"]


class TestValidation:
    def test_valid(self):
        assert validate(_product(), known_auth=KNOWN_AUTH) == []

    @pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"id": "Bad_Id"}, "kebab-case"),
            ({"theme": "weather"}, "unknown theme"),
            ({"citation": ""}, "citation is empty"),
            ({"sources": ()}, "no sources"),
            ({"loader": "easysnowdata.nope.missing"}, "does not import"),
            ({"sources": (Source("a", "raster_http", "u"),)}, "no health probe"),
            (
                {"sources": (Source("a", "ftp", "u", health=lambda: None),)},
                "unknown provider",
            ),
            (
                {"sources": (Source("a", "raster_http", "", health=lambda: None),)},
                "location is empty",
            ),
            (
                {
                    "sources": (
                        Source(
                            "a",
                            "raster_http",
                            "u",
                            requires=("x",),
                            health=lambda: None,
                        ),
                    )
                },
                "unknown auth provider",
            ),
            (
                {
                    "sources": (
                        Source("a", "raster_http", "u", health=lambda: None),
                        Source("a", "raster_http", "u"),
                    )
                },
                "duplicate source ids",
            ),
            (
                {
                    "variables": (
                        Variable("v", flag_values=(1, 2), flag_meanings=("a",)),
                    )
                },
                "flag_meanings length",
            ),
            (
                {
                    "variables": (
                        Variable(
                            "v",
                            flag_values=(1, 2),
                            flag_meanings=("a", "b"),
                            flag_colors=("#000",),
                        ),
                    )
                },
                "flag_colors length",
            ),
            (
                {
                    "variables": (
                        Variable("v", flag_values=(1, 1), flag_meanings=("a", "b")),
                    )
                },
                "duplicate flag_values",
            ),
        ],
    )
    def test_problems(self, overrides, fragment):
        problems = validate(_product(**overrides), known_auth=KNOWN_AUTH)
        assert any(fragment in p for p in problems), problems

    def test_malformed_probe(self):
        src = Source("a", "raster_http", "u", health=(Probe("", lambda: None),))
        assert any(
            "malformed health probe" in p for p in validate(_product(sources=(src,)))
        )

    def test_source_health_normalisation(self):
        assert Source("a", "raster_http", "u").health == ()
        one = Source("a", "raster_http", "u", health=Probe("p", lambda: None))
        assert len(one.health) == 1 and one.title == "a"
        bare = Source("a", "raster_http", "u", title="T", health=lambda: None)
        assert bare.health[0].label == "T"


class TestHealthRunner:
    @pytest.fixture
    def creds(self, monkeypatch, tmp_path):
        for var in (
            "EARTHDATA_TOKEN",
            "EARTHDATA_USERNAME",
            "EARTHDATA_PASSWORD",
            "EARTHENGINE_TOKEN",
            "NETRC",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        auth.reset()
        yield
        auth.reset()

    def test_run_pass_fail_skip(self, creds, monkeypatch):
        calls = []

        def ok():
            calls.append("ok")

        def bad():
            raise RuntimeError("Unreachable: HTTP 404")

        product = _product(
            id="tmp-health",
            sources=(
                Source(
                    "open",
                    "raster_http",
                    "u",
                    health=(Probe("Row A", ok), Probe("Row B", bad)),
                ),
                Source(
                    "gated",
                    "earthdata",
                    "u",
                    requires=("earthdata",),
                    health=Probe("Row C", ok),
                ),
                Source(
                    "gated-open-probe",
                    "stac",
                    "u",
                    requires=("earthdata",),
                    health=Probe("Row D", ok, requires=()),
                ),
            ),
        )
        catalog.register(product)
        lines = []
        try:
            results = health.run(["tmp-health"], progress=lines.append)
        finally:
            _registry._PRODUCTS.pop("tmp-health")
        by_label = {r["source"]: r for r in results}
        assert (
            by_label["Row A"]["status"] == "pass" and by_label["Row A"]["error"] is None
        )
        assert by_label["Row B"]["status"] == "fail"
        assert by_label["Row B"]["error"] == "RuntimeError: Unreachable: HTTP 404"
        assert by_label["Row C"]["status"] == "skip"
        assert "NASA Earthdata" in by_label["Row C"]["error"]
        assert by_label["Row D"]["status"] == "pass"  # probe-level requires override
        assert calls == ["ok", "ok"]
        assert all(r["product"] == "tmp-health" for r in results)
        assert [r["route"] for r in results] == [
            "open",
            "open",
            "gated",
            "gated-open-probe",
        ]
        assert (
            lines[0].startswith("  ✅ Row A")
            and "→ RuntimeError" in lines[1]
            and "⚠️" in lines[2]
        )
        assert health.summarize(results) == {"pass": 2, "fail": 1, "skip": 1}

    def test_credentialed_probes_skip_without_credentials(self, creds):
        results = health.run(["chili", "ucla-snow-reanalysis"])
        assert {r["status"] for r in results} == {"skip"}
        assert "EARTHENGINE_TOKEN" in results[0]["error"]

    def test_update_history_and_main(self, creds, tmp_path, monkeypatch, capsys):
        history = tmp_path / "data_status" / "history.json"
        first = [{"source": "A", "status": "pass", "error": None, "checked_at": "t"}]
        health.update_history(first, history)
        health.update_history(
            [{"source": "B", "status": "fail", "error": "x", "checked_at": "t"}],
            history,
            keep=1,
        )
        data = json.loads(history.read_text())
        assert len(data) == 1 and data[0][0]["source"] == "B"

        def fake_run(ids, progress=None):
            if progress is not None:
                progress("  ✅ X")
            return first

        monkeypatch.setattr(health, "run", fake_run)
        code = health.main(["--output", str(history), "--product", "era5"])
        out = capsys.readouterr().out
        assert (
            code == 0
            and "1 passed, 0 failed, 0 skipped" in out
            and "History written" in out
        )
        assert json.loads(history.read_text())[0] == first

        monkeypatch.setattr(
            health,
            "run",
            lambda ids, progress=None: [
                {"source": "A", "status": "fail", "error": "e", "checked_at": "t"}
            ],
        )
        assert health.main(["--no-history", "--strict"]) == 1
        assert health.main(["--no-history"]) == 0

    def test_generic_probes_offline(self, monkeypatch):
        """Each generic probe raises on an empty answer (transport monkeypatched)."""
        import requests

        class Resp:
            def __init__(self, status):
                self.status_code = status

            def close(self):
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(202))
        monkeypatch.setattr(requests, "head", lambda *a, **k: Resp(200))
        health.http_first_byte("https://x")  # HEAD fallback saves it
        monkeypatch.setattr(requests, "head", lambda *a, **k: Resp(400))
        with pytest.raises(RuntimeError, match="HTTP 202"):
            health.http_first_byte("https://x")
        monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(206))
        health.http_first_byte("https://x")

        import pystac_client

        class Search:
            def items(self):
                return iter([])

        class Client:
            @staticmethod
            def open(url, modifier=None):
                return Client()

            def search(self, **kw):
                assert kw["max_items"] == 1
                return Search()

        monkeypatch.setattr(pystac_client, "Client", Client)
        with pytest.raises(RuntimeError, match="No coll items"):
            health.stac_search(
                "https://api",
                "coll",
                datetime_range="2020/2021",
                query={"a": {"eq": 1}},
            )

        import xarray as xr

        monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: xr.Dataset())
        with pytest.raises(RuntimeError, match="no data variables"):
            health.zarr_metadata("gs://bucket/store.zarr")
        monkeypatch.setattr(
            xr, "open_zarr", lambda *a, **k: xr.Dataset({"t2m": ("x", [1.0])})
        )
        health.zarr_metadata("gs://bucket/store.zarr")

    def test_gee_and_earthdata_probes_offline(self, monkeypatch):
        import types

        import earthaccess
        import ee

        monkeypatch.setattr(auth.get("earthengine"), "ensure", lambda **kw: None)
        monkeypatch.setattr(auth.get("earthdata"), "ensure", lambda **kw: None)

        class Num:
            def __init__(self, n):
                self.n = n

            def getInfo(self):
                return self.n

        class Col:
            def __init__(self, asset):
                self.asset = asset

            def filterDate(self, a, b):
                return self

            def size(self):
                return Num(0 if "empty" in self.asset else 1)

            def limit(self, n):
                return self

            def getInfo(self):
                return {"features": [] if "empty" in self.asset else [1]}

        monkeypatch.setattr(ee, "ImageCollection", Col)
        monkeypatch.setattr(ee, "FeatureCollection", Col)
        monkeypatch.setattr(
            ee,
            "Image",
            lambda asset: types.SimpleNamespace(
                getInfo=lambda: {} if "empty" in asset else {"id": asset}
            ),
        )
        health.gee_asset("A", start="2020-01-01", end="2020-01-02")
        health.gee_asset("A", "image")
        health.gee_asset("A", "feature_collection")
        for kind in ("image", "image_collection", "feature_collection"):
            with pytest.raises(RuntimeError):
                health.gee_asset("empty", kind)

        monkeypatch.setattr(
            earthaccess,
            "search_data",
            lambda **kw: [] if kw["short_name"] == "EMPTY" else [1],
        )
        health.earthdata_search("WUS_UCLA_SR")
        with pytest.raises(RuntimeError, match="no granules"):
            health.earthdata_search(
                "EMPTY", cloud_hosted=False, bbox=None, temporal=None
            )
