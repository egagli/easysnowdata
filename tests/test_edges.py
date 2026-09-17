"""Edge cases that the main test modules do not reach (keeps the new modules at 100%)."""

from __future__ import annotations

import json
import types

import pytest
import shapely
import xarray as xr
from odc.geo import res_

from easysnowdata import _gdal, auth, catalog, temporal
from easysnowdata.aoi import estimate_utm_crs, parse_aoi
from easysnowdata.auth import earthengine as eeprov
from easysnowdata.auth._base import Detection, Provider
from easysnowdata.catalog import Product, Source, _registry
from easysnowdata.processing import optical
from easysnowdata.providers import earthdata, vector_http, zarr_cloud


def test_aoi_edges():
    assert estimate_utm_crs(
        parse_aoi((178.0, -20.0, -170.0, -10.0)).geometry
    ).is_projected  # centroid past 180°
    gb = parse_aoi((-121.94, 46.72, -121.54, 46.99)).to_geobox(
        resolution=res_(0.1), crs="EPSG:4326"
    )
    assert gb.resolution.x == 0.1
    buffered = parse_aoi((170.0, -20.0, -170.0, -10.0)).buffer(50_000)
    assert (
        buffered.crosses_antimeridian and buffered.geometry.geom_type == "MultiPolygon"
    )


def test_optical_and_temporal_edges():
    assert optical._coerce(None) is None
    start, _ = temporal.parse_time("2023-10-01T12:00:00+00:00")
    assert str(start) == "2023-10-01 12:00:00"


def test_earthengine_edges(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for var in ("EARTHENGINE_TOKEN", "EE_PROJECT_ID", "EARTHENGINE_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    provider = auth.get("earthengine")
    provider.reset()
    creds = eeprov.credentials_path()
    creds.parent.mkdir(parents=True)
    creds.write_text("{not json")
    assert provider.project_id() is None
    creds.write_text(json.dumps({"refresh_token": "rt", "project": "from-file"}))
    assert provider.xee_init_kwargs() == {
        "opt_url": eeprov.HIGH_VOLUME_URL,
        "project": "from-file",
    }


def test_auth_edges(monkeypatch):
    class Dummy(Provider):
        name = "dummy"
        title = "Dummy"

        def detect(self):
            return Detection(True, "env:X")

    d = Dummy()
    assert d.ensure() is True and d.ensure() is True  # cached second call

    import earthaccess

    provider = auth.get("earthdata")
    provider.reset()
    monkeypatch.setattr(provider, "detect", lambda: Detection(False))
    monkeypatch.setattr(
        earthaccess,
        "login",
        lambda strategy, persist=False: types.SimpleNamespace(authenticated=False),
    )
    with pytest.raises(auth.CredentialError, match="rejected the interactive"):
        provider.login(persist=False)

    monkeypatch.setattr(provider, "login", lambda **kw: None)
    auth.login("earthdata")  # the by-name branch returns after the provider's login


def test_catalog_edges():
    p = catalog.get("era5")
    assert p.source(None) is p.default_source
    tmp = Product(
        id="tmp-refs",
        theme="snow",
        title="T",
        description="d",
        sources=(Source("s", "raster_http", "u", health=lambda: None),),
        citation="c",
        license="l",
        loader="easysnowdata.aoi.parse_aoi",
        references=("https://example.com/paper",),
    )
    catalog.register(tmp)
    try:
        assert "- https://example.com/paper" in catalog.describe("tmp-refs")
    finally:
        _registry._PRODUCTS.pop("tmp-refs")


def test_provider_edges(fixtures_dir, monkeypatch):
    gdf = vector_http.read(
        fixtures_dir / "basins.geojson", layer="basins", use_arrow=False
    )
    assert len(gdf) == 6
    assert (
        vector_http._primary_geometry_column(fixtures_dir / "nope.parquet")
        == "geometry"
    )

    granule = earthdata._Granule = None  # noqa: F841 — no attribute of that name; keep linters quiet
    bare = {
        "umm": {
            "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"Points": []}}}
        }
    }
    assert earthdata._granule_geometry(bare) is None

    class Bad(dict):
        def size(self):
            raise RuntimeError("no size")

    out = earthdata.granules_to_geodataframe([Bad(meta={"native-id": "x"})])
    assert out["size_mb"].isna().all()

    seen = {}
    monkeypatch.setattr(
        xr, "open_zarr", lambda url, **kw: seen.update(kw) or xr.Dataset()
    )
    zarr_cloud.open("gs://bucket/store.zarr")
    assert seen["storage_options"] == {"token": "anon"}

    import odc.stac

    monkeypatch.setattr(_gdal, "dask_client", lambda: "client")
    monkeypatch.setattr(odc.stac, "configure_rio", lambda **kw: seen.update(rio=kw))
    with _gdal.gdal_env(GDAL_HTTP_MAX_RETRY="2"):
        pass
    assert (
        seen["rio"]["client"] == "client" and seen["rio"]["GDAL_HTTP_MAX_RETRY"] == "2"
    )
    assert shapely is not None
