"""easysnowdata.optical.planetscope — the Data API on scrubbed synthetic
responses, UDM2 decoding, and a delivery opened from synthetic COGs.

Planet imagery is not redistributable, so nothing here touches a real scene:
the Data API responses are hand-written (no real item ids), and the "delivery"
is the synthetic 4-band raster plus UDM2 mask from the fixture generator. The
live tier only *searches* (free); ordering spends quota and stays manual.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.optical import planetscope
from easysnowdata.processing import optical as optical_processing

RAINIER = (-121.94, 46.72, -121.54, 46.99)

# A synthetic Data API item: the id, the strip and the grid reference are made
# up, and the geometry is the test box. No Education & Research scene metadata
# appears in this repository.
SYNTHETIC_ITEM = {
    "id": "20230701_000000_00_0000",
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.0, 46.7],
                [-121.5, 46.7],
                [-121.5, 47.0],
                [-122.0, 47.0],
                [-122.0, 46.7],
            ]
        ],
    },
    "properties": {
        "acquired": "2023-07-01T18:30:12.000Z",
        "cloud_cover": 0.02,
        "cloud_percent": 2,
        "clear_percent": 95,
        "snow_ice_percent": 30,
        "item_type": "PSScene",
        "instrument": "PSB.SD",
        "pixel_resolution": 3.0,
        "epsg_code": 32610,
    },
    "assets": ["ortho_analytic_4b_sr", "ortho_udm2", "ortho_visual"],
    "_permissions": ["assets.ortho_analytic_4b_sr:download"],
}


@pytest.fixture
def fake_planet(monkeypatch, fake_credentials):
    """A stand-in Planet client: search returns synthetic items, orders record."""
    calls: dict[str, object] = {}

    class Data:
        def search(self, item_types, search_filter=None, limit=100, sort=None, **kw):
            calls["search"] = {
                "item_types": item_types,
                "filter": search_filter,
                "limit": limit,
            }
            return iter([SYNTHETIC_ITEM])

        def get_asset(self, item_type, item_id, asset_type):
            calls["get_asset"] = (item_type, item_id, asset_type)
            return {"status": "active", "location": calls["asset_location"]}

        def activate_asset(self, asset):  # pragma: no cover — already active
            calls["activated"] = asset

        def wait_asset(self, asset, **kw):  # pragma: no cover
            return asset

    class Orders:
        def create_order(self, request):
            calls["order_request"] = request
            return {"id": "order-0001", "state": "queued"}

        def wait(self, order_id, **kw):
            return "success"

        def download_order(self, order_id, directory=None, **kw):
            calls["downloaded"] = (order_id, directory)
            return list(calls.get("delivery_files", []))

    class Client:
        data = Data()
        orders = Orders()

    monkeypatch.setattr(planetscope.providers.planet, "ensure", lambda **kw: Client())
    monkeypatch.setattr(planetscope.providers.planet, "client", lambda **kw: Client())
    return calls


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("planetscope")
    assert product is planetscope.PRODUCT
    assert product.loader == "easysnowdata.optical.planetscope.load"
    assert product.resolve_loader() is planetscope.load
    assert [s.id for s in product.sources] == ["orders-api", "data-api"]
    assert product.requires == ("planet",)
    assert product.credential_free_sources == ()
    assert [p.label for s in product.sources for p in s.health] == [
        "PlanetScope (Planet Data API)"
    ]
    assert "not redistributable" in product.license
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_bundles_item_types_and_ids():
    assert planetscope._bundle("analytic_sr") == "analytic_sr_udm2"
    assert planetscope._bundle("analytic_8b_sr_udm2") == "analytic_8b_sr_udm2"
    with pytest.raises(ValueError, match="bundle must be one of"):
        planetscope._bundle("raw")
    assert planetscope._item_type("SkySatCollect") == "SkySatCollect"
    with pytest.raises(ValueError, match="item_type must be one of"):
        planetscope._item_type("REOrthoTile")
    assert planetscope._item_ids("abc") == ["abc"]
    assert planetscope._item_ids([SYNTHETIC_ITEM]) == ["20230701_000000_00_0000"]
    assert planetscope._item_ids(None) == []


def test_missing_credentials_are_named_before_any_call(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        planetscope.search(RAINIER, "2023-07")
    assert excinfo.value.provider == "planet"
    assert "planet auth login" in str(excinfo.value)


# ── search, on a synthetic Data API response ─────────────────────────────────


@pytest.mark.recorded
def test_search_returns_the_contract_frame(fake_planet):
    gdf = planetscope.search(
        RAINIER, "2023-07-01/2023-07-02", cloud_cover=20, asset_types=["ortho_udm2"]
    )
    assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326
    assert gdf["item_type"].iloc[0] == "PSScene"
    assert str(gdf["acquired"].iloc[0].date()) == "2023-07-01"
    assert gdf["snow_ice_percent"].iloc[0] == 30
    assert "planet_item" in gdf.columns
    assert gdf.attrs == {"source": "orders-api", "item_type": "PSScene"}

    sent = fake_planet["search"]
    assert sent["item_types"] == ["PSScene"] and sent["limit"] == 100
    # the filter carries geometry, date range, cloud cover, assets and permission
    kinds = {f["type"] for f in sent["filter"]["config"]}
    assert {"GeometryFilter", "DateRangeFilter", "RangeFilter", "AssetFilter"} <= kinds
    assert any(f["type"] == "PermissionFilter" for f in sent["filter"]["config"])
    cloud = next(
        f for f in sent["filter"]["config"] if f.get("field_name") == "cloud_cover"
    )
    assert cloud["config"]["lte"] == pytest.approx(0.2)  # percent → fraction


@pytest.mark.recorded
def test_search_filter_without_constraints(fake_planet):
    empty = planetscope.providers.planet.build_search_filter(permission=False)
    assert empty["type"] in {"AndFilter", "EmptyFilter"}
    one = planetscope.providers.planet.build_search_filter(RAINIER, permission=False)
    assert one["type"] == "GeometryFilter"


# ── ordering (the request is built and inspected; nothing is sent) ───────────


@pytest.mark.recorded
def test_order_builds_a_clipped_request(fake_planet, ts_fixtures):
    fake_planet["delivery_files"] = [
        ts_fixtures["planet_scene"],
        ts_fixtures["planet_udm2"],
    ]
    result = planetscope.order(RAINIER, items=["20230701_000000_00_0000"])
    request = fake_planet["order_request"]
    assert request["products"][0]["item_ids"] == ["20230701_000000_00_0000"]
    assert request["products"][0]["product_bundle"] == "analytic_sr_udm2"
    tools = {list(t)[0] for t in request["tools"]}
    assert "clip" in tools and "harmonize" in tools
    clip = next(t for t in request["tools"] if "clip" in t)["clip"]
    assert clip["aoi"]["type"] == "Polygon"
    assert result["order_id"] == "order-0001" and result["state"] == "success"
    assert len(result["files"]) == 2


@pytest.mark.recorded
def test_order_refuses_without_an_aoi(fake_planet):
    with pytest.raises(ValueError, match="AOI is required"):
        planetscope.order(None, items=["x"])
    with pytest.raises(ValueError, match="whole globe"):
        planetscope.order((-180, -90, 180, 90), items=["x"])
    with pytest.raises(ValueError, match="No item ids"):
        planetscope.order(RAINIER, items=[])


@pytest.mark.recorded
def test_load_never_orders_implicitly(fake_planet):
    with pytest.raises(ValueError, match="does not place orders"):
        planetscope.load(RAINIER, "2023-07")
    assert "order_request" not in fake_planet


# ── opening a delivery (synthetic COGs) ──────────────────────────────────────


@pytest.mark.recorded
def test_open_delivery_stacks_bands_and_decodes_udm2(fake_planet, ts_fixtures):
    ds = planetscope.open_delivery(ts_fixtures["planet_dir"], RAINIER)
    assert set(ds.data_vars) >= {
        "blue",
        "green",
        "red",
        "nir",
        "snow",
        "clear",
        "cloud",
    }
    assert ds["red"].dims == ("time", "y", "x")
    assert ds.sizes["time"] == 1
    assert str(ds["time"].values[0])[:10] == "2023-07-01"  # from the file name
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    assert ds["red"].chunks is not None
    # reflectance scaling and the nodata corner
    assert float(np.nanmax(ds["red"].values)) <= 1.0
    assert np.isnan(ds["red"].isel(time=0, y=0, x=0))
    # UDM2: the snow patch the fixture wrote
    assert ds["snow"].dtype == np.uint8
    assert int(ds["snow"].sum()) == 16
    assert ds["snow"].attrs["flag_meanings"] == "not_snow snow"
    assert ds.attrs["product_id"] == "planetscope"
    assert ds.attrs["source_id"] == "orders-api"
    assert "not redistributable" in ds.attrs["license"]
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_open_delivery_band_selection_and_errors(fake_planet, ts_fixtures):
    ds = planetscope.open_delivery(
        [ts_fixtures["planet_scene"]], RAINIER, bands=["red", "nir"], scale=False
    )
    assert set(ds.data_vars) == {"red", "nir"}
    assert ds["red"].dtype in (np.float32, np.float64)  # nodata masking makes it float
    raw = planetscope.open_delivery(
        [ts_fixtures["planet_scene"]], RAINIER, scale=False, mask_nodata=False
    )
    assert raw["red"].dtype == np.uint16
    with pytest.raises(ValueError, match="No GeoTIFFs found"):
        planetscope.open_delivery([], RAINIER)


@pytest.mark.recorded
def test_load_from_a_finished_order(fake_planet, ts_fixtures):
    order = {
        "order_id": "order-0001",
        "files": list(ts_fixtures["planet_dir"].glob("*.tif")),
    }
    ds = planetscope.load(RAINIER, order=order)
    assert "snow" in ds.data_vars and ds.attrs["product_id"] == "planetscope"


@pytest.mark.recorded
def test_load_data_api_reads_one_scene(fake_planet, ts_fixtures):
    fake_planet["asset_location"] = str(ts_fixtures["planet_scene"])
    ds = planetscope.load(
        RAINIER, "2023-07-01/2023-07-02", source="data-api", bands=["red"]
    )
    assert fake_planet["get_asset"] == (
        "PSScene",
        "20230701_000000_00_0000",
        "ortho_analytic_4b_sr",
    )
    assert set(ds.data_vars) == {"red"} and ds["red"].dims == ("time", "y", "x")
    assert ds.attrs["source_id"] == "data-api"
    assert ds.attrs["asset_type"] == "ortho_analytic_4b_sr"


# ── UDM2 decoding (pure processing) ──────────────────────────────────────────


def test_decode_udm2_bands_and_flags():
    data = np.zeros((8, 2, 2), dtype="uint8")
    data[0] = 1  # clear
    data[1, 0, 0] = 1  # snow
    data[6] = 88  # confidence
    data[7, 1, 1] = 1  # unusable: blackfill
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": range(1, 9), "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    ds = optical_processing.decode_udm2(da)
    assert list(ds.data_vars) == list(optical_processing.UDM2_BANDS)
    assert int(ds["snow"].isel(y=0, x=0)) == 1 and int(ds["snow"].sum()) == 1
    assert ds["snow"].attrs["flag_values"] == [0, 1]
    assert ds["confidence"].attrs["units"] == "%"
    assert int(optical_processing.udm1_bit(ds["unusable"], "blackfill").sum()) == 1
    assert ds.attrs["udm2_band_order"].startswith("clear snow")
    with pytest.raises(ValueError, match="Unknown UDM1 flag"):
        optical_processing.udm1_bit(ds["unusable"], "fog")
    with pytest.raises(ValueError, match="has 8 bands"):
        optical_processing.decode_udm2(da.isel(band=slice(0, 3)))
    with pytest.raises(ValueError, match="not a dimension"):
        optical_processing.decode_udm2(da.rename({"band": "b"}))
    # a Dataset of eight named variables decodes too
    assert "snow" in optical_processing.decode_udm2(
        da.to_dataset(dim="band").rename({i: f"band_{i}" for i in range(1, 9)})
    )


@pytest.mark.recorded
def test_snow_fraction_helper(fake_planet, ts_fixtures):
    ds = planetscope.open_delivery(ts_fixtures["planet_dir"], RAINIER)
    fraction = planetscope.snow_fraction(ds)
    assert fraction.dims == ("time",)
    assert 0.0 < float(fraction.isel(time=0)) < 1.0


# ── live smoke test: search only, never an order ─────────────────────────────


@pytest.mark.live
@pytest.mark.requires_planet
def test_live_planet_search_only():
    """Searches are free. Ordering spends quota and stays a manual step."""
    gdf = planetscope.search(RAINIER, "2023-07-01/2023-07-08", cloud_cover=50, limit=5)
    assert isinstance(gdf, gpd.GeoDataFrame)
    if len(gdf):
        assert gdf.crs.to_epsg() == 4326
        assert gdf["item_type"].iloc[0] == "PSScene"
        assert gdf["acquired"].notna().all()
