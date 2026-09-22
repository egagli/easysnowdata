"""easysnowdata.optical.hls — searches replayed from cassettes, the load path
on synthetic items, and the live smoke tests (Earthdata Login for the reads)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.optical import hls

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("hls")
    assert product is hls.PRODUCT
    assert product.loader == "easysnowdata.optical.hls.load"
    assert product.resolve_loader() is hls.load
    assert [s.id for s in product.sources] == [
        "lpcloud-cmr-stac",
        "planetary-computer",
    ]
    assert product.default_source.requires == ("earthdata",)
    # the probe label the weekly health check records stays the same, and the
    # CMR-STAC search itself needs no credentials
    probe = product.default_source.health[0]
    assert probe.label == "HLS L30 (CMR-STAC LPCLOUD)" and probe.requires == ()
    assert product.source("planetary-computer").requires == ()
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_band_tables_and_products():
    assert hls._products(None) == ["L30", "S30"]
    assert hls._products("S30") == ["S30"]
    with pytest.raises(ValueError, match="products must be in"):
        hls._products(["S31"])
    assert "swir16" in hls.COMMON_BANDS and "rededge1" not in hls.COMMON_BANDS
    assert hls.BAND_ASSETS["L30"]["nir08"] == "B05"  # Landsat OLI
    assert hls.BAND_ASSETS["S30"]["nir08"] == "B8A"  # Sentinel-2 narrow NIR
    assert hls.BAND_ASSETS["S30"]["swir16"] == "B11"
    cfg = hls._stac_cfg("HLSL30_2.0", "L30", ["blue", "Fmask", "SZA"])
    assert cfg["HLSL30_2.0"]["assets"]["*"]["nodata"] == -9999
    assert cfg["HLSL30_2.0"]["assets"]["Fmask"]["data_type"] == "uint8"
    assert cfg["HLSL30_2.0"]["assets"]["SZA"]["nodata"] == 40000
    assert cfg["HLSL30_2.0"]["aliases"] == {"blue": "B02"}
    # a band that only one product has is skipped for the other
    assert hls._asset_names("L30", ["rededge1", "blue", "Fmask"]) == ["blue", "Fmask"]
    assert hls._asset_names("S30", ["rededge1", "blue"]) == ["rededge1", "blue"]


def test_product_and_platform_from_metadata():
    cmr = {"id": "HLS.S30.T10TFT.2023215T190921.v2.0", "properties": {}}
    assert hls._product_of(cmr, "HLSS30_2.0") == "S30"
    assert hls._platform_of(cmr, "S30") == "sentinel-2a/b"
    pc = {"id": "HLS.L30.T10TES.2023215.v2.0", "properties": {"platform": "landsat-9"}}
    assert hls._product_of(pc, "hls2-l30") == "L30"
    assert hls._platform_of(pc, "L30") == "landsat-9"
    # an id that says nothing falls back to the collection
    assert hls._product_of({"id": "x", "properties": {}}, "hls2-s30") == "S30"
    assert hls._product_of({"id": "x", "properties": {}}, "nope") == ""


# ── searches, replayed ────────────────────────────────────────────────────────


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_cmr_lpcloud_is_open_and_labels_products():
    gdf = hls.search(RAINIER, "2023-08-01/2023-08-06", max_items=3)
    assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) >= 2
    assert set(gdf["product"]) <= {"L30", "S30"}
    assert set(gdf["product"]) == {"L30", "S30"}
    assert gdf["platform"].str.startswith(("landsat", "sentinel")).all()
    assert gdf["datetime"].is_monotonic_increasing
    assert "eo:cloud_cover" in gdf.columns  # STAC properties, not the XML scrape
    assert gdf.attrs["source"] == "lpcloud-cmr-stac"


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_planetary_computer_mirror():
    gdf = hls.search(
        RAINIER,
        "2023-08-01/2023-08-06",
        source="planetary-computer",
        products="S30",
        max_items=3,
    )
    assert len(gdf) >= 1 and set(gdf["product"]) == {"S30"}
    assert (gdf["collection"] == "hls2-s30").all()
    assert gdf["platform"].iloc[0].startswith("sentinel-2")


# ── the load path, on synthetic items ────────────────────────────────────────


def _item(item_id: str, when: str, collection: str):
    return {
        "id": item_id,
        "collection": collection,
        "properties": {"datetime": when, "eo:cloud_cover": 5},
        "assets": {"B04": {"href": "https://x/B04.tif"}},
    }


def _dataset(values: dict[str, list[int]], times: list[str]) -> xr.Dataset:
    time = pd.to_datetime(times).values
    data = {}
    for name, per_time in values.items():
        dtype = "uint8" if name == "Fmask" else "int16"
        block = np.array(
            [np.full((2, 2), v, dtype=dtype) for v in per_time], dtype=dtype
        )
        data[name] = (("time", "y", "x"), block)
    return xr.Dataset(
        data,
        coords={"time": time, "y": [5180400.0, 5180370.0], "x": [580000.0, 580030.0]},
    ).rio.write_crs("EPSG:32610")


@pytest.fixture
def fake_stac(monkeypatch, fake_credentials):
    """Capture the per-product odc-stac calls without touching the network."""
    calls: list[dict] = []
    items = {
        "L30": [_item("HLS.L30.T10TES.2023215.v2.0", "2023-08-03", "HLSL30_2.0")],
        "S30": [_item("HLS.S30.T10TES.2023215.v2.0", "2023-08-04", "HLSS30_2.0")],
    }

    def search(catalog_id, collection, aoi=None, time=None, **kwargs):
        product = "L30" if "L30" in collection or "l30" in collection else "S30"
        calls.append(
            {"kind": "search", "catalog": catalog_id, "collection": collection}
        )
        return items[product]

    def items_to_geodataframe(item_list):
        item_list = list(item_list)
        return gpd.GeoDataFrame(
            {
                "id": [i["id"] for i in item_list],
                "collection": [i.get("collection") for i in item_list],
                "datetime": pd.to_datetime(
                    [i["properties"]["datetime"] for i in item_list], utc=True
                ),
                "stac_item": item_list,
            },
            geometry=[None] * len(item_list),
            crs="EPSG:4326",
        )

    def load(item_arg, aoi=None, **kwargs):
        product = "L30" if "L30" in str(kwargs["stac_cfg"].keys()).upper() else "S30"
        calls.append({"kind": "load", "product": product, **kwargs})
        when = "2023-08-03" if product == "L30" else "2023-08-04"
        values = {"red": [3000], "Fmask": [0 if product == "L30" else 2]}
        wanted = {b: v for b, v in values.items() if b in kwargs["bands"]}
        return _dataset(wanted, [when])

    monkeypatch.setattr(hls.providers.stac, "search", search)
    monkeypatch.setattr(
        hls.providers.stac, "items_to_geodataframe", items_to_geodataframe
    )
    monkeypatch.setattr(hls.providers.stac, "load", load)
    return calls


@pytest.mark.recorded
def test_load_stacks_both_products_with_coordinates(fake_stac):
    ds = hls.load(RAINIER, "2023-08", bands=["red", "Fmask"])
    loads = [c for c in fake_stac if c["kind"] == "load"]
    assert [c["product"] for c in loads] == ["L30", "S30"]
    assert loads[0]["catalog"] == "cmr-lpcloud"
    assert loads[0]["resolution"] == 30 and loads[0]["crs"] == "utm"
    # a failed read raises (odc-stac's default) rather than filling with nodata
    assert "fail_on_error" not in loads[0]

    assert ds["red"].dims == ("time", "y", "x") and ds.sizes["time"] == 2
    assert list(ds["product"].values) == ["L30", "S30"]
    assert list(ds["platform"].values) == ["landsat-8/9", "sentinel-2a/b"]
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    # 3000 DN → 0.3 reflectance; Fmask keeps its sentinel
    assert float(ds["red"].isel(time=0, y=0, x=0)) == pytest.approx(0.3)
    assert ds["Fmask"].dtype == np.uint8 and ds["Fmask"].rio.nodata == 255
    assert ds.attrs["product_id"] == "hls"
    assert ds.attrs["source_id"] == "lpcloud-cmr-stac"
    assert ds.attrs["collections"] == "HLSL30_2.0 HLSS30_2.0"
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_load_one_product_and_the_mirror(fake_stac):
    ds = hls.load(
        RAINIER,
        "2023-08",
        bands=["red"],
        products="S30",
        source="planetary-computer",
        scale=False,
        mask_nodata=False,
    )
    loads = [c for c in fake_stac if c["kind"] == "load"]
    assert len(loads) == 1 and loads[0]["catalog"] == "planetary-computer"
    assert ds.sizes["time"] == 1 and ds["red"].dtype == np.int16
    assert int(ds["red"].isel(time=0, y=0, x=0)) == 3000
    assert ds["red"].rio.nodata == -9999
    assert ds.attrs["collections"] == "hls2-s30"


@pytest.mark.recorded
def test_fmask_masking(fake_stac):
    ds = hls.load(RAINIER, "2023-08", bands=["red", "Fmask"], mask="fmask-default")
    # the S30 scene has Fmask=2 (adjacent to cloud) → removed; L30 has 0 → kept
    assert not np.isnan(ds["red"].isel(time=0, y=0, x=0))
    assert np.isnan(ds["red"].isel(time=1, y=0, x=0))
    assert "cloud" in ds.attrs["masked_fmask_flags"]
    keep_all = hls.load(RAINIER, "2023-08", bands=["red", "Fmask"], mask=False)
    assert not np.isnan(keep_all["red"].isel(time=1, y=0, x=0))
    with pytest.raises(ValueError, match="Unknown Fmask flags"):
        hls.load(RAINIER, "2023-08", bands=["red", "Fmask"], mask=["fog"])
    with pytest.raises(ValueError, match="Unknown mask"):
        hls.load(RAINIER, "2023-08", bands=["red", "Fmask"], mask="everything")
    with pytest.raises(ValueError, match="needs the 'Fmask' band"):
        hls.load(RAINIER, "2023-08", bands=["red"], mask=True)


@pytest.mark.recorded
def test_load_without_items_raises(monkeypatch, fake_credentials):
    monkeypatch.setattr(hls.providers.stac, "search", lambda *a, **k: [])
    monkeypatch.setattr(
        hls.providers.stac,
        "items_to_geodataframe",
        lambda items: gpd.GeoDataFrame(
            {"id": [], "collection": [], "datetime": [], "stac_item": []},
            geometry=[],
            crs="EPSG:4326",
        ),
    )
    with pytest.raises(ValueError, match="No HLS items"):
        hls.load(RAINIER, "1999-01")


@pytest.mark.recorded
def test_load_without_credentials_names_the_free_alternative(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        hls.load(RAINIER, "2023-08")
    message = str(excinfo.value)
    assert excinfo.value.provider == "earthdata"
    assert 'source="planetary-computer"' in message and "hls" in message


# ── the deprecation shim ──────────────────────────────────────────────────────


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_hls_search_is_open_without_credentials():
    """The CMR-STAC search needs no Earthdata Login — only the reads do."""
    gdf = hls.search(RAINIER, "2023-08-01/2023-08-06", cloud_cover=60)
    assert len(gdf) >= 1
    assert set(gdf["product"]) <= {"L30", "S30"}
    assert gdf.crs.to_epsg() == 4326


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_hls_load_cmr_lpcloud():
    ds = hls.load(
        RAINIER,
        "2023-08-01/2023-08-06",
        bands=["green", "swir16", "Fmask"],
        resolution=120,
        mask="fmask-default",
    )
    assert ds["green"].dims == ("time", "y", "x")
    assert ds["green"].chunks is not None
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    assert ds["Fmask"].dtype == np.uint8
    assert set(np.unique(ds["product"].values)) <= {"L30", "S30"}
    assert (
        ds.attrs["product_id"] == "hls" and ds.attrs["source_id"] == "lpcloud-cmr-stac"
    )
    green = ds["green"].isel(time=0).compute()
    assert float(np.nanmax(green)) <= 2.0
    scene = ds.isel(time=0)
    ndsi = (
        (scene["green"] - scene["swir16"]) / (scene["green"] + scene["swir16"])
    ).compute()
    # Slightly negative surface reflectance over dark targets makes the
    # denominator vanish on a handful of pixels, so a raw NDSI can leave
    # [-1, 1] there; those must stay a handful rather than a hole in the scene.
    outside = float(((ndsi < -1) | (ndsi > 1)).mean())
    assert 0.0 <= outside < 0.01


@pytest.mark.live
def test_live_hls_planetary_computer_mirror_needs_no_credentials():
    ds = hls.load(
        RAINIER,
        "2023-08-01/2023-08-06",
        bands=["green", "Fmask"],
        products="S30",
        source="planetary-computer",
        resolution=120,
    )
    assert ds["green"].dims == ("time", "y", "x") and ds.sizes["time"] >= 1
    assert ds.attrs["source_id"] == "planetary-computer"
    assert float(np.nanmax(ds["green"].isel(time=0).compute())) <= 2.0
