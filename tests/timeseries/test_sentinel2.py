"""easysnowdata.optical.sentinel2 — searches replayed from cassettes, the
processing chain on synthetic items, and the live smoke tests.

Refresh the cassettes with::

    pixi run -e dev pytest tests/timeseries/test_sentinel2.py -m recorded --record-mode=rewrite
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.optical import sentinel2

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("sentinel-2-l2a")
    assert product is sentinel2.PRODUCT
    assert product.loader == "easysnowdata.optical.sentinel2.load"
    assert product.resolve_loader() is sentinel2.load
    assert [s.id for s in product.sources] == ["planetary-computer", "earth-search"]
    assert product.default_source.requires == ()
    assert [
        p.label for s in product.sources for p in s.health if p.kind == "health"
    ] == [
        "Sentinel-2 L2A (Planetary Computer)",
        "Sentinel-2 L2A (Earth Search)",
    ]
    scl = product.variables[-1]
    assert scl.name == "scl" and len(scl.flag_values) == 12
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_collection_validation():
    src, collection = sentinel2._catalog_and_collection(None, None)
    assert (src.id, collection) == ("planetary-computer", "sentinel-2-l2a")
    src, collection = sentinel2._catalog_and_collection(
        "earth-search", "sentinel-2-c1-l2a"
    )
    assert (src.id, collection) == ("earth-search", "sentinel-2-c1-l2a")
    with pytest.raises(ValueError, match="Collection-1 products are on Earth Search"):
        sentinel2._catalog_and_collection("planetary-computer", "sentinel-2-c1-l2a")
    with pytest.raises(ValueError, match="no source"):
        sentinel2._catalog_and_collection("aws-open-data", None)
    # the stac_cfg fallback is Planetary Computer only
    pc = sentinel2.PRODUCT.source("planetary-computer")
    assert "sentinel-2-l2a" in sentinel2._stac_cfg(pc, "sentinel-2-l2a")
    assert sentinel2._stac_cfg(sentinel2.PRODUCT.source("earth-search"), "x") is None


# ── searches, replayed ────────────────────────────────────────────────────────


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_planetary_computer():
    gdf = sentinel2.search(
        RAINIER, "2023-08-01/2023-08-12", cloud_cover=40, max_items=5
    )
    assert isinstance(gdf, gpd.GeoDataFrame) and 1 <= len(gdf) <= 5
    assert gdf.crs.to_epsg() == 4326
    assert (gdf["collection"] == "sentinel-2-l2a").all()
    assert (gdf["eo:cloud_cover"] < 40).all()
    assert gdf["datetime"].dt.year.eq(2023).all()
    assert "stac_item" in gdf.columns
    assert gdf.attrs == {"source": "planetary-computer", "collection": "sentinel-2-l2a"}
    # Planetary Computer items carry no raster:bands, hence the stac_cfg fallback
    assert sentinel2._offset_days(list(gdf["stac_item"])) is None


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_earth_search_collection_1():
    gdf = sentinel2.search(
        RAINIER,
        "2023-08-01/2023-08-12",
        source="earth-search",
        collection="sentinel-2-c1-l2a",
        max_items=5,
    )
    assert 1 <= len(gdf) <= 5
    assert (gdf["collection"] == "sentinel-2-c1-l2a").all()
    item = gdf["stac_item"].iloc[0]
    scale, offset = sentinel2._asset_scale_offset(item, "red")
    assert scale == pytest.approx(1e-4) and offset == pytest.approx(-0.1)
    # Collection-1 keeps the raw ESA values, so its post-2022 days need the fix
    days = sentinel2._offset_days(list(gdf["stac_item"]))
    assert days and all(isinstance(d, pd.Timestamp) for d in days)


# ── the processing chain, on synthetic items ─────────────────────────────────


def _item(when: str, *, offset: float | None, boa_applied: bool | None = None):
    properties: dict[str, object] = {"datetime": when}
    if boa_applied is not None:
        properties["earthsearch:boa_offset_applied"] = boa_applied
    raster = [{"nodata": 0, "data_type": "uint16", "scale": 1e-4}]
    if offset is not None:
        raster[0]["offset"] = offset
    return {
        "id": f"item-{when}",
        "properties": properties,
        "assets": {"red": {"href": "s3://x/red.tif", "raster:bands": raster}},
    }


def _dataset(values: dict[str, list[int]], times: list[str]) -> xr.Dataset:
    time = pd.to_datetime(times).values
    data = {}
    for name, per_time in values.items():
        dtype = "uint8" if name == "scl" else "uint16"
        block = np.array(
            [np.full((2, 2), v, dtype=dtype) for v in per_time], dtype=dtype
        )
        data[name] = (("time", "y", "x"), block)
    return xr.Dataset(
        data,
        coords={"time": time, "y": [5180400.0, 5180390.0], "x": [580000.0, 580010.0]},
    ).rio.write_crs("EPSG:32610")


def test_harmonization_from_metadata_and_from_the_date_rule():
    times = ["2021-06-01", "2023-08-10"]
    ds = _dataset({"red": [2000, 3000]}, times)

    # Planetary Computer: no raster:bands anywhere → the acquisition-date rule
    pc_items = [_item(t, offset=None) for t in times]
    out = sentinel2._harmonize(ds, pc_items, ["red"])
    assert list(out["red"].isel(y=0, x=0).values) == [2000, 2000]

    # Earth Search Collection-1: the post-baseline-4 item carries offset -0.1
    # (raw ESA values), the older one carries 0 → only the 2023 day is fixed
    c1_items = [_item("2021-06-01", offset=0.0), _item("2023-08-10", offset=-0.1)]
    out = sentinel2._harmonize(ds, c1_items, ["red"])
    assert list(out["red"].isel(y=0, x=0).values) == [2000, 2000]

    # Earth Search sentinel-2-l2a: boa_offset_applied → pixels already correct
    l2a_items = [_item(t, offset=-0.1, boa_applied=True) for t in times]
    out = sentinel2._harmonize(ds, l2a_items, ["red"])
    assert list(out["red"].isel(y=0, x=0).values) == [2000, 3000]

    # nothing to do without a time dimension or a reflectance band
    assert sentinel2._harmonize(ds, pc_items, ["scl"]) is ds


def test_scaling_uses_the_metadata_scale_then_falls_back():
    ds = _dataset({"red": [2000], "scl": [4]}, ["2021-06-01"])
    scaled = sentinel2._scale(ds, [_item("2021-06-01", offset=None)], ["red", "scl"])
    assert float(scaled["red"].isel(time=0, y=0, x=0)) == pytest.approx(0.2)
    assert scaled["red"].attrs["units"] == "1"
    assert int(scaled["scl"].isel(time=0, y=0, x=0)) == 4  # untouched
    no_metadata = sentinel2._scale(
        ds, [{"id": "x", "properties": {}, "assets": {}}], ["red"]
    )
    assert float(no_metadata["red"].isel(time=0, y=0, x=0)) == pytest.approx(0.2)


def test_scl_masking_rules():
    ds = _dataset({"red": [2000], "scl": [9]}, ["2021-06-01"])  # 9 = cloud high
    masked = sentinel2._apply_mask(ds, True, ["red", "scl"])
    assert np.isnan(masked["red"].isel(time=0, y=0, x=0))
    assert int(masked["scl"].isel(time=0, y=0, x=0)) == 9  # the mask band stays
    assert "cloud_high" in masked.attrs["masked_scl_classes"]
    kept = sentinel2._apply_mask(ds, ["snow_ice"], ["red", "scl"])
    assert int(kept["red"].isel(time=0, y=0, x=0)) == 2000
    assert sentinel2._apply_mask(ds, None, ["red"]) is ds
    with pytest.raises(ValueError, match="Unknown mask"):
        sentinel2._apply_mask(ds, "everything", ["red", "scl"])
    with pytest.raises(ValueError, match="needs the 'scl' band"):
        sentinel2._apply_mask(ds[["red"]], True, ["red"])


@pytest.fixture
def fake_stac(monkeypatch):
    """Capture the odc-stac load parameters without touching the network."""
    calls: dict[str, object] = {}
    items = [_item("2023-08-10", offset=None)]

    def search(catalog_id, collection, aoi=None, time=None, **kwargs):
        calls["search"] = (catalog_id, collection, aoi, time, kwargs)
        return items

    def items_to_geodataframe(item_list):
        return gpd.GeoDataFrame(
            {"id": [i["id"] for i in item_list], "stac_item": list(item_list)},
            geometry=[None] * len(item_list),
            crs="EPSG:4326",
        )

    def load(item_arg, aoi=None, **kwargs):
        calls["load"] = (item_arg, aoi, kwargs)
        return _dataset({"red": [3000], "scl": [4]}, ["2023-08-10"])

    monkeypatch.setattr(sentinel2.providers.stac, "search", search)
    monkeypatch.setattr(
        sentinel2.providers.stac, "items_to_geodataframe", items_to_geodataframe
    )
    monkeypatch.setattr(sentinel2.providers.stac, "load", load)
    return calls


@pytest.mark.recorded
def test_load_builds_the_right_call_and_output_contract(fake_stac):
    ds = sentinel2.load(RAINIER, "2023-08", bands=["red", "scl"], mask=True)
    catalog_id, collection, aoi, time, kwargs = fake_stac["search"]
    assert (catalog_id, collection) == ("planetary-computer", "sentinel-2-l2a")
    assert kwargs["query"] is None
    _, load_aoi, load_kwargs = fake_stac["load"]
    assert load_kwargs["bands"] == ["red", "scl"]
    assert load_kwargs["crs"] == "utm" and load_kwargs["groupby"] == "solar_day"
    assert "sentinel-2-l2a" in load_kwargs["stac_cfg"]  # PC needs the fallback
    assert load_kwargs["catalog"] == "planetary-computer"
    assert load_aoi.bounds == pytest.approx(RAINIER)

    assert ds["red"].dims == ("time", "y", "x")
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    # 3000 DN → harmonized (−1000) → scaled (×1e-4) → 0.2 reflectance
    assert float(ds["red"].isel(time=0, y=0, x=0)) == pytest.approx(0.2)
    assert ds["scl"].dtype == np.uint8 and ds["scl"].rio.nodata == 0
    assert ds["scl"].attrs["flag_meanings"].split()[-1] == "Snow_or_ice"
    assert ds.attrs["product_id"] == "sentinel-2-l2a"
    assert ds.attrs["source_id"] == "planetary-computer"
    assert ds.attrs["harmonized_to_old_baseline"] == "True"
    assert ds.attrs["scaled_to_reflectance"] == "True"
    assert all(not callable(v) for v in ds.attrs.values())


@pytest.mark.recorded
def test_load_flags_and_earth_search_route(fake_stac):
    raw = sentinel2.load(
        RAINIER,
        "2023-08",
        bands=["red", "scl"],
        source="earth-search",
        collection="sentinel-2-c1-l2a",
        harmonize=False,
        scale=False,
        mask_nodata=False,
        cloud_cover=20,
    )
    catalog_id, collection, _, _, kwargs = fake_stac["search"]
    assert (catalog_id, collection) == ("earth-search", "sentinel-2-c1-l2a")
    assert kwargs["query"] == {"eo:cloud_cover": {"lt": 20.0}}
    assert fake_stac["load"][2]["stac_cfg"] is None  # no fallback on Earth Search
    assert (
        raw["red"].dtype == np.uint16 and int(raw["red"].isel(time=0, y=0, x=0)) == 3000
    )
    assert raw["red"].rio.nodata == 0
    assert raw.attrs["harmonized_to_old_baseline"] == "False"


@pytest.mark.recorded
def test_load_with_prefetched_items_skips_the_search(fake_stac, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("search must not run when items= is given")

    gdf = gpd.GeoDataFrame(
        {"id": ["x"], "stac_item": [_item("2023-08-10", offset=None)]},
        geometry=[None],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(sentinel2.providers.stac, "search", boom)
    ds = sentinel2.load(RAINIER, items=gdf, bands=["red"])
    assert float(ds["red"].isel(time=0, y=0, x=0)) == pytest.approx(0.2)


@pytest.mark.recorded
def test_load_without_items_raises(monkeypatch):
    monkeypatch.setattr(sentinel2.providers.stac, "search", lambda *a, **k: [])
    monkeypatch.setattr(
        sentinel2.providers.stac,
        "items_to_geodataframe",
        lambda items: gpd.GeoDataFrame(
            {"id": [], "stac_item": []}, geometry=[], crs="EPSG:4326"
        ),
    )
    with pytest.raises(ValueError, match="No Sentinel-2 items"):
        sentinel2.load(RAINIER, "1999-01")


# ── the deprecation shim ──────────────────────────────────────────────────────


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_sentinel2_planetary_computer():
    ds = sentinel2.load(
        RAINIER,
        "2023-08-01/2023-08-12",
        bands=["green", "swir16", "scl"],
        resolution=60,
        mask="scl-default",
        cloud_cover=40,
    )
    assert ds["green"].dims == ("time", "y", "x")
    assert ds["green"].dtype == np.float32 or ds["green"].dtype == np.float64
    assert ds["green"].chunks is not None
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    assert ds["scl"].dtype == np.uint8 and ds["scl"].rio.nodata == 0
    assert ds.attrs["product_id"] == "sentinel-2-l2a" and ds.attrs["license"]
    green = ds["green"].isel(time=0).compute()
    assert 0 <= float(np.nanmin(green)) and float(np.nanmax(green)) <= 2.0
    scene = ds.isel(time=0)
    ndsi = (
        (scene["green"] - scene["swir16"]) / (scene["green"] + scene["swir16"])
    ).compute()
    assert float(np.nanmedian(ndsi)) == float(
        np.nanmedian(ndsi)
    )  # finite where both bands are


@pytest.mark.live
def test_live_sentinel2_catalogs_agree_after_harmonization():
    """The same tile and day from both catalogs must agree once harmonized.

    Earth Search's ``sentinel-2-l2a`` pixels already carry the baseline
    correction while Collection-1 and Planetary Computer do not; this is the
    regression test for that (they differ by ~1000 DN when it is wrong).
    """
    window = (-121.80, 46.82, -121.72, 46.88)
    common = dict(bands=["red"], resolution=60, crs="EPSG:32610")
    pc = sentinel2.load(window, "2023-08-10/2023-08-11", **common)
    es = sentinel2.load(
        window, "2023-08-10/2023-08-11", source="earth-search", **common
    )
    c1 = sentinel2.load(
        window,
        "2023-08-10/2023-08-11",
        source="earth-search",
        collection="sentinel-2-c1-l2a",
        **common,
    )
    medians = [
        float(np.nanmedian(d["red"].isel(time=0).compute().values))
        for d in (pc, es, c1)
    ]
    assert max(medians) - min(medians) < 0.02  # reflectance units


@pytest.mark.live
def test_live_sentinel2_search_returns_the_contract_frame():
    gdf = sentinel2.search(RAINIER, "2023-08-01/2023-08-12", cloud_cover=40)
    assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326
    assert (gdf["eo:cloud_cover"] < 40).all()
    # the frame can be filtered and handed back to load()
    subset = gdf.iloc[:1]
    ds = sentinel2.load(RAINIER, items=subset, bands=["red"], resolution=60)
    assert ds.sizes["time"] == 1
