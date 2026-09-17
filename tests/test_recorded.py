"""Recorded tier: STAC searches replayed from pytest-recording cassettes.

Record or refresh with network access::

    pixi run -e dev pytest tests/test_recorded.py --record-mode=rewrite

The offline tier replays them with ``--record-mode=none`` and sockets
disabled, so a missing or stale cassette fails instead of reaching the network.
Searches are unsigned (``sign=False``) so no SAS tokens land in git.
"""

from __future__ import annotations

import geopandas as gpd
import pystac
import pytest

from easysnowdata import catalog
from easysnowdata.catalog import health
from easysnowdata.providers import stac

RAINIER = (-121.94, 46.72, -121.54, 46.99)

pytestmark = [pytest.mark.recorded, pytest.mark.vcr]


def test_planetary_computer_dem_search():
    items = stac.search("planetary-computer", "cop-dem-glo-30", RAINIER, sign=False)
    assert isinstance(items, pystac.ItemCollection) and len(items) >= 1
    gdf = stac.items_to_geodataframe(items)
    assert isinstance(gdf, gpd.GeoDataFrame) and gdf.crs.to_epsg() == 4326
    assert (gdf["collection"] == "cop-dem-glo-30").all()
    assert gdf.intersects(
        gpd.GeoSeries.from_xy([-121.76], [46.85], crs="EPSG:4326").iloc[0]
    ).any()
    back = stac.geodataframe_to_items(gdf)
    assert {i.id for i in back} == set(gdf["id"])
    assert "data" in back[0].assets


def test_earth_search_sentinel2_search_with_time():
    items = stac.search(
        "earth-search",
        "sentinel-2-l2a",
        RAINIER,
        "2024-07-01/2024-07-10",
        query={"eo:cloud_cover": {"lt": 60}},
        max_items=5,
    )
    gdf = stac.items_to_geodataframe(items)
    assert 1 <= len(gdf) <= 5
    assert (
        gdf["datetime"].dt.month.eq(7).all() and gdf["datetime"].dt.year.eq(2024).all()
    )
    assert (gdf["eo:cloud_cover"] < 60).all()
    assert gdf["stac_item"].iloc[0]["assets"]["red"]["href"].startswith("s3://") or gdf[
        "stac_item"
    ].iloc[0]["assets"]["red"]["href"].startswith("https://")


def test_cmr_lpcloud_hls_search_is_open():
    """CMR-STAC search needs no Earthdata login (only the COG reads do)."""
    items = stac.search(
        "cmr-lpcloud",
        "HLSL30_2.0",
        RAINIER,
        "2024-07-01T00:00:00Z/2024-07-31T23:59:59Z",
        max_items=2,
    )
    assert len(items) >= 1
    assert all(item.collection_id == "HLSL30_2.0" for item in items)


def test_health_stac_probe_replays():
    """The catalog's Copernicus DEM probe, replayed unsigned."""
    health.stac_search(stac.PLANETARY_COMPUTER_URL, "cop-dem-glo-30")
    source = catalog.get("copernicus-dem").default_source
    assert source.health[0].label == "Copernicus DEM (Planetary Computer)"


def test_health_first_byte_probe_replays():
    health.http_first_byte(
        "https://zenodo.org/records/2626737/files/MODIS_mtnsnow_classes.zip"
    )
