"""OpenTopography's raster catalog: a static STAC catalog of public COGs.

OpenTopography publishes its whole raster library — the global DEMs and the
community lidar surveys — as a static STAC 1.1 catalog
(:data:`CATALOG_URL`), with the tiles as anonymous Cloud Optimized GeoTIFFs
on SDSC's S3-compatible store (:data:`BUCKET_URL`). No account, no signing.

It is **not a STAC API**: there is no ``/search`` endpoint, so
``pystac-client`` cannot query it. Each dataset is one collection holding
one item per product level (``COP30_hh``, ``SRTM_GL1_srtm``,
``NASADEM_be``), and every tile of that dataset is an *asset* of the one
item, with its own ``bbox``. The Copernicus GLO-30 item is 12 MB of JSON
with about 26 000 assets. :func:`search` therefore downloads the item once
into the cache (re-fetched after :data:`MAX_AGE`), picks the assets whose
``bbox`` meets the AOI, and turns each into a :class:`pystac.Item` that
``odc-stac`` can load. The assets carry no ``proj:`` metadata, so the grid
of each selected tile is read from its COG header (one small range request
per tile, in parallel).

Two more quirks worth knowing: an item's ``datetime`` is when OpenTopography
processed the dataset, not when it was acquired, so the items built here
take their dates from the collection's temporal extent instead; and each
item also lists a ``<item>.vrt`` mosaic of all its tiles (11 MB for
Copernicus), which this module does not use.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import shapely

from easysnowdata._gdal import gdal_env
from easysnowdata.aoi import parse_aoi
from easysnowdata.providers import raster_http

__all__ = [
    "CATALOG_URL",
    "BUCKET_URL",
    "MAX_AGE",
    "collection_id",
    "collection_url",
    "item_url",
    "collection",
    "item",
    "tiles",
    "search",
]

_logger = logging.getLogger(__name__)

STAC_ROOT = "https://portal.opentopography.org/stac"
#: The root of the static catalog.
CATALOG_URL = f"{STAC_ROOT}/raster_catalog.json"
#: Where the COGs (and the per-item VRT mosaics) live.
BUCKET_URL = "https://opentopography.s3.sdsc.edu/raster"
#: Seconds a cached catalog JSON is trusted before it is fetched again: the
#: items are regenerated when OpenTopography adds or replaces tiles.
MAX_AGE = 30 * 24 * 3600
#: Header reads in flight at once when building items.
MAX_WORKERS = 8


def collection_id(item_id: str) -> str:
    """The collection an item belongs to: ``"COP30_hh"`` → ``"COP30"``.

    Item ids are the collection's short name plus a product-level suffix
    (``_hh`` surface, ``_be`` bare earth, ``_srtm``, ``_global``).
    """
    return item_id.rsplit("_", 1)[0]


def collection_url(name: str) -> str:
    """URL of the collection JSON for a short name such as ``"SRTM_GL1"``."""
    return f"{STAC_ROOT}/{name}_collection.json"


def item_url(item_id: str) -> str:
    """URL of the item JSON for an id such as ``"SRTM_GL1_srtm"``."""
    return f"{STAC_ROOT}/items/{item_id}.json"


def _fetch_json(url: str) -> dict[str, Any]:
    """Download *url* into the cache (at most every :data:`MAX_AGE`) and parse it."""
    path = raster_http.fetch(
        url,
        fname=url.rsplit("/", 1)[-1],
        subdir="opentopography",
        progressbar=False,
        max_age=MAX_AGE,
    )
    return _parse(str(path), Path(path).stat().st_mtime)


@lru_cache(maxsize=8)
def _parse(path: str, mtime: float) -> dict[str, Any]:  # noqa: ARG001 — cache key
    return json.loads(Path(path).read_text())


def collection(name: str) -> dict[str, Any]:
    """The collection JSON for a short name (``"COP30"``), from the cache."""
    return _fetch_json(collection_url(name))


def item(item_id: str) -> dict[str, Any]:
    """The item JSON for an id (``"COP30_hh"``), from the cache."""
    return _fetch_json(item_url(item_id))


def tiles(item_id: str, aoi: Any = None) -> list[dict[str, Any]]:
    """The data assets of *item_id* whose ``bbox`` meets *aoi*.

    Each is a dict with ``id`` (the file name without ``.tif``), ``href``,
    ``bbox`` and ``type``. ``aoi=None`` or a global AOI returns every tile.
    """
    geometry = None
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            geometry = parsed.geometry
    found = []
    for key, asset in item(item_id).get("assets", {}).items():
        bbox = asset.get("bbox")
        if bbox is None or "data" not in asset.get("roles", ()):
            continue  # the .vrt mosaic and any metadata assets
        if geometry is not None and not shapely.box(*bbox).intersects(geometry):
            continue
        found.append(
            {
                "id": key.removesuffix(".tif"),
                "href": asset["href"],
                "bbox": list(bbox),
                "type": asset.get("type"),
            }
        )
    return found


def _interval(name: str) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Acquisition start and end from the collection's temporal extent."""
    try:
        start, end = collection(name)["extent"]["temporal"]["interval"][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None

    def parse(value: str | None) -> dt.datetime | None:
        if not value:
            return None
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))

    return parse(start), parse(end)


def _grid(href: str) -> dict[str, Any]:
    """``proj:`` fields for one COG, read from its header."""
    import rasterio  # noqa: PLC0415

    # gdal_env is a rasterio.Env, which is thread-local: enter it per worker.
    with gdal_env(), rasterio.open(href) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        return {
            "proj:code": f"EPSG:{epsg}" if epsg else None,
            "proj:wkt2": None if epsg else (src.crs.to_wkt() if src.crs else None),
            "proj:transform": list(src.transform)[:6],
            "proj:shape": [src.height, src.width],
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }


def search(item_id: str, aoi: Any = None) -> Any:
    """The tiles of *item_id* covering *aoi* as a :class:`pystac.ItemCollection`.

    One item per tile, in collection :func:`collection_id` of *item_id*, with
    a single ``data`` asset, the projection extension filled from the COG
    header, and ``start_datetime``/``end_datetime`` from the collection's
    temporal extent (``datetime`` is the start, so every tile of a dataset
    groups into one time step).
    """
    import pystac  # noqa: PLC0415

    name = collection_id(item_id)
    selected = tiles(item_id, aoi)
    if not selected:
        return pystac.ItemCollection([])
    start, end = _interval(name)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        grids = list(pool.map(_grid, [t["href"] for t in selected]))
    items = []
    for tile, grid in zip(selected, grids):
        properties: dict[str, Any] = {
            k: v for k, v in grid.items() if k.startswith("proj:") and v is not None
        }
        if start is not None:
            properties["start_datetime"] = start.isoformat()
        if end is not None:
            properties["end_datetime"] = end.isoformat()
        stac_item = pystac.Item(
            id=tile["id"],
            geometry=shapely.geometry.mapping(shapely.box(*tile["bbox"])),
            bbox=tile["bbox"],
            datetime=start or dt.datetime(1970, 1, 1, tzinfo=dt.UTC),
            properties=properties,
            collection=name,
            stac_extensions=[
                "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
                "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            ],
        )
        raster_info = {"data_type": grid["dtype"]}
        if grid["nodata"] is not None:
            raster_info["nodata"] = grid["nodata"]
        stac_item.add_asset(
            "data",
            pystac.Asset(
                tile["href"],
                media_type=tile["type"],
                roles=["data"],
                extra_fields={"raster:bands": [raster_info]},
            ),
        )
        items.append(stac_item)
    _logger.debug("OpenTopography %s: %d tiles", item_id, len(items))
    return pystac.ItemCollection(items)
