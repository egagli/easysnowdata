"""STAC search and load: pystac-client for search, odc-stac for lazy loading."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from easysnowdata import auth, temporal
from easysnowdata._gdal import gdal_env
from easysnowdata.aoi import AOI, parse_aoi
from easysnowdata.auth.planetary_computer import STAC_URL as PLANETARY_COMPUTER_URL

__all__ = [
    "CATALOGS",
    "open_catalog",
    "search",
    "items_to_geodataframe",
    "geodataframe_to_items",
    "load",
    "odc_load",
    "read_env",
]

_logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac"
CMR_CLOUDSTAC_URL = "https://cmr.earthdata.nasa.gov/cloudstac"

#: Known catalogs: url, whether hrefs need signing, auth providers reads need,
#: and GDAL options for their object store.
CATALOGS: dict[str, dict[str, Any]] = {
    "planetary-computer": {
        "url": PLANETARY_COMPUTER_URL,
        "sign": True,
        "requires": ("planetary_computer",),
        "gdal": {},
    },
    "earth-search": {
        "url": EARTH_SEARCH_URL,
        "sign": False,
        "requires": (),
        "gdal": {"AWS_NO_SIGN_REQUEST": "YES", "AWS_REGION": "us-west-2"},
    },
    "cmr-lpcloud": {
        "url": f"{CMR_STAC_URL}/LPCLOUD",
        "sign": False,
        "requires": ("earthdata",),
        "gdal": {},
    },
    "cmr-asf": {
        "url": f"{CMR_CLOUDSTAC_URL}/ASF",
        "sign": False,
        "requires": ("earthdata",),
        # datapool.asf.alaska.edu answers HEAD with 403 (API Gateway
        # "MissingAuthenticationTokenException"), and GDAL's first request on a
        # /vsicurl/ file is a HEAD; a ranged GET follows the EDL redirects fine.
        "gdal": {"CPL_VSIL_CURL_USE_HEAD": "NO"},
    },
    "cmr-nsidc": {
        "url": f"{CMR_CLOUDSTAC_URL}/NSIDC_CPRD",
        "sign": False,
        "requires": ("earthdata",),
        "gdal": {},
    },
}


def _catalog_spec(catalog: str) -> dict[str, Any]:
    if catalog in CATALOGS:
        return CATALOGS[catalog]
    for spec in CATALOGS.values():
        if spec["url"].rstrip("/") == catalog.rstrip("/"):
            return spec
    return {"url": catalog, "sign": False, "requires": (), "gdal": {}}


def open_catalog(
    catalog: str = "planetary-computer", *, sign: bool | None = None, **kwargs: Any
) -> Any:
    """Open a STAC API by name (``"planetary-computer"``, ``"earth-search"``,
    ``"cmr-lpcloud"``, ``"cmr-asf"``, ``"cmr-nsidc"``) or URL.

    Planetary Computer hrefs are signed in place unless ``sign=False``.
    """
    import pystac_client  # noqa: PLC0415

    spec = _catalog_spec(catalog)
    do_sign = spec["sign"] if sign is None else sign
    modifier = auth.get("planetary_computer").sign if do_sign else None
    return pystac_client.Client.open(spec["url"], modifier=modifier, **kwargs)


def search(
    catalog: str,
    collections: str | Sequence[str],
    aoi: Any = None,
    time: Any = None,
    *,
    query: dict[str, Any] | None = None,
    filter: dict[str, Any] | str | None = None,  # noqa: A002 — STAC's own name
    max_items: int | None = None,
    sign: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Search a STAC API and return a :class:`pystac.ItemCollection`.

    ``aoi`` (any form :func:`~easysnowdata.aoi.parse_aoi` accepts) becomes an
    ``intersects`` geometry — a MultiPolygon across the antimeridian — and
    ``time`` a ``start/end`` interval (:func:`~easysnowdata.temporal.parse_time`).
    ``None`` for either leaves that constraint out.
    """
    client = open_catalog(catalog, sign=sign)
    params: dict[str, Any] = {
        "collections": [collections]
        if isinstance(collections, str)
        else list(collections),
        "max_items": max_items,
    }
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            params["intersects"] = parsed.stac_intersects
    if time is not None:
        params["datetime"] = temporal.to_stac_datetime(time)
    if query:
        params["query"] = query
    if filter:
        params["filter"] = filter
    params.update(kwargs)
    _logger.debug("STAC search %s: %s", catalog, params)
    return client.search(**params).item_collection()


def items_to_geodataframe(items: Any) -> gpd.GeoDataFrame:
    """Items → GeoDataFrame (EPSG:4326) with ``id``, ``collection``, ``datetime``,
    the item properties as columns, and the full item dict in ``stac_item`` so
    :func:`load` can take a filtered frame back."""
    rows = []
    for item in _iter_items(items):
        d = (
            item.to_dict(transform_hrefs=False)
            if hasattr(item, "to_dict")
            else dict(item)
        )
        props = dict(d.get("properties", {}))
        row = {
            "id": d["id"],
            "collection": d.get("collection"),
            **props,
            "stac_item": d,
        }
        row["geometry"] = (
            shapely.geometry.shape(d["geometry"]) if d.get("geometry") else None
        )
        rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(
            {"id": [], "collection": [], "datetime": [], "stac_item": []},
            geometry=[],
            crs="EPSG:4326",
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if "datetime" in gdf.columns:
        gdf["datetime"] = pd.to_datetime(gdf["datetime"], utc=True, errors="coerce")
    return gdf.set_index("id", drop=False).rename_axis(None)


def geodataframe_to_items(gdf: gpd.GeoDataFrame) -> list[Any]:
    """Rebuild :class:`pystac.Item` objects from a frame made by :func:`items_to_geodataframe`."""
    import pystac  # noqa: PLC0415

    if "stac_item" not in gdf.columns:
        raise ValueError(
            "GeoDataFrame has no 'stac_item' column; use items_to_geodataframe() output."
        )
    return [pystac.Item.from_dict(d) for d in gdf["stac_item"]]


def _iter_items(items: Any) -> Iterable[Any]:
    import pystac  # noqa: PLC0415

    if isinstance(items, gpd.GeoDataFrame):
        return geodataframe_to_items(items)
    if isinstance(items, pystac.ItemCollection):
        return list(items)
    if hasattr(items, "item_collection"):  # pystac_client.ItemSearch
        return list(items.item_collection())
    if isinstance(items, pystac.Item):
        return [items]
    return list(items)


@contextlib.contextmanager
def read_env(
    catalog: str | None = None, requires: Iterable[str] = (), **gdal: Any
) -> Iterator[dict[str, Any]]:
    """GDAL + credential environment for reading assets found in *catalog*."""
    spec = _catalog_spec(catalog) if catalog else {"requires": (), "gdal": {}}
    names = tuple(dict.fromkeys([*spec["requires"], *requires]))
    with auth.env(*names) as auth_opts, gdal_env(**{**spec["gdal"], **gdal}) as opts:
        yield {**auth_opts, **opts}


def odc_load(
    items: Any,
    *,
    catalog: str | None = None,
    requires: Iterable[str] = (),
    gdal: dict[str, Any] | None = None,
    **load_params: Any,
) -> Any:
    """``odc.stac.load`` with the read environment applied (a low-level helper
    for callers that build their own ``load_params``)."""
    import odc.stac  # noqa: PLC0415

    with read_env(catalog, requires, **(gdal or {})):
        return odc.stac.load(_iter_items(items), **load_params)


def load(
    items: Any,
    aoi: Any = None,
    *,
    bands: str | Sequence[str] | None = None,
    resolution: float | None = None,
    crs: Any = None,
    chunks: dict[str, Any] | None = None,
    groupby: str | None = "time",
    stac_cfg: dict[str, Any] | None = None,
    catalog: str | None = None,
    requires: Iterable[str] = (),
    **kwargs: Any,
) -> Any:
    """Lazily load STAC items into an ``xarray.Dataset`` with odc-stac.

    Parameters
    ----------
    items
        A ``pystac.ItemCollection``, list of items, ``ItemSearch`` or a
        GeoDataFrame from :func:`items_to_geodataframe` (filtered as you like).
    aoi
        Spatial subset. With ``clip=True`` (default) the grid is cut to the
        footprint (``geopolygon``); with ``clip=False`` the covering tiles are
        returned whole.
    bands, resolution, crs, chunks, groupby, stac_cfg, **kwargs
        Passed to ``odc.stac.load``. ``crs="utm"`` picks the AOI's UTM zone;
        ``chunks`` defaults to ``{}`` (Dask, native chunking).
    catalog, requires
        Which catalog the items came from (sets signing/GDAL options) and any
        extra auth providers whose ``env()`` the reads need.
    """
    params: dict[str, Any] = {
        "bands": bands,
        "chunks": {} if chunks is None else chunks,
        "groupby": groupby,
        "stac_cfg": stac_cfg,
    }
    if aoi is not None:
        parsed = aoi if isinstance(aoi, AOI) else parse_aoi(aoi)
        if crs is not None and isinstance(crs, str) and crs.lower() == "utm":
            crs = str(parsed.utm_crs)
        if not parsed.is_global:
            if parsed.clip:
                from odc.geo.geom import Geometry  # noqa: PLC0415

                params["geopolygon"] = Geometry(parsed.geometry, crs="EPSG:4326")
            else:
                params["intersects"] = parsed.stac_intersects
    if crs is not None:
        params["crs"] = crs
    if resolution is not None:
        params["resolution"] = resolution
    params.update(kwargs)
    params = {k: v for k, v in params.items() if v is not None or k == "groupby"}
    return odc_load(items, catalog=catalog, requires=requires, **params)
