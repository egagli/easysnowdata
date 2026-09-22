"""Planet Labs via the ``planet`` SDK 3.x: Data API search, Orders API delivery.

Thin and generic, like the other providers: it knows how to talk to Planet,
not what PlanetScope is. The snow knowledge lives in
:mod:`easysnowdata.optical.planetscope`.

Everything goes through one ``planet.Planet`` client built by the ``planet``
auth provider (:mod:`easysnowdata.auth.planet`), which delegates credential
detection and login to the SDK's own stack. Nothing here configures GDAL: the
APIs hand out signed download URLs.

**Quota.** A Data API *search* is free. Activating a Data API asset charges
the whole scene against the account's quota, and an *order* charges the
clipped area, which is why :func:`create_order` refuses to run without an AOI
and why ordering is never part of a test.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from easysnowdata import auth, config, temporal
from easysnowdata.aoi import parse_aoi

__all__ = [
    "ensure",
    "client",
    "sdk",
    "build_search_filter",
    "search",
    "items_to_geodataframe",
    "product",
    "create_order",
    "find_order",
    "wait_order",
    "download_order",
    "asset_url",
]

_logger = logging.getLogger(__name__)

DATA_API_URL = "https://api.planet.com/data/v1/"
TILES_URL = (
    "https://tiles.planet.com/data/v1/{item_type}/{item_id}/{{z}}/{{x}}/{{y}}.png"
)


def sdk() -> Any:
    """Return the imported ``planet`` module (a required dependency)."""
    import planet  # noqa: PLC0415

    return planet


def ensure(**kwargs: Any) -> Any:
    """Return the process-wide ``planet.Planet`` client, logging in if needed."""
    return auth.get("planet").ensure(**kwargs)


def client(**kwargs: Any) -> Any:
    """Alias of :func:`ensure`, for symmetry with the other providers."""
    return ensure(**kwargs)


# ── search ────────────────────────────────────────────────────────────────────


def build_search_filter(
    aoi: Any = None,
    time: Any = None,
    *,
    cloud_cover: float | None = None,
    asset_types: Sequence[str] | None = None,
    permission: bool = True,
    extra: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build a Data API ``search_filter`` from the package's AOI/time inputs.

    ``permission=True`` keeps only items the account may actually download,
    which is what makes a search result safe to hand to :func:`create_order`.
    """
    planet = sdk()
    filters: list[dict[str, Any]] = []
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            filters.append(
                planet.data_filter.geometry_filter(
                    shapely.geometry.mapping(parsed.geometry)
                )
            )
    if time is not None:
        start, end = temporal.parse_time(time)
        filters.append(
            planet.data_filter.date_range_filter(
                "acquired",
                gte=None if start is None else start.to_pydatetime(),
                lte=end.to_pydatetime(),
            )
        )
    if cloud_cover is not None:
        filters.append(
            planet.data_filter.range_filter("cloud_cover", lte=float(cloud_cover) / 100)
        )
    if asset_types:
        filters.append(planet.data_filter.asset_filter(list(asset_types)))
    if permission:
        filters.append(planet.data_filter.permission_filter())
    filters.extend(extra)
    if not filters:
        return planet.data_filter.empty_filter()
    if len(filters) == 1:
        return filters[0]
    return planet.data_filter.and_filter(filters)


def search(
    item_types: Sequence[str],
    search_filter: dict[str, Any],
    *,
    limit: int = 100,
    sort: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run a Data API quick-search and return the items as dicts (free)."""
    pl = ensure()
    _logger.debug("Planet Data API search: item_types=%s limit=%s", item_types, limit)
    return list(
        pl.data.search(
            item_types=list(item_types),
            search_filter=search_filter,
            limit=limit,
            sort=sort,
            **kwargs,
        )
    )


def items_to_geodataframe(items: Iterable[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Data API items → GeoDataFrame with ``id``, ``item_type``, ``acquired`` and
    the item properties as columns (the same shape every ``search_*`` returns)."""
    rows = []
    for item in items:
        properties = dict(item.get("properties", {}))
        rows.append(
            {
                "id": item.get("id"),
                "item_type": properties.get("item_type")
                or item.get("_permissions", [""])[0].split(":")[0]
                or None,
                **properties,
                "assets": list(item.get("assets", []) or []),
                "planet_item": item,
                "geometry": shapely.geometry.shape(item["geometry"])
                if item.get("geometry")
                else None,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            {"id": [], "item_type": [], "acquired": [], "planet_item": []},
            geometry=[],
            crs="EPSG:4326",
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if "acquired" in gdf.columns:
        gdf["acquired"] = pd.to_datetime(gdf["acquired"], utc=True, errors="coerce")
    return gdf.set_index("id", drop=False).rename_axis(None)


# ── orders (quota-spending; never exercised by tests) ────────────────────────


def product(
    item_ids: Sequence[str],
    *,
    item_type: str,
    product_bundle: str,
    validate: bool = False,
) -> dict[str, Any]:
    """One ``products`` entry of an order request.

    The SDK's own ``order_request.product()`` validates the item type and
    bundle by fetching ``https://api.planet.com/compute/ops/bundles/spec``
    *while building the request* (SDK 3.6 ``specs._LazyBundlesLoader``), which
    makes request building fail offline and whenever that endpoint is down.
    The dict it returns is the three keys below, so this builds them directly
    and leaves the live check to ``validate=True``.
    """
    if validate:
        planet = sdk()
        return planet.order_request.product(
            item_ids=list(item_ids),
            product_bundle=product_bundle,
            item_type=item_type,
        )
    return {
        "item_ids": list(item_ids),
        "item_type": item_type,
        "product_bundle": product_bundle,
    }


def create_order(
    item_ids: Sequence[str],
    *,
    item_type: str,
    product_bundle: str,
    aoi: Any,
    name: str | None = None,
    harmonize: str | None = None,
    composite: bool = False,
    extra_tools: Sequence[dict[str, Any]] = (),
    validate_bundle: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create an Orders API order clipped to *aoi*, and return the order dict.

    **This spends the account's quota.** The AOI is required: without it
    Planet would deliver whole scenes, which is exactly the waste the Orders
    route exists to avoid (companion §B.11).

    ``validate_bundle=True`` asks the SDK to check the item type and bundle
    against Planet's live bundle spec; see :func:`product` for why that is
    not the default.
    """
    if aoi is None:
        raise ValueError(
            "An AOI is required to order Planet imagery: the clip tool is what keeps "
            "an order from charging for whole scenes."
        )
    parsed = parse_aoi(aoi)
    if parsed.is_global:
        raise ValueError("Refusing to order Planet imagery for the whole globe.")
    if not item_ids:
        raise ValueError("No item ids to order.")
    planet = sdk()
    tools = [planet.order_request.clip_tool(shapely.geometry.mapping(parsed.geometry))]
    if harmonize:
        tools.append(planet.order_request.harmonize_tool(harmonize))
    if composite:
        tools.append(planet.order_request.composite_tool())
    tools.extend(extra_tools)
    request = planet.order_request.build_request(
        name=name or f"easysnowdata-{temporal.now():%Y%m%dT%H%M%S}",
        products=[
            product(
                item_ids,
                item_type=item_type,
                product_bundle=product_bundle,
                validate=validate_bundle,
            )
        ],
        tools=tools,
        **kwargs,
    )
    pl = ensure()
    _logger.info(
        "Creating a Planet order for %d %s item(s) clipped to the AOI (spends quota).",
        len(item_ids),
        item_type,
    )
    return pl.orders.create_order(request)


def find_order(name: str, *, state: str | None = "success") -> dict[str, Any] | None:
    """Return the newest order called *name* (in *state*, if given), or ``None``.

    A free ``GET`` of the account's order list; it is what lets a repeated
    :func:`easysnowdata.optical.planetscope.order` call with the same *name*
    re-download an earlier delivery instead of spending quota again.
    """
    pl = ensure()
    matches = [
        o
        for o in pl.orders.list_orders(name=name, state=state)
        if o.get("name") == name
    ]
    if not matches:
        return None
    return max(matches, key=lambda o: o.get("created_on", ""))


def wait_order(
    order_id: str, *, delay: int = 15, max_attempts: int = 0, **kwargs: Any
) -> str:
    """Block until an order reaches a final state and return that state.

    A clipped PlanetScope order usually takes ten to forty minutes to run.
    The SDK's own defaults (200 polls, five seconds apart) raise after about
    seventeen minutes, before many orders finish, so this polls every
    *delay* seconds with no attempt limit (``max_attempts=0``). Pass a
    positive ``max_attempts`` to bound the wait.
    """
    pl = ensure()
    return pl.orders.wait(order_id, delay=delay, max_attempts=max_attempts, **kwargs)


def download_order(
    order_id: str, *, subdir: str = "planet", **kwargs: Any
) -> list[Path]:
    """Download a finished order into the easysnowdata cache and return the paths."""
    pl = ensure()
    target = config.cache_dir(subdir, order_id)
    return [
        Path(p) for p in pl.orders.download_order(order_id, directory=target, **kwargs)
    ]


# ── single-asset reads (Data API) ────────────────────────────────────────────


def asset_url(
    item_id: str,
    *,
    item_type: str,
    asset_type: str,
    activate: bool = True,
    wait: bool = True,
    **kwargs: Any,
) -> str:
    """Return a signed download URL for one asset of one scene.

    Activation charges the **whole scene** against the account's quota, so
    this is the quick-look route (``source="data-api"``); ordering with a clip
    is the default for anything loaded as an array.
    """
    pl = ensure()
    asset = pl.data.get_asset(item_type, item_id, asset_type)
    if activate and asset.get("status") != "active":
        _logger.info(
            "Activating %s/%s (%s): this charges the whole scene against the quota.",
            item_type,
            item_id,
            asset_type,
        )
        pl.data.activate_asset(asset)
        if wait:
            asset = pl.data.wait_asset(asset, **kwargs)
    return asset["location"]


def tile_url(item_id: str, *, item_type: str) -> str:
    """The XYZ tile template for a scene (for folium/ipyleaflet quick looks)."""
    return TILES_URL.format(item_type=item_type, item_id=item_id)


def iter_pages(pages: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Flatten the SDK's paged iterators (kept for callers that need raw pages)."""
    for page in pages:
        yield from page.get("features", [page])
