"""PlanetScope (and SkySat) imagery from Planet Labs.

Planet is the first key-gated commercial source in the package (plan §12 Q17,
companion §B.11). Two routes, and the difference is quota:

``orders-api`` (default)
    Search with the Data API, then **order** the scenes with the ``clip``
    tool so Planet delivers COGs cut to the AOI, optionally ``harmonize``d to
    Sentinel-2. An order is charged for the clipped area only. Because an
    order spends quota it is never implicit: :func:`load` refuses to place one
    unless ``order=True``, and :func:`order` is the explicit entry point.
``data-api``
    Activate and read a whole scene's asset directly. This is the quick-look
    route for a single scene; activation charges the **full scene** area, and
    the AOI is only applied after the bytes have been streamed.

::

    import easysnowdata as esd
    scenes_gdf = esd.optical.planetscope.search(aoi, "2023-07-01/2023-07-02")
    order = esd.optical.planetscope.order(aoi, items=scenes_gdf.head(1))  # spends quota
    ps_ds = esd.optical.planetscope.load(aoi, order=order)
    udm2_ds = esd.processing.decode_udm2(ps_ds["udm2"])  # snow band

Planet imagery is not redistributable, so nothing in this repository contains
a real scene: the tests run on synthetic deliveries and scrubbed Data API
responses.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.processing import optical as optical_processing

__all__ = [
    "PRODUCT",
    "ITEM_TYPES",
    "BUNDLES",
    "BAND_NAMES",
    "search",
    "order",
    "load",
    "open_delivery",
]

_logger = logging.getLogger(__name__)

#: Item types this module supports, newest use first (companion §B.11).
ITEM_TYPES = ("PSScene", "SkySatCollect")
#: Order bundles: 4-band and 8-band analytic surface reflectance, each with UDM2.
BUNDLES = {
    "analytic_sr": "analytic_sr_udm2",
    "analytic_8b_sr": "analytic_8b_sr_udm2",
    "analytic": "analytic_udm2",
    "visual": "visual",
}
#: Band order of the PlanetScope analytic products.
BAND_NAMES = {
    4: ("blue", "green", "red", "nir"),
    8: (
        "coastal_blue",
        "blue",
        "green_i",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir",
    ),
}
SCENE_SCALE = 1e-4  # surface-reflectance scaling of the analytic_sr bundles
SCENE_NODATA = 0

_PLANET_DOCS = "https://developers.planet.com/docs/data/psscene/"
_UDM2_DOCS = "https://developers.planet.com/docs/data/udm-2/"


def _health_probe() -> None:
    """One authenticated ``GET /data/v1/searches`` through the SDK — no quota spent.

    Goes through ``planet.Planet().data`` so the SDK's own auth stack signs
    the request. The previous probe read the API key off a private
    ``Auth.value`` attribute, which the SDK deprecated and now raises on.
    """
    client = providers.planet.ensure()
    # A generator; pulling one page is one request against the Data API.
    next(iter(client.data.list_searches(limit=1)), None)


PRODUCT = Product(
    id="planetscope",
    theme="optical",
    title="PlanetScope surface reflectance (Planet Labs)",
    description=(
        "PlanetScope 3 m 4- and 8-band imagery with the UDM2 usable-data mask "
        "(clear / snow / shadow / haze / cloud), ordered clipped to the AOI through "
        "the Orders API or read scene-by-scene through the Data API. Commercial: "
        "needs a Planet account with data access, and every order spends quota."
    ),
    sources=(
        Source(
            id="orders-api",
            provider="planet",
            location="https://api.planet.com/compute/ops/orders/v2",
            requires=("planet",),
            resolution_m=3,
            temporal="2016/present",
            latency="hours to a day",
            notes=(
                "Data API search plus an Orders API clip: Planet delivers COGs cut to "
                "the AOI and charges the clipped area. Ordering is explicit (order=True)"
            ),
            title="Planet Orders API (clip)",
            health=Probe("PlanetScope (Planet Data API)", _health_probe),
        ),
        Source(
            id="data-api",
            provider="planet",
            location=providers.planet.DATA_API_URL,
            requires=("planet",),
            resolution_m=3,
            temporal="2016/present",
            latency="hours",
            notes=(
                "activate and read a whole scene; the full scene area is charged and "
                "the AOI is applied client-side. Quick looks and single scenes only"
            ),
            title="Planet Data API (whole scene)",
        ),
    ),
    variables=(
        Variable("blue", units="1", dtype="uint16", nodata=SCENE_NODATA),
        Variable("green", units="1", dtype="uint16", nodata=SCENE_NODATA),
        Variable("red", units="1", dtype="uint16", nodata=SCENE_NODATA),
        Variable("nir", units="1", dtype="uint16", nodata=SCENE_NODATA),
        Variable(
            "snow",
            dtype="uint8",
            long_name="UDM2 snow mask",
            flag_values=(0, 1),
            flag_meanings=("not_snow", "snow"),
            flag_colors=("#00000000", "#1f78b4"),
        ),
    ),
    citation=(
        "Planet Team (2017). Planet Application Program Interface: In Space for Life "
        "on Earth. San Francisco, CA. https://api.planet.com"
    ),
    license="Planet licence (not redistributable; Education & Research programme)",
    references=(_PLANET_DOCS, _UDM2_DOCS),
    loader="easysnowdata.optical.planetscope.load",
    examples=("optical/plot_planetscope.py",),
    tags=("optical", "planet", "planetscope", "commercial", "udm2"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _bundle(bundle: str) -> str:
    if bundle in BUNDLES:
        return BUNDLES[bundle]
    if bundle in BUNDLES.values():
        return bundle
    raise ValueError(f"bundle must be one of {list(BUNDLES)}, got {bundle!r}.")


def _item_type(item_type: str) -> str:
    if item_type not in ITEM_TYPES:
        raise ValueError(f"item_type must be one of {ITEM_TYPES}, got {item_type!r}.")
    return item_type


def _item_ids(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    if isinstance(items, gpd.GeoDataFrame):
        return list(items["id"])
    if isinstance(items, pd.Series):
        return list(items)
    out = []
    for item in items:
        out.append(item["id"] if isinstance(item, dict) else str(item))
    return out


def _acquired(item: Any) -> pd.Timestamp | None:
    if isinstance(item, dict):
        value = item.get("properties", {}).get("acquired") or item.get("acquired")
        return pd.Timestamp(value).tz_localize(None) if value else None
    return None


def _time_from_filename(path: Path) -> pd.Timestamp | None:
    """Planet delivery names start ``YYYYMMDD_HHMMSS_``."""
    stem = path.name
    try:
        return pd.Timestamp(
            f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}T{stem[9:11]}:{stem[11:13]}:{stem[13:15]}"
        )
    except (ValueError, IndexError):
        return None


def _delivery_files(order: Any) -> dict[str, list[Path]]:
    """Group a delivered order's GeoTIFFs into ``scene`` and ``udm2`` lists.

    A delivery also carries ``manifest.json``, the item metadata JSON and the
    ``AnalyticMS_metadata`` XML; only the rasters are kept, whether *order* is
    a directory or the file list an :func:`order` result holds.
    """
    if isinstance(order, (str, Path)):
        paths = sorted(Path(order).rglob("*"))
    else:
        paths = [Path(p) for p in order]
    paths = [p for p in paths if p.suffix.lower() in (".tif", ".tiff")]
    scenes = [p for p in paths if "udm2" not in p.name.lower()]
    udm2 = [p for p in paths if "udm2" in p.name.lower()]
    return {"scene": scenes, "udm2": udm2}


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    item_type: str = "PSScene",
    cloud_cover: float | None = None,
    asset_types: Sequence[str] | None = None,
    limit: int = 100,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search the Planet Data API and return the scenes as a ``GeoDataFrame``.

    A search is free — it spends no quota — and the frame it returns is what
    :func:`order` and :func:`load` take. Only scenes the account may download
    are returned (the Data API permission filter).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    item_type
        ``"PSScene"`` (default) or ``"SkySatCollect"``.
    cloud_cover
        Keep scenes with less than this cloud percentage.
    asset_types
        Require these assets (e.g. ``["ortho_analytic_4b_sr", "ortho_udm2"]``).
    limit
        Maximum number of scenes.

    Notes
    -----
    When *aoi* is given, an ``aoi_cover`` column holds the fraction of the
    AOI inside each scene's footprint. Planet's ``clear_percent`` and
    ``cloud_percent`` describe the whole scene, so a scene can be 98 % clear
    and still touch only a corner of the box; sort on ``aoi_cover`` first
    when choosing what to order.
    """
    src = resolve_source(PRODUCT, source)
    item_type = _item_type(item_type)
    ensure_source(PRODUCT, src)
    search_filter = providers.planet.build_search_filter(
        aoi, time, cloud_cover=cloud_cover, asset_types=asset_types
    )
    items = providers.planet.search([item_type], search_filter, limit=limit, **kwargs)
    gdf = providers.planet.items_to_geodataframe(items)
    if aoi is not None and len(gdf):
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            geom = parsed.geometry
            gdf["aoi_cover"] = gdf.geometry.apply(
                lambda g: (
                    g.intersection(geom).area / geom.area if g is not None else 0.0
                )
            ).astype("float64")
    gdf.attrs = {"source": src.id, "item_type": item_type}
    return gdf


def order(
    aoi: Any,
    *,
    items: Any = None,
    time: Any = None,
    item_type: str = "PSScene",
    bundle: str = "analytic_sr",
    harmonize: str | None = "Sentinel-2",
    composite: bool = False,
    name: str | None = None,
    reuse: bool = True,
    wait: bool = True,
    download: bool = True,
    cloud_cover: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Order scenes clipped to *aoi* and (by default) wait for and download them.

    **This spends the account's quota.** Returns a dict with ``order_id``,
    ``state``, ``files`` (when downloaded) and the request that was sent.

    Give the order a *name* to make the call repeatable: with ``reuse=True``
    (the default) a *name* that already belongs to a successful order on the
    account is re-downloaded instead of re-ordered, so a notebook or a docs
    build that runs again spends nothing. Planet keeps delivered results for
    a limited time; when the old order is gone a new one is placed.

    Parameters
    ----------
    aoi
        Required: the clip geometry. Ordering without one is refused.
    items
        Scenes to order (a :func:`search` frame, ids, or item dicts). When
        omitted, *time* and *cloud_cover* drive a fresh search.
    bundle
        ``"analytic_sr"`` (4-band surface reflectance, default),
        ``"analytic_8b_sr"``, ``"analytic"`` or ``"visual"``; each analytic
        bundle includes the UDM2 mask.
    harmonize
        Radiometric harmonisation target (``"Sentinel-2"`` by default, or
        ``None`` to skip it). Planet's ``harmonize`` tool rescales PS2.SD and
        PSB.SD surface reflectance, band by band, onto Sentinel-2's
        radiometry so the two sensors can be mixed in one time series.
    name, reuse
        An order name, and whether an existing successful order of that name
        is re-downloaded instead of ordered again (see above).
    """
    src = resolve_source(PRODUCT, "orders-api")
    item_type = _item_type(item_type)
    ensure_source(PRODUCT, src)
    if name and reuse:
        previous = providers.planet.find_order(name)
        if previous is not None:
            _logger.info(
                "Reusing Planet order %r (%s, created %s); nothing is charged.",
                name,
                previous["id"],
                previous.get("created_on", "?")[:10],
            )
            result = {
                "order_id": previous["id"],
                "state": previous.get("state"),
                "item_ids": [
                    i for prod in previous.get("products", []) for i in prod["item_ids"]
                ],
                "item_type": item_type,
                "bundle": _bundle(bundle),
                "order": previous,
                "reused": True,
            }
            if download:
                result["files"] = providers.planet.download_order(previous["id"])
            return result
    if items is None:
        items = search(
            aoi, time, item_type=item_type, cloud_cover=cloud_cover, limit=50
        )
    ids = _item_ids(items)
    created = providers.planet.create_order(
        ids,
        item_type=item_type,
        product_bundle=_bundle(bundle),
        aoi=aoi,
        name=name,
        harmonize=harmonize,
        composite=composite,
        **kwargs,
    )
    order_id = created["id"]
    result: dict[str, Any] = {
        "order_id": order_id,
        "state": created.get("state"),
        "item_ids": ids,
        "item_type": item_type,
        "bundle": _bundle(bundle),
        "order": created,
        "reused": False,
    }
    if wait:
        result["state"] = providers.planet.wait_order(order_id)
    if download:
        result["files"] = providers.planet.download_order(order_id)
    return result


def open_delivery(
    delivery: Any,
    aoi: Any = None,
    *,
    bands: Sequence[str] | None = None,
    scale: bool = True,
    mask_nodata: bool = True,
    chunks: Any = True,
    source: Any = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Open delivered Planet COGs (a directory, file list, or order result).

    Scenes are stacked on ``time`` from their file names, the analytic bands
    are named (:data:`BAND_NAMES`), and a delivered ``udm2`` mask is decoded
    into its named layers (:func:`easysnowdata.processing.decode_udm2`).
    """
    src = source if source is not None else resolve_source(PRODUCT, "orders-api")
    files = _delivery_files(
        delivery["files"]
        if isinstance(delivery, dict) and "files" in delivery
        else delivery
    )
    if not files["scene"]:
        raise ValueError(f"No GeoTIFFs found in the delivery {delivery!r}.")
    parsed = parse_aoi(aoi) if aoi is not None else None

    scenes, times = [], []
    for path in files["scene"]:
        da = providers.raster_http.open(
            path, parsed, chunks=chunks, squeeze=False, **kwargs
        )
        names = BAND_NAMES.get(da.sizes.get("band", 0))
        if names is None:
            raise ValueError(
                f"{path.name} has {da.sizes.get('band')} bands; expected 4 or 8 "
                f"({list(BAND_NAMES)})."
            )
        ds = da.assign_coords(band=list(names)).to_dataset(dim="band")
        scenes.append(ds)
        times.append(_time_from_filename(path) or pd.Timestamp("1970-01-01"))
    ds = xr.concat(
        [
            scene_ds.expand_dims(time=[t])
            for scene_ds, t in zip(scenes, times, strict=True)
        ],
        dim="time",
    ).sortby("time")

    if files["udm2"]:
        masks_by_time = []
        for path, when in zip(files["udm2"], times, strict=False):
            udm2_da = providers.raster_http.open(
                path, parsed, chunks=chunks, squeeze=False, **kwargs
            )
            decoded_ds = optical_processing.decode_udm2(udm2_da)
            masks_by_time.append(decoded_ds.expand_dims(time=[when]))
        udm2_ds = xr.concat(masks_by_time, dim="time").sortby("time")
        ds = ds.merge(udm2_ds, compat="override", join="left")

    if bands is not None:
        keep = [b for b in bands if b in ds.data_vars]
        ds = ds[keep]
    if mask_nodata:
        for band in list(ds.data_vars):
            if band in BAND_NAMES[4] or band in BAND_NAMES[8]:
                ds[band] = contract.mask_continuous(ds[band], SCENE_NODATA)
    if scale:
        for band in list(ds.data_vars):
            if band in BAND_NAMES[4] or band in BAND_NAMES[8]:
                ds[band] = (ds[band] * SCENE_SCALE).assign_attrs(
                    {**ds[band].attrs, "units": "1", "scale": SCENE_SCALE}
                )
    ds = contract.finalize(
        ds,
        PRODUCT,
        src,
        variables=(),
        source_url=_PLANET_DOCS,
        attrs={
            "scaled_to_reflectance": str(bool(scale)),
            "udm2_bands": " ".join(optical_processing.UDM2_BANDS)
            if files["udm2"]
            else None,
        },
    )
    return ds


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    items: Any = None,
    order: Any = None,  # noqa: A002 — the order to load
    source: str | None = None,
    item_type: str = "PSScene",
    bundle: str = "analytic_sr",
    asset_type: str = "ortho_analytic_4b_sr",
    bands: Sequence[str] | None = None,
    scale: bool = True,
    mask_nodata: bool = True,
    chunks: Any = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Load PlanetScope imagery as an ``xarray.Dataset`` (``time``, ``y``, ``x``).

    Parameters
    ----------
    order
        A finished order (the dict :func:`order` returns, a directory, or a
        list of files) to open. This is the normal path: search, order once,
        then load the delivery as often as you like.
    source
        ``"orders-api"`` (default) or ``"data-api"``. The Data API route
        activates and reads whole scenes, so it takes a single scene and
        charges the full scene area against the quota.
    items
        For ``source="data-api"``: the scene (or scenes) to read.

    Notes
    -----
    ``load`` never places an order by itself — pass ``order=`` with a finished
    order, or call :func:`order` explicitly. That keeps quota spending visible
    in the calling code.
    """
    src = resolve_source(PRODUCT, source)
    ensure_source(PRODUCT, src)
    if order is not None:
        return open_delivery(
            order,
            aoi,
            bands=bands,
            scale=scale,
            mask_nodata=mask_nodata,
            chunks=chunks,
            source=src,
            **kwargs,
        )
    if src.id != "data-api":
        raise ValueError(
            "load() does not place orders: call easysnowdata.optical.planetscope.order("
            "aoi, ...) first (it spends quota) and pass the result as order=, or use "
            'source="data-api" to read a single whole scene.'
        )
    if items is None:
        items = search(aoi, time, item_type=item_type, limit=1)
    ids = _item_ids(items)
    if not ids:
        raise ValueError("No Planet scenes for this AOI and time.")
    if len(ids) > 1:
        _logger.warning(
            "source='data-api' activates whole scenes; loading %d of them charges "
            "%d full scene areas against the quota.",
            len(ids),
            len(ids),
        )
    arrays, times = [], []
    for item_id in ids:
        url = providers.planet.asset_url(
            item_id, item_type=item_type, asset_type=asset_type
        )
        da = providers.raster_http.open(
            url,
            parse_aoi(aoi) if aoi is not None else None,
            chunks=chunks,
            squeeze=False,
            **kwargs,
        )
        names = BAND_NAMES.get(da.sizes.get("band", 0), BAND_NAMES[4])
        arrays.append(da.assign_coords(band=list(names)).to_dataset(dim="band"))
        times.append(
            _acquired(items[0] if isinstance(items, list) else None)
            or pd.Timestamp("1970-01-01")
        )
    ds = xr.concat(
        [
            scene_ds.expand_dims(time=[t])
            for scene_ds, t in zip(arrays, times, strict=True)
        ],
        dim="time",
    )
    if mask_nodata:
        for band in list(ds.data_vars):
            ds[band] = contract.mask_continuous(ds[band], SCENE_NODATA)
    if scale:
        for band in list(ds.data_vars):
            ds[band] = (ds[band] * SCENE_SCALE).assign_attrs(
                {**ds[band].attrs, "units": "1", "scale": SCENE_SCALE}
            )
    if bands is not None:
        ds = ds[[b for b in bands if b in ds.data_vars]]
    return contract.finalize(
        ds,
        PRODUCT,
        src,
        variables=(),
        source_url=_PLANET_DOCS,
        attrs={"asset_type": asset_type, "scaled_to_reflectance": str(bool(scale))},
    )


def snow_fraction(
    udm2: xr.Dataset, *, dims: Sequence[str] = ("y", "x")
) -> xr.DataArray:
    """Fraction of usable pixels UDM2 calls snow, per time step.

    A convenience over the decoded mask: ``snow / (clear + snow)`` computed on
    the pixels UDM2 considers usable at all.
    """
    snow_da = udm2["snow"].astype("float32")
    usable_da = (udm2["clear"].astype("float32") + snow_da).where(
        lambda total_da: total_da > 0
    )
    return (snow_da / usable_da).mean(dim=list(dims)).rename("snow_fraction")
