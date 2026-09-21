"""Digital elevation models: Copernicus DEM, NASADEM, SRTM, 3DEP and ALOS World 3D.

Five DEMs, one ``load``. Copernicus GLO-30 is the default because it is the
most recent global model with the fewest voids, but it is a choice, not the
only option, and the others differ in ways that matter for snow work: 3DEP is
10 m over the United States, NASADEM and SRTM are the year-2000 radar
surface that most older studies used, and ALOS World 3D is the optical
stereo model CHILI is derived from. ``PRODUCTS`` lists them and
:func:`compare` says how they differ::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    dem = esd.terrain.dem.load(aoi)                                # Copernicus GLO-30
    dem = esd.terrain.dem.load(aoi, product="nasadem")             # SRTM heritage, no account
    dem = esd.terrain.dem.load(aoi, product="3dep", resolution=10) # US 10 m
    dem = esd.terrain.dem.load(aoi, product="srtm")                # SRTM GL1 v3 on Earth Engine
    dem = esd.terrain.dem.load(aoi, source="earth-search")         # the same GLO-30, unsigned AWS
    items = esd.terrain.dem.search(aoi, product="3dep")            # the covering tiles

Every product returns ``elevation`` in metres with the same dims, CRS
handling and provenance attributes; the vertical datum is in
``attrs["vertical_datum"]`` (EGM2008 for Copernicus, EGM96 for the SRTM
family and ALOS, NAVD88 for 3DEP), which is the first thing to reconcile
before differencing two of them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import geopandas as gpd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT_ID",
    "PRODUCTS",
    "ALIASES",
    "COLLECTIONS",
    "NODATA",
    "search",
    "load",
    "compare",
]

_logger = logging.getLogger(__name__)

#: The default product.
PRODUCT_ID = "copernicus-dem"
#: Copernicus DEM resolution in metres → STAC collection id (same on both catalogs).
COLLECTIONS: dict[int, str] = {30: "cop-dem-glo-30", 90: "cop-dem-glo-90"}
#: The sentinel the Copernicus DEM COGs use for sea and missing tiles.
NODATA = -32767.0

#: Short names accepted by ``product=``.
ALIASES: dict[str, str] = {
    "copernicus": "copernicus-dem",
    "cop-dem": "copernicus-dem",
    "glo30": "copernicus-dem",
    "glo-30": "copernicus-dem",
    "nasadem": "nasadem",
    "srtm": "srtm",
    "srtmgl1": "srtm",
    "3dep": "3dep",
    "alos": "alos-dem",
    "aw3d30": "alos-dem",
    "alos-dem": "alos-dem",
}

_PC_STAC = providers.stac.PLANETARY_COMPUTER_URL
_EARTH_SEARCH_STAC = providers.stac.EARTH_SEARCH_URL
_GEE_CATALOG = "https://developers.google.com/earth-engine/datasets/catalog/"


# ── routes ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StacRoute:
    """A DEM served as COG tiles behind a STAC API."""

    collections: dict[int, str]
    """resolution in metres → collection id."""
    asset: str
    nodata: float | None
    query: Callable[[int], dict[str, Any]] | None = None
    """Extra STAC ``query`` for a resolution (3DEP keeps 10 m and 30 m in one collection)."""


@dataclass(frozen=True)
class _GeeRoute:
    """A DEM served as an Earth Engine image or tiled image collection."""

    asset: str
    band: str
    resolution: int
    mosaic: bool = False
    """True when *asset* is an ImageCollection of tiles to mosaic first."""


_ROUTES: dict[tuple[str, str], _StacRoute | _GeeRoute] = {
    ("copernicus-dem", "planetary-computer"): _StacRoute(COLLECTIONS, "data", NODATA),
    ("copernicus-dem", "earth-search"): _StacRoute(COLLECTIONS, "data", NODATA),
    ("copernicus-dem", "gee"): _GeeRoute(
        "COPERNICUS/DEM/GLO30_2024_1", "DEM", 30, mosaic=True
    ),
    ("nasadem", "planetary-computer"): _StacRoute(
        {30: "nasadem"}, "elevation", -32768.0
    ),
    ("nasadem", "gee"): _GeeRoute("NASA/NASADEM_HGT/001", "elevation", 30),
    ("srtm", "gee"): _GeeRoute("USGS/SRTMGL1_003", "elevation", 30),
    ("3dep", "planetary-computer"): _StacRoute(
        {10: "3dep-seamless", 30: "3dep-seamless"},
        "data",
        -999999.0,  # the USGS fill value; the COGs carry no nodata tag
        query=lambda res: {"gsd": {"eq": res}},
    ),
    ("3dep", "gee"): _GeeRoute(
        "USGS/3DEP/10m_collection", "elevation", 10, mosaic=True
    ),
    ("alos-dem", "planetary-computer"): _StacRoute({30: "alos-dem"}, "data", -9999.0),
    ("alos-dem", "gee"): _GeeRoute("JAXA/ALOS/AW3D30/V4_1", "DSM", 30, mosaic=True),
}

#: Per-product facts that go into the attrs and the comparison table.
_FACTS: dict[str, dict[str, Any]] = {
    "copernicus-dem": {
        "vertical_datum": "EGM2008",
        "surface": "digital surface model (X-band InSAR, TanDEM-X 2011-2015)",
        "default_resolution": 30,
    },
    "nasadem": {
        "vertical_datum": "EGM96",
        "surface": "digital surface model (C-band InSAR, SRTM February 2000, reprocessed and void-filled)",
        "default_resolution": 30,
    },
    "srtm": {
        "vertical_datum": "EGM96",
        "surface": "digital surface model (C-band InSAR, SRTM February 2000, v3 void-filled)",
        "default_resolution": 30,
    },
    "3dep": {
        "vertical_datum": "NAVD88",
        "surface": "bare-earth digital terrain model (lidar and legacy sources, mixed dates)",
        "default_resolution": 10,
    },
    "alos-dem": {
        "vertical_datum": "EGM96",
        "surface": "digital surface model (optical stereo, ALOS PRISM 2006-2011)",
        "default_resolution": 30,
    },
}


def _product_id(product: str | None) -> str:
    if product is None:
        return PRODUCT_ID
    key = str(product).lower()
    if key in ALIASES:
        return ALIASES[key]
    if key in _FACTS:
        return key
    raise ValueError(
        f"Unknown DEM product {product!r}; choose from {', '.join(sorted(_FACTS))} "
        f"(aliases: {', '.join(sorted(ALIASES))})."
    )


def _route(product_id: str, source_id: str) -> _StacRoute | _GeeRoute:
    return _ROUTES[product_id, source_id]


def _resolution(route: _StacRoute | _GeeRoute, product_id: str, resolution: Any) -> int:
    """The DEM resolution to serve, validated against what the route offers."""
    offered = (
        sorted(route.collections)
        if isinstance(route, _StacRoute)
        else [route.resolution]
    )
    if resolution is None:
        default = _FACTS[product_id]["default_resolution"]
        return default if default in offered else offered[0]
    try:
        wanted = int(resolution)
        if wanted != float(resolution):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"{product_id} is available at {_join_m(offered)} only, got {resolution!r}."
        ) from None
    if wanted not in offered:
        raise ValueError(
            f"{product_id} is available at {_join_m(offered)} only, got {wanted} m."
        )
    return wanted


def _join_m(values: list[int]) -> str:
    return " and ".join(f"{v} m" for v in values)


# ── search ────────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    *,
    product: str | None = None,
    source: str | None = None,
    resolution: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Return the DEM tiles covering *aoi* as a GeoDataFrame of STAC items.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    product
        ``"copernicus-dem"`` (default), ``"nasadem"``, ``"3dep"`` or
        ``"alos-dem"`` — the products with a STAC route. ``"srtm"`` is served
        by Earth Engine only, which has no tile search.
    source
        A STAC source of that product (``"planetary-computer"`` is the
        default of every product that has one; Copernicus also has
        ``"earth-search"``).
    resolution
        The DEM resolution in metres where a product offers more than one
        (Copernicus 30/90, 3DEP 10/30).
    **kwargs
        Passed to :func:`easysnowdata.providers.stac.search`.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per tile, filterable and accepted back by :func:`load`.
    """
    pid = _product_id(product)
    prod = catalog.get(pid)
    src = resolve_source(prod, source)
    route = _route(pid, src.id)
    if not isinstance(route, _StacRoute):
        raise ValueError(
            f"{pid} on source={src.id!r} is an Earth Engine asset, which has no "
            "tile search; call load() directly."
        )
    res = _resolution(route, pid, resolution)
    if route.query is not None:
        kwargs.setdefault("query", route.query(res))
    items = providers.stac.search(src.id, route.collections[res], aoi, **kwargs)
    return providers.stac.items_to_geodataframe(items)


# ── load ──────────────────────────────────────────────────────────────────────


def load(
    aoi: Any = None,
    *,
    product: str | None = None,
    source: str | None = None,
    resolution: int | None = None,
    items: Any = None,
    crs: Any = None,
    grid_resolution: float | None = None,
    chunks: Any = contract.DEFAULT,
    mask: bool = True,
    resampling: str = "bilinear",
    **kwargs: Any,
) -> xr.DataArray:
    """Load elevation for *aoi* from one of the DEM products.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. ``clip=False`` on
        an :class:`~easysnowdata.aoi.AOI` returns the covering tiles whole
        (STAC routes).
    product
        Which DEM: ``"copernicus-dem"`` (default), ``"nasadem"``, ``"srtm"``,
        ``"3dep"`` or ``"alos-dem"``. Short forms (``"copernicus"``,
        ``"alos"``) work too; see :data:`ALIASES`.
    source
        Which route to that product: the product's default when ``None``.
        Copernicus: ``"planetary-computer"``, ``"earth-search"`` (unsigned
        AWS, no account) or ``"gee"`` (the newer 2024_1 release). NASADEM, 3DEP
        and ALOS: ``"planetary-computer"`` or ``"gee"``. SRTM: ``"gee"``.
    resolution
        The DEM product's resolution in metres, not the output grid:
        Copernicus 30 or 90, 3DEP 10 or 30, the rest 30. ``None`` picks the
        product's finest.
    items
        Tiles from :func:`search` (a GeoDataFrame or ``ItemCollection``); when
        given, no search is made. STAC routes only.
    crs, grid_resolution
        Output grid. ``crs="utm"`` reprojects to the AOI's UTM zone; the
        native grid is kept when both are ``None``.
    chunks
        Dask chunks; the default is the source's native chunking and ``None``
        loads eagerly.
    mask
        ``True`` (default, §2.5 for continuous products) replaces the nodata
        sentinel with NaN and keeps it as ``rio.encoded_nodata``; ``False``
        keeps the sentinel with ``rio.nodata`` set. Earth Engine routes
        already return NaN outside the asset.
    resampling
        How pixels are interpolated when *crs* or *grid_resolution* puts the
        DEM on a new grid: ``"bilinear"`` (default) or ``"cubic"`` for a
        continuous field like elevation; ``"nearest"`` only reproduces the
        blocky staircase of the source pixels. Both the STAC and the Earth
        Engine routes use the same method, so two copies of one DEM land on
        a common grid identically. Ignored on the native grid.
    **kwargs
        Passed to ``odc.stac.load`` (STAC routes) or
        ``xarray.open_dataset(engine="ee")`` (Earth Engine routes).

    Returns
    -------
    xarray.DataArray
        ``elevation`` in metres, dims ``latitude``/``longitude`` (or ``y``/``x``
        when the grid is projected), lazy unless ``chunks=None``, with
        ``vertical_datum``, ``resolution_m`` and the usual provenance attrs.
    """
    pid = _product_id(product)
    prod = catalog.get(pid)
    src = resolve_source(prod, source)
    route = _route(pid, src.id)
    res = _resolution(route, pid, resolution)
    eager = chunks is None
    if isinstance(route, _StacRoute):
        if crs is not None or grid_resolution is not None:
            kwargs.setdefault("resampling", resampling)
        da, source_url = _load_stac(
            prod,
            src,
            route,
            res,
            aoi,
            items,
            crs,
            grid_resolution,
            chunks,
            mask,
            kwargs,
        )
    else:
        if items is not None:
            raise ValueError(
                f"items= is for the STAC routes; {pid} on source={src.id!r} is an "
                "Earth Engine asset."
            )
        da, source_url = _load_gee(
            prod, src, route, aoi, crs, grid_resolution, chunks, resampling, kwargs
        )
    facts = _FACTS[pid]
    da = da.rename("elevation")
    da.attrs.update(
        contract.provenance(
            prod,
            src,
            source_url=source_url,
            long_name=f"elevation above the {facts['vertical_datum']} geoid"
            if facts["vertical_datum"] != "NAVD88"
            else "elevation above NAVD88",
            units="m",
            vertical_datum=facts["vertical_datum"],
            surface=facts["surface"],
            resolution_m=int(res),
        )
    )
    if eager:
        da = da.compute()
    return da


def _load_stac(
    prod: Product,
    src: Source,
    route: _StacRoute,
    res: int,
    aoi: Any,
    items: Any,
    crs: Any,
    grid_resolution: float | None,
    chunks: Any,
    mask: bool,
    kwargs: dict[str, Any],
) -> tuple[xr.DataArray, str]:
    collection = route.collections[res]
    if items is None:
        query = route.query(res) if route.query is not None else None
        items = providers.stac.search(
            src.id, collection, aoi, **({"query": query} if query else {})
        )
    if len(items) == 0:
        raise ValueError(
            f"No {collection} items cover this AOI on {src.title}"
            + (f" at {res} m" if len(route.collections) > 1 else "")
            + f". {prod.title} covers {src.extent}."
        )
    ds = providers.stac.load(
        items,
        aoi,
        bands=route.asset,
        crs=crs,
        resolution=grid_resolution,
        chunks=None if chunks is contract.DEFAULT else chunks,
        groupby="time",  # every tile of a release shares one datetime
        catalog=src.id,
        **kwargs,
    )
    da = ds[route.asset]
    if "time" in da.dims:
        if da.sizes["time"] == 1:
            da = da.squeeze("time")  # static product: keep the date as a scalar coord
        else:
            # 3DEP tiles carry the date of their newest source, so a box that
            # straddles tiles comes back as several "times": one surface.
            da = da.max("time", keep_attrs=True)
    da = contract.write_crs(da, da.rio.crs)
    nodata = route.nodata if route.nodata is not None else da.rio.nodata
    if nodata is not None:
        da = (
            contract.mask_continuous(da, nodata)
            if mask
            else contract.set_categorical_nodata(da, nodata)
        )
    elif mask and not da.dtype.kind == "f":
        da = da.astype("float32")
    da.attrs["collection"] = collection
    url = f"{providers.stac.CATALOGS[src.id]['url']}/collections/{collection}"
    return da, url


def _load_gee(
    prod: Product,
    src: Source,
    route: _GeeRoute,
    aoi: Any,
    crs: Any,
    grid_resolution: float | None,
    chunks: Any,
    resampling: str,
    kwargs: dict[str, Any],
) -> tuple[xr.DataArray, str]:
    ensure_source(prod, src)
    ee = providers.gee.ee()
    if route.mosaic:
        tiles = ee.ImageCollection(route.asset)
        template = tiles.first().select(route.band)
        image = tiles.select(route.band).mosaic()
    else:
        template = image = ee.Image(route.asset).select(route.band)
    reproject = crs is not None or grid_resolution is not None
    # Fetch a margin of native pixels when the result will be reprojected, or
    # the rotated corners of a projected grid come back NaN.
    fetch_aoi = _with_margin(aoi) if reproject else aoi
    grid = providers.gee.grid_params(template, fetch_aoi)
    params: dict[str, Any] = {"grid": grid}
    if chunks is not contract.DEFAULT and chunks is not None:
        params["chunks"] = chunks
    ds = providers.gee.open_dataset(
        ee.ImageCollection(image), fetch_aoi, **params, **kwargs
    )
    da = ds[route.band]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)  # a single static image
    da = contract.write_crs(da, grid["crs"])
    if da.dtype.kind != "f":
        da = da.astype("float32")
    if reproject:
        da = _reproject(da, aoi, crs, grid_resolution, resampling)
    da.attrs["asset"] = route.asset
    return da, _GEE_CATALOG + route.asset.replace("/", "_")


def _reproject(
    da: xr.DataArray,
    aoi: Any,
    crs: Any,
    grid_resolution: float | None,
    resampling: str = "bilinear",
) -> xr.DataArray:
    """Put an Earth Engine result on the same kind of output grid the STAC routes offer."""
    from odc.geo.xr import xr_reproject  # noqa: PLC0415

    from easysnowdata.aoi import parse_aoi  # noqa: PLC0415

    parsed = parse_aoi(aoi)
    if grid_resolution is None:
        # Keep the native pixel size, expressed in the target CRS.
        native = da.odc.geobox
        target = parsed.utm_crs if str(crs).lower() == "utm" else crs
        grid_resolution = (
            abs(native.resolution.x)
            if native.crs.geographic == _is_geographic(target)
            else (
                abs(native.resolution.x) * 111_320
                if native.crs.geographic
                else abs(native.resolution.x) / 111_320
            )
        )
    geobox = parsed.to_geobox(resolution=grid_resolution, crs=crs)
    # The same method the STAC routes hand odc.stac.load, so the same DEM from
    # two routes lands on a common grid identically.
    out = xr_reproject(da, geobox, resampling=resampling)
    return contract.write_crs(out, geobox.crs)


def _is_geographic(crs: Any) -> bool:
    from pyproj import CRS  # noqa: PLC0415

    return CRS.from_user_input(crs).is_geographic


def _with_margin(aoi: Any, fraction: float = 0.05) -> Any:
    """*aoi* grown by *fraction* of its longer side, so edge pixels survive a rotation.

    ``AOI.buffer`` works in metres; the span is converted from degrees at
    111 km per degree, which is generous enough at any latitude.
    """
    from easysnowdata.aoi import parse_aoi  # noqa: PLC0415

    parsed = parse_aoi(aoi)
    if parsed.is_global:
        return parsed
    west, south, east, north = parsed.bounds
    span_m = max(east - west, north - south) * 111_000.0
    return parsed.buffer(span_m * fraction)


# ── comparison ────────────────────────────────────────────────────────────────


def compare() -> Any:
    """One row per DEM product: resolution, extent, dates, datum, routes and credentials.

    The table :func:`load` chooses from, for deciding which DEM a study
    should use. It is the same information the catalog pages carry, in one
    place.
    """
    import pandas as pd  # noqa: PLC0415

    rows = []
    for pid in _FACTS:
        prod = catalog.get(pid)
        default = prod.default_source
        rows.append(
            {
                "product": pid,
                "title": prod.title,
                "resolution_m": " / ".join(
                    str(r)
                    for r in sorted(
                        {
                            r
                            for s in prod.sources
                            for r in (
                                _route(pid, s.id).collections
                                if isinstance(_route(pid, s.id), _StacRoute)
                                else [_route(pid, s.id).resolution]
                            )
                        }
                    )
                ),
                "extent": default.extent,
                "acquired": default.temporal,
                "vertical_datum": _FACTS[pid]["vertical_datum"],
                "surface": _FACTS[pid]["surface"],
                "sources": ", ".join(s.id for s in prod.sources),
                "credential_free": bool(prod.credential_free_sources),
            }
        )
    return pd.DataFrame(rows).set_index("product")


# ── catalog entries ───────────────────────────────────────────────────────────


def _elevation(nodata: float | None, datum: str) -> tuple[Variable, ...]:
    return (
        Variable(
            "elevation",
            units="m",
            dtype="float32",
            nodata=None if nodata is None else int(nodata),
            long_name=f"elevation above {datum}",
        ),
    )


def _gee_source(
    asset: str,
    *,
    title: str,
    resolution_m: int,
    extent: str,
    temporal: str,
    notes: str,
    label: str,
    kind: str = "image",
) -> Source:
    return Source(
        id="gee",
        provider="gee",
        location=asset,
        requires=("earthengine",),
        resolution_m=resolution_m,
        extent=extent,
        temporal=temporal,
        notes=notes,
        title=title,
        health=Probe(
            label, partial(health.gee_asset, asset, kind), requires=("earthengine",)
        ),
    )


COPERNICUS_PRODUCT = Product(
    id="copernicus-dem",
    theme="terrain",
    title="Copernicus DEM GLO-30 / GLO-90",
    description=(
        "Global 30 m and 90 m digital surface model from the Copernicus DEM "
        "(WorldDEM heritage: TanDEM-X X-band InSAR, 2011-2015), 2021 release "
        "on the STAC routes and the 2024_1 release on Earth Engine. "
        "Elevations are referenced to the EGM2008 geoid. The default DEM here "
        "because it is the most recent global model with the fewest voids; "
        "see esd.terrain.dem.compare() for the alternatives."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="cop-dem-glo-30 / cop-dem-glo-90",
            resolution_m=30,
            temporal="static (2021 release; acquired 2011-2015)",
            notes="signed hrefs, anonymous; the 2021 release",
            title="Planetary Computer",
            health=Probe(
                "Copernicus DEM (Planetary Computer)",
                partial(health.stac_search, _PC_STAC, "cop-dem-glo-30", sign=True),
            ),
        ),
        Source(
            id="earth-search",
            provider="stac",
            location="cop-dem-glo-30 / cop-dem-glo-90",
            resolution_m=30,
            temporal="static (2021 release; acquired 2011-2015)",
            notes=(
                "Element 84's index of the unsigned AWS Open Data buckets "
                "copernicus-dem-30m / -90m; no signing, no account"
            ),
            title="Earth Search (AWS Open Data)",
            health=Probe(
                "Copernicus DEM (Earth Search)",
                partial(health.stac_search, _EARTH_SEARCH_STAC, "cop-dem-glo-30"),
            ),
        ),
        _gee_source(
            "COPERNICUS/DEM/GLO30_2024_1",
            title="Earth Engine (2024_1 release)",
            resolution_m=30,
            extent="global",
            temporal="static (2024_1 release; acquired 2011-2015)",
            notes=(
                "the newer 2024_1 edit of GLO-30, tiled as an ImageCollection and "
                "mosaicked here; 30 m only"
            ),
            label="Copernicus DEM (Earth Engine)",
            kind="image_collection",
        ),
    ),
    variables=_elevation(NODATA, "the EGM2008 geoid"),
    citation=(
        "European Space Agency, Sinergise (2021). Copernicus Global Digital "
        "Elevation Model. Distributed by OpenTopography."
    ),
    license="Copernicus DEM licence (free, attribution)",
    doi="10.5069/G9028PQB",
    loader="easysnowdata.terrain.dem.load",
    examples=("terrain/plot_dem.py",),
    references=(
        "https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30",
        "https://registry.opendata.aws/copernicus-dem/",
        f"{_GEE_CATALOG}COPERNICUS_DEM_GLO30_2024_1",
    ),
    tags=("dem", "elevation", "terrain", "copernicus"),
)

NASADEM_PRODUCT = Product(
    id="nasadem",
    theme="terrain",
    title="NASADEM (SRTM reprocessed)",
    description=(
        "NASA's 2020 reprocessing of the February 2000 SRTM C-band radar "
        "mission: 30 m, 60°N to 56°S, voids filled with ASTER GDEM and other "
        "sources, referenced to the EGM96 geoid. The same surface as SRTM GL1 "
        "with better geolocation and fewer voids, and the credential-free way "
        "to get an SRTM-heritage DEM."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="nasadem",
            resolution_m=30,
            extent="60°N to 56°S",
            temporal="static (acquired February 2000; released 2020)",
            notes="signed hrefs, anonymous; one COG per 1° tile",
            title="Planetary Computer",
            health=Probe(
                "NASADEM (Planetary Computer)",
                partial(health.stac_search, _PC_STAC, "nasadem", sign=True),
            ),
        ),
        _gee_source(
            "NASA/NASADEM_HGT/001",
            title="Earth Engine",
            resolution_m=30,
            extent="60°N to 56°S",
            temporal="static (acquired February 2000; released 2020)",
            notes="the same product as one global image",
            label="NASADEM (Earth Engine)",
        ),
    ),
    variables=_elevation(-32768.0, "the EGM96 geoid"),
    citation=(
        "NASA JPL (2020). NASADEM Merged DEM Global 1 arc second V001. NASA "
        "EOSDIS Land Processes DAAC. https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001"
    ),
    license="Public domain (NASA data)",
    doi="10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001",
    loader="easysnowdata.terrain.dem.load",
    examples=("terrain/plot_dem.py",),
    references=(
        "https://planetarycomputer.microsoft.com/dataset/nasadem",
        "https://lpdaac.usgs.gov/products/nasadem_hgtv001/",
    ),
    tags=("dem", "elevation", "terrain", "srtm", "nasadem"),
)

SRTM_PRODUCT = Product(
    id="srtm",
    theme="terrain",
    title="SRTM GL1 v3 (30 m)",
    description=(
        "The Shuttle Radar Topography Mission 1 arc-second global DEM, version "
        "3 (void-filled), as served by Earth Engine: the February 2000 C-band "
        "radar surface, 60°N to 56°S, EGM96 geoid. Kept under its own name "
        "because so much of the snow literature is built on it; NASADEM is the "
        "same mission reprocessed, and the one to use when no Earth Engine "
        "account is at hand."
    ),
    sources=(
        _gee_source(
            "USGS/SRTMGL1_003",
            title="Earth Engine",
            resolution_m=30,
            extent="60°N to 56°S",
            temporal="static (acquired February 2000; v3 2013)",
            notes=(
                "one global image; the only cloud-native route to SRTM proper "
                "without an Earthdata download of per-tile NetCDF"
            ),
            label="SRTM GL1 (Earth Engine)",
        ),
    ),
    variables=_elevation(None, "the EGM96 geoid"),
    citation=(
        "NASA JPL (2013). NASA Shuttle Radar Topography Mission Global 1 arc "
        "second V003. NASA EOSDIS Land Processes DAAC. "
        "https://doi.org/10.5067/MEaSUREs/SRTM/SRTMGL1.003"
    ),
    license="Public domain (NASA data)",
    doi="10.5067/MEaSUREs/SRTM/SRTMGL1.003",
    loader="easysnowdata.terrain.dem.load",
    examples=("terrain/plot_dem.py",),
    references=(
        f"{_GEE_CATALOG}USGS_SRTMGL1_003",
        "https://lpdaac.usgs.gov/products/srtmgl1v003/",
    ),
    tags=("dem", "elevation", "terrain", "srtm"),
)

THREEDEP_PRODUCT = Product(
    id="3dep",
    theme="terrain",
    title="USGS 3DEP seamless DEM (10 m / 30 m, United States)",
    description=(
        "The USGS 3D Elevation Program seamless bare-earth DEM at 1/3 "
        "arc-second (~10 m) and 1 arc-second (~30 m): lidar where it has been "
        "flown, legacy sources elsewhere, so the acquisition date varies by "
        "tile. Conterminous US, Alaska (mostly coarser), Hawaii and "
        "territories; NAD83 horizontal, NAVD88 vertical. A terrain model, not "
        "a surface model: forest canopy is removed, which the global DEMs "
        "above do not do."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="3dep-seamless",
            resolution_m=10,
            extent="United States",
            temporal="static (tile dates 1925-2020; updated as lidar arrives)",
            notes=(
                "signed hrefs, anonymous; one collection holding both "
                "resolutions, selected by gsd"
            ),
            title="Planetary Computer",
            health=Probe(
                "3DEP seamless (Planetary Computer)",
                partial(
                    health.stac_search,
                    _PC_STAC,
                    "3dep-seamless",
                    sign=True,
                    query={"gsd": {"eq": 10}},
                ),
            ),
        ),
        _gee_source(
            "USGS/3DEP/10m_collection",
            title="Earth Engine",
            resolution_m=10,
            extent="United States",
            temporal="static (tile dates 1925-2020)",
            notes=(
                "the 1/3 arc-second layer as a tiled ImageCollection, mosaicked "
                "here; 10 m only"
            ),
            label="3DEP 10 m (Earth Engine)",
            kind="image_collection",
        ),
    ),
    variables=_elevation(-999999.0, "NAVD88"),
    citation=(
        "U.S. Geological Survey (2019). 3D Elevation Program 1/3 arc-second "
        "and 1 arc-second Digital Elevation Models. https://www.usgs.gov/3d-elevation-program"
    ),
    license="Public domain (US government data)",
    loader="easysnowdata.terrain.dem.load",
    examples=("terrain/plot_dem.py",),
    references=(
        "https://planetarycomputer.microsoft.com/dataset/3dep-seamless",
        "https://www.usgs.gov/3d-elevation-program",
    ),
    tags=("dem", "elevation", "terrain", "3dep", "lidar", "united states"),
)

ALOS_PRODUCT = Product(
    id="alos-dem",
    theme="terrain",
    title="ALOS World 3D-30m (AW3D30)",
    description=(
        "JAXA's global 30 m digital surface model from ALOS PRISM optical "
        "stereo imagery (2006-2011), EGM96 geoid: version 3.2 on the "
        "Planetary Computer, version 4.1 on Earth Engine. The DEM the CHILI "
        "heat-load index is derived from, and an independent surface to check "
        "the radar DEMs against."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="alos-dem",
            resolution_m=30,
            temporal="static (acquired 2006-2011; v3.2 2021)",
            notes="signed hrefs, anonymous; one COG per 1° tile",
            title="Planetary Computer",
            health=Probe(
                "ALOS World 3D (Planetary Computer)",
                partial(health.stac_search, _PC_STAC, "alos-dem", sign=True),
            ),
        ),
        _gee_source(
            "JAXA/ALOS/AW3D30/V4_1",
            title="Earth Engine (v4.1)",
            resolution_m=30,
            extent="global",
            temporal="static (acquired 2006-2011; v4.1 2024)",
            notes="the newer v4.1, tiled as an ImageCollection and mosaicked here",
            label="ALOS World 3D (Earth Engine)",
            kind="image_collection",
        ),
    ),
    variables=_elevation(-9999.0, "the EGM96 geoid"),
    citation=(
        "Japan Aerospace Exploration Agency (2021). ALOS World 3D 30 meter DEM "
        "V3.2. Distributed by OpenTopography. https://doi.org/10.5069/G94M92HB"
    ),
    license="JAXA AW3D30 terms of use (free, attribution)",
    doi="10.5069/G94M92HB",
    loader="easysnowdata.terrain.dem.load",
    examples=("terrain/plot_dem.py",),
    references=(
        "https://planetarycomputer.microsoft.com/dataset/alos-dem",
        "https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm",
    ),
    tags=("dem", "elevation", "terrain", "alos", "aw3d30"),
)

#: The default product's entry, under the name 0.2 exported.
PRODUCT = COPERNICUS_PRODUCT

#: Every DEM product, by id.
PRODUCTS: dict[str, Product] = {
    p.id: p
    for p in (
        COPERNICUS_PRODUCT,
        NASADEM_PRODUCT,
        SRTM_PRODUCT,
        THREEDEP_PRODUCT,
        ALOS_PRODUCT,
    )
}

catalog.register(*PRODUCTS.values(), replace=True)
