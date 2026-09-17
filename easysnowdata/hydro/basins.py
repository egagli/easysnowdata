"""Basin and watershed boundaries (§4.7, companion §B.10).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    huc = esd.hydro.basins.huc(aoi, level=12)          # USGS WBD, no credentials
    atlas = esd.hydro.basins.hydrobasins(aoi, level=5)  # BasinATLAS attributes
    major = esd.hydro.basins.grdc_major(aoi)            # GRDC major river basins
    wmo = esd.hydro.basins.grdc_wmo(aoi)                # GRDC / WMO basins

Four products share this module because they answer the same question with
different geographies. Every loader returns a :class:`geopandas.GeoDataFrame`
in EPSG:4326 holding the features that *intersect* the AOI — whole basins, not
polygons cut at the AOI edge — with the provenance attrs of §2.5 in
``.attrs``.

The HUC default moved off Earth Engine (§12 Q7): the USGS Watershed Boundary
Dataset is public through its own ArcGIS REST service, so HUC boundaries need
no credentials at all. Earth Engine stays as ``source="gee"``.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import geopandas as gpd
import shapely

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = [
    "DATASETS",
    "WBD_SERVICE",
    "WBD_LAYERS",
    "HYDROSHEDS_REGIONS",
    "huc",
    "hydrobasins",
    "grdc_major",
    "grdc_wmo",
    "load",
]

_logger = logging.getLogger(__name__)

# ── HUC (USGS Watershed Boundary Dataset) ────────────────────────────────────

WBD_SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
#: HUC level → the service's layer id.
WBD_LAYERS: dict[int, int] = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8}
#: The columns ``get_huc_geometries`` returned, in its order.
HUC_COLUMNS = ("name", "areasqkm", "states", "tnmid")
GEE_WBD_ASSET = "USGS/WBD/2017/HUC{level:02d}"

# ── HydroBASINS / BasinATLAS ─────────────────────────────────────────────────

BASINATLAS_URL = (
    "https://ndownloader.figshare.com/files/20082137/BasinATLAS_Data_v10.gdb.zip"
)
HYDROSHEDS_URL = "https://data.hydrosheds.org/file/HydroBASINS/standard"
#: HydroSHEDS region code → title and approximate extent (west, south, east, north).
HYDROSHEDS_REGIONS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "af": ("Africa", (-19.0, -35.0, 55.0, 38.0)),
    "ar": ("North American Arctic", (-180.0, 50.0, -60.0, 84.0)),
    "as": ("Central and South-East Asia", (57.0, -12.0, 180.0, 61.0)),
    "au": ("Australia and Oceania", (94.0, -56.0, 180.0, 21.0)),
    "eu": ("Europe and Middle East", (-25.0, 12.0, 70.0, 62.0)),
    "gr": ("Greenland", (-75.0, 59.0, -10.0, 84.0)),
    "na": ("North and Central America", (-138.0, 5.0, -52.0, 62.0)),
    "sa": ("South America", (-93.0, -56.0, -32.0, 15.0)),
    "si": ("Siberia", (58.0, 45.0, 180.0, 81.0)),
}
GEE_HYDROATLAS_ASSET = "WWF/HydroATLAS/v1/Basins/level{level:02d}"

# ── GRDC ─────────────────────────────────────────────────────────────────────

GRDC_MAJOR_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0041426/DR0051689/"
    "major_basins_of_the_world_0_0_0.zip"
)
GRDC_WMO_URL = "https://grdc.bafg.de/downloads/wmobb_json.zip"
GRDC_WMO_MEMBER = "wmobb_basins.json"

#: Dataset name → (product id, loader) for :func:`load`.
DATASETS = ("huc", "hydrobasins", "grdc-major-river-basins", "grdc-wmo-basins")


def _resolve(product_id: str, source: str | None) -> tuple[Any, Any]:
    """The catalog product and the chosen access route."""
    product = catalog.get(product_id)
    return product, resolve_source(product, source)


def _finish(gdf: gpd.GeoDataFrame, product, src, **extra: Any) -> gpd.GeoDataFrame:
    """EPSG:4326 plus the provenance attrs every product carries (§2.5)."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.attrs.update(contract.provenance(product, src, **extra))
    return gdf


def _level(level: Any, *, low: int = 1, high: int = 12) -> int:
    value = int(str(level).lstrip("0") or "0")
    if not low <= value <= high:
        raise ValueError(f"Level must be between {low} and {high}, got {level}")
    return value


# ── HUC ──────────────────────────────────────────────────────────────────────


def _wbd_query(
    layer: int,
    bounds: tuple[float, float, float, float] | None,
    *,
    fields: str,
    page_size: int,
    geometry_precision: int,
    timeout: float,
    where: str | None,
) -> list[dict[str, Any]]:
    """Page through one WBD layer, returning GeoJSON features.

    The service answers HTTP 500 when a page takes too long, so each page is
    retried before the query is given up on.
    """
    import requests  # noqa: PLC0415

    params: dict[str, Any] = {
        "outFields": fields,
        "outSR": "4326",
        "f": "geojson",
        "geometryPrecision": geometry_precision,
        "orderByFields": "objectid",
        "where": where or "1=1",
    }
    if bounds is not None:
        params.update(
            geometry=",".join(f"{v:.6f}" for v in bounds),
            geometryType="esriGeometryEnvelope",
            inSR="4326",
            spatialRel="esriSpatialRelIntersects",
        )
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = dict(params, resultOffset=offset, resultRecordCount=page_size)
        payload = None
        for attempt in (1, 2, 3):
            try:
                response = requests.get(
                    f"{WBD_SERVICE}/{layer}/query", params=page, timeout=timeout
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001 — the service 500s on slow pages
                _logger.debug("WBD page at offset %s failed: %s", offset, exc)
                if attempt < 3:
                    time.sleep(2 * attempt)
        if payload is None:
            raise RuntimeError(
                f"The USGS WBD service failed on the page at offset {offset}. It "
                "times out on large queries: ask for a smaller AOI, a coarser "
                'level, or use source="gee".'
            )
        if "error" in payload:
            raise RuntimeError(f"USGS WBD service error: {payload['error']}")
        batch = payload.get("features", [])
        features.extend(batch)
        truncated = payload.get("exceededTransferLimit") or payload.get(
            "properties", {}
        ).get("exceededTransferLimit")
        if not truncated or not batch:
            return features
        offset += len(batch)


def huc(
    aoi: Any = None,
    *,
    level: int | str = 8,
    source: str | None = None,
    columns: Any = None,
    where: str | None = None,
    page_size: int = 200,
    geometry_precision: int = 6,
    timeout: float = 120,
) -> gpd.GeoDataFrame:
    """Hydrologic unit (HUC) boundaries for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` asks for
        the whole United States, which only the coarse levels can answer.
    level
        2, 4, 6, 8 (default), 10, 12, 14 or 16 — the number of HUC digits.
        Strings (``"02"``) are accepted.
    source
        ``"usgs-wbd"`` (default, the public ArcGIS REST service, no
        credentials) or ``"gee"``.
    columns
        Attributes to keep. ``None`` keeps the columns the old
        ``get_huc_geometries`` returned; ``"all"`` keeps everything.
    where
        Extra SQL filter for the REST service (``"states LIKE '%WA%'"``).
    page_size, geometry_precision, timeout
        REST paging controls. The service times out on large pages, so it is
        paged in blocks of *page_size* features with coordinates rounded to
        *geometry_precision* decimals.

    Returns
    -------
    geopandas.GeoDataFrame
        The hydrologic units intersecting the AOI, in EPSG:4326.
    """
    product, src = _resolve("huc", source)
    digits = _level(level, low=2, high=16)
    if digits not in WBD_LAYERS:
        raise ValueError(f"HUC level must be one of {sorted(WBD_LAYERS)}, got {level}.")
    code = f"huc{digits}"
    parsed = parse_aoi(aoi)
    if src.provider == "gee":
        asset = GEE_WBD_ASSET.format(level=digits)
        gdf = providers.gee.features_to_geodataframe(asset, parsed)
        source_url = f"https://developers.google.com/earth-engine/datasets/catalog/{asset.replace('/', '_')}"
    else:
        wanted = "*" if columns == "all" else ",".join([code, *HUC_COLUMNS])
        bounds = None if parsed.is_global or not parsed.clip else parsed.bounds
        features = _wbd_query(
            WBD_LAYERS[digits],
            bounds,
            fields=wanted,
            page_size=page_size,
            geometry_precision=geometry_precision,
            timeout=timeout,
            where=where,
        )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        source_url = f"{WBD_SERVICE}/{WBD_LAYERS[digits]}"
    if len(gdf) and columns is None:
        keep = [c for c in ("name", code, *HUC_COLUMNS[1:]) if c in gdf.columns]
        gdf = gdf[[*keep, "geometry"]]
    elif isinstance(columns, (list, tuple)):
        gdf = gdf[[*columns, "geometry"]]
    return _finish(gdf, product, src, source_url=source_url, huc_level=digits)


# ── HydroBASINS ──────────────────────────────────────────────────────────────


def _hydrosheds_region(aoi: Any) -> str:
    """The HydroSHEDS region whose extent overlaps the AOI most."""
    parsed = parse_aoi(aoi)
    if parsed.is_global:
        raise ValueError(
            "The HydroSHEDS route is per region; pass region= (one of "
            f"{', '.join(HYDROSHEDS_REGIONS)}) or use the global BasinATLAS source."
        )
    geometry = parsed.geometry
    overlaps = {
        code: shapely.box(*bounds).intersection(geometry).area
        for code, (_, bounds) in HYDROSHEDS_REGIONS.items()
    }
    best = max(overlaps, key=lambda code: overlaps[code])
    if overlaps[best] <= 0:
        raise ValueError(
            "No HydroSHEDS region covers this AOI; pass region= explicitly "
            f"(one of {', '.join(HYDROSHEDS_REGIONS)})."
        )
    _logger.info("HydroSHEDS region %s (%s).", best, HYDROSHEDS_REGIONS[best][0])
    return best


def hydrobasins(
    aoi: Any = None,
    *,
    level: int = 5,
    source: str | None = None,
    region: str | None = None,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """HydroBASINS / BasinATLAS sub-basin boundaries for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    level
        Pfafstetter level 1-12; higher is finer.
    source
        ``"figshare-basinatlas"`` (default; one 2.7 GB global geodatabase with
        the BasinATLAS attributes, read with mask pushdown),
        ``"hydrosheds"`` (per-region zips, much smaller, geometry only) or
        ``"gee"``.
    region
        HydroSHEDS route only: one of :data:`HYDROSHEDS_REGIONS`. Inferred
        from the AOI when omitted.
    columns
        Attributes to keep (BasinATLAS ships several hundred).
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        The sub-basins intersecting the AOI, in EPSG:4326.
    """
    product, src = _resolve("hydrobasins", source)
    value = _level(level)
    if src.provider == "gee":
        asset = GEE_HYDROATLAS_ASSET.format(level=value)
        gdf = providers.gee.features_to_geodataframe(asset, aoi)
        source_url = f"https://developers.google.com/earth-engine/datasets/catalog/{asset.replace('/', '_')}"
    elif src.id == "hydrosheds":
        code = region or _hydrosheds_region(aoi)
        if code not in HYDROSHEDS_REGIONS:
            raise ValueError(
                f"Unknown HydroSHEDS region {code!r}; available: "
                f"{', '.join(HYDROSHEDS_REGIONS)}."
            )
        archive = f"hybas_{code}_lev01-12_v1c.zip"
        source_url = f"{HYDROSHEDS_URL}/{archive}"
        path = providers.raster_http.fetch(source_url, archive, subdir="hydrosheds")
        gdf = providers.vector_http.read(
            f"zip://{path}!hybas_{code}_lev{value:02d}_v1c.shp",
            aoi,
            columns=columns,
            **kwargs,
        )
        if not len(gdf):
            _logger.warning(
                "No level-%s basins in HydroSHEDS region %s for this AOI; "
                "pass region= if the AOI sits in another region.",
                value,
                code,
            )
    else:
        source_url = BASINATLAS_URL
        gdf = providers.vector_http.read(
            f"zip+{BASINATLAS_URL}",
            aoi,
            layer=f"BasinATLAS_v10_lev{value:02d}",
            columns=columns,
            **kwargs,
        )
    return _finish(gdf, product, src, source_url=source_url, level=value)


# ── GRDC ─────────────────────────────────────────────────────────────────────


def grdc_major(
    aoi: Any = None, *, source: str | None = None, **kwargs: Any
) -> gpd.GeoDataFrame:
    """The GRDC major river basins of the world that intersect *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` returns
        all 405 basins.
    source
        Only ``"world-bank"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        Whole basins (not clipped to the AOI), in EPSG:4326.
    """
    product, src = _resolve("grdc-major-river-basins", source)
    gdf = providers.vector_http.read(f"zip+{GRDC_MAJOR_URL}", aoi, **kwargs)
    return _finish(gdf, product, src, source_url=GRDC_MAJOR_URL)


def grdc_wmo(
    aoi: Any = None, *, source: str | None = None, **kwargs: Any
) -> gpd.GeoDataFrame:
    """The GRDC / WMO basins and sub-basins that intersect *aoi*.

    The GRDC server answers HEAD with 400, which GDAL's ``/vsicurl`` sends
    before any range read, so the archive is fetched once with a plain GET
    into the easysnowdata cache and read from there.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        Only ``"grdc"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        Whole basins (not clipped to the AOI), in EPSG:4326.
    """
    product, src = _resolve("grdc-wmo-basins", source)
    path = providers.raster_http.fetch(GRDC_WMO_URL, "wmobb_json.zip", subdir="grdc")
    gdf = providers.vector_http.read(f"zip://{path}!{GRDC_WMO_MEMBER}", aoi, **kwargs)
    return _finish(gdf, product, src, source_url=GRDC_WMO_URL)


_LOADERS = {
    "huc": huc,
    "hydrobasins": hydrobasins,
    "grdc-major-river-basins": grdc_major,
    "grdc-wmo-basins": grdc_wmo,
}


def load(aoi: Any = None, *, dataset: str = "huc", **kwargs: Any) -> gpd.GeoDataFrame:
    """Load one of the basin datasets by catalog id.

    ``dataset`` is one of :data:`DATASETS`; the remaining keyword arguments go
    to that dataset's own loader (:func:`huc`, :func:`hydrobasins`,
    :func:`grdc_major`, :func:`grdc_wmo`).
    """
    try:
        loader = _LOADERS[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown basin dataset {dataset!r}; available: {', '.join(DATASETS)}."
        ) from None
    return loader(aoi, **kwargs)


HUC_PRODUCT = Product(
    id="huc",
    theme="hydro",
    title="USGS Watershed Boundary Dataset (HUC)",
    description=(
        "Hydrologic unit boundaries (HUC2 to HUC16) for the United States, from "
        "the USGS/NRCS Watershed Boundary Dataset."
    ),
    sources=(
        Source(
            id="usgs-wbd",
            provider="vector_http",
            location=WBD_SERVICE,
            extent="United States",
            temporal="continuously updated",
            notes=(
                "the public ArcGIS REST service: no credentials, server-side "
                "spatial query, paged because it times out on large pages"
            ),
            title="USGS WBD (ArcGIS REST)",
            health=Probe(
                "HUC geometries (USGS WBD REST)",
                partial(
                    health.http_first_byte,
                    f"{WBD_SERVICE}/4/query?where=1%3D1&outFields=huc8&"
                    "resultRecordCount=1&returnGeometry=false&f=json",
                ),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location="USGS/WBD/2017/HUC{level}",
            requires=("earthengine",),
            extent="United States",
            temporal="static (2017)",
            notes="the 2017 snapshot; the REST service is newer and needs no account",
            title="Earth Engine",
            health=Probe(
                "HUC geometries (GEE/USGS WBD)",
                partial(health.gee_asset, "USGS/WBD/2017/HUC08", "feature_collection"),
            ),
        ),
    ),
    variables=(
        Variable("huc", long_name="hydrologic unit code"),
        Variable("name", long_name="hydrologic unit name"),
        Variable("areasqkm", units="km2", long_name="area"),
    ),
    citation=(
        "Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D. (2022). Federal "
        "standards and procedures for the national Watershed Boundary Dataset "
        "(WBD), 5th ed. U.S. Geological Survey Techniques and Methods 11-A3."
    ),
    license="Public domain (US government data)",
    doi="10.3133/tm11A3",
    loader="easysnowdata.hydro.basins.huc",
    examples=("hydro/plot_basins.py",),
    references=(
        "https://www.usgs.gov/national-hydrography/watershed-boundary-dataset",
    ),
    tags=("basins", "watersheds", "huc"),
)

HYDROBASINS_PRODUCT = Product(
    id="hydrobasins",
    theme="hydro",
    title="HydroBASINS / BasinATLAS",
    description=(
        "Global nested sub-basin polygons (Pfafstetter levels 1-12). BasinATLAS "
        "adds several hundred hydro-environmental attributes per basin."
    ),
    sources=(
        Source(
            id="figshare-basinatlas",
            provider="vector_http",
            location=BASINATLAS_URL,
            extent="global",
            temporal="static (v1.0, 2019)",
            notes=(
                "2.7 GB file geodatabase read with mask pushdown, so only the "
                "intersecting basins are fetched; carries the BasinATLAS attributes"
            ),
            title="figshare BasinATLAS gdb",
            health=Probe(
                "HydroATLAS basins (figshare)",
                partial(health.http_first_byte, BASINATLAS_URL),
            ),
        ),
        Source(
            id="hydrosheds",
            provider="vector_http",
            location=f"{HYDROSHEDS_URL}/hybas_<region>_lev01-12_v1c.zip",
            extent="per region (9 regions)",
            temporal="static (v1c)",
            notes=(
                "~50-300 MB per region instead of 2.7 GB, cached on first use; "
                "geometry and the basic Pfafstetter fields, no BasinATLAS attributes"
            ),
            title="HydroSHEDS regional zips",
            health=Probe(
                "HydroBASINS (HydroSHEDS regional zip)",
                partial(
                    health.http_first_byte,
                    f"{HYDROSHEDS_URL}/hybas_na_lev01-12_v1c.zip",
                ),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location="WWF/HydroATLAS/v1/Basins/level{level}",
            requires=("earthengine",),
            extent="global",
            temporal="static (v1.0)",
            notes="server-side query, no download",
            title="Earth Engine",
            health=Probe(
                "HydroBASINS (GEE/HydroATLAS)",
                partial(
                    health.gee_asset,
                    "WWF/HydroATLAS/v1/Basins/level05",
                    "feature_collection",
                ),
            ),
        ),
    ),
    citation=(
        "Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global "
        "hydro-environmental sub-basin and river reach characteristics at high "
        "spatial resolution. Scientific Data 6, 283."
    ),
    license="CC BY 4.0",
    doi="10.1038/s41597-019-0300-6",
    loader="easysnowdata.hydro.basins.hydrobasins",
    references=("https://www.hydrosheds.org/products/hydrobasins",),
    tags=("basins", "watersheds", "hydrobasins"),
)

GRDC_MAJOR_PRODUCT = Product(
    id="grdc-major-river-basins",
    theme="hydro",
    title="GRDC major river basins of the world",
    description=(
        "The major river and lake basins of the world (GRDC, 2nd revised "
        "edition), distributed through the World Bank data catalog."
    ),
    sources=(
        Source(
            id="world-bank",
            provider="vector_http",
            location=GRDC_MAJOR_URL,
            temporal="static (2020)",
            notes="zipped shapefile read in place",
            title="World Bank zip",
            health=Probe(
                "GRDC major river basins (World Bank)",
                partial(health.http_first_byte, GRDC_MAJOR_URL),
            ),
        ),
    ),
    citation=(
        "GRDC (2020). Major River Basins of the World / Global Runoff Data "
        "Centre, 2nd rev. ext. ed. Koblenz: BfG."
    ),
    license="Open (attribution)",
    loader="easysnowdata.hydro.basins.grdc_major",
    examples=("hydro/plot_basins.py",),
    references=("https://www.bafg.de/GRDC",),
    tags=("basins", "rivers"),
)

GRDC_WMO_PRODUCT = Product(
    id="grdc-wmo-basins",
    theme="hydro",
    title="GRDC / WMO basins and sub-basins",
    description=(
        "WMO basins and sub-basins (GRDC, 3rd revised edition): hydrographic "
        "regions with their WMO region numbers and drainage areas."
    ),
    sources=(
        Source(
            id="grdc",
            provider="vector_http",
            location=GRDC_WMO_URL,
            temporal="static (2020)",
            notes=(
                "grdc.bafg.de answers HEAD with 400, so the archive is fetched "
                "once with a plain GET into the cache and read from there"
            ),
            title="GRDC zip",
            health=Probe(
                "GRDC WMO basins", partial(health.http_first_byte, GRDC_WMO_URL)
            ),
        ),
    ),
    citation=(
        "GRDC (2020). WMO Basins and Sub-Basins / Global Runoff Data Centre, "
        "3rd rev. ext. ed. Koblenz: BfG."
    ),
    license="Open (attribution)",
    loader="easysnowdata.hydro.basins.grdc_wmo",
    references=("https://www.bafg.de/GRDC",),
    tags=("basins", "wmo"),
)

catalog.register(
    HUC_PRODUCT,
    HYDROBASINS_PRODUCT,
    GRDC_MAJOR_PRODUCT,
    GRDC_WMO_PRODUCT,
    replace=True,
)
