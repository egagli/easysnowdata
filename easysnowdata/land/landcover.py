"""ESA WorldCover global 10 m land cover (§4.4, companion §B.7).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    lc = esd.land.landcover.load(aoi)                          # v200 (2021), PC
    lc = esd.land.landcover.load(aoi, version="v100")          # v100 (2020)
    lc = esd.land.landcover.load(aoi, source="aws-open-data")  # unsigned bucket
    esd.plotting.categorical(lc)

WorldCover ended at v200 (2021); ESA names Copernicus LCFM on CDSE as the
successor, and Dynamic World (Earth Engine) is the route to "what is the land
cover *now*" (§B.7).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.catalog._products import WORLDCOVER_CLASSES
from easysnowdata.land import _common

__all__ = ["PRODUCT_ID", "VERSIONS", "COLLECTION", "search", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "esa-worldcover"
COLLECTION = "esa-worldcover"
#: WorldCover version → the year it maps.
VERSIONS: dict[str, str] = {"v100": "2020", "v200": "2021"}
#: The source sentinel; WorldCover writes 0 outside its footprint.
NODATA = 0
AWS_BUCKET_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
GRID_URL = f"{AWS_BUCKET_URL}/esa_worldcover_grid.geojson"
_PC_STAC = providers.stac.PLANETARY_COMPUTER_URL


def _year(version: str) -> str:
    try:
        return VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"Incorrect version number. Please provide one of {list(VERSIONS)}."
        ) from None


def tile_url(tile: str, version: str) -> str:
    """URL of one 3°×3° map tile in the public AWS bucket (``tile`` is e.g. ``N45W123``)."""
    year = _year(version)
    return f"{AWS_BUCKET_URL}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"


def _tiles(aoi: Any, version: str) -> gpd.GeoDataFrame:
    """The bucket's own tile grid, cached on first use, filtered to *aoi*."""
    path = providers.raster_http.fetch(
        GRID_URL, "esa_worldcover_grid.geojson", subdir="worldcover", progressbar=False
    )
    grid = providers.vector_http.read(path, aoi)
    grid = grid.rename(columns={"ll_tile": "tile"})
    grid["url"] = [tile_url(tile, version) for tile in grid["tile"]]
    return grid


def _version_of(item: Any) -> str | None:
    """The WorldCover version of a STAC item (a pystac Item or an item dict)."""
    if hasattr(item, "properties"):
        props = item.properties
    else:
        props = item.get("properties", item)
    version = props.get("esa_worldcover:product_version")
    if version:  # "1.0.0" / "2.0.0"
        return f"v{version.split('.')[0]}00"
    stamp = props.get("start_datetime") or props.get("datetime")
    if stamp:
        year = str(pd.Timestamp(stamp).year)
        for name, mapped in VERSIONS.items():
            if mapped == year:
                return name
    return None


def search(
    aoi: Any = None,
    *,
    source: str | None = None,
    version: str = "v200",
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Return the WorldCover tiles covering *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        ``"planetary-computer"`` (default) or ``"aws-open-data"``.
    version
        ``"v200"`` (2021) or ``"v100"`` (2020).
    **kwargs
        Passed to :func:`easysnowdata.providers.stac.search` (STAC route only).

    Returns
    -------
    geopandas.GeoDataFrame
        STAC items for the Planetary Computer route, or the bucket's tile grid
        (``tile``, ``url``) for the AWS route. Either is accepted by :func:`load`.
    """
    _, src = _common.resolve(PRODUCT_ID, source)
    _year(version)
    if src.provider != "stac":
        return _tiles(aoi, version)
    items = providers.stac.search(src.id, COLLECTION, aoi, **kwargs)
    keep = [item for item in items if _version_of(item) == version]
    return providers.stac.items_to_geodataframe(keep)


def _load_stac(items, aoi, src, version, crs, grid_resolution, chunks, kwargs):
    if not isinstance(items, gpd.GeoDataFrame):
        items = providers.stac.items_to_geodataframe(items)
    if "stac_item" in items.columns and len(items):
        versions = items["stac_item"].map(_version_of)
        if versions.notna().any():  # keep items whose version the catalog omits
            items = items[versions.isna() | (versions == version)]
    if not len(items):
        raise ValueError(
            f"No ESA WorldCover {version} tiles cover this AOI on {src.title}."
        )
    ds = providers.stac.load(
        items,
        aoi,
        bands="map",
        crs=crs,
        resolution=grid_resolution,
        chunks=chunks,
        groupby="time",
        catalog=src.id,
        **kwargs,
    )
    da = ds["map"]
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.squeeze("time")  # static product: keep the year as a scalar coord
    return da


def _load_tiles(tiles, aoi, version, chunks, kwargs):
    if isinstance(tiles, gpd.GeoDataFrame):
        urls = (
            list(tiles["url"])
            if "url" in tiles
            else [tile_url(t, version) for t in tiles["tile"]]
        )
    else:
        urls = [
            t if str(t).startswith("http") else tile_url(str(t), version) for t in tiles
        ]
    if not urls:
        raise ValueError(
            f"No ESA WorldCover {version} tiles cover this AOI in the AWS bucket "
            "(the grid covers land only)."
        )
    parts = [
        providers.raster_http.open(
            url, aoi, chunks=True if chunks is None else chunks, **kwargs
        )
        for url in urls
    ]
    if len(parts) == 1:
        return parts[0]
    merged = xr.combine_by_coords(
        [part.to_dataset(name="map") for part in parts], combine_attrs="drop_conflicts"
    )["map"]
    return merged.rio.write_crs(parts[0].rio.crs)


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    version: str = "v200",
    items: Any = None,
    crs: Any = None,
    grid_resolution: float | None = None,
    chunks: Any = _common.DEFAULT,
    mask: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Load ESA WorldCover land cover for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        ``"planetary-computer"`` (default) or ``"aws-open-data"`` (unsigned
        COGs read straight from the bucket, no STAC).
    version
        ``"v200"`` (2021) or ``"v100"`` (2020).
    items
        Tiles from :func:`search`; when given, no search is made.
    crs, grid_resolution
        Output grid for the STAC route (``crs="utm"`` picks the AOI's zone).
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        ``False`` (default, §2.5 for categorical products) keeps the 0
        sentinel with ``rio.nodata`` set; ``True`` masks it to NaN and keeps it
        as ``rio.encoded_nodata``.
    **kwargs
        Passed to ``odc.stac.load`` (STAC) or ``rioxarray.open_rasterio`` (AWS).

    Returns
    -------
    xarray.DataArray
        ``landcover``, uint8 class values carrying CF ``flag_values`` /
        ``flag_meanings`` / ``flag_colors`` for
        :func:`easysnowdata.plotting.categorical`.
    """
    product, src = _common.resolve(PRODUCT_ID, source)
    year = _year(version)
    eager = chunks is None
    load_chunks = None if chunks is _common.DEFAULT else chunks
    if src.provider == "stac":
        if items is None:
            items = providers.stac.search(src.id, COLLECTION, aoi)
        da = _load_stac(
            items, aoi, src, version, crs, grid_resolution, load_chunks, kwargs
        )
    else:
        if items is None:
            items = _tiles(aoi, version)
        da = _load_tiles(items, aoi, version, load_chunks, kwargs)
        da = da.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    da = _common.standardize(da)
    da = _common.apply_nodata(da, NODATA, mask=mask)
    da = da.rename("landcover")
    da.attrs.update(
        _common.attrs_for(
            product,
            src,
            source_url=(
                f"{_PC_STAC}/collections/{COLLECTION}"
                if src.provider == "stac"
                else f"{AWS_BUCKET_URL}/{version}/{year}/map"
            ),
            version=version,
            year=year,
        )
    )
    da = _common.set_variable_flags(da, product, "landcover")
    if eager:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="land",
    title="ESA WorldCover land cover",
    description=(
        "Global 10 m land cover for 2020 (v100) and 2021 (v200), 11 classes from "
        "the FAO Land Cover Classification System. The project ended at v200; "
        "Copernicus LCFM on CDSE is the successor."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="esa-worldcover",
            resolution_m=10,
            temporal="2020, 2021",
            notes="signed hrefs, anonymous; both versions in one collection",
            title="Planetary Computer",
            health=Probe(
                "ESA WorldCover (Planetary Computer)",
                partial(health.stac_search, _PC_STAC, COLLECTION, sign=True),
            ),
        ),
        Source(
            id="aws-open-data",
            provider="raster_http",
            location=f"{AWS_BUCKET_URL}/<version>/<year>/map",
            resolution_m=10,
            temporal="2020, 2021",
            notes=(
                "public eu-central-1 bucket, no signing and no STAC; the tile "
                "grid (esa_worldcover_grid.geojson) is cached on first use"
            ),
            title="AWS Open Data bucket",
            health=Probe(
                "ESA WorldCover (AWS bucket)",
                partial(
                    health.http_first_byte,
                    f"{AWS_BUCKET_URL}/v200/2021/map/ESA_WorldCover_10m_2021_v200_N45W123_Map.tif",
                ),
            ),
        ),
    ),
    variables=(
        Variable(
            "landcover",
            dtype="uint8",
            nodata=0,
            long_name="land cover class",
            flag_values=tuple(WORLDCOVER_CLASSES),
            flag_meanings=tuple(name for name, _ in WORLDCOVER_CLASSES.values()),
            flag_colors=tuple(color for _, color in WORLDCOVER_CLASSES.values()),
        ),
    ),
    citation="Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.",
    license="CC BY 4.0",
    doi="10.5281/zenodo.7254221",
    loader="easysnowdata.land.landcover.load",
    references=(
        "https://esa-worldcover.org/",
        "https://registry.opendata.aws/esa-worldcover-vito/",
    ),
    tags=("land cover", "worldcover"),
)

catalog.register(PRODUCT, replace=True)
