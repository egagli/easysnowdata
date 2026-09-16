"""Copernicus DEM GLO-30 / GLO-90 (§4.5, companion §B.8).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    items = esd.terrain.dem.search(aoi)                       # GeoDataFrame
    dem = esd.terrain.dem.load(aoi)                           # Planetary Computer
    dem = esd.terrain.dem.load(aoi, source="earth-search")    # unsigned AWS COGs
    dem = esd.terrain.dem.load(aoi, resolution=90, crs="utm", grid_resolution=90)

Both routes serve the 2021 release of the same global digital surface model;
the newer GLO-30 2024_1 release exists only on Earth Engine and CDSE (§B.8).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import geopandas as gpd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.terrain import _common

__all__ = ["PRODUCT_ID", "COLLECTIONS", "search", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "copernicus-dem"
#: DEM resolution in metres → STAC collection id (the same ids on both catalogs).
COLLECTIONS: dict[int, str] = {30: "cop-dem-glo-30", 90: "cop-dem-glo-90"}
#: The sentinel the Copernicus DEM COGs use for sea and missing tiles.
NODATA = -32767.0

_PC_STAC = providers.stac.PLANETARY_COMPUTER_URL
_EARTH_SEARCH_STAC = providers.stac.EARTH_SEARCH_URL


def _collection(resolution: int | float) -> str:
    try:
        return COLLECTIONS[int(resolution)]
    except (KeyError, ValueError):
        raise ValueError(
            f"Copernicus DEM is available at 30 m and 90 m only, got {resolution} m."
        ) from None


def search(
    aoi: Any = None,
    *,
    source: str | None = None,
    resolution: int = 30,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Return the DEM tiles covering *aoi* as a GeoDataFrame of STAC items.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        ``"planetary-computer"`` (default) or ``"earth-search"``.
    resolution
        30 or 90 (metres) — which Copernicus DEM collection to search.
    **kwargs
        Passed to :func:`easysnowdata.providers.stac.search`.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per tile, filterable and accepted back by :func:`load`.
    """
    _, src = _common.resolve(PRODUCT_ID, source)
    items = providers.stac.search(src.id, _collection(resolution), aoi, **kwargs)
    return providers.stac.items_to_geodataframe(items)


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    resolution: int = 30,
    items: Any = None,
    crs: Any = None,
    grid_resolution: float | None = None,
    chunks: Any = _common.DEFAULT,
    mask: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Load Copernicus DEM elevation for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. ``clip=False`` on
        an :class:`~easysnowdata.aoi.AOI` returns the covering tiles whole.
    source
        ``"planetary-computer"`` (default, signed) or ``"earth-search"``
        (unsigned AWS Open Data COGs). Both carry the 2021 release.
    resolution
        30 or 90 — the DEM product, not the output grid.
    items
        Tiles from :func:`search` (a GeoDataFrame or ``ItemCollection``); when
        given, no search is made.
    crs, grid_resolution
        Output grid. ``crs="utm"`` reprojects to the AOI's UTM zone; the
        native grid (EPSG:4326) is kept when both are ``None``.
    chunks
        Dask chunks; the default is the source's native chunking and ``None``
        loads eagerly.
    mask
        ``True`` (default, §2.5 for continuous products) replaces the -32767
        sentinel with NaN and keeps it as ``rio.encoded_nodata``; ``False``
        keeps the sentinel with ``rio.nodata`` set.
    **kwargs
        Passed to ``odc.stac.load``.

    Returns
    -------
    xarray.DataArray
        ``elevation`` in metres, dims ``latitude``/``longitude`` (or ``y``/``x``
        when reprojected), lazy unless ``chunks=None``.
    """
    product, src = _common.resolve(PRODUCT_ID, source)
    collection = _collection(resolution)
    if items is None:
        items = providers.stac.search(src.id, collection, aoi)
    if len(items) == 0:
        raise ValueError(
            f"No {collection} items cover this AOI on {src.title}. The Copernicus "
            "DEM covers land between 90°N and 90°S; ocean-only boxes have no tiles."
        )
    eager = chunks is None
    ds = providers.stac.load(
        items,
        aoi,
        bands="data",
        crs=crs,
        resolution=grid_resolution,
        chunks=None if chunks is _common.DEFAULT else chunks,
        groupby="time",  # every tile of a release shares one datetime
        catalog=src.id,
        **kwargs,
    )
    da = ds["data"]
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.squeeze("time")  # static product: keep the date as a scalar coord
    da = _common.standardize(da)
    da = _common.apply_nodata(da, NODATA, mask=mask)
    da = da.rename("elevation")
    da.attrs.update(
        _common.attrs_for(
            product,
            src,
            source_url=f"{providers.stac.CATALOGS[src.id]['url']}/collections/{collection}",
            long_name="elevation above the EGM2008 geoid",
            units="m",
            collection=collection,
            resolution_m=int(resolution),
        )
    )
    if eager:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="terrain",
    title="Copernicus DEM GLO-30 / GLO-90",
    description=(
        "Global 30 m and 90 m digital surface model from the Copernicus DEM "
        "(WorldDEM heritage), 2021 release. Elevations are referenced to the "
        "EGM2008 geoid."
    ),
    sources=(
        Source(
            id="planetary-computer",
            provider="stac",
            location="cop-dem-glo-30 / cop-dem-glo-90",
            resolution_m=30,
            temporal="static (2021 release)",
            notes=(
                "signed hrefs, anonymous; the 2024_1 release is only on GEE and "
                "CDSE (§B.8)"
            ),
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
            temporal="static (2021 release)",
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
    ),
    variables=(
        Variable(
            "elevation",
            units="m",
            dtype="float32",
            nodata=-32767,
            long_name="elevation above the EGM2008 geoid",
        ),
    ),
    citation=(
        "European Space Agency, Sinergise (2021). Copernicus Global Digital "
        "Elevation Model. Distributed by OpenTopography."
    ),
    license="Copernicus DEM licence (free, attribution)",
    doi="10.5069/G9028PQB",
    loader="easysnowdata.terrain.dem.load",
    references=(
        "https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30",
        "https://registry.opendata.aws/copernicus-dem/",
    ),
    tags=("dem", "elevation", "terrain"),
)

catalog.register(PRODUCT, replace=True)
