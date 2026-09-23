"""Forest cover fraction — Copernicus Global Land Service LC100 (§4.4).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    forest_da = esd.land.forest_cover.load(aoi)                # Zenodo GeoTIFF, 2019
    forest_da = esd.land.forest_cover.load(aoi, source="gee")  # 2015-2019 epochs

The tree-cover fraction layer of CGLS-LC100 collection 3, epoch 2019 — the
last epoch the service produced. The Zenodo GeoTIFF needs no credentials and
is read with range requests; the Earth Engine collection adds the 2015-2018
epochs and the other cover fractions.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers, temporal
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = ["PRODUCT_ID", "ZENODO_URL", "GEE_ASSET", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "forest-cover-fraction"
ZENODO_URL = (
    "https://zenodo.org/record/3939050/files/"
    "PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif"
)
GEE_ASSET = "COPERNICUS/Landcover/100m/Proba-V-C3/Global"
GEE_BAND = "tree-coverfraction"
#: CGLS-LC100 writes 255 outside its footprint; cover fractions are 0-100 %.
NODATA = 255
EPOCH = "2019"


def _load_gee(aoi, time, chunks, kwargs):
    ee = providers.gee.ee()
    collection = ee.ImageCollection(GEE_ASSET).select([GEE_BAND])
    keep_time = time is not None
    if keep_time:
        start, end = temporal.parse_time(time)
        collection = collection.filterDate(
            "2015-01-01" if start is None else start.strftime("%Y-%m-%d"),
            (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
    else:
        collection = collection.sort("system:time_start", False).limit(1)
    grid = providers.gee.grid_params(collection.first(), aoi)
    params: dict[str, Any] = {"grid": grid}
    if chunks is not contract.DEFAULT:
        params["chunks"] = chunks
    ds = providers.gee.open_dataset(collection, aoi, **params, **kwargs)
    names = list(ds.data_vars)
    da = ds[GEE_BAND if GEE_BAND in names else names[0]]
    if "time" in da.dims and (not keep_time or da.sizes["time"] == 1):
        da = da.isel(time=0)
    return contract.write_crs(da, grid["crs"])


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    time: Any = None,
    chunks: Any = contract.DEFAULT,
    mask: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Load the CGLS-LC100 tree-cover fraction for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        ``"zenodo"`` (default, credential-free) or ``"gee"``.
    time
        Earth Engine route only: which epochs to keep (2015-2019). ``None``
        returns the newest as a 2-D map.
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        ``True`` (default, §2.5 for continuous products) masks the 255
        sentinel to NaN and keeps it as ``rio.encoded_nodata``; ``False``
        keeps the uint8 values with ``rio.nodata`` set.
    **kwargs
        Passed to ``rioxarray.open_rasterio`` (Zenodo) or
        ``xarray.open_dataset(engine="ee")`` (Earth Engine).

    Returns
    -------
    xarray.DataArray
        ``tree_cover_fraction`` in percent.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    eager = chunks is None
    if src.provider == "gee":
        da = _load_gee(aoi, time, chunks, kwargs)
    else:
        da = providers.raster_http.open(
            ZENODO_URL,
            aoi,
            chunks=True if chunks in (None, contract.DEFAULT) else chunks,
            **kwargs,
        )
        da = contract.write_crs(da, da.rio.crs)
        da = da.assign_coords(time=pd.Timestamp(f"{EPOCH}-01-01"))
    da = (
        contract.mask_continuous(da, NODATA)
        if mask
        else contract.set_categorical_nodata(da, NODATA)
    )
    da = da.rename("tree_cover_fraction")
    da.attrs.update(
        contract.provenance(
            product,
            src,
            source_url=ZENODO_URL if src.provider != "gee" else src.location,
            long_name="tree cover fraction",
            units="%",
            epoch=EPOCH if src.provider != "gee" else None,
        )
    )
    if eager:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="land",
    title="Forest cover fraction (CGLS-LC100)",
    description=(
        "Copernicus Global Land Service 100 m tree-cover fraction, collection 3. "
        "Epoch 2019 is the last one the service produced; Earth Engine also "
        "serves the 2015-2018 epochs."
    ),
    sources=(
        Source(
            id="zenodo",
            provider="raster_http",
            location=ZENODO_URL,
            resolution_m=100,
            temporal="static (2019)",
            notes="one global GeoTIFF read with range requests; no credentials",
            title="Zenodo GeoTIFF",
            health=Probe(
                "Forest cover fraction (Zenodo)",
                partial(health.http_first_byte, ZENODO_URL),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location=GEE_ASSET,
            requires=("earthengine",),
            resolution_m=100,
            temporal="2015-2019 (annual epochs)",
            notes="adds the earlier epochs and the other LC100 cover fractions",
            title="Earth Engine",
            health=Probe(
                "Forest cover fraction (GEE/CGLS-LC100)",
                partial(health.gee_asset, GEE_ASSET),
            ),
        ),
    ),
    variables=(
        Variable(
            "tree_cover_fraction",
            units="%",
            dtype="uint8",
            nodata=255,
            long_name="tree cover fraction",
        ),
    ),
    citation=(
        "Buchhorn, M., et al. (2020). Copernicus Global Land Service: Land Cover "
        "100m: collection 3: epoch 2019: Globe (V3.0.1). Zenodo."
    ),
    license="CC BY 4.0",
    doi="10.5281/zenodo.3939050",
    loader="easysnowdata.land.forest_cover.load",
    examples=("land/plot_forest_cover.py",),
    references=("https://land.copernicus.eu/global/products/lc",),
    tags=("forest", "land cover", "vegetation"),
)

catalog.register(PRODUCT, replace=True)
