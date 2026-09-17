"""Wrzesien global seasonal mountain snow mask (§4.3).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    mask = esd.snow.mountain_snow_mask.load(aoi)                   # mountains
    snow = esd.snow.mountain_snow_mask.load(aoi, layer="snow")     # all terrain
    cloud = esd.snow.mountain_snow_mask.load(aoi, layer="clouds")  # indeterminacy
    esd.plotting.categorical(mask)

MODIS MOD10A2 snow-cover extent and the GTOPO30 elevation model, classified at
30 arcsec into little-to-no, ephemeral and seasonal snow, with mountains
separated from all terrain (Wrzesien et al. 2019).

**The archive is one zipped GeoTIFF per layer on Zenodo.** Each is fetched once
into the easysnowdata cache (~14-26 MB) instead of being re-read over HTTPS on
every call, which is what made the old function slow; ``cache=False`` restores
the direct ``zip+https://`` read.

**Upstream fill values.** The two class rasters are written as 32-bit with
nodata 256 (and 265 in places) rather than 255, so unmasked they arrive as
uint32. :func:`easysnowdata.processing.snow.repair_fill_values` puts every
value above the last class back to 255 and returns uint8; the loader applies it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract
from easysnowdata.processing.snow import repair_fill_values

__all__ = ["PRODUCT_ID", "LAYERS", "ZENODO_FILES", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "mountain-snow-mask"
ZENODO_FILES = "https://zenodo.org/records/2626737/files"
#: Layer → (zip file, member GeoTIFF). The clouds member is spelled without an underscore.
LAYERS: dict[str, tuple[str, str]] = {
    "mountain_snow": ("MODIS_mtnsnow_classes.zip", "MODIS_mtnsnow_classes.tif"),
    "snow": ("MODIS_snow_classes.zip", "MODIS_snow_classes.tif"),
    "clouds": ("MODIS_clouds.zip", "MODISclouds.tif"),
}
#: The mountain classes and the all-terrain classes: value → (name, colour).
MOUNTAIN_SNOW_CLASSES: dict[int, tuple[str, str]] = {
    0: ("Mountains with little-to-no snow", "#030303"),
    1: ("Indeterminate due to clouds", "#755F4A"),
    2: ("Mountains with ephemeral snow", "#792B8E"),
    3: ("Mountains with seasonal snow", "#679ACF"),
    255: ("Fill", "#ffffff"),
}
SNOW_CLASSES: dict[int, tuple[str, str]] = {
    0: ("Little-to-no snow", "#030303"),
    1: ("Indeterminate due to clouds", "#755F4A"),
    2: ("Ephemeral snow", "#792B8E"),
    3: ("Seasonal snow", "#679ACF"),
    255: ("Fill", "#ffffff"),
}
#: The class table's "Fill" value, and the highest real class in the two masks.
NODATA = 255
LAST_CLASS = 3
#: The years the MOD10A2 climatology covers.
PERIOD = "2000-2016"


def _layer(name: str) -> tuple[str, str]:
    try:
        return LAYERS[name]
    except KeyError:
        raise ValueError(
            f"Invalid layer {name!r}. Choose from {', '.join(repr(k) for k in LAYERS)}."
        ) from None


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    layer: str = "mountain_snow",
    cache: bool = True,
    chunks: Any = contract.DEFAULT,
    mask: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Load one layer of the mountain snow mask for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        Only ``"zenodo"`` today.
    layer
        ``"mountain_snow"`` (default), ``"snow"`` or ``"clouds"``. The first
        two are categorical; ``"clouds"`` holds the cloud-indeterminacy level
        0-6, which the upstream record does not define further.
    cache
        ``True`` (default) downloads the zip once into the easysnowdata cache
        and reads it locally; ``False`` reads it over HTTPS every call.
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        ``False`` (default, §2.5 for categorical products) keeps the 255 (Fill)
        sentinel with ``rio.nodata`` set; ``True`` masks it to NaN.
    **kwargs
        Passed to ``rioxarray.open_rasterio``.

    Returns
    -------
    xarray.DataArray
        The layer as uint8, with CF flag attributes on the two class layers.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    archive, member = _layer(layer)
    url = f"{ZENODO_FILES}/{archive}"
    if cache:
        _logger.info("Fetching %s into the easysnowdata cache (once).", archive)
        path = providers.raster_http.fetch(url, archive, subdir="mountain_snow_mask")
        target = f"zip://{path}!{member}"
    else:
        target = providers.raster_http.zip_url(url, member)
    da = providers.raster_http.open(
        target,
        aoi,
        chunks=True if chunks in (None, contract.DEFAULT) else chunks,
        **kwargs,
    )
    if layer != "clouds":
        da = repair_fill_values(da, last_class=LAST_CLASS, fill=NODATA)
    da = contract.write_crs(da, da.rio.crs)
    da = (
        contract.mask_continuous(da, NODATA)
        if mask
        else contract.set_categorical_nodata(da, NODATA)
    )
    da = da.rename(layer)
    da.attrs.update(
        contract.provenance(
            product,
            src,
            source_url=url,
            layer=layer,
            file=member,
            period=PERIOD,
            long_name=(
                "cloud indeterminacy level (0-6, undefined upstream)"
                if layer == "clouds"
                else None
            ),
            units="1" if layer == "clouds" else None,
        )
    )
    if layer != "clouds":
        da = da.assign_attrs(product.variable(layer).cf_attrs())
    if chunks is None:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="snow",
    title="Wrzesien global seasonal mountain snow mask",
    description=(
        "MODIS MOD10A2-derived masks of mountains with seasonal, ephemeral or "
        "little snow (Wrzesien et al. 2019) at 30 arcsec, the same classes over "
        "all terrain, and the cloud-indeterminacy layer."
    ),
    sources=(
        Source(
            id="zenodo",
            provider="raster_http",
            location=f"{ZENODO_FILES}/MODIS_<layer>_classes.zip",
            resolution_m=500,
            temporal=f"static ({PERIOD} climatology)",
            notes=(
                "one zipped GeoTIFF per layer, cached on first use; the "
                "published nodata is 256/265, which the loader repairs to 255"
            ),
            title="Zenodo zip",
            health=Probe(
                "Mountain snow mask (Zenodo)",
                partial(
                    health.http_first_byte, f"{ZENODO_FILES}/MODIS_mtnsnow_classes.zip"
                ),
            ),
        ),
    ),
    variables=(
        Variable(
            "mountain_snow",
            dtype="uint8",
            nodata=255,
            long_name="mountain snow class",
            flag_values=tuple(MOUNTAIN_SNOW_CLASSES),
            flag_meanings=tuple(n for n, _ in MOUNTAIN_SNOW_CLASSES.values()),
            flag_colors=tuple(c for _, c in MOUNTAIN_SNOW_CLASSES.values()),
        ),
        Variable(
            "snow",
            dtype="uint8",
            nodata=255,
            long_name="snow class",
            flag_values=tuple(SNOW_CLASSES),
            flag_meanings=tuple(n for n, _ in SNOW_CLASSES.values()),
            flag_colors=tuple(c for _, c in SNOW_CLASSES.values()),
        ),
        Variable(
            "clouds",
            units="1",
            dtype="uint8",
            nodata=255,
            long_name="cloud indeterminacy level (0-6, undefined upstream)",
        ),
    ),
    citation=(
        "Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. "
        "(2019). Global Seasonal Mountain Snow Mask from MODIS MOD10A2. Zenodo."
    ),
    license="CC BY 4.0",
    doi="10.5281/zenodo.2626737",
    loader="easysnowdata.snow.mountain_snow_mask.load",
    examples=("snow/plot_mountain_snow_mask.py",),
    references=("https://doi.org/10.1029/2019GL082649",),
    tags=("snow mask", "mountains", "modis"),
)

catalog.register(PRODUCT, replace=True)
