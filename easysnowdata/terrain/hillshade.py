"""Natural Earth global shaded relief — a hillshade basemap layer.

::

    import easysnowdata as esd

    aoi = (-123.0, 46.0, -120.5, 48.0)
    hillshade_da = esd.terrain.hillshade.load(aoi)                  # 1:10m, gray earth
    plain_da = esd.terrain.hillshade.load(aoi, style="shaded-relief")
    world_da = esd.terrain.hillshade.load(scale="50m")               # whole globe

The Natural Earth raster collection (Patterson & Kelso) ships global
grayscale shaded relief in several styles, from a plain hillshade to the
"Gray Earth" family that adds hypsometric tints, an ocean-bottom relief and
drainages. They are 8-bit, one band, in geographic coordinates (EPSG:4326):
1 arcmin (~1.85 km) at the 1:10m scale and 2 arcmin at 1:50m. Brightness is
a cartographic shading, not a physical quantity, so there is no nodata value
and nothing is masked; use the layer under other data, not in analysis.

This is the layer ``download_and_preprocess_hillshade.ipynb`` in the global
snowmelt runoff onset project draws its maps on. That notebook reprojects it
to Robinson; this loader returns the native grid, because ``odc.reproject``
onto whatever the map uses is one line (see the gallery example).

**Each style is one zip on the Natural Earth S3 bucket** (18-92 MB). It is
fetched once into the easysnowdata cache and read in place, which clips an
AOI in a few seconds; ``cache=False`` reads it over HTTPS every call instead,
which is only sensible for a one-off small AOI.

For a hillshade at DEM resolution, shade a DEM from :mod:`~easysnowdata.terrain.dem`
instead; this layer is for context maps from a basin up to the globe.
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

__all__ = ["PRODUCT_ID", "NATURAL_EARTH_RASTERS", "STYLES", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "hillshade"
NATURAL_EARTH_RASTERS = "https://naturalearth.s3.amazonaws.com"
#: style → (description, {scale: archive stem}). The stem names both the zip and
#: the GeoTIFF inside it; the ocean-bottom-and-drainages style exists at 1:10m only.
STYLES: dict[str, tuple[str, dict[str, str]]] = {
    "gray-earth-ocean-drainages": (
        "Gray Earth with shaded relief, hypsography, ocean bottom and drainages",
        {"10m": "GRAY_HR_SR_OB_DR"},
    ),
    "gray-earth-ocean": (
        "Gray Earth with shaded relief, hypsography and ocean bottom",
        {"10m": "GRAY_HR_SR_OB", "50m": "GRAY_50M_SR_OB"},
    ),
    "gray-earth-water": (
        "Gray Earth with shaded relief, hypsography and flat water",
        {"10m": "GRAY_HR_SR_W", "50m": "GRAY_50M_SR_W"},
    ),
    "gray-earth": (
        "Gray Earth with shaded relief and hypsography (land only)",
        {"10m": "GRAY_HR_SR", "50m": "GRAY_50M_SR"},
    ),
    "shaded-relief": (
        "plain shaded relief",
        {"10m": "SR_HR", "50m": "SR_50M"},
    ),
}
#: Natural Earth's scale names (1:10 million, 1:50 million) → grid spacing.
SCALES: dict[str, float] = {"10m": 1 / 60, "50m": 1 / 30}


def _stem(style: str, scale: str) -> str:
    if style not in STYLES:
        raise ValueError(
            f"Invalid style {style!r}. Choose from {', '.join(repr(k) for k in STYLES)}."
        )
    if scale not in SCALES:
        raise ValueError(
            f"Invalid scale {scale!r}. Choose from {', '.join(repr(k) for k in SCALES)} "
            "(Natural Earth's 1:10 million and 1:50 million)."
        )
    stems = STYLES[style][1]
    if scale not in stems:
        raise ValueError(
            f"Style {style!r} exists at scale {', '.join(repr(k) for k in stems)} only."
        )
    return stems[scale]


def url(style: str = "gray-earth-ocean-drainages", scale: str = "10m") -> str:
    """The Natural Earth zip holding *style* at *scale*."""
    return f"{NATURAL_EARTH_RASTERS}/{scale}_raster/{_stem(style, scale)}.zip"


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    style: str = "gray-earth-ocean-drainages",
    scale: str = "10m",
    cache: bool = True,
    chunks: Any = contract.DEFAULT,
    **kwargs: Any,
) -> xr.DataArray:
    """Load Natural Earth shaded relief for *aoi* (the globe when ``None``).

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` returns
        the whole globe.
    source
        Only ``"natural-earth"`` today.
    style
        ``"gray-earth-ocean-drainages"`` (default; 1:10m only),
        ``"gray-earth-ocean"``, ``"gray-earth-water"``, ``"gray-earth"`` or
        ``"shaded-relief"`` (a plain hillshade). See :data:`STYLES`.
    scale
        ``"10m"`` (default, 1 arcmin) or ``"50m"`` (2 arcmin) — Natural
        Earth's 1:10 million and 1:50 million scales, not metres.
    cache
        ``True`` (default) downloads the zip once into the easysnowdata cache
        and reads it locally; ``False`` reads it over HTTPS every call.
    chunks
        Dask chunks; ``None`` loads eagerly.
    **kwargs
        Passed to ``rioxarray.open_rasterio``.

    Returns
    -------
    xarray.DataArray
        ``hillshade`` as uint8 brightness (0 dark - 255 light) on
        ``latitude``/``longitude`` in EPSG:4326, with no nodata.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    stem = _stem(style, scale)
    archive_url = url(style, scale)
    member = f"{stem}.tif"
    if cache:
        _logger.info("Fetching %s.zip into the easysnowdata cache (once).", stem)
        path = providers.raster_http.fetch(
            archive_url, f"{stem}.zip", subdir="natural_earth"
        )
        target = f"zip://{path}!{member}"
    else:
        target = providers.raster_http.zip_url(archive_url, member)
    hillshade_da = providers.raster_http.open(
        target,
        aoi,
        chunks=True if chunks in (None, contract.DEFAULT) else chunks,
        **kwargs,
    )
    # The GeoTIFFs carry Photoshop TIFF tags and a neutral scale/offset; keep
    # only what describes the data.
    hillshade_da.attrs = {}
    hillshade_da = contract.write_crs(hillshade_da, hillshade_da.rio.crs or "EPSG:4326")
    hillshade_da = hillshade_da.rename("hillshade")
    hillshade_da.attrs.update(
        contract.provenance(
            product,
            src,
            source_url=archive_url,
            style=style,
            scale=f"1:{scale}",
            file=member,
        )
    )
    hillshade_da = hillshade_da.assign_attrs(product.variable("hillshade").cf_attrs())
    hillshade_da.attrs["long_name"] = f"shaded relief brightness ({STYLES[style][0]})"
    if chunks is None:
        hillshade_da = hillshade_da.compute()
    return hillshade_da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="terrain",
    title="Natural Earth global shaded relief (hillshade)",
    description=(
        "Global 8-bit grayscale shaded relief from Natural Earth at 1 or 2 "
        "arcmin, from a plain hillshade to Gray Earth with hypsography, ocean "
        "bottom and drainages; a basemap layer for maps, not an analysis input."
    ),
    sources=(
        Source(
            id="natural-earth",
            provider="raster_http",
            location=f"{NATURAL_EARTH_RASTERS}/<scale>_raster/<style>.zip",
            resolution_m=1850,
            temporal="static",
            notes=(
                "one zipped GeoTIFF per style and scale (18-92 MB), cached on "
                "first use and read in place; EPSG:4326, no nodata"
            ),
            title="Natural Earth S3 bucket",
            health=Probe(
                "Natural Earth hillshade (S3)",
                partial(health.http_first_byte, url()),
            ),
        ),
    ),
    variables=(
        Variable(
            "hillshade",
            units="1",
            dtype="uint8",
            long_name="shaded relief brightness",
        ),
    ),
    citation=(
        "Patterson, T., & Kelso, N. V. Natural Earth raster data: Gray Earth "
        "with shaded relief, hypsography, ocean bottom and drainages. "
        "https://www.naturalearthdata.com/"
    ),
    license="Public domain",
    loader="easysnowdata.terrain.hillshade.load",
    examples=("terrain/plot_hillshade.py",),
    references=(
        "https://www.naturalearthdata.com/downloads/10m-raster-data/10m-gray-earth/",
        "https://www.naturalearthdata.com/downloads/10m-raster-data/10m-shaded-relief/",
    ),
    tags=("hillshade", "shaded relief", "basemap", "natural earth"),
)

catalog.register(PRODUCT, replace=True)
