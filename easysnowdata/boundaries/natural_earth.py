"""Any Natural Earth vector layer: lakes, rivers, coastline, glaciated areas, places.

::

    import easysnowdata as esd

    aoi = (-123.0, 46.0, -120.5, 48.0)
    lakes_gdf = esd.boundaries.natural_earth.load(aoi)                  # layer="lakes"
    rivers_gdf = esd.boundaries.natural_earth.load(aoi, layer="rivers_lake_centerlines")
    ice_gdf = esd.boundaries.natural_earth.load(aoi, layer="glaciated_areas")
    places_gdf = esd.boundaries.natural_earth.load(aoi, layer="populated_places_simple")
    coast_gdf = esd.boundaries.natural_earth.load(layer="coastline", scale="110m")

Natural Earth publishes some 130 public-domain cartographic layers at three
scales (1:10m, 1:50m, 1:110m), each a zipped shapefile named
``ne_<scale>_<layer>.zip`` under a ``cultural`` or ``physical`` folder. This
module reads any of them by name; :data:`LAYERS` lists the common ones and
the category is looked up there (pass ``category=`` for anything else). The
countries and states in :mod:`~easysnowdata.boundaries.admin` come from the
same archive, with normalized columns on top. The point-of-view editions of
the countries layer (``layer="admin_0_countries_ind"``, ``_usa``, ``_chn`` …,
1:10m) draw disputed borders as that country does.

These are map layers, generalized for display at their scale: use them for
outlines and labels, not for measurements. For glacier outlines to analyse,
use :mod:`~easysnowdata.boundaries.glaciers` (RGI); for rivers and lakes as
hydrography, HydroRIVERS and HydroLAKES are the analysis-grade route.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import geopandas as gpd

from easysnowdata import catalog
from easysnowdata.boundaries.admin import (
    NATURAL_EARTH_SCALES,
    NATURAL_EARTH_URL,
    _read_zip,
    natural_earth_url,
)
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source
from easysnowdata.processing import contract

__all__ = ["PRODUCT_ID", "LAYERS", "load", "url"]

PRODUCT_ID = "natural-earth-vectors"
#: Common layers → (category, scales available).
LAYERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "admin_0_countries": ("cultural", ("10m", "50m", "110m")),
    "admin_0_boundary_lines_land": ("cultural", ("10m", "50m", "110m")),
    "admin_1_states_provinces": ("cultural", ("10m", "50m")),
    "admin_1_states_provinces_lines": ("cultural", ("10m", "50m", "110m")),
    "populated_places_simple": ("cultural", ("10m", "50m", "110m")),
    "urban_areas": ("cultural", ("10m", "50m")),
    "roads": ("cultural", ("10m",)),
    "coastline": ("physical", ("10m", "50m", "110m")),
    "land": ("physical", ("10m", "50m", "110m")),
    "ocean": ("physical", ("10m", "50m", "110m")),
    "lakes": ("physical", ("10m", "50m", "110m")),
    "rivers_lake_centerlines": ("physical", ("10m", "50m", "110m")),
    "glaciated_areas": ("physical", ("10m", "50m", "110m")),
    "antarctic_ice_shelves_polys": ("physical", ("10m", "50m")),
    "geography_regions_polys": ("physical", ("10m", "50m", "110m")),
    "geography_regions_elevation_points": ("physical", ("10m", "50m", "110m")),
}


def url(layer: str, scale: str = "10m", category: str | None = None) -> str:
    """The Natural Earth zip for *layer* at *scale*."""
    if category is None and layer.startswith("admin_0_countries_"):
        # The point-of-view editions (admin_0_countries_ind, _usa, _chn …):
        # each country's own view of disputed borders, at 1:10m.
        category = "cultural"
    if category is None:
        if layer not in LAYERS:
            raise ValueError(
                f"Unknown layer {layer!r}: pass category='cultural' or 'physical' "
                f"for layers outside LAYERS ({', '.join(LAYERS)})."
            )
        category, scales = LAYERS[layer]
        if scale not in scales:
            raise ValueError(
                f"{layer!r} exists at {', '.join(repr(s) for s in scales)} only."
            )
    return natural_earth_url(layer, scale, category)


def load(
    aoi: Any = None,
    *,
    layer: str = "lakes",
    scale: str = "10m",
    category: str | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read the Natural Earth *layer* for *aoi* (the world when ``None``).

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; features that
        intersect it are returned whole.
    layer
        The layer name without the ``ne_<scale>_`` prefix: ``"lakes"``
        (default), ``"rivers_lake_centerlines"``, ``"glaciated_areas"`` …
        (see :data:`LAYERS`).
    scale
        ``"10m"`` (default), ``"50m"`` or ``"110m"``.
    category
        ``"cultural"`` or ``"physical"``; only needed outside :data:`LAYERS`.
    source
        Only ``"natural-earth"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        The layer's own columns, in EPSG:4326.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    if scale not in NATURAL_EARTH_SCALES:
        raise ValueError(
            f"Invalid scale {scale!r}. Choose from "
            f"{', '.join(repr(s) for s in NATURAL_EARTH_SCALES)}."
        )
    layer_url = url(layer, scale, category)
    layer_gdf = _read_zip(layer_url, aoi, subdir="natural_earth", **kwargs)
    return contract.finalize_frame(
        layer_gdf, product, src, source_url=layer_url, layer=layer, scale=f"1:{scale}"
    )


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="boundaries",
    title="Natural Earth vector layers",
    description=(
        "Public-domain cartographic vectors at 1:10m, 1:50m and 1:110m: lakes, "
        "rivers, coastline, land and ocean, glaciated areas, populated places, "
        "boundary lines and more, read by layer name."
    ),
    sources=(
        Source(
            id="natural-earth",
            provider="vector_http",
            location=f"{NATURAL_EARTH_URL}/<scale>/<category>/ne_<scale>_<layer>.zip",
            temporal="static (v5.1)",
            notes="one zipped shapefile per layer and scale, cached on first use",
            title="Natural Earth",
            health=Probe(
                "Natural Earth lakes",
                partial(health.http_first_byte, url("lakes", "110m")),
            ),
        ),
    ),
    citation="Natural Earth. Free vector and raster map data @ naturalearthdata.com.",
    license="Public domain",
    loader="easysnowdata.boundaries.natural_earth.load",
    examples=("boundaries/plot_natural_earth.py",),
    references=("https://www.naturalearthdata.com/downloads/",),
    tags=("natural earth", "lakes", "rivers", "coastline", "basemap"),
)

catalog.register(PRODUCT, replace=True)
