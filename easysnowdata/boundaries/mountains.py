"""GMBA Mountain Inventory v2: named mountain-range polygons worldwide.

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    ranges_gdf = esd.boundaries.mountains.load(aoi)                  # smallest units
    major_gdf = esd.boundaries.mountains.load(aoi, subset="300")     # major systems
    cascades_gdf = esd.boundaries.mountains.load(aoi, subset="all", level=4)
    broad_gdf = esd.boundaries.mountains.load(aoi, extent="broad")

The Global Mountain Biodiversity Assessment inventory (Snethlage et al. 2022)
delineates 8,327 mountain ranges as a hierarchy up to ten levels deep
(North America > American Cordillera > Pacific Coast Ranges > Cascade Range >
South Washington Cascades > …). It comes in three subsets:

``"basic"`` (default)
    6,717 non-overlapping polygons, the smallest unit wherever there is one
    (around Rainier: the Mount Rainier Massif and its foothills).
``"300"``
    291 non-overlapping major systems (the Cascade Range).
``"all"``
    every range at every level, overlapping; ``level=`` picks one level.

and two extents: ``"standard"`` follows the GMBA v2 mountain definition,
``"broad"`` extends each polygon well into the surrounding terrain (meant to be
intersected with another mountain definition, such as the Wrzesien mask).

Each variant is one zipped shapefile on EarthEnv (22-186 MB), fetched once
into the easysnowdata cache.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = ["PRODUCT_ID", "GMBA_URL", "SUBSETS", "EXTENTS", "load", "url"]

PRODUCT_ID = "gmba-mountains"
GMBA_URL = "https://data.earthenv.org/mountains"
#: subset → the file-name suffix.
SUBSETS: dict[str, str] = {"basic": "_basic", "300": "_300", "all": ""}
EXTENTS = ("standard", "broad")


def url(subset: str = "basic", extent: str = "standard") -> str:
    """The EarthEnv zip for *subset* and *extent*."""
    if subset not in SUBSETS:
        raise ValueError(
            f"Invalid subset {subset!r}. Choose from {', '.join(repr(s) for s in SUBSETS)}."
        )
    if extent not in EXTENTS:
        raise ValueError(
            f"Invalid extent {extent!r}. Choose from {', '.join(repr(e) for e in EXTENTS)}."
        )
    return f"{GMBA_URL}/{extent}/GMBA_Inventory_v2.0_{extent}{SUBSETS[subset]}.zip"


def load(
    aoi: Any = None,
    *,
    subset: str = "basic",
    extent: str = "standard",
    level: int | None = None,
    name: str | list[str] | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """GMBA mountain ranges intersecting *aoi* (the world when ``None``).

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    subset
        ``"basic"`` (default, smallest non-overlapping units), ``"300"``
        (major systems) or ``"all"`` (every level, overlapping).
    extent
        ``"standard"`` (default) or ``"broad"``.
    level
        Keep only ranges at this hierarchy level (1 = continent … 10);
        most useful with ``subset="all"``.
    name
        Keep only ranges with this name (``"Cascade Range"``), case-insensitive.
    source
        Only ``"earthenv"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``name`` (``MapName``), ``gmba_id`` (``GMBA_V2_ID``), ``level``
        (``Hier_Lvl``) and ``path`` (the hierarchy, ``"North America > …"``),
        then the inventory's columns (names in eight languages, countries,
        elevation range, ``Level_01`` … ``Level_10``).
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    zip_url = url(subset, extent)
    member = zip_url.rsplit("/", 1)[-1].replace(".zip", ".shp")
    path = providers.raster_http.fetch(zip_url, subdir="boundaries/gmba")
    raw_gdf = providers.vector_http.read(f"zip://{path}!{member}", aoi, **kwargs)
    lead = pd.DataFrame(
        {
            "name": raw_gdf["MapName"],
            "gmba_id": raw_gdf["GMBA_V2_ID"],
            "level": pd.to_numeric(raw_gdf["Hier_Lvl"], errors="coerce").astype(
                "Int64"
            ),
            "path": raw_gdf["Path"],
        },
        index=raw_gdf.index,
    )
    ranges_gdf = gpd.GeoDataFrame(pd.concat([lead, raw_gdf], axis=1), crs=raw_gdf.crs)
    if level is not None:
        ranges_gdf = ranges_gdf[ranges_gdf["level"] == int(level)]
    if name is not None:
        options = [name] if isinstance(name, str) else list(name)
        wanted = [str(o).casefold() for o in options]
        ranges_gdf = ranges_gdf[
            ranges_gdf["name"].astype(str).str.casefold().isin(wanted)
        ]
    return contract.finalize_frame(
        ranges_gdf.reset_index(drop=True),
        product,
        src,
        source_url=zip_url,
        subset=subset,
        extent=extent,
        level=level,
    )


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="boundaries",
    title="GMBA Mountain Inventory v2",
    description=(
        "8,327 named mountain-range polygons in a hierarchy up to ten levels "
        "deep (Snethlage et al. 2022), in basic, major-system (300) and "
        "all-level subsets and standard or broad extents."
    ),
    sources=(
        Source(
            id="earthenv",
            provider="vector_http",
            location=f"{GMBA_URL}/<extent>/GMBA_Inventory_v2.0_<extent>[_basic|_300].zip",
            temporal="static (v2.0, 2022)",
            notes="one zipped shapefile per subset and extent (22-186 MB), cached on first use",
            title="EarthEnv",
            health=Probe(
                "GMBA mountains (EarthEnv)",
                partial(health.http_first_byte, url("300")),
            ),
        ),
    ),
    variables=(
        Variable("name", long_name="mountain range name"),
        Variable("gmba_id", long_name="GMBA v2 identifier"),
        Variable("level", long_name="hierarchy level (1 = continent)"),
    ),
    citation=(
        "Snethlage, M. A., Geschke, J., Ranipeta, A., Jetz, W., Yoccoz, N. G., "
        "Körner, C., Spehn, E. M., Fischer, M., & Urbach, D. (2022). A hierarchical "
        "inventory of the world's mountains for global comparative mountain "
        "science. Scientific Data 9, 149. Data: GMBA Mountain Inventory v2, "
        "EarthEnv, doi:10.48601/earthenv-t9k2-1407."
    ),
    license="CC BY 4.0",
    doi="10.1038/s41597-022-01256-y",
    loader="easysnowdata.boundaries.mountains.load",
    examples=("boundaries/plot_mountains.py",),
    references=(
        "https://www.earthenv.org/mountains",
        "https://doi.org/10.48601/earthenv-t9k2-1407",
    ),
    tags=("mountains", "mountain ranges", "gmba"),
)

catalog.register(PRODUCT, replace=True)
