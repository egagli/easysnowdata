"""Countries, states and provinces, US counties, and admin levels anywhere.

::

    import easysnowdata as esd

    aoi = (-123.0, 46.0, -120.5, 48.0)
    world_gdf = esd.boundaries.admin.countries()                    # Natural Earth, 1:110m
    wa_gdf = esd.boundaries.admin.states(aoi)                       # US AOI → Census
    bc_gdf = esd.boundaries.admin.states(country="CAN", name="British Columbia")
    counties_gdf = esd.boundaries.admin.counties(aoi, state="WA")   # Census
    kommuner_gdf = esd.boundaries.admin.admin(country="NOR", level=2)  # geoBoundaries

Four products, one question each: which country, which state or province,
which county, which administrative unit at a given level. Every function
returns a :class:`geopandas.GeoDataFrame` in EPSG:4326 of the units that
*intersect* the AOI (whole units, not cut at its edge; ``None`` means the
world), with provenance in ``.attrs``. The first columns are the same
everywhere: ``name``, ``iso3`` (the country, ISO 3166-1 alpha-3) and
``admin_level`` (0 country, 1 state/province, 2 county…), followed by the
source's own attributes.

**Sources.** Natural Earth (public domain) for countries and for states and
provinces worldwide; the US Census Bureau's cartographic boundary files
(public domain, yearly) for US states and counties; geoBoundaries (gbOpen,
CC BY 4.0 for most countries — each unit carries its own ``boundaryLicense``)
for any administrative level of any country. Each archive is fetched once
into the easysnowdata cache.

**Natural Earth's point of view.** Its countries follow one de facto view of
disputed borders; Natural Earth also publishes per-country point-of-view
variants, which :func:`easysnowdata.boundaries.natural_earth.load` reaches
(``layer="admin_0_countries_ind"``, ``"admin_0_countries_usa"`` …).

**Natural Earth's ISO codes.** ``ISO_A3`` is ``-99`` for France, Norway and a
few others; ``iso3`` is taken from ``ADM0_A3``, which is always set.

A boundary is an AOI: ``esd.hydro.basins.huc(wa_gdf, level=4)``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = [
    "NATURAL_EARTH_URL",
    "CENSUS_URL",
    "GEOBOUNDARIES_API",
    "US_STATES",
    "countries",
    "states",
    "counties",
    "admin",
    "natural_earth_url",
    "census_url",
]

_logger = logging.getLogger(__name__)

NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth"
CENSUS_URL = "https://www2.census.gov/geo/tiger"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen"
#: The newest cartographic boundary release; ``year=`` picks another.
CENSUS_YEAR = 2025
#: Census cartographic resolutions (1:500,000, 1:5,000,000, 1:20,000,000).
CENSUS_RESOLUTIONS = ("500k", "5m", "20m")
#: Natural Earth scales (1:10, 1:50 and 1:110 million).
NATURAL_EARTH_SCALES = ("10m", "50m", "110m")
#: Two-letter postal code → state name, for ``counties(state=...)``.
US_STATES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}  # fmt: skip


# ── shared helpers ────────────────────────────────────────────────────────────


def natural_earth_url(
    layer: str, scale: str = "10m", category: str = "cultural"
) -> str:
    """The Natural Earth zip for *layer* (``"admin_0_countries"``, ``"lakes"``…)."""
    if scale not in NATURAL_EARTH_SCALES:
        raise ValueError(
            f"Invalid scale {scale!r}. Choose from "
            f"{', '.join(repr(s) for s in NATURAL_EARTH_SCALES)} (1:10, 1:50, 1:110 million)."
        )
    return f"{NATURAL_EARTH_URL}/{scale}/{category}/ne_{scale}_{layer}.zip"


def census_url(layer: str, resolution: str = "5m", year: int = CENSUS_YEAR) -> str:
    """The Census cartographic boundary zip for *layer* (``"state"``, ``"county"``)."""
    if resolution not in CENSUS_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution {resolution!r}. Choose from "
            f"{', '.join(repr(r) for r in CENSUS_RESOLUTIONS)}."
        )
    return f"{CENSUS_URL}/GENZ{year}/shp/cb_{year}_us_{layer}_{resolution}.zip"


def _read_zip(url: str, aoi: Any, *, subdir: str, **kwargs: Any) -> gpd.GeoDataFrame:
    """Fetch a zipped shapefile once into the cache and read the AOI from it."""
    path = providers.raster_http.fetch(url, subdir=f"boundaries/{subdir}")
    return providers.vector_http.read(f"zip://{path}", aoi, **kwargs)


def _lead(
    gdf: gpd.GeoDataFrame,
    *,
    name: Any,
    iso3: Any,
    admin_level: int,
) -> gpd.GeoDataFrame:
    """Put the three shared columns first, keeping the source's own after them."""
    lead = pd.DataFrame(
        {"name": name, "iso3": iso3, "admin_level": admin_level}, index=gdf.index
    )
    rest = gdf.drop(columns=[c for c in lead.columns if c in gdf.columns])
    return gpd.GeoDataFrame(pd.concat([lead, rest], axis=1), crs=gdf.crs)


def _matches(values: pd.Series, wanted: Any) -> pd.Series:
    """Case-insensitive membership of *values* in *wanted* (a str or a list)."""
    options = [wanted] if isinstance(wanted, str) else list(wanted)
    return values.astype(str).str.casefold().isin([str(w).casefold() for w in options])


def _filter(
    gdf: gpd.GeoDataFrame, column: str, wanted: Any, what: str
) -> gpd.GeoDataFrame:
    if wanted is None:
        return gdf
    kept_gdf = gdf[_matches(gdf[column], wanted)]
    if not len(kept_gdf):
        _logger.warning("No %s matched %r.", what, wanted)
    return kept_gdf


def _natural_earth_scale(scale: str | None, aoi: Any) -> str:
    if scale is not None:
        return scale
    return "110m" if aoi is None or parse_aoi(aoi).is_global else "10m"


# ── countries ─────────────────────────────────────────────────────────────────


def countries(
    aoi: Any = None,
    *,
    name: str | list[str] | None = None,
    iso3: str | list[str] | None = None,
    scale: str | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Country boundaries intersecting *aoi* (the world when ``None``).

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    name, iso3
        Keep only these countries, by name (``"Norway"``) or ISO 3166-1
        alpha-3 code (``"NOR"``); a string or a list. Case-insensitive.
    scale
        Natural Earth scale: ``"110m"`` (the default for the world),
        ``"50m"`` or ``"10m"`` (the default when an AOI is given).
    source
        ``"natural-earth"`` (default) or ``"geoboundaries"`` (ADM0, one
        country at a time; needs *iso3* or an AOI).
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``name``, ``iso3``, ``admin_level`` (0), then the source's columns.
    """
    product = catalog.get("countries")
    src = resolve_source(product, source)
    if src.id == "geoboundaries":
        return _geoboundaries(aoi, iso3, 0, product, src, name=name)
    scale = _natural_earth_scale(scale, aoi)
    url = natural_earth_url("admin_0_countries", scale)
    raw_gdf = _read_zip(url, aoi, subdir="natural_earth", **kwargs)
    countries_gdf = _lead(
        raw_gdf, name=raw_gdf["NAME"], iso3=raw_gdf["ADM0_A3"], admin_level=0
    )
    countries_gdf = _filter(countries_gdf, "name", name, "country name")
    countries_gdf = _filter(countries_gdf, "iso3", iso3, "ISO3 code")
    return contract.finalize_frame(
        countries_gdf, product, src, source_url=url, scale=f"1:{scale}"
    )


# ── states and provinces ──────────────────────────────────────────────────────


def _countries_in(aoi: Any) -> set[str]:
    """ISO3 codes of the countries the AOI touches (Natural Earth 1:110m)."""
    countries_gdf = countries(aoi, scale="110m")
    return set(countries_gdf["iso3"])


def _default_states_source(aoi: Any, country: Any) -> str:
    """The Census for the United States, Natural Earth everywhere else."""
    if country is not None:
        options = [country] if isinstance(country, str) else list(country)
        us = {str(c).upper() for c in options} <= {"USA", "US"}
        return "us-census" if us else "natural-earth"
    if aoi is None or parse_aoi(aoi).is_global:
        return "natural-earth"
    touched = _countries_in(aoi)
    return "us-census" if touched and touched <= {"USA"} else "natural-earth"


def states(
    aoi: Any = None,
    *,
    country: str | list[str] | None = None,
    name: str | list[str] | None = None,
    scale: str = "10m",
    resolution: str = "5m",
    year: int = CENSUS_YEAR,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """First-level subdivisions (states, provinces, regions) intersecting *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    country
        Keep only these countries (ISO3, ``"CAN"``; a string or a list).
    name
        Keep only these units by name (``"British Columbia"``).
    scale
        Natural Earth scale, ``"10m"`` (default) or ``"50m"``.
    resolution, year
        Census cartographic resolution (``"500k"``, ``"5m"`` default,
        ``"20m"``) and release year.
    source
        ``None`` (default) picks ``"us-census"`` when *country* is the US or
        the AOI lies only in the US, and ``"natural-earth"`` otherwise;
        ``"geoboundaries"`` reads ADM1 for each country in *country* (or
        touched by the AOI).
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``name``, ``iso3``, ``admin_level`` (1), then the source's columns.
    """
    product = catalog.get("states-provinces")
    src = resolve_source(product, source or _default_states_source(aoi, country))
    if src.id == "geoboundaries":
        return _geoboundaries(aoi, country, 1, product, src, name=name)
    if src.id == "us-census":
        url = census_url("state", resolution, year)
        raw_gdf = _read_zip(url, aoi, subdir="census", **kwargs)
        states_gdf = _lead(raw_gdf, name=raw_gdf["NAME"], iso3="USA", admin_level=1)
        extra = {"resolution": f"1:{resolution}", "year": year}
    else:
        if scale == "110m":
            raise ValueError(
                'Natural Earth has states and provinces at "10m" and "50m" only.'
            )
        url = natural_earth_url("admin_1_states_provinces", scale)
        raw_gdf = _read_zip(url, aoi, subdir="natural_earth", **kwargs)
        states_gdf = _lead(
            raw_gdf, name=raw_gdf["name"], iso3=raw_gdf["adm0_a3"], admin_level=1
        )
        extra = {"scale": f"1:{scale}"}
    states_gdf = _filter(states_gdf, "iso3", country, "country")
    states_gdf = _filter(states_gdf, "name", name, "state or province name")
    return contract.finalize_frame(states_gdf, product, src, source_url=url, **extra)


# ── US counties ───────────────────────────────────────────────────────────────


def counties(
    aoi: Any = None,
    *,
    state: str | list[str] | None = None,
    name: str | list[str] | None = None,
    resolution: str = "5m",
    year: int = CENSUS_YEAR,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """US counties (and county equivalents) intersecting *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` returns
        every county.
    state
        Keep only these states, by postal code (``"WA"``) or name.
    name
        Keep only these counties by name (``"Pierce"``).
    resolution, year
        Census cartographic resolution (``"500k"``, ``"5m"`` default,
        ``"20m"``) and release year.
    source
        Only ``"us-census"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``name``, ``iso3`` (``"USA"``), ``admin_level`` (2), then the Census
        columns (``STUSPS``, ``STATE_NAME``, ``GEOID``, ``ALAND`` …).
    """
    product = catalog.get("us-counties")
    src = resolve_source(product, source)
    url = census_url("county", resolution, year)
    raw_gdf = _read_zip(url, aoi, subdir="census", **kwargs)
    counties_gdf = _lead(raw_gdf, name=raw_gdf["NAME"], iso3="USA", admin_level=2)
    if state is not None:
        options = [state] if isinstance(state, str) else list(state)
        codes = {
            code
            for code, full in US_STATES.items()
            for option in options
            if str(option).casefold() in (code.casefold(), full.casefold())
        }
        counties_gdf = _filter(
            counties_gdf, "STUSPS", sorted(codes) or options, "state"
        )
    counties_gdf = _filter(counties_gdf, "name", name, "county name")
    return contract.finalize_frame(
        counties_gdf,
        product,
        src,
        source_url=url,
        resolution=f"1:{resolution}",
        year=year,
    )


# ── geoBoundaries: any level, any country ─────────────────────────────────────


def _geoboundaries_meta(
    iso3: str, level: int, *, timeout: float = 60
) -> dict[str, Any]:
    """The gbOpen API record for one country and level."""
    import requests  # noqa: PLC0415

    url = f"{GEOBOUNDARIES_API}/{iso3.upper()}/ADM{level}/"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404 or not response.text.strip().startswith("{"):
        raise ValueError(
            f"geoBoundaries has no ADM{level} boundaries for {iso3.upper()} "
            f"(gbOpen). Try a lower level."
        )
    response.raise_for_status()
    return response.json()


def _geoboundaries(
    aoi: Any,
    country: Any,
    level: int,
    product: Product,
    src: Source,
    *,
    name: Any = None,
    simplified: bool = False,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    if not 0 <= int(level) <= 5:
        raise ValueError(f"level must be between 0 and 5, got {level}.")
    if country is None:
        if aoi is None or parse_aoi(aoi).is_global:
            raise ValueError(
                "geoBoundaries is per country: pass country= (ISO3, e.g. 'NOR') "
                "or an AOI to infer it from."
            )
        codes = sorted(_countries_in(aoi))
    else:
        codes = [country] if isinstance(country, str) else list(country)
    frames = []
    urls = []
    for code in codes:
        meta = _geoboundaries_meta(code, level)
        url = meta["simplifiedGeometryGeoJSON" if simplified else "gjDownloadURL"]
        # The release commit is part of the URL; keep it in the file name so a
        # new release is a new cache entry rather than a stale hit.
        commit = url.split("/raw/", 1)[-1].split("/", 1)[0]
        fname = f"{commit}-{url.rsplit('/', 1)[-1]}"
        path = providers.raster_http.fetch(
            url, fname, subdir="boundaries/geoboundaries"
        )
        # Full-resolution coastlines (Norway's ADM0) exceed GDAL's default 200 MB
        # limit on one GeoJSON object.
        raw_gdf = providers.vector_http.read(
            path, aoi, gdal={"OGR_GEOJSON_MAX_OBJ_SIZE": "0"}, **kwargs
        )
        raw_gdf["boundaryLicense"] = meta.get("boundaryLicense")
        raw_gdf["boundarySource"] = meta.get("boundarySource")
        frames.append(raw_gdf)
        urls.append(url)
    raw_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    units_gdf = _lead(
        raw_gdf,
        name=raw_gdf["shapeName"],
        iso3=raw_gdf["shapeGroup"],
        admin_level=int(level),
    )
    units_gdf = _filter(units_gdf, "name", name, "unit name")
    return contract.finalize_frame(
        units_gdf, product, src, source_url=" ".join(urls), admin_level=int(level)
    )


def admin(
    aoi: Any = None,
    *,
    country: str | list[str] | None = None,
    level: int = 1,
    name: str | list[str] | None = None,
    simplified: bool = False,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Administrative units at *level* (0-5) from geoBoundaries.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. Without
        *country* the countries it touches are read.
    country
        ISO3 code(s): ``"NOR"`` or ``["NOR", "SWE"]``.
    level
        0 (country) to 5; how deep a country goes varies (Norway stops at 2).
    name
        Keep only these units by name.
    simplified
        Read geoBoundaries' simplified geometries (a fraction of the size).
    source
        Only ``"geoboundaries"`` today.
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``name``, ``iso3``, ``admin_level``, then ``shapeName``, ``shapeISO``,
        ``shapeID``, ``shapeGroup``, ``shapeType`` and the record's
        ``boundaryLicense`` and ``boundarySource``.
    """
    product = catalog.get("admin-boundaries")
    src = resolve_source(product, source)
    return _geoboundaries(
        aoi, country, level, product, src, name=name, simplified=simplified, **kwargs
    )


# ── catalog entries ──────────────────────────────────────────────────────────

_NATURAL_EARTH_CITATION = (
    "Natural Earth. Free vector and raster map data @ naturalearthdata.com."
)
_CENSUS_CITATION = (
    "U.S. Census Bureau (2025). Cartographic Boundary Files. "
    "https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html"
)
_GEOBOUNDARIES_CITATION = (
    "Runfola, D. et al. (2020). geoBoundaries: A global database of political "
    "administrative boundaries. PLoS ONE 15(4): e0231866."
)


def _natural_earth_source(layer: str, what: str, scale_note: str) -> Source:
    return Source(
        id="natural-earth",
        provider="vector_http",
        location=f"{NATURAL_EARTH_URL}/<scale>/cultural/ne_<scale>_{layer}.zip",
        temporal="static (v5.1.1; a few layers 5.0.0)",
        notes=f"zipped shapefile, cached on first use; {scale_note}",
        title="Natural Earth",
        health=Probe(
            f"Natural Earth {what} (naciscdn)",
            partial(health.http_first_byte, natural_earth_url(layer, "50m")),
        ),
    )


def _census_source(layer: str, what: str) -> Source:
    return Source(
        id="us-census",
        provider="vector_http",
        location=f"{CENSUS_URL}/GENZ<year>/shp/cb_<year>_us_{layer}_<resolution>.zip",
        extent="United States",
        temporal="yearly releases",
        notes="1:500k, 1:5m or 1:20m zipped shapefile, NAD83 reprojected to EPSG:4326",
        title="US Census cartographic boundaries",
        health=Probe(
            f"US Census {what} (cartographic boundaries)",
            partial(health.http_first_byte, census_url(layer, "20m")),
        ),
    )


def _probe_geoboundaries(iso3: str, level: int) -> None:
    """The gbOpen record answers, and the GeoJSON it points to is reachable.

    The API and the files live on different hosts (the files are GitHub
    release assets), so checking the API alone would miss a broken download.
    """
    meta = _geoboundaries_meta(iso3, level)
    health.http_first_byte(meta["gjDownloadURL"])


def _geoboundaries_source(levels: str, what: str, level: int) -> Source:
    return Source(
        id="geoboundaries",
        provider="vector_http",
        location=f"{GEOBOUNDARIES_API}/<ISO3>/ADM<level>/",
        temporal="continuously updated",
        notes=(
            f"{levels}; one country per request through the gbOpen API, GeoJSON "
            "cached per release"
        ),
        title="geoBoundaries (gbOpen)",
        health=Probe(
            f"geoBoundaries {what} (ADM{level}, gbOpen)",
            partial(_probe_geoboundaries, "NOR", level),
        ),
    )


_LEAD_VARIABLES = (
    Variable("name", long_name="unit name"),
    Variable("iso3", long_name="country, ISO 3166-1 alpha-3"),
    Variable("admin_level", long_name="administrative level (0 country)"),
)

COUNTRIES_PRODUCT = Product(
    id="countries",
    theme="boundaries",
    title="Country boundaries",
    description=(
        "Country polygons worldwide: Natural Earth at 1:10m, 1:50m or 1:110m, or "
        "geoBoundaries ADM0 country by country."
    ),
    sources=(
        _natural_earth_source(
            "admin_0_countries", "countries", "1:10m, 1:50m and 1:110m"
        ),
        _geoboundaries_source("ADM0", "countries", 0),
    ),
    variables=_LEAD_VARIABLES,
    citation=_NATURAL_EARTH_CITATION,
    license="Public domain (Natural Earth); CC BY 4.0 (geoBoundaries gbOpen)",
    loader="easysnowdata.boundaries.admin.countries",
    examples=("boundaries/plot_admin.py",),
    references=(
        "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/",
        "https://www.geoboundaries.org/",
    ),
    tags=("boundaries", "countries", "admin-0", "natural earth"),
)

STATES_PRODUCT = Product(
    id="states-provinces",
    theme="boundaries",
    title="States, provinces and first-level subdivisions",
    description=(
        "First-level administrative units: US states from the Census Bureau, "
        "states and provinces worldwide from Natural Earth, or geoBoundaries ADM1."
    ),
    sources=(
        _natural_earth_source(
            "admin_1_states_provinces", "states and provinces", "1:10m and 1:50m"
        ),
        _census_source("state", "states"),
        _geoboundaries_source("ADM1", "states and provinces", 1),
    ),
    variables=_LEAD_VARIABLES,
    citation=f"{_NATURAL_EARTH_CITATION} {_CENSUS_CITATION}",
    license="Public domain (Natural Earth, US Census); CC BY 4.0 (geoBoundaries)",
    loader="easysnowdata.boundaries.admin.states",
    examples=("boundaries/plot_admin.py",),
    references=(
        "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/",
        "https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html",
    ),
    tags=("boundaries", "states", "provinces", "admin-1"),
)

COUNTIES_PRODUCT = Product(
    id="us-counties",
    theme="boundaries",
    title="US counties (Census cartographic boundaries)",
    description="US counties and county equivalents from the Census Bureau.",
    sources=(_census_source("county", "counties"),),
    variables=_LEAD_VARIABLES,
    citation=_CENSUS_CITATION,
    license="Public domain (US government data)",
    loader="easysnowdata.boundaries.admin.counties",
    examples=("boundaries/plot_admin.py",),
    references=(
        "https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html",
    ),
    tags=("boundaries", "counties", "admin-2", "census"),
)

ADMIN_PRODUCT = Product(
    id="admin-boundaries",
    theme="boundaries",
    title="Administrative boundaries at any level (geoBoundaries)",
    description=(
        "ADM0 to ADM5 units for any country from geoBoundaries' open release "
        "(counties, districts, municipalities…)."
    ),
    sources=(_geoboundaries_source("ADM0-ADM5", "admin units", 2),),
    variables=_LEAD_VARIABLES,
    citation=_GEOBOUNDARIES_CITATION,
    license="CC BY 4.0 for most countries (each unit carries its boundaryLicense)",
    doi="10.1371/journal.pone.0231866",
    loader="easysnowdata.boundaries.admin.admin",
    examples=("boundaries/plot_admin.py",),
    references=("https://www.geoboundaries.org/api.html",),
    tags=("boundaries", "admin", "geoboundaries"),
)

for _product in (COUNTRIES_PRODUCT, STATES_PRODUCT, COUNTIES_PRODUCT, ADMIN_PRODUCT):
    catalog.register(_product, replace=True)
