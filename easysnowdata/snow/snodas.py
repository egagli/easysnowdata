"""SNODAS: NOHRSC daily 1 km snow water equivalent and snow depth.

Sources (companion file §B.5):

``nsidc`` (default, new)
    The authoritative NSIDC G02158 archive: one ``.tar`` per day holding
    gzipped flat-binary grids with their own text headers. No account, no
    Earth Engine — the cost is a small download per day and a reader
    (:mod:`easysnowdata.processing.snow`) instead of an analysis-ready cube.
``gee-climate-engine``
    Climate Engine's community re-hosting on Earth Engine: fast, lazy and
    server-side subset, about a day behind, but a non-authoritative mirror
    behind Earth Engine credentials.

Both should agree; the gallery example checks that they do.

**Glaciers saturate the grid.** SNODAS never melts perennial ice out, so SWE
grows without bound over glaciers and eventually saturates the 16-bit field at
32767 mm (32.767 m). Verified 2026-09-16 on 2024-03-15: eight CONUS pixels are
saturated and they sit on Mount Rainier's and Mount Baker's summit ice caps,
while the Rainier box's median is 0.86 m. Those values are the model's, not a
reader artefact, so they are passed through untouched; mask them yourself
(``ds["SWE"].where(ds["SWE"] < 30)``) when a basin contains glaciers.

::

    import easysnowdata as esd
    swe = esd.snow.snodas.load(aoi, "2024-03-01/2024-03-07")            # NSIDC
    swe = esd.snow.snodas.load(aoi, "2024-03", source="gee-climate-engine")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, config, providers, temporal
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.processing import snow as snow_processing

__all__ = [
    "PRODUCT",
    "VARIABLES",
    "REGIONS",
    "tar_url",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

NSIDC_ROOT = "https://noaadata.apps.nsidc.org/NOAA/G02158"
GEE_COLLECTION = (
    "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"
)
_NSIDC_DOCS = "https://nsidc.org/data/g02158/versions/1"
_GEE_DOCS = "https://gee-community-catalog.org/projects/snodas/"

#: Which grid to read: the CONUS-only ``masked`` product or the larger
#: ``unmasked`` domain (which also covers southern Canada and northern Mexico).
REGIONS = ("masked", "unmasked")

#: Variable name → (SNODAS product code, units, long name).
VARIABLES: dict[str, tuple[str, str, str]] = {
    "SWE": ("1034", "m", "snow water equivalent, total of snow layers"),
    "snow_depth": ("1036", "m", "snow layer thickness, total of snow layers"),
    "snowpack_average_temperature": ("1038", "K", "snowpack average temperature"),
    "blowing_snow_sublimation": ("1039", "m", "sublimation of blowing snow"),
    "snow_melt": ("1044", "m", "snow melt runoff at the base of the snowpack"),
    "snowpack_sublimation": ("1050", "m", "sublimation from the snowpack"),
}
#: The Earth Engine mirror's band names for the two variables it carries.
GEE_BANDS = {"SWE": "SWE", "snow_depth": "Snow_Depth"}
DEFAULT_VARIABLES = ("SWE", "snow_depth")

PRODUCT = Product(
    id="snodas",
    theme="snow",
    title="SNODAS snow water equivalent and snow depth",
    description=(
        "NOHRSC Snow Data Assimilation System daily 1 km snow water equivalent, snow "
        "depth, melt, sublimation and snowpack temperature over the CONUS (masked) or "
        "the wider modelling domain (unmasked), 2003-10 onward."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="raster_http",
            location=f"{NSIDC_ROOT}/{{region}}/YYYY/MM_Mon/SNODAS_YYYYMMDD.tar",
            resolution_m=1000,
            extent="CONUS (masked) / North America (unmasked)",
            temporal="2003-10/present",
            latency="~1 day",
            notes=(
                "authoritative NOHRSC archive: one tar per day of gzipped flat-binary "
                "grids plus text headers; no credentials, no cloud-native mirror exists"
            ),
            title="NSIDC G02158 (direct)",
            health=Probe(
                "SNODAS (NSIDC G02158)",
                partial(
                    health.http_first_byte,
                    f"{NSIDC_ROOT}/masked/2024/03_Mar/SNODAS_20240315.tar",
                ),
            ),
        ),
        Source(
            id="gee-climate-engine",
            provider="gee",
            location=GEE_COLLECTION,
            requires=("earthengine",),
            resolution_m=1000,
            extent="CONUS",
            temporal="2003-10/present",
            latency="~1 day",
            notes=(
                "Climate Engine's community re-hosting: analysis-ready and lazy, but "
                "a non-authoritative mirror with only SWE and snow depth"
            ),
            title="Earth Engine (Climate Engine)",
            health=Probe(
                "SNODAS (GEE/Climate Engine)",
                partial(
                    health.gee_asset,
                    GEE_COLLECTION,
                    start="2020-01-01",
                    end="2020-01-02",
                ),
            ),
        ),
    ),
    variables=(
        Variable("SWE", units="m", dtype="float32", long_name="snow water equivalent"),
        Variable("snow_depth", units="m", dtype="float32", long_name="snow depth"),
        Variable(
            "snow_melt",
            units="m",
            dtype="float32",
            long_name="snow melt runoff at the base of the snowpack",
        ),
    ),
    citation=(
        "National Operational Hydrologic Remote Sensing Center (2004). Snow Data "
        "Assimilation System (SNODAS) Data Products at NSIDC, Version 1. Boulder, "
        "Colorado USA. NSIDC."
    ),
    license="Public domain (US government data)",
    doi="10.7265/N5TB14TC",
    references=(_NSIDC_DOCS, _GEE_DOCS),
    loader="easysnowdata.snow.snodas.load",
    examples=("snow/plot_snodas.py",),
    tags=("swe", "snow depth", "snodas", "nohrsc"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _variables(variables: str | Sequence[str] | None) -> list[str]:
    names = (
        list(DEFAULT_VARIABLES)
        if variables is None
        else ([variables] if isinstance(variables, str) else list(variables))
    )
    unknown = [n for n in names if n not in VARIABLES]
    if unknown:
        raise ValueError(
            f"Invalid variables: {set(unknown)}. Available variables: {list(VARIABLES)}"
        )
    return names


def _region(region: str) -> str:
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    return region


def tar_url(day: Any, *, region: str = "masked") -> str:
    """The NSIDC URL of one SNODAS day (``…/2024/03_Mar/SNODAS_20240315.tar``)."""
    when = pd.Timestamp(day)
    region = _region(region)
    stem = "SNODAS" if region == "masked" else "SNODAS_unmasked"
    return (
        f"{NSIDC_ROOT}/{region}/{when:%Y}/{when:%m}_{when:%b}/{stem}_{when:%Y%m%d}.tar"
    )


def _days(time: Any) -> pd.DatetimeIndex:
    start, end = temporal.parse_time(time)
    if start is None:
        raise ValueError(
            "SNODAS needs a start date: it is one file per day, so an open-ended "
            "request would download the whole archive."
        )
    return pd.date_range(start.normalize(), end.normalize(), freq="D")


def _variable_of(stem: str) -> str | None:
    """Map a member name (``us_ssmv11034tS__T0001…``) to a variable name."""
    for name, (code, _, _) in VARIABLES.items():
        if f"ssmv1{code}" in stem or f"ssmv0{code}" in stem:
            return name
    return None


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    region: str = "masked",
) -> gpd.GeoDataFrame | pd.DataFrame:
    """List the SNODAS days a request covers, with their archive URLs.

    Nothing is downloaded: the archive is one file per day at a predictable
    URL, so the "search" is the day list plus where each day lives.
    """
    src = resolve_source(PRODUCT, source)
    if aoi is not None:
        parse_aoi(aoi)
    days = _days(time)
    frame = pd.DataFrame(
        {
            "date": days,
            "url": [tar_url(day, region=region) for day in days]
            if src.id == "nsidc"
            else [GEE_COLLECTION] * len(days),
        }
    )
    frame.attrs = {"source": src.id, "region": region}
    return frame


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    variables: str | Sequence[str] | None = None,
    source: str | None = None,
    region: str = "masked",
    chunks: Any = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load SNODAS as an ``xarray.Dataset`` (``time``, ``latitude``, ``longitude``).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs. *time* must have a start: the
        NSIDC archive is one file per day.
    variables
        Any of :data:`VARIABLES` (``"SWE"`` and ``"snow_depth"`` by default).
        The Earth Engine mirror carries only those two.
    source
        ``"nsidc"`` (default, no credentials) or ``"gee-climate-engine"``.
    region
        ``"masked"`` (CONUS, the usual product) or ``"unmasked"`` (the wider
        modelling domain). NSIDC route only.
    """
    src = resolve_source(PRODUCT, source)
    names = _variables(variables)
    region = _region(region)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if src.id == "nsidc":
        ds = _load_nsidc(parsed, time, names, region, chunks, **kwargs)
    else:
        ds = _load_gee(parsed, time, names, chunks, **kwargs)
    return contract.finalize(
        ds,
        PRODUCT,
        src,
        crs="EPSG:4326",
        variables=(),
        mask=False,
        source_url=_NSIDC_DOCS if src.id == "nsidc" else _GEE_DOCS,
        attrs={"region": region if src.id == "nsidc" else None},
    )


def _load_nsidc(
    parsed: Any,
    time: Any,
    names: Sequence[str],
    region: str,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    """Fetch one tar per day, read the flat-binary grids, stack them on time."""
    days = _days(time)
    wanted = {VARIABLES[name][0]: name for name in names}
    per_day = []
    for day in days:
        url = tar_url(day, region=region)
        try:
            path = providers.raster_http.fetch(
                url, subdir="snodas", progressbar=False, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 — a missing day must not kill the series
            _logger.warning(
                "SNODAS %s is not available (%s); skipping.", day.date(), exc
            )
            continue
        members = snow_processing.snodas_members(path.read_bytes())
        arrays = {}
        for stem, parts in members.items():
            variable = _variable_of(stem)
            if variable is None or variable not in wanted.values():
                continue
            if "dat" not in parts or "txt" not in parts:  # pragma: no cover — defensive
                continue
            header = snow_processing.parse_snodas_header(parts["txt"].decode())
            header.setdefault("description", VARIABLES[variable][2])
            da = snow_processing.snodas_array(parts["dat"], header, name=variable)
            if parsed is not None and parsed.clip and not parsed.is_global:
                west, south, east, north = parsed.total_bounds()
                da = da.rio.clip_box(
                    west, south, east, north, allow_one_dimensional_raster=True
                )
            arrays[variable] = da
        if not arrays:
            _logger.warning("SNODAS %s has none of %s; skipping.", day.date(), names)
            continue
        per_day.append(xr.Dataset(arrays).expand_dims(time=[day]))
    if not per_day:
        raise ValueError(
            f"No SNODAS days available between {days[0].date()} and {days[-1].date()}."
        )
    ds = xr.concat(per_day, dim="time") if len(per_day) > 1 else per_day[0]
    ds = ds.sortby("time")
    for name in ds.data_vars:
        _, units, long_name = VARIABLES[name]
        ds[name].attrs.update({"units": units, "long_name": long_name})
    if chunks is not None:
        ds = ds.chunk(chunks)
    elif ds.chunks is None or not ds.chunks:
        ds = ds.chunk({"time": 1})
    ds.attrs["cache_dir"] = str(config.cache_dir("snodas"))
    return ds


def _load_gee(
    parsed: Any, time: Any, names: Sequence[str], chunks: Any, **kwargs: Any
) -> xr.Dataset:
    ee = providers.gee.ee()
    collection = ee.ImageCollection(GEE_COLLECTION)
    start, end = temporal.parse_time(time)
    if time is not None:
        collection = collection.filterDate(
            (start or pd.Timestamp("2003-10-01")).strftime("%Y-%m-%d"),
            (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
    bands = [GEE_BANDS[name] for name in names if name in GEE_BANDS]
    if not bands:
        raise ValueError(
            f"The Earth Engine mirror carries only {list(GEE_BANDS)}; "
            f"use source='nsidc' for {list(names)}."
        )
    collection = collection.select(bands)
    ds = providers.gee.open_dataset(
        collection, parsed, chunks={} if chunks is None else chunks, **kwargs
    )
    rename = {value: key for key, value in GEE_BANDS.items() if value in ds.data_vars}
    ds = ds.rename(rename) if rename else ds
    for name in ds.data_vars:
        if name in VARIABLES:
            _, units, long_name = VARIABLES[name]
            ds[name].attrs.update({"units": units, "long_name": long_name})
    ds.attrs["collection"] = GEE_COLLECTION
    return ds
