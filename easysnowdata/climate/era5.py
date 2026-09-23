"""ERA5 and ERA5-Land reanalysis.

Sources (companion file §B.9):

``arco-era5-gcs`` (default)
    Google's ARCO-ERA5 Zarr store on GCS: hourly ERA5 at 0.25°, 1940 →
    ERA5T (about one week behind real time), anonymous access, 273
    variables. Only *hourly ERA5* lives here.
``gee``
    Earth Engine ``ECMWF/ERA5*`` and ``ECMWF/ERA5_LAND*`` collections: the
    ERA5-Land route and the daily / monthly aggregates. Needs Earth Engine
    credentials.

::

    import easysnowdata as esd
    t2m_ds = esd.climate.era5.load(aoi, "2020-01", variables=["2m_temperature"])
    land_ds = esd.climate.era5.load(aoi, "2020-01", version="ERA5_LAND",
                                    cadence="daily", variables=["temperature_2m"])
    esd.climate.era5.search()                        # variable inventory

The ERA5 / ERA5T boundary of the ARCO store is exposed in the result attrs
(``valid_time_stop`` is the last final-ERA5 day, ``valid_time_stop_era5t``
the last preliminary day).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import pandas as pd
import xarray as xr

from easysnowdata import auth, catalog, providers, temporal
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT",
    "ARCO_URL",
    "GEE_COLLECTIONS",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_DOCS = "https://github.com/google-research/arco-era5"
GEE_DOCS = (
    "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY"
)

#: ``(version, cadence)`` → Earth Engine collection id.
GEE_COLLECTIONS: dict[tuple[str, str], str] = {
    ("ERA5", "hourly"): "ECMWF/ERA5/HOURLY",
    ("ERA5", "daily"): "ECMWF/ERA5/DAILY",
    ("ERA5", "monthly"): "ECMWF/ERA5/MONTHLY",
    ("ERA5_LAND", "hourly"): "ECMWF/ERA5_LAND/HOURLY",
    ("ERA5_LAND", "daily"): "ECMWF/ERA5_LAND/DAILY_AGGR",
    ("ERA5_LAND", "monthly"): "ECMWF/ERA5_LAND/MONTHLY_AGGR",
}
VERSIONS = ("ERA5", "ERA5_LAND")
CADENCES = ("hourly", "daily", "monthly")

PRODUCT = Product(
    id="era5",
    theme="climate",
    title="ERA5 / ERA5-Land reanalysis",
    description=(
        "ECMWF ERA5 hourly reanalysis (ARCO-ERA5 on Google Cloud Storage, 0.25°, "
        "1940 to ERA5T) and the ERA5 / ERA5-Land hourly, daily and monthly "
        "aggregates on Earth Engine (the only ERA5-Land route today)."
    ),
    sources=(
        Source(
            id="arco-era5-gcs",
            provider="zarr_cloud",
            location=ARCO_URL,
            resolution_m=27750,
            temporal="1940/present (ERA5T)",
            latency="~1 week (ERA5T); ~3 months (final ERA5)",
            notes="Zarr v2, consolidated, anonymous; hourly ERA5 only",
            title="ARCO-ERA5 (GCS)",
            health=(
                Probe(
                    "ARCO-ERA5 (GCS anonymous)",
                    partial(health.zarr_metadata, ARCO_URL),
                ),
                # §8 asks a third-party Zarr store for its latest time; for a
                # reanalysis that is also exactly its latency, so one read
                # answers both. ERA5T runs about five days behind real time.
                Probe(
                    "ARCO-ERA5 latency (GCS anonymous)",
                    partial(
                        health.zarr_latest,
                        ARCO_URL,
                        attrs=("valid_time_stop_era5t", "valid_time_stop"),
                    ),
                    kind="latency",
                ),
            ),
        ),
        Source(
            id="gee",
            provider="gee",
            location="ECMWF/ERA5_LAND/HOURLY (and ERA5 / ERA5_LAND daily and monthly aggregates)",
            requires=("earthengine",),
            resolution_m=11132,
            temporal="1950/present",
            latency="~1 week",
            notes="the only ERA5-Land route; daily and monthly aggregates",
            title="Earth Engine",
            health=Probe(
                "ERA5 (Google Earth Engine)",
                partial(
                    health.gee_asset,
                    "ECMWF/ERA5_LAND/HOURLY",
                    start="2020-01-01",
                    end="2020-01-02",
                ),
            ),
        ),
    ),
    variables=(
        Variable("2m_temperature", units="K", long_name="2 metre temperature"),
        Variable("total_precipitation", units="m", long_name="total precipitation"),
        Variable("snow_depth", units="m of water equivalent", long_name="snow depth"),
        Variable("snowfall", units="m of water equivalent", long_name="snowfall"),
        Variable("snowmelt", units="m of water equivalent", long_name="snowmelt"),
        Variable("temperature_2m", units="K", long_name="2 metre temperature (GEE)"),
        Variable(
            "snow_depth_water_equivalent",
            units="m of water equivalent",
            long_name="snow depth water equivalent (GEE)",
        ),
    ),
    citation=(
        "Hersbach, H., et al. (2020). The ERA5 global reanalysis. QJRMS 146, 1999–2049. "
        "Muñoz-Sabater, J., et al. (2021). ERA5-Land. ESSD 13, 4349–4383. "
        "Carver, R. W. and Merose, A. (2023). ARCO-ERA5: An Analysis-Ready Cloud-Optimized "
        "Reanalysis Dataset. 22nd Conf. on AI for Env. Science, AMS."
    ),
    license="Copernicus licence",
    doi="10.1002/qj.3803",
    references=(ARCO_DOCS, GEE_DOCS),
    loader="easysnowdata.climate.era5.load",
    examples=("climate/plot_era5.py",),
    tags=("reanalysis", "temperature", "precipitation", "snow depth"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _normalise(version: str, cadence: str) -> tuple[str, str]:
    version_u = str(version).upper().replace("-", "_")
    if version_u == "ERA5LAND":
        version_u = "ERA5_LAND"
    cadence_l = str(cadence).lower()
    if version_u not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}.")
    if cadence_l not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, got {cadence!r}.")
    return version_u, cadence_l


def _pick_source(source: Any, version: str, cadence: str) -> Source:
    if source is None or (isinstance(source, str) and source.lower() == "auto"):
        source = "arco-era5-gcs" if (version, cadence) == ("ERA5", "hourly") else "gee"
    src = resolve_source(PRODUCT, source)
    if src.id == "arco-era5-gcs" and (version, cadence) != ("ERA5", "hourly"):
        raise ValueError(
            "source='arco-era5-gcs' only serves hourly ERA5; use source='gee' for "
            f"{version} {cadence}."
        )
    return src


def _variables(variables: str | Sequence[str] | None) -> list[str] | None:
    if variables is None:
        return None
    return [variables] if isinstance(variables, str) else list(variables)


def _wrap_longitudes(ds: xr.Dataset, keep_0_360: bool = False) -> xr.Dataset:
    """Re-label 0–360° longitudes as −180…180 and sort them.

    *keep_0_360* is set for an AOI that straddles ±180°: there, wrapping and
    sorting would tear the subset in two, so the 0–360 labels are kept and
    recorded in ``attrs["longitude_convention"]``. A global or one-sided
    subset always comes back in −180…180.
    """
    if "longitude" not in ds.coords or ds.sizes.get("longitude", 0) == 0:
        return ds
    lon_da = ds["longitude"]
    if float(lon_da.max()) <= 180.0:
        return ds
    if keep_0_360:
        ds.attrs["longitude_convention"] = "0-360"
        return ds
    return ds.assign_coords(longitude=((lon_da + 180) % 360) - 180).sortby("longitude")


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    source: str | None = None,
    version: str = "ERA5",
    cadence: str = "hourly",
) -> pd.DataFrame:
    """Inventory of the selected route: one row per variable.

    ERA5 is a continuous store, not a granule archive, so ``search`` returns
    the variables (name, units, long name) with the route's time span. For
    the ARCO store the span comes from its metadata; for Earth Engine from
    the collection. *aoi* is accepted for symmetry and only validated.
    """
    version, cadence = _normalise(version, cadence)
    src = _pick_source(source, version, cadence)
    if aoi is not None:
        parse_aoi(aoi)
    ensure_source(PRODUCT, src)
    if src.id == "arco-era5-gcs":
        ds = providers.zarr_cloud.open(ARCO_URL, chunks=None, consolidated=True)
        rows = [
            {
                "variable": name,
                "units": var_da.attrs.get("units"),
                "long_name": var_da.attrs.get("long_name"),
                "dims": tuple(var_da.dims),
            }
            for name, var_da in ds.data_vars.items()
        ]
        inventory_df = pd.DataFrame(rows).set_index("variable")
        inventory_df.attrs = {
            "source": src.id,
            "time_start": ds.attrs.get("valid_time_start"),
            "time_stop": ds.attrs.get("valid_time_stop_era5t")
            or ds.attrs.get("valid_time_stop"),
            "time_stop_final": ds.attrs.get("valid_time_stop"),
        }
        return inventory_df
    collection = GEE_COLLECTIONS[(version, cadence)]
    ee = providers.gee.ee()
    image = ee.ImageCollection(collection).first()
    info = image.getInfo() or {}
    rows = [
        {"variable": band.get("id"), "units": None, "long_name": None, "dims": ()}
        for band in info.get("bands", [])
    ]
    inventory_df = pd.DataFrame(
        rows, columns=["variable", "units", "long_name", "dims"]
    )
    inventory_df = (
        inventory_df.set_index("variable") if len(inventory_df) else inventory_df
    )
    inventory_df.attrs = {"source": src.id, "collection": collection}
    return inventory_df


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    variables: str | Sequence[str] | None = None,
    source: str | None = None,
    version: str = "ERA5",
    cadence: str = "hourly",
    chunks: Any = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load ERA5 / ERA5-Land as a lazy ``xarray.Dataset`` (``time``, ``latitude``, ``longitude``).

    Parameters
    ----------
    aoi
        Any form :func:`~easysnowdata.aoi.parse_aoi` accepts; ``None`` is global.
        The subset is the smallest block of native cells covering the AOI.
    time
        Any form :func:`~easysnowdata.temporal.parse_time` accepts. ``None``
        means the store's full valid span (ARCO) or the collection's (GEE).
    variables
        Variable name(s); ``None`` keeps every variable.
    source
        ``"arco-era5-gcs"`` (hourly ERA5, default), ``"gee"``, or ``None`` /
        ``"auto"`` (ARCO for hourly ERA5, GEE otherwise).
    version, cadence
        ``"ERA5"`` or ``"ERA5_LAND"``; ``"hourly"``, ``"daily"`` or ``"monthly"``.
    chunks
        Dask chunking; ``None`` (default) keeps the store's native chunking,
        so the result is always Dask-backed (§2.3).
    **kwargs
        Forwarded to ``xarray.open_zarr`` (ARCO) or ``xarray.open_dataset``
        with ``engine="ee"`` (GEE).
    """
    version, cadence = _normalise(version, cadence)
    src = _pick_source(source, version, cadence)
    names = _variables(variables)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)
    if src.id == "arco-era5-gcs":
        ds = _load_arco(parsed, time, names, chunks, **kwargs)
        url = ARCO_DOCS
    else:
        ds = _load_gee(parsed, time, names, version, cadence, chunks, **kwargs)
        url = GEE_DOCS
    ds = contract.finalize(
        ds,
        PRODUCT,
        src,
        crs="EPSG:4326",
        mask=False,
        source_url=url,
        attrs={"version": version, "cadence": cadence},
    )
    return ds


def _load_arco(
    parsed: Any, time: Any, names: list[str] | None, chunks: Any, **kwargs: Any
) -> xr.Dataset:
    # Open without Dask first. ARCO-ERA5 holds 273 variables on a 1940-onward
    # hourly grid, so chunking before the variable and time subset builds a
    # graph big enough to exhaust memory; the result is chunked below instead.
    open_kwargs: dict[str, Any] = {
        "chunks": None,
        "consolidated": True,
        **kwargs,
    }
    ds = providers.zarr_cloud.open(ARCO_URL, **open_kwargs)
    store_attrs = dict(ds.attrs)
    if names is not None:
        missing = [n for n in names if n not in ds.data_vars]
        if missing:
            raise KeyError(
                f"Variables {missing} are not in ARCO-ERA5; see climate.era5.search()."
            )
        ds = ds[names]
    start, end = temporal.parse_time(time)
    valid_start = store_attrs.get("valid_time_start")
    valid_stop = store_attrs.get("valid_time_stop_era5t") or store_attrs.get(
        "valid_time_stop"
    )
    if start is None and valid_start:
        start = pd.Timestamp(valid_start)
    if valid_stop:
        stop_ts = pd.Timestamp(valid_stop) + pd.Timedelta(hours=23)
        if time is None or end > stop_ts:
            end = stop_ts
    ds = providers.zarr_cloud.select(ds, parsed, (start, end))
    ds = _wrap_longitudes(
        ds, keep_0_360=bool(parsed is not None and parsed.crosses_antimeridian)
    )
    # Chunk the subset, not the store (§2.3: Dask-backed by default).
    ds = ds.chunk({"time": 24} if chunks is None else chunks)
    kept = {k: v for k, v in ds.attrs.items() if k in ("longitude_convention",)}
    ds.attrs = {
        k: store_attrs[k]
        for k in (
            "valid_time_start",
            "valid_time_stop",
            "valid_time_stop_era5t",
            "last_updated",
        )
        if k in store_attrs
    } | kept
    final_stop = store_attrs.get("valid_time_stop")
    if final_stop and end > pd.Timestamp(final_stop) + pd.Timedelta(hours=23):
        ds.attrs["era5t_from"] = str(
            (pd.Timestamp(final_stop) + pd.Timedelta(days=1)).date()
        )
    return ds


def _load_gee(
    parsed: Any,
    time: Any,
    names: list[str] | None,
    version: str,
    cadence: str,
    chunks: Any,
    **kwargs: Any,
) -> xr.Dataset:
    auth.ensure("earthengine")
    ee = providers.gee.ee()
    collection = ee.ImageCollection(GEE_COLLECTIONS[(version, cadence)])
    start, end = temporal.parse_time(time)
    if time is not None:
        start_s = (start or pd.Timestamp("1950-01-01")).strftime("%Y-%m-%dT%H:%M:%S")
        end_s = (end + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
        collection = collection.filterDate(start_s, end_s)
    if names is not None:
        collection = collection.select(names)
    open_kwargs: dict[str, Any] = {"chunks": {} if chunks is None else chunks}
    open_kwargs.update(kwargs)
    ds = providers.gee.open_dataset(collection, parsed, **open_kwargs)
    ds.attrs = {"collection": GEE_COLLECTIONS[(version, cadence)]}
    return ds
