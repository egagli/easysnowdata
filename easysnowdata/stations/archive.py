"""The daily station archive: everything daily, without hitting five APIs.

``global_snow_networks`` pre-downloads daily SWE and snow depth for every
station its probe has verified as daily-or-better, and publishes two
artefacts that this module reads (§9 step 3):

``all_snow_stations.geojson``
    Every station from all five networks, periodic sites included, with the
    26 normalized properties of that repo's DESIGN.md §6.1 — including
    ``daily_or_better``, which is the **probe's verdict** rather than what a
    network advertises about itself.
``data/all_station_csvs.tar.xz``
    One ``date,wteq_cm,snwd_cm`` CSV per daily-or-better station, bundled.
    About 28 MB, and the cheapest way by far to get every station's whole
    record — which, for a few long snow courses, reaches back to 1896.

::

    import easysnowdata as esd

    inv = esd.stations.archive.inventory(daily_only=True)   # one request
    ds = esd.stations.archive.load(aoi=(-121.94, 46.72, -121.54, 46.99))

Both artefacts are refreshed daily by that repo's CI, so the archive is at
most a day behind and covers only SWE and snow depth. For anything else — a
different variable, an interval other than daily, quality flags, or today's
observation — use :func:`easysnowdata.stations.load`, which goes to the
networks themselves.
"""

from __future__ import annotations

import io
import logging
import tarfile
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract, wateryear
from easysnowdata.stations import _frames, networks

__all__ = [
    "ARCHIVE_URL",
    "CSV_BASE",
    "INVENTORY_URL",
    "PRODUCT",
    "REPO",
    "csv_url",
    "inventory",
    "load",
    "network_of",
]

_logger = logging.getLogger(__name__)

REPO = "https://github.com/egagli/global_snow_networks"
INVENTORY_URL = f"{REPO}/raw/main/all_snow_stations.geojson"
ARCHIVE_URL = f"{REPO}/raw/main/data/all_station_csvs.tar.xz"
CSV_BASE = (
    "https://raw.githubusercontent.com/egagli/global_snow_networks/main/data/stations/"
)

#: Archive CSV column -> (standardized type, units).
COLUMNS = {
    "wteq_cm": ("swe", "cm"),
    "snwd_cm": ("snwd", "cm"),
}

#: Inventory properties kept as columns, on top of the geometry.
_KEEP = (
    "code",
    "name",
    "client",
    "network_code",
    "network_name",
    "latitude",
    "longitude",
    "elevation_m",
    "state",
    "operator",
    "data_provider",
    "status",
    "is_active",
    "begin_date",
    "end_date",
    "earliest_record_date",
    "latest_record_date",
    "station_url",
    "has_daily_swe",
    "has_daily_snwd",
    "daily_or_better",
    "daily_verified",
    "daily_provenance",
    "metadata_fetched_at",
)

_INVENTORY_CACHE: dict[str, gpd.GeoDataFrame] = {}


def csv_url(code: str) -> str:
    """URL of one station's archive CSV."""
    return f"{CSV_BASE}{code}.csv"


def _read_inventory(**kwargs: Any) -> gpd.GeoDataFrame:
    """The published inventory, memoized per process for the keyword set used."""
    key = repr(sorted(kwargs.items()))
    cached = _INVENTORY_CACHE.get(key)
    if cached is not None:
        return cached
    _logger.info("Reading the station inventory from %s", INVENTORY_URL)
    gdf = providers.vector_http.read(INVENTORY_URL, **kwargs)
    if "code" in gdf.columns:
        gdf = gdf.set_index("code")
    # Careful: the published inventory already has a `network` property, and
    # it means something else — a Yukon-specific display name ("Yukon Snow
    # Survey Network"), null for the other four clients. This package's
    # `network` is the access path, which that repo calls `client`. Keep the
    # upstream value under `network_name` and let `network` mean one thing.
    if "client" in gdf.columns:
        if "network" in gdf.columns:
            gdf = gdf.rename(columns={"network": "network_name"})
        gdf["network"] = gdf["client"]
    _INVENTORY_CACHE[key] = gdf
    return gdf


def inventory(
    aoi: Any = None,
    *,
    networks: Any = None,
    daily_only: bool = False,
    active_only: bool = False,
    columns: Any = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """The published station inventory as a GeoDataFrame indexed by code.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` keeps
        every station.
    networks
        Keep only these networks (``"awdb"``, ``"cdec"``, ``"databc"``,
        ``"nve"``, ``"yukon"``).
    daily_only
        Keep only stations whose daily record the pipeline has actually
        verified (``daily_or_better``), which is what the archive holds CSVs
        for. Unlike the live route, this verdict is probe-based, not
        advertised.
    active_only
        Keep only stations the network reports as active.
    columns
        Columns to keep. ``None`` keeps the standard set; ``"all"`` keeps
        every property, including ``data_variables`` and
        ``possible_duplicates``.
    **kwargs
        Passed to ``geopandas.read_file`` (``rows=``, ``where=``, …).

    Returns
    -------
    geopandas.GeoDataFrame
        Stations in EPSG:4326 indexed by ``code``.
    """
    gdf = _read_inventory(**kwargs)
    if networks is not None:
        wanted = [networks] if isinstance(networks, str) else list(networks)
        gdf = gdf[gdf["network"].isin(wanted)]
    if daily_only and "daily_or_better" in gdf.columns:
        gdf = gdf[gdf["daily_or_better"].fillna(False).astype(bool)]
    if active_only and "is_active" in gdf.columns:
        gdf = gdf[gdf["is_active"].fillna(False).astype(bool)]
    if aoi is not None:
        parsed = parse_aoi(aoi)
        if not parsed.is_global:
            gdf = gdf[gdf.intersects(parsed.geometry)]
    if columns == "all":
        pass
    elif columns is None:
        keep = [c for c in ("station_id", "network", *_KEEP) if c in gdf.columns]
        gdf = gdf[[*dict.fromkeys(keep), "geometry"]]
    else:
        gdf = gdf[[*columns, "geometry"]]
    gdf = gdf.copy()
    if "station_id" not in gdf.columns and "network" in gdf.columns:
        gdf["station_id"] = [
            networks_module().to_station_id(str(code), str(net))
            if str(net) in networks_module().NETWORKS
            else str(code)
            for code, net in zip(gdf.index, gdf["network"], strict=True)
        ]
    product = catalog.get("snow-station-archive")
    gdf.attrs.update(
        contract.provenance(
            product, resolve_source(product, "github-tarball"), source_url=INVENTORY_URL
        )
    )
    return gdf


def networks_module() -> Any:
    """:mod:`easysnowdata.stations.networks` (the keyword shadows the name)."""
    return networks


def network_of(codes: list[str]) -> dict[str, str]:
    """``{code: network}`` from the inventory, for codes whose shape is ambiguous."""
    gdf = _read_inventory()
    found = gdf.reindex([str(c) for c in codes])["network"]
    return {str(code): str(net) for code, net in found.dropna().items()}


# ── reading the data ─────────────────────────────────────────────────────────


def _parse_csv(text: str, code: str) -> pd.DataFrame:
    """One station CSV as a date-indexed frame (``date,wteq_cm,snwd_cm``)."""
    frame = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    return frame.set_index("date").sort_index()


def _from_tarball(codes: set[str] | None) -> dict[str, pd.DataFrame]:
    """Every wanted station's CSV, out of the one bundled archive.

    Downloaded once into the package cache by pooch, so a second call in the
    same environment re-reads the local file. ``EASYSNOWDATA_CACHE_DIR`` moves
    the cache root.
    """
    path = providers.raster_http.fetch(
        ARCHIVE_URL, "all_station_csvs.tar.xz", subdir="stations"
    )
    _logger.info("Reading the bundled station archive at %s", path)
    frames: dict[str, pd.DataFrame] = {}
    with tarfile.open(path, mode="r:xz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            code = Path(member.name).stem
            if codes is not None and code not in codes:
                continue
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover — directories skipped above
                continue
            frames[code] = _parse_csv(handle.read().decode("utf-8"), code)
    _logger.info("Read %d station CSVs from the archive", len(frames))
    return frames


def _from_csvs(codes: list[str]) -> dict[str, pd.DataFrame]:
    """One HTTP request per station, for when only a few are wanted."""
    import requests  # noqa: PLC0415

    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        response = requests.get(csv_url(code), timeout=60)
        if response.status_code == 404:
            _logger.warning(
                "%s has no archive CSV (the probe has not verified it as "
                "daily-or-better); its row will be all-NaN.",
                code,
            )
            continue
        response.raise_for_status()
        frames[code] = _parse_csv(response.text, code)
    return frames


def load(
    stations: Any = None,
    *,
    aoi: Any = None,
    variables: Any = None,
    time: Any = None,
    networks: Any = None,
    source: str | None = None,
    hemisphere: str = "northern",
) -> xr.Dataset:
    """Daily SWE and snow depth for many stations, from the published archive.

    Parameters
    ----------
    stations
        Station codes, or an inventory frame. ``None`` with no *aoi* means
        every station in the archive — which is the point of this route.
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; picks the
        stations when *stations* is ``None``.
    variables
        ``"swe"``, ``"snwd"`` or both (the default). The archive holds
        nothing else; :func:`easysnowdata.stations.load` does.
    time
        Any form :func:`easysnowdata.temporal.parse_time` accepts. Each
        station's series is cut to the window before the dense grid is built,
        so a narrow window is cheap — but the bundle is still downloaded
        whole, because that is how it is published.
    networks
        Keep only these networks.
    source
        ``"github-tarball"`` (default) downloads the one ~28 MB bundle and
        reads every wanted CSV out of it — right for more than a handful of
        stations, and the only sane route for all of them.
        ``"github-csv"`` fetches one CSV per station instead, which is
        cheaper for a few stations and needs no temporary file.
    hemisphere
        Which water year to attach.

    Returns
    -------
    xarray.Dataset
        ``swe`` and ``snwd`` in centimetres with dims ``(station, time)``,
        station metadata on ``station`` and water-year coordinates on
        ``time``.

    Notes
    -----
    The whole archive is about 1 550 stations by 47 000 days, so asking for
    all of it materializes roughly 1.2 GB of float64 (measured: 31 s and
    under 2 GB of peak memory). Pass *time*, *aoi* or *stations* when you do
    not need every cell.

    Examples
    --------
    >>> import easysnowdata as esd
    >>> ds = esd.stations.archive.load()          # every daily station  # doctest: +SKIP
    """
    product = catalog.get("snow-station-archive")
    src = resolve_source(product, source)
    wanted_types = _types(variables)

    inv = stations if hasattr(stations, "columns") else None
    codes: list[str] | None = None
    if inv is not None:
        codes = [str(c) for c in inv.index]
    elif stations is not None:
        codes = [
            str(s) for s in ([stations] if isinstance(stations, str) else stations)
        ]
    elif aoi is not None or networks is not None:
        inv = inventory(aoi, networks=networks, daily_only=True)
        codes = [str(c) for c in inv.index]

    if inv is None:
        inv = inventory(daily_only=False)
        if codes is not None:
            inv = inv.reindex(codes)

    if src.id == "github-csv":
        if codes is None:
            raise ValueError(
                'source="github-csv" fetches one CSV per station, so it needs '
                "stations= or aoi=. Use the default tarball route for the "
                "whole archive."
            )
        frames = _from_csvs(codes)
    else:
        frames = _from_tarball(set(codes) if codes is not None else None)

    window = None
    if time is not None:
        from easysnowdata.temporal import parse_time  # noqa: PLC0415

        window = parse_time(time)
    ds = _to_dataset(frames, wanted_types, codes, inv, hemisphere, window)
    ds.attrs.update(contract.provenance(product, src, source_url=_url_of(src)))
    ds.attrs["interval"] = "daily"
    return ds


def _url_of(src: Any) -> str:
    return ARCHIVE_URL if src.id == "github-tarball" else CSV_BASE


def _types(variables: Any) -> list[str]:
    known = {type_name for type_name, _ in COLUMNS.values()}
    if variables is None:
        return sorted(known)
    wanted = [variables] if isinstance(variables, str) else list(variables)
    unknown = [v for v in wanted if v not in known]
    if unknown:
        raise ValueError(
            f"The archive holds only {', '.join(sorted(known))}; it has no "
            f"{', '.join(unknown)}. Use easysnowdata.stations.load() for those."
        )
    return list(wanted)


def _to_dataset(
    frames: dict[str, pd.DataFrame],
    types: list[str],
    codes: list[str] | None,
    inv: gpd.GeoDataFrame | None,
    hemisphere: str,
    window: tuple[Any, Any] | None = None,
) -> xr.Dataset:
    """The parsed CSVs as the same ``(station, time)`` Dataset ``load()`` returns.

    Each station's series is written straight into a preallocated dense
    array. Concatenating first and pivoting would work for a handful of
    stations and build a 30-million-row intermediate for the whole archive,
    which is the case this route exists to serve.
    """
    import numpy as np  # noqa: PLC0415

    if window is not None:
        start, end = window
        frames = {
            code: frame.loc[start:end]  # type: ignore[misc]
            for code, frame in frames.items()
        }
    station_axis = codes if codes is not None else sorted(frames)
    if frames:
        times = pd.DatetimeIndex(
            sorted({t for frame in frames.values() for t in frame.index}), name="time"
        )
    else:
        times = pd.DatetimeIndex([], name="time")
    positions = {code: i for i, code in enumerate(station_axis)}
    time_positions = pd.Series(range(len(times)), index=times)

    arrays: dict[str, Any] = {}
    for column, (type_name, _units) in COLUMNS.items():
        if type_name not in types:
            continue
        values = np.full((len(station_axis), len(times)), np.nan, dtype="float64")
        for code, frame in frames.items():
            row = positions.get(code)
            if row is None or column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            values[row, time_positions.reindex(series.index).to_numpy()] = (
                series.to_numpy("float64")
            )
        arrays[type_name] = (("station", "time"), values)

    ds = xr.Dataset(
        arrays,
        coords={"station": np.array(station_axis, dtype="str"), "time": times},
    )
    for column, (type_name, units) in COLUMNS.items():
        if type_name in ds.data_vars:
            ds[type_name].attrs.update(
                {
                    "units": units,
                    "long_name": networks.TYPES[type_name][1],
                    "native_variables": column,
                    "interval": "daily",
                }
            )
    ds = _frames._attach_metadata(ds, inv, {})
    if ds.sizes.get("time", 0):
        ds = wateryear.add_water_year_coords(ds, hemisphere)
    return ds


# ── catalog entry ────────────────────────────────────────────────────────────

PRODUCT = Product(
    id="snow-station-archive",
    theme="stations",
    title="Daily snow-station archive (global snow networks)",
    description=(
        "Daily SWE and snow depth for every station whose daily "
        "record has been probe-verified, across all five networks, "
        "pre-downloaded and published by global_snow_networks: a normalized "
        "station inventory as GeoJSON and one CSV per station, bundled into a "
        "single ~28 MB archive. The fast path for 'everything daily' without "
        "hitting five APIs, refreshed daily. Most stations start around 1980; "
        "a few long snow courses reach back to 1896. SWE and snow depth only; for "
        "other variables, other intervals or quality flags, use "
        "easysnowdata.stations.load()."
    ),
    sources=(
        Source(
            id="github-tarball",
            provider="raster_http",
            location=ARCHIVE_URL,
            extent="western US, western Canada, Norway, Yukon, California, BC",
            temporal="1896/present (most stations from ~1980)",
            latency="daily rebuild",
            notes=(
                "one ~28 MB download holding every station CSV; cached, so "
                "the whole archive costs one request"
            ),
            title="global_snow_networks bundled archive",
            # These two labels are inherited from the retired
            # `snotel-ccss-stations` entry on purpose. They are the keys
            # data_status/history.json and the README status table have
            # recorded weekly since 2026-06, and the route they name — a
            # station list and a station CSV on GitHub — is still exactly
            # what they probe, just published by global_snow_networks now.
            # Renaming them would start two fresh rows and orphan the
            # history.
            health=(
                Probe(
                    "SNOTEL/CCSS station list (GitHub)",
                    partial(health.http_first_byte, INVENTORY_URL),
                ),
                Probe(
                    "Snow station archive tarball (global_snow_networks)",
                    partial(health.http_first_byte, ARCHIVE_URL),
                ),
            ),
        ),
        Source(
            id="github-csv",
            provider="vector_http",
            location=CSV_BASE,
            extent="western US, western Canada, Norway, Yukon, California, BC",
            temporal="1896/present (most stations from ~1980)",
            latency="daily rebuild",
            notes="one request per station; cheaper than the bundle for a few",
            title="global_snow_networks per-station CSVs",
            health=Probe(
                "SNOTEL/CCSS station CSV (GitHub)",
                partial(health.http_first_byte, f"{CSV_BASE}679_WA_SNTL.csv"),
            ),
        ),
    ),
    variables=(
        Variable("swe", units="cm", dtype="float64", long_name="snow water equivalent"),
        Variable("snwd", units="cm", dtype="float64", long_name="snow depth"),
    ),
    citation=(
        "Gagliano, E. Global snow networks: a documented inventory and daily "
        "archive of public snow-station observations. "
        "https://github.com/egagli/global_snow_networks. Underlying "
        "observations are the networks' own; cite them as well."
    ),
    license="Per contributing network; see each network's product entry",
    loader="easysnowdata.stations.archive.load",
    examples=("stations/plot_station_archive.py",),
    references=(REPO, f"{REPO}/blob/main/DESIGN.md"),
    tags=("swe", "snow depth", "stations", "archive", "daily"),
)

catalog.register(PRODUCT, replace=True)
