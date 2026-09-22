"""The daily station archive: everything daily, without hitting five APIs.

``global_snow_networks`` pre-downloads daily SWE and snow depth for every
station its probe has verified as daily-or-better, and publishes three
artefacts that this module reads (§9 step 3):

``all_snow_stations.geojson``
    Every station from all five networks, periodic sites included, with the
    26 normalized properties of that repo's DESIGN.md §6.1 — including
    ``daily_or_better``, which is the **probe's verdict** rather than what a
    network advertises about itself.
``archive/*.zarr`` on that repo's GitHub Pages site
    The same observations as the CSVs below, chunked (that repo's DESIGN.md
    §6.5): ``by_time.zarr`` for "one water year, every station" and
    ``by_station.zarr`` for "this station's whole record". Pages serves
    range requests, so a query fetches the chunks it touches — well under a
    megabyte for either of those — instead of the whole bundle. The default
    route.
``all_station_csvs.tar.xz`` on that repo's latest GitHub Release
    One ``date,wteq_cm,snwd_cm`` CSV per daily-or-better station, bundled:
    the periodic, DOI'd snapshot (a few times a year), about 28 MB and
    whole-file only. The fallback when Pages is down, since Releases are
    served from elsewhere — and the route to use when a result should be
    pinned to a citable snapshot rather than to today's archive.

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
from easysnowdata.stations import _frames
from easysnowdata.stations import networks as _nets

__all__ = [
    "ARCHIVE_URL",
    "CSV_BASE",
    "INVENTORY_URL",
    "MANIFEST_URL",
    "PAGES_BASE",
    "PRODUCT",
    "RELEASES_API",
    "REPO",
    "ZARR_URLS",
    "csv_url",
    "inventory",
    "latest_snapshot",
    "load",
    "manifest",
    "network_of",
]

_logger = logging.getLogger(__name__)

REPO = "https://github.com/egagli/global_snow_networks"
INVENTORY_URL = f"{REPO}/raw/main/all_snow_stations.geojson"
#: The bundled CSVs of the latest snapshot release. Fixed-name asset, so this
#: URL always resolves to the newest release; :func:`latest_snapshot` asks the
#: API for the tag and date behind it. Until 2026-09 the bundle was a file
#: committed daily on ``main``; that stopped once the Pages store took over
#: the daily role (that repo's docs/STORAGE.md §3).
ARCHIVE_URL = f"{REPO}/releases/latest/download/all_station_csvs.tar.xz"
RELEASES_API = (
    "https://api.github.com/repos/egagli/global_snow_networks/releases/latest"
)
CSV_BASE = (
    "https://raw.githubusercontent.com/egagli/global_snow_networks/main/data/stations/"
)
#: The chunked archive, rebuilt into that repo's Pages artefact on every
#: deploy (its DESIGN.md §6.5). Two stores, same observations, two chunk
#: layouts; ``archive.json`` beside them names the commit they were built from.
PAGES_BASE = "https://egagli.github.io/global_snow_networks/archive"
ZARR_URLS = {
    "by_time": f"{PAGES_BASE}/by_time.zarr",
    "by_station": f"{PAGES_BASE}/by_station.zarr",
}
MANIFEST_URL = f"{PAGES_BASE}/archive.json"

#: Named stations up to this many read from ``by_station.zarr`` (one chunk of
#: 64 stations holds a whole record); more than that, or every station, reads
#: from ``by_time.zarr``, whose chunks hold 366 days of every station.
_BY_STATION_LIMIT = 64

#: Store array -> archive CSV column. The store spells snow depth out; the
#: output contract keeps the two-letter names the CSV columns abbreviate.
_STORE_COLUMNS = {"swe": "wteq_cm", "snow_depth": "snwd_cm"}

#: How long a downloaded bundle is trusted before it is fetched again, when
#: the release it came from could not be identified (the API was unreachable)
#: and the cache file therefore cannot be named after its tag. A tag-named
#: file is immutable and never re-fetched.
ARCHIVE_MAX_AGE = 24 * 3600

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
            _nets.to_station_id(str(code), str(net))
            if str(net) in _nets.NETWORKS
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


def network_of(codes: list[str]) -> dict[str, str]:
    """``{code: network}`` from the inventory, for codes whose shape is ambiguous."""
    gdf = _read_inventory()
    found = gdf.reindex([str(c) for c in codes])["network"]
    return {str(code): str(net) for code, net in found.dropna().items()}


def manifest() -> dict[str, Any]:
    """``archive.json`` from the Pages store: when it was built, from which commit.

    Useful for a data-availability statement — the commit pins exactly which
    CSVs the store was built from — and for telling a stale Pages deploy from
    a stale bundle.
    """
    return _fetch_json(MANIFEST_URL)


def _fetch_json(url: str) -> dict[str, Any]:
    import requests  # noqa: PLC0415

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def latest_snapshot() -> dict[str, Any]:
    """The newest snapshot release of ``global_snow_networks``.

    ``{"tag", "published_at", "html_url", "tarball_url"}`` from the GitHub
    releases API — the tag is what a Zenodo version DOI corresponds to, so
    this is what to cite when a result came from the bundle. ``tarball_url``
    is the bundled-CSV asset; ``None`` when the release carries none.
    """
    release = _fetch_json(RELEASES_API)
    assets = release.get("assets") or []
    tarball = None
    for asset in assets:
        name = str(asset.get("name", ""))
        if name.startswith("all_station_csvs") and name.endswith(".tar.xz"):
            tarball = asset.get("browser_download_url")
            if name == "all_station_csvs.tar.xz":
                break
    return {
        "tag": str(release.get("tag_name") or ""),
        "published_at": str(release.get("published_at") or "")[:10],
        "html_url": str(release.get("html_url") or ""),
        "tarball_url": tarball,
    }


# ── reading the data ─────────────────────────────────────────────────────────


def _parse_csv(text: str, code: str) -> pd.DataFrame:
    """One station CSV as a date-indexed frame (``date,wteq_cm,snwd_cm``)."""
    frame = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    return frame.set_index("date").sort_index()


def _snapshot_tarball() -> tuple[str, dict[str, Any]]:
    """URL of the latest snapshot's bundle, and what is known about the release.

    When the releases API cannot be reached the fixed-name download URL is
    used blind — it still resolves to the newest release — and the returned
    metadata is empty.
    """
    try:
        info = latest_snapshot()
    except Exception as exc:  # noqa: BLE001 — the download URL works without it
        _logger.warning(
            "Could not read the latest snapshot release (%s); downloading the "
            "bundle without knowing its tag.",
            exc,
        )
        return ARCHIVE_URL, {}
    return info.get("tarball_url") or ARCHIVE_URL, info


def _from_tarball(
    codes: set[str] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Every wanted station's CSV, out of the latest snapshot's bundle.

    Downloaded into the package cache by pooch. A bundle whose release tag is
    known is cached under that tag and never re-fetched; one downloaded blind
    is re-fetched after :data:`ARCHIVE_MAX_AGE`. ``EASYSNOWDATA_CACHE_DIR``
    moves the cache root. Returns the frames and the release metadata.
    """
    url, info = _snapshot_tarball()
    tag = info.get("tag")
    path = providers.raster_http.fetch(
        url,
        f"all_station_csvs-{tag}.tar.xz" if tag else "all_station_csvs.tar.xz",
        subdir="stations",
        max_age=None if tag else ARCHIVE_MAX_AGE,
    )
    if tag:
        _logger.info(
            "Reading the bundled station archive of snapshot %s (%s) at %s; the "
            "daily archive on Pages may be ahead of it.",
            tag,
            info.get("published_at") or "date unknown",
            path,
        )
    else:
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
    return frames, {**info, "tarball_url": url}


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


def _open_store(url: str) -> xr.Dataset:
    """The store at *url*, which is a URL on Pages or a local path in tests.

    A path without a scheme is opened as an explicit local store: left to
    URL parsing, a Windows drive letter (``C:\\…``) reads as a protocol.
    """
    if "://" in url:
        return providers.zarr_cloud.open(url, consolidated=True)
    from zarr.storage import LocalStore  # noqa: PLC0415

    return xr.open_zarr(LocalStore(Path(url)), consolidated=True)


def _layout_for(codes: list[str] | None) -> str:
    """Which store answers a request cheapest (see :data:`_BY_STATION_LIMIT`)."""
    if codes is not None and len(codes) <= _BY_STATION_LIMIT:
        return "by_station"
    return "by_time"


def _from_zarr(
    codes: list[str] | None,
    types: list[str],
    window: tuple[Any, Any] | None,
    layout: str,
) -> xr.Dataset:
    """The wanted stations and days out of the Pages store, as a bare grid.

    Only the chunks the selection touches are fetched. The result has
    ``swe`` / ``snwd`` on ``(station, time)`` in float64 and nothing else;
    :func:`_finish` puts the inventory metadata and water-year coordinates
    on, the same way the CSV routes get them.
    """
    import numpy as np  # noqa: PLC0415

    url = ZARR_URLS[layout]
    _logger.info("Reading the chunked station archive at %s", url)
    store = _open_store(url)
    wanted = [
        name for name, column in _STORE_COLUMNS.items() if COLUMNS[column][0] in types
    ]
    ds = store[wanted]
    # reindex, not sel: a code the store lacks (the probe has not verified it)
    # becomes an all-NaN row, as on the CSV routes. With no codes named, the
    # station axis is sorted, as the tarball route sorts it.
    axis = (
        [str(c) for c in codes]
        if codes is not None
        else sorted(str(c) for c in ds["station"].values)
    )
    ds = ds.reindex(station=axis)
    if window is not None:
        start, end = window
        ds = ds.sel(time=slice(start, end))
    ds = ds.reset_coords(drop=True).load()
    ds = ds.rename({name: COLUMNS[_STORE_COLUMNS[name]][0] for name in wanted})
    ds = ds.astype("float64")
    times = pd.DatetimeIndex(ds["time"].values, name="time")
    return xr.Dataset(
        {name: (("station", "time"), ds[name].values) for name in ds.data_vars},
        coords={
            "station": np.array([str(c) for c in ds["station"].values], dtype="str"),
            "time": times,
        },
    )


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
        so a narrow window is cheap — on the bundle route it is still downloaded
        whole, because that is how it is published.
    networks
        Keep only these networks.
    source
        ``"github-pages-zarr"`` (default) reads the chunked store on that
        repo's Pages site and fetches only the chunks the request touches:
        named stations up to 64 come from ``by_station.zarr`` (a whole record
        is one chunk), anything wider from ``by_time.zarr`` (one water year of
        every station is one or two chunks). Its ``time`` axis is complete
        and daily between the first and last observation, and its values are
        stored as float32 (returned as float64), so they agree with the CSV
        routes to float32 precision. When the store cannot be read and no
        *source* was asked for, the snapshot route below is used instead,
        with a warning naming the snapshot.
        ``"github-tarball"`` downloads the ~28 MB bundle of CSVs attached to
        that repo's **latest snapshot release** — a few times a year, each
        with a Zenodo DOI — and reads every wanted CSV out of it. Ask for it
        when a result should be pinned to a citable snapshot; the Dataset's
        ``snapshot_tag`` and ``snapshot_published_at`` say which one. Its
        ``time`` axis holds only days on which some station observed.
        ``"github-csv"`` fetches one CSV per station instead, which needs
        neither a store nor a temporary file.
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
    all of it materializes roughly 1.2 GB of float64 (measured on the tarball
    route: 31 s and under 2 GB of peak memory). Pass *time*, *aoi* or
    *stations* when you do not need every cell — on the default route that
    also cuts the download to the chunks touched, measured at 0.7 MB for one
    water year of every station and 0.5 MB for one station's whole record.

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

    window = None
    if time is not None:
        from easysnowdata.temporal import parse_time  # noqa: PLC0415

        window = parse_time(time)

    source_url: str
    if src.id == "github-pages-zarr":
        layout = _layout_for(codes)
        try:
            grid = _from_zarr(codes, wanted_types, window, layout)
        except Exception as exc:  # noqa: BLE001 — any read failure: fall back
            if source is not None:
                raise
            # The store is a build artefact of a Pages deploy; the bundle is
            # a GitHub Release asset, served from elsewhere. So a Pages outage
            # (or a deploy that has not happened yet) degrades to the latest
            # snapshot — possibly months behind, and the warning says so —
            # rather than to an error (REVAMP_PLAN §9.3).
            src = product.source("github-tarball")
            frames, snapshot = _from_tarball(set(codes) if codes is not None else None)
            _logger.warning(
                "Could not read the chunked archive at %s (%s); read the bundled "
                "CSVs of snapshot %s (%s) instead, which may be behind the daily "
                "archive.",
                ZARR_URLS[layout],
                exc,
                snapshot.get("tag") or "unknown",
                snapshot.get("published_at") or "date unknown",
            )
            ds = _to_dataset(frames, wanted_types, codes, inv, hemisphere, window)
            source_url = snapshot["tarball_url"]
            ds.attrs.update(_snapshot_attrs(snapshot))
        else:
            ds = _finish(grid, inv, hemisphere)
            source_url = ZARR_URLS[layout]
    elif src.id == "github-csv":
        if codes is None:
            raise ValueError(
                'source="github-csv" fetches one CSV per station, so it needs '
                "stations= or aoi=. Use the default route for the whole archive."
            )
        frames = _from_csvs(codes)
        ds = _to_dataset(frames, wanted_types, codes, inv, hemisphere, window)
        source_url = CSV_BASE
    else:
        frames, snapshot = _from_tarball(set(codes) if codes is not None else None)
        ds = _to_dataset(frames, wanted_types, codes, inv, hemisphere, window)
        source_url = snapshot["tarball_url"]
        ds.attrs.update(_snapshot_attrs(snapshot))

    ds.attrs.update(contract.provenance(product, src, source_url=source_url))
    ds.attrs["interval"] = "daily"
    return ds


def _snapshot_attrs(snapshot: dict[str, Any]) -> dict[str, str]:
    """Which snapshot release a bundle-route Dataset came from, for its attrs."""
    return {
        "snapshot_tag": str(snapshot.get("tag") or ""),
        "snapshot_published_at": str(snapshot.get("published_at") or ""),
        "snapshot_url": str(snapshot.get("html_url") or ""),
    }


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
    return _finish(ds, inv, hemisphere)


def _finish(
    ds: xr.Dataset, inv: gpd.GeoDataFrame | None, hemisphere: str
) -> xr.Dataset:
    """Variable attributes, station metadata and water-year coordinates.

    The last step of every route, so a Dataset from the store and one from
    the CSVs are the same shape with the same coordinates.
    """
    for column, (type_name, units) in COLUMNS.items():
        if type_name in ds.data_vars:
            ds[type_name].attrs.update(
                {
                    "units": units,
                    "long_name": _nets.TYPES[type_name][1],
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
        "station inventory as GeoJSON, a chunked Zarr store on its Pages site "
        "that a query reads only the touched chunks of, and one CSV per "
        "station bundled into a single ~28 MB archive on each snapshot release. "
        "The fast path for "
        "'everything daily' without hitting five APIs, refreshed daily. Most "
        "stations start around 1980; "
        "a few long snow courses reach back to 1896. SWE and snow depth only; for "
        "other variables, other intervals or quality flags, use "
        "easysnowdata.stations.load()."
    ),
    sources=(
        Source(
            id="github-pages-zarr",
            provider="zarr_cloud",
            location=PAGES_BASE,
            extent="western US, western Canada, Norway, Yukon, California, BC",
            temporal="1896/present (most stations from ~1980)",
            latency="daily rebuild",
            notes=(
                "Zarr v3 on GitHub Pages, two chunk layouts; a query fetches "
                "only the chunks it touches (one water year of every station "
                "~0.7 MB, one station's whole record ~0.5 MB)"
            ),
            title="global_snow_networks chunked archive (Pages)",
            health=(
                Probe(
                    "Snow station Zarr archive (global_snow_networks)",
                    partial(
                        health.http_first_byte, f"{ZARR_URLS['by_time']}/zarr.json"
                    ),
                ),
            ),
        ),
        Source(
            id="github-tarball",
            provider="raster_http",
            location=ARCHIVE_URL,
            extent="western US, western Canada, Norway, Yukon, California, BC",
            temporal="1896/present (most stations from ~1980)",
            latency="per snapshot release (a few times a year)",
            notes=(
                "one ~28 MB download holding every station CSV of the latest "
                "snapshot release (each has a Zenodo DOI); cached per tag. The "
                "citable route, and the fallback when the Pages store cannot "
                "be read"
            ),
            title="global_snow_networks snapshot bundle (latest release)",
            # The inventory and CSV probes were once labelled "SNOTEL/CCSS …"
            # after the retired snotel_ccss_stations entry; the labels were
            # renamed in 0.3 and data_status/history.json rewritten with them,
            # so their history since 2026-06 is continuous.
            health=(
                Probe(
                    "Snow station inventory (global_snow_networks)",
                    partial(health.http_first_byte, INVENTORY_URL),
                ),
                Probe(
                    "Snow station archive tarball (global_snow_networks)",
                    partial(health.http_first_byte, RELEASES_API),
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
                "Snow station CSV (global_snow_networks)",
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
    examples=("stations/plot_station_archive.py", "stations/plot_all_networks.py"),
    references=(REPO, f"{REPO}/blob/main/DESIGN.md"),
    tags=("swe", "snow depth", "stations", "archive", "daily"),
)

catalog.register(PRODUCT, replace=True)
