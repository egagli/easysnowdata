"""``inventory()`` and ``load()``: the xarray/GeoDataFrame layer over the clients.

The clients answer in dict records (``global_snow_networks``' DESIGN.md §3.4)
because that repo's pipeline wants them that way. Library users want the
package's own return types, so this module is the only place that knows both:
it takes an ``aoi`` and a ``time`` like every other loader (§2.1, §2.2), fans
the request out over the networks the stations belong to, and brings the
answer onto the output contract.

The ``networks`` module is imported under an alias throughout, because
``networks`` is also the name of the public keyword argument §9 step 2 gives
:func:`inventory` and :func:`load`.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import auth, catalog
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.stations import _frames
from easysnowdata.stations import networks as _nets
from easysnowdata.temporal import parse_time

__all__ = ["PRODUCT_IDS", "inventory", "load", "metadata"]

_logger = logging.getLogger(__name__)

#: Catalog product id per network.
PRODUCT_IDS = {
    "awdb": "awdb-stations",
    "cdec": "cdec-stations",
    "databc": "databc-stations",
    "nve": "nve-stations",
    "yukon": "yukon-stations",
}

_DEFAULT_VARIABLES = ("swe", "snwd")


def _network_ids(names: Any) -> list[str]:
    if names is None:
        return list(_nets.NETWORKS)
    if isinstance(names, str):
        names = [names]
    return [_nets.get(name).id for name in names]


def _bbox(aoi: Any) -> tuple[float, float, float, float] | None:
    """The clients' ``(min_lon, min_lat, max_lon, max_lat)``, or ``None``."""
    if aoi is None:
        return None
    parsed = parse_aoi(aoi)
    if parsed.is_global:
        return None
    return tuple(float(v) for v in parsed.bounds)  # type: ignore[return-value]


def _client(network: str) -> Any:
    """An initialised client for *network*, credentials ensured first (§5.2)."""
    net = _nets.get(network)
    product = catalog.get(PRODUCT_IDS[network])
    ensure_source(product, resolve_source(product, net.id))
    return net.client_class()()


def _archive_lookup(codes: list[str]) -> dict[str, str]:
    """``{code: network}`` for codes whose shape does not give the network away.

    The published inventory is the cheap way to answer this — one request that
    already knows every code. When it cannot be reached, fall back to asking
    the candidate networks themselves, so an unreachable
    ``global_snow_networks`` degrades a shortcut rather than breaking the load.
    """
    from easysnowdata.stations import archive  # noqa: PLC0415

    try:
        found = archive.network_of(codes)
    except Exception as exc:  # noqa: BLE001 — the live networks can still answer
        _logger.warning(
            "Could not read the station inventory to resolve %s (%s); asking "
            "the networks whose codes have that shape instead.",
            ", ".join(sorted(codes)),
            exc,
        )
        found = {}
    missing = [code for code in codes if code not in found]
    if missing:
        found.update(_lookup_by_asking_networks(missing))
    return found


#: Networks whose station codes cannot be told apart by shape (see
#: :func:`easysnowdata.stations.networks.guess_network`).
_AMBIGUOUS_NETWORKS = ("cdec", "databc")


def _lookup_by_asking_networks(codes: list[str]) -> dict[str, str]:
    """``{code: network}`` by listing the networks whose codes share a shape."""
    found: dict[str, str] = {}
    wanted = set(codes)
    for network in _AMBIGUOUS_NETWORKS:
        if not wanted:
            break
        try:
            client = _client(network)
            with auth.env(*_nets.get(network).requires):
                stations = client.get_all_stations()
        except Exception as exc:  # noqa: BLE001 — try the next network
            _logger.warning("Could not list %s stations (%s).", network, exc)
            continue
        net = _nets.get(network)
        for station in stations:
            code = station.get(net.id_key) or station.get("station_id")
            if code is not None and str(code) in wanted:
                found[str(code)] = network
                wanted.discard(str(code))
    return found


def _metadata_frame(codes: list[str]) -> gpd.GeoDataFrame | None:
    """Metadata for the stations being loaded, from the published inventory."""
    from easysnowdata.stations import archive  # noqa: PLC0415

    try:
        return archive.inventory(daily_only=False).reindex(codes)
    except Exception as exc:  # noqa: BLE001 — metadata is a nicety, not the data
        _logger.warning(
            "Could not read the station inventory for the metadata "
            "coordinates (%s); the Dataset carries only `network`.",
            exc,
        )
        return None


def _version() -> str:
    from easysnowdata import __version__  # noqa: PLC0415

    return __version__


# ── inventory ────────────────────────────────────────────────────────────────


def inventory(
    aoi: Any = None,
    *,
    networks: Any = None,
    daily_only: bool = False,
    active_only: bool = False,
    source: str = "archive",
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Snow stations as a GeoDataFrame, indexed by their global station code.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts; ``None`` asks for
        every station in the chosen networks.
    networks
        Which networks to list — any of ``"awdb"``, ``"cdec"``, ``"databc"``,
        ``"nve"``, ``"yukon"``. ``None`` (default) means all five.
    daily_only
        Keep only stations with a daily-or-better SWE or snow-depth record.
        On ``source="archive"`` this is the **probe-verified** verdict
        (``global_snow_networks``' DESIGN.md §4: a station counts as daily
        only once the pipeline has actually retrieved daily values). The live
        route has no such verdict, so there it filters on what each network
        *advertises* about itself, and the frame's ``daily`` column and
        ``daily_flag`` attribute say so.
    active_only
        Keep only stations the network reports as active. Inactive stations
        with a long record are usually still worth having.
    source
        ``"archive"`` (default) reads the daily-refreshed inventory
        ``global_snow_networks`` publishes: one HTTP request, normalized
        columns and probe-verified daily flags, at most a day old.
        ``"clients"`` asks the five APIs for today's station list instead —
        five sweeps, slower, and NVE needs its key.
    **kwargs
        On the archive route, passed to ``geopandas.read_file`` (``rows=``,
        ``columns=``, …).

    Returns
    -------
    geopandas.GeoDataFrame
        Stations in EPSG:4326 indexed by ``code`` (``"679_WA_SNTL"``), with
        the native ``station_id`` and the columns listed in
        :data:`easysnowdata.stations._frames.INVENTORY_COLUMNS`.

    Examples
    --------
    >>> import easysnowdata as esd
    >>> aoi = (-121.94, 46.72, -121.54, 46.99)
    >>> esd.stations.inventory(aoi, daily_only=True)          # doctest: +SKIP
    """
    wanted = _network_ids(networks)
    if source == "archive":
        from easysnowdata.stations import archive  # noqa: PLC0415

        return archive.inventory(
            aoi,
            networks=wanted,
            daily_only=daily_only,
            active_only=active_only,
            **kwargs,
        )
    if source != "clients":
        raise ValueError(
            f"Unknown inventory source {source!r}; use 'archive' or 'clients'."
        )
    bbox = _bbox(aoi)
    frames = []
    for network in wanted:
        client = _client(network)
        with auth.env(*_nets.get(network).requires):
            stations = client.get_all_stations(active_only=active_only, bbox=bbox)
        _logger.info("%s: %d stations", network, len(stations))
        frame = _frames.stations_to_geodataframe(stations, network)
        with auth.env(*_nets.get(network).requires):
            frame["daily"] = _advertised_daily(stations, network, client)
        frames.append(frame)
    gdf = (
        pd.concat(frames)
        if frames
        else _frames.stations_to_geodataframe([], "awdb").assign(daily=None)
    )
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    if daily_only:
        gdf = gdf[gdf["daily"].fillna(False).astype(bool)]
    gdf.attrs.update(
        {
            "source": "clients",
            "source_title": "the network APIs, live",
            "networks": " ".join(wanted),
            "daily_flag": "advertised (not probe-verified; see source='archive')",
            "easysnowdata_version": _version(),
        }
    )
    return gdf


#: AWDB element codes for SWE and snow depth, the two the archive is about.
_AWDB_SNOW_ELEMENTS = ("WTEQ", "SNWD")


def _awdb_daily_triplets(client: Any, stations: list[dict]) -> set[str] | None:
    """Triplets whose AWDB element inventory lists a DAILY WTEQ or SNWD.

    AWDB's station list says nothing about elements, so this is one extra
    ``/stations`` call per 150 triplets with ``returnStationElements`` — about
    50 requests and 50 s for the whole network (measured 2026-09-22: 1 217 of
    7 146 stations). ``None`` when the calls fail, so the flag degrades to
    "unknown" rather than the inventory failing.
    """
    triplets = [str(s["stationTriplet"]) for s in stations if s.get("stationTriplet")]
    if not triplets:
        return set()
    try:
        metadata = client.get_metadata(
            triplets, elements=list(_AWDB_SNOW_ELEMENTS), durations=["DAILY"]
        )
    except Exception as exc:  # noqa: BLE001 — a hint, not the verdict
        _logger.warning(
            "Could not read AWDB station elements (%s); the `daily` flag is "
            "unknown for AWDB stations.",
            exc,
        )
        return None
    if isinstance(metadata, dict):
        metadata = [metadata]
    return {
        str(m["stationTriplet"])
        for m in metadata
        if m.get("stationTriplet") and m.get("stationElements")
    }


def _advertised_daily(
    stations: list[dict], network: str, client: Any = None
) -> list[Any]:
    """What each network's station list *says* about daily SWE / snow depth.

    Advertised, never verified — DESIGN.md §4 is explicit that capability
    flags are hints. Four networks answer from their station list alone. AWDB
    does not carry elements there, so given a *client* the flag comes from
    one batched element-inventory call (:func:`_awdb_daily_triplets`);
    without one, or when that call fails, it is ``None``.
    """
    out: list[Any] = []
    if network == "awdb":
        daily = _awdb_daily_triplets(client, stations) if client is not None else None
        for station in stations:
            out.append(
                None if daily is None else str(station.get("stationTriplet")) in daily
            )
        return out
    for station in stations:
        if network == "cdec":
            out.append(
                bool(station.get("has_daily_swe") or station.get("has_daily_snwd"))
            )
        elif network == "nve":
            out.append(bool(set(station.get("daily_parameters") or ()) & {2002, 2003}))
        elif network == "yukon":
            out.append(
                any(
                    str(s.get("interval", "")) in ("daily", "hourly", "sub_daily")
                    for s in (station.get("series") or ())
                    if isinstance(s, dict)
                )
            )
        elif network == "databc":
            out.append(str(station.get("station_type", "")).upper() == "ASWS")
        else:
            out.append(None)
    return out


def metadata(stations: Any, *, networks: Any = None) -> dict[str, Any]:
    """Full per-station metadata from the owning network's API.

    A near-pass-through to each client's ``get_metadata``; returns
    ``{code: metadata dict}``, including the station's variable inventory.
    """
    grouped = _split(stations, networks)
    out: dict[str, Any] = {}
    for net, ids in grouped.items():
        client = _client(net)
        with auth.env(*_nets.get(net).requires):
            for station_id in ids:
                out[_nets.to_code(station_id, net)] = _one_metadata(
                    client.get_metadata(station_id), net, station_id
                )
    return out


def _one_metadata(value: Any, network: str, station_id: str) -> Any:
    """One station's metadata as a dict, whatever shape the client used.

    DESIGN.md §3.4 says ``get_metadata(station_id) -> dict`` and all five
    clients now do that — AWDB used to return a one-element list and was fixed
    upstream. This stays as a cheap guard so a client that regresses, or a
    future one that batches, cannot leak two shapes to callers.
    """
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        _logger.warning(
            "%s returned %d metadata records for %s; DESIGN.md §3.4 expects "
            "one dict, so the list is passed through unchanged.",
            network,
            len(value),
            station_id,
        )
    return value


def _split(stations: Any, names: Any) -> dict[str, list[str]]:
    """``{network: [station_id]}`` for *stations*, honouring a pinned network."""
    if names is None:
        return _nets.split_by_network(stations, lookup=_archive_lookup)
    wanted = _network_ids(names)
    if len(wanted) == 1:
        return _nets.split_by_network(stations, network=wanted[0])
    grouped = _nets.split_by_network(stations, lookup=_archive_lookup)
    return {net: ids for net, ids in grouped.items() if net in wanted}


# ── load ─────────────────────────────────────────────────────────────────────


def load(
    stations: Any = None,
    *,
    aoi: Any = None,
    variables: Any = _DEFAULT_VARIABLES,
    time: Any = None,
    interval: str = "daily",
    networks: Any = None,
    include_flags: bool = False,
    hemisphere: str = "northern",
    daily_only: bool = True,
) -> xr.Dataset:
    """Station observations as a ``(station, time)`` Dataset.

    Parameters
    ----------
    stations
        Station codes (``"679_WA_SNTL"``), native station ids
        (``"679:WA:SNTL"``), or the frame :func:`inventory` returned — which
        is the cheapest form, because it already says which network each
        station belongs to. ``None`` reads every station in *aoi*.
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. Picks the
        stations when *stations* is ``None``; ignored otherwise.
    variables
        Standardized types (``"swe"``, ``"snwd"``, ``"temp"``, … — the
        vocabulary in :data:`easysnowdata.stations.networks.TYPES`) or a
        network's own native names. Default: SWE and snow depth. The returned
        data variables are always named for the *type*, whichever spelling
        went in; the native names that fed each one are in its
        ``native_variables`` attribute.
    time
        Any form :func:`easysnowdata.temporal.parse_time` accepts:
        ``"2023-10"``, ``("2023-10-01", "2024-06-30")``, ``"2023-10/2024-06"``.
        ``None`` asks each network for its whole record.
    interval
        ``"daily"`` (default), ``"hourly"``, ``"monthly"``, … — the shared
        interval vocabulary. A network that cannot serve it raises its own
        ``{Client}Error`` rather than quietly resampling.
    networks
        Pin every station to this network (or these networks) instead of
        working out which one each code belongs to.
    include_flags
        Ask the clients for per-value quality flags. The Dataset keeps the
        values; use the clients directly when you need the flags themselves.
    hemisphere
        Which water year to attach: ``"northern"`` (1 October, default) or
        ``"southern"`` (1 April).
    daily_only
        When the stations come from *aoi* rather than being named, keep only
        the daily-or-better ones. Reading every periodic snow course in a
        large AOI is rarely what anyone wants.

    Returns
    -------
    xarray.Dataset
        Dims ``(station, time)``; one data variable per standardized type,
        each carrying ``units``, ``long_name``, ``native_variables`` and
        ``interval``; station metadata (name, latitude, longitude,
        elevation_m, network, …) as non-dimension coordinates on ``station``;
        ``water_year`` and ``dowy`` on ``time``.

    Examples
    --------
    >>> import easysnowdata as esd
    >>> obs = esd.stations.load(["679_WA_SNTL", "642_WA_SNTL"],
    ...                         variables=["swe", "snwd"],
    ...                         time="2023-10/2024-06")       # doctest: +SKIP
    """
    inv: gpd.GeoDataFrame | None = None
    if stations is None:
        if aoi is None:
            raise ValueError(
                "load() needs either stations= (codes, ids or an inventory "
                "frame) or aoi= to choose them with."
            )
        try:
            inv = inventory(aoi, networks=networks, daily_only=daily_only)
        except Exception as exc:  # noqa: BLE001 — the networks can list themselves
            _logger.warning(
                "Could not read the station inventory to choose stations for "
                "this AOI (%s); asking the networks directly instead.",
                exc,
            )
            inv = inventory(
                aoi, networks=networks, daily_only=daily_only, source="clients"
            )
        if inv.empty:
            raise ValueError("No stations found in that area of interest.")
        stations = inv
    elif hasattr(stations, "columns"):
        inv = stations

    grouped = _split(stations, networks)
    begin, end = _window(time)
    wanted = _variable_names(variables)

    records: dict[str, list[dict]] = {}
    products: list[Any] = []
    sources: list[Any] = []
    for network, ids in grouped.items():
        client = _client(network)
        product = catalog.get(PRODUCT_IDS[network])
        products.append(product)
        sources.append(resolve_source(product, network))
        _logger.info(
            "%s: fetching %s for %d station(s), %s to %s",
            network,
            ", ".join(wanted),
            len(ids),
            begin or "start of record",
            end or "today",
        )
        with auth.env(*_nets.get(network).requires):
            records[network] = client.get_data(
                station_ids=ids,
                variables=list(wanted),
                begin_date=begin,
                end_date=end,
                interval=interval,
                include_flags=include_flags,
            )

    codes = [
        _nets.to_code(station_id, network)
        for network, ids in grouped.items()
        for station_id in ids
    ]
    if inv is None:
        inv = _metadata_frame(codes)
    ds = _frames.records_to_dataset(
        records, metadata=inv, hemisphere=hemisphere, stations=codes
    )
    return _finish(ds, products, sources, interval=interval)


def _variable_names(variables: Any) -> tuple[str, ...]:
    if variables is None:
        return _DEFAULT_VARIABLES
    if isinstance(variables, str):
        return (variables,)
    return tuple(str(v) for v in variables)


def _window(time: Any) -> tuple[str | None, str | None]:
    """``time`` as the clients' ``(begin_date, end_date)`` day strings."""
    if time is None:
        return None, None
    start, end = parse_time(time)
    return (
        None if start is None else pd.Timestamp(start).strftime("%Y-%m-%d"),
        None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
    )


def _finish(
    ds: xr.Dataset, products: list[Any], sources: list[Any], *, interval: str
) -> xr.Dataset:
    """Provenance attrs (§2.5) for one network, or for several.

    With a single network the attrs are the usual ones, so
    ``ds.attrs["source"]`` hands straight back to ``load(networks=...)``.
    Across networks the per-station truth is the ``network`` coordinate, and
    the joined ids are recorded here so nothing is lost.
    """
    if len(products) == 1:
        ds.attrs.update(contract.provenance(products[0], sources[0]))
    elif products:
        ds.attrs.update(
            {
                "source": " ".join(s.id for s in sources),
                "source_id": " ".join(s.id for s in sources),
                "source_title": "; ".join(s.title for s in sources),
                "source_url": " ".join(s.location for s in sources),
                "product_id": " ".join(p.id for p in products),
                "title": "Snow station observations",
                "data_citation": " ".join(p.citation for p in products),
                "license": "; ".join(sorted({p.license for p in products})),
                "easysnowdata_version": _version(),
            }
        )
    ds.attrs["interval"] = interval
    return ds
