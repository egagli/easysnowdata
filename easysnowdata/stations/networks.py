"""The five snow-station networks, and how a station code maps to one of them.

Each :class:`Network` ties a catalog source id to the vendored client that
speaks to it (:mod:`easysnowdata.stations.clients`) and to the spelling of a
station identifier in that network. Two spellings matter:

``code``
    The globally unique code the inventory and the archive use, e.g.
    ``"679_WA_SNTL"``. Unique across all five networks.
``station_id``
    What the network's own API calls the station, e.g. the AWDB triplet
    ``"679:WA:SNTL"``. Only AWDB's two spellings differ; for the other four
    the code *is* the station id.

:func:`split_by_network` turns a mix of codes, station ids and inventory rows
into ``{network_id: [station_id, …]}``, which is what :func:`easysnowdata.stations.load`
fans out over.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NETWORKS",
    "Network",
    "TYPES",
    "canonical_units",
    "get",
    "split_by_network",
    "to_code",
    "to_station_id",
]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Network:
    """One snow-station network and the client that reads it."""

    id: str
    title: str
    client: str
    """Class name in :mod:`easysnowdata.stations.clients`."""
    requires: tuple[str, ...]
    """Auth providers the API needs (§5)."""
    extent: str
    data_provider: str
    api_url: str
    #: Key in the client's station dicts that holds the native station id.
    id_key: str
    #: True when ``code`` is the station id with ``:`` swapped for ``_``.
    colon_code: bool = False
    #: Regex a *code* of this network matches; ``None`` when it cannot be
    #: told apart from another network's by shape alone.
    code_pattern: str | None = None

    def client_class(self) -> Any:
        """Import and return the client class."""
        from easysnowdata.stations import clients  # noqa: PLC0415

        return getattr(clients, self.client)

    def station_id(self, station: str) -> str:
        """The native station id for *station* (a code or an id already)."""
        return station.replace("_", ":") if self.colon_code else station

    def code(self, station_id: str) -> str:
        """The globally unique code for a native *station_id*."""
        return station_id.replace(":", "_") if self.colon_code else station_id


NETWORKS: dict[str, Network] = {
    "awdb": Network(
        id="awdb",
        title="USDA NRCS AWDB (SNOTEL, SCAN, snow courses)",
        client="AWDBClient",
        requires=(),
        extent="western US, western Canada",
        data_provider="USDA NRCS National Water and Climate Center",
        api_url="https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1",
        id_key="stationTriplet",
        colon_code=True,
        # "679_WA_SNTL" / "679:WA:SNTL" — three parts, the last a network code
        code_pattern=r"^[A-Za-z0-9]+[:_][A-Z]{2}[:_][A-Z]+$",
    ),
    "cdec": Network(
        id="cdec",
        title="California Data Exchange Center (CCSS)",
        client="CDECClient",
        requires=(),
        extent="California",
        data_provider="California Department of Water Resources",
        api_url="https://cdec.water.ca.gov",
        id_key="station_id",
        # CDEC ids ("QUA") and DataBC ids ("1A01P") overlap, so neither can be
        # told from the other by shape; both are resolved by lookup instead.
        code_pattern=None,
    ),
    "databc": Network(
        id="databc",
        title="BC automated snow weather stations and manual snow surveys",
        client="DataBCClient",
        requires=(),
        extent="British Columbia",
        data_provider="BC Ministry of Environment / BC Data Catalogue",
        api_url="https://openmaps.gov.bc.ca/geo/pub",
        id_key="location_id",
        code_pattern=None,
    ),
    "nve": Network(
        id="nve",
        title="NVE HydAPI (Norway)",
        client="NVEClient",
        requires=("nve",),
        extent="Norway",
        data_provider="Norwegian Water Resources and Energy Directorate",
        api_url="https://hydapi.nve.no/api/v1",
        id_key="station_id",
        # "12.142.0" — dotted, digits only
        code_pattern=r"^\d+(\.\d+)+$",
    ),
    "yukon": Network(
        id="yukon",
        title="Yukon Water Data (AquaCache)",
        client="YukonClient",
        requires=(),
        extent="Yukon and neighbouring BC/Alaska courses",
        data_provider="Government of Yukon",
        api_url="https://service.yukon.ca/water-data/api/v1",
        id_key="station_id",
        # "09AA-M1", "08AA-SC01", "10MD-MET-00004" — hyphenated
        code_pattern=r"^[0-9]{2}[A-Z]{2}-[A-Z0-9-]+$",
    ),
}

#: Standardized variable type -> (canonical units, long name). The vocabulary
#: and the units are ``global_snow_networks``' DESIGN.md §3.2 and §3.5.
TYPES: dict[str, tuple[str, str]] = {
    "swe": ("cm", "snow water equivalent"),
    "snwd": ("cm", "snow depth"),
    "temp": ("°C", "air temperature"),
    "temp_max": ("°C", "maximum air temperature"),
    "temp_min": ("°C", "minimum air temperature"),
    "precip": ("mm", "precipitation"),
    "rh": ("%", "relative humidity"),
    "wind_spd": ("km/h", "wind speed"),
    "wind_gust": ("km/h", "wind gust speed"),
    "wind_dir": ("degrees", "wind direction"),
    "wind_run": ("km", "wind run"),
    "solar": ("W/m²", "solar radiation"),
    "baro": ("hPa", "barometric pressure"),
    "density": ("%", "snow density"),
    "snow_line": ("m", "snow line elevation"),
    "soil_moisture": ("%", "soil moisture"),
    "other": ("", "other measurement"),
}

#: Conversions that are safe *for a given type*, keyed ``(type, from, to)``.
#:
#: Keyed by type on purpose. A length conversion that is right for one
#: quantity is wrong for another: ``swe`` in cm and in m are the same depth of
#: water, but ``precip`` in mm (depth of water) and in cm (depth of new
#: snowfall, which is what Yukon's ``precip_snow_cm`` measures) are different
#: quantities, and multiplying one by ten to look like the other would report
#: 5 cm of snow as 50 mm of water. Anything not listed here raises rather than
#: being fabricated.
_UNIT_FACTORS: dict[tuple[str, str, str], float] = {
    # Barometric pressure: the same quantity, and Yukon emits kPa where
    # DataBC and DESIGN.md §3.5 use hPa.
    ("baro", "kPa", "hPa"): 10.0,
    ("baro", "hPa", "kPa"): 0.1,
    # Snow water equivalent and snow depth: plain depth, safely rescaled.
    ("swe", "m", "cm"): 100.0,
    ("swe", "cm", "m"): 0.01,
    ("swe", "mm", "cm"): 0.1,
    ("swe", "cm", "mm"): 10.0,
    ("snwd", "m", "cm"): 100.0,
    ("snwd", "cm", "m"): 0.01,
    ("snwd", "mm", "cm"): 0.1,
    ("snwd", "cm", "mm"): 10.0,
}


def canonical_units(type_name: str) -> str:
    """The unit :func:`easysnowdata.stations.load` emits for *type_name*."""
    return TYPES.get(type_name, ("", ""))[0]


def unit_factor(
    type_name: str, from_units: str, to_units: str, *, variables: str = ""
) -> float:
    """Factor converting *from_units* to *to_units* for *type_name*.

    Raises when the conversion is not one this package considers safe for that
    type — which includes conversions that are arithmetically obvious but
    physically wrong, such as treating a depth of snowfall as a depth of water.
    """
    if from_units == to_units or not from_units or not to_units:
        return 1.0
    try:
        return _UNIT_FACTORS[type_name, from_units, to_units]
    except KeyError:
        detail = f" ({variables})" if variables else ""
        raise ValueError(
            f"{type_name!r} came back in both {from_units!r} and "
            f"{to_units!r}{detail}, and converting between them is not safe "
            f"for {type_name!r} — they may not measure the same quantity. "
            "Request one native variable by name instead of the standardized "
            "type, or use the clients directly."
        ) from None


def get(network: str) -> Network:
    """Return the :class:`Network` called *network*."""
    try:
        return NETWORKS[network]
    except KeyError:
        raise ValueError(
            f"Unknown station network {network!r}; available: {', '.join(NETWORKS)}."
        ) from None


def to_station_id(station: str, network: str) -> str:
    """The native station id of *station* in *network*."""
    return get(network).station_id(str(station))


def to_code(station_id: str, network: str) -> str:
    """The globally unique code of *station_id* in *network*."""
    return get(network).code(str(station_id))


def guess_network(station: str) -> str | None:
    """The network a code belongs to when its shape gives it away, else ``None``.

    CDEC and DataBC codes overlap (``"1A01P"`` is a valid spelling of both),
    so a short alphanumeric code returns ``None`` and has to be looked up.
    """
    text = str(station).strip()
    for net in NETWORKS.values():
        if net.code_pattern and re.match(net.code_pattern, text):
            return net.id
    return None


def _from_frame(stations: Any) -> dict[str, list[str]] | None:
    """``{network: [station_id]}`` from an inventory frame, or ``None``."""
    columns = getattr(stations, "columns", None)
    if columns is None:
        return None
    # Prefer whichever column actually holds network ids. `network` is this
    # package's name for the access path and `client` is global_snow_networks',
    # but a frame can carry a `network` column meaning something else — the
    # published inventory's Yukon display name, or the "SNOTEL"/"CCSS" labels
    # the automatic_weather_stations shim restores.
    candidates = [c for c in ("network", "client") if c in columns]
    column = next(
        (c for c in candidates if set(stations[c].dropna()) <= set(NETWORKS)),
        None,
    )
    if column is None:
        column = candidates[0] if candidates else None
    if column is None:
        return None
    index = stations.index
    codes = index if index.name in ("code", "station") else stations.get("code", index)
    grouped: dict[str, list[str]] = {}
    for code, network in zip(codes, stations[column], strict=True):
        if network not in NETWORKS:
            raise ValueError(
                f"Station {code!r} names network {network!r}, which is not one "
                f"of {', '.join(NETWORKS)}."
            )
        grouped.setdefault(network, []).append(to_station_id(str(code), network))
    return grouped


def split_by_network(
    stations: Any,
    *,
    network: str | None = None,
    lookup: Any = None,
) -> dict[str, list[str]]:
    """Group *stations* by network, as ``{network_id: [station_id, …]}``.

    Parameters
    ----------
    stations
        A code or station id, a sequence of them, or a station frame carrying
        a ``network``/``client`` column (what :func:`easysnowdata.stations.inventory`
        returns) — in which case that column decides and nothing is guessed.
    network
        Pin every station to this network instead of working it out.
    lookup
        Callable taking the codes that could not be told apart by shape and
        returning ``{code: network}``. :func:`easysnowdata.stations.load`
        passes the archive inventory's lookup; without one, an ambiguous code
        raises.
    """
    if network is not None:
        net = get(network).id
        return {net: [to_station_id(str(s), net) for s in _as_list(stations)]}
    from_frame = _from_frame(stations)
    if from_frame is not None:
        return from_frame
    items = _as_list(stations)
    grouped: dict[str, list[str]] = {}
    unknown: list[str] = []
    for item in items:
        guess = guess_network(item)
        if guess is None:
            unknown.append(str(item))
        else:
            grouped.setdefault(guess, []).append(to_station_id(str(item), guess))
    if unknown:
        if lookup is None:
            raise ValueError(
                f"Cannot tell which network these station codes belong to: "
                f"{', '.join(sorted(unknown))}. CDEC and DataBC codes have the "
                "same shape, so pass networks=… , or hand load() the frame "
                "inventory() returned instead of bare codes."
            )
        found = lookup(unknown)
        missing = [code for code in unknown if code not in found]
        if missing:
            raise ValueError(
                f"Unknown station code(s): {', '.join(sorted(missing))}. They "
                "are in none of the five networks' inventories."
            )
        for code in unknown:
            net = found[code]
            grouped.setdefault(net, []).append(to_station_id(code, net))
    return grouped


def _as_list(stations: Any) -> list[str]:
    if isinstance(stations, str):
        return [stations]
    index = getattr(stations, "index", None)
    if index is not None and not hasattr(stations, "__iter__"):
        return [str(s) for s in index]
    if hasattr(stations, "columns"):  # a frame without a network column
        return [str(s) for s in stations.index]
    return [str(s) for s in stations]
