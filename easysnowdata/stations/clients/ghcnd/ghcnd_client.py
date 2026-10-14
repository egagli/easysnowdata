"""
clients/ghcnd/ghcnd_client.py
=============================
Python client for GHCN-Daily — NOAA NCEI's Global Historical Climatology
Network daily station archive.

Primary use case: global daily snow depth (``SNWD``) and, at a smaller set
of stations, daily snow water equivalent (``WESD``) from national
meteorological services, the U.S. COOP and CoCoRaHS networks, ECA&D and
others.  GHCN-D also mirrors SNOTEL (network code ``S``), so its U.S.
SNOTEL stations duplicate what the AWDB client already serves.

Data source       : https://www.ncei.noaa.gov/pub/data/ghcn/daily/
Format reference  : https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
Authentication    : none
Licence           : none stated in ``readme.txt``; NCEI asks users to cite
                    Menne et al. (2012), doi:10.1175/JTECH-D-11-00103.1, and
                    the dataset version, doi:10.7289/V5D21VHZ.

Files used
----------
- ``ghcnd-stations.txt``  — fixed-width station list (id, lat, lon,
  elevation, state, name, GSN/HCN flags, WMO id).  ~11 MB.
- ``ghcnd-inventory.txt`` — fixed-width (station, element, first year,
  last year) table.  ~36 MB.  This is what tells snow stations apart.
- ``by_station/{ID}.csv.gz`` — one gzipped CSV per station holding every
  element for the full period of record.  Rows are
  ``ID,YYYYMMDD,ELEMENT,VALUE,MFLAG,QFLAG,SFLAG,OBS_TIME``, no header.

Both inventory files are downloaded once per client instance and cached in
memory.  A station's full record is downloaded for every ``get_data()``
call regardless of the date range — GHCN-D serves no windowed per-station
files — so callers looping over many date windows should reuse the
records rather than re-request.

Station ids
-----------
GHCN-D ids are 11 characters: a 2-letter FIPS country code, a 1-character
network code (see :data:`GHCN_NETWORKS`) and an 8-character station
number, e.g. ``USS0021D08S`` (a SNOTEL site), ``RSM00027612`` (Moscow,
WMO 27612).

Design principles
-----------------
- Metric-first: SNWD (mm) and SNOW (mm) are returned in cm; WESD
  (tenths of mm) in cm; PRCP and WESF (tenths of mm) in mm; temperatures
  (tenths of °C) in °C.
- Values that failed a GHCN-D quality check (non-blank ``QFLAG``) are
  returned as ``None`` unless the client is built with
  ``keep_qc_failed=True``.  ``include_flags=True`` exposes the raw
  ``MFLAG,QFLAG,SFLAG`` triple either way.
- Physically implausible snow values are nulled with per-type clamps.
- Only ``interval="daily"`` exists; anything else raises.
- Missing values (``-9999``) are ``None``; HTTP retries use the shared
  ``_common.request_with_retries`` loop (except the streamed ``by_year``
  read, which ``get_year_data`` performs directly).
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

import requests

from .._common import (
    coerce_list as _coerce_list,
)
from .._common import (
    date_str as _date_str,
)
from .._common import (
    filter_by_bbox as _filter_by_bbox,
)
from .._common import (
    request_with_retries,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

_STATIONS_FILE = "ghcnd-stations.txt"
_INVENTORY_FILE = "ghcnd-inventory.txt"
_BY_STATION = "by_station/{station_id}.csv.gz"
_BY_YEAR = "by_year/{year}.csv.gz"

# GHCN-D marks a missing elevation as -999.9 in ghcnd-stations.txt.
_MISSING_ELEVATION = -999.9

# A station counts as Active when its latest snow-element year is at least
# this many years before the current one.  The inventory is year-resolution
# only, so "active" here means "reported this year or last".
_ACTIVE_LAG_YEARS = 1

# Plausibility bounds (output units).  World-record snow depth is ~11.8 m
# and the largest SNOTEL SWE is ~4 m; daily snowfall above 5 m is a glitch.
_MAX_PLAUSIBLE = {
    "swe": 1500.0,  # cm
    "snwd": 1500.0,  # cm
    "snowfall": 500.0,  # cm
}

_GHCND_SOURCE = "NOAA NCEI GHCN-Daily — " + BASE_URL + "/by_station/"

#: Elements that make a station a *snow* station for get_all_stations().
SNOW_ELEMENTS: tuple[str, ...] = ("WESD", "SNWD")

# ── Public variable / flag tables ────────────────────────────────────────────

#: GHCN-D elements this client knows, keyed by the native element code.
#: ``scale`` converts the stored integer to ``output_units``.
VARIABLES: dict[str, dict] = {
    "WESD": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "tenths of mm",
        "output_units": "cm",
        "scale": 0.01,
        "source": _GHCND_SOURCE,
        "description": "Water equivalent of snow on the ground.",
        "notes": (
            "Stored as tenths of mm; returned in cm (× 0.01). Most WESD "
            "records outside the U.S. are sparse or periodic."
        ),
    },
    "SNWD": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Depth of snow on the ground.",
        "notes": "Stored as mm; returned in cm (× 0.1).",
    },
    "SNOW": {
        "name": "Snowfall",
        "type": "snowfall",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "New snow over the observation day, as a depth of snow.",
        "notes": "Stored as mm; returned in cm (× 0.1).",
    },
    "WESF": {
        "name": "Snowfall Water Equivalent",
        "type": "other",
        "units": "tenths of mm",
        "output_units": "mm",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Water equivalent of the day's snowfall.",
        "notes": (
            "Stored as tenths of mm; returned in mm (× 0.1). Typed 'other' "
            "rather than 'precip' so it is never mistaken for total PRCP."
        ),
    },
    "PRCP": {
        "name": "Precipitation",
        "type": "precip",
        "units": "tenths of mm",
        "output_units": "mm",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Total daily precipitation (rain and melted snow).",
        "notes": "Stored as tenths of mm; returned in mm (× 0.1).",
    },
    "TAVG": {
        "name": "Average Air Temperature",
        "type": "temp",
        "units": "tenths of degC",
        "output_units": "°C",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Daily mean air temperature.",
        "notes": (
            "Stored as tenths of °C; returned in °C. Computed differently "
            "by different sources (hourly mean vs. (max+min)/2)."
        ),
    },
    "TMAX": {
        "name": "Maximum Air Temperature",
        "type": "temp_max",
        "units": "tenths of degC",
        "output_units": "°C",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Daily maximum air temperature.",
        "notes": "Stored as tenths of °C; returned in °C.",
    },
    "TMIN": {
        "name": "Minimum Air Temperature",
        "type": "temp_min",
        "units": "tenths of degC",
        "output_units": "°C",
        "scale": 0.1,
        "source": _GHCND_SOURCE,
        "description": "Daily minimum air temperature.",
        "notes": "Stored as tenths of °C; returned in °C.",
    },
}

#: Standardized type → GHCN-D element(s), priority order.
_TYPE_TO_ELEMENTS: dict[str, list[str]] = {}
for _elem, _info in VARIABLES.items():
    _TYPE_TO_ELEMENTS.setdefault(_info["type"], []).append(_elem)

#: Default variables when ``variables=None`` (DESIGN.md §3.4: snow only).
_DEFAULT_ELEMENTS: list[str] = ["WESD", "SNWD"]

#: GHCN-D quality flags (``QFLAG``).  Blank means the value passed QC.
#: The emitted ``flag`` string is ``"MFLAG,QFLAG,SFLAG"`` — the same layout
#: NCEI's Access Data Service uses for its ``*_ATTRIBUTES`` fields.
DATA_FLAGS: dict[str, str] = {
    "D": "Failed duplicate check",
    "G": "Failed gap check",
    "I": "Failed internal consistency check",
    "K": "Failed streak/frequent-value check",
    "L": "Failed check on length of multiday period",
    "M": "Failed megaconsistency check",
    "N": "Failed naught check",
    "O": "Failed climatological outlier check",
    "R": "Failed lagged range check",
    "S": "Failed spatial consistency check",
    "T": "Failed temporal consistency check",
    "W": "Temperature too warm for snow",
    "X": "Failed bounds check",
    "Z": "Flagged as a result of an official Datzilla investigation",
}

#: The 3rd character of a GHCN-D id: which numbering system the id uses.
#: It is a reasonable proxy for the observing network, and the way to drop
#: the SNOTEL mirror (``"S"``) when AWDB is also in use.
GHCN_NETWORKS: dict[str, str] = {
    "0": "Unspecified",
    "1": "CoCoRaHS",
    "C": "U.S. Cooperative Observer Network (COOP)",
    "E": "ECA&D (European Climate Assessment & Dataset)",
    "M": "WMO station id",
    "N": "National Meteorological or Hydrological Center",
    "P": "Pre-COOP (NCEI internal id)",
    "R": "U.S. RAWS",
    "S": "USDA NRCS SNOTEL",
    "W": "WBAN",
}


# ── Fixed-width parsers ──────────────────────────────────────────────────────


def _parse_stations_txt(text: str) -> dict[str, dict]:
    """Parse ``ghcnd-stations.txt`` into ``{station_id: station dict}``.

    Column layout (1-based, from the GHCN-D readme): ID 1–11, LATITUDE
    13–20, LONGITUDE 22–30, ELEVATION 32–37, STATE 39–40, NAME 42–71,
    GSN FLAG 73–75, HCN/CRN FLAG 77–79, WMO ID 81–85.
    """
    stations: dict[str, dict] = {}
    for line in text.splitlines():
        if len(line) < 30:
            continue
        sid = line[0:11].strip()
        try:
            lat = float(line[12:20])
            lon = float(line[21:30])
        except ValueError:
            continue
        try:
            elev: float | None = float(line[31:37])
        except ValueError:
            elev = None
        if elev is not None and elev <= _MISSING_ELEVATION:
            elev = None
        stations[sid] = {
            "station_id": sid,
            "name": line[41:71].strip(),
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
            "state": line[38:40].strip() or None,
            "country_code": sid[:2],
            "ghcn_network": sid[2:3],
            "ghcn_network_name": GHCN_NETWORKS.get(sid[2:3], "Unknown"),
            "gsn": line[72:75].strip() == "GSN",
            "hcn_crn": line[76:79].strip() or None,
            "wmo_id": line[80:85].strip() or None,
        }
    return stations


def _parse_inventory_txt(
    text: str, elements: set[str] | None = None
) -> dict[str, dict[str, tuple[int, int]]]:
    """Parse ``ghcnd-inventory.txt`` into ``{station_id: {element: (first, last)}}``.

    Column layout: ID 1–11, LATITUDE 13–20, LONGITUDE 22–30, ELEMENT 32–35,
    FIRSTYEAR 37–40, LASTYEAR 42–45.  When ``elements`` is given, other
    elements are skipped (the file has ~800 000 rows).
    """
    inv: dict[str, dict[str, tuple[int, int]]] = {}
    for line in text.splitlines():
        if len(line) < 45:
            continue
        elem = line[31:35]
        if elements is not None and elem not in elements:
            continue
        try:
            first, last = int(line[36:40]), int(line[41:45])
        except ValueError:
            continue
        inv.setdefault(line[0:11].strip(), {})[elem] = (first, last)
    return inv


def _iter_rows(
    lines: Any,
    elements: set[str],
    begin: str | None,
    end: str | None,
    station_ids: set[str] | None = None,
) -> Iterator[tuple[str, str, str, int, str, str, str]]:
    """Yield ``(station_id, YYYY-MM-DD, element, raw_value, mflag, qflag, sflag)``.

    ``lines`` is any iterable of CSV text lines in the GHCN-D
    ``ID,YYYYMMDD,ELEMENT,VALUE,MFLAG,QFLAG,SFLAG,OBS_TIME`` layout shared by
    the ``by_station/`` and ``by_year/`` files.  Rows outside ``[begin, end]``
    (``YYYY-MM-DD``, either may be ``None``), for other elements, or for
    stations outside ``station_ids`` are skipped.
    """
    begin_key = begin.replace("-", "") if begin else None
    end_key = end.replace("-", "") if end else None
    for rec in csv.reader(lines):
        if len(rec) < 4 or rec[2] not in elements:
            continue
        if station_ids is not None and rec[0] not in station_ids:
            continue
        day = rec[1]
        if begin_key and day < begin_key:
            continue
        if end_key and day > end_key:
            continue
        try:
            value = int(rec[3])
        except ValueError:
            continue
        mflag = rec[4] if len(rec) > 4 else ""
        qflag = rec[5] if len(rec) > 5 else ""
        sflag = rec[6] if len(rec) > 6 else ""
        yield (
            rec[0],
            f"{day[:4]}-{day[4:6]}-{day[6:8]}",
            rec[2],
            value,
            mflag,
            qflag,
            sflag,
        )


def _parse_by_station_csv(
    raw: bytes,
    elements: set[str],
    begin: str | None,
    end: str | None,
) -> list[tuple[str, str, int, str, str, str]]:
    """Parse a ``by_station/{ID}.csv.gz`` payload.

    Returns ``(YYYY-MM-DD, element, raw_value, mflag, qflag, sflag)`` rows
    for the wanted elements inside ``[begin, end]`` (``YYYY-MM-DD`` strings,
    either may be ``None``).
    """
    try:
        text = gzip.decompress(raw).decode("ascii", errors="replace")
    except (OSError, EOFError) as exc:
        raise GHCNDError(f"Could not decompress GHCN-D station file: {exc}") from exc
    return [row[1:] for row in _iter_rows(io.StringIO(text), elements, begin, end)]


def _convert(element: str, raw: int) -> float | None:
    """Scale a stored GHCN-D integer to output units, applying snow clamps."""
    if raw == -9999:
        return None
    info = VARIABLES[element]
    value = round(raw * info["scale"], 3)
    bound = _MAX_PLAUSIBLE.get(info["type"])
    if bound is not None and not (0.0 <= value <= bound):
        return None
    return value


# ── Client ───────────────────────────────────────────────────────────────────


class GHCNDClient:
    """
    Client for NOAA NCEI GHCN-Daily station files.

    No credentials are needed.

    Parameters
    ----------
    base_url : str
        Root of the GHCN-D directory.  Override for a mirror.
    timeout : int
        HTTP request timeout in seconds.  The inventory file is ~36 MB, so
        the default is generous.
    max_retries : int
        Number of retry attempts on transient errors (5xx, network).
    backoff : int
        Base backoff delay in seconds.  Actual delay = backoff × attempt.
    session : requests.Session or None
        Optional pre-configured session.
    keep_qc_failed : bool
        If False (default), values carrying a non-blank GHCN-D ``QFLAG``
        are returned as ``None``.  If True they are returned as stored.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
        keep_qc_failed: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.keep_qc_failed = keep_qc_failed
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._stations_cache: dict[str, dict] | None = None
        self._inventory_cache: dict[str, dict[str, tuple[int, int]]] | None = None

    # ── Inventory files ──────────────────────────────────────────────────────

    def _station_table(self) -> dict[str, dict]:
        if self._stations_cache is None:
            text = self._get(_STATIONS_FILE).text
            self._stations_cache = _parse_stations_txt(text)
            logger.info("GHCN-D: parsed %d stations", len(self._stations_cache))
        return self._stations_cache

    def _inventory(self) -> dict[str, dict[str, tuple[int, int]]]:
        if self._inventory_cache is None:
            text = self._get(_INVENTORY_FILE).text
            self._inventory_cache = _parse_inventory_txt(text, set(VARIABLES))
            logger.info(
                "GHCN-D: parsed inventory for %d stations", len(self._inventory_cache)
            )
        return self._inventory_cache

    def _build_station(self, sid: str) -> dict | None:
        base = self._station_table().get(sid)
        if base is None:
            return None
        elems = self._inventory().get(sid, {})
        snow_last = [elems[e][1] for e in SNOW_ELEMENTS if e in elems]
        active = bool(snow_last) and (
            max(snow_last) >= date.today().year - _ACTIVE_LAG_YEARS
        )
        return {
            **base,
            "status": "Active" if active else "Inactive",
            "has_swe": "WESD" in elems,
            "has_snwd": "SNWD" in elems,
            "elements": {
                e: {"first_year": f, "last_year": la} for e, (f, la) in elems.items()
            },
        }

    # ── Public API — stations ────────────────────────────────────────────────

    def get_stations(
        self,
        elements: list[str] | tuple[str, ...] | str = SNOW_ELEMENTS,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        exclude_networks: list[str] | tuple[str, ...] | str | None = None,
    ) -> list[dict]:
        """
        Stations reporting any of ``elements``.

        Parameters
        ----------
        elements : list[str] or str
            GHCN-D element codes; a station is kept if it reports any of
            them.  Defaults to ``("WESD", "SNWD")``.
        active_only : bool
            Keep only stations whose latest snow-element year is this year
            or last.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        exclude_networks : list[str] or str, optional
            GHCN-D network codes to drop (see :data:`GHCN_NETWORKS`), e.g.
            ``"S"`` to drop the SNOTEL mirror when AWDB is used alongside.

        Returns
        -------
        list[dict]
            Station dicts with ``station_id``, ``name``, ``latitude``,
            ``longitude``, ``elevation_m``, ``status``, ``state``,
            ``country_code``, ``ghcn_network``, ``ghcn_network_name``,
            ``gsn``, ``hcn_crn``, ``wmo_id``, ``has_swe``, ``has_snwd``,
            ``elements`` (``{code: {first_year, last_year}}``) and
            ``station_url``.

        Raises
        ------
        GHCNDError
            On an unknown element code or a failed inventory download.
        """
        wanted = set(_coerce_list(elements))
        unknown = wanted - set(VARIABLES)
        if unknown:
            raise GHCNDError(
                f"Unknown GHCN-D element(s) {sorted(unknown)} — expected some "
                f"of {sorted(VARIABLES)}."
            )
        excluded = set(_coerce_list(exclude_networks)) if exclude_networks else set()
        inventory = self._inventory()
        out: list[dict] = []
        for sid, elems in inventory.items():
            if not wanted & set(elems):
                continue
            if sid[2:3] in excluded:
                continue
            sta = self._build_station(sid)
            if sta is None:
                continue
            if active_only and sta["status"] != "Active":
                continue
            sta["station_url"] = (
                "https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/"
                f"GHCND:{sid}/detail"
            )
            out.append(sta)
        return _filter_by_bbox(out, bbox)

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        All GHCN-D stations reporting snow depth or SWE (any network).

        Parameters
        ----------
        active_only : bool
            Keep only stations whose latest snow-element year is this year
            or last.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            Same schema as :meth:`get_stations` (which can also drop
            networks).

        Raises
        ------
        GHCNDError
            If the station or inventory file cannot be downloaded.
        """
        return self.get_stations(active_only=active_only, bbox=bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Metadata and element inventory for one station.

        Parameters
        ----------
        station_id : str
            GHCN-D station id, e.g. ``"USS0021D08S"``.

        Returns
        -------
        dict
            The :meth:`get_stations` dict plus ``data_variables``: one
            entry per known element with ``name``, ``type``, ``interval``,
            ``units`` (emitted), ``description``, ``notes``, ``begin_date``,
            ``end_date`` and ``n_obs`` (``None`` — the inventory does not
            say).  Dates are year-resolution (``"YYYY-01-01"`` /
            ``"YYYY-12-31"``).

        Raises
        ------
        GHCNDError
            If the station id is not in ``ghcnd-stations.txt``.
        """
        sta = self._build_station(station_id)
        if sta is None:
            raise GHCNDError(f"Station {station_id!r} not found in GHCN-D")
        sta["station_url"] = (
            "https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/"
            f"GHCND:{station_id}/detail"
        )
        sta["data_variables"] = [
            {
                "name": elem,
                "type": VARIABLES[elem]["type"],
                "interval": "daily",
                "units": VARIABLES[elem]["output_units"],
                "description": VARIABLES[elem]["description"],
                "notes": VARIABLES[elem]["notes"],
                "begin_date": f"{yrs['first_year']}-01-01",
                "end_date": f"{yrs['last_year']}-12-31",
                "n_obs": None,
            }
            for elem, yrs in sta["elements"].items()
        ]
        return sta

    # ── Public API — data ────────────────────────────────────────────────────

    def get_station_file(self, station_id: str) -> bytes:
        """
        Raw ``by_station/{ID}.csv.gz`` bytes for one station.

        Parameters
        ----------
        station_id : str
            GHCN-D station id.

        Returns
        -------
        bytes
            The gzipped CSV, full period of record.

        Raises
        ------
        GHCNDError
            If the file cannot be downloaded (HTTP 404 for an unknown id).
        """
        return self._get(_BY_STATION.format(station_id=station_id)).content

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "daily",
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Standardized data fetch — returns a flat list of observation records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            GHCN-D station id(s), e.g. ``"USS0021D08S"``.  Required unless
            ``bbox`` is provided.
        variables : list[str] or str or None
            Element codes (``"SNWD"``) or standardized types (``"snwd"``).
            ``None`` means snow only: ``WESD`` and ``SNWD``.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``; fetches every snow
            station in the box.  GHCN-D is dense — a large box means many
            full-record downloads.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds.  The whole station file is
            downloaded either way; the bounds only filter rows.
        interval : str
            Only ``"daily"`` is supported.
        include_flags : bool
            Add ``flag`` = ``"MFLAG,QFLAG,SFLAG"`` to each record.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            records (plus ``flag``).  Stations whose file cannot be fetched
            are logged and skipped.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        GHCNDError
            On an unknown variable or interval.

        Example
        -------
        >>> client = GHCNDClient()
        >>> records = client.get_data(
        ...     station_ids="USS0021D08S",
        ...     variables=["snwd"],
        ...     begin_date="2024-01-01",
        ...     end_date="2024-01-15",
        ... )
        >>> records[0]["units"]
        'cm'
        """
        if interval.lower() != "daily":
            raise GHCNDError(
                f"Unsupported interval {interval!r} for GHCN-D — only 'daily' exists."
            )
        elements = _resolve_variables(variables)

        if station_ids is None and bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        elif station_ids is not None:
            ids = _coerce_list(station_ids)
        else:
            raise ValueError("Provide station_ids or bbox.")

        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        wanted = set(elements)

        records: list[dict] = []
        for sid in ids:
            try:
                raw = self.get_station_file(sid)
                rows = _parse_by_station_csv(raw, wanted, begin, end)
            except GHCNDError as exc:
                logger.warning("GHCN-D: skipping station %s: %s", sid, exc)
                continue
            records.extend(
                self._record(sid, *row, include_flags=include_flags) for row in rows
            )
        return records

    def get_year_data(
        self,
        year: int,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        include_flags: bool = False,
    ) -> Iterator[dict]:
        """
        Stream one ``by_year/{year}.csv.gz`` file as observation records.

        The bulk path for refreshing many stations: one download (~115–170 MB
        compressed for a recent year) instead of one full-record file per
        station.  NCEI rebuilds the current year's file daily.  Late-arriving
        sources (COOP paper forms, ECA&D batches) can still add values to the
        previous year's file for a month or two.

        Parameters
        ----------
        year : int
            Calendar year of the file.
        station_ids : list[str] or str or None
            Keep only these stations.  ``None`` keeps every station.
        variables : list[str] or str or None
            As in :meth:`get_data`; ``None`` means ``WESD`` and ``SNWD``.
        begin_date, end_date : str or date, optional
            Inclusive row filter within the year.
        include_flags : bool
            Add ``flag`` = ``"MFLAG,QFLAG,SFLAG"`` to each record.

        Yields
        ------
        dict
            The same records :meth:`get_data` returns, in file order.

        Raises
        ------
        GHCNDError
            On an unknown variable or a failed download.
        """
        wanted = set(_resolve_variables(variables))
        keep = set(_coerce_list(station_ids)) if station_ids is not None else None
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        url = f"{self.base_url}/{_BY_YEAR.format(year=int(year))}"
        try:
            with self._session.get(url, stream=True, timeout=self.timeout) as resp:
                if not resp.ok:
                    raise GHCNDError(f"HTTP {resp.status_code} from {url}")
                resp.raw.decode_content = True
                with gzip.open(
                    resp.raw, "rt", encoding="ascii", errors="replace"
                ) as fh:
                    for row in _iter_rows(fh, wanted, begin, end, keep):
                        yield self._record(*row, include_flags=include_flags)
        except (requests.exceptions.RequestException, OSError, EOFError) as exc:
            raise GHCNDError(f"Reading {url} failed: {exc}") from exc

    def _record(
        self,
        sid: str,
        day: str,
        elem: str,
        stored: int,
        mflag: str,
        qflag: str,
        sflag: str,
        *,
        include_flags: bool,
    ) -> dict[str, Any]:
        info = VARIABLES[elem]
        value = _convert(elem, stored)
        if qflag.strip() and not self.keep_qc_failed:
            value = None
        rec: dict[str, Any] = {
            "station_id": sid,
            "date": day,
            "variable": elem,
            "type": info["type"],
            "value": value,
            "units": info["output_units"],
            "interval": "daily",
        }
        if include_flags:
            rec["flag"] = f"{mflag},{qflag},{sflag}"
        return rec

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get(self, path: str) -> requests.Response:
        return request_with_retries(
            self._session,
            f"{self.base_url}/{path}",
            error_cls=GHCNDError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )


# ── Exception ────────────────────────────────────────────────────────────────


class GHCNDError(Exception):
    """Raised when a GHCN-D file cannot be fetched or parsed."""


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Translate element codes and/or standardized types to element codes."""
    if variables is None:
        return list(_DEFAULT_ELEMENTS)
    raw = [variables] if isinstance(variables, str) else list(variables)
    if not raw:
        return list(_DEFAULT_ELEMENTS)
    out: list[str] = []
    for v in raw:
        if v in VARIABLES:
            candidates = [v]
        elif v in _TYPE_TO_ELEMENTS:
            candidates = _TYPE_TO_ELEMENTS[v]
        else:
            # No silent fallback (DESIGN.md §3.6)
            raise GHCNDError(
                f"Unknown variable {v!r} for GHCN-D — expected one of "
                f"{sorted(VARIABLES)} or a standardized type "
                f"({sorted(_TYPE_TO_ELEMENTS)})."
            )
        out.extend(c for c in candidates if c not in out)
    return out
