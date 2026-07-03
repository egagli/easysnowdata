# -*- coding: utf-8 -*-
"""
clients/nve/nve_client.py
=========================
Python client for NVE HydAPI — the Norwegian Water Resources and Energy
Directorate (Norges vassdrags- og energidirektorat) hydrological data service.

Primary use case: Norwegian snow pillow stations with automated daily SWE
and snow depth measurements.

API documentation : https://hydapi.nve.no/
Base URL          : https://hydapi.nve.no/api/v1
Authentication    : API key required — pass via X-API-Key header.
                   Register for a free key at https://hydapi.nve.no/
                   Set the NVE_API_KEY environment variable or pass
                   ``api_key`` to NVEClient().

Key parameters (verified against GET /Parameters, 2026-07-03)
--------------
- Parameter 2002 : Snow depth / "Snødybde" (cm)
- Parameter 2003 : Snow Water Equivalent / "Snøens vannekvivalent" (m)
- (Parameter 2001 is soil water / "Markfuktighet" — NOT snow depth)
- ResolutionTime 1440 : Daily (1440 min)
- ResolutionTime 60   : Hourly (60 min)

Station URL convention
----------------------
Each station has a public page on the NVE Sildre portal:
  ``https://sildre.nve.no/station/{station_id}``

Design principles
-----------------
- Returns plain Python objects (dicts / lists).
- Metric-first: SWE returned in cm (converted from native metres × 100).
  Snow depth returned in cm as-is.
- Missing / sentinel values (None, -9999, NaN) are normalised to ``None``,
  as are physically implausible snow values (negative or > 15 m) that
  slip through NVE's own quality control.
- ``include_flags=True`` on ``get_data()`` adds a ``flag`` key to each
  value record.
- HTTP retry logic is applied to all requests.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://hydapi.nve.no/api/v1"

_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

# NVE HydAPI rate limit: 5 requests per second per API key.
# A small inter-request delay keeps us safely under the limit.
_REQUEST_DELAY = 0.25  # seconds between Observations requests

# The API caps the number of data points one Observations request may
# return ("the request will terminate with an error if the query reaches
# this limit" — the exact cap is not documented).  Long date ranges are
# split into windows sized to stay under this budget.
_MAX_POINTS_PER_REQUEST = 20_000

# When get_data() targets at most this many stations, series availability
# is looked up per station instead of via nationwide per-parameter lists.
_SERIES_PER_STATION_MAX = 10

# Sentinel / missing value used by some NVE responses
_MISSING_VALUES = {-9999, -9999.0}

# Plausibility bound for converted snow values (cm).  The NVE archive
# contains glitches that pass its own quality control (e.g. station
# 123.93.0 reports ~145 m SWE flagged "secondary controlled" in Jan
# 2018); world-record snow depth is ~11.8 m.  Values outside
# [0, _MAX_PLAUSIBLE_CM] are normalised to None, matching the DataBC
# client's negative-value handling.
_MAX_PLAUSIBLE_CM = 1500.0

# NVE parameter IDs for snow variables
_PARAM_SWE   = 2003  # Snow Water Equivalent, "Snøens vannekvivalent" (m)
_PARAM_SNWD  = 2002  # Snow depth, "Snødybde" (cm)

# Temporal resolution in minutes
_RESOLUTION_DAILY  = 1440
_RESOLUTION_HOURLY = 60

# Sildre station URL template
_SILDRE_URL = "https://sildre.nve.no/station/{station_id}"

_NVE_DATA_SOURCE = "NVE HydAPI v1 — https://hydapi.nve.no/api/v1/Observations"

# ── Public variable / flag tables ────────────────────────────────────────────

#: Known NVE hydrological parameters relevant to snow monitoring.
VARIABLES: dict[str, dict] = {
    "swe_m": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "cm",
        "source": _NVE_DATA_SOURCE + " (ParameterId=2003)",
        "description": (
            "Snow water equivalent from automated snow pillow. "
            "Native API unit is metres; returned here in cm (× 100)."
        ),
        "notes": "Parameter ID 2003 (Snøens vannekvivalent). Native units: m.",
    },
    "snwd_cm": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "source": _NVE_DATA_SOURCE + " (ParameterId=2002)",
        "description": "Snow depth from automated sensor. Native API unit is cm.",
        "notes": "Parameter ID 2002 (Snødybde). Native units: cm.",
    },
}

#: Mapping from standardized type → NVE variable key(s) (priority order).
_TYPE_TO_NVE_VARS: dict[str, list[str]] = {
    "swe":  ["swe_m"],
    "snwd": ["snwd_cm"],
}

#: Mapping from NVE parameter ID → variable key.
_PARAM_TO_VAR: dict[int, str] = {
    _PARAM_SWE:  "swe_m",
    _PARAM_SNWD: "snwd_cm",
}

#: Mapping from variable key → NVE parameter ID.
_VAR_TO_PARAM: dict[str, int] = {v: k for k, v in _PARAM_TO_VAR.items()}

#: Standardized interval → NVE ResolutionTime (minutes).
_INTERVAL_TO_RESOLUTION: dict[str, int] = {
    "daily":  _RESOLUTION_DAILY,
    "hourly": _RESOLUTION_HOURLY,
}

#: NVE ResolutionTime → standardized interval name.
_RESOLUTION_TO_INTERVAL: dict[int, str] = {
    _RESOLUTION_DAILY:  "daily",
    _RESOLUTION_HOURLY: "hourly",
}

#: NVE data quality flags returned in the ``quality`` field of observations.
DATA_FLAGS: dict[str, str] = {
    "0": "Unknown — quality status not determined",
    "1": "Uncontrolled",
    "2": "Primary controlled",
    "3": "Secondary controlled (quality assured)",
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _coerce_list(value: list | str) -> list[str]:
    """Ensure value is a list of strings."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _date_str(d: str | date | datetime) -> str:
    """Normalize a date-like object to ``YYYY-MM-DD`` string."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def _reference_windows(
    begin_date: str | date | None,
    end_date: str | date | None,
    data_from: str,
    resolution: int,
) -> list[tuple[str | None, str | None]]:
    """
    Split a requested observation period into ReferenceTime windows.

    Clips the requested begin to the series' actual data start
    (``data_from``, ISO date string from /Series, may be empty) and splits
    the period into windows small enough to stay under
    ``_MAX_POINTS_PER_REQUEST`` data points at the given resolution.
    The end is never clipped to the series' ``dataToTime``: that metadata
    can lag the newest observations.

    Returns a list of ``(begin, end)`` ISO-date tuples.  ``(None, None)``
    means "omit ReferenceTime" (the API then returns the most recent
    observation), preserving the no-dates behaviour of ``get_data``.
    An empty list means the requested period is empty.
    """
    if begin_date is None and end_date is None:
        return [(None, None)]

    begin = _date_str(begin_date) if begin_date is not None else (data_from or None)
    if begin is None:
        # Open start ("/end") — cannot chunk without a start date.
        return [(None, _date_str(end_date))]

    if data_from and begin < data_from:
        begin = data_from
    end = _date_str(end_date) if end_date is not None else date.today().isoformat()
    if begin > end:
        return []

    # e.g. daily (1440 min): 20 000-day windows; hourly (60 min): 833 days
    step_days = max(1, _MAX_POINTS_PER_REQUEST * resolution // 1440)
    windows: list[tuple[str | None, str | None]] = []
    cur = date.fromisoformat(begin)
    stop = date.fromisoformat(end)
    while cur <= stop:
        win_end = min(cur + timedelta(days=step_days - 1), stop)
        windows.append((cur.isoformat(), win_end.isoformat()))
        cur = win_end + timedelta(days=1)
    return windows


def _filter_by_bbox(
    stations: list[dict],
    bbox: tuple[float, float, float, float],
    lat_key: str = "latitude",
    lon_key: str = "longitude",
) -> list[dict]:
    """Return stations whose lat/lon fall within (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return [
        s for s in stations
        if s.get(lat_key) is not None and s.get(lon_key) is not None
        and min_lat <= float(s[lat_key]) <= max_lat
        and min_lon <= float(s[lon_key]) <= max_lon
    ]


def _normalize_value(v: Any) -> float | None:
    """Return None for missing/sentinel values; otherwise return float."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv in _MISSING_VALUES or fv != fv:  # NaN check
        return None
    return fv


def _enrich_station(raw: dict) -> dict:
    """
    Build a normalised station dict from a raw NVE /Stations entry.

    Parameters
    ----------
    raw : dict
        A single entry from the ``data`` array returned by /Stations.

    Returns
    -------
    dict
        Normalised station record with keys:
        ``station_id``, ``name``, ``latitude``, ``longitude``,
        ``elevation_m``, ``drainage_basin_key``, ``status``,
        ``station_url``, ``parameters``, ``daily_parameters``.
    """
    sid = str(raw.get("stationId") or "")
    # Normalise lat/lon (NVE may return as float or null)
    lat = _normalize_value(raw.get("latitude"))
    lon = _normalize_value(raw.get("longitude"))
    # Elevation: NVE reports in metres
    elev = _normalize_value(raw.get("elevation"))

    # Active if no decommission date
    active = raw.get("active", True)
    status = "Active" if active else "Inactive"

    # Build parameter availability from seriesList.  Each entry is a
    # SerieShort whose parameter ID lives under the key ``parameter``
    # (NOT ``parameterId``), with a ``resolutionList`` of supported time
    # resolutions (resTime in minutes: 0/60/1440).
    series_list = raw.get("seriesList") or []
    param_ids: set[int] = set()
    daily_param_ids: set[int] = set()
    for s in series_list:
        pid = s.get("parameter")
        if pid is None:
            continue
        pid = int(pid)
        param_ids.add(pid)
        resolutions = s.get("resolutionList") or []
        if any(
            _normalize_value(r.get("resTime")) == _RESOLUTION_DAILY
            for r in resolutions
        ):
            daily_param_ids.add(pid)

    return {
        "station_id": sid,
        "name": raw.get("stationName") or raw.get("name") or "",
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elev,
        "drainage_basin_key": raw.get("drainageBasinKey") or "",
        "status": status,
        "station_url": _SILDRE_URL.format(station_id=sid) if sid else "",
        "parameters": sorted(param_ids),
        "daily_parameters": sorted(daily_param_ids),
    }


# ── Client ───────────────────────────────────────────────────────────────────

class NVEClient:
    """
    Client for the NVE HydAPI (Norwegian hydrological data service).

    An API key is required for all endpoints — set the ``NVE_API_KEY``
    environment variable or pass ``api_key``.  Register for a free key
    at https://hydapi.nve.no/.

    Parameters
    ----------
    base_url : str
        Base URL of the NVE HydAPI.  Override for staging environments.
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Number of retry attempts on transient server errors (5xx).
    backoff : int
        Base backoff delay in seconds.  Actual delay = backoff × attempt_number.
    session : requests.Session or None
        Optional pre-configured session (useful for proxies, etc.).
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        resolved_key = api_key or os.environ.get("NVE_API_KEY", "")
        headers: dict[str, str] = {
            "accept": "application/json",
            "User-Agent": "global-snow-networks/1.0",
        }
        if resolved_key:
            headers["X-API-Key"] = resolved_key
        else:
            logger.warning(
                "No NVE API key configured (NVE_API_KEY unset and no "
                "api_key given) — all HydAPI requests will fail with "
                "HTTP 401. Register for a free key at https://hydapi.nve.no/"
            )
        self._session.headers.update(headers)

    # ── Public API — station lists ────────────────────────────────────────────

    def get_stations(
        self,
        parameter_ids: list[int] | int | None = None,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        List NVE hydrological stations, optionally filtered by parameter.

        Parameters
        ----------
        parameter_ids : list[int] or int, optional
            NVE parameter ID(s) to filter stations by.
            - 2001 : Snow depth
            - 2002 : Snow Water Equivalent (SWE)
            Defaults to no parameter filter (all stations).
        active_only : bool
            If True, return only currently active stations.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` bounding box filter.

        Returns
        -------
        list[dict]
            One dict per station with keys: ``station_id``, ``name``,
            ``latitude``, ``longitude``, ``elevation_m``,
            ``drainage_basin_key``, ``status``, ``station_url``,
            ``parameters``.

        Example
        -------
        >>> client = NVEClient()
        >>> swe_stations = client.get_stations(parameter_ids=2002)
        >>> len(swe_stations) > 20
        True
        """
        params: dict[str, Any] = {}
        if parameter_ids is not None:
            pid_list = (
                [parameter_ids]
                if isinstance(parameter_ids, int)
                else list(parameter_ids)
            )
            # The NVE API accepts a single ParameterId; make multiple calls if needed
            all_stations: dict[str, dict] = {}
            for pid in pid_list:
                raw = self._get("Stations", {"ParameterId": pid})
                for item in raw.get("data") or []:
                    sta = _enrich_station(item)
                    if sta["station_id"]:
                        # Safety net: guarantee the queried parameter is
                        # present even if seriesList is missing/unparseable.
                        if pid not in sta["parameters"]:
                            sta["parameters"] = sorted(
                                set(sta["parameters"]) | {pid}
                            )
                        all_stations[sta["station_id"]] = sta
            stations = list(all_stations.values())
        else:
            raw = self._get("Stations", params)
            stations = [
                _enrich_station(item)
                for item in (raw.get("data") or [])
                if (item.get("stationId") or "")
            ]

        if active_only:
            stations = [s for s in stations if s.get("status") == "Active"]
        if bbox is not None:
            stations = _filter_by_bbox(stations, bbox)
        return stations

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Get all NVE stations with snow parameters (SWE and/or snow depth).

        This is the recommended entry point for discovering snow monitoring
        stations.  It fetches stations with parameter 2002 (SWE) and
        parameter 2001 (snow depth) and deduplicates.

        Parameters
        ----------
        active_only : bool
            If True, return only Active stations.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` bounding box filter.

        Returns
        -------
        list[dict]
            Same schema as :meth:`get_stations`.
        """
        return self.get_stations(
            parameter_ids=[_PARAM_SWE, _PARAM_SNWD],
            active_only=active_only,
            bbox=bbox,
        )

    def get_metadata(self, station_id: str) -> dict:
        """
        Retrieve full metadata for a single station.

        Parameters
        ----------
        station_id : str
            NVE station ID, e.g. ``"2.11.0"``.

        Returns
        -------
        dict
            Station metadata including keys: ``station_id``, ``name``,
            ``latitude``, ``longitude``, ``elevation_m``,
            ``drainage_basin_key``, ``status``, ``station_url``,
            ``parameters``, ``series_list``.

        Raises
        ------
        NVEError
            If the station is not found or the request fails.
        """
        raw = self._get("Stations", {"StationId": station_id})
        data = raw.get("data") or []
        if not data:
            raise NVEError(f"Station {station_id!r} not found")
        sta = _enrich_station(data[0])
        sta["series_list"] = data[0].get("seriesList") or []
        return sta

    def get_series(
        self,
        parameter: int | None = None,
        station_id: str | None = None,
    ) -> list[dict]:
        """
        List time series available in HydAPI (GET /Series).

        A *series* is one (station, parameter, version) combination with a
        list of supported time resolutions and the data range covered at
        each resolution.

        Parameters
        ----------
        parameter : int, optional
            NVE parameter ID (e.g. 2002 for SWE, 2001 for snow depth).
            The endpoint accepts a single parameter per request.
        station_id : str, optional
            NVE station ID, e.g. ``"2.11.0"``.

        Returns
        -------
        list[dict]
            One dict per series with keys: ``station_id``, ``station_name``,
            ``parameter``, ``parameter_name``, ``version_no``, ``unit``,
            ``serie_from``, ``serie_to``, ``resolutions``.  ``resolutions``
            is the raw ``resolutionList`` — each entry has ``resTime``
            (minutes: 0/60/1440), ``dataFromTime`` and ``dataToTime``.
        """
        params: dict[str, Any] = {}
        if parameter is not None:
            params["Parameter"] = parameter
        if station_id is not None:
            params["StationId"] = station_id
        raw = self._get("Series", params)
        series: list[dict] = []
        for item in raw.get("data") or []:
            series.append({
                "station_id": str(item.get("stationId") or ""),
                "station_name": item.get("stationName") or "",
                "parameter": item.get("parameter"),
                "parameter_name": item.get("parameterName") or "",
                "version_no": item.get("versionNo"),
                "unit": item.get("unit") or "",
                "serie_from": item.get("serieFrom") or "",
                "serie_to": item.get("serieTo") or "",
                "resolutions": item.get("resolutionList") or [],
            })
        return series

    def _series_index(
        self,
        param_ids: set[int],
        station_ids: set[str],
        resolution: int,
    ) -> dict[tuple[str, int], str]:
        """
        Map ``(station_id, param_id)`` → earliest ``dataFromTime`` date
        string (``""`` when unknown) for series that exist at
        ``resolution``.  For a handful of stations the /Series lookups go
        by station; for large batches one nationwide /Series call per
        parameter is cheaper.

        Raises
        ------
        NVEError
            If a /Series request fails (after retries) — a silent partial
            index would make ``get_data`` skip every affected station and
            look like "no data".
        """
        index: dict[tuple[str, int], str] = {}

        def _ingest(series: list[dict]) -> None:
            for serie in series:
                sid = serie["station_id"]
                pid = serie["parameter"]
                if sid not in station_ids or pid not in param_ids:
                    continue
                for res in serie["resolutions"]:
                    if _normalize_value(res.get("resTime")) != resolution:
                        continue
                    frm = str(res.get("dataFromTime") or "")[:10]
                    prev = index.get((sid, pid))
                    if prev is not None:
                        # Merge versions: unknown start ("") wins, else earliest
                        frm = min(frm, prev) if (frm and prev) else ""
                    index[(sid, pid)] = frm

        try:
            if len(station_ids) <= _SERIES_PER_STATION_MAX:
                for i, sid in enumerate(sorted(station_ids)):
                    if i > 0:
                        time.sleep(_REQUEST_DELAY)
                    _ingest(self.get_series(station_id=sid))
            else:
                for i, pid in enumerate(sorted(param_ids)):
                    if i > 0:
                        time.sleep(_REQUEST_DELAY)
                    _ingest(self.get_series(parameter=pid))
        except NVEError as exc:
            raise NVEError(f"Failed to list available series: {exc}") from exc
        return index

    # ── Public API — time-series data ─────────────────────────────────────────

    def get_observations(
        self,
        station_id: str,
        parameter_id: int,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        resolution: int = _RESOLUTION_DAILY,
    ) -> list[dict]:
        """
        Fetch raw observations for a single station and parameter.

        Parameters
        ----------
        station_id : str
            NVE station ID, e.g. ``"2.11.0"``.
        parameter_id : int
            NVE parameter ID (e.g. 2002 for SWE, 2001 for snow depth).
        begin_date : str or date, optional
            Start of the observation window (``"YYYY-MM-DD"``).
        end_date : str or date, optional
            End of the observation window (inclusive).
        resolution : int
            Temporal resolution in minutes.
            1440 = daily; 60 = hourly.

        Returns
        -------
        list[dict]
            Observation records (one per timestamp), each with ``time``,
            ``value``, ``correction`` and ``quality`` fields.

        Raises
        ------
        NVEError
            On request failure.
        """
        params: dict[str, Any] = {
            "StationId": station_id,
            "Parameter": parameter_id,
            # Minutes as a string ("1440"/"60"), matching NVE's reference
            # client (github.com/NVE/HydAPI).
            "ResolutionTime": str(resolution),
        }
        # The Observations endpoint uses ReferenceTime in ISO-8601 interval
        # format ("start/end"), NOT separate StartDate/EndDate parameters.
        if begin_date is not None or end_date is not None:
            start = _date_str(begin_date) if begin_date is not None else ""
            end = _date_str(end_date) if end_date is not None else ""
            params["ReferenceTime"] = f"{start}/{end}"

        raw = self._get("Observations", params)
        # ``data`` items are per-series wrappers ({stationId, parameter,
        # unit, observationCount, observations: [...]}) — flatten to the
        # actual observation records.
        return [
            obs
            for serie in raw.get("data") or []
            for obs in serie.get("observations") or []
        ]

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
            NVE station ID(s), e.g. ``"2.11.0"``.
            Required unless ``bbox`` is provided.
        variables : list[str] or str or None
            NVE variable key(s) (e.g. ``"swe_m"``) **or** standardized
            types (e.g. ``"swe"``).  ``None`` returns all snow variables
            (SWE and snow depth).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.  Alternative to
            ``station_ids``; fetches data for all snow stations in the box.
        begin_date : str or date, optional
            Start date (``"YYYY-MM-DD"``).
        end_date : str or date, optional
            End date (inclusive).
        interval : str
            Temporal resolution: ``"daily"`` (default) or ``"hourly"``.
        include_flags : bool
            If True, add a ``"flag"`` key with the NVE quality code to each
            record.

        Returns
        -------
        list[dict]
            Flat list of observation records::

                {
                    "station_id": "2.11.0",
                    "date": "2024-01-15",
                    "variable": "swe_m",
                    "type": "swe",
                    "value": 12.5,   # cm (converted from m × 100)
                    "units": "cm",
                    "interval": "daily",
                    # "flag": "0"  (only present when include_flags=True)
                }

        Notes
        -----
        SWE values (parameter 2003) are stored by NVE in metres.  This
        method converts them to cm (× 100) so that ``"units"`` is always
        ``"cm"`` for type ``"swe"``.

        Snow depth values (parameter 2002) are stored by NVE in cm and are
        returned as-is.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        NVEError
            On network / API failure.

        Example
        -------
        >>> client = NVEClient()
        >>> records = client.get_data(
        ...     station_ids="2.11.0",
        ...     variables=["swe"],
        ...     begin_date="2024-01-01",
        ...     end_date="2024-01-15",
        ... )
        >>> records[0].keys()
        dict_keys(['station_id', 'date', 'variable', 'type', 'value', 'units', 'interval'])
        """
        # ── Resolve station IDs ────────────────────────────────────────────
        if station_ids is None and bbox is not None:
            ids: list[str] = [
                s["station_id"]
                for s in self.get_all_stations(bbox=bbox)
            ]
        elif station_ids is not None:
            ids = _coerce_list(station_ids)
        else:
            raise ValueError("Provide station_ids or bbox.")

        if not ids:
            return []

        # ── Resolve variables → (var_key, param_id, converter) tuples ─────
        var_jobs = _resolve_variables(variables)

        resolution = _INTERVAL_TO_RESOLUTION.get(interval.lower(), _RESOLUTION_DAILY)
        std_interval = _RESOLUTION_TO_INTERVAL.get(resolution, interval.lower())

        # ── Discover which series actually exist ──────────────────────────
        # Requesting a series that does not exist returns HTTP 404, and a
        # very long ReferenceTime window can exceed the API's per-request
        # data-point limit.  /Series tells us which stations have data at
        # the requested resolution and since when, so we only request
        # observations that exist, clipped to their actual data start and
        # chunked into bounded windows.
        wanted_params = {param_id for _, param_id, _ in var_jobs}
        series_index = self._series_index(wanted_params, set(ids), resolution)
        n_pairs = len(ids) * len(var_jobs)
        logger.info(
            "NVE series index: %d of %d requested station+parameter pairs "
            "have a series at resolution %d",
            len(series_index), n_pairs, resolution,
        )
        if not series_index and ids:
            # /Series listed nothing for stations that our own inventory
            # says have data — distrust the index rather than silently
            # return no records.  Blind requests are noisier (missing
            # series 404 with a verbose body) but self-diagnosing.
            logger.warning(
                "/Series matched none of the %d requested stations at "
                "resolution %d — falling back to direct observation "
                "requests for all of them",
                len(ids), resolution,
            )
            series_index = {
                (sid, pid): "" for sid in ids for pid in wanted_params
            }

        records: list[dict] = []
        request_count = 0
        for sid in ids:
            for var_key, param_id, converter in var_jobs:
                var_info = VARIABLES[var_key]
                std_type = var_info["type"]
                units = var_info["units"]

                if (sid, param_id) not in series_index:
                    logger.debug(
                        "Station %s has no %s series at resolution %d — skipping",
                        sid, var_key, resolution,
                    )
                    continue
                data_from = series_index[(sid, param_id)]

                obs_list: list[dict] = []
                for win_begin, win_end in _reference_windows(
                    begin_date, end_date, data_from, resolution
                ):
                    if request_count > 0:
                        time.sleep(_REQUEST_DELAY)
                    request_count += 1
                    try:
                        obs_list.extend(self.get_observations(
                            station_id=sid,
                            parameter_id=param_id,
                            begin_date=win_begin,
                            end_date=win_end,
                            resolution=resolution,
                        ))
                    except NVEError as exc:
                        logger.warning(
                            "Failed to fetch %s for station %s (%s/%s): %s",
                            var_key, sid, win_begin, win_end, exc,
                        )

                for obs in obs_list:
                    raw_val = _normalize_value(obs.get("value"))
                    value = converter(raw_val) if raw_val is not None else None
                    if value is not None and not (
                        0 <= value <= _MAX_PLAUSIBLE_CM
                    ):
                        value = None
                    # Extract date part from ISO timestamp
                    ts = str(obs.get("time") or obs.get("dateTime") or "")
                    date_str = ts[:10] if ts else ""
                    if not date_str:
                        continue
                    rec: dict = {
                        "station_id": sid,
                        "date": date_str,
                        "variable": var_key,
                        "type": std_type,
                        "value": value,
                        "units": units,
                        "interval": std_interval,
                    }
                    if include_flags:
                        rec["flag"] = str(obs.get("quality") or obs.get("flag") or "")
                    records.append(rec)
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict:
        """
        Make a GET request to the given endpoint with retry logic.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without leading slash), e.g. ``"Stations"``.
        params : dict
            Query parameters.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        NVEError
            On non-retryable HTTP errors or after all retries are exhausted.
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(
                    url, params=params, timeout=self.timeout
                )
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise NVEError(
                        f"Request to {url} failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff * attempt)
                continue

            if response.ok:
                return response.json()

            # Non-retryable client errors
            if response.status_code == 400:
                try:
                    body = response.json()
                    errors = body.get("errors") or {}
                    msg = (
                        f"{body.get('title', '')} {errors}"
                        if errors
                        else body.get("title") or response.text[:500]
                    )
                except Exception:
                    msg = response.text[:500]
                raise NVEError(f"HTTP 400 Bad Request: {msg}")

            if response.status_code == 404:
                # The API 404s e.g. when a requested series does not exist;
                # the body explains which — keep it in the error.
                raise NVEError(
                    f"HTTP 404 Not Found: {url} "
                    f"(params={params!r}): {response.text[:300]}"
                )

            # Rate limited — honour Retry-After if present, then retry
            if response.status_code == 429:
                if attempt < self.max_retries:
                    try:
                        delay = float(response.headers.get("Retry-After", ""))
                    except ValueError:
                        delay = float(self.backoff * attempt)
                    logger.warning(
                        "HTTP 429 from %s (attempt %d/%d) — retrying in %.1fs",
                        url, attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                raise NVEError(
                    f"HTTP 429 Too Many Requests from {url} "
                    f"after {self.max_retries} attempts"
                )

            # Retryable server errors (5xx)
            if response.status_code >= 500:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d) — retrying in %ds",
                    response.status_code, url,
                    attempt, self.max_retries, self.backoff * attempt,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise NVEError(
                    f"HTTP {response.status_code} from {url} "
                    f"after {self.max_retries} attempts"
                )

            # Other errors (401, 403, etc.)
            raise NVEError(
                f"HTTP {response.status_code} from {url}: {response.text[:200]}"
            )

        raise NVEError(f"Exhausted retries for {url}")  # should not reach here


# ── Exception ────────────────────────────────────────────────────────────────

class NVEError(Exception):
    """Raised when the NVE HydAPI returns an error or a request fails."""


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_variables(
    variables: list[str] | str | None,
) -> list[tuple[str, int, Any]]:
    """
    Translate a variable list (native keys or types) to
    ``(var_key, param_id, converter)`` tuples.

    The converter is a callable that takes a float and returns a float
    (applies unit conversion if needed).

    Parameters
    ----------
    variables : list[str] or str or None
        Variable key(s) (e.g. ``"swe_m"``) or standardized type(s)
        (e.g. ``"swe"``).  ``None`` returns all snow variables.

    Returns
    -------
    list of (var_key, param_id, converter) tuples
    """
    # Converters: swe is m → cm (× 100); snwd is cm → cm (identity)
    _converters: dict[str, Any] = {
        "swe_m":   lambda x: round(x * 100.0, 3),
        "snwd_cm": lambda x: round(x, 3),
    }

    if variables is None:
        # Default: all snow variables
        return [
            (vk, _VAR_TO_PARAM[vk], _converters[vk])
            for vk in ["swe_m", "snwd_cm"]
        ]

    raw_vars = [variables] if isinstance(variables, str) else list(variables)
    jobs: list[tuple[str, int, Any]] = []
    seen: set[str] = set()
    for v in raw_vars:
        if v in VARIABLES and v in _VAR_TO_PARAM:
            if v not in seen:
                jobs.append((v, _VAR_TO_PARAM[v], _converters[v]))
                seen.add(v)
        elif v in _TYPE_TO_NVE_VARS:
            for vk in _TYPE_TO_NVE_VARS[v]:
                if vk not in seen and vk in _VAR_TO_PARAM:
                    jobs.append((vk, _VAR_TO_PARAM[vk], _converters[vk]))
                    seen.add(vk)
        else:
            logger.warning("Unknown variable %r — skipping", v)
    return jobs or [
        (vk, _VAR_TO_PARAM[vk], _converters[vk])
        for vk in ["swe_m", "snwd_cm"]
    ]
