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
Authentication    : None — fully open public API

Key parameters
--------------
- Parameter 2001 : Snow depth (cm)
- Parameter 2002 : Snow Water Equivalent / SWE (mm)
- ResolutionTime 1440 : Daily (1440 min)
- ResolutionTime 60   : Hourly (60 min)

Station URL convention
----------------------
Each station has a public page on the NVE Sildre portal:
  ``https://sildre.nve.no/station/{station_id}``

Design principles
-----------------
- Returns plain Python objects (dicts / lists).
- Metric-first: SWE returned in cm (converted from mm ÷ 10). Snow depth
  returned in cm as-is. All other variables in their native SI units.
- Missing / sentinel values (None, -9999, NaN) are normalised to ``None``.
- ``include_flags=True`` on ``get_data()`` adds a ``flag`` key to each
  value record.
- HTTP retry logic is applied to all requests.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://hydapi.nve.no/api/v1"

_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

# Sentinel / missing value used by some NVE responses
_MISSING_VALUES = {-9999, -9999.0}

# NVE parameter IDs for snow variables
_PARAM_SWE   = 2002  # Snow Water Equivalent (mm)
_PARAM_SNWD  = 2001  # Snow depth (cm)

# Temporal resolution in minutes
_RESOLUTION_DAILY  = 1440
_RESOLUTION_HOURLY = 60

# Sildre station URL template
_SILDRE_URL = "https://sildre.nve.no/station/{station_id}"

_NVE_DATA_SOURCE = "NVE HydAPI v1 — https://hydapi.nve.no/api/v1/Observations"

# ── Public variable / flag tables ────────────────────────────────────────────

#: Known NVE hydrological parameters relevant to snow monitoring.
VARIABLES: dict[str, dict] = {
    "swe_mm": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "cm",
        "source": _NVE_DATA_SOURCE + " (ParameterId=2002)",
        "description": (
            "Snow water equivalent from automated snow pillow. "
            "Native API unit is mm; returned here in cm (÷ 10)."
        ),
        "notes": "Parameter ID 2002. Native units: mm.",
    },
    "snwd_cm": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "source": _NVE_DATA_SOURCE + " (ParameterId=2001)",
        "description": "Snow depth from automated sensor. Native API unit is cm.",
        "notes": "Parameter ID 2001. Native units: cm.",
    },
}

#: Mapping from standardized type → NVE variable key(s) (priority order).
_TYPE_TO_NVE_VARS: dict[str, list[str]] = {
    "swe":  ["swe_mm"],
    "snwd": ["snwd_cm"],
}

#: Mapping from NVE parameter ID → variable key.
_PARAM_TO_VAR: dict[int, str] = {
    _PARAM_SWE:  "swe_mm",
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
    "0": "No flag / good data",
    "1": "Interpolated",
    "2": "Estimated / corrected",
    "3": "Dubious",
    "4": "Missing",
    "9": "No data",
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
        ``station_url``, ``parameters``.
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

    # Build list of parameter IDs available at this station
    series_list = raw.get("seriesList") or []
    param_ids: list[int] = sorted(
        {int(s["parameterId"]) for s in series_list if s.get("parameterId") is not None}
    )

    return {
        "station_id": sid,
        "name": raw.get("stationName") or raw.get("name") or "",
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elev,
        "drainage_basin_key": raw.get("drainageBasinKey") or "",
        "status": status,
        "station_url": _SILDRE_URL.format(station_id=sid) if sid else "",
        "parameters": param_ids,
    }


# ── Client ───────────────────────────────────────────────────────────────────

class NVEClient:
    """
    Client for the NVE HydAPI (Norwegian hydrological data service).

    No authentication is required.  All endpoints are publicly accessible.

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({
            "accept": "application/json",
            "User-Agent": "global-snow-networks/1.0",
        })

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
            Raw observation records as returned by the API (one per timestamp),
            each with at least ``time``, ``value``, and optionally
            ``quality`` fields.

        Raises
        ------
        NVEError
            On request failure.
        """
        params: dict[str, Any] = {
            "StationId": station_id,
            "ParameterId": parameter_id,
            "ResolutionTime": resolution,
        }
        if begin_date is not None:
            params["StartDate"] = f"{_date_str(begin_date)}T00:00"
        if end_date is not None:
            params["EndDate"] = f"{_date_str(end_date)}T23:59"

        raw = self._get("Observations", params)
        return raw.get("data") or []

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
            NVE variable key(s) (e.g. ``"swe_mm"``) **or** standardized
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
                    "variable": "swe_mm",
                    "type": "swe",
                    "value": 12.5,   # cm (converted from mm ÷ 10)
                    "units": "cm",
                    "interval": "daily",
                    # "flag": "0"  (only present when include_flags=True)
                }

        Notes
        -----
        SWE values (parameter 2002) are stored by NVE in mm.  This method
        converts them to cm (÷ 10) so that ``"units"`` is always ``"cm"``
        for type ``"swe"``.

        Snow depth values (parameter 2001) are stored by NVE in cm and are
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

        records: list[dict] = []
        for sid in ids:
            for var_key, param_id, converter in var_jobs:
                var_info = VARIABLES[var_key]
                std_type = var_info["type"]
                units = var_info["units"]
                try:
                    obs_list = self.get_observations(
                        station_id=sid,
                        parameter_id=param_id,
                        begin_date=begin_date,
                        end_date=end_date,
                        resolution=resolution,
                    )
                except NVEError as exc:
                    logger.warning(
                        "Failed to fetch %s for station %s: %s",
                        var_key, sid, exc,
                    )
                    continue

                for obs in obs_list:
                    raw_val = _normalize_value(obs.get("value"))
                    value = converter(raw_val) if raw_val is not None else None
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
                    msg = response.json().get("message", response.text[:200])
                except Exception:
                    msg = response.text[:200]
                raise NVEError(f"HTTP 400 Bad Request: {msg}")

            if response.status_code == 404:
                raise NVEError(f"HTTP 404 Not Found: {url}")

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
        Variable key(s) (e.g. ``"swe_mm"``) or standardized type(s)
        (e.g. ``"swe"``).  ``None`` returns all snow variables.

    Returns
    -------
    list of (var_key, param_id, converter) tuples
    """
    # Converters: swe is mm → cm (÷ 10); snwd is cm → cm (identity)
    _converters: dict[str, Any] = {
        "swe_mm":  lambda x: round(x / 10.0, 3),
        "snwd_cm": lambda x: round(x, 3),
    }

    if variables is None:
        # Default: all snow variables
        return [
            (vk, _VAR_TO_PARAM[vk], _converters[vk])
            for vk in ["swe_mm", "snwd_cm"]
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
        for vk in ["swe_mm", "snwd_cm"]
    ]
