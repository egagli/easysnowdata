# -*- coding: utf-8 -*-
"""
clients/awdb_client.py
======================
A clean, flexible Python client for the USDA NRCS AWDB REST API v1.

API documentation : https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
Demo notebooks    : https://github.com/nrcs-nwcc/iow_awdb_rest_api_demo

Design principles
-----------------
- All methods return plain Python objects (dicts / lists) so callers can
  decide how to parse or convert to pandas/xarray.
- Batching and retry logic are handled internally; callers just ask for data.
- No hard-coded element codes or network codes — everything is parameterized.
- The 500,000 value-per-request API limit is enforced automatically; the
  client splits requests and re-assembles the results transparently.
- Metric-first outputs (DESIGN.md §3.5): imperial AWDB values are converted
  in-client — WTEQ/SNWD in centimetres, temperatures in °C, precipitation
  in mm, wind speeds in km/h.

Intended usage
--------------
    from clients.awdb import AWDBClient

    client = AWDBClient()

    # List all active SNOTEL stations in Colorado
    stations = client.get_stations(networks=["SNTL"], states=["CO"])

    # Get metadata + element inventory for a batch of stations
    meta = client.get_metadata(
        triplets=["303:CO:SNTL", "713:CO:SNTL"],
        elements=["WTEQ", "SNWD"],
        durations=["DAILY"],
    )

    # Fetch daily SWE for WY2024 (values in centimetres)
    records = client.get_data(
        station_ids=["303:CO:SNTL"],
        variables=["swe"],
        interval="daily",
        begin_date="2023-10-01",
        end_date="2024-09-30",
    )
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

# Maximum number of (stations × elements × days) the API will serve in one
# request.  We keep a comfortable margin below the documented 500,000 limit.
_MAX_VALUES = 450_000

_DEFAULT_TIMEOUT = 180      # seconds
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 6        # seconds (multiplied by attempt number)

#: Known AWDB element codes with standardized type and unit metadata.
#: Elements not listed here are returned with ``type="other"``.
#: ``units`` is the native unit AWDB serves; ``output_units`` is what
#: :meth:`AWDBClient.get_data` emits after in-client conversion
#: (DESIGN.md §3.5 — fully metric, no imperial passthrough).
_AWDB_DATA_SOURCE = "USDA NRCS AWDB REST API v1 — /data endpoint"

VARIABLES: dict[str, dict] = {
    "WTEQ":   {"name": "Snow Water Equivalent",          "type": "swe",       "units": "in",      "output_units": "cm",      "description": "Snow water equivalent. Converted in-client from inches to cm.", "notes": "",  "source": _AWDB_DATA_SOURCE},
    "SNWD":   {"name": "Snow Depth",                     "type": "snwd",      "units": "in",      "output_units": "cm",      "description": "Snow depth. Converted in-client from inches to cm.",            "notes": "",  "source": _AWDB_DATA_SOURCE},
    "TOBS":   {"name": "Air Temperature (observed)",     "type": "temp",      "units": "°F",      "output_units": "°C",      "description": "Instantaneous air temperature at observation time. Converted in-client to °C.", "notes": "",  "source": _AWDB_DATA_SOURCE},
    "TMAX":   {"name": "Maximum Air Temperature",        "type": "temp_max",  "units": "°F",      "output_units": "°C",      "description": "Daily maximum air temperature. Converted in-client to °C.",     "notes": "",  "source": _AWDB_DATA_SOURCE},
    "TMIN":   {"name": "Minimum Air Temperature",        "type": "temp_min",  "units": "°F",      "output_units": "°C",      "description": "Daily minimum air temperature. Converted in-client to °C.",     "notes": "",  "source": _AWDB_DATA_SOURCE},
    "PREC":   {"name": "Precipitation Accumulation",     "type": "precip",    "units": "in",      "output_units": "mm",      "description": "Cumulative seasonal precipitation accumulation. Converted in-client to mm.", "notes": "",  "source": _AWDB_DATA_SOURCE},
    "PRCP":   {"name": "Precipitation Increment",        "type": "precip",    "units": "in",      "output_units": "mm",      "description": "Precipitation increment since last observation. Converted in-client to mm.", "notes": "",  "source": _AWDB_DATA_SOURCE},
    "PRCPSA": {"name": "Precipitation Accumulation (storm)", "type": "precip","units": "in",      "output_units": "mm",      "description": "Storm-period precipitation accumulation. Converted in-client to mm.", "notes": "",  "source": _AWDB_DATA_SOURCE},
    "RHUM":   {"name": "Relative Humidity",              "type": "rh",        "units": "%",       "output_units": "%",       "description": "Relative humidity percentage.",                                 "notes": "",  "source": _AWDB_DATA_SOURCE},
    "WSPDV":  {"name": "Wind Speed Average",             "type": "wind_spd",  "units": "mph",     "output_units": "km/h",    "description": "Average wind speed. Converted in-client to km/h.",              "notes": "",  "source": _AWDB_DATA_SOURCE},
    "WSPDX":  {"name": "Wind Speed Maximum (Gust)",      "type": "wind_gust", "units": "mph",     "output_units": "km/h",    "description": "Maximum (gust) wind speed. Converted in-client to km/h.",       "notes": "",  "source": _AWDB_DATA_SOURCE},
    "WDIRV":  {"name": "Wind Direction",                 "type": "wind_dir",  "units": "degrees", "output_units": "degrees", "description": "Wind direction in degrees from north (0–360).",                "notes": "",  "source": _AWDB_DATA_SOURCE},
    "SRADV":  {"name": "Solar Radiation Average",        "type": "solar",     "units": "W/m²",    "output_units": "W/m²",    "description": "Average solar radiation.",                                      "notes": "",  "source": _AWDB_DATA_SOURCE},
}

# In-client metric conversions (DESIGN.md §3.5), keyed by element code.
# (transform, emitted unit).  Elements not listed pass through with their
# registry output_units (they are already metric).
_METRIC_CONVERSIONS: dict[str, tuple] = {
    "WTEQ":   (lambda v: v * 2.54, "cm"),
    "SNWD":   (lambda v: v * 2.54, "cm"),
    "TOBS":   (lambda v: (v - 32.0) * 5.0 / 9.0, "°C"),
    "TMAX":   (lambda v: (v - 32.0) * 5.0 / 9.0, "°C"),
    "TMIN":   (lambda v: (v - 32.0) * 5.0 / 9.0, "°C"),
    "PREC":   (lambda v: v * 25.4, "mm"),
    "PRCP":   (lambda v: v * 25.4, "mm"),
    "PRCPSA": (lambda v: v * 25.4, "mm"),
    "WSPDV":  (lambda v: v * 1.609344, "km/h"),
    "WSPDX":  (lambda v: v * 1.609344, "km/h"),
}

# Unit codes that mean the payload is ALREADY metric — if AWDB ever serves
# these for a convertible element, the transform must be skipped.
_ALREADY_METRIC_UNIT_CODES = {
    "cm", "mm", "m", "degC", "°C", "km/h", "kph", "metric",
}

#: The AWDB REST API v1 does not return per-value QC flags, so this
#: registry is empty.  It exists so the documented
#: ``from clients.awdb.awdb_client import DATA_FLAGS`` pattern works
#: uniformly across all clients.
DATA_FLAGS: dict[str, str] = {}

# Standardized interval → AWDB duration name
_INTERVAL_TO_AWDB_DURATION: dict[str, str] = {
    "daily":        "DAILY",
    "hourly":       "HOURLY",
    "monthly":      "MONTHLY",
    "semi_monthly": "SEMIMONTHLY",
    "annual":       "ANNUAL",
    "sub_daily":    "HOURLY",
}
# AWDB duration name → standardized interval
_AWDB_DURATION_TO_INTERVAL: dict[str, str] = {
    "DAILY":        "daily",
    "HOURLY":       "hourly",
    "MONTHLY":      "monthly",
    "SEMIMONTHLY":  "semi_monthly",
    "ANNUAL":       "annual",
    "WATER_YEAR":   "annual",
    "EVENT":        "sub_daily",
}
# Standardized type → AWDB element code(s)
_TYPE_TO_ELEMENTS: dict[str, list[str]] = {
    "swe":       ["WTEQ"],
    "snwd":      ["SNWD"],
    "temp":      ["TOBS", "TMAX", "TMIN"],
    "temp_max":  ["TMAX"],
    "temp_min":  ["TMIN"],
    "precip":    ["PREC", "PRCP", "PRCPSA"],
    "rh":        ["RHUM"],
    "wind_spd":  ["WSPDV"],
    "wind_gust": ["WSPDX"],
    "wind_dir":  ["WDIRV"],
    "solar":     ["SRADV"],
}


def _resolve_variables_to_awdb(variables: list[str] | str | None) -> list[str]:
    """Translate a variables list (native codes or types) to AWDB element codes."""
    if variables is None:
        return list(VARIABLES.keys())
    elems: list[str] = []
    seen: set[str] = set()
    var_list = (
        _coerce_list(variables) if isinstance(variables, str) else list(variables)
    )
    if not var_list:
        return list(VARIABLES.keys())
    for v in var_list:
        if v in VARIABLES:
            if v not in seen:
                elems.append(v)
                seen.add(v)
        elif v in _TYPE_TO_ELEMENTS:
            for e in _TYPE_TO_ELEMENTS[v]:
                if e not in seen:
                    elems.append(e)
                    seen.add(e)
        else:
            # No silent fallback (DESIGN.md §3.6): an unknown variable
            # must not quietly expand into "fetch everything".
            raise AWDBError(
                f"Unknown variable {v!r} for AWDB — expected an element "
                f"code ({sorted(VARIABLES)}) or a standardized type "
                f"({sorted(_TYPE_TO_ELEMENTS)})."
            )
    return elems


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


# ── Client ───────────────────────────────────────────────────────────────────

class AWDBClient:
    """
    Client for the USDA NRCS AWDB REST API v1.

    Parameters
    ----------
    base_url : str
        Base URL of the AWDB REST API.  Override for staging / mirror endpoints.
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Number of retry attempts on transient server errors (5xx).
    backoff : int
        Base backoff delay in seconds.  Actual delay = backoff × attempt_number.
    session : requests.Session or None
        Optional pre-configured session (useful for auth headers, proxies, etc.).
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

    # ── Public API ────────────────────────────────────────────────────────────

    def get_reference_data(
        self,
        tables: list[str] | str = "all",
    ) -> dict[str, Any]:
        """
        Fetch AWDB reference/lookup tables.

        Parameters
        ----------
        tables : list[str] or str
            One or more reference table names, or ``"all"`` for everything.
            Common tables: ``"networks"``, ``"elements"``, ``"states"``,
            ``"counties"``, ``"durations"``, ``"units"``.

        Returns
        -------
        dict
            Parsed JSON response.  Keys are table names; values are lists of
            ``{code, name, description}`` dicts.

        Example
        -------
        >>> client = AWDBClient()
        >>> ref = client.get_reference_data(["networks", "elements"])
        >>> ref["networks"][0]
        {'code': 'SNTL', 'name': 'SNOTEL', ...}
        """
        if isinstance(tables, list):
            tables = ",".join(tables)
        return self._get("reference-data", {"referenceLists": tables})

    def get_stations(
        self,
        networks: list[str] | str | None = None,
        states: list[str] | str | None = None,
        huc: str | None = None,
        county_name: str | None = None,
        active_only: bool = False,
        station_triplets: list[str] | str | None = None,
    ) -> list[dict]:
        """
        List stations matching the given filters.

        Returns only basic identification fields (no element inventory).
        Use :meth:`get_metadata` to retrieve full metadata including elements.

        Parameters
        ----------
        networks : list[str] or str, optional
            Network code(s) to filter by, e.g. ``["SNTL", "SNTLT"]``.
            Defaults to all networks (``"*"``).
        states : list[str] or str, optional
            Two-letter state code(s), e.g. ``["CO", "UT"]``.
        huc : str, optional
            Hydrologic Unit Code prefix (2–12 digits).  Stations whose HUC
            starts with this prefix are returned.
        county_name : str, optional
            County name filter (case-insensitive prefix match).
        active_only : bool
            If True, only return stations with no ``endDate``.
        station_triplets : list[str] or str, optional
            Explicit list of station triplets to retrieve.  Overrides network
            and state filters when provided.

        Returns
        -------
        list[dict]
            List of station dicts with keys: ``stationTriplet``,
            ``stationId``, ``networkCode``, ``name``, ``stateCode``,
            ``countyName``, ``huc``, ``latitude``, ``longitude``,
            ``elevation``, ``beginDate``, ``endDate``.

        Example
        -------
        >>> stations = client.get_stations(networks=["SNTL"], states=["CO"])
        >>> len(stations)
        117
        """
        params: dict[str, str] = {}

        if station_triplets is not None:
            triplets = _coerce_list(station_triplets)
            params["stationTriplets"] = ",".join(triplets)
        elif networks is not None:
            nets = _coerce_list(networks)
            params["stationTriplets"] = ",".join(f"*:*:{n}" for n in nets)
        else:
            params["stationTriplets"] = "*:*:*"

        if states:
            params["stateCodes"] = ",".join(_coerce_list(states))
        if huc:
            params["hucs"] = huc
        if county_name:
            params["countyNames"] = county_name
        if not active_only:
            params["activeOnly"] = "false"

        results = self._get("stations", params)
        for sta in results:
            _enrich_awdb_station(sta)
        # Post-filter by state: the API ignores stateCodes when
        # stationTriplets is a wildcard pattern.
        if states:
            states_set = {s.upper() for s in _coerce_list(states)}
            results = [
                s for s in results
                if s.get("stateCode", "").upper() in states_set
            ]
        return results

    def get_metadata(
        self,
        triplets: list[str] | str,
        elements: list[str] | str = "*",
        durations: list[str] | str = "*",
        include_forecast_point: bool = False,
        include_reservoir: bool = False,
        active_only: bool = False,
    ) -> list[dict]:
        """
        Retrieve full station metadata including the element inventory.

        Parameters
        ----------
        triplets : list[str] or str
            Station triplet(s), e.g. ``"303:CO:SNTL"`` or a list.
        elements : list[str] or str
            Element code(s) to filter the station element list by, e.g.
            ``["WTEQ", "SNWD"]``.  Pass ``"*"`` for all elements.
        durations : list[str] or str
            Duration name(s) to filter by, e.g. ``["DAILY", "MONTHLY"]``.
            Pass ``"*"`` for all durations.
        include_forecast_point : bool
            Include forecast point metadata if available.
        include_reservoir : bool
            Include reservoir metadata if available.
        active_only : bool
            If True, only return active elements.

        Returns
        -------
        list[dict]
            One dict per station.  Each dict contains all AWDB metadata fields
            plus a ``stationElements`` list (one entry per matching element ×
            duration combination).

        Notes
        -----
        Only stations that have at least one matching element are returned.
        Requests are automatically split into batches of at most 150 triplets
        and then split further on request-size errors to avoid URL-length and
        payload-size limits.

        Example
        -------
        >>> meta = client.get_metadata(
        ...     ["303:CO:SNTL", "713:CO:SNTL"],
        ...     elements=["WTEQ", "SNWD"],
        ...     durations=["DAILY"],
        ... )
        >>> meta[0]["stationElements"][0]["elementCode"]
        'SNWD'
        """
        triplets = _coerce_list(triplets)
        elements_str = ",".join(_coerce_list(elements)) if elements != "*" else "*"
        durations_str = ",".join(_coerce_list(durations)) if durations != "*" else "*"

        def is_request_too_large_error(exc: AWDBError) -> bool:
            message = str(exc).lower()
            return (
                "413" in message
                or "414" in message
                or ("400" in message and "exceeds the amount allowed" in message)
                or "request entity too large" in message
                or "request-uri too large" in message
            )

        def fetch_batch(batch: list[str]) -> list[dict]:
            params = {
                "stationTriplets": ",".join(batch),
                "elements": elements_str,
                "returnStationElements": "true",
                "returnForecastPointMetadata": str(
                    include_forecast_point
                ).lower(),
                "returnReservoirMetadata": str(
                    include_reservoir
                ).lower(),
                "activeOnly": str(active_only).lower(),
            }
            if durations != "*":
                params["durations"] = durations_str

            try:
                batch_result = self._get("stations", params)
            except AWDBError as exc:
                if not is_request_too_large_error(exc):
                    raise
                if len(batch) == 1:
                    raise
                mid = max(1, len(batch) // 2)
                left = fetch_batch(batch[:mid])
                right = fetch_batch(batch[mid:])
                return left + right

            return batch_result if isinstance(batch_result, list) else []

        results = []
        for batch in _chunk(triplets, 150):
            results.extend(fetch_batch(batch))

        return results

    def _get_data_awdb(
        self,
        triplets: list[str] | str,
        elements: list[str] | str,
        duration: str = "DAILY",
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        period_ref: str = "START",
        central_tendency_type: str | None = None,
    ) -> list[dict]:
        """
        Fetch time-series data for one or more stations and elements.

        Automatically batches requests to respect the 500,000-value API limit.
        All batches are merged and returned as a single list.

        Parameters
        ----------
        triplets : list[str] or str
            Station triplet(s).
        elements : list[str] or str
            Element code(s), e.g. ``["WTEQ", "SNWD"]`` or ``"WTEQ"``.
        duration : str
            Temporal resolution: ``"DAILY"``, ``"MONTHLY"``, ``"HOURLY"``,
            ``"SEMIMONTHLY"``, ``"ANNUAL"``, ``"WATER_YEAR"``.
        begin_date : str or date, optional
            Start date (``"YYYY-MM-DD"`` or ``datetime.date``).
            Defaults to the earliest available data.
        end_date : str or date, optional
            End date (inclusive).  Defaults to today.
        period_ref : str
            Period reference alignment: ``"START"`` or ``"END"``.
        central_tendency_type : str, optional
            Include normals alongside observed data: ``"MEDIAN"``,
            ``"AVERAGE"``.  Pass ``None`` to omit normals.

        Returns
        -------
        list[dict]
            One dict per station.  Each dict has the structure::

                {
                    "stationTriplet": "303:CO:SNTL",
                    "data": [
                        {
                            "stationElement": {
                                "elementCode": "WTEQ",
                                "durationName": "DAILY",
                                "originalUnitCode": "cm",
                                ...
                            },
                            "values": [
                                {"date": "2024-01-01", "value": 14.2},
                                ...
                            ]
                        },
                        ...   # one block per element
                    ]
                }

        Raises
        ------
        AWDBError
            If a batch request fails after all retries.

        Notes
        -----
        Returned units are metric-normalized for known imperial snow elements:
        ``WTEQ`` and ``SNWD`` are converted from inches to centimeters.

        The AWDB API enforces a hard limit of 500,000 data values per request:
        ``n_stations × n_elements × n_days ≤ 500,000``.  This client
        automatically splits the station list into batches that stay under the
        limit.  For long time series (many years) with multiple elements, the
        effective batch size may be as small as 2–5 stations.

        Example
        -------
        >>> data = client._get_data_awdb(
        ...     triplets=["303:CO:SNTL", "713:CO:SNTL"],
        ...     elements=["WTEQ", "SNWD"],
        ...     begin_date="2023-10-01",
        ...     end_date="2024-09-30",
        ... )
        >>> station = data[0]
        >>> station["stationTriplet"]
        '303:CO:SNTL'
        >>> station["data"][0]["stationElement"]["elementCode"]
        'SNWD'
        >>> station["data"][0]["values"][:2]
        [{'date': '2023-10-01', 'value': 0.0}, {'date': '2023-10-02', 'value': 0.0}]
        """
        triplets = _coerce_list(triplets)
        elements = _coerce_list(elements)

        begin_str = _date_str(begin_date) if begin_date else "1800-01-01"
        end_str   = _date_str(end_date)   if end_date   else date.today().isoformat()

        n_days     = (date.fromisoformat(end_str) - date.fromisoformat(begin_str)).days + 1
        n_elements = len(elements)
        # Cap at 75 stations per batch to avoid HTTP 414 (URL too long)
        batch_size = min(75, max(1, _MAX_VALUES // (n_elements * max(n_days, 1))))

        logger.debug(
            "get_data: %d stations, %d elements, %d days → batch_size=%d",
            len(triplets), n_elements, n_days, batch_size,
        )

        # Accumulate results keyed by triplet so we can merge batches
        results_by_triplet: dict[str, dict] = {}

        for batch in _chunk(triplets, batch_size):
            params: dict[str, str] = {
                "stationTriplets": ",".join(batch),
                "elements":        ",".join(elements),
                "duration":        duration,
                "beginDate":       begin_str,
                "endDate":         end_str,
                "periodRef":       period_ref,
            }
            if central_tendency_type:
                params["centralTendencyType"] = central_tendency_type

            batch_result = self._get("data", params)

            if not isinstance(batch_result, list):
                logger.warning("Unexpected response type from /data: %s", type(batch_result))
                continue

            self._convert_data_response_to_metric(batch_result)

            for station_data in batch_result:
                triplet = station_data.get("stationTriplet")
                if triplet:
                    results_by_triplet[triplet] = station_data

        return list(results_by_triplet.values())

    def _convert_data_response_to_metric(self, station_blocks: list[dict]) -> None:
        """Convert in-place AWDB /data payloads to metric (DESIGN.md §3.5).

        Every convertible element (snow in inches, temperature in °F,
        precipitation in inches, wind in mph) is transformed; elements
        already metric pass through.  Snow elements use cm (archive
        convention) rather than mm.
        """
        for station_data in station_blocks:
            for data_block in station_data.get("data", []):
                station_element = data_block.get("stationElement", {})
                element_code = str(station_element.get("elementCode") or "")
                conversion = _METRIC_CONVERSIONS.get(element_code)
                if conversion is None:
                    continue
                # Guard: skip if AWDB reports the payload already metric.
                unit_code = str(
                    station_element.get("storedUnitCode")
                    or station_element.get("originalUnitCode")
                    or ""
                )
                if unit_code in _ALREADY_METRIC_UNIT_CODES:
                    continue
                transform, out_unit = conversion

                values = data_block.get("values", [])
                if values and isinstance(values[0], dict):
                    for rec in values:
                        self._convert_value_record(rec, transform)
                elif values:
                    data_block["values"] = [
                        self._convert_scalar_value(v, transform)
                        for v in values
                    ]

                if station_element.get("originalUnitCode"):
                    station_element["originalUnitCode"] = out_unit
                if station_element.get("storedUnitCode"):
                    station_element["storedUnitCode"] = out_unit
                station_element["convertedUnitCode"] = out_unit

    @staticmethod
    def _convert_value_record(rec: dict, transform) -> None:
        for key in ("value", "average", "median"):
            if key in rec:
                rec[key] = AWDBClient._convert_scalar_value(
                    rec.get(key), transform
                )

    @staticmethod
    def _convert_scalar_value(value: Any, transform) -> float | None:
        if value is None:
            return None
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(fval):
            return None
        return round(transform(fval), 3)

    def get_data_by_water_year(
        self,
        triplets: list[str] | str,
        elements: list[str] | str,
        water_year: int,
        duration: str = "DAILY",
        **kwargs,
    ) -> list[dict]:
        """
        Convenience wrapper: fetch data for a single water year.

        A water year runs from October 1 of the *previous* calendar year to
        September 30 of ``water_year``.  For example, WY2024 spans
        2023-10-01 through 2024-09-30.

        Parameters
        ----------
        triplets : list[str] or str
        elements : list[str] or str
        water_year : int
            The water year integer (e.g., 2024).
        duration : str
        **kwargs
            Additional keyword arguments passed to :meth:`_get_data_awdb`.

        Returns
        -------
        list[dict]
            Same structure as :meth:`_get_data_awdb` (nested AWDB payload,
            not the flat records of :meth:`get_data`).
        """
        begin = f"{water_year - 1}-10-01"
        end   = f"{water_year}-09-30"
        return self._get_data_awdb(
            triplets=triplets,
            elements=elements,
            duration=duration,
            begin_date=begin,
            end_date=end,
            **kwargs,
        )


    def get_normals(
        self,
        triplets: list[str] | str,
        elements: list[str] | str,
        duration: str = "DAILY",
        normal_period: str = "1991-2020",
        central_tendency_type: str = "MEDIAN",
    ) -> list[dict]:
        """
        Retrieve climatological normals (medians or averages) for stations.

        Parameters
        ----------
        triplets : list[str] or str
        elements : list[str] or str
        duration : str
        normal_period : str
            Reference period string: ``"1991-2020"``, ``"1981-2010"``,
            or ``"1971-2000"``.
        central_tendency_type : str
            ``"MEDIAN"`` or ``"AVERAGE"``.

        Returns
        -------
        list[dict]
            Same structure as :meth:`_get_data_awdb` (nested AWDB payload)
            with ``median``/``average`` fields alongside ``value`` in the
            values list.

        Example
        -------
        >>> norms = client.get_normals(
        ...     "303:CO:SNTL", ["WTEQ"], normal_period="1991-2020"
        ... )
        """
        begin_year, end_year = normal_period.split("-")
        return self._get_data_awdb(
            triplets=triplets,
            elements=elements,
            duration=duration,
            begin_date=f"{begin_year}-10-01",
            end_date=f"{end_year}-09-30",
            central_tendency_type=central_tendency_type,
        )


    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Standardized station list.

        Parameters
        ----------
        active_only : bool
            If True, only return active stations.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` bounding box filter.

        Returns
        -------
        list[dict]
            Station dicts (same schema as :meth:`get_stations`).
        """
        stations = self.get_stations(active_only=active_only)
        if bbox is not None:
            stations = _filter_by_bbox(stations, bbox)
        return stations

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
            AWDB station triplet(s), e.g. ``"303:CO:SNTL"``.
            Required unless ``bbox`` is provided.
        variables : list[str] or str or None
            Element codes (e.g. ``"WTEQ"``) **or** standardized types
            (e.g. ``"swe"``).  ``None`` returns all elements in
            :data:`VARIABLES`.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.  Alternative to
            ``station_ids``; fetches data for all stations in the box.
        begin_date : str or date, optional
            Start date (``"YYYY-MM-DD"``).  Defaults to earliest available.
        end_date : str or date, optional
            End date (inclusive).  Defaults to today.
        interval : str
            Temporal resolution: ``"daily"``, ``"hourly"``, ``"monthly"``,
            etc.  Mapped to the AWDB ``duration`` parameter.
        include_flags : bool
            Reserved; the AWDB REST API does not return per-value QC flags.

        Returns
        -------
        list[dict]
            Flat list of observation records::

                {
                    "station_id": "303:CO:SNTL",
                    "date": "2024-01-15",
                    "variable": "WTEQ",
                    "type": "swe",
                    "value": 14.2,
                    "units": "cm",
                    "interval": "daily",
                    # "flag": None  (only present when include_flags=True)
                }

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        AWDBError
            If a batch request fails after all retries.
        """
        if station_ids is None and bbox is not None:
            ids = [s["stationTriplet"] for s in self.get_all_stations(bbox=bbox)]
        elif station_ids is not None:
            ids = _coerce_list(station_ids)
        else:
            raise ValueError("Provide station_ids or bbox.")
        if not ids:
            return []

        elements = _resolve_variables_to_awdb(variables)
        try:
            duration = _INTERVAL_TO_AWDB_DURATION[interval.lower()]
        except KeyError:
            raise AWDBError(
                f"Unsupported interval {interval!r} for AWDB — expected "
                f"one of {sorted(_INTERVAL_TO_AWDB_DURATION)}."
            ) from None
        sub_daily = duration == "HOURLY"
        raw = self._get_data_awdb(ids, elements, duration, begin_date, end_date)

        records: list[dict] = []
        for station_data in raw:
            triplet = station_data.get("stationTriplet", "")
            for block in station_data.get("data", []):
                elem_info = block.get("stationElement", {})
                elem_code = str(elem_info.get("elementCode") or "")
                dur_name = str(
                    elem_info.get("durationName") or "DAILY"
                ).upper()
                var_info = VARIABLES.get(elem_code, {})
                units = str(
                    elem_info.get("convertedUnitCode")
                    or var_info.get("output_units")
                    or elem_info.get("originalUnitCode")
                    or ""
                )
                std_interval = _AWDB_DURATION_TO_INTERVAL.get(
                    dur_name, dur_name.lower()
                )
                std_type = var_info.get("type", "other")
                for rec in block.get("values", []):
                    ts = str(rec.get("date", ""))
                    r: dict = {
                        "station_id": triplet,
                        "date": ts[:10],
                        "variable": elem_code,
                        "type": std_type,
                        "value": rec.get("value"),
                        "units": units,
                        "interval": std_interval,
                    }
                    if sub_daily:
                        r["datetime"] = ts
                    if include_flags:
                        r["flag"] = None
                    records.append(r)
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict[str, str]) -> Any:
        """
        Make a GET request to the given endpoint with retry logic.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without leading slash), e.g. ``"stations"``.
        params : dict[str, str]
            Query parameters.

        Returns
        -------
        Any
            Parsed JSON response body.

        Raises
        ------
        AWDBError
            On non-retryable HTTP errors or after all retries are exhausted.
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                logger.warning("Request failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    raise AWDBError(f"Request to {url} failed after {self.max_retries} attempts: {exc}") from exc
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
                raise AWDBError(f"HTTP 400 Bad Request: {msg}")

            if response.status_code == 404:
                raise AWDBError(f"HTTP 404 Not Found: {url}")

            # Retryable server errors (5xx)
            if response.status_code >= 500:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d) — retrying in %ds",
                    response.status_code, url, attempt, self.max_retries,
                    self.backoff * attempt,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise AWDBError(
                    f"HTTP {response.status_code} from {url} after {self.max_retries} attempts"
                )

            # Other errors (e.g. 401, 403)
            raise AWDBError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")

        raise AWDBError(f"Exhausted retries for {url}")  # should not reach here


# ── Exceptions ────────────────────────────────────────────────────────────────

class AWDBError(Exception):
    """Raised when the AWDB API returns an error or a request fails."""


# ── Private helpers ───────────────────────────────────────────────────────────

def _enrich_awdb_station(sta: dict) -> None:
    """Add station_url, station_image_url, elevation_m, status in-place."""
    network = str(sta.get("networkCode") or "")
    sid = str(sta.get("stationId") or "").strip()
    # station_url — NRCS site page for SNOTEL networks
    if not sta.get("station_url"):
        if network in {"SNTL", "SNTLT"} and sid:
            sta["station_url"] = (
                f"https://wcc.sc.egov.usda.gov/nwcc/site?sitenum={sid}"
            )
        else:
            sta["station_url"] = ""
    # station_image_url — predictable NRCS SNOTEL photo URL
    if not sta.get("station_image_url"):
        if network in {"SNTL", "SNTLT"} and sid and sid.isdigit():
            sta["station_image_url"] = (
                f"https://www.wcc.nrcs.usda.gov/siteimages/{sid}.jpg"
            )
    # elevation_m — convert feet to metres
    if "elevation_m" not in sta:
        elev_ft = sta.get("elevation")
        if elev_ft is not None:
            try:
                sta["elevation_m"] = round(float(elev_ft) * 0.3048, 1)
            except (TypeError, ValueError):
                pass
    # status — Active if no endDate, else Inactive
    if "status" not in sta:
        sta["status"] = "Active" if not sta.get("endDate") else "Inactive"


def _coerce_list(value: list | str) -> list[str]:
    """Ensure value is a list of strings."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _chunk(lst: list, size: int):
    """Yield successive sublists of at most ``size`` items."""
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def _date_str(d: str | date | datetime) -> str:
    """Normalize a date-like object to ``YYYY-MM-DD`` string."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()
