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
- Metric-first outputs: AWDB values that are known to be imperial are converted
    in-client; specifically WTEQ and SNWD are returned in centimeters.

Intended usage
--------------
    from global_snow_point_obs.clients import AWDBClient

    client = AWDBClient()

    # List all active SNOTEL stations in Colorado
    stations = client.get_stations(networks=["SNTL"], states=["CO"])

    # Get metadata + element inventory for a batch of stations
    meta = client.get_metadata(
        triplets=["303:CO:SNTL", "713:CO:SNTL"],
        elements=["WTEQ", "SNWD"],
        durations=["DAILY"],
    )

    # Fetch daily SWE for WY2024
    data = client.get_data(
        triplets=["303:CO:SNTL"],
        elements=["WTEQ"],
        duration="DAILY",
        begin_date="2023-10-01",
        end_date="2024-09-30",
    )
    # WTEQ values are returned in centimeters.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

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

# AWDB commonly serves snow values in inches; normalize to metric at source.
_CM_CONVERSIONS: dict[str, float] = {
    "WTEQ": 2.54,
    "SNWD": 2.54,
}


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

        return self._get("stations", params)

    def get_metadata(
        self,
        triplets: list[str] | str,
        elements: list[str] | str = "*",
        durations: list[str] | str = "*",
        include_forecast_point: bool = True,
        include_reservoir: bool = True,
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
        to avoid URL-length limits.

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

        results = []
        for batch in _chunk(triplets, 150):
            params = {
                "stationTriplets": ",".join(batch),
                "elements": elements_str,
                "durations": durations_str,
                "returnStationElements": "true",
                "returnForecastPointMetadata": str(include_forecast_point).lower(),
                "returnReservoirMetadata": str(include_reservoir).lower(),
                "activeOnly": str(active_only).lower(),
            }
            batch_result = self._get("stations", params)
            if isinstance(batch_result, list):
                results.extend(batch_result)

        return results

    def get_data(
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
        >>> data = client.get_data(
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
        batch_size = max(1, _MAX_VALUES // (n_elements * max(n_days, 1)))

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
        """Convert in-place AWDB /data payloads to metric where applicable."""
        for station_data in station_blocks:
            for data_block in station_data.get("data", []):
                station_element = data_block.get("stationElement", {})
                element_code = station_element.get("elementCode")
                factor = _CM_CONVERSIONS.get(str(element_code or ""))
                if factor is None:
                    continue

                values = data_block.get("values", [])
                if values and isinstance(values[0], dict):
                    for rec in values:
                        self._convert_value_record(rec, factor)
                elif values:
                    converted: list[float | None] = []
                    for v in values:
                        converted.append(self._convert_scalar_value(v, factor))
                    data_block["values"] = converted

                if station_element.get("originalUnitCode"):
                    station_element["originalUnitCode"] = "cm"
                if station_element.get("storedUnitCode"):
                    station_element["storedUnitCode"] = "cm"
                station_element["convertedUnitCode"] = "cm"

    @staticmethod
    def _convert_value_record(rec: dict, factor: float) -> None:
        for key in ("value", "average", "median"):
            if key in rec:
                rec[key] = AWDBClient._convert_scalar_value(rec.get(key), factor)

    @staticmethod
    def _convert_scalar_value(value: Any, factor: float) -> float | None:
        if value is None:
            return None
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(fval):
            return None
        return round(fval * factor, 3)

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
            Additional keyword arguments passed to :meth:`get_data`.

        Returns
        -------
        list[dict]
            Same structure as :meth:`get_data`.
        """
        begin = f"{water_year - 1}-10-01"
        end   = f"{water_year}-09-30"
        return self.get_data(
            triplets=triplets,
            elements=elements,
            duration=duration,
            begin_date=begin,
            end_date=end,
            **kwargs,
        )

    def get_forecasts(
        self,
        triplets: list[str] | str,
        elements: list[str] | str = "SRVO",
        begin_publication_date: str | date | None = None,
        end_publication_date: str | date | None = None,
    ) -> list[dict]:
        """
        Retrieve seasonal streamflow or SWE forecasts.

        Parameters
        ----------
        triplets : list[str] or str
            Forecast point station triplet(s).
        elements : list[str] or str
            Forecast element code(s).  Common values:
            ``"SRVO"`` (natural flow volume, kaf),
            ``"WTEQ"`` (SWE forecast).
        begin_publication_date : str or date, optional
            Start of the publication date range (``"YYYY-MM-DD"``).
        end_publication_date : str or date, optional
            End of the publication date range.

        Returns
        -------
        list[dict]
            One dict per station with ``publicationDate``, ``forecastPeriod``,
            and ``forecastValues`` (exceedance probabilities).

        Example
        -------
        >>> fcst = client.get_forecasts(
        ...     "09085000:CO:USGS",
        ...     begin_publication_date="2024-01-01",
        ...     end_publication_date="2024-06-30",
        ... )
        """
        triplets = _coerce_list(triplets)

        begin_str = _date_str(begin_publication_date) if begin_publication_date else None
        end_str   = _date_str(end_publication_date)   if end_publication_date   else None

        results = []
        for batch in _chunk(triplets, 50):
            params: dict[str, str] = {
                "stationTriplets": ",".join(batch),
                "elementCodes":    ",".join(_coerce_list(elements)),
            }
            if begin_str:
                params["beginPublicationDate"] = begin_str
            if end_str:
                params["endPublicationDate"] = end_str

            batch_result = self._get("forecasts", params)
            if isinstance(batch_result, list):
                results.extend(batch_result)

        return results

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
            Same structure as :meth:`get_data` with ``median``/``average``
            fields alongside ``value`` in the values list.

        Example
        -------
        >>> norms = client.get_normals(
        ...     "303:CO:SNTL", ["WTEQ"], normal_period="1991-2020"
        ... )
        """
        begin_year, end_year = normal_period.split("-")
        return self.get_data(
            triplets=triplets,
            elements=elements,
            duration=duration,
            begin_date=f"{begin_year}-10-01",
            end_date=f"{end_year}-09-30",
            central_tendency_type=central_tendency_type,
        )

    def get_reservoir_metadata(
        self,
        triplets: list[str] | str,
    ) -> list[dict]:
        """
        Retrieve reservoir-specific metadata (capacity, storage levels, etc.).

        Parameters
        ----------
        triplets : list[str] or str
            Station triplet(s) for reservoir stations (typically BOR network).

        Returns
        -------
        list[dict]
            Full metadata dicts with ``reservoir`` block populated.
        """
        return self.get_metadata(
            triplets=triplets,
            elements="RESC",
            durations="MONTHLY",
            include_reservoir=True,
        )

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
