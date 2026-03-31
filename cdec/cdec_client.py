# -*- coding: utf-8 -*-
"""
clients/cdec/cdec_client.py
===========================
Python client for CDEC (California Data Exchange Center), operated by the
California Department of Water Resources (CA DWR).

Primary use case: California Cooperative Snow Surveys (CCSS) — both automated
snow pillow stations (daily SWE) and manual snow course sites (periodic).

API documentation : https://cdec.water.ca.gov
Station metadata  : https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=QUA
Data service      : https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet
Snow courses list : https://cdec.water.ca.gov/reportapp/javareports?name=SnowCourses
Snow sensors list : https://cdec.water.ca.gov/reportapp/javareports?name=SnowSensors

Key snow sensors
----------------
- Sensor  3 : Snow Water Content (raw pillow reading, inches SWE)
- Sensor 18 : Snow Depth (ultrasonic, inches)
- Sensor 82 : SNO ADJ — Snow Water Content Revised (quality-controlled SWE,
              inches).  This is the preferred SWE variable for analysis and
              is stored as ``wteq_cm`` in per-station CSVs.

SWE vs. SNO ADJ
---------------
Sensor 3 (SNOW WC) is the raw telemetered reading from the snow pillow load
cell.  Sensor 82 (SNO ADJ) is a revised version with a constant calibration
offset applied after manual QC; it carries the ``r`` (revised) data flag.
Most CCSS pillow stations have both sensors 3 and 82; sensor 82 is preferred.
Snow course sites (manual surveys) only report periodic measurements and do not
have automated daily sensors.

Design principles
-----------------
- Returns plain Python objects (dicts / lists).
- Metric-first: all returned values are in centimetres (× 2.54 from inches).
- ``include_flags=True`` on ``get_data()`` adds a ``flag`` key to each value
  record.  Flag values are NOT stored in per-station CSVs.
- Missing values (-9999) are normalised to ``None``.
- HTML endpoints are parsed with pandas.read_html(); station IDs are matched
  with a short regex to avoid header/footer rows.
"""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://cdec.water.ca.gov"

_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4
_MISSING = -9999
_INCHES_TO_CM = 2.54

# Regex for valid CDEC station IDs (2–5 uppercase alphanumeric characters)
_STATION_ID_RE = re.compile(r"^[A-Z0-9]{2,5}$")


# ── Public sensor / flag / duration tables ───────────────────────────────────

#: Known snow-relevant CDEC sensors.
SENSORS: dict[int, dict[str, str]] = {
    3: {
        "name": "Snow Water Content",
        "short_name": "SNOW WC",
        "units": "in",
        "variable": "swe_raw",
        "description": "Raw snow pillow reading (SWE, inches).",
    },
    18: {
        "name": "Snow Depth",
        "short_name": "SNOW DP",
        "units": "in",
        "variable": "snwd",
        "description": "Ultrasonic snow depth sensor (inches).",
    },
    82: {
        "name": "Snow Water Content (Adjusted)",
        "short_name": "SNO ADJ",
        "units": "in",
        "variable": "swe",
        "description": (
            "Quality-controlled SWE with calibration offset applied "
            "(preferred over sensor 3 for analysis)."
        ),
    },
}

#: CDEC data quality flags.
DATA_FLAGS: dict[str, str] = {
    " ": "Unreviewed / provisional",
    "A": "Precipitation accumulation period",
    "L": "Awaiting observer response",
    "N": "Error in data",
    "c": "Calculated (gridded precipitation)",
    "e": "Estimated",
    "o": "Calibration offset applied",
    "q": "New rating table applied",
    "r": "Revised",
    "s": "New shift applied",
    "t": "Trace precipitation",
    "v": "Out of valid range",
}

#: CDEC duration codes.
DURATION_CODES: dict[str, str] = {
    "D": "Daily",
    "M": "Monthly",
    "H": "Hourly",
    "E": "Event (sub-hourly)",
}


# ── Client ───────────────────────────────────────────────────────────────────

class CDECClient:
    """
    Client for CDEC (California Data Exchange Center).

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Retry attempts on transient server errors (5xx / connection errors).
    backoff : int
        Base backoff delay in seconds (actual delay = backoff × attempt).
    session : requests.Session or None
        Optional pre-configured session.
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "global-snow-networks/1.0"})

    # ── Public API ────────────────────────────────────────────────────────────

    def get_snow_courses(self) -> list[dict]:
        """
        Fetch the CCSS manual snow course station list.

        Source: CDEC SnowCourses report page (HTML table).

        Returns
        -------
        list[dict]
            One dict per snow course with keys:
            ``station_id``, ``course_number``, ``name``, ``elevation_ft``,
            ``latitude``, ``longitude``, ``april1_avg_swe_in``,
            ``measuring_agency``, ``is_snow_course``, ``station_url``.
        """
        url = f"{BASE_URL}/reportapp/javareports?name=SnowCourses"
        html = self._get_html(url)
        courses = []
        for t in _read_html_tables(html):
            # Identify the snow courses table by looking for an "ID" column
            cols_lower = [str(c).lower().strip() for c in t.columns]
            if "id" not in cols_lower:
                continue
            t = _normalise_snow_courses_table(t)
            if t is None:
                continue
            for _, row in t.iterrows():
                sid = str(row.get("station_id", "")).strip().upper()
                if not _STATION_ID_RE.match(sid):
                    continue
                courses.append(
                    {
                        "station_id": sid,
                        "course_number": _str(row.get("course_number")),
                        "name": _str(row.get("name")),
                        "elevation_ft": _to_float(
                            str(row.get("elevation_ft", "")).replace(",", "")
                        ),
                        "latitude": _to_float(row.get("latitude")),
                        "longitude": _to_float(row.get("longitude")),
                        "april1_avg_swe_in": _to_float(
                            row.get("april1_avg_swe_in")
                        ),
                        "measuring_agency": _str(row.get("measuring_agency")),
                        "is_snow_course": True,
                        "is_snow_pillow": False,
                        "has_daily_swe": False,
                        "has_daily_snwd": False,
                        "station_url": (
                            f"{BASE_URL}/dynamicapp/staMeta?station_id={sid}"
                        ),
                    }
                )
            if courses:
                break
        return courses

    def get_snow_pillows(self) -> list[dict]:
        """
        Fetch the CDEC automated snow pillow (CCSS sensor) station list.

        Source: CDEC SnowSensors report page (HTML table).

        Returns
        -------
        list[dict]
            One dict per station with keys:
            ``station_id``, ``name``, ``elevation_ft``, ``latitude``,
            ``longitude``, ``april1_avg_swe_in``, ``operator``,
            ``is_snow_pillow``, ``has_daily_swe``, ``station_url``.
        """
        url = f"{BASE_URL}/reportapp/javareports?name=SnowSensors"
        html = self._get_html(url)
        pillows = []
        for t in _read_html_tables(html):
            cols_lower = [str(c).lower().strip() for c in t.columns]
            if "id" not in cols_lower and not any(
                "station" in c for c in cols_lower
            ):
                continue
            t = _normalise_snow_sensors_table(t)
            if t is None:
                continue
            for _, row in t.iterrows():
                sid = str(row.get("station_id", "")).strip().upper()
                if not _STATION_ID_RE.match(sid):
                    continue
                pillows.append(
                    {
                        "station_id": sid,
                        "name": _str(row.get("name")),
                        "elevation_ft": _to_float(
                            str(row.get("elevation_ft", "")).replace(",", "")
                        ),
                        "latitude": _to_float(row.get("latitude")),
                        "longitude": _to_float(row.get("longitude")),
                        "april1_avg_swe_in": _to_float(
                            row.get("april1_avg_swe_in")
                        ),
                        "operator": _str(row.get("operator")),
                        "is_snow_course": False,
                        "is_snow_pillow": True,
                        "has_daily_swe": True,
                        "has_daily_snwd": False,
                        "station_url": (
                            f"{BASE_URL}/dynamicapp/staMeta?station_id={sid}"
                        ),
                    }
                )
            if pillows:
                break
        return pillows

    def get_stations(
        self,
        sensors: tuple[int, ...] = (3, 18, 82),
        active_only: bool = False,
    ) -> list[dict]:
        """
        Get all CDEC stations that have any of the specified snow sensors.

        Queries the CDEC station search for each sensor number and merges the
        results.  Supplements with the official SnowCourses and SnowSensors
        report pages to flag CCSS membership.

        Parameters
        ----------
        sensors : tuple[int, ...]
            CDEC sensor numbers to include.  Defaults to (3, 18, 82).
        active_only : bool
            If True, request only currently active stations.

        Returns
        -------
        list[dict]
            One dict per unique station with keys from staSearch plus:
            ``sensors`` (sorted list of found sensor numbers),
            ``has_daily_swe``, ``has_daily_snwd``,
            ``is_snow_course``, ``is_snow_pillow``, ``station_url``.
        """
        all_by_id: dict[str, dict] = {}

        for sensor_num in sensors:
            params: dict[str, str] = {
                "sensor_chk": "on",
                "sensor": str(sensor_num),
                "elev1": "-5",
                "elev2": "99000",
                "numRecord": "2000",
                "submit_btn": "Search",
            }
            if active_only:
                params["active"] = "Y"

            try:
                html = self._get_html(
                    f"{BASE_URL}/dynamicapp/staSearch", params=params
                )
            except CDECError as exc:
                logger.warning(
                    "Station search for sensor %d failed: %s", sensor_num, exc
                )
                continue

            rows = _parse_station_search_html(html)
            for row in rows:
                sid = row["station_id"]
                if sid not in all_by_id:
                    all_by_id[sid] = dict(row)
                    all_by_id[sid]["sensors"] = set()
                    all_by_id[sid]["is_snow_course"] = False
                    all_by_id[sid]["is_snow_pillow"] = False
                all_by_id[sid]["sensors"].add(sensor_num)

        # Mark snow courses and snow pillows from official lists
        try:
            for course in self.get_snow_courses():
                sid = course["station_id"]
                if sid in all_by_id:
                    all_by_id[sid]["is_snow_course"] = True
                    all_by_id[sid].setdefault(
                        "course_number", course.get("course_number")
                    )
                    all_by_id[sid].setdefault(
                        "measuring_agency", course.get("measuring_agency")
                    )
                    all_by_id[sid].setdefault(
                        "april1_avg_swe_in", course.get("april1_avg_swe_in")
                    )
        except CDECError as exc:
            logger.warning("Could not fetch snow courses list: %s", exc)

        try:
            for pillow in self.get_snow_pillows():
                sid = pillow["station_id"]
                if sid in all_by_id:
                    all_by_id[sid]["is_snow_pillow"] = True
        except CDECError as exc:
            logger.warning("Could not fetch snow pillows list: %s", exc)

        # Finalise derived flags and clean up sensor sets
        for sid, sta in all_by_id.items():
            sset = sta.get("sensors", set())
            sta["sensors"] = sorted(sset)
            sta["has_daily_swe"] = bool({3, 82} & sset)
            sta["has_daily_snwd"] = bool({18} & sset)
            sta.setdefault(
                "station_url",
                f"{BASE_URL}/dynamicapp/staMeta?station_id={sid}",
            )

        return list(all_by_id.values())

    def get_metadata(self, station_id: str) -> dict:
        """
        Fetch full station metadata by scraping the CDEC staMeta page.

        Returns all metadata available on the station page including the
        sensor inventory (sensor number, description, duration, date range).

        Parameters
        ----------
        station_id : str
            CDEC station ID (e.g. ``"QUA"``).

        Returns
        -------
        dict
            Keys: ``station_id``, ``name``, ``elevation_ft``, ``river_basin``,
            ``county``, ``hydrologic_area``, ``nearby_city``, ``latitude``,
            ``longitude``, ``operator``, ``maintenance``, ``sensors``
            (list of sensor inventory dicts), ``station_url``.
        """
        url = f"{BASE_URL}/dynamicapp/staMeta"
        html = self._get_html(url, params={"station_id": station_id})
        return _parse_sta_meta_html(station_id, html)

    def get_data(
        self,
        station_ids: list[str] | str,
        sensors: list[int] | int,
        duration: str = "D",
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Fetch time-series data for one or more stations and sensors.

        Values are converted from inches to centimetres in-place.
        Missing observations (-9999) are normalised to ``None``.

        Parameters
        ----------
        station_ids : list[str] or str
            CDEC station ID(s), e.g. ``["QUA", "BLC"]`` or ``"QUA"``.
        sensors : list[int] or int
            Sensor number(s) to retrieve.  For snow: 3, 18, 82.
        duration : str
            Duration code: ``"D"`` (daily), ``"H"`` (hourly), ``"M"``
            (monthly), ``"E"`` (event).  Monthly data (``"M"``) is not
            available for sensors 3, 18, 82 — use ``"D"`` for daily values.
        begin_date : str or date, optional
            Start date (``"YYYY-MM-DD"``).  Defaults to earliest available.
        end_date : str or date, optional
            End date (inclusive).  Defaults to today.
        include_flags : bool
            If True, each value record includes a ``"flag"`` key.  Flags are
            not stored in per-station CSVs but are useful for QC analysis.

        Returns
        -------
        list[dict]
            One dict per station::

                {
                    "stationId": "QUA",
                    "data": [
                        {
                            "stationElement": {
                                "sensorNum": 82,
                                "sensorType": "SNO ADJ",
                                "sensorName": "Snow Water Content (Adjusted)",
                                "durationCode": "D",
                                "durationName": "Daily",
                                "units": "cm",
                            },
                            "values": [
                                {"date": "2024-01-01", "value": 24.7},
                                ...
                            ]
                        },
                        ...
                    ]
                }

        Notes
        -----
        Sensor 82 (SNO ADJ) is the preferred SWE variable.  If a station only
        has sensor 3 (raw SWE), that is returned instead.  Sensor 18 provides
        snow depth.
        """
        if isinstance(station_ids, str):
            station_ids = [station_ids]
        if isinstance(sensors, int):
            sensors = [sensors]

        begin_str = _date_str(begin_date) if begin_date else "1900-01-01"
        end_str = _date_str(end_date) if end_date else date.today().isoformat()

        # CDEC JSONDataServlet accepts comma-separated station IDs and sensors.
        # Batch by station to keep URL lengths reasonable.
        batch_size = 20
        results_by_id: dict[str, dict] = {}

        for i in range(0, len(station_ids), batch_size):
            batch = station_ids[i: i + batch_size]
            params = {
                "Stations": ",".join(batch),
                "SensorNums": ",".join(str(s) for s in sensors),
                "dur_code": duration,
                "Start": begin_str,
                "End": end_str,
            }
            url = f"{BASE_URL}/dynamicapp/req/JSONDataServlet"
            raw = self._get_json(url, params=params)

            if not isinstance(raw, list):
                logger.warning(
                    "Unexpected response from JSONDataServlet: %s", type(raw)
                )
                continue

            for record in raw:
                sid = str(record.get("stationId", "")).strip().upper()
                if not sid:
                    continue
                sensor_num = int(record.get("SENSOR_NUM", 0))
                raw_val = record.get("value")

                # Normalise missing value and convert inches → cm
                if raw_val is None or raw_val == _MISSING:
                    value_cm: float | None = None
                else:
                    try:
                        fv = float(raw_val)
                        value_cm = (
                            None
                            if fv == _MISSING
                            else round(fv * _INCHES_TO_CM, 3)
                        )
                    except (TypeError, ValueError):
                        value_cm = None

                v: dict[str, Any] = {
                    "date": _normalise_cdec_date(
                        str(record.get("date", ""))
                    ),
                    "value": value_cm,
                }
                if include_flags:
                    v["flag"] = str(record.get("dataFlag", "")).strip()

                if sid not in results_by_id:
                    results_by_id[sid] = {"stationId": sid, "data": {}}

                sensor_key = (sensor_num, duration)
                if sensor_key not in results_by_id[sid]["data"]:
                    si = SENSORS.get(sensor_num, {})
                    results_by_id[sid]["data"][sensor_key] = {
                        "stationElement": {
                            "sensorNum": sensor_num,
                            "sensorType": si.get("short_name", ""),
                            "sensorName": si.get("name", ""),
                            "durationCode": duration,
                            "durationName": DURATION_CODES.get(
                                duration, duration
                            ),
                            "units": "cm",
                        },
                        "values": [],
                    }
                results_by_id[sid]["data"][sensor_key]["values"].append(v)

        # Convert nested dict to list format
        output = []
        for sid, payload in results_by_id.items():
            output.append(
                {
                    "stationId": sid,
                    "data": list(payload["data"].values()),
                }
            )
        return output

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_html(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> str:
        resp = self._request("GET", url, params=params)
        return resp.text

    def _get_json(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        resp = self._request("GET", url, params=params)
        return resp.json()

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
    ) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(
                    method, url, params=params, timeout=self.timeout
                )
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise CDECError(
                        f"Request to {url} failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff * attempt)
                continue

            if resp.ok:
                return resp

            if resp.status_code in (400, 404):
                raise CDECError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
                )

            if resp.status_code >= 500:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt,
                    self.max_retries,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise CDECError(
                    f"HTTP {resp.status_code} from {url} after "
                    f"{self.max_retries} attempts"
                )

            raise CDECError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
            )

        raise CDECError(f"Exhausted retries for {url}")


# ── Exception ─────────────────────────────────────────────────────────────────

class CDECError(Exception):
    """Raised when the CDEC API returns an error or a request fails."""


# ── HTML parsing helpers ──────────────────────────────────────────────────────

def _read_html_tables(html: str) -> list[pd.DataFrame]:
    """Return list of DataFrames parsed from HTML, or empty list on failure."""
    try:
        return pd.read_html(io.StringIO(html))
    except Exception:
        return []


def _parse_station_search_html(html: str) -> list[dict]:
    """
    Parse the CDEC staSearch result HTML into a list of station dicts.

    Expected table columns (by position):
    0: ID, 1: Station Name, 2: River Basin, 3: County,
    4: Longitude, 5: Latitude, 6: Elevation (ft), 7: Operator, 8: Map
    """
    tables = _read_html_tables(html)
    for t in tables:
        if len(t.columns) < 7:
            continue
        # Identify station table: first data column should look like IDs
        col0 = t.iloc[1:, 0].dropna().astype(str).str.strip()
        if col0.str.match(r"^[A-Z0-9]{2,5}$").sum() < 3:
            continue

        # Assign fixed column names by position
        names = [
            "station_id",
            "name",
            "river_basin",
            "county",
            "longitude",
            "latitude",
            "elevation_ft",
            "operator",
        ]
        extra = [f"_col{i}" for i in range(8, len(t.columns))]
        t.columns = names + extra

        stations = []
        for _, row in t.iterrows():
            sid = str(row.get("station_id", "")).strip().upper()
            if not _STATION_ID_RE.match(sid):
                continue
            elev_raw = str(row.get("elevation_ft", "")).replace(",", "")
            stations.append(
                {
                    "station_id": sid,
                    "name": _str(row.get("name")),
                    "river_basin": _str(row.get("river_basin")),
                    "county": _str(row.get("county")),
                    "longitude": _to_float(row.get("longitude")),
                    "latitude": _to_float(row.get("latitude")),
                    "elevation_ft": _to_float(elev_raw),
                    "operator": _str(row.get("operator")),
                }
            )
        if stations:
            return stations
    return []


def _normalise_snow_courses_table(
    t: pd.DataFrame,
) -> pd.DataFrame | None:
    """Rename columns of the SnowCourses table to standard names."""
    col_map: dict[Any, str] = {}
    for col in t.columns:
        c = str(col).lower().strip()
        if c == "id":
            col_map[col] = "station_id"
        elif c in ("course #", "course#", "course"):
            col_map[col] = "course_number"
        elif c == "station":
            col_map[col] = "name"
        elif "elev" in c:
            col_map[col] = "elevation_ft"
        elif "lat" in c:
            col_map[col] = "latitude"
        elif "lon" in c:
            col_map[col] = "longitude"
        elif "april" in c or "avg" in c:
            col_map[col] = "april1_avg_swe_in"
        elif "agenc" in c or "measur" in c:
            col_map[col] = "measuring_agency"
    if "station_id" not in col_map.values():
        return None
    return t.rename(columns=col_map)


def _normalise_snow_sensors_table(
    t: pd.DataFrame,
) -> pd.DataFrame | None:
    """Rename columns of the SnowSensors table to standard names."""
    col_map: dict[Any, str] = {}
    for col in t.columns:
        c = str(col).lower().strip()
        if c == "id":
            col_map[col] = "station_id"
        elif c == "station":
            col_map[col] = "name"
        elif "elev" in c:
            col_map[col] = "elevation_ft"
        elif "lat" in c:
            col_map[col] = "latitude"
        elif "lon" in c:
            col_map[col] = "longitude"
        elif "april" in c or "avg" in c:
            col_map[col] = "april1_avg_swe_in"
        elif "agenc" in c or "oper" in c:
            col_map[col] = "operator"
    if "station_id" not in col_map.values():
        return None
    return t.rename(columns=col_map)


def _parse_sta_meta_html(station_id: str, html: str) -> dict:
    """
    Parse the staMeta HTML page for a single station.

    The page contains four tables:
    - Table 0: station info (key-value pairs in a grid layout)
    - Table 2: sensor inventory (one row per sensor × duration)
    """
    meta: dict[str, Any] = {
        "station_id": station_id.upper(),
        "station_url": (
            f"{BASE_URL}/dynamicapp/staMeta?station_id={station_id}"
        ),
    }
    tables = _read_html_tables(html)
    if not tables:
        return meta

    # Table 0: station info
    # Typically a 4-column table (label, value, label, value)
    if tables:
        t0 = tables[0]
        flat: dict[str, str] = {}
        for _, row in t0.iterrows():
            vals = [str(v).strip() for v in row if str(v).strip() not in ("", "nan")]
            for k, v in zip(vals[::2], vals[1::2]):
                flat[k.lower()] = v
        meta["elevation_ft"] = _to_float(
            flat.get("elevation", "").replace("ft", "").replace(",", "")
        )
        meta["river_basin"] = flat.get("river basin", "")
        meta["county"] = flat.get("county", "")
        meta["hydrologic_area"] = flat.get("hydrologic area", "")
        meta["nearby_city"] = flat.get("nearby city", "")
        lat_str = flat.get("latitude", "").replace("°", "")
        lon_str = flat.get("longitude", "").replace("°", "")
        meta["latitude"] = _to_float(lat_str)
        meta["longitude"] = _to_float(lon_str)
        meta["operator"] = flat.get("operator", "")
        meta["maintenance"] = flat.get("maintenance", "")

    # Table 2 (index 2 if present): sensor inventory
    if len(tables) > 2:
        t2 = tables[2]
        # Sensor inventory columns: Sensor Description | Sensor Number |
        #   Duration | Short Name | Data Collection | Data Available
        sensor_col_map: dict[Any, str] = {}
        for col in t2.columns:
            c = str(col).lower()
            if "desc" in c or "sensor d" in c:
                sensor_col_map[col] = "sensor_description"
            elif "number" in c or "num" in c:
                sensor_col_map[col] = "sensor_num"
            elif "dur" in c:
                sensor_col_map[col] = "duration"
            elif "short" in c:
                sensor_col_map[col] = "short_name"
            elif "collect" in c:
                sensor_col_map[col] = "data_collection"
            elif "avail" in c:
                sensor_col_map[col] = "data_available"
        t2 = t2.rename(columns=sensor_col_map)

        sensors_list = []
        for _, row in t2.iterrows():
            snum_raw = str(row.get("sensor_num", "")).strip()
            if not snum_raw or snum_raw.lower() in ("nan", "sensor number"):
                continue
            try:
                snum = int(float(snum_raw))
            except ValueError:
                continue
            sensors_list.append(
                {
                    "sensor_num": snum,
                    "sensor_description": _str(
                        row.get("sensor_description")
                    ),
                    "duration": _str(row.get("duration")),
                    "short_name": _str(row.get("short_name")),
                    "data_collection": _str(row.get("data_collection")),
                    "data_available": _str(row.get("data_available")),
                }
            )
        meta["sensor_inventory"] = sensors_list
    else:
        meta["sensor_inventory"] = []

    return meta


# ── Utility helpers ───────────────────────────────────────────────────────────

def _normalise_cdec_date(date_str: str) -> str:
    """Convert CDEC date strings like '2023-1-1 00:00' to 'YYYY-MM-DD'."""
    s = date_str.strip().split(" ")[0]
    parts = s.split("-")
    if len(parts) == 3:
        try:
            return (
                f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            )
        except ValueError:
            pass
    return s[:10]


def _date_str(d: str | date | datetime) -> str:
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", "").strip())
        return None if (f != f) else f  # NaN check
    except (ValueError, TypeError):
        return None


def _str(val: Any) -> str:
    s = str(val).strip()
    return "" if s.lower() == "nan" else s
