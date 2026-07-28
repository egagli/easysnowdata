# -*- coding: utf-8 -*-
"""
clients/yukon/yukon_client.py
=============================
Python client for the Yukon Water Data (AquaCache) API — the public data
service behind the Government of Yukon "Water Data Explorer".

Primary use cases:

1. **Yukon Snow Survey Network** — 90+ manual snow courses (10-point
   surveys, records from 1964) plus a small network of automated
   snow-weather stations with hourly snow-pillow SWE (records from 1980).
2. **ECCC Meteorology Network (Yukon)** — Environment and Climate Change
   Canada climate stations mirrored into AquaCache, several with long
   daily snow-depth records reaching the Arctic coast (Herschel Island
   69.57°N; Komakuk Beach from 1963).

API documentation : https://service.yukon.ca/water-data/api/v1/openapi.json
Base URL          : https://service.yukon.ca/water-data/api/v1
Authentication    : None — the API is fully open.

Endpoints used
--------------
- ``GET /locations``                    — all monitoring locations
- ``GET /timeseries``                   — continuous-series catalogue
- ``GET /timeseries/measurementsDaily`` — daily aggregates
- ``GET /timeseries/measurements``      — instantaneous measurements
- ``GET /snow-survey/metadata``         — snow course metadata
- ``GET /snow-survey/data``             — snow course survey archive
- ``GET /snow-survey/stats``            — per-course summary statistics
- ``GET /snow-survey/trends``           — per-course Mann-Kendall trends
- ``GET /grades`` / ``/approvals`` / ``/qualifiers`` — flag vocabularies

Station URL convention
----------------------
The Water Data Explorer is a Shiny application served behind a Cloudflare
JS challenge, so it has no scripted-accessible per-station permalink.
``station_url`` therefore points at the Explorer entry point, and
``dataset_url`` at the matching Open Yukon dataset landing page (which is
robot-accessible).  See ``_EXPLORER_URL`` / ``_DATASET_URLS``.

Design principles
-----------------
- Returns plain Python objects (dicts / lists).
- Metric-first: SWE is returned in **cm** (converted from native mm ÷ 10);
  snow depth is natively cm and passes through.  As in the DataBC client,
  other met variables keep their native units.
- ``VARIABLES[key]["units"]`` documents the **native** unit and
  ``["output_units"]`` the unit ``get_data()`` actually emits, so callers
  and the GeoJSON builder read one source of truth.
- Missing / sentinel values are normalised to ``None``, as are physically
  implausible snow values (negative or > 15 m).
- ``include_flags=True`` on ``get_data()`` adds a ``flag`` key to each
  record.
- HTTP retry logic is applied to all requests.

Source quirks handled here (all verified against the live API)
-------------------------------------------------------------
- ``/snow-survey/*`` responses are CSV preceded by **quoted** comment
  lines (``"# Description: ..."``).  The comment block ends with a line
  containing exactly ``""``, which a naive ``strip()`` filter mistakes
  for content — see :func:`_strip_api_comments`.
- ``/snow-survey/data`` reports an **empty** ``units`` field for snow
  depth.  It is cm, corroborated by the ``/snow-survey/stats`` field name
  ``max_DEPTH_cm``.  The unit is taken from :data:`VARIABLES`, never from
  the response.
- ``/locations`` serialises ``networks`` and ``projects`` as Postgres
  array literals (``{"Yukon Snow Survey Network","..."}`` or ``{NULL}``)
  — see :func:`_parse_pg_array`.
- Snow-survey ``month`` includes the non-integer value ``5.5`` meaning
  May 15, so it is parsed as a float, not an int.
- ``measurementsDaily.value`` is a **daily mean** over the *local* day
  (``day_timezone``, UTC-07 for Yukon sites), not an instantaneous
  snapshot — verified by reconstructing 2024-03-01 for timeseries 20 from
  ``/timeseries/measurements``.
- ``/timeseries`` may hold several series of the same parameter at one
  location distinguished only by ``aggregation_type`` (ECCC daily air
  temperature exists as minimum, maximum and ``(min+max)/2``).  Records
  therefore carry ``aggregation`` and ``timeseries_id`` alongside the
  standard keys.
- ``/organizations`` and ``/csw-layer`` are slow and are never on the
  critical path.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import date, datetime
from typing import Any

import requests

from clients._common import (
    chunk as _chunked,
    coerce_list as _coerce_list,
    date_str as _date_str,
    filter_by_bbox as _filter_by_bbox,
    request_with_retries,
    to_float as _to_float,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://service.yukon.ca/water-data/api/v1"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: Network names as they appear in the ``/locations`` ``networks`` array.
SNOW_SURVEY_NETWORK = "Yukon Snow Survey Network"
ECCC_NETWORK = "ECCC Meteorology Network"

#: ``location_type`` used by AquaCache for manual snow courses.
_SNOWPACK_TYPE = "snowpack"

#: Water Data Explorer entry point.  The app sits behind a Cloudflare JS
#: challenge and exposes no scripted per-station permalink, so every
#: station shares this URL.
_EXPLORER_URL = "https://service.yukon.ca/water-data/shiny/?page=home&lang=en"

#: Robot-accessible Open Yukon dataset landing pages, by station type.
_DATASET_URLS: dict[str, str] = {
    "SC": "https://open.yukon.ca/data/yukon-snow-survey-network",
    "AWS": "https://open.yukon.ca/data/yukon-snow-survey-network",
    "ECCC": "https://open.yukon.ca/data/water-monitoring-sites",
}

#: Operators, by station type.
_OPERATORS: dict[str, str] = {
    "SC": (
        "Yukon Government Department of Environment, "
        "Water Science and Stewardship"
    ),
    "AWS": (
        "Yukon Government Department of Environment, "
        "Water Science and Stewardship"
    ),
    "ECCC": "Environment and Climate Change Canada",
}

#: ``networkCode`` used downstream by the live map, by station type.
NETWORK_CODES: dict[str, str] = {"SC": "YSS", "AWS": "YSS", "ECCC": "YKEC"}

#: Yukon's approximate extent, ``(min_lon, min_lat, max_lon, max_lat)``:
#: 60°N in the south, 141°W against Alaska, ~123.8°W against the NWT.
_YUKON_BOX = (-141.0, 60.0, -123.8, 69.7)

#: Jurisdiction declared in a station's own name by the data provider.
_STATE_FROM_NAME: dict[str, str] = {
    "(b.c.)": "BC",
    "(bc)": "BC",
    "(alaska)": "AK",
}

#: Stations the Yukon Snow Survey operates outside Yukon whose names do not
#: declare a jurisdiction.  Each is assigned from its published position;
#: a name that later declares its own jurisdiction takes precedence, so an
#: upstream fix supersedes these entries automatically.
_STATE_OVERRIDES: dict[str, str] = {
    # 59.93°N 136.80°W — northern BC, ~8 km south of the Yukon border in
    # the Haines Road corridor.
    "08AC-SC01": "BC",
    # 58.28°N 134.53°W — Eaglecrest Ski Area, Juneau, Alaska.
    "08AK-SC01": "AK",
    # 59.52°N 135.25°W — Klondike Highway south of the White Pass summit,
    # on the Alaska side of the border.
    "08AK-SC02": "AK",
    # 59.76°N 134.97°W — co-located with course 09AA-SC03, which AquaCache
    # names "Log Cabin (B.C.) Snow Course".
    "09AA-M2": "BC",
}

#: Column set of the informational envelope the API returns instead of an
#: empty CSV when a query matches no rows.
_STATUS_ENVELOPE_FIELDS = {"status", "message"}

#: Values that mean "no observation".
_MISSING_VALUES = {-9999, -9999.0, -999, -999.0}

#: Plausibility bound for snow values in cm.  World-record snow depth is
#: ~11.8 m; anything outside [0, _MAX_PLAUSIBLE_CM] is normalised to
#: ``None``, matching the NVE and DataBC clients.
_MAX_PLAUSIBLE_CM = 1500.0

#: A snow course is considered active if it was surveyed within this many
#: years of today (``/locations`` carries no status field).
_COURSE_ACTIVE_WINDOW_YEARS = 2

#: Timeseries IDs per ``/timeseries/measurementsDaily`` request.  Kept
#: small so a single station's history cannot approach ``_ROW_LIMIT``.
_DAILY_ID_BATCH = 8

#: ``limit`` passed to the measurement endpoints (the documented maximum).
_ROW_LIMIT = 1_000_000

#: Earliest date used when no ``begin_date`` is given.  Komakuk Beach snow
#: depth starts 1963-12-26, so 1900 is comfortably early.
_EPOCH = "1900-01-01"

_YUKON_DATA_SOURCE = f"Yukon Water Data (AquaCache) API v1 — {BASE_URL}"

# ── Public variable / flag tables ────────────────────────────────────────────

#: AquaCache parameters exposed by this client.  Keys are named after the
#: **native** unit (as in the DataBC client); ``units`` is the native unit
#: and ``output_units`` is what :meth:`YukonClient.get_data` emits.
VARIABLES: dict[str, dict] = {
    "swe_mm": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "source": _YUKON_DATA_SOURCE + " (parameter 'snow water equivalent')",
        "description": (
            "Snow water equivalent. At automated snow-weather stations this "
            "comes from a snow pillow recorded hourly to 3-hourly; at snow "
            "courses it is the average of a 10-point manual survey. Native "
            "API unit is mm; returned here in cm (÷ 10)."
        ),
        "notes": (
            "Native units: mm. Snow course values carry a flag of 'Actual' "
            "(measured depth and SWE) or 'Estimated SWE' (measured depth "
            "with SWE estimated from historical average density)."
        ),
    },
    "snwd_cm": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "source": _YUKON_DATA_SOURCE + " (parameter 'snow depth')",
        "description": (
            "Snow depth from an automated sensor (Yukon and ECCC stations) "
            "or from a 10-point manual snow course survey. Native API unit "
            "is cm."
        ),
        "notes": (
            "Native units: cm. NB: /snow-survey/data leaves the units field "
            "empty for this parameter; cm is confirmed by the "
            "/snow-survey/stats field name 'max_DEPTH_cm'."
        ),
    },
    "air_temp_degc": {
        "name": "Air Temperature",
        "type": "temp",
        "units": "°C",
        "output_units": "°C",
        "source": _YUKON_DATA_SOURCE + " (parameter 'temperature, air')",
        "description": "Air temperature.",
        "notes": (
            "Instantaneous series, or the ECCC daily '(min+max)/2' mean. "
            "Native units: °C."
        ),
    },
    "air_temp_max_degc": {
        "name": "Air Temperature (daily maximum)",
        "type": "temp_max",
        "units": "°C",
        "output_units": "°C",
        "source": (
            _YUKON_DATA_SOURCE
            + " (parameter 'temperature, air', aggregation 'maximum')"
        ),
        "description": "Daily maximum air temperature.",
        "notes": "ECCC daily series only. Native units: °C.",
    },
    "air_temp_min_degc": {
        "name": "Air Temperature (daily minimum)",
        "type": "temp_min",
        "units": "°C",
        "output_units": "°C",
        "source": (
            _YUKON_DATA_SOURCE
            + " (parameter 'temperature, air', aggregation 'minimum')"
        ),
        "description": "Daily minimum air temperature.",
        "notes": "ECCC daily series only. Native units: °C.",
    },
    "precip_total_mm": {
        "name": "Precipitation (total)",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "source": _YUKON_DATA_SOURCE + " (parameter 'precipitation, total')",
        "description": "Total precipitation accumulated over the recording interval.",
        "notes": (
            "Aggregation type 'sum'. Returned in native mm without "
            "conversion, as in the DataBC client."
        ),
    },
    "precip_rain_mm": {
        "name": "Precipitation (rain)",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "source": _YUKON_DATA_SOURCE + " (parameter 'precipitation, rain')",
        "description": "Rainfall accumulated over the recording interval.",
        "notes": "Aggregation type 'sum'. Native units: mm.",
    },
    "precip_snow_cm": {
        "name": "Precipitation (snowfall)",
        "type": "precip",
        "units": "cm",
        "output_units": "cm",
        "source": _YUKON_DATA_SOURCE + " (parameter 'precipitation, snow')",
        "description": "New snowfall accumulated over the recording interval.",
        "notes": (
            "ECCC daily series. The /timeseries units field is empty for "
            "this parameter; cm comes from /parameters (id 1221)."
        ),
    },
    "rel_humidity_pct": {
        "name": "Relative Humidity",
        "type": "rh",
        "units": "%",
        "output_units": "%",
        "source": _YUKON_DATA_SOURCE + " (parameter 'relative humidity')",
        "description": "Relative humidity.",
        "notes": "Native units: %.",
    },
    "wind_spd_kmh": {
        "name": "Wind Speed",
        "type": "wind_spd",
        "units": "km/h",
        "output_units": "km/h",
        "source": _YUKON_DATA_SOURCE + " (parameter 'velocity, wind')",
        "description": "Wind speed.",
        "notes": "Native units: km/h.",
    },
    "wind_dir_deg": {
        "name": "Wind Direction",
        "type": "wind_dir",
        "units": "degrees",
        "output_units": "degrees",
        "source": (
            _YUKON_DATA_SOURCE
            + " (parameter 'wind direction (direction from, expressed 0-360 deg)')"
        ),
        "description": "Wind direction, degrees clockwise from north (0-360).",
        "notes": "Direction the wind blows *from*. Native units: degrees.",
    },
    "baro_press_kpa": {
        "name": "Barometric Pressure",
        "type": "baro",
        "units": "kPa",
        "output_units": "kPa",
        "source": _YUKON_DATA_SOURCE + " (parameter 'barometric pressure')",
        "description": "Barometric pressure.",
        "notes": (
            "Native units: kPa — note the DataBC client reports hPa for the "
            "same quantity (1 kPa = 10 hPa)."
        ),
    },
    "soil_moisture_pct": {
        "name": "Soil Moisture Content",
        "type": "other",
        "units": "%",
        "output_units": "%",
        "source": _YUKON_DATA_SOURCE + " (parameter 'moisture content')",
        "description": "Volumetric soil moisture content.",
        "notes": "Native units: %.",
    },
}

#: Mapping from standardized type → Yukon variable key(s) (priority order).
_TYPE_TO_YUKON_VARS: dict[str, list[str]] = {
    "swe": ["swe_mm"],
    "snwd": ["snwd_cm"],
    "temp": ["air_temp_degc"],
    "temp_max": ["air_temp_max_degc"],
    "temp_min": ["air_temp_min_degc"],
    "precip": ["precip_total_mm", "precip_rain_mm", "precip_snow_cm"],
    "rh": ["rel_humidity_pct"],
    "wind_spd": ["wind_spd_kmh"],
    "wind_dir": ["wind_dir_deg"],
    "baro": ["baro_press_kpa"],
    "other": ["soil_moisture_pct"],
}

#: Variable keys that carry snow information — the default for
#: :meth:`YukonClient.get_data` and the set archived to per-station CSVs.
SNOW_VARIABLES: list[str] = ["swe_mm", "snwd_cm"]

#: AquaCache parameter name → variable key, for parameters whose meaning
#: does not depend on ``aggregation_type``.
_PARAM_TO_VAR: dict[str, str] = {
    "snow water equivalent": "swe_mm",
    "snow depth": "snwd_cm",
    "precipitation, total": "precip_total_mm",
    "precipitation, rain": "precip_rain_mm",
    "precipitation, snow": "precip_snow_cm",
    "relative humidity": "rel_humidity_pct",
    "velocity, wind": "wind_spd_kmh",
    "wind direction (direction from, expressed 0-360 deg)": "wind_dir_deg",
    "barometric pressure": "baro_press_kpa",
    "moisture content": "soil_moisture_pct",
}

#: ``(parameter name, aggregation_type)`` → variable key, for parameters
#: that AquaCache stores as several series distinguished only by their
#: aggregation (ECCC daily air temperature).
_PARAM_AGG_TO_VAR: dict[tuple[str, str], str] = {
    ("temperature, air", "maximum"): "air_temp_max_degc",
    ("temperature, air", "minimum"): "air_temp_min_degc",
}

#: Fallback for parameters present in :data:`_PARAM_AGG_TO_VAR`.
_PARAM_AGG_DEFAULT: dict[str, str] = {"temperature, air": "air_temp_degc"}

#: AquaCache ``recording_rate`` → standardized interval name.  Anything
#: finer than hourly is reported as ``"sub_daily"``; the repo treats
#: ``daily``, ``sub_daily`` and ``hourly`` alike when deciding whether a
#: station has daily data.
_RATE_TO_INTERVAL: dict[str, str] = {
    "1 day": "daily",
    "01:00:00": "hourly",
    "00:30:00": "sub_daily",
    "00:15:00": "sub_daily",
    "00:05:00": "sub_daily",
    "00:01:00": "sub_daily",
    "03:00:00": "sub_daily",
}

#: ``month`` value in ``/snow-survey/data`` → survey period label, in the
#: ``DD-Mon`` form used by the DataBC client's ``survey_period``.
_SURVEY_PERIODS: dict[float, str] = {
    2.0: "01-Feb",
    3.0: "01-Mar",
    4.0: "01-Apr",
    5.0: "01-May",
    5.5: "15-May",
}

#: Grade codes from ``GET /grades`` (data quality).
GRADE_FLAGS: dict[str, str] = {
    "A": "Excellent",
    "B": "Good",
    "C": "Fair",
    "D": "Poor",
    "N": "Unusable",
    "UNK": "Unknown",
    "UNS": "Unspecified",
    "MISS": "Missing data",
    "E": "Estimated",
    "HD": "Historical records, daily mean",
    "HI": "Historical records, instantaneous",
}

#: Approval codes from ``GET /approvals`` (review status).
APPROVAL_FLAGS: dict[str, str] = {
    "A": "Approved",
    "C": "Ready for review",
    "R": "Reviewed, pending approval",
    "N": "Not reviewed",
    "UNK": "Unknown",
    "UNS": "Unspecified",
    "RR": "Reviewed, requires revision",
    "S": "Auto screened, no human review",
}

#: Qualifier codes from ``GET /qualifiers`` (conditions affecting a value).
QUALIFIER_FLAGS: dict[str, str] = {
    "ICE": "Ice present",
    "ICE-EST": "Ice interpolation/estimation",
    "DRY": "Dry well/stream/lake (not only sensor out of water)",
    "OOW": "Sensor out of water",
    "SUS": "Suspect measurements",
    "EST": "Estimated",
    "DD": "Draw-down after pumping",
    "BW": "Backwater affecting measurements",
    "INT": "Interpolated data",
    "HW-MISS": "High water missed (peak not recorded)",
    "LW-MISS": "Low water missed (trough not recorded)",
    "PMMAX": "Peak monthly maximum",
    "PMMIN": "Peak monthly minimum",
    "PYMAX": "Peak yearly maximum",
    "PYMIN": "Peak yearly minimum",
    "REL": "Release of water, e.g. beaver dam breaking",
    "UNS": "Unspecified",
    "UNK": "Unknown",
    "N": "Unusable",
    "US-DISTURB": "Upstream disturbances (obstructions and/or diversions)",
}

#: Flags specific to ``/snow-survey/data``.
SNOW_SURVEY_FLAGS: dict[str, str] = {
    "Actual": (
        "Averaged from actual snow depth and snow water equivalent readings"
    ),
    "Estimated SWE": (
        "Averaged from actual snow depth readings with snow water equivalent "
        "estimated from current depth and average historical density"
    ),
}

#: Flags emitted by ``/timeseries/measurementsDaily``.
DAILY_FLAGS: dict[str, str] = {
    "imputed": "Daily value was imputed rather than computed from measurements",
    "": "No flag / value computed from measurements",
}

#: Union of every flag vocabulary this client can emit, keyed by code.
#: ``get_data()`` sets ``flag`` from the vocabulary appropriate to the
#: interval: :data:`SNOW_SURVEY_FLAGS` for ``"periodic"``,
#: :data:`DAILY_FLAGS` for ``"daily"``, and the grade / approval /
#: qualifier codes for instantaneous requests.
DATA_FLAGS: dict[str, str] = {
    **DAILY_FLAGS,
    **SNOW_SURVEY_FLAGS,
    **{f"grade:{k}": v for k, v in GRADE_FLAGS.items()},
    **{f"approval:{k}": v for k, v in APPROVAL_FLAGS.items()},
    **{f"qualifier:{k}": v for k, v in QUALIFIER_FLAGS.items()},
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _to_bool(value: Any) -> bool:
    """Parse an AquaCache ``TRUE``/``FALSE`` CSV field."""
    return str(value).strip().upper() == "TRUE"


def _normalize_value(value: Any) -> float | None:
    """Convert a raw field to float, mapping sentinels to ``None``."""
    num = _to_float(value)
    if num is None or num in _MISSING_VALUES:
        return None
    return num


def _clamp_snow(value: float | None) -> float | None:
    """Null out physically implausible snow values (in cm)."""
    if value is None:
        return None
    if not 0 <= value <= _MAX_PLAUSIBLE_CM:
        return None
    return value


def _strip_api_comments(text: str) -> str:
    """
    Remove the quoted comment header that precedes ``/snow-survey/*`` CSVs.

    The block looks like::

        "# Description: Snow survey measurements for ..."
        "# Generated at : 2026-07-27 15:14 MST"
        "# "
        ""
        location_code,location_name,parameter,...

    Note the terminating line containing exactly ``""`` — filtering on
    ``line.strip()`` alone treats it as content and misparses the header,
    so it is matched explicitly.

    Parameters
    ----------
    text : str
        Raw response body.

    Returns
    -------
    str
        The body from its real header row onwards.
    """
    lines = text.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"#') or stripped.startswith("#") \
                or stripped == '""' or stripped == "":
            start = idx + 1
            continue
        break
    return "\n".join(lines[start:])


def _parse_csv(text: str) -> list[dict]:
    """
    Parse an AquaCache CSV response (comment header tolerated) to dicts.

    A query that matches nothing does not return an empty CSV — the API
    replies ``status,message`` with a single informational row, e.g.
    ``info,"No daily measurements found for the specified timeseries and
    date range."``.  That envelope is recognised and reduced to ``[]`` so
    callers never see it as data.
    """
    body = _strip_api_comments(text)
    if not body.strip():
        return []
    reader = csv.DictReader(io.StringIO(body))
    if set(reader.fieldnames or []) == _STATUS_ENVELOPE_FIELDS:
        for row in reader:
            logger.debug(
                "API returned a status envelope: %s", row.get("message", "")
            )
        return []
    return [dict(row) for row in reader]


def _parse_pg_array(value: Any) -> list[str]:
    """
    Parse a Postgres array literal as serialised by the AquaCache API.

    ``/locations`` and ``/timeseries`` return array columns like
    ``{"Yukon Snow Survey Network","Yukon Small Stream Network"}``,
    ``{NULL}`` or ``[]``.

    Parameters
    ----------
    value : Any
        Raw field value.

    Returns
    -------
    list[str]
        Parsed elements, with ``NULL`` entries dropped.

    Examples
    --------
    >>> _parse_pg_array('{"Yukon Snow Survey Network"}')
    ['Yukon Snow Survey Network']
    >>> _parse_pg_array('{NULL}')
    []
    >>> _parse_pg_array('[]')
    []
    """
    if value is None:
        return []
    text = str(value).strip()
    if text in {"", "{}", "[]", "{NULL}", "NULL"}:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if not text:
        return []
    # The elements are comma-separated with optional double quoting, which
    # csv.reader handles including embedded commas.
    try:
        parts = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        return []
    return [p.strip() for p in parts if p.strip() and p.strip().upper() != "NULL"]


def _series_variable_key(parameter: str, aggregation: str) -> str | None:
    """
    Map an AquaCache ``(parameter_name, aggregation_type)`` pair to a
    :data:`VARIABLES` key.

    Returns ``None`` for parameters this client does not expose (water
    flow, water level, pH, turbidity and the rest of the hydrometric and
    water-quality catalogue).
    """
    param = (parameter or "").strip().lower()
    agg = (aggregation or "").strip().lower()
    if (param, agg) in _PARAM_AGG_TO_VAR:
        return _PARAM_AGG_TO_VAR[(param, agg)]
    if param in _PARAM_AGG_DEFAULT:
        return _PARAM_AGG_DEFAULT[param]
    return _PARAM_TO_VAR.get(param)


def _series_interval(recording_rate: str) -> str:
    """Map an AquaCache ``recording_rate`` to a standardized interval name."""
    rate = (recording_rate or "").strip()
    if rate in _RATE_TO_INTERVAL:
        return _RATE_TO_INTERVAL[rate]
    if rate.endswith("day") or rate.endswith("days"):
        return "daily"
    # HH:MM:SS forms not in the table: hourly if exactly one hour, else
    # sub_daily.
    if rate.startswith("01:00"):
        return "hourly"
    return "sub_daily" if ":" in rate else "periodic"


def _survey_period(month: Any) -> str:
    """Map a ``/snow-survey/data`` ``month`` value to a ``DD-Mon`` label."""
    num = _to_float(month)
    if num is None:
        return ""
    return _SURVEY_PERIODS.get(num, "")


def _station_state(
    station_id: str,
    name: str,
    latitude: float | None,
    longitude: float | None,
) -> str:
    """
    Derive the province / territory / state code for a station.

    The Yukon Snow Survey operates a handful of courses outside Yukon —
    "Atlin (B.C.)", "Boundary (Alaska)", the Eaglecrest course at Juneau —
    so a blanket ``"YT"`` would mislabel them.  Resolution order:

    1. A jurisdiction declared in the station's own name (authoritative,
       and it tracks upstream naming changes).
    2. Inside :data:`_YUKON_BOX` → ``"YT"``.
    3. An entry in :data:`_STATE_OVERRIDES`.

    Parameters
    ----------
    station_id : str
        AquaCache ``location_code``.
    name : str
        Station name.
    latitude, longitude : float or None
        WGS-84 position.

    Returns
    -------
    str
        Two-letter code, or ``""`` when the station lies outside Yukon and
        no jurisdiction can be established.

    Examples
    --------
    >>> _station_state("09AA-SC04", "Atlin (B.C.) Snow Course", 59.6, -133.7)
    'BC'
    >>> _station_state("08AA-SC01", "Canyon Lake Snow Course", 61.1, -137.0)
    'YT'
    """
    lowered = (name or "").lower()
    for marker, code in _STATE_FROM_NAME.items():
        if marker in lowered:
            return code

    if latitude is not None and longitude is not None:
        min_lon, min_lat, max_lon, max_lat = _YUKON_BOX
        if min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon:
            return "YT"

    override = _STATE_OVERRIDES.get(station_id)
    if override:
        return override

    logger.debug(
        "Station %s (%s) lies outside Yukon with no declared jurisdiction",
        station_id, name,
    )
    return ""


def _course_status(last_survey: str | None) -> str:
    """Derive ``Active`` / ``Inactive`` for a snow course from its last survey."""
    year = _to_float((last_survey or "")[:4])
    if year is None:
        return "Inactive"
    return (
        "Active"
        if year >= date.today().year - _COURSE_ACTIVE_WINDOW_YEARS
        else "Inactive"
    )


# ── Client ───────────────────────────────────────────────────────────────────

class YukonClient:
    """
    Client for the Yukon Water Data (AquaCache) API.

    Parameters
    ----------
    base_url : str
        API base URL.  Defaults to :data:`BASE_URL`.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Number of attempts for retryable failures.
    backoff : int
        Linear backoff multiplier, in seconds, between attempts.
    session : requests.Session, optional
        Session to reuse.  A new one is created when omitted.

    Examples
    --------
    >>> client = YukonClient()
    >>> stations = client.get_all_stations()
    >>> records = client.get_data(
    ...     station_ids=["09AA-M1"], variables=["swe"], interval="daily",
    ...     begin_date="2024-01-01", end_date="2024-01-15",
    ... )
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
        self._session.headers.update({"Accept": "text/csv"})

        # Response caches — the catalogue endpoints are queried repeatedly
        # while resolving stations and variables.
        self._locations_cache: list[dict] | None = None
        self._timeseries_cache: list[dict] | None = None
        self._survey_cache: list[dict] | None = None
        self._course_meta_cache: list[dict] | None = None

    # ── Catalogue endpoints ───────────────────────────────────────────────

    def get_locations(
        self,
        location_types: list[str] | str | None = None,
        networks: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Return monitoring locations from ``GET /locations``.

        Parameters
        ----------
        location_types : list[str] or str, optional
            Keep only these ``location_type`` values, e.g. ``"snowpack"``
            or ``"meteorological station"``.
        networks : list[str] or str, optional
            Keep only locations belonging to at least one of these
            networks, e.g. :data:`SNOW_SURVEY_NETWORK`.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            One dict per location with keys ``location_id``,
            ``location_code``, ``name``, ``alias``, ``location_type``,
            ``latitude``, ``longitude``, ``elevation_m``, ``datum``,
            ``note``, ``networks`` and ``projects``.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        if self._locations_cache is None:
            rows = self._get_csv("locations", {"lang": "en"})
            self._locations_cache = [self._parse_location(r) for r in rows]

        result = list(self._locations_cache)

        if location_types is not None:
            wanted_types = {t.lower() for t in _coerce_list(location_types)}
            result = [
                loc for loc in result
                if (loc.get("location_type") or "").lower() in wanted_types
            ]
        if networks is not None:
            wanted_nets = set(_coerce_list(networks))
            result = [
                loc for loc in result
                if wanted_nets & set(loc.get("networks") or [])
            ]
        return _filter_by_bbox(result, bbox)

    def get_timeseries(
        self,
        location_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        publicly_visible_only: bool = True,
    ) -> list[dict]:
        """
        Return the continuous-series catalogue from ``GET /timeseries``.

        Only series whose parameter is exposed by this client (see
        :data:`VARIABLES`) are returned — the endpoint also lists water
        flow, water level, groundwater and water-quality series that have
        no place in a snow archive.

        Parameters
        ----------
        location_ids : list[str] or str, optional
            Restrict to these AquaCache numeric ``location_id`` values.
        variables : list[str] or str, optional
            Restrict to these :data:`VARIABLES` keys or standardized types.
        publicly_visible_only : bool
            Drop series flagged ``publicly_visible = FALSE`` (default True).

        Returns
        -------
        list[dict]
            One dict per series with keys ``timeseries_id``,
            ``location_id``, ``location_name``, ``variable``, ``type``,
            ``parameter_name``, ``aggregation``, ``recording_rate``,
            ``interval``, ``units``, ``output_units``, ``start_datetime``,
            ``end_datetime``, ``active``, ``publicly_visible``, ``owner``
            and ``networks``.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        if self._timeseries_cache is None:
            rows = self._get_csv("timeseries", {"lang": "en"})
            parsed: list[dict] = []
            for row in rows:
                series = self._parse_series(row)
                if series is not None:
                    parsed.append(series)
            self._timeseries_cache = parsed

        result = list(self._timeseries_cache)

        if publicly_visible_only:
            result = [s for s in result if s.get("publicly_visible")]
        if location_ids is not None:
            wanted_locs = set(_coerce_list(location_ids))
            result = [
                s for s in result if str(s.get("location_id")) in wanted_locs
            ]
        if variables is not None:
            wanted_vars = set(_resolve_variables(variables))
            result = [s for s in result if s.get("variable") in wanted_vars]
        return result

    # ── Station endpoints ─────────────────────────────────────────────────

    def get_snow_course_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Return Yukon snow courses (``location_type = "snowpack"``).

        The list comes from ``/locations``, which is authoritative, and is
        enriched from ``/snow-survey/metadata`` where a course appears
        there — that endpoint adds the first and last survey dates,
        per-target-date survey counts and the sub-basin used in the Yukon
        Snow Bulletins.  Not every course is present in it: composite
        records such as ``09DC-SC01`` (Mayo Airport, the unweighted average
        of 09DC-SC01A and 09DC-SC01B) are listed only in ``/locations``,
        so building from the metadata endpoint alone would silently drop
        them.  Courses missing metadata fall back to their first/last
        survey dates derived from ``/snow-survey/data``.

        Parameters
        ----------
        active_only : bool
            Keep only courses surveyed within the last
            :data:`_COURSE_ACTIVE_WINDOW_YEARS` years.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            Station dicts with ``station_type = "SC"``.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        if self._course_meta_cache is None:
            self._course_meta_cache = self._get_csv("snow-survey/metadata", {})

        meta_by_code = {
            (row.get("location_code") or "").strip(): row
            for row in self._course_meta_cache
            if (row.get("location_code") or "").strip()
        }

        stations: list[dict] = []
        for loc in self.get_locations(location_types=_SNOWPACK_TYPE):
            code = loc["location_code"]
            if not code:
                continue
            meta = meta_by_code.get(code, {})
            if not meta:
                logger.debug(
                    "Snow course %s absent from /snow-survey/metadata — "
                    "deriving survey dates from /snow-survey/data", code,
                )
            first_survey = (meta.get("first_survey") or "").strip()
            last_survey = (meta.get("last_survey") or "").strip()
            if not last_survey:
                first_survey, last_survey = self._course_survey_span(code)
            surveys = {
                period: int(_to_float(meta.get(field)) or 0)
                for period, field in (
                    ("01-Feb", "feb1_surveys"),
                    ("01-Mar", "march1_surveys"),
                    ("01-Apr", "april1_surveys"),
                    ("01-May", "may1_surveys"),
                    ("15-May", "may15_surveys"),
                )
            }
            stations.append({
                "station_id": code,
                "location_id": loc["location_id"],
                "location_code": code,
                # /locations carries the canonical name; metadata repeats it.
                "name": loc["name"] or (meta.get("location_name") or "").strip(),
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "elevation_m": loc["elevation_m"],
                "state": _station_state(
                    code, loc["name"], loc["latitude"], loc["longitude"]
                ),
                "station_type": "SC",
                "network": SNOW_SURVEY_NETWORK,
                "network_code": NETWORK_CODES["SC"],
                "operator": _OPERATORS["SC"],
                "status": _course_status(last_survey),
                "note": loc.get("note") or (meta.get("note") or "").strip(),
                "datum": loc.get("datum", ""),
                "sub_basin": (meta.get("sub_basin") or "").strip(),
                "first_survey": first_survey,
                "last_survey": last_survey,
                "survey_counts": surveys,
                "has_survey_metadata": bool(meta),
                "station_url": _EXPLORER_URL,
                "dataset_url": _DATASET_URLS["SC"],
                # Courses are measured manually for SWE and depth alike.
                "variables": list(SNOW_VARIABLES),
                "series": [],
            })

        if active_only:
            stations = [s for s in stations if s["status"] == "Active"]
        return _filter_by_bbox(
            sorted(stations, key=lambda s: s["station_id"]), bbox
        )

    def _course_survey_span(self, location_code: str) -> tuple[str, str]:
        """Return ``(first, last)`` survey dates for a course from its data."""
        dates = sorted(
            rec["date"]
            for rec in self.get_snow_survey_data(station_ids=location_code)
            if rec.get("date")
        )
        return (dates[0], dates[-1]) if dates else ("", "")

    def get_automated_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        networks: list[str] | str | None = None,
    ) -> list[dict]:
        """
        Return locations that carry a continuous snow series.

        A location qualifies when ``/timeseries`` lists a publicly visible
        SWE or snow-depth series for it.  Yukon Snow Survey sites are
        tagged ``station_type = "AWS"`` (automated snow-weather station,
        snow-pillow SWE) and ECCC climate stations ``"ECCC"``.

        Parameters
        ----------
        active_only : bool
            Keep only locations with at least one active snow series.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        networks : list[str] or str, optional
            Restrict to these AquaCache network names.  Defaults to the
            Yukon Snow Survey and ECCC Meteorology networks.

        Returns
        -------
        list[dict]
            Station dicts with a populated ``series`` list.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        wanted_nets = (
            set(_coerce_list(networks))
            if networks is not None
            else {SNOW_SURVEY_NETWORK, ECCC_NETWORK}
        )

        series_by_loc: dict[str, list[dict]] = {}
        for series in self.get_timeseries():
            series_by_loc.setdefault(str(series["location_id"]), []).append(series)

        locations = {
            str(loc["location_id"]): loc for loc in self.get_locations()
        }

        stations: list[dict] = []
        for loc_id, all_series in series_by_loc.items():
            snow_series = [
                s for s in all_series if s["variable"] in SNOW_VARIABLES
            ]
            if not snow_series:
                continue
            loc = locations.get(loc_id)
            if loc is None:
                logger.debug(
                    "Timeseries reference location_id %s absent from "
                    "/locations — skipping", loc_id,
                )
                continue
            loc_nets = set(loc.get("networks") or [])
            if wanted_nets and not (wanted_nets & loc_nets):
                continue

            stype = "ECCC" if ECCC_NETWORK in loc_nets else "AWS"
            network = ECCC_NETWORK if stype == "ECCC" else SNOW_SURVEY_NETWORK
            status = (
                "Active"
                if any(s.get("active") for s in snow_series)
                else "Inactive"
            )
            stations.append({
                "station_id": loc["location_code"],
                "location_id": loc["location_id"],
                "location_code": loc["location_code"],
                "name": loc["name"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "elevation_m": loc["elevation_m"],
                "state": _station_state(
                    loc["location_code"], loc["name"],
                    loc["latitude"], loc["longitude"],
                ),
                "station_type": stype,
                "network": network,
                "network_code": NETWORK_CODES[stype],
                "operator": _OPERATORS[stype],
                "status": status,
                "note": loc.get("note", ""),
                "alias": loc.get("alias", ""),
                "datum": loc.get("datum", ""),
                "networks": sorted(loc_nets),
                "station_url": _EXPLORER_URL,
                "dataset_url": _DATASET_URLS[stype],
                "variables": sorted({s["variable"] for s in all_series}),
                "series": sorted(all_series, key=lambda s: s["timeseries_id"]),
            })

        if active_only:
            stations = [s for s in stations if s["status"] == "Active"]
        return _filter_by_bbox(
            sorted(stations, key=lambda s: s["station_id"]), bbox
        )

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Return every snow station: courses, automated Yukon sites and ECCC.

        Parameters
        ----------
        active_only : bool
            Keep only stations reporting recently (see
            :meth:`get_snow_course_stations` and
            :meth:`get_automated_stations`).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            Combined station list, sorted by ``station_id``.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        courses = self.get_snow_course_stations(active_only=active_only, bbox=bbox)
        automated = self.get_automated_stations(active_only=active_only, bbox=bbox)
        return sorted(courses + automated, key=lambda s: s["station_id"])

    def get_metadata(self, station_id: str) -> dict:
        """
        Return full metadata for one station, including its series list.

        Parameters
        ----------
        station_id : str
            AquaCache ``location_code``, e.g. ``"09AA-M1"`` or
            ``"08AA-SC01"``.

        Returns
        -------
        dict
            The station dict, or ``{}`` when the code is unknown.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        code = str(station_id).strip()
        for sta in self.get_all_stations():
            if sta["station_id"] == code:
                return sta
        return {}

    def get_station_image_url(self, station_id: str) -> None:
        """
        Return the station photo URL — always ``None`` for this source.

        The AquaCache API exposes no station imagery, and the Water Data
        Explorer that would host it sits behind a Cloudflare JS challenge.
        The method exists so callers can treat every client alike.
        """
        logger.debug(
            "Yukon AquaCache exposes no station imagery (station %s)", station_id
        )
        return None

    # ── Snow survey (periodic) endpoints ──────────────────────────────────

    def get_snow_survey_data(
        self,
        station_ids: list[str] | str | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Return the manual snow course archive from ``GET /snow-survey/data``.

        The endpoint takes no query parameters and returns the whole
        archive (~22,000 rows, ~2 MB) in one response, so filtering happens
        client-side and the response is cached on the instance.

        Parameters
        ----------
        station_ids : list[str] or str, optional
            Restrict to these ``location_code`` values.
        begin_date, end_date : str or date, optional
            Restrict by ``sample_date`` (inclusive).
        include_flags : bool
            Add the source ``flag`` (``"Actual"`` / ``"Estimated SWE"``).

        Returns
        -------
        list[dict]
            One dict per measurement with keys ``station_id``, ``name``,
            ``date`` (true sample date), ``target_date``, ``survey_period``,
            ``year``, ``month``, ``variable``, ``type``, ``value``,
            ``units`` and ``interval`` (always ``"periodic"``), plus
            ``flag`` when requested.

        Raises
        ------
        YukonError
            On network / API failure.

        Examples
        --------
        >>> client = YukonClient()
        >>> rows = client.get_snow_survey_data(station_ids=["08AA-SC01"])
        >>> apr1 = [r for r in rows if r["survey_period"] == "01-Apr"]
        """
        if self._survey_cache is None:
            self._survey_cache = self._get_csv("snow-survey/data", {})

        wanted = set(_coerce_list(station_ids)) if station_ids is not None else None
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None

        records: list[dict] = []
        for row in self._survey_cache:
            code = (row.get("location_code") or "").strip()
            if not code or (wanted is not None and code not in wanted):
                continue

            var_key = _series_variable_key(row.get("parameter", ""), "")
            if var_key is None:
                continue
            var_info = VARIABLES[var_key]

            sample_date = (row.get("sample_date") or "").strip()[:10]
            if not sample_date:
                continue
            if begin and sample_date < begin:
                continue
            if end and sample_date > end:
                continue

            value = _normalize_value(row.get("result"))
            if var_key == "swe_mm" and value is not None:
                value = round(value / 10.0, 3)
            if var_info["type"] in {"swe", "snwd"}:
                value = _clamp_snow(value)

            rec: dict = {
                "station_id": code,
                "name": (row.get("location_name") or "").strip(),
                "date": sample_date,
                "target_date": (row.get("target_date") or "").strip()[:10],
                "survey_period": _survey_period(row.get("month")),
                "year": int(_to_float(row.get("year")) or 0) or None,
                "month": _to_float(row.get("month")),
                "variable": var_key,
                "type": var_info["type"],
                "value": value,
                # Never trust the response units field here: it is empty
                # for snow depth.
                "units": var_info["output_units"],
                "interval": "periodic",
            }
            if include_flags:
                rec["flag"] = (row.get("flag") or "").strip()
            records.append(rec)
        return records

    def get_snow_survey_stats(self) -> list[dict]:
        """
        Return per-course summary statistics from ``GET /snow-survey/stats``.

        Fields include ``total_record_yrs``, ``start``, ``end``,
        ``missing_yrs``, ``sample_months``, ``max_SWE_mm``,
        ``mean_max_SWE_mm``, ``median_max_SWE_mm`` and the matching
        ``*_DEPTH_cm`` columns.  Only complete years are considered.

        Returns
        -------
        list[dict]
            One dict per course, values left as source strings.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        return self._get_csv("snow-survey/stats", {})

    def get_snow_survey_trends(self) -> list[dict]:
        """
        Return per-course trend analysis from ``GET /snow-survey/trends``.

        Fields include Mann-Kendall ``p.value_SWE_max`` /
        ``p.value_DEPTH_max``, Sen's slope estimates, the number of years
        used, and estimated annual percent change.

        Returns
        -------
        list[dict]
            One dict per course, values left as source strings.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        return self._get_csv("snow-survey/trends", {})

    # ── Continuous measurement endpoints ──────────────────────────────────

    def get_daily_measurements(
        self,
        timeseries_ids: list[str] | str | int,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        stats: bool = False,
    ) -> list[dict]:
        """
        Fetch daily aggregates from ``GET /timeseries/measurementsDaily``.

        Parameters
        ----------
        timeseries_ids : list[str] or str or int
            AquaCache ``timeseries_id`` value(s).
        begin_date : str or date, optional
            Start date (inclusive).  Defaults to :data:`_EPOCH`.
        end_date : str or date, optional
            End date (inclusive).  Defaults to today.
        stats : bool
            Include the historical range statistics columns (percentiles,
            30-year normals, day-of-year counts).

        Returns
        -------
        list[dict]
            Raw rows: ``timeseries_id``, ``date``, ``day_timezone``,
            ``value``, ``imputed`` (plus statistics columns when
            ``stats=True``).

        Notes
        -----
        ``value`` is the mean over the **local** day given by
        ``day_timezone`` (UTC-07 for Yukon Snow Survey sites), not an
        instantaneous reading.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        params: dict[str, Any] = {
            "id": ",".join(_coerce_list(timeseries_ids)),
            "start": _date_str(begin_date) if begin_date else _EPOCH,
            "limit": _ROW_LIMIT,
        }
        if end_date:
            params["end"] = _date_str(end_date)
        if stats:
            params["stats"] = "true"
        return self._get_csv("timeseries/measurementsDaily", params)

    def get_measurements(
        self,
        timeseries_ids: list[str] | str | int,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> list[dict]:
        """
        Fetch instantaneous values from ``GET /timeseries/measurements``.

        Parameters
        ----------
        timeseries_ids : list[str] or str or int
            AquaCache ``timeseries_id`` value(s).
        begin_date : str or date, optional
            Start date/time (inclusive).  Defaults to :data:`_EPOCH`.
        end_date : str or date, optional
            End date/time (inclusive).  Defaults to now.

        Returns
        -------
        list[dict]
            Raw rows including ``datetime``, ``value_raw``,
            ``value_corrected``, ``grade_type_description``,
            ``approval_type_description`` and
            ``qualifier_type_descriptions``.

        Raises
        ------
        YukonError
            On network / API failure.
        """
        params: dict[str, Any] = {
            "id": ",".join(_coerce_list(timeseries_ids)),
            "start": (
                str(begin_date)[:16] if begin_date else f"{_EPOCH} 00:00"
            ),
            "limit": _ROW_LIMIT,
        }
        if end_date:
            params["end"] = str(end_date)[:16]
        return self._get_csv("timeseries/measurements", params)

    # ── Standardized data fetch ───────────────────────────────────────────

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
            AquaCache ``location_code`` value(s), e.g. ``"09AA-M1"`` for an
            automated snow-weather station or ``"08AA-SC01"`` for a snow
            course.  Required unless ``bbox`` is provided.
        variables : list[str] or str or None
            :data:`VARIABLES` key(s) (e.g. ``"swe_mm"``) **or** standardized
            types (e.g. ``"swe"``).  ``None`` returns all snow variables.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.  Alternative to
            ``station_ids``; fetches data for all snow stations in the box.
        begin_date : str or date, optional
            Start date (``"YYYY-MM-DD"``).
        end_date : str or date, optional
            End date (inclusive).
        interval : str
            ``"daily"`` (default) reads the daily aggregates;
            ``"hourly"`` / ``"sub_daily"`` reads instantaneous values;
            ``"periodic"`` reads the manual snow course archive.
        include_flags : bool
            If True, add a ``"flag"`` key to each record.

        Returns
        -------
        list[dict]
            Flat list of observation records::

                {
                    "station_id": "09AA-M1",
                    "date": "2024-01-15",
                    "variable": "swe_mm",
                    "type": "swe",
                    "value": 12.5,     # cm (converted from mm ÷ 10)
                    "units": "cm",
                    "interval": "daily",
                    "aggregation": "instantaneous",
                    "timeseries_id": "20",
                    # "flag": "" (only present when include_flags=True)
                }

            ``aggregation`` and ``timeseries_id`` are extra keys that
            disambiguate locations holding several series of the same
            parameter (ECCC daily air temperature exists as minimum,
            maximum and mean).  Sub-daily records additionally carry
            ``datetime``.

        Notes
        -----
        SWE is stored by AquaCache in mm and is converted to cm, so
        ``"units"`` is always ``"cm"`` for type ``"swe"``.  Snow depth is
        natively cm and is returned as-is.  Other met variables keep their
        native units, as in the DataBC client.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        YukonError
            On network / API failure.

        Examples
        --------
        >>> client = YukonClient()
        >>> records = client.get_data(
        ...     station_ids="09AA-M1",
        ...     variables=["swe"],
        ...     begin_date="2024-01-01",
        ...     end_date="2024-01-15",
        ... )
        >>> sorted(records[0])[:4]
        ['aggregation', 'date', 'interval', 'station_id']
        """
        # ── Resolve station codes ──────────────────────────────────────────
        if station_ids is None and bbox is not None:
            codes = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        elif station_ids is not None:
            codes = _coerce_list(station_ids)
        else:
            raise ValueError("Provide station_ids or bbox.")

        if not codes:
            return []

        var_keys = _resolve_variables(variables)
        if not var_keys:
            return []

        interval_key = str(interval or "daily").lower()
        if interval_key not in ("daily", "hourly", "sub_daily",
                                "instantaneous", "periodic"):
            # No silent fallback (DESIGN.md §3.6): anything unrecognised
            # used to be routed to the instantaneous endpoint.
            raise YukonError(
                f"Unsupported interval {interval!r} for Yukon — expected "
                "'daily', 'hourly', 'sub_daily', 'instantaneous', or "
                "'periodic'."
            )

        return self._get_data_yukon(
            codes=codes,
            var_keys=var_keys,
            begin_date=begin_date,
            end_date=end_date,
            interval=interval_key,
            include_flags=include_flags,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_data_yukon(
        self,
        codes: list[str],
        var_keys: list[str],
        begin_date: str | date | None,
        end_date: str | date | None,
        interval: str,
        include_flags: bool,
    ) -> list[dict]:
        """Native data fetch behind :meth:`get_data`."""
        if interval == "periodic":
            return self.get_snow_survey_data(
                station_ids=codes,
                begin_date=begin_date,
                end_date=end_date,
                include_flags=include_flags,
            )

        # Map requested location codes to their series.  ``series`` rows
        # carry only the numeric location_id, so keep a reverse index.
        stations = self.get_all_stations()
        code_by_loc_id = {
            str(sta.get("location_id")): sta["station_id"] for sta in stations
        }

        wanted_codes = set(codes)
        series: list[dict] = []
        for sta in stations:
            if sta["station_id"] not in wanted_codes:
                continue
            for ser in sta.get("series") or []:
                if ser["variable"] in var_keys:
                    series.append(ser)

        if not series:
            logger.info(
                "No continuous series for %d requested station(s) and "
                "variables %s — snow courses hold periodic data only "
                "(interval='periodic')",
                len(codes), var_keys,
            )
            return []

        want_daily = interval == "daily"
        series_by_id = {str(s["timeseries_id"]): s for s in series}
        ids = sorted(series_by_id, key=int)

        records: list[dict] = []
        for batch in _chunked(ids, _DAILY_ID_BATCH):
            if want_daily:
                rows = self.get_daily_measurements(
                    batch, begin_date=begin_date, end_date=end_date
                )
            else:
                rows = self.get_measurements(
                    batch, begin_date=begin_date, end_date=end_date
                )
            if len(rows) >= _ROW_LIMIT:
                # The endpoint silently truncates at ``limit``; splitting
                # the batch keeps every series whole rather than reporting
                # a quietly short archive.
                logger.warning(
                    "Row limit %d reached for timeseries %s — refetching "
                    "each series individually",
                    _ROW_LIMIT, batch,
                )
                rows = []
                for one in batch:
                    rows.extend(
                        self.get_daily_measurements(
                            one, begin_date=begin_date, end_date=end_date
                        )
                        if want_daily
                        else self.get_measurements(
                            one, begin_date=begin_date, end_date=end_date
                        )
                    )

            for row in rows:
                ts_id = str(row.get("timeseries_id") or "").strip()
                ser = series_by_id.get(ts_id)
                if ser is None:
                    continue
                code = code_by_loc_id.get(str(ser["location_id"]))
                if code is None:
                    continue
                rec = self._build_record(
                    row, ser, code, want_daily, include_flags
                )
                if rec is not None:
                    records.append(rec)
        return records

    def _build_record(
        self,
        row: dict,
        series: dict,
        code: str,
        want_daily: bool,
        include_flags: bool,
    ) -> dict | None:
        """Turn one measurement row into a standardized record."""
        var_key = series["variable"]
        var_info = VARIABLES[var_key]

        if want_daily:
            timestamp = (row.get("date") or "").strip()
            raw_value = row.get("value")
        else:
            timestamp = (row.get("datetime") or "").strip()
            raw_value = row.get("value_corrected")
            if _to_float(raw_value) is None:
                raw_value = row.get("value_raw")

        date_str = timestamp[:10]
        if not date_str:
            return None

        value = _normalize_value(raw_value)
        if var_key == "swe_mm" and value is not None:
            value = round(value / 10.0, 3)
        if var_info["type"] in {"swe", "snwd"}:
            value = _clamp_snow(value)

        rec: dict = {
            "station_id": code,
            "date": date_str,
            "variable": var_key,
            "type": var_info["type"],
            "value": value,
            "units": var_info["output_units"],
            "interval": "daily" if want_daily else series["interval"],
            "aggregation": series["aggregation"],
            "timeseries_id": str(series["timeseries_id"]),
        }
        if not want_daily:
            rec["datetime"] = timestamp
        if include_flags:
            rec["flag"] = self._row_flag(row, want_daily)
        return rec

    @staticmethod
    def _row_flag(row: dict, want_daily: bool) -> str:
        """Build the ``flag`` string for a measurement row."""
        if want_daily:
            return "imputed" if _to_bool(row.get("imputed")) else ""
        parts: list[str] = []
        grade = (row.get("grade_type_description") or "").strip()
        approval = (row.get("approval_type_description") or "").strip()
        qualifiers = (row.get("qualifier_type_descriptions") or "").strip()
        if grade:
            parts.append(f"grade:{grade}")
        if approval:
            parts.append(f"approval:{approval}")
        for qual in _parse_pg_array(qualifiers) or (
            [qualifiers] if qualifiers else []
        ):
            parts.append(f"qualifier:{qual}")
        return "; ".join(parts)

    def _parse_location(self, row: dict) -> dict:
        """Normalize one ``/locations`` row."""
        return {
            "location_id": (row.get("location_id") or "").strip(),
            "location_code": (row.get("location_code") or "").strip(),
            "name": (row.get("name") or "").strip(),
            "alias": (row.get("alias") or "").strip(),
            "location_type": (row.get("location_type") or "").strip(),
            "latitude": _to_float(row.get("latitude")),
            "longitude": _to_float(row.get("longitude")),
            # Real low elevations exist here (Herschel Island 1.2 m,
            # Komakuk Beach 13.2 m, Fishing Branch River 0 m) — they are
            # not sentinels and must not be nulled.
            "elevation_m": _to_float(row.get("elevation")),
            "datum": (row.get("datum") or "").strip(),
            "note": (row.get("note") or "").strip(),
            "networks": _parse_pg_array(row.get("networks")),
            "projects": _parse_pg_array(row.get("projects")),
        }

    def _parse_series(self, row: dict) -> dict | None:
        """
        Normalize one ``/timeseries`` row, or return ``None`` when its
        parameter is not exposed by this client.
        """
        parameter = (row.get("parameter_name") or "").strip()
        aggregation = (row.get("aggregation_type") or "").strip()
        var_key = _series_variable_key(parameter, aggregation)
        if var_key is None:
            return None
        var_info = VARIABLES[var_key]
        return {
            "timeseries_id": (row.get("timeseries_id") or "").strip(),
            "location_id": (row.get("location_id") or "").strip(),
            "location_name": (row.get("location_name") or "").strip(),
            "variable": var_key,
            "type": var_info["type"],
            "parameter_name": parameter,
            "aggregation": aggregation,
            "recording_rate": (row.get("recording_rate") or "").strip(),
            "interval": _series_interval(row.get("recording_rate", "")),
            # The response units field is unreliable (empty for
            # 'precipitation, snow'), so both come from VARIABLES.
            "units": var_info["units"],
            "output_units": var_info["output_units"],
            "start_datetime": (row.get("start_datetime") or "").strip(),
            "end_datetime": (row.get("end_datetime") or "").strip(),
            "active": _to_bool(row.get("active")),
            "publicly_visible": _to_bool(row.get("publicly_visible")),
            "owner": (row.get("default_owner") or "").strip(),
            "networks": _parse_pg_array(row.get("networks")),
        }

    def _get_csv(self, endpoint: str, params: dict[str, Any]) -> list[dict]:
        """
        GET an endpoint and parse its CSV body into a list of dicts.

        Parameters
        ----------
        endpoint : str
            API path without a leading slash, e.g. ``"locations"``.
        params : dict
            Query parameters.

        Returns
        -------
        list[dict]

        Raises
        ------
        YukonError
            On non-retryable HTTP errors or after all retries are exhausted.
        """
        return _parse_csv(self._get(endpoint, params))

    def _get(self, endpoint: str, params: dict[str, Any]) -> str:
        """
        Make a GET request to the given endpoint with retry logic.

        Parameters
        ----------
        endpoint : str
            API path without a leading slash, e.g. ``"locations"``.
        params : dict
            Query parameters.

        Returns
        -------
        str
            Response body text.

        Raises
        ------
        YukonError
            On non-retryable HTTP errors or after all retries are exhausted.
        """
        url = f"{self.base_url}/{endpoint}"
        response = request_with_retries(
            self._session, url, params=params, error_cls=YukonError,
            timeout=self.timeout, max_retries=self.max_retries,
            backoff=self.backoff,
        )
        return response.text


# ── Exception ────────────────────────────────────────────────────────────────

class YukonError(Exception):
    """Raised when the Yukon AquaCache API returns an error or a request fails."""


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """
    Translate a variable list (native keys or standardized types) to
    :data:`VARIABLES` keys.

    Parameters
    ----------
    variables : list[str] or str or None
        Variable key(s) (e.g. ``"swe_mm"``) or standardized type(s)
        (e.g. ``"swe"``).  ``None`` returns all snow variables.

    Returns
    -------
    list[str]
        Ordered, de-duplicated variable keys.  Falls back to
        :data:`SNOW_VARIABLES` when nothing resolves.
    """
    if variables is None:
        return list(SNOW_VARIABLES)

    raw_vars = [variables] if isinstance(variables, str) else list(variables)
    if not raw_vars:
        return list(SNOW_VARIABLES)
    keys: list[str] = []
    seen: set[str] = set()
    for var in raw_vars:
        if var in VARIABLES:
            if var not in seen:
                keys.append(var)
                seen.add(var)
        elif var in _TYPE_TO_YUKON_VARS:
            for key in _TYPE_TO_YUKON_VARS[var]:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        else:
            # No silent fallback (DESIGN.md §3.6)
            raise YukonError(
                f"Unknown variable {var!r} for Yukon — expected one of "
                f"{sorted(VARIABLES)} or a standardized type "
                f"({sorted(_TYPE_TO_YUKON_VARS)})."
            )
    return keys
