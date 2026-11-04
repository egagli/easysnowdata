"""
clients/fmi/fmi_client.py
=========================
Python client for the Finnish Meteorological Institute (FMI, Ilmatieteen
laitos) open-data WFS download service — daily snow depth for Finland,
with the other daily and hourly weather parameters the service serves.

Primary use case: daily snow depth (parameter ``snow``) at FMI's automatic
weather stations and its manual precipitation and climate stations, from the
1930s at the long-record stations.  FMI publishes no snow water equivalent
through this service; Finnish snow-course SWE comes from Syke
(:mod:`easysnowdata.stations.clients.syke`), and Sodankylä's research SWE
record from FMI's separate LITDB portal (https://litdb.fmi.fi).

Data source       : https://opendata.fmi.fi/wfs (WFS 2.0 stored queries)
Documentation     : https://en.ilmatieteenlaitos.fi/open-data-manual-fmi-wfs-services
Parameter metadata: https://opendata.fmi.fi/meta?observableProperty=observation&param=<name>&language=eng
Station elevations: https://en.ilmatieteenlaitos.fi/observation-stations (the
                    public station table; the WFS station list has none)
Authentication    : none (no API key since 2017)
Licence           : Creative Commons Attribution 4.0 International (CC BY 4.0) —
                    https://en.ilmatieteenlaitos.fi/open-data-licence.
                    Attribute "Finnish Meteorological Institute".
Rate limits       : 20 000 download requests per day and 600 requests per
                    5 minutes, download and view services combined (the
                    open-data manual).  The client spaces its requests
                    ``min_interval`` seconds apart (0.5 s = 600 per 5 min).

Stored queries used
-------------------
- ``fmi::ef::stations`` — station metadata (INSPIRE EF): ``fmisid``, name,
  WMO code, region, position (EPSG:4258, ETRS89 — within a metre of WGS84
  in Finland), one ``OperationalActivityPeriod`` and one ``belongsTo``
  network per network membership.  Without a time range it lists only
  stations active now (437 in November 2026); with
  ``starttime=1900-01-01`` it lists closed stations too (2 715).
- ``fmi::observations::weather::daily::timevaluepair`` — daily values
  (``interval="daily"``).  At most 8 928 h (372 days) per request.
- ``fmi::observations::weather::timevaluepair`` — instantaneous values
  every 10 minutes, or every hour with ``timestep=60``
  (``interval="sub_hourly"`` / ``"hourly"``).  At most 168 h (7 days) per
  request.

Both observation queries answer HTTP 400 "Too long time interval
requested!" above their limit and HTTP 400 "Unknown 'fmisid' value!" when
any one of the requested stations is not in the observation database — the
whole request fails, not just that station.  At most 99 ``fmisid``
parameters are accepted per request ("specified more than 99 times").
Values come as WaterML 2.0 time-value pairs, one series per station and
parameter; a missing value is ``NaN``.

Snow depth and the ``-1`` code
------------------------------
Daily ``snow`` and sub-daily ``snow_aws`` are depths in cm.  Both use
``-1`` on snow-free ground: every one of 1 792 daily values in Finland on
2024-07-01..10 was ``-1`` (no other value occurred), and in November the
``-1`` days sit between positive depths.  ``0`` is a separate, valid value.
The client returns ``-1`` as ``0.0`` cm and, with ``include_flags=True``,
flags it ``"snow_free"``.  ``rrday`` uses the same ``-1`` for a day without
precipitation (691 of 1 198 values at 60 stations in June 2024), returned as
``0.0`` mm and flagged ``"no_precipitation"``.  FMI's open-data pages do
not state either convention; it is read off the data (2026-11-04).

What a daily ``snow`` value is
------------------------------
At automatic weather stations the daily ``snow`` value dated D equals the
``snow_aws`` reading at 06:00 UTC on D (checked against the hourly series at
Sodankylä Tähtelä and Helsinki Kaisaniemi, 15 days in January–April 2024:
every day matched the 06 UTC reading, and none matched the day's mean).
Parameter metadata calls it an instant value aggregated over ``P1D``.  At
manual stations it is the observer's reading; its clock time is not stated
by the service.  Dates are UTC days (``tz`` is not passed).

Stations
--------
:meth:`FMIClient.get_all_stations` lists the stations of the networks that
report snow depth (:data:`SNOW_NETWORKS`: automatic weather stations,
manned weather stations, climate stations and precipitation stations), open
and closed.  ``station_id`` is the ``fmisid`` as a string.  Elevations come
from FMI's public station table (``z``, metres); a station missing from it
has ``elevation_m = None``.  Some closed stations are listed by
``fmi::ef::stations`` but unknown to the observation service (stations
closed before about 2011, though not all of them);
:meth:`FMIClient.screen_station_ids` finds them, and :meth:`get_data`
skips them with a log message rather than failing the whole request.

Design principles
-----------------
- Metric-first: depths in cm, temperatures in °C, precipitation in mm, wind
  speeds converted from m/s to km/h (× 3.6), pressure in hPa.
- Snow depth outside 0–15 m is nulled.
- ``variables=None`` means snow depth only.  Hourly variables are not
  available as daily and vice versa; asking for one at the wrong interval
  raises :class:`FMIError`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any
from xml.etree import ElementTree as ET

import requests

from .._common import (
    chunk as _chunk,
)
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
from .._common import (
    to_float as _to_float,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://opendata.fmi.fi/wfs"
STATIONS_PAGE_URL = "https://en.ilmatieteenlaitos.fi/observation-stations"

STATIONS_QUERY = "fmi::ef::stations"
DAILY_QUERY = "fmi::observations::weather::daily::timevaluepair"
INSTANT_QUERY = "fmi::observations::weather::timevaluepair"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4
#: Seconds between requests: 600 per 5 minutes is FMI's combined limit.
_DEFAULT_MIN_INTERVAL = 0.5

#: ``fmisid`` parameters accepted per request (the service rejects 100).
_MAX_IDS = 99

#: Request windows.  The daily query allows 8 928 h; a window of 365 days
#: from 00:00 to 00:00 (8 760 h) covers 366 dates.  The instantaneous query
#: allows exactly 168 h.
_DAILY_WINDOW = timedelta(days=365)
_INSTANT_WINDOW = timedelta(hours=168)

#: Earliest date asked for when no begin date is given and a station's
#: period of record is unknown.
_EARLIEST = date(1900, 1, 1)

#: Plausibility bound for snow depth, cm (DESIGN.md: 0–15 m).
_MAX_SNWD_CM = 1500.0

#: Networks (``networkid``) whose stations report snow depth, → (Finnish
#: name as the service gives it, English, short code used as
#: ``network_code``).
SNOW_NETWORKS: dict[int, tuple[str, str, str]] = {
    121: ("Automaattinen sääasema", "automatic weather station", "AWS"),
    122: ("Sääasema", "manned weather station", "WS"),
    123: ("Ilmastoasema", "climate station", "CLIM"),
    124: ("Sadeasema", "precipitation station", "PRCP"),
}
#: Priority when a station belongs to several snow networks.
_NETWORK_PRIORITY = (121, 123, 122, 124)

_SOURCE_DAILY = f"{BASE_URL} — stored query {DAILY_QUERY}"
_SOURCE_INSTANT = f"{BASE_URL} — stored query {INSTANT_QUERY}"

_NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "ef": "http://inspire.ec.europa.eu/schemas/ef/4.0",
    "wml2": "http://www.opengis.net/waterml/2.0",
    "xlink": "http://www.w3.org/1999/xlink",
}
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_XLINK_TITLE = "{http://www.w3.org/1999/xlink}title"

# ── Public variable / flag tables ────────────────────────────────────────────

#: Variables, keyed by the FMI parameter name.  ``interval`` lists the
#: intervals the parameter is served at (daily parameters come from the
#: daily stored query, the others from the instantaneous one); ``scale``
#: converts the native value to ``output_units``; ``no_value_code`` is the
#: native code meaning "none" (snow-free ground, no precipitation).
VARIABLES: dict[str, dict] = {
    "snow": {
        "name": "Snow Depth (daily)",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "interval": ("daily",),
        "no_value_code": -1.0,
        "source": _SOURCE_DAILY,
        "description": "Daily snow depth (instant value, P1D).",
        "notes": (
            "At automatic stations the 06:00 UTC reading of the dated day "
            "(checked against snow_aws, 2024); at manual stations the "
            "observer's reading, time not stated. -1 (snow-free ground) is "
            "returned as 0.0, flag 'snow_free'."
        ),
    },
    "tday": {
        "name": "Air Temperature (daily mean)",
        "type": "temp",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "interval": ("daily",),
        "source": _SOURCE_DAILY,
        "description": "Daily mean air temperature (avg, P1D).",
        "notes": "",
    },
    "tmin": {
        "name": "Minimum Air Temperature",
        "type": "temp_min",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "interval": ("daily",),
        "source": _SOURCE_DAILY,
        "description": "Daily minimum air temperature (min, PT24H).",
        "notes": "",
    },
    "tmax": {
        "name": "Maximum Air Temperature",
        "type": "temp_max",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "interval": ("daily",),
        "source": _SOURCE_DAILY,
        "description": "Daily maximum air temperature (max, PT24H).",
        "notes": "",
    },
    "rrday": {
        "name": "Precipitation (daily)",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "interval": ("daily",),
        "no_value_code": -1.0,
        "source": _SOURCE_DAILY,
        "description": "Daily precipitation amount (sum, PT24H).",
        "notes": (
            "-1 (no precipitation) is returned as 0.0, flag "
            "'no_precipitation'. The 24 h period's boundaries are not "
            "stated by the service."
        ),
    },
    "snow_aws": {
        "name": "Snow Depth (automatic)",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "interval": ("hourly", "sub_hourly"),
        "no_value_code": -1.0,
        "source": _SOURCE_INSTANT,
        "description": "Snow depth from the automatic sensor (instant).",
        "notes": (
            "-1 (snow-free ground) is returned as 0.0, flag 'snow_free'. "
            "Not accepted by the daily query (HTTP 400 'Unrecognized "
            "parameter'); use 'snow' for daily values."
        ),
    },
    "t2m": {
        "name": "Air Temperature",
        "type": "temp",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Air temperature at 2 m (avg, PT1M).",
        "notes": "",
    },
    "rh": {
        "name": "Relative Humidity",
        "type": "rh",
        "units": "%",
        "output_units": "%",
        "scale": 1.0,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Relative humidity (avg, PT1M).",
        "notes": "",
    },
    "ws_10min": {
        "name": "Wind Speed",
        "type": "wind_spd",
        "units": "m/s",
        "output_units": "km/h",
        "scale": 3.6,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Wind speed, 10-minute mean.",
        "notes": "Served in m/s; returned in km/h (× 3.6).",
    },
    "wg_10min": {
        "name": "Wind Gust",
        "type": "wind_gust",
        "units": "m/s",
        "output_units": "km/h",
        "scale": 3.6,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Gust speed, 10-minute maximum.",
        "notes": "Served in m/s; returned in km/h (× 3.6).",
    },
    "wd_10min": {
        "name": "Wind Direction",
        "type": "wind_dir",
        "units": "degrees",
        "output_units": "degrees",
        "scale": 1.0,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Wind direction, 10-minute mean.",
        "notes": "",
    },
    "r_1h": {
        "name": "Precipitation (1 h)",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "interval": ("hourly",),
        "source": _SOURCE_INSTANT,
        "description": "Precipitation amount accumulated over the past hour.",
        "notes": "Hourly only: a 1 h accumulation sampled every 10 min would repeat.",
    },
    "p_sea": {
        "name": "Sea-Level Pressure",
        "type": "baro",
        "units": "hPa",
        "output_units": "hPa",
        "scale": 1.0,
        "interval": ("hourly", "sub_hourly"),
        "source": _SOURCE_INSTANT,
        "description": "Air pressure reduced to mean sea level (avg, PT1M).",
        "notes": "",
    },
}

#: Flags this client sets (the service publishes no quality flags in the
#: time-value-pair format).  ``None`` = a plain value.
DATA_FLAGS: dict[str, str] = {
    "snow_free": "Source value -1: snow-free ground, returned as 0.0 cm.",
    "no_precipitation": "Source value -1: no precipitation, returned as 0.0 mm.",
}

_FLAG_FOR_TYPE = {"snwd": "snow_free", "precip": "no_precipitation"}

_INTERVALS = ("daily", "hourly", "sub_hourly")
#: ``timestep`` (minutes) sent to the instantaneous query.
_TIMESTEP = {"hourly": 60, "sub_hourly": 10}

_TYPE_TO_VARS: dict[str, list[str]] = {}
for _code, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_code)

#: Defaults when ``variables`` is None (DESIGN.md §3.4: snow only).
_DEFAULT_VARS: dict[str, list[str]] = {
    "daily": ["snow"],
    "hourly": ["snow_aws"],
    "sub_hourly": ["snow_aws"],
}


class FMIError(Exception):
    """Raised for FMI open-data failures and bad requests."""


class _UnknownStationError(FMIError):
    """The observation service does not know one of the requested fmisids."""


# ── Parsers ──────────────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None, interval: str) -> list[str]:
    """Map native names or standardized types to native names at *interval*."""
    items = _coerce_list(variables) if variables is not None else []
    if not items:
        return list(_DEFAULT_VARS[interval])
    out: list[str] = []
    for v in items:
        key = v.lower()
        if key in VARIABLES:
            if interval not in VARIABLES[key]["interval"]:
                raise FMIError(
                    f"FMI variable {v!r} is served at "
                    f"{list(VARIABLES[key]['interval'])}, not {interval!r}."
                )
            codes = [key]
        elif key in _TYPE_TO_VARS and key != "other":
            codes = [
                c for c in _TYPE_TO_VARS[key] if interval in VARIABLES[c]["interval"]
            ]
            if not codes:
                raise FMIError(
                    f"FMI serves no {key!r} variable at interval {interval!r}."
                )
        else:
            raise FMIError(
                f"Unknown FMI variable {v!r} — expected one of "
                f"{sorted(VARIABLES)} or types {sorted(_TYPE_TO_VARS)}. "
                "FMI publishes no snow water equivalent here (see SYKEClient)."
            )
        for c in codes:
            if c not in out:
                out.append(c)
    return out


def _resolve_interval(interval: str) -> str:
    iv = str(interval).lower()
    if iv not in _INTERVALS:
        raise FMIError(
            f"Unsupported interval {interval!r} for FMI — expected one of {list(_INTERVALS)}."
        )
    return iv


def _convert(code: str, raw: Any) -> tuple[float | None, str | None]:
    """Native value text → (emitted value, flag)."""
    info = VARIABLES[code]
    v = _to_float(raw)
    if v is None:
        return None, None
    nv = info.get("no_value_code")
    if nv is not None and v == nv:
        return 0.0, _FLAG_FOR_TYPE.get(info["type"])
    out = round(v * info["scale"], 3)
    if info["type"] == "snwd" and not (0.0 <= out <= _MAX_SNWD_CM):
        return None, None
    return out, None


def _iso(d: date | datetime) -> str:
    if isinstance(d, datetime):
        return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{d.isoformat()}T00:00:00Z"


def _windows(begin: date, end: date, interval: str) -> list[tuple[str, str]]:
    """``(starttime, endtime)`` pairs covering *begin*..*end* inclusive."""
    out: list[tuple[str, str]] = []
    if interval == "daily":
        cur = begin
        while cur <= end:
            stop = min(cur + _DAILY_WINDOW, end)
            out.append((_iso(cur), _iso(stop)))
            cur = stop + timedelta(days=1)
        return out
    cur_dt = datetime(begin.year, begin.month, begin.day, tzinfo=UTC)
    last = datetime(end.year, end.month, end.day, 23, 59, tzinfo=UTC)
    step = timedelta(minutes=_TIMESTEP[interval])
    while cur_dt <= last:
        stop_dt = min(cur_dt + _INSTANT_WINDOW, last)
        out.append((_iso(cur_dt), _iso(stop_dt)))
        cur_dt = stop_dt + step
    return out


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_stations(content: bytes) -> list[dict]:
    """``fmi::ef::stations`` response → raw station dicts (all networks)."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise FMIError(f"Unreadable station list: {exc}") from exc
    out: list[dict] = []
    for fac in root.iter(f"{{{_NS['ef']}}}EnvironmentalMonitoringFacility"):
        names: dict[str, str] = {}
        fmisid = None
        for el in fac:
            tag = _local(el.tag)
            space = el.get("codeSpace", "")
            if tag == "identifier" and space.endswith("/fmisid"):
                fmisid = (el.text or "").strip()
            elif tag == "name" and space:
                names[space.rsplit("/", 1)[-1]] = (el.text or "").strip()
        if not fmisid:
            continue
        pos = fac.find(".//gml:pos", _NS)
        lat = lon = None
        if pos is not None and pos.text:
            parts = pos.text.split()
            if len(parts) >= 2:
                lat, lon = _to_float(parts[0]), _to_float(parts[1])
        periods: list[tuple[str | None, str | None]] = []
        for per in fac.iter(f"{{{_NS['gml']}}}TimePeriod"):
            b = per.find("gml:beginPosition", _NS)
            e = per.find("gml:endPosition", _NS)
            begin = (b.text or "").strip()[:10] if b is not None and b.text else None
            if e is not None and e.get("indeterminatePosition") == "now":
                end = "now"
            else:
                end = (e.text or "").strip()[:10] if e is not None and e.text else None
            periods.append((begin, end))
        networks: list[tuple[int, str]] = []
        for bt in fac.iter(f"{{{_NS['ef']}}}belongsTo"):
            m = re.search(r"networkid=(\d+)", bt.get(_XLINK_HREF, ""))
            if m:
                nid = int(m.group(1))
                if all(nid != n for n, _ in networks):
                    networks.append((nid, bt.get(_XLINK_TITLE, "")))
        out.append(
            {
                "fmisid": fmisid,
                "name": names.get("name")
                or (fac.findtext("ef:name", "", _NS) or "").strip(),
                "wmo": names.get("wmo") or None,
                "region": names.get("region") or None,
                "country": names.get("country") or None,
                "latitude": lat,
                "longitude": lon,
                "periods": periods,
                "networks": networks,
            }
        )
    return out


def _station_from_raw(raw: dict, elevations: dict[str, dict]) -> dict:
    """Raw station + elevation-table row → the client's station dict."""
    nids = [n for n, _ in raw["networks"]]
    snow_nets = [n for n in _NETWORK_PRIORITY if n in nids]
    active = any(end == "now" for _, end in raw["periods"])
    begins = [b for b, _ in raw["periods"] if b]
    ends = [e for _, e in raw["periods"] if e and e != "now"]
    extra = elevations.get(raw["fmisid"], {})
    return {
        "station_id": raw["fmisid"],
        "name": raw["name"],
        "latitude": raw["latitude"],
        "longitude": raw["longitude"],
        "elevation_m": extra.get("elevation_m"),
        "status": "Active" if active else "Inactive",
        "begin_date": min(begins) if begins else None,
        "end_date": None if active else (max(ends) if ends else None),
        "wmo_id": raw["wmo"],
        "region": raw["region"],
        "network_code": SNOW_NETWORKS[snow_nets[0]][2] if snow_nets else None,
        "network_ids": nids,
        "networks": [title for _, title in raw["networks"]],
        "station_groups": extra.get("groups"),
        "operator": "Finnish Meteorological Institute",
        "station_url": None,
    }


def _devalue(payload: list, idx: Any) -> Any:
    """Read one scalar from a Nuxt ``devalue`` payload (negative = special)."""
    if not isinstance(idx, int) or idx < 0 or idx >= len(payload):
        return None
    v = payload[idx]
    return v if isinstance(v, (str, int, float)) or v is None else None


def _parse_station_table(html: str) -> dict[str, dict]:
    """FMI's public station table → ``{fmisid: {elevation_m, groups}}``.

    The page is a Nuxt app; its ``__NUXT_DATA__`` script is a ``devalue``
    array in which each station is an object whose values are indices into
    the array (``{"fmisid": 11805, "z": 7215, …}``).  ``z`` is the station
    elevation in metres (Sodankylä Tähtelä: 179).
    """
    m = re.search(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise FMIError("FMI station table: no __NUXT_DATA__ payload found")
    try:
        payload = json.loads(m.group(1))
    except ValueError as exc:
        raise FMIError(f"FMI station table: unreadable payload: {exc}") from exc
    out: dict[str, dict] = {}
    for obj in payload:
        if not (isinstance(obj, dict) and "fmisid" in obj):
            continue
        fmisid = _devalue(payload, obj["fmisid"])
        if fmisid is None:
            continue
        z = _devalue(payload, obj.get("z"))
        out[str(fmisid)] = {
            "elevation_m": float(z) if isinstance(z, (int, float)) else None,
            "groups": _devalue(payload, obj.get("groups")),
        }
    if not out:
        raise FMIError("FMI station table: no stations in the payload")
    return out


def _parse_series(content: bytes) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Observation response → ``[(fmisid, parameter, [(time, value), …]), …]``."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise FMIError(f"Unreadable observation response: {exc}") from exc
    out: list[tuple[str, str, list[tuple[str, str]]]] = []
    for member in root.iter(f"{{{_NS['wfs']}}}member"):
        param = None
        fmisid = None
        for el in member.iter():
            tag = _local(el.tag)
            if param is None and tag == "observedProperty":
                m = re.search(r"[?&]param=([^&]+)", el.get(_XLINK_HREF, ""))
                param = m.group(1).lower() if m else None
            elif (
                fmisid is None
                and tag == "identifier"
                and el.get("codeSpace", "").endswith("/fmisid")
            ):
                fmisid = (el.text or "").strip()
        if not param or not fmisid:
            continue
        pairs: list[tuple[str, str]] = []
        for tvp in member.iter(f"{{{_NS['wml2']}}}MeasurementTVP"):
            t = tvp.findtext("wml2:time", None, _NS)
            v = tvp.findtext("wml2:value", None, _NS)
            if t:
                pairs.append((t.strip(), (v or "").strip()))
        out.append((fmisid, param, pairs))
    return out


# ── Client ───────────────────────────────────────────────────────────────────


class FMIClient:
    """
    Client for FMI's open-data WFS (daily snow depth and weather).

    No credentials are needed.

    Parameters
    ----------
    base_url : str
        WFS endpoint.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    min_interval : float
        Minimum seconds between requests (FMI allows 600 per 5 minutes).
    session : requests.Session or None
        Optional pre-configured session.
    allow_partial : bool
        If False (default), a failure to read FMI's public station table
        (the only source of elevations) raises :class:`FMIError`; if True
        it is logged and stations get ``elevation_m = None``.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        min_interval: float = _DEFAULT_MIN_INTERVAL,
        session: requests.Session | None = None,
        allow_partial: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.allow_partial = allow_partial
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._last_request = 0.0
        self._stations_cache: list[dict] | None = None
        self._unknown_ids: set[str] = set()
        self._known_ids: set[str] = set()

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _wfs(self, query: str, params: list[tuple[str, Any]]) -> bytes:
        """GET one stored query; ``params`` may repeat keys (``fmisid``)."""
        base = [
            ("service", "WFS"),
            ("version", "2.0.0"),
            ("request", "getFeature"),
            ("storedquery_id", query),
        ]
        self._throttle()
        try:
            resp = request_with_retries(
                self._session,
                self.base_url,
                params=base + params,
                error_cls=FMIError,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff=self.backoff,
            )
        except FMIError as exc:
            # The 400 body names the cause; its ExceptionText sits ~360
            # characters in, inside the 500 the retry loop keeps.
            text = str(exc)
            if "Unknown &apos;fmisid&apos;" in text or "Unknown 'fmisid'" in text:
                raise _UnknownStationError(text) from exc
            raise
        return resp.content

    # ── Stations ─────────────────────────────────────────────────────────────

    def _elevations(self) -> dict[str, dict]:
        self._throttle()
        try:
            resp = request_with_retries(
                self._session,
                STATIONS_PAGE_URL,
                error_cls=FMIError,
                timeout=self.timeout,
                max_retries=self.max_retries,
                backoff=self.backoff,
            )
            return _parse_station_table(resp.text)
        except FMIError as exc:
            if not self.allow_partial:
                raise FMIError(
                    f"FMI station table (elevations) could not be read: {exc}"
                ) from exc
            logger.warning(
                "FMI: station table unreadable, elevations left empty "
                "(allow_partial=True): %s",
                exc,
            )
            return {}

    def _stations(self) -> list[dict]:
        if self._stations_cache is None:
            content = self._wfs(
                STATIONS_QUERY,
                [
                    ("starttime", "1900-01-01T00:00:00Z"),
                    ("endtime", _iso(datetime.now(UTC))),
                ],
            )
            raw = _parse_stations(content)
            if not raw:
                raise FMIError("FMI station list is empty")
            elevations = self._elevations()
            self._stations_cache = [
                _station_from_raw(r, elevations)
                for r in raw
                if any(n in SNOW_NETWORKS for n, _ in r["networks"])
            ]
            logger.info(
                "FMI: %d stations in snow-depth networks (of %d listed)",
                len(self._stations_cache),
                len(raw),
            )
        return self._stations_cache

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        All stations of FMI's snow-depth networks, open and closed.

        Parameters
        ----------
        active_only : bool
            Keep stations with an activity period open to now.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (fmisid), ``name``, ``latitude``, ``longitude``
            (ETRS89), ``elevation_m`` (FMI's station table; ``None`` when
            absent), ``status``, ``begin_date``, ``end_date`` (``None``
            while active), ``wmo_id``, ``region``, ``network_code``
            (``AWS``/``CLIM``/``WS``/``PRCP``, see :data:`SNOW_NETWORKS`),
            ``network_ids``, ``networks`` (Finnish names, every network the
            station belongs to), ``station_groups``, ``operator`` and
            ``station_url`` (``None``: FMI has no per-station page).

        Raises
        ------
        FMIError
            On an API failure, or when the station table cannot be read and
            ``allow_partial`` is False.
        """
        stations = [dict(s) for s in self._stations()]
        if active_only:
            stations = [s for s in stations if s["status"] == "Active"]
        return _filter_by_bbox(stations, bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Station metadata plus ``data_variables``.

        The service has no per-series inventory, so ``data_variables``
        lists what FMI's snow-depth networks report, with the station's
        period of activity as ``begin_date`` / ``end_date``; ``n_obs`` is
        ``None``.  Whether a series holds values is only known by fetching
        it (closed stations may be unknown to the observation service — see
        :meth:`screen_station_ids`).

        Parameters
        ----------
        station_id : str
            fmisid.

        Returns
        -------
        dict
            The station dict plus ``data_variables``: daily ``snow``,
            ``tday``, ``tmin``, ``tmax`` and ``rrday`` for every station,
            and the hourly variables for automatic weather stations.

        Raises
        ------
        FMIError
            If the station is not in a snow-depth network.
        """
        sid = str(station_id).strip()
        match = [s for s in self._stations() if s["station_id"] == sid]
        if not match:
            raise FMIError(f"Station {sid!r} is not in FMI's snow-depth networks")
        sta = dict(match[0])
        aws = 121 in sta["network_ids"]
        dvars = []
        for code, info in VARIABLES.items():
            for interval in info["interval"]:
                if interval != "daily" and not aws:
                    continue
                dvars.append(
                    {
                        "name": code,
                        "type": info["type"],
                        "interval": interval,
                        "units": info["output_units"],
                        "description": info["description"],
                        "notes": info["notes"],
                        "begin_date": sta["begin_date"],
                        "end_date": sta["end_date"],
                        "n_obs": None,
                    }
                )
        sta["data_variables"] = dvars
        return sta

    def screen_station_ids(
        self, station_ids: list[str] | str
    ) -> tuple[list[str], list[str]]:
        """
        Split fmisids into those the observation service knows and those it
        does not.

        The service rejects a whole request when any one fmisid is unknown,
        so ids are tried in batches with a one-day window and a failing
        batch is halved until the unknown ids are isolated.  Results are
        cached on the client.

        Parameters
        ----------
        station_ids : list[str] or str
            fmisids.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(known, unknown)``, each in input order.

        Raises
        ------
        FMIError
            On an API failure other than an unknown id.
        """
        ids = list(dict.fromkeys(_coerce_list(station_ids)))
        todo = [
            i for i in ids if i not in self._known_ids and i not in self._unknown_ids
        ]
        day = _iso(datetime.now(UTC).date() - timedelta(days=1))

        def probe(group: list[str]) -> None:
            try:
                self._wfs(
                    DAILY_QUERY,
                    [("parameters", "snow"), ("starttime", day), ("endtime", day)]
                    + [("fmisid", i) for i in group],
                )
            except _UnknownStationError:
                if len(group) == 1:
                    self._unknown_ids.add(group[0])
                    return
                mid = len(group) // 2
                probe(group[:mid])
                probe(group[mid:])
                return
            self._known_ids.update(group)

        for group in _chunk(todo, _MAX_IDS):
            probe(group)
        known = [i for i in ids if i in self._known_ids]
        unknown = [i for i in ids if i in self._unknown_ids]
        return known, unknown

    # ── Data ─────────────────────────────────────────────────────────────────

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
        Observations as flat records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            fmisid(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            FMI parameter names (``"snow"``, ``"tday"``, ``"snow_aws"``, …)
            or standardized types (``"snwd"``, ``"temp"``, …).  ``None`` =
            snow depth (``snow`` daily, ``snow_aws`` sub-daily).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds.  Without them the request
            covers the stations' joint activity period: from the earliest
            ``begin_date`` to the latest ``end_date``, or to today when any
            station is active (the whole record).  Ranges are split into
            366-day (daily) or 7-day (sub-daily) requests.
        interval : str
            ``"daily"`` (default), ``"hourly"`` or ``"sub_hourly"``
            (10-minute values).
        include_flags : bool
            Add ``flag``: ``"snow_free"`` / ``"no_precipitation"`` where the
            source sent its ``-1`` code, else ``None``.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``datetime`` for sub-daily, ``flag``), one per non-missing
            value.  Stations the observation service does not know are
            logged and skipped.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        FMIError
            On an unknown variable, unsupported interval, a variable not
            served at the interval, a non-numeric station id, or an API
            failure.

        Example
        -------
        >>> client = FMIClient()
        >>> records = client.get_data(
        ...     station_ids="101932",  # Sodankylä Tähtelä
        ...     begin_date="2024-01-01",
        ...     end_date="2024-04-30",
        ... )
        >>> records[0]["units"]
        'cm'
        """
        iv = _resolve_interval(interval)
        codes = _resolve_variables(variables, iv)
        if station_ids is not None:
            ids = list(dict.fromkeys(s.strip() for s in _coerce_list(station_ids)))
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        for i in ids:
            if not i.isdigit():
                raise FMIError(f"FMI station ids are integers (fmisid); got {i!r}")
        if not ids:
            return []

        if begin_date is None or end_date is None:
            span_begin, span_end = self._record_span(ids)
        begin = date.fromisoformat(_date_str(begin_date)) if begin_date else span_begin
        end = date.fromisoformat(_date_str(end_date)) if end_date else span_end
        if begin > end:
            return []

        known, unknown = self.screen_station_ids(ids)
        if unknown:
            logger.warning(
                "FMI: %d station(s) unknown to the observation service, skipped: %s",
                len(unknown),
                ", ".join(unknown[:20]) + (" …" if len(unknown) > 20 else ""),
            )
        if not known:
            return []

        query = DAILY_QUERY if iv == "daily" else INSTANT_QUERY
        extra: list[tuple[str, Any]] = [("parameters", ",".join(codes))]
        if iv != "daily":
            extra.append(("timestep", _TIMESTEP[iv]))
        records: list[dict] = []
        for group in _chunk(known, _MAX_IDS):
            for start, stop in _windows(begin, end, iv):
                content = self._wfs(
                    query,
                    extra
                    + [("starttime", start), ("endtime", stop)]
                    + [("fmisid", i) for i in group],
                )
                for fmisid, param, pairs in _parse_series(content):
                    if param not in VARIABLES:
                        continue
                    records.extend(
                        _records(fmisid, param, pairs, iv, include_flags, begin, end)
                    )
        records.sort(
            key=lambda r: (
                r["station_id"],
                r["variable"],
                r.get("datetime") or r["date"],
            )
        )
        return records

    def _record_span(self, ids: list[str]) -> tuple[date, date]:
        """Default ``(begin, end)``: the stations' joint activity period.

        From the earliest ``begin_date`` to the latest ``end_date``, or to
        today when any station is active or its period is unknown, so a
        station closed in 1996 is not asked for 30 empty years.
        """
        by_id = {s["station_id"]: s for s in self._stations()}
        today = date.today()
        listed = [by_id[i] for i in ids if i in by_id]
        begins = [s["begin_date"] for s in listed if s["begin_date"]]
        begin = (
            max(_EARLIEST, date.fromisoformat(min(begins)))
            if len(begins) == len(ids)
            else _EARLIEST
        )
        ends = [s["end_date"] for s in listed if s["end_date"]]
        end = (
            min(today, date.fromisoformat(max(ends)))
            if len(ends) == len(ids)
            else today
        )
        return begin, end


def _records(
    fmisid: str,
    param: str,
    pairs: list[tuple[str, str]],
    interval: str,
    include_flags: bool,
    begin: date,
    end: date,
) -> list[dict]:
    info = VARIABLES[param]
    lo, hi = begin.isoformat(), end.isoformat()
    out: list[dict] = []
    for t, raw in pairs:
        day = t[:10]
        if day < lo or day > hi:
            continue
        value, flag = _convert(param, raw)
        if value is None:
            continue
        rec: dict[str, Any] = {
            "station_id": fmisid,
            "date": day,
            "variable": param,
            "type": info["type"],
            "value": value,
            "units": info["output_units"],
            "interval": interval,
        }
        if interval != "daily":
            rec["datetime"] = t
        if include_flags:
            rec["flag"] = flag
        out.append(rec)
    return out
