"""
clients/alberta/alberta_client.py
=================================
Python client for the snow data behind Alberta Environment and Protected
Areas' (AEPA) *Alberta River Basins* web app (https://rivers.alberta.ca/).

Primary use case: automated snow-pillow SWE and snow depth in the Alberta
Rockies and foothills (near-real-time, 15-minute, records back to the
1980s at most AEPA pillows) plus the manual snow-survey (snow course) SWE
and depth record (back to the 1960s–1970s at many courses).

Data source       : https://rivers.alberta.ca/ (Alberta River Basins)
Authentication    : none
Documented API    : none.  Everything here is what the public web page's
                    own JavaScript calls (``/js/ArbPublicPageScripts``,
                    inspected 2026-10); it may change without notice.

Endpoints used (the same ones the public map uses)
--------------------------------------------------
- Station catalog — ArcGIS MapServer
  ``https://geospatial.alberta.ca/europa/rest/services/arb/alberta_river_basins/MapServer``
  reached through the site's own proxy::

      https://rivers.alberta.ca/proxy.ashx?<MapServer>/<layer>/query?where=...&f=json

  The MapServer itself answers ``499 Token Required``; the proxy needs a
  ``Referer: https://rivers.alberta.ca/`` header (it answers 403
  otherwise), exactly as a browser on the page sends.

  * layer **39** ``ARB_STATIONS_ENVDATA_TIMESERIES_TABLE`` — one row per
    station × parameter time series: ``station_number``, ``ParNumber``
    (``SW``, ``SD``, ``SW-SS``, ``SD-SS``, ``TA``, ``PC`` …), ``tsID``
    (the WISKI time-series id), ``TsIdLowres`` (its daily aggregate),
    ``TsUnitSymbol``, ``TsMinDataAsCurrentTime``/``TsMaxDataAsCurrentTime``
    (first/last value, MST wall clock encoded as epoch ms),
    ``CurrentData`` (``current``/``not_current``), ``station_status``,
    ``SiteNumber`` (province/state: AB, BC, MT, SK).
  * layer **11** ``ARB_STATIONS_ENVDATA`` — station name, latitude,
    longitude, status.  18 BC stations listed in layer 39 are missing
    here and therefore have no coordinates.

- Data — ``https://rivers.alberta.ca/WiskiLiveDataService/Download``
  ``?tsId=<tsID>&from=YYYY-MM-DD&to=YYYY-MM-DD&filename=..&zip=false&json=true``
  (the page's "Data Download" button).  Returns
  ``[{"station_no", "stationparameter_name", "ts_name", "ts_unitsymbol",
  "rows", "columns": "Timestamp,Value", "data": [["YYYY-MM-DD HH:MM:SS",
  value|null], ...]}]``.  ``to`` is inclusive of the whole day.  No row
  cap was hit (7 years of 15-minute data, 145 000 rows, in ~8 s).  An
  unknown ``tsId`` gives HTTP 500.

Time convention
---------------
Timestamps are **MST (UTC−7) all year, no daylight saving**.  Verified by
lag-correlating hourly air temperature at the two SNOTEL sites mirrored
here (Many Glacier 13A27S, Flattop Mtn 13A19S) against NRCS AWDB (PST):
the Alberta timestamps lead AWDB by exactly 1 h in both March and August
2026.  Sub-daily records carry ``datetime`` as ISO 8601 with ``-07:00``.
(The catalog's ``TsMinDateUtc``/``TsMaxDateUtc`` fields are MST + 6 h, not
true UTC; this client uses the ``*AsCurrentTime`` fields instead.)

Intervals
---------
- ``"sub_hourly"`` — the native series (``C.Corrected-Seasonal`` for
  SW/SD): 15-minute at AEPA pillows since ~2010s, hourly in older years
  and at mirrored SNOTEL sites.
- ``"hourly"`` — the native samples that fall exactly on HH:00 MST
  (instantaneous, not an hourly mean); hours without an :00 sample are
  absent.
- ``"daily"`` — for SW, SD and PC the source's own WISKI daily aggregate
  (``TsIdLowres``: ``A.DayMean`` for SW/SD, ``A.DayTotal`` for PC),
  labelled at 00:00 MST of the day it summarises (a few early years are
  labelled 01:00; the date part is used either way).  ``A.DayMean`` is
  AEPA's own (time-weighted) mean and differs by a fraction of a mm from a
  plain arithmetic mean of the 15-minute values.  For TA — whose
  ``TsIdLowres`` sometimes points at a *different* sensor (e.g. ``TA2``) —
  and for any series whose daily aggregate is missing or mislabelled, the
  daily value is computed here as the arithmetic mean of the native
  samples in the MST calendar day, kept only when they span at least 20
  distinct hours.
- ``"periodic"`` — manual snow survey ``SW-SS`` / ``SD-SS`` (``P.Survey``),
  one value per visit (typically monthly, Jan–Jun).

Variables
---------
See :data:`VARIABLES`.  Native units: SW / SW-SS mm, SD / SD-SS cm, TA
°C, PC mm.  Output: SWE and depth in cm, TA in °C, PC in mm.  Snow values
outside 0–1500 cm (including small negative pillow readings) are nulled.
``PC`` (precipitation gauge) is offered only as the daily total
(``A.DayTotal``) because the native series is a cumulative gauge level.

Stations
--------
At 2026-10-14 the catalog listed 291 snow stations (any of SW, SD, SW-SS,
SD-SS): 55 with automated SWE (19 AEPA pillows, 27 BC Snow Survey sites,
7 NRCS SNOTEL sites in Montana, 2 Global Water Futures Observatory
sites), 129 with automated snow depth, 202 manual snow courses.  The
operator is the suffix of the station name ("... - EPA", "- BC",
"- Snotel", "- MSC", "- Merchant", "- AGI" ...); it is passed through as
``operator_code`` and only the codes whose meaning is unambiguous are
expanded in ``operator``.  **Elevation is not published** by the app and
is ``None``.

``status`` is ``"Active"`` when any automated snow series is flagged
``CurrentData == "current"`` or the latest snow survey is within the last
550 days; the catalog's own status (Active / Discontinued / Inactive /
Unknown / Pending / Decommissioned — "Unknown" for most BC sites) is kept
as ``source_status``.

Licence / terms
---------------
No licence is stated for this web app or its download.  The page shows a
"Download Disclaimer": data are provisional, automatically generated by
remote equipment that may not be under Government of Alberta control, not
reviewed, may change, and are used at your own risk.  The footer links
the general Government of Alberta "Copyright and Disclaimer" page
(https://www.alberta.ca/disclaimer.aspx).  Alberta's Open Government
Licence – Alberta applies to datasets published on open.alberta.ca; the
ARB snow data are *not* published there (only PDF protocols are), so OGL–
Alberta has not been verified to cover this data.  Mirrored BC and SNOTEL
series are also available from their own operators.

Quality flags
-------------
The download carries no quality flags.  :data:`DATA_FLAGS` is empty and
``include_flags=True`` adds ``flag: None``.

Design principles
-----------------
- Returns plain Python objects (dicts / lists).
- Metric-first: SWE returned in cm (native mm × 0.1); depth cm, TA °C and
  PC mm pass through.
- Missing values are ``None``; snow values outside 0–15 m are nulled.
- Unknown variables or intervals raise :class:`AlbertaError` (no
  fallback); a named parameter not offered at ``interval`` raises too.
- HTTP retries use the shared ``_common.request_with_retries`` loop.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

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

BASE_URL = "https://rivers.alberta.ca"
MAPSERVER_URL = (
    "https://geospatial.alberta.ca/europa/rest/services/arb/"
    "alberta_river_basins/MapServer"
)
#: Station × time-series table (layer 39) and station point layer (layer 11).
TIMESERIES_LAYER = 39
STATIONS_LAYER = 11

_PROXY_PATH = "/proxy.ashx?"
_DOWNLOAD_PATH = "/WiskiLiveDataService/Download"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4
_PAGE_SIZE = 2000  # MapServer maxRecordCount
_IN_CHUNK = 100  # station ids per IN (...) clause

#: Fixed MST offset used by every timestamp on the site (no DST).
MST = timezone(timedelta(hours=-7), "MST")

# Snow plausibility (output units, cm): 0–15 m.
_MAX_SNOW_CM = 1500.0

# A manual-survey-only station counts as Active if surveyed this recently.
_SURVEY_ACTIVE_DAYS = 550

# Client-side daily means need native samples in at least this many hours.
_MIN_HOURS_FOR_DAILY = 20

_SOURCE = "Alberta Environment and Protected Areas — Alberta River Basins, " + (
    BASE_URL + _DOWNLOAD_PATH
)

# ── Public variable / flag tables ────────────────────────────────────────────

#: Native ARB parameters (``ParNumber``) this client serves.  ``scale``
#: converts native units to ``output_units``; ``intervals`` lists the
#: intervals each parameter supports.
VARIABLES: dict[str, dict] = {
    "SW": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "intervals": ("sub_hourly", "hourly", "daily"),
        "source": _SOURCE,
        "description": (
            "Automated snow water equivalent (snow pillow at AEPA sites; "
            "mirrored BC, SNOTEL and GWFO sensors)."
        ),
        "notes": (
            "Native C.Corrected-Seasonal series (provisional), mm → cm. "
            "Daily = AEPA's A.DayMean aggregate (00:00 MST label). "
            "Values outside 0–1500 cm are nulled."
        ),
    },
    "SD": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "intervals": ("sub_hourly", "hourly", "daily"),
        "source": _SOURCE,
        "description": "Automated (ultrasonic) snow depth.",
        "notes": (
            "Native C.Corrected-Seasonal series (provisional), cm. Daily = "
            "AEPA's A.DayMean aggregate. Values outside 0–1500 cm are nulled."
        ),
    },
    "SW-SS": {
        "name": "Snow Water Equivalent (snow survey)",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "intervals": ("periodic",),
        "source": _SOURCE,
        "description": "Manual snow-course SWE, one value per survey visit.",
        "notes": "P.Survey series, mm → cm. Survey date in MST.",
    },
    "SD-SS": {
        "name": "Snow Depth (snow survey)",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "intervals": ("periodic",),
        "source": _SOURCE,
        "description": "Manual snow-course depth, one value per survey visit.",
        "notes": "P.Survey series, cm.",
    },
    "TA": {
        "name": "Air Temperature",
        "type": "temp",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "intervals": ("sub_hourly", "hourly", "daily"),
        "source": _SOURCE,
        "description": "Air temperature at snow stations.",
        "notes": (
            "Native C.Corrected-Sensor or P.NRT series. Daily = arithmetic "
            "mean of native samples in the MST day, computed in-client "
            "(kept when they span >= 20 hours); the source's own daily "
            "series sometimes belongs to a secondary sensor (TA2)."
        ),
    },
    "PC": {
        "name": "Precipitation",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "intervals": ("daily",),
        "source": _SOURCE,
        "description": "Daily precipitation total from the station gauge.",
        "notes": (
            "AEPA's A.DayTotal aggregate of the cumulative gauge series. "
            "Sub-daily precipitation is not offered (the native series is "
            "cumulative gauge level)."
        ),
    },
}

#: Parameters that make a station a *snow* station.
SNOW_PARAMETERS: tuple[str, ...] = ("SW", "SD", "SW-SS", "SD-SS")
_AUTOMATED_SNOW = ("SW", "SD")
_SURVEY_SNOW = ("SW-SS", "SD-SS")

#: Default variables when ``variables=None`` (snow only), per interval.
_DEFAULT_BY_INTERVAL: dict[str, list[str]] = {
    "sub_hourly": ["SW", "SD"],
    "hourly": ["SW", "SD"],
    "daily": ["SW", "SD"],
    "periodic": ["SW-SS", "SD-SS"],
}

#: Daily aggregate ``ts_name`` the source uses, per parameter.
_DAILY_TS_NAME = {"SW": "A.DayMean", "SD": "A.DayMean", "PC": "A.DayTotal"}

#: The download exposes no quality flags.
DATA_FLAGS: dict[str, str] = {}

#: Station-name suffix → operator.  Only unambiguous codes are expanded.
OPERATORS: dict[str, str] = {
    "EPA": "Alberta Environment and Protected Areas",
    "BC": "British Columbia Snow Survey Network",
    "Snotel": "USDA NRCS SNOTEL",
    "GWFO": "Global Water Futures Observatories",
    "OSM": "Oil Sands Monitoring Program",
}


# ── Parsers / helpers ────────────────────────────────────────────────────────


def _epoch_ms_to_mst_wall(ms: Any) -> datetime | None:
    """Decode a ``*AsCurrentTime`` field: MST wall clock stored as epoch ms."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, UTC).replace(tzinfo=None)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _split_name(name: str | None) -> tuple[str | None, str | None]:
    """``"Sunshine Village - EPA"`` → ``("Sunshine Village", "EPA")``."""
    if not name:
        return name, None
    m = re.match(r"^(.*?)\s+-\s+([A-Za-z]+)$", name.strip())
    if m:
        return m.group(1), m.group(2)
    return name.strip(), None


def _parse_download_json(payload: Any) -> tuple[dict, list[tuple[datetime, Any]]]:
    """Parse a ``Download?...&json=true`` payload.

    Returns ``(header, rows)`` with ``rows`` as ``(naive MST datetime, raw
    value)``.  Raises :class:`AlbertaError` on an unexpected shape.
    """
    if isinstance(payload, list):
        if not payload:
            return {}, []
        payload = payload[0]
    if not isinstance(payload, dict) or "data" not in payload:
        raise AlbertaError("Unexpected WISKI download payload (no 'data' field)")
    header = {k: v for k, v in payload.items() if k != "data"}
    rows: list[tuple[datetime, Any]] = []
    for item in payload.get("data") or []:
        if not item or len(item) < 2:
            continue
        try:
            ts = datetime.strptime(str(item[0])[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        rows.append((ts, item[1]))
    return header, rows


def _convert(param: str, raw: Any) -> float | None:
    """Native value → output units, nulling implausible snow values."""
    value = _to_float(raw)
    if value is None:
        return None
    info = VARIABLES[param]
    value = round(value * info["scale"], 3)
    if info["type"] in ("swe", "snwd") and not (0.0 <= value <= _MAX_SNOW_CM):
        return None
    return value


def _daily_means(
    rows: list[tuple[datetime, Any]], param: str
) -> list[tuple[datetime, float | None]]:
    """Arithmetic daily mean of native samples per MST calendar day.

    A day is kept only when its non-missing samples span at least
    ``_MIN_HOURS_FOR_DAILY`` distinct hours.
    """
    by_day: dict[date, list[tuple[int, float]]] = {}
    for ts, raw in rows:
        v = _to_float(raw)
        if v is None:
            continue
        by_day.setdefault(ts.date(), []).append((ts.hour, v))
    out: list[tuple[datetime, float | None]] = []
    for day in sorted(by_day):
        samples = by_day[day]
        if len({h for h, _ in samples}) < _MIN_HOURS_FOR_DAILY:
            continue
        mean = sum(v for _, v in samples) / len(samples)
        out.append((datetime(day.year, day.month, day.day), mean))
    return out


def _resolve_variables(variables: list[str] | str | None, interval: str) -> list[str]:
    """Resolve native names / standardized types to native parameters.

    Raises :class:`AlbertaError` for unknown names, for a native parameter
    that does not support ``interval``, or for a type with no parameter at
    that interval.
    """
    items = _coerce_list(variables) if variables is not None else []
    if not items:
        return list(_DEFAULT_BY_INTERVAL[interval])
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key.upper() in VARIABLES:
            native = key.upper()
            if interval not in VARIABLES[native]["intervals"]:
                raise AlbertaError(
                    f"Variable {native!r} is not available at interval "
                    f"{interval!r} (supported: {list(VARIABLES[native]['intervals'])})."
                )
            matches = [native]
        else:
            matches = [
                p
                for p, info in VARIABLES.items()
                if info["type"] == key.lower() and interval in info["intervals"]
            ]
            if not matches:
                known_type = any(i["type"] == key.lower() for i in VARIABLES.values())
                if known_type:
                    raise AlbertaError(
                        f"No Alberta variable of type {key!r} at interval {interval!r}."
                    )
                raise AlbertaError(
                    f"Unknown variable {key!r} — expected one of "
                    f"{sorted(VARIABLES)} or their types."
                )
        for m in matches:
            if m not in out:
                out.append(m)
    return out


def _check_interval(interval: str) -> str:
    iv = (interval or "").lower()
    supported = set(_DEFAULT_BY_INTERVAL)
    if iv not in supported:
        raise AlbertaError(
            f"Unsupported interval {interval!r} for Alberta — expected one of "
            f"{sorted(supported)}."
        )
    return iv


class AlbertaError(Exception):
    """Raised for Alberta River Basins source failures and bad requests."""


# ── Client ───────────────────────────────────────────────────────────────────


class AlbertaClient:
    """
    Client for snow data on the Alberta River Basins site (rivers.alberta.ca).

    No credentials are needed.

    Parameters
    ----------
    base_url : str
        Site root (``https://rivers.alberta.ca``).
    timeout : int
        HTTP timeout in seconds.
    max_retries : int
        Retry attempts on transient errors (5xx, network).
    backoff : int
        Base backoff delay in seconds (delay = backoff × attempt).
    session : requests.Session or None
        Optional pre-configured session.
    allow_partial : bool
        :meth:`get_data` downloads one series per station and parameter.
        By default a failed download raises :class:`AlbertaError` rather
        than return the other series without it.  ``True`` logs a warning
        and skips that series, for interactive use.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
        allow_partial: bool = False,
    ) -> None:
        self.allow_partial = allow_partial
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        # The site's proxy refuses requests without a referer from the page.
        self._session.headers.update(
            {"User-Agent": "easysnowdata", "Referer": self.base_url + "/"}
        )
        self._series_cache: dict[str, dict[str, dict]] | None = None
        self._stations_cache: dict[str, dict] | None = None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _get(self, url: str) -> requests.Response:
        return request_with_retries(
            self._session,
            url,
            error_cls=AlbertaError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )

    def _query_layer(self, layer: int, where: str, out_fields: str = "*") -> list[dict]:
        """Run an ArcGIS ``query`` on the ARB MapServer via the site proxy.

        Pages with ``resultOffset`` until the server stops reporting
        ``exceededTransferLimit``.  Returns the feature ``attributes``.
        """
        rows: list[dict] = []
        offset = 0
        while True:
            qs = urllib.parse.urlencode(
                {
                    "where": where,
                    "outFields": out_fields,
                    "returnGeometry": "false",
                    "orderByFields": "OBJECTID",
                    "resultOffset": offset,
                    "resultRecordCount": _PAGE_SIZE,
                    "f": "json",
                }
            )
            url = f"{self.base_url}{_PROXY_PATH}{MAPSERVER_URL}/{layer}/query?{qs}"
            try:
                payload = self._get(url).json()
            except ValueError as exc:
                raise AlbertaError(f"Non-JSON reply from ARB layer {layer}") from exc
            if "error" in payload:
                raise AlbertaError(
                    f"ARB layer {layer} query failed: {payload['error']}"
                )
            feats = payload.get("features") or []
            rows.extend(f.get("attributes", {}) for f in feats)
            if not payload.get("exceededTransferLimit") or not feats:
                return rows
            offset += len(feats)

    # ── Catalog ──────────────────────────────────────────────────────────────

    def _series(self) -> dict[str, dict[str, dict]]:
        """``{station_number: {ParNumber: layer-39 row}}`` for snow stations."""
        if self._series_cache is not None:
            return self._series_cache
        snow_in = ",".join(f"'{p}'" for p in SNOW_PARAMETERS)
        rows = self._query_layer(TIMESERIES_LAYER, f"ParNumber IN ({snow_in})")
        series: dict[str, dict[str, dict]] = {}
        for r in rows:
            sid, par = r.get("station_number"), r.get("ParNumber")
            if sid and par:
                series.setdefault(sid, {}).setdefault(par, r)
        extra_in = ",".join(f"'{p}'" for p in VARIABLES if p not in SNOW_PARAMETERS)
        for ids in _chunk(sorted(series), _IN_CHUNK):
            id_in = ",".join(f"'{i}'" for i in ids)
            for r in self._query_layer(
                TIMESERIES_LAYER,
                f"ParNumber IN ({extra_in}) AND station_number IN ({id_in})",
            ):
                sid, par = r.get("station_number"), r.get("ParNumber")
                if sid in series and par:
                    series[sid].setdefault(par, r)
        logger.info("Alberta: %d snow stations in the ARB catalog", len(series))
        self._series_cache = series
        return series

    def _station_points(self, ids: list[str]) -> dict[str, dict]:
        points: dict[str, dict] = {}
        for group in _chunk(ids, _IN_CHUNK):
            id_in = ",".join(f"'{i}'" for i in group)
            for r in self._query_layer(
                STATIONS_LAYER,
                f"station_number IN ({id_in})",
                "station_number,station_name,station_latitude,"
                "station_longitude,station_status,data_type",
            ):
                if r.get("station_number"):
                    points.setdefault(r["station_number"], r)
        return points

    def _stations(self) -> dict[str, dict]:
        if self._stations_cache is not None:
            return self._stations_cache
        series = self._series()
        points = self._station_points(sorted(series))
        out: dict[str, dict] = {}
        for sid, pars in series.items():
            out[sid] = _build_station(sid, pars, points.get(sid))
        self._stations_cache = out
        return out

    # ── Public API — stations ────────────────────────────────────────────────

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        All ARB stations with automated or manual SWE / snow depth.

        Parameters
        ----------
        active_only : bool
            Keep stations with a ``current`` automated snow series or a snow
            survey in the last 550 days.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id``, ``name``, ``latitude``, ``longitude``,
            ``elevation_m`` (always ``None``: not published), ``status``,
            ``source_status``, ``state`` (AB/BC/MT/SK), ``operator``,
            ``operator_code``, ``measurement_method``, ``has_swe``,
            ``has_snwd``, ``has_pillow_swe``, ``has_survey``,
            ``begin_date``, ``end_date``, ``parameters`` and
            ``station_url``.  Stations without coordinates are dropped when
            ``bbox`` is given.

        Raises
        ------
        AlbertaError
            On a catalog query failure.
        """
        stations = list(self._stations().values())
        if active_only:
            stations = [s for s in stations if s["status"] == "Active"]
        stations = [dict(s) for s in stations]
        return _filter_by_bbox(stations, bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Metadata and series inventory for one station.

        Parameters
        ----------
        station_id : str
            ARB station number, e.g. ``"05BB803"``.

        Returns
        -------
        dict
            The :meth:`get_all_stations` dict plus ``data_variables`` (one
            entry per parameter × supported interval) with ``name``,
            ``type``, ``interval``, ``units`` (emitted), ``description``,
            ``notes``, ``begin_date``, ``end_date`` and ``n_obs``
            (``None``), plus the WISKI ``ts_id`` / ``ts_name``.  Dates are
            MST.

        Raises
        ------
        AlbertaError
            If the station is not a snow station in the ARB catalog.
        """
        sid = str(station_id).strip()
        stations = self._stations()
        if sid not in stations:
            raise AlbertaError(f"Station {sid!r} not found in the ARB snow catalog")
        meta = dict(stations[sid])
        dvars: list[dict] = []
        for par, row in self._series()[sid].items():
            if par not in VARIABLES:
                continue
            info = VARIABLES[par]
            begin = _epoch_ms_to_mst_wall(row.get("TsMinDataAsCurrentTime"))
            end = _epoch_ms_to_mst_wall(row.get("TsMaxDataAsCurrentTime"))
            for iv in info["intervals"]:
                dvars.append(
                    {
                        "name": par,
                        "type": info["type"],
                        "interval": iv,
                        "units": info["output_units"],
                        "description": info["description"],
                        "notes": info["notes"],
                        "begin_date": begin.date().isoformat() if begin else None,
                        "end_date": end.date().isoformat() if end else None,
                        "n_obs": None,
                        "ts_id": row.get("tsID"),
                        "ts_name": row.get("TsName"),
                    }
                )
        meta["data_variables"] = dvars
        return meta

    # ── Public API — data ────────────────────────────────────────────────────

    def download_series(
        self, ts_id: str, begin_date: str, end_date: str
    ) -> tuple[dict, list[tuple[datetime, Any]]]:
        """
        Raw WISKI download for one time series.

        Parameters
        ----------
        ts_id : str
            WISKI time-series id (``tsID`` / ``TsIdLowres`` from the catalog).
        begin_date, end_date : str
            Inclusive ``YYYY-MM-DD`` bounds (MST).

        Returns
        -------
        tuple[dict, list[tuple[datetime, Any]]]
            ``(header, rows)``; ``header`` is the download's metadata
            (``station_no``, ``stationparameter_name``, ``ts_name``, …) and
            rows are ``(naive MST datetime, raw value)`` in native units.

        Raises
        ------
        AlbertaError
            On a failed request (HTTP 500 for an unknown ``tsId``) or an
            unexpected payload.
        """
        qs = urllib.parse.urlencode(
            {
                "tsId": ts_id,
                "from": begin_date,
                "to": end_date,
                "filename": "easysnowdata",
                "zip": "false",
                "json": "true",
            }
        )
        resp = self._get(f"{self.base_url}{_DOWNLOAD_PATH}?{qs}")
        if not resp.content:
            # Seen from 2026-10-26: HTTP 200, text/csv, zero bytes, for
            # every series, including from the site's own download button.
            raise AlbertaError(
                f"Empty WISKI download for tsId {ts_id} (HTTP {resp.status_code}, "
                f"{resp.headers.get('Content-Type', 'no content type')})"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise AlbertaError(f"Non-JSON WISKI download for tsId {ts_id}") from exc
        return _parse_download_json(payload)

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
            ARB station numbers, e.g. ``"05BB803"``.  Required unless
            ``bbox`` is given.
        variables : list[str] or str or None
            Native parameters (``"SW"``, ``"SW-SS"``) or standardized types
            (``"swe"``).  A type resolves to the parameters available at
            ``interval`` (``swe`` → ``SW`` for daily/hourly, ``SW-SS`` for
            periodic).  ``None`` = snow only.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds in MST.  Default: the series'
            first value / today.
        interval : str
            ``"daily"`` (default), ``"hourly"``, ``"sub_hourly"`` or
            ``"periodic"``.  See the module docstring for conventions.
        include_flags : bool
            Add ``flag`` (always ``None``: the source has no flags).

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``datetime`` ISO 8601 with ``-07:00`` for hourly/sub_hourly).
            Unknown stations are logged and skipped; parameters a station
            does not have are skipped silently.  A failed download raises,
            unless the client was built with ``allow_partial=True``.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        AlbertaError
            On an unknown variable or unsupported interval, or a failed
            download (see ``allow_partial``).

        Example
        -------
        >>> client = AlbertaClient()
        >>> records = client.get_data(
        ...     station_ids="05BB803",
        ...     variables=["swe"],
        ...     begin_date="2026-03-01",
        ...     end_date="2026-03-15",
        ... )
        >>> records[0]["units"]
        'cm'
        """
        iv = _check_interval(interval)
        params = _resolve_variables(variables, iv)
        if station_ids is not None:
            ids = _coerce_list(station_ids)
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")

        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        series = self._series()

        records: list[dict] = []
        for sid in ids:
            if sid not in series:
                logger.warning("Alberta: station %s not in the snow catalog", sid)
                continue
            for par in params:
                row = series[sid].get(par)
                if row is None:
                    continue
                try:
                    records.extend(
                        self._fetch(sid, par, row, iv, begin, end, include_flags)
                    )
                except AlbertaError as exc:
                    if not self.allow_partial:
                        raise
                    logger.warning(
                        "Alberta: skipping %s %s: %s (allow_partial=True)",
                        sid,
                        par,
                        exc,
                    )
        return records

    # ── Internals ────────────────────────────────────────────────────────────

    def _fetch(
        self,
        sid: str,
        par: str,
        row: dict,
        interval: str,
        begin: str | None,
        end: str | None,
        include_flags: bool,
    ) -> list[dict]:
        first = _epoch_ms_to_mst_wall(row.get("TsMinDataAsCurrentTime"))
        b = begin or (first.date().isoformat() if first else "1900-01-01")
        e = end or (datetime.now(MST).date() + timedelta(days=1)).isoformat()
        native_id = str(row.get("tsID") or "")
        if not native_id:
            return []

        sub_daily = interval in ("hourly", "sub_hourly")
        if interval == "daily":
            rows = self._daily_rows(par, row, native_id, b, e)
        else:
            _, rows = self.download_series(native_id, b, e)
            if interval == "hourly":
                rows = [(ts, v) for ts, v in rows if ts.minute == 0 and ts.second == 0]

        info = VARIABLES[par]
        out: list[dict] = []
        for ts, raw in rows:
            day = ts.date().isoformat()
            if (begin and day < begin) or (end and day > end):
                continue
            rec = {
                "station_id": sid,
                "date": day,
                "variable": par,
                "type": info["type"],
                "value": _convert(par, raw),
                "units": info["output_units"],
                "interval": interval,
            }
            if sub_daily:
                rec["datetime"] = ts.replace(tzinfo=MST).isoformat()
            if include_flags:
                rec["flag"] = None
            out.append(rec)
        return out

    def _daily_rows(
        self, par: str, row: dict, native_id: str, b: str, e: str
    ) -> list[tuple[datetime, Any]]:
        """Daily rows: the source's aggregate when trustworthy, else computed."""
        lowres = str(row.get("TsIdLowres") or "")
        wanted = _DAILY_TS_NAME.get(par)
        if wanted and lowres and lowres != native_id:
            header, rows = self.download_series(lowres, b, e)
            if (
                str(header.get("stationparameter_name", "")).upper() == par
                and header.get("ts_name") == wanted
            ):
                # One row per day; keep the first if a day repeats.
                seen: set[date] = set()
                kept = []
                for ts, v in rows:
                    if ts.date() not in seen:
                        seen.add(ts.date())
                        kept.append((ts, v))
                return kept
            logger.info(
                "Alberta: daily series %s for %s is %s/%s, computing daily means",
                lowres,
                par,
                header.get("stationparameter_name"),
                header.get("ts_name"),
            )
        if par == "PC":
            # A mean of a cumulative gauge is meaningless; no fallback.
            return []
        _, native = self.download_series(native_id, b, e)
        return _daily_means(native, par)


def _build_station(sid: str, pars: dict[str, dict], point: dict | None) -> dict:
    """Assemble one station dict from its layer-39 rows and layer-11 point."""
    any_row = next(iter(pars.values()))
    raw_name = (point or {}).get("station_name") or None
    name, op_code = _split_name(raw_name)
    lat = _to_float((point or {}).get("station_latitude"))
    lon = _to_float((point or {}).get("station_longitude"))

    has_pillow = "SW" in pars
    has_survey = any(p in pars for p in _SURVEY_SNOW)
    current = any(
        pars[p].get("CurrentData") == "current" for p in _AUTOMATED_SNOW if p in pars
    )
    survey_end = [
        _epoch_ms_to_mst_wall(pars[p].get("TsMaxDataAsCurrentTime"))
        for p in _SURVEY_SNOW
        if p in pars
    ]
    recent_survey = any(
        d is not None
        and (datetime.now(MST).replace(tzinfo=None) - d).days <= _SURVEY_ACTIVE_DAYS
        for d in survey_end
    )
    starts = [
        _epoch_ms_to_mst_wall(r.get("TsMinDataAsCurrentTime"))
        for p, r in pars.items()
        if p in SNOW_PARAMETERS
    ]
    ends = [
        _epoch_ms_to_mst_wall(r.get("TsMaxDataAsCurrentTime"))
        for p, r in pars.items()
        if p in SNOW_PARAMETERS
    ]
    starts = [d for d in starts if d]
    ends = [d for d in ends if d]

    methods = []
    if has_pillow:
        methods.append("automated SWE (snow pillow)")
    if "SD" in pars:
        methods.append("automated snow depth")
    if has_survey:
        methods.append("manual snow survey")

    return {
        "station_id": sid,
        "name": name or sid,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": None,
        "status": "Active" if (current or recent_survey) else "Inactive",
        "source_status": (point or {}).get("station_status")
        or any_row.get("station_status"),
        "state": any_row.get("SiteNumber"),
        "operator": OPERATORS.get(op_code or "", op_code),
        "operator_code": op_code,
        "measurement_method": "; ".join(methods),
        "has_swe": has_pillow or "SW-SS" in pars,
        "has_snwd": "SD" in pars or "SD-SS" in pars,
        "has_pillow_swe": has_pillow,
        "has_survey": has_survey,
        "begin_date": min(starts).date().isoformat() if starts else None,
        "end_date": max(ends).date().isoformat() if ends else None,
        "parameters": sorted(p for p in pars if p in VARIABLES),
        "station_url": f"{BASE_URL}/?StationNumber={sid}",
    }
