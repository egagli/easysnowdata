"""
clients/hydroquebec/hydroquebec_client.py
=========================================
Python client for Hydro-Québec's open hydrometeorological feed — the
utility's own network of weather and hydrometric stations across its
hydroelectric watersheds in Québec (Côte-Nord, Nord-du-Québec, Mauricie,
Outaouais, Saguenay–Lac-Saint-Jean, …).

Primary use case: near-real-time snow water equivalent (SWE) and snow
depth from ~95 GMON stations in northern and central Québec.

Data source    : Opendatasoft portal, dataset ``donnees-hydrometeorologiques``
                 https://donnees.hydroquebec.com/explore/dataset/donnees-hydrometeorologiques/
API            : https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/donnees-hydrometeorologiques
                 (``/records`` and ``/exports/json``; ODSQL ``where``/``group_by``)
Upstream file  : https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/Donnees_VUE_STATIONS_ET_TARAGES.json
                 (the nested JSON the portal flattens; also what Données Québec links to)
Authentication : none
Licence        : CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) —
                 attribution required, **no commercial use**.  The portal adds
                 "Les données fournies sont sans garantie de qualité et peuvent
                 changer sans préavis" (no quality guarantee; may change without
                 notice).

ROLLING WINDOW — NO HISTORY
---------------------------
The feed holds only a rolling **~10 days** of hourly data ("Couverture
temporelle : dix jours").  Older values disappear; there is no archive
endpoint, and no other Hydro-Québec dataset on the portal (checked:
``historique-donnees-meteo`` is ECCC weather without snow,
``donnees-hydrometriques`` is inflows/flows) or on Données Québec carries
snow history.  Anyone who needs a time series must **accumulate it
themselves** by calling :meth:`HydroQuebecClient.get_data` at least every
few days (or let ``global_snow_networks`` archive it).  The historical
GMON record through 2025 is available in **CanSWE** (Vionnet et al.),
which includes these Hydro-Québec stations — use that client for history.

Record layout
-------------
One flat row per (station, variable, hour)::

    identifiant, nom, xcoord (lon), ycoord (lat), zcoord (m, often null),
    coderegionqc, regionqc, date_debut, date_fin,
    composition_depil_type_point_donnee   (French variable label)
    composition_depil_nom_unite_mesure    (unit)
    composition_depil_pas_temps           ("Horaire" for every row seen)
    composition_depil_type_mesure         (Instantanée/Moyenne/Somme/Maximum/Minimum)
    date (UTC, ISO with +00:00), valeur (text)

Quirks handled here
-------------------
- ``valeur`` is **text**; ``-999.00`` means missing.
- The flattening emits ~4 **junk rows per series** with ``date = null``
  and ``valeur`` set to a label ("Horaire", "Moyenne", "mm", …).  They are
  dropped (``date is not null`` in every query).
- ``date`` is UTC (field description: "date et heure de la mesure, en
  UTC"; the upstream JSON uses ``Z`` suffixes and values match 1:1).
- Variable labels are French and contain dots/accents
  (``"Humidité relative.2 mètres"``); they are the native ``variable``
  names here.
- SWE is published as an **hourly** series but the GMON value only
  changes every **6 h** (at ~01, 07, 13, 19 UTC); the intermediate hours
  repeat the last reading.  The portal text says SWE is "recueillies
  toutes les six heures" and depth hourly.
- Snow depth hovers around zero with negative noise (median −0.3 cm in a
  snow-free October); negatives are nulled by the 0–15 m clamp.  SWE in
  snow-free conditions can show spurious non-zero GMON readings (seen:
  one station alternating 0 / 75 mm every 6 h with depth ≈ 0) —
  callers should cross-check SWE against depth before trusting
  early-season values.
- ``date_debut`` is the site commissioning date; ``1800-01-01`` is a
  placeholder and is returned as ``None``.
- ``zcoord`` (elevation) is null for ~55 % of sites (mostly hydrometric
  ones; only one SWE station lacks it).
- The upstream JSON is refreshed hourly; the portal re-processes it
  about twice a day (``update_frequency = DAILY_2``), so the newest portal
  value is typically 1–10 h old.

Units
-----
SWE native mm → **cm**; snow depth native cm → cm; precipitation mm;
temperatures °C; RH %; wind km/h and degrees; level m; discharge m³/s.

Daily aggregation (``interval="daily"``)
----------------------------------------
Hydro-Québec publishes no daily product for these stations (every row is
``pas_temps = "Horaire"``), so the client aggregates hourly values over
the **local calendar day in America/Toronto** (the dataset's declared
timezone; Eastern time, with DST) using:

- SWE, snow depth, level, water temperature at instant → **last valid
  value of the local day** (a snow-state convention);
- precipitation (hourly sums) → **sum**;
- ``Température Maximum`` → max; ``Température Minimum`` → min;
- RH, wind speed, discharge, mean level / water temperature → mean;
- wind direction → circular (unit-vector) mean.

Non-``last`` aggregations require at least :data:`MIN_HOURS_DAILY` valid
hours in the day, otherwise the value is ``None`` (this mainly affects the
first and last day of the rolling window).
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from .._common import (
    to_float as _to_float,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PORTAL_URL = "https://donnees.hydroquebec.com/api/explore/v2.1"
DATASET_ID = "donnees-hydrometeorologiques"
DATASET_PAGE = (
    "https://donnees.hydroquebec.com/explore/dataset/donnees-hydrometeorologiques/"
)
UPSTREAM_JSON = (
    "https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/"
    "Donnees_VUE_STATIONS_ET_TARAGES.json"
)
LICENSE = "CC BY-NC 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: Rolling window length the portal states ("dix jours").
WINDOW_DAYS = 10

#: Minimum valid hours for a sum/mean/max/min daily value.
MIN_HOURS_DAILY = 20

_LOCAL_TZ_NAME = "America/Toronto"
try:
    _LOCAL_TZ: Any = ZoneInfo(_LOCAL_TZ_NAME)
except ZoneInfoNotFoundError:  # pragma: no cover - Windows without tzdata
    logger.warning(
        "tz database lacks %s; daily aggregation falls back to fixed UTC-5",
        _LOCAL_TZ_NAME,
    )
    _LOCAL_TZ = timezone(timedelta(hours=-5), "EST")

# Field names in the flattened portal records.
_F_VAR = "composition_depil_type_point_donnee"
_F_UNIT = "composition_depil_nom_unite_mesure"
_F_STEP = "composition_depil_pas_temps"
_F_KIND = "composition_depil_type_mesure"

_SWE = "Équivalent en eau de la neige"
_SNWD = "Épaisseur de neige"

# Plausibility bounds in output units (DESIGN.md: SWE/depth 0–15 m).
_SNOW_MAX_CM = 1500.0

_SRC = f"Hydro-Québec open data — {DATASET_PAGE}"

_NAME_CODE_RE = re.compile(r"\(([A-Z0-9]{3,5})\)\s*$")

# ── Public variable / flag tables ────────────────────────────────────────────

#: Variables in the feed, keyed by the native French label
#: (``composition_depil_type_point_donnee``).  ``scale`` converts native to
#: output units; ``daily`` is the daily aggregation (see module docstring);
#: ``measure`` is the native ``type_mesure``.
VARIABLES: dict[str, dict] = {
    _SWE: {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "daily": "last",
        "measure": "Instantanée",
        "source": _SRC,
        "description": "Snow water equivalent from a GMON passive gamma sensor.",
        "notes": (
            "Native mm, returned in cm. Published hourly but the GMON value "
            "changes only every 6 h (~01/07/13/19 UTC); hours in between "
            "repeat the last reading. Daily = last valid value of the "
            "America/Toronto local day. Early-season GMON readings can be "
            "spurious when the ground is bare. Rolling ~10-day window only."
        ),
    },
    _SNWD: {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "daily": "last",
        "measure": "Instantanée",
        "source": _SRC,
        "description": "Depth of snow on the ground, hourly instantaneous.",
        "notes": (
            "Native cm. Sensor type not stated by Hydro-Québec (the portal "
            "says GMON measures both depth and SWE). Negative near-zero "
            "noise is nulled. Daily = last valid value of the local day."
        ),
    },
    "Précipitation": {
        "name": "Precipitation",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "daily": "sum",
        "measure": "Somme",
        "source": _SRC,
        "description": "Hourly total precipitation (rain and snow water).",
        "notes": (
            "Hourly sums; daily = sum over the local day (needs "
            f">= {MIN_HOURS_DAILY} valid hours). Negative values are nulled."
        ),
    },
    "Température Maximum": {
        "name": "Maximum Air Temperature",
        "type": "temp_max",
        "units": "°C",
        "output_units": "degC",
        "scale": 1.0,
        "daily": "max",
        "measure": "Maximum",
        "source": _SRC,
        "description": "Hourly maximum air temperature.",
        "notes": "Daily = max of hourly maxima over the local day.",
    },
    "Température Minimum": {
        "name": "Minimum Air Temperature",
        "type": "temp_min",
        "units": "°C",
        "output_units": "degC",
        "scale": 1.0,
        "daily": "min",
        "measure": "Minimum",
        "source": _SRC,
        "description": "Hourly minimum air temperature.",
        "notes": "Daily = min of hourly minima over the local day.",
    },
    "Humidité relative.2 mètres": {
        "name": "Relative Humidity (2 m)",
        "type": "rh",
        "units": "%",
        "output_units": "%",
        "scale": 1.0,
        "daily": "mean",
        "measure": "Instantanée",
        "source": _SRC,
        "description": "Relative humidity at 2 m, hourly instantaneous.",
        "notes": "Daily = mean over the local day.",
    },
    "Vitesse du vent.10 mètres": {
        "name": "Wind Speed (10 m)",
        "type": "wind_spd",
        "units": "km/h",
        "output_units": "km/h",
        "scale": 1.0,
        "daily": "mean",
        "measure": "Moyenne",
        "source": _SRC,
        "description": "Hourly mean wind speed at 10 m.",
        "notes": (
            "Daily = mean over the local day. The portal text mentions "
            "2.5 m wind, but only 10 m series are present in the feed."
        ),
    },
    "Direction du vent.10 mètres": {
        "name": "Wind Direction (10 m)",
        "type": "wind_dir",
        "units": "°",
        "output_units": "deg",
        "scale": 1.0,
        "daily": "circmean",
        "measure": "Instantanée",
        "source": _SRC,
        "description": "Wind direction at 10 m, hourly instantaneous.",
        "notes": "Daily = unweighted circular mean over the local day.",
    },
    "Niveau": {
        "name": "Water Level",
        "type": "other",
        "units": "m",
        "output_units": "m",
        "scale": 1.0,
        "daily": "mean",
        "measure": "Moyenne",
        "source": _SRC,
        "description": "Water level (lake, reservoir or river), hourly.",
        "notes": (
            "Mostly hourly means; two sites report instantaneous level "
            "(``Instantanée``). Daily = mean over the local day."
        ),
    },
    "Débit": {
        "name": "Discharge",
        "type": "other",
        "units": "m³/s",
        "output_units": "m3/s",
        "scale": 1.0,
        "daily": "mean",
        "measure": "Moyenne",
        "source": _SRC,
        "description": "River discharge computed from water level, hourly mean.",
        "notes": "Daily = mean over the local day.",
    },
    "Température de l'eau": {
        "name": "Water Temperature",
        "type": "other",
        "units": "°C",
        "output_units": "degC",
        "scale": 1.0,
        "daily": "mean",
        "measure": "Moyenne",
        "source": _SRC,
        "description": "Water temperature, hourly mean (one site).",
        "notes": "Daily = mean over the local day.",
    },
}

#: Standardized type → native labels.
_TYPE_TO_VARS: dict[str, list[str]] = {}
for _var, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_var)

#: Default variables when ``variables=None`` (DESIGN.md §3.4: snow only).
_DEFAULT_VARS: list[str] = [_SWE, _SNWD]

#: The feed carries no quality flags.  ``include_flags=True`` emits the
#: native ``type_mesure`` (Instantanée/Moyenne/Somme/Maximum/Minimum) as
#: ``flag`` on hourly records (``None`` on daily ones) so callers can
#: tell the two level series apart.
DATA_FLAGS: dict[str, str] = {}

_SUPPORTED_INTERVALS = ("hourly", "daily")

# ── Parsing helpers ──────────────────────────────────────────────────────────


def _parse_utc(text: str | None) -> datetime | None:
    """Parse the portal's ``2026-10-04T08:00:00+00:00`` into aware UTC."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _site_date(text: str | None) -> str | None:
    """Commissioning/decommissioning date; ``1800-01-01`` placeholders → None."""
    if not text:
        return None
    day = str(text)[:10]
    if day < "1900-01-01":
        return None
    return day


def _convert(variable: str, raw: Any) -> float | None:
    """Native text value → output units, with sentinel and plausibility nulling."""
    value = _to_float(raw)
    if value is None:
        return None
    info = VARIABLES[variable]
    value = round(value * info["scale"], 4)
    if info["type"] in ("swe", "snwd") and not (0.0 <= value <= _SNOW_MAX_CM):
        return None
    if info["type"] == "precip" and value < 0:
        return None
    if info["type"] == "rh" and not (0.0 <= value <= 100.0):
        return None
    return value


def _aggregate(method: str, values: list[float]) -> float | None:
    """Daily aggregation of valid hourly values (chronological order)."""
    if not values:
        return None
    if method == "last":
        return values[-1]
    if len(values) < MIN_HOURS_DAILY:
        return None
    if method == "sum":
        return round(sum(values), 3)
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if method == "mean":
        return round(sum(values) / len(values), 3)
    if method == "circmean":
        s = sum(math.sin(math.radians(v)) for v in values)
        c = sum(math.cos(math.radians(v)) for v in values)
        if abs(s) < 1e-9 and abs(c) < 1e-9:
            return None
        return round(math.degrees(math.atan2(s, c)) % 360.0, 1)
    raise HydroQuebecError(f"Unknown daily aggregation {method!r}")


def _station_code(name: str | None) -> str | None:
    """The 4-character code Hydro-Québec appends to met-station names, e.g. ``(CM3V)``."""
    if not name:
        return None
    m = _NAME_CODE_RE.search(name)
    return m.group(1) if m else None


def _quote_list(values: list[str]) -> str:
    return ", ".join('"' + v.replace('"', '\\"') + '"' for v in values)


def _build_stations(all_rows: list[dict], valid_rows: list[dict]) -> dict[str, dict]:
    """Merge two ``group_by`` exports into ``{station_id: station dict}``.

    ``all_rows``: every row per (station, variable), incl. junk/-999 rows.
    ``valid_rows``: rows with a timestamp and a non-sentinel value, with
    ``n_obs``, ``first`` and ``last``.
    """
    valid: dict[tuple[str, str], dict] = {
        (r["identifiant"], r["variable"]): r for r in valid_rows
    }
    stations: dict[str, dict] = {}
    for r in all_rows:
        sid = r.get("identifiant")
        var = r.get("variable")
        if not sid:
            continue
        sta = stations.get(sid)
        if sta is None:
            lon = _to_float(r.get("xcoord"))
            lat = _to_float(r.get("ycoord"))
            name = r.get("nom")
            end = _site_date(r.get("date_fin"))
            sta = {
                "station_id": sid,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": _to_float(r.get("zcoord")),
                "status": "Inactive",
                "region": r.get("regionqc"),
                "region_code": r.get("coderegionqc"),
                "hq_code": _station_code(name),
                "operator": "Hydro-Québec",
                "network": "Hydro-Québec hydrometeorological network",
                "measurement_method": None,
                "commissioned": _site_date(r.get("date_debut")),
                "decommissioned": end,
                "variables": {},
                "has_swe": False,
                "has_snwd": False,
                "begin_date": None,
                "end_date": None,
                "latest_observation": None,
                "license": LICENSE,
                "station_url": DATASET_PAGE,
            }
            stations[sid] = sta
        if var not in VARIABLES:
            continue
        v = valid.get((sid, var))
        sta["variables"][var] = {
            "n_obs": int(v["n_obs"]) if v else 0,
            "first": v["first"] if v else None,
            "last": v["last"] if v else None,
        }
        if var == _SWE:
            sta["has_swe"] = True
        if var == _SNWD:
            sta["has_snwd"] = True
    for sta in stations.values():
        firsts = [x["first"] for x in sta["variables"].values() if x["first"]]
        lasts = [x["last"] for x in sta["variables"].values() if x["last"]]
        if lasts:
            sta["begin_date"] = min(firsts)[:10]
            sta["end_date"] = max(lasts)[:10]
            sta["latest_observation"] = max(lasts)
        if lasts and sta["decommissioned"] is None:
            sta["status"] = "Active"
        if sta["has_swe"]:
            sta["measurement_method"] = "GMON passive gamma (SWE, 6-hourly)"
        elif sta["has_snwd"]:
            sta["measurement_method"] = "automatic snow depth (sensor unspecified)"
    return stations


# ── Client ───────────────────────────────────────────────────────────────────


class HydroQuebecClient:
    """
    Client for Hydro-Québec's open hydrometeorological feed (Opendatasoft).

    No credentials are needed.  The feed is a rolling ~10-day window of
    hourly data; see the module docstring for why history must be
    accumulated by the caller (or taken from CanSWE).  Data are licensed
    CC BY-NC 4.0 (non-commercial).

    Parameters
    ----------
    base_url : str
        Opendatasoft API root (``.../api/explore/v2.1``).
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session or None
        Optional pre-configured session.
    """

    def __init__(
        self,
        base_url: str = PORTAL_URL,
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
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._stations_cache: dict[str, dict] | None = None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _export(self, params: dict) -> list[dict]:
        url = f"{self.base_url}/catalog/datasets/{DATASET_ID}/exports/json"
        resp = request_with_retries(
            self._session,
            url,
            params=params,
            error_cls=HydroQuebecError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise HydroQuebecError(f"Non-JSON response from {url}: {exc}") from exc
        if isinstance(payload, dict):  # ODS error envelope
            raise HydroQuebecError(
                f"Hydro-Québec portal error: {payload.get('message', payload)}"
            )
        return payload

    # ── Stations ─────────────────────────────────────────────────────────────

    def _station_table(self) -> dict[str, dict]:
        if self._stations_cache is None:
            keys = (
                "identifiant,nom,xcoord,ycoord,zcoord,regionqc,coderegionqc,"
                f"date_debut,date_fin,{_F_VAR}"
            )
            select_base = keys.replace(_F_VAR, f"{_F_VAR} as variable")
            all_rows = self._export(
                {"select": f"{select_base},count(*) as n_rows", "group_by": keys}
            )
            valid_rows = self._export(
                {
                    "select": (
                        f"{select_base},count(*) as n_obs,"
                        "min(date) as first,max(date) as last"
                    ),
                    "where": 'date is not null and valeur != "-999.00"',
                    "group_by": keys,
                }
            )
            self._stations_cache = _build_stations(all_rows, valid_rows)
            logger.info("Hydro-Québec: %d stations", len(self._stations_cache))
        return self._stations_cache

    def get_stations(
        self,
        variables: list[str] | str | None = None,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Stations listing any of ``variables`` in the current window.

        Parameters
        ----------
        variables : list[str] or str or None
            Native labels or standardized types; ``None`` = SWE or snow
            depth.  Pass ``"all"`` for every station (incl. hydrometric).
        active_only : bool
            Keep stations with at least one valid value in the window.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (e.g. ``"1-10613"``), ``name``, ``latitude``,
            ``longitude``, ``elevation_m``, ``status``, ``region``,
            ``region_code``, ``hq_code`` (code in the name, e.g. ``CM3V``),
            ``operator``, ``network``, ``measurement_method``,
            ``commissioned``, ``decommissioned``, ``has_swe``, ``has_snwd``,
            ``variables`` (``{label: {n_obs, first, last}}``),
            ``begin_date``/``end_date`` (the window held, not the period of
            record), ``latest_observation``, ``license``, ``station_url``.
        """
        if variables == "all":
            wanted = None
        else:
            wanted = set(_resolve_variables(variables))
        out = []
        for sta in self._station_table().values():
            if wanted is not None and not wanted & set(sta["variables"]):
                continue
            if active_only and sta["status"] != "Active":
                continue
            out.append(dict(sta))
        return _filter_by_bbox(out, bbox)

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """All stations listing SWE or snow depth.

        ``status`` is ``"Active"`` when the station has at least one valid
        (non ``-999``) value of any variable in the rolling window.  See
        :meth:`get_stations` for the schema and for non-snow stations.
        """
        return self.get_stations(active_only=active_only, bbox=bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Metadata and variable inventory for one station.

        ``data_variables`` has one entry per variable the station lists,
        with ``begin_date``/``end_date``/``n_obs`` describing the
        **current rolling window** only (valid hourly values).
        """
        sta = self._station_table().get(station_id)
        if sta is None:
            raise HydroQuebecError(
                f"Station {station_id!r} not found in the Hydro-Québec feed"
            )
        meta = dict(sta)
        meta["data_variables"] = [
            {
                "name": var,
                "type": VARIABLES[var]["type"],
                "interval": "hourly",
                "units": VARIABLES[var]["output_units"],
                "description": VARIABLES[var]["description"],
                "notes": VARIABLES[var]["notes"],
                "begin_date": inv["first"][:10] if inv["first"] else None,
                "end_date": inv["last"][:10] if inv["last"] else None,
                "n_obs": inv["n_obs"],
            }
            for var, inv in sta["variables"].items()
        ]
        return meta

    # ── Data ─────────────────────────────────────────────────────────────────

    def get_raw_records(
        self,
        station_ids: list[str],
        variables: list[str],
        begin_utc: datetime | None = None,
        end_utc: datetime | None = None,
    ) -> list[dict]:
        """Raw portal rows (junk ``date = null`` rows excluded)."""
        where = [
            "date is not null",
            f"identifiant in ({_quote_list(station_ids)})",
            f"{_F_VAR} in ({_quote_list(variables)})",
        ]
        if begin_utc is not None:
            where.append(f'date >= "{_iso_utc(begin_utc)}"')
        if end_utc is not None:
            where.append(f'date < "{_iso_utc(end_utc)}"')
        return self._export(
            {
                "select": f"identifiant,{_F_VAR},{_F_KIND},{_F_UNIT},date,valeur",
                "where": " and ".join(where),
                "order_by": "date",
            }
        )

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "hourly",
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Standardized data fetch — a flat list of observation records.

        Only the last ~10 days exist; dates outside the window simply
        return nothing.  Accumulate history yourself (or use CanSWE).

        Parameters
        ----------
        station_ids : list[str] or str or None
            Hydro-Québec ``identifiant`` values (``"1-10613"``).  Required
            unless ``bbox`` is given.
        variables : list[str] or str or None
            Native French labels or standardized types (``"swe"``,
            ``"temp_max"``, …).  ``None`` = SWE and snow depth.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds — on the UTC date for
            ``"hourly"`` and on the America/Toronto local date for
            ``"daily"``.
        interval : str
            ``"hourly"`` (native; records carry ``datetime`` in UTC,
            ``YYYY-MM-DDTHH:MM:SSZ``, and ``date`` is its UTC date) or
            ``"daily"`` (local-day aggregation; see module docstring:
            SWE/depth = last valid value of the local day, precip = sum,
            Tmax = max, Tmin = min, others = mean, wind direction =
            circular mean; sum/mean/max/min need >= ``MIN_HOURS_DAILY``
            valid hours).
        include_flags : bool
            Add ``flag``: the native ``type_mesure`` for hourly records,
            ``None`` for daily ones.  The feed has no QC flags.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``datetime`` for hourly, + ``flag``).

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        HydroQuebecError
            On an unknown variable, unsupported interval or portal error.
        """
        interval = (interval or "").lower()
        if interval not in _SUPPORTED_INTERVALS:
            raise HydroQuebecError(
                f"Unsupported interval {interval!r} for Hydro-Québec — "
                f"expected one of {list(_SUPPORTED_INTERVALS)}."
            )
        wanted = _resolve_variables(variables)

        if station_ids is not None:
            ids = _coerce_list(station_ids)
        elif bbox is not None:
            ids = [
                s["station_id"] for s in self.get_stations(variables=wanted, bbox=bbox)
            ]
        else:
            raise ValueError("Provide station_ids or bbox.")
        if not ids:
            return []

        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        # Pad the UTC query by a day either side so local days are complete.
        begin_utc = (
            datetime.fromisoformat(begin).replace(tzinfo=UTC) - timedelta(days=1)
            if begin
            else None
        )
        end_utc = (
            datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=2)
            if end
            else None
        )
        id_set, var_set = set(ids), set(wanted)
        rows = [
            r
            for r in self.get_raw_records(ids, wanted, begin_utc, end_utc)
            if r.get("identifiant") in id_set and r.get(_F_VAR) in var_set
        ]
        if interval == "hourly":
            return _hourly_records(rows, begin, end, include_flags)
        return _daily_records(rows, begin, end, include_flags)


# ── Exception ────────────────────────────────────────────────────────────────


class HydroQuebecError(Exception):
    """Raised when the Hydro-Québec portal fails or a request is invalid."""


# ── Record builders ──────────────────────────────────────────────────────────


def _iter_rows(rows: list[dict]):
    """Yield ``(sid, variable, utc datetime, measure, value)`` for usable rows,
    de-duplicated on (sid, variable, datetime) preferring the variable's
    native ``measure`` (two sites carry both mean and instantaneous level)."""
    seen: dict[tuple[str, str, datetime], tuple[str, float | None]] = {}
    for r in rows:
        var = r.get(_F_VAR)
        if var not in VARIABLES:
            continue
        dt = _parse_utc(r.get("date"))
        if dt is None:
            continue
        sid = r.get("identifiant")
        key = (sid, var, dt)
        measure = r.get(_F_KIND)
        value = _convert(var, r.get("valeur"))
        if key in seen and seen[key][0] == VARIABLES[var]["measure"]:
            continue
        seen[key] = (measure, value)
    for (sid, var, dt), (measure, value) in sorted(
        seen.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        yield sid, var, dt, measure, value


def _hourly_records(
    rows: list[dict], begin: str | None, end: str | None, include_flags: bool
) -> list[dict]:
    out: list[dict] = []
    for sid, var, dt, measure, value in _iter_rows(rows):
        day = dt.date().isoformat()
        if (begin and day < begin) or (end and day > end):
            continue
        info = VARIABLES[var]
        rec: dict[str, Any] = {
            "station_id": sid,
            "date": day,
            "datetime": _iso_utc(dt),
            "variable": var,
            "type": info["type"],
            "value": value,
            "units": info["output_units"],
            "interval": "hourly",
        }
        if include_flags:
            rec["flag"] = measure
        out.append(rec)
    return out


def _daily_records(
    rows: list[dict], begin: str | None, end: str | None, include_flags: bool
) -> list[dict]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    order: list[tuple[str, str, str]] = []
    for sid, var, dt, _measure, value in _iter_rows(rows):
        day = dt.astimezone(_LOCAL_TZ).date().isoformat()
        if (begin and day < begin) or (end and day > end):
            continue
        key = (sid, var, day)
        if key not in groups:
            order.append(key)
            groups[key] = []
        if value is not None:
            groups[key].append(value)
    out: list[dict] = []
    for key in sorted(order):
        sid, var, day = key
        info = VARIABLES[var]
        rec: dict[str, Any] = {
            "station_id": sid,
            "date": day,
            "variable": var,
            "type": info["type"],
            "value": _aggregate(info["daily"], groups[key]),
            "units": info["output_units"],
            "interval": "daily",
        }
        if include_flags:
            rec["flag"] = None
        out.append(rec)
    return out


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Translate native labels and/or standardized types to native labels."""
    if variables is None:
        return list(_DEFAULT_VARS)
    raw = [variables] if isinstance(variables, str) else list(variables)
    if not raw:
        return list(_DEFAULT_VARS)
    out: list[str] = []
    for v in raw:
        if v in VARIABLES:
            candidates = [v]
        elif v in _TYPE_TO_VARS:
            candidates = _TYPE_TO_VARS[v]
        else:
            raise HydroQuebecError(
                f"Unknown variable {v!r} for Hydro-Québec — expected one of "
                f"{sorted(VARIABLES)} or a standardized type "
                f"({sorted(_TYPE_TO_VARS)})."
            )
        out.extend(c for c in candidates if c not in out)
    return out
