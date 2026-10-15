"""
clients/ceaza/ceaza_client.py
=============================
Python client for CEAZA-Met, the station network of the Centro de Estudios
Avanzados en Zonas Áridas (CEAZA, La Serena, Chile), through its open
monitoring-data web service.

Primary use case: automatic snow depth and snow water equivalent from the
high Andes of the Coquimbo region (30–32° S, 2,284–4,306 m): the
glacier, hydrology and snow stations El Tapado, Tascadero, Guandacol, Llano
de Las Liebres, La Laguna and others.  The same stations also measure air
temperature (and, at Tascadero, precipitation), which this client serves as
non-default variables.  The network has about 95 stations in all (coast,
valleys and islands included); by default the client lists only the 17 that
carry a snow sensor.

Data source       : https://www.ceazamet.cl/ws/pop_ws.php (``http://`` also
                    works; this client uses ``https://``)
Documentation     : https://www.ceazamet.cl/ws/pop_ws_doc.php
                    ("Documentación webservice [WS] Datos monitoreo
                    CEAZA-Met")
Authentication    : none needed.  The service takes an optional ``user``
                    parameter (an e-mail address); without it, queries "are
                    limited in the maximum number of records returned and in
                    the number of queries per day" (documentation page).
                    Pass ``user=`` once you have registered (see Licence).
Licence           : no licence is stated.  The documentation page's
                    "Condiciones de uso" (terms of use) say that users of
                    the system are required to request a user name by
                    e-mailing ceazamet@ceaza.cl with their name, e-mail,
                    organisation, sector, intended use of the data, download
                    frequency and amount of data downloaded per day, and
                    that the data source must be credited with a link (e.g.
                    "Datos provistos por el laboratorio de meteorología
                    CEAZA", linked to https://www.ceazamet.cl).  A
                    disclaimer states that the services come with no
                    guarantee of data reliability or availability and may
                    change at the author's discretion.  Every response
                    header repeats "Revisar condiciones de uso en
                    www.ceazamet.cl".

Functions used
--------------
All three are ``GET pop_ws.php?fn=<function>&p_cod=ceazamet&…``.

- ``GetListaEstaciones`` — station list, CSV.  ``c0``…``c9`` choose and
  order the columns (``e_cod, e_nombre, e_lat, e_lon, e_altitud,
  e_primera_lectura, e_ultima_lectura, e_cod_provincia, e_owner``); this
  client always names them, as the documentation recommends.
- ``GetListaSensores`` — sensor list, CSV.  Documented per station
  (``e_cod``), but without ``e_cod`` it returns every sensor of the network
  in one response (1,515 sensors on 99 station codes in October 2026; four
  codes are not in the station list).  Columns ``e_cod, s_cod, tf_nombre``
  (measured quantity), ``um_notacion`` (units), ``s_altura`` (sensor height,
  m), ``s_primera_lectura``, ``s_ultima_lectura``, ``tm_cod`` (measurement
  type code, the key of :data:`VARIABLES`).
- ``GetSerieSensor`` — one sensor's series between ``fecha_inicio`` and
  ``fecha_fin`` (both inclusive; ``YYYY-MM-DD``), ``interv=hora`` (default)
  or ``dia``.  This client asks for ``tipo_resp=json``: rows of
  ``{fecha, min, prom, max, data_pc, fecha_min, fecha_max}``.

Known quirks (checked October 2026)
-----------------------------------
- The **CSV** form of ``GetSerieSensor`` returns at most 24 rows per
  request without ``user`` (one day hourly, or 24 days daily), whatever the
  date range and also with the documentation's example ``user`` values.
  The **JSON** form returned full ranges (13 years daily for Tascadero
  snow depth, 4,748 rows, in 3.5 s).  The client uses JSON and splits
  requests into windows of 10 years (daily) or one year (hourly).
- Errors come back as HTTP 200: an unsupported ``interv`` gives an HTML
  ``<p>ERROR en lib_datos…</p>`` before the JSON (no ``serie``); an unknown
  sensor gives ``e_cod: null`` and a ``serie`` of empty rows.  Both raise
  :class:`CEAZAError`.
- The service returns a row for every interval of the request, including
  intervals with no data (``prom: null``, ``data_pc: 0``); those rows are
  dropped.
- Times are UTC−4 (``gmt-4``, the documented default, no daylight saving):
  a daily row is the aggregate of the hourly rows labelled 00:00–23:00 of
  that UTC−4 day.  The daily ``prom`` of Tascadero snow depth on 2025-07-01
  is exactly the mean of its 24 hourly values.  For precipitation
  (``pp_mm``) every aggregate is a **sum**: ``min``, ``prom`` and ``max``
  all hold the interval total (Tascadero 2025-06-12: 28.2 mm daily = the
  sum of its hourly rows).
- ``data_pc`` is the share (%) of the expected records present in the
  interval (daily: of 24 hours; hourly: of the logger's sub-hourly
  records).
- The values are raw logger output, not quality controlled.  Bare-ground
  snow depth is often slightly negative (zero offset; 875 of 1,952 daily
  Tascadero depths), snow-weight and SWE sensors drift below zero in summer
  (El Tapado ``TPFSWE_K`` down to −113 mm), and there are spikes (El Jote
  depth 4,021 cm, El Tapado air temperature −82.8 °C).  The plausibility
  bounds below null these; negatives are **not** set to zero.
- ``s_ultima_lectura`` is not always the last real value: El Tapado's
  ``TPFSNOWD`` lists 2025-10-08 but its last daily value is 2023-10-25.

Variables
---------
Keyed by ``tm_cod``.  Snow (default): ``ani_cm`` "Altura de Nieve" (depth,
cm, 18 sensors at 17 stations), ``ae_mm`` "Agua equivalente" (SWE, mm:
El Tapado ``TPFSWE_K`` and ``TPFSWE_TL``, Tascadero ``TASCPN``), ``ae_cm``
"Agua equivalente" (SWE, cm: Hualtatas ``HUALPN`` 2014–2019, La Laguna
``LAGELSP`` 2013–2017), ``pnkg`` "Peso de la Nieve" (snow weight, kg/m²,
i.e. mm of water: Guandacol ``HDCPN``, Llano de Las Liebres
``LLLSW1Wins1h`` and ``LLLSW2Wins1h``).  The units were checked against
the co-located depth sensors: the implied median snow densities are
190–370 kg/m³.  Non-default: ``ta_c`` (air temperature) and ``pp_mm``
(precipitation).  Not served: ``ppa_mm`` "Precipitación Acumulada" (a
cumulative gauge level, not a per-interval amount), ``acol_mm`` "Agua
Colectada", and ``an_mm`` "Agua Nieve" (only at Puerto Williams, sea level,
undocumented, values up to 7,999.7).

A station can have two sensors of one type (El Tapado: two depth and two
SWE sensors; Llano de Las Liebres: two snow-weight sensors), so every record
carries ``sensor`` (the ``s_cod``).  ``get_data(sensors=…)`` picks one.

Design principles
-----------------
- Metric-first: SWE and snow weight are returned in cm (mm and kg/m² ×
  0.1), depth in cm, temperature in °C, precipitation in mm.
- ``interval="daily"`` (``interv=dia``) and ``"hourly"`` (``interv=hora``);
  anything else raises :class:`CEAZAError`.  Hourly records carry
  ``datetime`` with the ``-04:00`` offset, labelled as the service labels
  them (the service does not document whether a label opens or closes its
  hour).
- ``value`` is the interval mean (``prom``; the total for precipitation).
  Records also carry ``value_min`` and ``value_max`` (the interval extremes,
  same units and bounds; ``None`` for precipitation) and ``data_pct``.
- ``include_flags=True`` adds ``flag``: "complete" (``data_pc`` = 100) or
  "partial" (1–99), see :data:`DATA_FLAGS`.
- Plausibility bounds: SWE and depth 0–15 m, air temperature −60 to 50 °C,
  precipitation 0–500 mm per interval; values outside are ``None``.
- ``status`` is "Active" when the station's latest reading
  (``e_ultima_lectura``) is at most 90 days old, otherwise "Inactive".
  High stations are often down for weeks in winter, hence the long window.
- Not wired into ``stations.networks``, the catalog or
  ``global_snow_networks``.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date, timedelta
from typing import Any

import requests

from .._common import coerce_list as _coerce_list
from .._common import date_str as _date_str
from .._common import filter_by_bbox as _filter_by_bbox
from .._common import request_with_retries
from .._common import to_float as _to_float

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.ceazamet.cl/ws/pop_ws.php"
DOC_URL = "https://www.ceazamet.cl/ws/pop_ws_doc.php"
HOME_URL = "https://www.ceazamet.cl/"
STATION_URL = (
    "https://www.ceazamet.cl/index.php?pag=mod_estacion&e_cod={}&p_cod=ceazamet"
)

#: The project (``p_cod``) this client queries.
PROJECT = "ceazamet"

#: Attribution text the terms of use ask for (with a link to HOME_URL).
ATTRIBUTION = "Datos provistos por el laboratorio de meteorología CEAZA"

#: Columns requested from GetListaEstaciones / GetListaSensores, in order.
STATION_COLUMNS = (
    "e_cod",
    "e_nombre",
    "e_lat",
    "e_lon",
    "e_altitud",
    "e_primera_lectura",
    "e_ultima_lectura",
    "e_cod_provincia",
    "e_owner",
)
SENSOR_COLUMNS = (
    "e_cod",
    "s_cod",
    "tf_nombre",
    "um_notacion",
    "s_altura",
    "s_primera_lectura",
    "s_ultima_lectura",
    "tm_cod",
)

#: Interval enum → ``interv`` code, and the request window per interval.
_INTERV: dict[str, str] = {"daily": "dia", "hourly": "hora"}
_WINDOW_DAYS: dict[str, int] = {"daily": 3650, "hourly": 365}

#: Earliest date the documentation allows for ``fecha_inicio``.
_EARLIEST = "2004-01-31"

#: Fixed offset of the service's timestamps (``gmt-4``).
UTC_OFFSET = "-04:00"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: A station whose latest reading is at most this old is Active.
_ACTIVE_DAYS = 90

#: Plausibility bounds per type, output units (DESIGN.md §3.6).
_BOUNDS: dict[str, tuple[float, float]] = {
    "swe": (0.0, 1500.0),  # cm
    "snwd": (0.0, 1500.0),  # cm
    "temp": (-60.0, 50.0),  # °C
    "precip": (0.0, 500.0),  # mm per interval
}

_SOURCE = "CEAZA-Met web service (GetSerieSensor) — " + BASE_URL

#: ``e_owner`` codes of the station list whose owner is named in the
#: station names or on the CEAZA-Met site.  Other codes (``uconce``,
#: ``ulagos``, ``upla``, ``jvre``) are kept as given in ``owner``.
OWNERS: dict[str, str] = {
    "ceaza": "Centro de Estudios Avanzados en Zonas Áridas (CEAZA)",
    "inia": "Instituto de Investigaciones Agropecuarias (INIA)",
    "dga": "Dirección General de Aguas (DGA)",
    "ieb": "Instituto de Ecología y Biodiversidad (IEB)",
    "uls": "Universidad de La Serena (ULS)",
}

# ── Public variable / flag tables ────────────────────────────────────────────

#: Variables served by :meth:`CEAZAClient.get_data`, keyed by ``tm_cod``.
VARIABLES: dict[str, dict] = {
    "ani_cm": {
        "name": "Altura de Nieve",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "source": _SOURCE,
        "description": "Snow depth from an automatic snow-depth sensor.",
        "notes": (
            "Raw logger values. Bare ground often reads slightly negative "
            "(zero offset); those values are nulled by the 0–15 m bound, not "
            "set to zero. El Tapado has two sensors (TPFSNOWD to 2023, "
            "TPFANSHM31 from 2018)."
        ),
    },
    "ae_mm": {
        "name": "Agua equivalente",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "source": _SOURCE,
        "description": (
            "Snow water equivalent from an automatic sensor (El Tapado "
            "TPFSWE_K and TPFSWE_TL, Tascadero TASCPN)."
        ),
        "notes": (
            "Published in mm; returned in cm (× 0.1). The sensor type is not "
            "given by the service. Summer values drift below zero and are "
            "nulled. El Tapado's two sensors disagree: median difference "
            "15 mm, largest 210 mm, on days when either reads over 50 mm."
        ),
    },
    "ae_cm": {
        "name": "Agua equivalente",
        "type": "swe",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "source": _SOURCE,
        "description": (
            "Snow water equivalent, published in cm (Hualtatas HUALPN "
            "2014–2019, La Laguna LAGELSP 2013–2017)."
        ),
        "notes": (
            "Same quantity as ae_mm but served in cm; the unit is confirmed "
            "by the co-located depth sensors (median density about 260 "
            "kg/m³ at Hualtatas)."
        ),
    },
    "pnkg": {
        "name": "Peso de la Nieve",
        "type": "swe",
        "units": "kg/m²",
        "output_units": "cm",
        "scale": 0.1,
        "source": _SOURCE,
        "description": (
            "Snow weight per unit area (Guandacol HDCPN, Llano de Las "
            "Liebres LLLSW1Wins1h and LLLSW2Wins1h), i.e. SWE: 1 kg/m² = "
            "1 mm of water."
        ),
        "notes": (
            "Published in kg/m²; returned as SWE in cm (× 0.1). The two "
            "Llano de Las Liebres sensors disagree: on days when either "
            "reads over 10 kg/m² the median difference is 32 kg/m² and the "
            "largest 182 kg/m² (2024-08-02)."
        ),
    },
    "ta_c": {
        "name": "Temperatura del Aire",
        "type": "temp",
        "units": "°C",
        "output_units": "°C",
        "scale": 1.0,
        "source": _SOURCE,
        "description": "Air temperature; value is the interval mean.",
        "notes": (
            "value_min / value_max are the interval extremes. Values outside "
            "−60 to 50 °C are nulled."
        ),
    },
    "pp_mm": {
        "name": "Precipitación",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "source": _SOURCE,
        "description": "Precipitation total over the interval (the service sums).",
        "notes": (
            "Only Tascadero (TASCPP) among the snow stations. The gauge "
            "type is not given by the service. A partial day (data_pct < "
            "100) undercounts. Not the cumulative ppa_mm gauge level, "
            "which is not served."
        ),
    },
}

#: Flag text from the service's ``data_pc`` (share of expected records).
DATA_FLAGS: dict[str, str] = {
    "complete": "data_pc = 100: every expected record of the interval was present",
    "partial": (
        "data_pc 1–99: some records of the interval were missing; the "
        "percentage is in data_pct (a partial precipitation total undercounts)"
    ),
}

_TYPE_TO_VARS: dict[str, list[str]] = {}
for _code, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_code)

#: Measurement types that make a station a snow station.
SNOW_VARIABLES: tuple[str, ...] = ("ani_cm", "ae_mm", "ae_cm", "pnkg")
_SWE_VARS = frozenset(_TYPE_TO_VARS["swe"])

_DEFAULT_VARS: list[str] = list(SNOW_VARIABLES)


class CEAZAError(Exception):
    """Raised for CEAZA-Met web-service failures and bad requests."""


# ── Parsers ──────────────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Map ``tm_cod`` codes or standardized types to ``tm_cod`` codes."""
    items = _coerce_list(variables) if variables is not None else []
    if not items:
        return list(_DEFAULT_VARS)
    out: list[str] = []
    for v in items:
        key = v.strip().lower()
        if key in VARIABLES:
            codes = [key]
        elif key in _TYPE_TO_VARS:
            codes = _TYPE_TO_VARS[key]
        else:
            raise CEAZAError(
                f"Unknown CEAZA-Met variable {v!r} — expected one of "
                f"{sorted(VARIABLES)} or types {sorted(_TYPE_TO_VARS)}."
            )
        for c in codes:
            if c not in out:
                out.append(c)
    return out


def _convert(code: str, raw: Any) -> float | None:
    """Native value → output units, nulling missing and implausible values."""
    value = _to_float(raw)
    if value is None:
        return None
    info = VARIABLES[code]
    value = round(value * info["scale"], 3)
    low, high = _BOUNDS[info["type"]]
    if not low <= value <= high:
        return None
    return value


def _text(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def parse_csv(text: str) -> list[dict]:
    """Parse a CEAZA-Met CSV response into row dicts.

    Lines starting with ``#`` are metadata; the last of them holds the
    column names (with a trailing comma, as does every data row).
    """
    header: list[str] | None = None
    body: list[str] = []
    for line in text.lstrip("﻿").splitlines():
        if line.startswith("#"):
            header = [h.strip() for h in line[1:].split(",")]
        elif line.strip():
            body.append(line)
    if header is None:
        raise CEAZAError(f"CEAZA-Met response has no header line: {text[:200]!r}")
    while header and not header[-1]:
        header.pop()
    rows: list[dict] = []
    for fields in csv.reader(body):
        rows.append(
            {h: (fields[i] if i < len(fields) else "") for i, h in enumerate(header)}
        )
    return rows


def parse_series(text: str) -> dict:
    """Parse a ``GetSerieSensor`` JSON response.

    Raises
    ------
    CEAZAError
        On an error message from the service, non-JSON text, or an unknown
        sensor (``e_cod`` null).
    """
    body = text.lstrip("﻿").strip()
    start = body.find("{")
    if start != 0:
        message = body[: start if start > 0 else 300]
        raise CEAZAError(f"CEAZA-Met error: {message.strip()[:300]}")
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise CEAZAError(f"CEAZA-Met returned non-JSON: {body[:200]!r}") from exc
    if not payload.get("e_cod"):
        raise CEAZAError(f"Unknown CEAZA-Met sensor {payload.get('s_cod')!r}")
    if "serie" not in payload:
        raise CEAZAError(
            f"CEAZA-Met response for {payload.get('s_cod')!r} has no 'serie'"
        )
    return payload


def _day(stamp: str | None) -> str | None:
    """``"2016-10-29 20:00:00"`` → ``"2016-10-29"``."""
    text = (stamp or "").strip()
    return text[:10] if text else None


def _parse_station(row: dict) -> dict:
    """GetListaEstaciones row → station dict (no sensor fields yet)."""
    sid = row["e_cod"].strip()
    owner = _text(row.get("e_owner"))
    return {
        "station_id": sid,
        "name": _text(row.get("e_nombre")),
        "latitude": _to_float(row.get("e_lat")),
        "longitude": _to_float(row.get("e_lon")),
        "elevation_m": _to_float(row.get("e_altitud")),
        "network": "CEAZA-Met",
        "owner": owner,
        "operator": OWNERS.get(owner or "", owner),
        "province_code": _text(row.get("e_cod_provincia")),
        "first_reading": _text(row.get("e_primera_lectura")),
        "last_reading": _text(row.get("e_ultima_lectura")),
        "begin_date": _day(row.get("e_primera_lectura")),
        "end_date": _day(row.get("e_ultima_lectura")),
        "station_url": STATION_URL.format(sid),
    }


def _parse_sensor(row: dict) -> dict:
    """GetListaSensores row → sensor dict."""
    return {
        "station_id": row["e_cod"].strip(),
        "sensor": row["s_cod"].strip(),
        "variable": _text(row.get("tm_cod")),
        "name": _text(row.get("tf_nombre")),
        "units": _text(row.get("um_notacion")),
        "height_m": _to_float(row.get("s_altura")),
        "begin_date": _day(row.get("s_primera_lectura")),
        "end_date": _day(row.get("s_ultima_lectura")),
    }


def _method(variables: set[str]) -> str:
    parts = []
    if "ani_cm" in variables:
        parts.append("automatic snow depth")
    if variables & {"ae_mm", "ae_cm"}:
        parts.append("automatic SWE sensor")
    if "pnkg" in variables:
        parts.append("snow-weight sensor (SWE)")
    return "; ".join(parts) or "automatic weather station"


def _windows(begin: str, end: str, days: int) -> list[tuple[str, str]]:
    """Split ``[begin, end]`` into inclusive windows of at most ``days``."""
    out: list[tuple[str, str]] = []
    start = date.fromisoformat(begin)
    last = date.fromisoformat(end)
    while start <= last:
        stop = min(start + timedelta(days=days - 1), last)
        out.append((start.isoformat(), stop.isoformat()))
        start = stop + timedelta(days=1)
    return out


# ── Client ───────────────────────────────────────────────────────────────────


class CEAZAClient:
    """
    Client for the CEAZA-Met web service (Coquimbo region, Chile).

    No credentials are needed; see the module docstring for the terms of
    use (registration by e-mail, attribution).

    Parameters
    ----------
    user : str, optional
        The e-mail address registered with CEAZA-Met, sent as ``user=``.
        Without it the service applies its anonymous limits.
    base_url : str
        Web-service URL.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session, optional
        Pre-configured session.
    """

    def __init__(
        self,
        user: str | None = None,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self.user = user
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._stations: list[dict] | None = None
        self._sensors: dict[str, list[dict]] | None = None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _get(self, fn: str, **params: Any) -> str:
        query: dict[str, Any] = {"fn": fn, "p_cod": PROJECT, **params}
        if self.user:
            query["user"] = self.user
        resp = request_with_retries(
            self._session,
            self.base_url,
            params=query,
            error_cls=CEAZAError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    @staticmethod
    def _columns(names: tuple[str, ...]) -> dict[str, str]:
        return {f"c{i}": name for i, name in enumerate(names)}

    def _load(self) -> None:
        """Station list and network-wide sensor list (two requests, cached)."""
        if self._stations is not None:
            return
        stations = [
            _parse_station(r)
            for r in parse_csv(
                self._get("GetListaEstaciones", **self._columns(STATION_COLUMNS))
            )
            if (r.get("e_cod") or "").strip()
        ]
        sensors: dict[str, list[dict]] = {}
        for r in parse_csv(
            self._get("GetListaSensores", **self._columns(SENSOR_COLUMNS))
        ):
            if (r.get("e_cod") or "").strip() and (r.get("s_cod") or "").strip():
                sen = _parse_sensor(r)
                sensors.setdefault(sen["station_id"], []).append(sen)
        if not stations:
            raise CEAZAError("CEAZA-Met returned an empty station list")
        cutoff = (date.today() - timedelta(days=_ACTIVE_DAYS)).isoformat()
        for sta in stations:
            own = sensors.get(sta["station_id"], [])
            kinds = {s["variable"] for s in own}
            served = [s for s in own if s["variable"] in VARIABLES]
            sta.update(
                {
                    "status": (
                        "Active"
                        if sta["end_date"] and sta["end_date"] >= cutoff
                        else "Inactive"
                    ),
                    "measurement_method": _method(kinds),
                    "has_swe": bool(kinds & _SWE_VARS),
                    "has_snwd": "ani_cm" in kinds,
                    "has_temp": "ta_c" in kinds,
                    "has_precip": "pp_mm" in kinds,
                    "is_snow_station": bool(kinds & set(SNOW_VARIABLES)),
                    "snow_sensors": [
                        s["sensor"] for s in own if s["variable"] in SNOW_VARIABLES
                    ],
                    "snow_variables": sorted(kinds & set(SNOW_VARIABLES)),
                    "n_sensors": len(own),
                    "served_sensors": [s["sensor"] for s in served],
                }
            )
        self._stations = stations
        self._sensors = sensors
        logger.info(
            "CEAZA-Met: %d stations (%d with snow sensors)",
            len(stations),
            sum(s["is_snow_station"] for s in stations),
        )

    # ── Public API — stations ────────────────────────────────────────────────

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        snow_only: bool = True,
    ) -> list[dict]:
        """
        CEAZA-Met stations; by default only those with a snow sensor.

        Parameters
        ----------
        active_only : bool
            Keep stations whose latest reading is at most 90 days old.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        snow_only : bool
            Keep stations with a snow-depth, SWE or snow-weight sensor
            (default).  ``False`` lists the whole network.

        Returns
        -------
        list[dict]
            ``station_id`` (``e_cod``), ``name``, ``latitude``,
            ``longitude``, ``elevation_m``, ``status``, ``network``
            (``"CEAZA-Met"``), ``owner`` (``e_owner`` code), ``operator``
            (from :data:`OWNERS`, else the code), ``province_code``,
            ``first_reading`` / ``last_reading`` (as served) and
            ``begin_date`` / ``end_date`` (their dates),
            ``measurement_method``, ``has_swe``, ``has_snwd``,
            ``has_temp``, ``has_precip``, ``is_snow_station``,
            ``snow_sensors`` (``s_cod`` list), ``snow_variables``
            (``tm_cod`` list), ``n_sensors``, ``served_sensors`` and
            ``station_url``.

        Raises
        ------
        CEAZAError
            If the service cannot be reached or returns nothing.
        """
        self._load()
        assert self._stations is not None
        out = [dict(s) for s in self._stations]
        if snow_only:
            out = [s for s in out if s["is_snow_station"]]
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    def get_sensors(self, station_id: str) -> list[dict]:
        """
        Every sensor of a station, as the network-wide sensor list gives it.

        Returns
        -------
        list[dict]
            ``station_id``, ``sensor`` (``s_cod``), ``variable``
            (``tm_cod``), ``name`` (``tf_nombre``), ``units``
            (``um_notacion``), ``height_m``, ``begin_date``, ``end_date``.
        """
        self._load()
        assert self._sensors is not None
        return [dict(s) for s in self._sensors.get(str(station_id).strip(), [])]

    def get_metadata(self, station_id: str) -> dict:
        """
        Station metadata plus ``data_variables`` and ``sensors``.

        Parameters
        ----------
        station_id : str
            ``e_cod`` (e.g. ``"TPF"``, ``"TASC"``); any station, snow or
            not.

        Returns
        -------
        dict
            The station dict plus ``data_variables`` (one entry per sensor
            this client serves: ``name`` (``tm_cod``), ``sensor``,
            ``native_name``, ``type``, ``units`` (native),
            ``output_units``, ``height_m``, ``begin_date``, ``end_date``,
            ``intervals``) and ``sensors`` (every sensor, see
            :meth:`get_sensors`).

        Raises
        ------
        CEAZAError
            If the station is not in the station list.
        """
        sid = str(station_id).strip()
        match = [
            s for s in self.get_all_stations(snow_only=False) if s["station_id"] == sid
        ]
        if not match:
            raise CEAZAError(f"Station {sid!r} not in the CEAZA-Met station list")
        sta = match[0]
        sensors = self.get_sensors(sid)
        sta["sensors"] = sensors
        sta["data_variables"] = [
            {
                "name": s["variable"],
                "sensor": s["sensor"],
                "native_name": s["name"],
                "type": VARIABLES[s["variable"]]["type"],
                "units": s["units"],
                "output_units": VARIABLES[s["variable"]]["output_units"],
                "height_m": s["height_m"],
                "begin_date": s["begin_date"],
                "end_date": s["end_date"],
                "intervals": list(_INTERV),
            }
            for s in sensors
            if s["variable"] in VARIABLES
        ]
        return sta

    # ── Public API — data ────────────────────────────────────────────────────

    def get_series(
        self,
        sensor: str,
        begin_date: str | date,
        end_date: str | date,
        interval: str = "daily",
    ) -> dict:
        """
        Raw ``GetSerieSensor`` JSON for one sensor and one window.

        Returns
        -------
        dict
            The service's payload (``e_cod``, ``s_cod``, ``tf_nombre``,
            ``um_notacion``, ``serie``, …).

        Raises
        ------
        CEAZAError
            On an unsupported interval, an unknown sensor or a service error.
        """
        if interval not in _INTERV:
            raise CEAZAError(
                f"Unsupported interval {interval!r} for CEAZA-Met — expected "
                f"one of {sorted(_INTERV)}."
            )
        text = self._get(
            "GetSerieSensor",
            s_cod=sensor,
            fecha_inicio=_date_str(begin_date),
            fecha_fin=_date_str(end_date),
            interv=_INTERV[interval],
            tipo_resp="json",
        )
        return parse_series(text)

    def _record(
        self, sid: str, sensor: str, code: str, row: dict, interval: str, flags: bool
    ) -> dict | None:
        stamp = (row.get("fecha") or "").strip()
        if not stamp or row.get("prom") is None:
            return None  # an interval without data
        info = VARIABLES[code]
        precip = info["type"] == "precip"
        pct = row.get("data_pc")
        try:
            pct = int(pct) if pct is not None else None
        except (TypeError, ValueError):
            pct = None
        rec: dict[str, Any] = {
            "station_id": sid,
            "date": stamp[:10],
        }
        if interval != "daily":
            rec["datetime"] = stamp.replace(" ", "T") + UTC_OFFSET
        rec.update(
            {
                "variable": code,
                "type": info["type"],
                "value": _convert(code, row.get("prom")),
                "units": info["output_units"],
                "interval": interval,
                "sensor": sensor,
                "value_min": None if precip else _convert(code, row.get("min")),
                "value_max": None if precip else _convert(code, row.get("max")),
                "data_pct": pct,
            }
        )
        if flags:
            rec["flag"] = "complete" if pct == 100 else "partial" if pct else None
        return rec

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "daily",
        include_flags: bool = False,
        sensors: list[str] | str | None = None,
    ) -> list[dict]:
        """
        Station data as flat records, one per sensor and interval with data.

        Parameters
        ----------
        station_ids : list[str] or str or None
            ``e_cod`` code(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            ``tm_cod`` codes (``"ani_cm"``, ``"ae_mm"``, ``"ae_cm"``,
            ``"pnkg"``, ``"ta_c"``, ``"pp_mm"``) or types (``"snwd"``,
            ``"swe"``, ``"temp"``, ``"precip"``).  ``None`` = the snow
            variables (``snwd`` + ``swe``).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``; selects snow stations.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds (UTC−4 days).  Default: each
            sensor's first reading to today.
        interval : str
            ``"daily"`` (default) or ``"hourly"``.
        include_flags : bool
            Add ``flag`` ("complete" / "partial", see :data:`DATA_FLAGS`).
        sensors : list[str] or str, optional
            Keep only these sensor codes (``s_cod``), e.g. ``"TPFSWE_K"``.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval,
            sensor, value_min, value_max, data_pct}`` (+ ``datetime`` for
            hourly, ``flag``).  ``value`` is the interval mean (precipitation:
            the total) and is ``None`` outside the plausibility bounds.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        CEAZAError
            On an unknown variable, station or sensor, an unsupported
            interval, or a service error.

        Example
        -------
        >>> client = CEAZAClient()
        >>> recs = client.get_data(
        ...     "TASC", variables="snwd",
        ...     begin_date="2025-08-01", end_date="2025-08-31",
        ... )
        >>> recs[0]["units"], recs[0]["sensor"]
        ('cm', 'TASCAN')
        """
        interval = str(interval).lower()
        if interval not in _INTERV:
            raise CEAZAError(
                f"Unsupported interval {interval!r} for CEAZA-Met — expected "
                f"one of {sorted(_INTERV)}."
            )
        codes = _resolve_variables(variables)
        if station_ids is not None:
            ids = [str(s).strip() for s in _coerce_list(station_ids)]
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        self._load()
        assert self._stations is not None and self._sensors is not None
        known = {s["station_id"] for s in self._stations}
        unknown = [sid for sid in ids if sid not in known]
        if unknown:
            raise CEAZAError(f"Station(s) {unknown} not in the CEAZA-Met station list")
        wanted = {s.strip() for s in _coerce_list(sensors)} if sensors else None
        if wanted:
            have = {s["sensor"] for sid in ids for s in self._sensors.get(sid, [])}
            if wanted - have:
                raise CEAZAError(
                    f"Sensor(s) {sorted(wanted - have)} not at station(s) {ids}"
                )
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else date.today().isoformat()
        records: list[dict] = []
        for sid in ids:
            for sen in self._sensors.get(sid, []):
                code = sen["variable"]
                if code not in codes or (wanted and sen["sensor"] not in wanted):
                    continue
                lo = max(begin or "", sen["begin_date"] or _EARLIEST)
                hi = min(end, sen["end_date"] or end)
                if lo > hi:
                    continue
                for w_begin, w_end in _windows(lo, hi, _WINDOW_DAYS[interval]):
                    payload = self.get_series(sen["sensor"], w_begin, w_end, interval)
                    for row in payload["serie"]:
                        rec = self._record(
                            sid, sen["sensor"], code, row, interval, include_flags
                        )
                        if rec is not None:
                            records.append(rec)
        return records
