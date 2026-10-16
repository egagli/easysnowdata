"""
clients/snih/snih_client.py
===========================
Python client for Argentina's Sistema Nacional de Información Hídrica
(SNIH) and its Base de Datos Hidrológica Integrada, the national
hydrometeorological data portal of the Secretaría de Infraestructura y
Política Hídrica (SIPH): https://snih.hidricosargentina.gob.ar

Primary use case: snow water equivalent from the 34 snow pillows along the
Andes of Argentina (Jujuy to Santa Cruz, 24–52° S) of the Red Hidrológica
Nacional (24) and the Red Meteorológica RMET (10).  Validated daily
records reach back to 1989 (Toscas, Laguna Atuel), 1990 (Valle Hermoso,
Laguna Diamante) and 1999 (Paso Pehuenche) at the five Mendoza pillows;
the same five sites have snow-course SWE, depth and density from 1951
(latest surveys 2008–2024); and the pillows' unvalidated sub-daily
telemetry runs up to the current day (29 of 34 reported in the 30 days to
2026-10-15).  The portal serves 1,441 hydrometeorological stations (1,439
distinct codes); this client is scoped to the **50 snow stations**, those
the service associates with a snow parameter (codes 70–74).

Data source       : https://snih.hidricosargentina.gob.ar (ASP.NET page
                    methods behind the public map; no documented API)
Authentication    : none.  The web map asks visitors to tick "Acepto los
                    Términos y Condiciones de Uso"; the page methods
                    themselves need no cookie or token.
Licence           : the portal's "Términos y condiciones de uso"
                    (https://snih.hidricosargentina.gob.ar/Terminos.aspx,
                    read October 2026), not a named standard licence.
                    Section 4: the information is "de carácter público,
                    libre y gratuito".  Section 7: any use is allowed,
                    including commercial use, derived works and
                    redistribution "sin ninguna restricción", on condition
                    that users keep the original meaning of the data, cite
                    the source (SIPH and/or the contributing networks),
                    state the date of the last update, and keep the source,
                    date and conditions metadata.  Section 8: the providing
                    agency is responsible for the content; SIPH disclaims
                    liability.  The start page adds that real-time
                    ("tiempo útil") values are raw, unvalidated sensor
                    records.

Status            : not wired into ``stations.networks``, the catalog or
                    ``global_snow_networks``.

Endpoints used
--------------
Every call is ``POST <base>/<Page>.aspx/<Method>`` with
``Content-Type: application/json``; the answer is ``{"d": ...}``.
Timestamps are .NET ``"/Date(<ms since epoch>)/"`` instants.

- ``Filtros.aspx/LeerEstaciones`` (body ``{}``): every station —
  ``Codigo``, ``Descripcion``, ``Latitud`` / ``Longitud`` (decimal degrees,
  **published positive**: Argentina is entirely south and west, so both are
  negated), ``Cota`` (m; ``-999`` = unknown), ``Red`` (network id),
  ``Provincia``, ``Cuenca``, ``Sistema``, ``Regional``, ``Rio``, ``Lugar``,
  ``Tipo`` (letters: H hidrométrico, M meteorológico, G geohidrométrico,
  A calidad de agua, N nivométrico), ``Transmision`` (``T`` telemetric;
  the portal treats anything else as conventional) and ``Habilitada``.
- ``Filtros.aspx/leerListaRedes``: network id → name.
- ``MuestraDatos.aspx/LeerListaAsociaciones``: station ↔ parameter-code
  pairs with ``Desde`` (start), ``Minimo`` / ``Maximo`` (range limits).
  This is what defines the snow-station scope.
- ``MuestraDatos.aspx/LeerDatosHistoricos`` (``fechaDesde``,
  ``fechaHasta``, ``estacion``, ``codigo``, ``validados``): the historical
  (validated) series — ``FechaHora``, ``Medicion``, ``Calificador``,
  ``Validado``.  ``fechaHasta`` is an instant at 00:00, so a value later
  that day is excluded; the client asks for one day more and filters.
  The full 1900–2026 range of one series comes back in about 2 s (Toscas
  SWE: 5,318 values).
- ``MuestraDatos.aspx/LeerUltimosRegistros`` (``fechaDesde``,
  ``fechaHasta``, ``estacion``, ``codigo``): raw telemetry, one row per
  transmission (every 4 or 6 h) — ``FechaHora`` and ``Mediciones``
  (``[{"Codigo": 73, "Valor": ...}]``).

Snow parameter codes (``MuestraDatos.aspx/LeerCodigosMedicion``, ``Tipo`` 5,
"Nivométrico"): 70 "Espesor de nieve" (cm), 71 "Long. media del núcleo
cateador" (cm), 72 "Densidad media de nieve" (g/cm3), 73 "Altura E.A.N. en
Snow Pillow" (mm) and 74 "E.A.N. resultante de cateos" (mm).  E.A.N. is
"equivalente en agua de nieve", snow water equivalent.  There is no other
snow parameter in the 179-code list.

Known quirks (October 2026)
---------------------------
- **Validated daily pillow SWE stops in 2023–2024** at every station; the
  ten pillows of the Red Meteorológica RMET (2024–2026) have no
  historical values at all.  Everything after that exists only as raw
  telemetry (``interval="sub_daily"``).
- ``LeerUltimosRegistros`` returns only one contiguous block of records per
  call: when the logger configuration changes inside the requested range
  (Toscas: 6-hourly until 2025-09-27, then 4-hourly from 2026-05-02) the
  answer stops at the end of the first block, and a range that starts long
  before a station's first record can come back empty (Las Lágrimas,
  2009-01-01 to 2026-10-15: no rows; its record starts 2021-04-27).  It
  also returns up to a day past ``fechaHasta``.  The client asks in
  windows of at most a year, asks again from the last day received
  whenever an answer stops early, and filters to the requested dates.
  Checked against 30-day windows at seven stations: no record missed.
- The ``validados`` flag of ``LeerDatosHistoricos`` made no difference for
  any snow station checked ("true" and "false" gave byte-identical
  answers), and every historical value carried ``Validado: true``.  The
  portal itself sends ``"true"`` only for networks 1, 21 and 22; this
  client always sends ``"true"``.
- ``Calificador`` was always blank (``""`` or ``" "``) on snow values.
- 17 days of the Toscas daily record (2005 and 2008) carry two values;
  at Túnel Internacional and La Hoya (2020) pairs at 00:00 and 00:01
  differ.  ``"daily"`` keeps the earliest value of each local day.
- Raw telemetry is noisy: step changes of exactly 10 mm, long runs of 0,
  and spikes (Laguna Diamante reached 14,370 mm, Las Leñas 4,007 mm).  The
  only clamp is SWE outside 0–15 m.  Lago Bagillt (282337) reports values
  near 0.2, possibly a different unit; this is not documented by the
  service.
- 16 river and weather stations are associated with code 70 (snow depth)
  but returned no value from either endpoint.  They stay in the inventory
  (``has_snwd`` follows the association) and are always "Inactive".
- ``Cota`` is ``-999`` or ``0`` at 22 of the 50 snow stations (e.g. Túnel
  Internacional, a pillow near 3,200 m, has 0); both become ``None``.
  Cerro Piltriquitrón's 999 m is kept as published.
- Two snow-associated codes (151492, 151495) are not in the station list
  and are left out.

Design principles
-----------------
- Metric-first: SWE is published in mm and returned in cm (× 0.1); snow
  depth and core length in cm; density in g/cm³, returned in % (× 100).
- One source per interval, never mixed: ``"daily"`` is the validated
  historical pillow SWE (``LeerDatosHistoricos``); ``"sub_daily"`` is the
  raw telemetry (``LeerUltimosRegistros``), with a ``datetime`` in
  Argentina time (UTC−3) and flag ``"raw"``; ``"periodic"`` is the
  snow-course survey (SWE, depth, density, core length), also from the
  historical table.  Raw values are never used to fill the validated daily
  series.
- Dates are the local day in Argentina time (UTC−3, ``ART``).  Argentina
  has been on UTC−3 in winter since 1969 and all year since 2009; for the
  Toscas record the fixed offset and the IANA ``America/Argentina/Mendoza``
  zone give the same day for every value.
- ``station_id`` is the SNIH ``Codigo`` as a string.
- ``status`` is "Active" when the station's raw pillow telemetry (code 73)
  holds a record in the last 30 days, else "Inactive" (all snow-course and
  snow-depth-only stations).  The portal's own ``LeerDatosActuales``
  "current value" is not used: it returned a placeholder (1900-01-01,
  −999) for Toscas while its telemetry was current.
- ``operator`` is the network name the service gives (``leerListaRedes``):
  "Red Hidrológica Nacional" (40 snow stations) or "Red Meteorológica
  RMET" (10).  The agencies that run individual stations are not stated
  by the service and are not guessed.
- There is no per-station page: the portal opens a station from its map
  through cookies, so ``station_url`` is the portal itself.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import requests

from .._common import coerce_list as _coerce_list
from .._common import date_str as _date_str
from .._common import filter_by_bbox as _filter_by_bbox
from .._common import request_with_retries
from .._common import to_float as _to_float

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://snih.hidricosargentina.gob.ar"
TERMS_URL = BASE_URL + "/Terminos.aspx"

EP_STATIONS = "Filtros.aspx/LeerEstaciones"
EP_NETWORKS = "Filtros.aspx/leerListaRedes"
EP_ASSOCIATIONS = "MuestraDatos.aspx/LeerListaAsociaciones"
EP_HISTORICAL = "MuestraDatos.aspx/LeerDatosHistoricos"
EP_RECENT = "MuestraDatos.aspx/LeerUltimosRegistros"

#: Argentina time.  Timestamps are converted to it before taking the date.
ART = timezone(timedelta(hours=-3), "ART")
#: The Unix epoch, for /Date(ms)/ values (some are before 1970).
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: A station is Active when its raw pillow telemetry has a record this recent.
_ACTIVE_DAYS = 30

#: Parallel requests used for the status check (one per pillow station).
_STATUS_WORKERS = 4

#: No raw telemetry exists before this (earliest record seen: 2009-02-19).
_RAW_START = date(2009, 1, 1)

#: Longest range asked of ``LeerUltimosRegistros`` in one go.  A 17-year
#: range at Las Lágrimas (11485) came back empty; one-year windows returned
#: every record that 30-day windows did at the seven stations compared.
_RAW_WINDOW_DAYS = 366

#: Values below this are the portal's no-data marker (its ``NoDataMin``).
_NO_DATA_BELOW = -998.999

#: Plausibility bounds per emitted type (DESIGN.md: SWE/depth 0–15 m).
_BOUNDS: dict[str, tuple[float, float]] = {
    "swe": (0.0, 1500.0),
    "snwd": (0.0, 1500.0),
    "density": (0.0, 100.0),
}

_SOURCE = "SNIH — Sistema Nacional de Información Hídrica (SIPH, Argentina) — "

#: ``Provincia`` code → name, as the portal's own filter list has it
#: (``Script/Filtros.js``, ``leerProvincias``).
PROVINCES: dict[int, str] = {
    1: "Jujuy",
    2: "Catamarca",
    3: "Tucumán",
    4: "Salta",
    5: "Santiago del Estero",
    6: "La Rioja",
    7: "San Juan",
    8: "Mendoza",
    9: "San Luis",
    10: "Río Negro",
    11: "Neuquén",
    12: "Chubut",
    13: "Buenos Aires",
    14: "Formosa",
    15: "Santa Cruz",
    16: "Santa Fe",
    17: "Tierra del Fuego",
    18: "Misiones",
    19: "La Pampa",
    20: "Corrientes",
    21: "Chaco",
    22: "Córdoba",
    23: "Entre Ríos",
}

#: ``Tipo`` letters, as the portal's "Tipo de Parámetro" filter names them.
STATION_TYPES: dict[str, str] = {
    "H": "Hidrométrico",
    "M": "Meteorológico",
    "G": "Geohidrométrico",
    "A": "Calidad de Agua",
    "N": "Nivométrico",
}

# ── Public variable / flag tables ────────────────────────────────────────────

#: Snow variables, keyed by a native name; ``code`` is the SNIH parameter
#: code and ``abbreviation`` its ``Abreviatura`` (both accepted as aliases).
VARIABLES: dict[str, dict] = {
    "SWE_PILLOW": {
        "name": "Altura E.A.N. en Snow Pillow",
        "code": 73,
        "abbreviation": "h_spilow",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "intervals": ("daily", "sub_daily"),
        "source": _SOURCE + EP_HISTORICAL + " (daily), " + EP_RECENT + " (sub_daily)",
        "description": (
            "Snow water equivalent from a snow pillow (parameter 73). "
            "'daily': the validated historical record; 'sub_daily': raw "
            "telemetry every 4 or 6 hours."
        ),
        "notes": (
            "Published in mm; returned in cm (x 0.1). Validated daily values "
            "end in 2023-2024; later values are raw telemetry only, which is "
            "noisy (10 mm steps, runs of 0, spikes)."
        ),
    },
    "SWE_COURSE": {
        "name": "E.A.N. resultante de cateos",
        "code": 74,
        "abbreviation": "EAN_Cat",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "intervals": ("periodic",),
        "source": _SOURCE + EP_HISTORICAL,
        "description": (
            "Snow water equivalent from a manual snow-course survey "
            "(cateo, snow coring), parameter 74."
        ),
        "notes": (
            "Published in mm; returned in cm (x 0.1). A few surveys in some "
            "winters at the five Mendoza courses from 1951 (Toscas: 35 "
            "values, 1951-2008); single values at three other stations."
        ),
    },
    "SNWD": {
        "name": "Espesor de nieve",
        "code": 70,
        "abbreviation": "EspNieve",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "intervals": ("periodic",),
        "source": _SOURCE + EP_HISTORICAL,
        "description": "Snow depth from a snow-course survey (parameter 70).",
        "notes": (
            "Published and returned in cm. Values exist only at the five "
            "Mendoza snow courses; the other stations associated with code "
            "70 returned none."
        ),
    },
    "DENSITY": {
        "name": "Densidad media de nieve",
        "code": 72,
        "abbreviation": "DensNie",
        "type": "density",
        "units": "g/cm3",
        "output_units": "%",
        "scale": 100.0,
        "intervals": ("periodic",),
        "source": _SOURCE + EP_HISTORICAL,
        "description": (
            "Mean snow density of a snow-course survey (parameter 72), "
            "consistent with SWE / depth."
        ),
        "notes": "Published in g/cm3 (0.19-0.70 seen); returned in % (x 100).",
    },
    "CORE_LENGTH": {
        "name": "Long. media del núcleo cateador",
        "code": 71,
        "abbreviation": "LongNucl",
        "type": "other",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "intervals": ("periodic",),
        "source": _SOURCE + EP_HISTORICAL,
        "description": (
            "Mean length of the snow core taken by the sampling tube in a "
            "snow-course survey (parameter 71)."
        ),
        "notes": "Not snow depth: the core is usually shorter than the snowpack.",
    },
}

_CODE_TO_VAR: dict[int, str] = {info["code"]: k for k, info in VARIABLES.items()}

#: Snow codes that place a station in scope (71 never occurs without them).
_SCOPE_CODES = frozenset({70, 72, 73, 74})

#: Qualifier codes (``Calificador``) as the portal renders them
#: (``Script/MuestraDatos.js``: ``ValorCalificadorATexto`` and
#: ``TextoCalificador``), plus two flags this client sets.  Only blank
#: qualifiers occurred on snow values in October 2026.
DATA_FLAGS: dict[str, str] = {
    "D": "Dudoso — doubtful value",
    "T": "Desbordado — overflow",
    "A": "Acumulado — accumulated value",
    "C": "Congelado — frozen",
    "S": "Río seco — dry river",
    "<": "Value is an upper bound (less than)",
    ">": "Value is a lower bound (greater than)",
    "&": "Shown by the portal as 'VNC' without a value (not expanded there)",
    "@": "Shown by the portal as 'ND' without a value (not expanded there)",
    ")": "Shown by the portal as 'LDND' without a value (not expanded there)",
    "%": "Shown by the portal as 'SD' without a value (not expanded there)",
    "(": "Shown by the portal as 'NC' without a value (not expanded there)",
    "unvalidated": "Historical value served with Validado = false",
    "raw": (
        "Unvalidated raw telemetry record (interval 'sub_daily', LeerUltimosRegistros)"
    ),
}

#: Qualifiers the portal displays as text only; their value is not shown.
_TEXT_ONLY_FLAGS = frozenset({"&", "@", ")", "%", "("})

_INTERVALS = ("daily", "sub_daily", "periodic")

#: ``variables=None``: the snow (SWE + depth) variables of each interval.
_DEFAULT_VARS: dict[str, list[str]] = {
    "daily": ["SWE_PILLOW"],
    "sub_daily": ["SWE_PILLOW"],
    "periodic": ["SWE_COURSE", "SNWD"],
}


class SNIHError(Exception):
    """Raised for SNIH portal failures and bad requests."""


# ── Parsers ──────────────────────────────────────────────────────────────────


def _lookup(name: str) -> str | None:
    """Native name, code or abbreviation → native name (case-insensitive)."""
    key = name.strip()
    for var, info in VARIABLES.items():
        if key.upper() == var or key == str(info["code"]):
            return var
        if key.lower() == info["abbreviation"].lower():
            return var
    return None


def _resolve_variables(variables: list[str] | str | None, interval: str) -> list[str]:
    """Map native names, codes or standardized types to native names.

    Raises
    ------
    SNIHError
        On an unknown name, or a variable the interval does not offer.
    """
    items = _coerce_list(variables) if variables is not None else []
    if not items:
        return list(_DEFAULT_VARS[interval])
    types = {info["type"] for info in VARIABLES.values()}
    out: list[str] = []
    for v in items:
        var = _lookup(v)
        if var is not None:
            names = [var]
        elif v.strip().lower() in types:
            names = [
                k
                for k, info in VARIABLES.items()
                if info["type"] == v.strip().lower() and interval in info["intervals"]
            ]
            if not names:
                raise SNIHError(
                    f"No SNIH {v!r} variable at interval {interval!r}; "
                    + ", ".join(
                        f"{k}: {'/'.join(i['intervals'])}"
                        for k, i in VARIABLES.items()
                        if i["type"] == v.strip().lower()
                    )
                )
        else:
            raise SNIHError(
                f"Unknown SNIH variable {v!r} — expected one of "
                f"{sorted(VARIABLES)}, a parameter code "
                f"({sorted(i['code'] for i in VARIABLES.values())}) or a type "
                f"({sorted(types)})."
            )
        for name in names:
            if interval not in VARIABLES[name]["intervals"]:
                raise SNIHError(
                    f"SNIH variable {name} is not offered at interval "
                    f"{interval!r}; it is served as "
                    f"{list(VARIABLES[name]['intervals'])}."
                )
            if name not in out:
                out.append(name)
    return out


def parse_ms_date(text: Any) -> datetime | None:
    """``"/Date(1435762800000)/"`` → aware datetime in Argentina time."""
    s = str(text or "").strip()
    if not (s.startswith("/Date(") and s.endswith(")/")):
        return None
    body = s[6:-2]
    # A "+hhmm" suffix may follow the milliseconds; the instant is UTC anyway.
    for sep in ("+", "-"):
        idx = body.find(sep, 1)
        if idx > 0:
            body = body[:idx]
    try:
        ms = int(body)
    except ValueError:
        return None
    # Epoch + offset, not fromtimestamp: Windows rejects negative (pre-1970)
    # timestamps, and the snow courses go back to 1951.
    return (_EPOCH + timedelta(milliseconds=ms)).astimezone(ART)


def _convert(raw: Any, var: str) -> float | None:
    """Native value → emitted units, or ``None`` (no-data or implausible)."""
    v = _to_float(raw)
    if v is None or v < _NO_DATA_BELOW:
        return None
    info = VARIABLES[var]
    out = round(v * info["scale"], 3)
    lo, hi = _BOUNDS.get(info["type"], (float("-inf"), float("inf")))
    if not lo <= out <= hi:
        return None
    return out


def _coord(value: Any) -> float | None:
    """Published positive degrees → negative (south / west), ``None`` if 0."""
    v = _to_float(value)
    if v is None or v == 0:
        return None
    return round(-abs(v), 6)


def _elevation(value: Any) -> float | None:
    """``Cota`` → metres; ``-999`` and ``0`` (both used for unknown) → None."""
    v = _to_float(value)
    if v is None or v <= 0:
        return None
    return v


def _method(codes: set[int]) -> str:
    parts = []
    if 73 in codes:
        parts.append("Snow pillow (telemetered SWE)")
    if codes & {71, 72, 74}:
        parts.append("Snow course (manual coring)")
    elif 70 in codes:
        parts.append("Snow depth (method not stated by the service)")
    return "; ".join(parts)


def parse_station(
    row: dict,
    networks: dict[int, str],
    assoc: dict[int, str],
) -> dict:
    """Turn a ``LeerEstaciones`` row into a station dict (no ``status`` yet).

    ``assoc`` maps the station's snow codes to their association start date.
    """
    codes = set(assoc)
    red = row.get("Red")
    tipo = (row.get("Tipo") or "").strip()
    prov = row.get("Provincia")
    trans = (row.get("Transmision") or "").strip()
    variables = [_CODE_TO_VAR[c] for c in sorted(codes) if c in _CODE_TO_VAR]
    return {
        "station_id": str(row["Codigo"]),
        "name": (row.get("Descripcion") or "").strip() or None,
        "latitude": _coord(row.get("Latitud")),
        "longitude": _coord(row.get("Longitud")),
        "elevation_m": _elevation(row.get("Cota")),
        "network_id": red,
        "network": networks.get(red),
        "operator": networks.get(red),
        "data_provider": (
            "Sistema Nacional de Información Hídrica (SNIH), Secretaría de "
            "Infraestructura y Política Hídrica, Argentina"
        ),
        "country": "AR",
        "province": PROVINCES.get(prov),
        "province_code": prov,
        "basin_code": row.get("Cuenca"),
        "system_code": row.get("Sistema"),
        "regional_code": row.get("Regional"),
        "river": (row.get("Rio") or "").strip() or None,
        "place": (row.get("Lugar") or "").strip() or None,
        "station_type": tipo or None,
        "station_type_description": ", ".join(
            STATION_TYPES[c] for c in tipo if c in STATION_TYPES
        )
        or None,
        "transmission": "telemetric" if trans == "T" else "conventional",
        "transmission_code": trans or None,
        "snow_codes": sorted(codes),
        "variables": variables,
        "variables_since": {_CODE_TO_VAR[c]: assoc[c] for c in sorted(codes)},
        "measurement_method": _method(codes),
        "has_swe": bool(codes & {73, 74}),
        "has_snwd": 70 in codes,
        "station_url": BASE_URL + "/",
    }


def _check(d: Any, what: str) -> dict:
    """Check a ``RespuestaMediciones`` payload and return it."""
    if not isinstance(d, dict):
        raise SNIHError(f"Unexpected SNIH answer to {what}: {str(d)[:200]}")
    if str(d.get("RespuestaOK")).lower() != "true":
        raise SNIHError(f"SNIH {what} failed: {d.get('MsgErr') or d}")
    return d


def parse_historical(d: Any) -> list[tuple[datetime, Any, str, bool]]:
    """``LeerDatosHistoricos`` payload → ``(time, value, qualifier, validated)``."""
    out = []
    for m in _check(d, "LeerDatosHistoricos").get("Mediciones") or []:
        ts = parse_ms_date(m.get("FechaHora"))
        if ts is None:
            continue
        cal = str(m.get("Calificador") or "").strip()
        out.append((ts, m.get("Medicion"), cal, bool(m.get("Validado", True))))
    out.sort(key=lambda r: r[0])
    return out


def parse_recent(d: Any, code: int) -> list[tuple[datetime, Any]]:
    """``LeerUltimosRegistros`` payload → ``(time, value)`` for one code."""
    out = []
    for row in _check(d, "LeerUltimosRegistros").get("Mediciones") or []:
        ts = parse_ms_date(row.get("FechaHora"))
        if ts is None:
            continue
        for m in row.get("Mediciones") or []:
            if m.get("Codigo") == code:
                out.append((ts, m.get("Valor")))
    out.sort(key=lambda r: r[0])
    return out


def _today() -> date:
    return datetime.now(ART).date()


# ── Client ───────────────────────────────────────────────────────────────────


class SNIHClient:
    """
    Client for the SNIH snow stations of Argentina.

    No credentials are needed.  By using the data you accept the portal's
    terms of use (:data:`TERMS_URL`): cite SIPH and the contributing network,
    keep the original meaning and state the date of the last update.

    Parameters
    ----------
    base_url : str
        Portal root.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session, optional
        Pre-configured session.
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
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._stations: list[dict] | None = None
        self._last_raw: dict[str, str | None] | None = None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _post(self, endpoint: str, body: dict | None = None) -> Any:
        """POST a page method and return its ``d`` member."""
        url = f"{self.base_url}/{endpoint}"
        resp = request_with_retries(
            self._session,
            url,
            method="POST",
            json=body or {},
            error_cls=SNIHError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SNIHError(
                f"Non-JSON response from {url}: {resp.text[:200]!r}"
            ) from exc
        if not isinstance(payload, dict) or "d" not in payload:
            raise SNIHError(f"Unexpected payload from {url}: {str(payload)[:200]}")
        return payload["d"]

    # ── Stations ─────────────────────────────────────────────────────────────

    def _snow_stations(self) -> list[dict]:
        """Snow stations without ``status`` (three requests, cached)."""
        if self._stations is not None:
            return self._stations
        rows = self._post(EP_STATIONS) or []
        networks = {
            r["Codigo"]: (r.get("Descripcion") or "").strip()
            for r in self._post(EP_NETWORKS) or []
        }
        assoc: dict[int, dict[int, str]] = {}
        for a in self._post(EP_ASSOCIATIONS) or []:
            code = a.get("Codigo")
            if code not in _CODE_TO_VAR:
                continue
            since = parse_ms_date(a.get("Desde"))
            day = since.date().isoformat() if since else None
            codes = assoc.setdefault(a.get("Estacion"), {})
            if code not in codes or (day and day < (codes[code] or day)):
                codes[code] = day
        out: list[dict] = []
        seen: set[int] = set()
        for row in rows:
            sid = row.get("Codigo")
            codes = assoc.get(sid)
            if sid in seen or not codes or not set(codes) & _SCOPE_CODES:
                continue  # the list repeats two (non-snow) codes
            seen.add(sid)
            out.append(parse_station(row, networks, codes))
        missing = sorted(
            s for s, c in assoc.items() if set(c) & _SCOPE_CODES and s not in seen
        )
        if missing:
            logger.info("SNIH: snow codes for unlisted stations %s ignored", missing)
        logger.info("SNIH: %d snow stations of %d", len(out), len(rows))
        self._stations = out
        return out

    def _latest_raw(self, station_id: str) -> str | None:
        """Time of the newest raw pillow record in the last 30 days, or None."""
        end = _today()
        rows = self._recent(station_id, 73, end - timedelta(days=_ACTIVE_DAYS), end)
        return rows[-1][0].isoformat() if rows else None

    def _statuses(self) -> dict[str, str | None]:
        if self._last_raw is None:
            pillows = [
                s["station_id"] for s in self._snow_stations() if 73 in s["snow_codes"]
            ]
            with ThreadPoolExecutor(_STATUS_WORKERS) as pool:
                latest = list(pool.map(self._latest_raw, pillows))
            self._last_raw = dict(zip(pillows, latest))
        return self._last_raw

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Every SNIH station associated with a snow parameter (codes 70–74).

        The first call reads the station list, the network names and the
        station–parameter associations (about 5 MB), then checks the last
        30 days of raw telemetry at each pillow station for ``status`` (one
        request each, four in parallel).  Both are cached on the client.

        Parameters
        ----------
        active_only : bool
            Keep stations whose pillow reported in the last 30 days.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (SNIH ``Codigo``), ``name``, ``latitude``,
            ``longitude`` (negated from the published positive values),
            ``elevation_m`` (``None`` where ``Cota`` is −999 or 0),
            ``status``, ``last_raw_obs`` (newest raw pillow record in the
            last 30 days, ISO time, or ``None``), ``network_id``,
            ``network`` / ``operator`` (the network name), ``data_provider``,
            ``country``, ``province`` / ``province_code``, ``basin_code``,
            ``system_code``, ``regional_code``, ``river``, ``place``,
            ``station_type`` (``Tipo`` letters) and its description,
            ``transmission`` / ``transmission_code``, ``snow_codes``,
            ``variables``, ``variables_since`` (association start per
            variable), ``measurement_method``, ``has_swe``, ``has_snwd`` and
            ``station_url``.

        Raises
        ------
        SNIHError
            On a portal failure.
        """
        status = self._statuses()
        out = []
        for s in self._snow_stations():
            last = status.get(s["station_id"])
            out.append(
                {**s, "status": "Active" if last else "Inactive", "last_raw_obs": last}
            )
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Station metadata plus ``data_variables`` (with period of record).

        Parameters
        ----------
        station_id : str
            SNIH station code (e.g. ``"19145"``, Toscas).

        Returns
        -------
        dict
            The station dict plus ``data_variables``: one entry per variable
            and interval with ``name``, ``code``, ``type``, ``interval``,
            ``units`` (emitted), ``description``, ``notes``,
            ``associated_since`` (the association start the service gives),
            and ``begin_date`` / ``end_date`` / ``n_obs``.  For ``"daily"``
            and ``"periodic"`` those come from the full historical series
            (``n_obs`` counts days for ``"daily"``); for ``"sub_daily"``
            ``end_date`` is the newest raw record of the last 30 days and
            ``begin_date`` / ``n_obs`` are ``None`` (counting would mean
            reading the whole raw record).

        Raises
        ------
        SNIHError
            If the station is not a SNIH snow station, or on a portal failure.
        """
        sid = str(station_id).strip()
        match = [s for s in self._snow_stations() if s["station_id"] == sid]
        if not match:
            raise SNIHError(f"Station {sid!r} is not a SNIH snow station")
        sta = dict(match[0])
        last = self._latest_raw(sid) if 73 in sta["snow_codes"] else None
        sta["status"] = "Active" if last else "Inactive"
        sta["last_raw_obs"] = last
        entries = []
        for var in sta["variables"]:
            info = VARIABLES[var]
            for interval in info["intervals"]:
                entry = {
                    "name": var,
                    "code": info["code"],
                    "type": info["type"],
                    "interval": interval,
                    "units": info["output_units"],
                    "description": info["description"],
                    "notes": info["notes"],
                    "associated_since": sta["variables_since"].get(var),
                }
                if interval == "sub_daily":
                    entry.update(
                        begin_date=None,
                        end_date=last[:10] if last else None,
                        n_obs=None,
                    )
                else:
                    recs = self._series(sid, var, interval, None, None, False)
                    entry.update(
                        begin_date=recs[0]["date"] if recs else None,
                        end_date=recs[-1]["date"] if recs else None,
                        n_obs=len(recs),
                    )
                entries.append(entry)
        sta["data_variables"] = entries
        return sta

    # ── Data ─────────────────────────────────────────────────────────────────

    def _historical(
        self, sid: str, code: int, begin: date, end: date
    ) -> list[tuple[datetime, Any, str, bool]]:
        body = {
            "fechaDesde": begin.isoformat(),
            # fechaHasta is an instant at 00:00: ask one day more, filter below.
            "fechaHasta": (end + timedelta(days=1)).isoformat(),
            "estacion": sid,
            "codigo": str(code),
            "validados": "true",
        }
        rows = parse_historical(self._post(EP_HISTORICAL, body))
        return [r for r in rows if begin <= r[0].date() <= end]

    def _recent_call(
        self, sid: str, code: int, begin: date, end: date
    ) -> list[tuple[datetime, Any]]:
        body = {
            "fechaDesde": begin.isoformat(),
            "fechaHasta": end.isoformat(),
            "estacion": sid,
            "codigo": str(code),
        }
        rows = parse_recent(self._post(EP_RECENT, body), code)
        return [r for r in rows if begin <= r[0].date() <= end]

    def _recent_window(
        self,
        sid: str,
        code: int,
        begin: date,
        end: date,
        found: dict[datetime, Any],
    ) -> None:
        """Read ``begin..end`` block by block (see the module notes)."""
        cursor = begin
        while cursor <= end:
            rows = self._recent_call(sid, code, cursor, end)
            if not rows:
                return
            new = [ts for ts, _ in rows if ts not in found]
            for ts, value in rows:
                found.setdefault(ts, value)
            last_day = rows[-1][0].date()
            if last_day >= end:
                return
            # The answer stopped early: a block ended.  Ask again from that
            # day (a new block may start later the same day), then move on.
            cursor = (
                last_day if new and last_day > cursor else last_day + timedelta(days=1)
            )

    def _recent(
        self, sid: str, code: int, begin: date, end: date
    ) -> list[tuple[datetime, Any]]:
        """Raw telemetry for ``begin..end`` (local dates), in time order."""
        found: dict[datetime, Any] = {}
        start = max(begin, _RAW_START)
        step = timedelta(days=_RAW_WINDOW_DAYS)
        while start <= end:
            stop = min(start + step - timedelta(days=1), end)
            self._recent_window(sid, code, start, stop, found)
            start = stop + timedelta(days=1)
        return sorted(found.items())

    def _series(
        self,
        sid: str,
        var: str,
        interval: str,
        begin: date | None,
        end: date | None,
        include_flags: bool,
    ) -> list[dict]:
        info = VARIABLES[var]
        code = info["code"]
        end = end or _today()
        records: list[dict] = []

        def record(ts: datetime, value: Any, flag: str | None) -> dict:
            rec: dict[str, Any] = {
                "station_id": sid,
                "date": ts.date().isoformat(),
                "variable": var,
                "type": info["type"],
                "value": value,
                "units": info["output_units"],
                "interval": interval,
            }
            if interval == "sub_daily":
                rec["datetime"] = ts.isoformat()
            if include_flags:
                rec["flag"] = flag
            return rec

        if interval == "sub_daily":
            for ts, raw in self._recent(sid, code, begin or _RAW_START, end):
                records.append(record(ts, _convert(raw, var), "raw"))
            return records

        rows = self._historical(sid, code, begin or date(1900, 1, 1), end)
        days: set[str] = set()
        for ts, raw, cal, validated in rows:
            day = ts.date().isoformat()
            if interval == "daily":
                if day in days:
                    continue  # keep the earliest value of the local day
                days.add(day)
            value = None if cal in _TEXT_ONLY_FLAGS else _convert(raw, var)
            flags = [
                f for f in (cal or None, None if validated else "unvalidated") if f
            ]
            records.append(record(ts, value, ",".join(flags) or None))
        return records

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
        Snow data as flat records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            SNIH station code(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            Native names (:data:`VARIABLES`), parameter codes (``"73"``),
            abbreviations or types.  ``None`` = the SWE and depth variables
            of ``interval``: ``SWE_PILLOW`` for ``"daily"`` and
            ``"sub_daily"``, ``SWE_COURSE`` + ``SNWD`` for ``"periodic"``.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive local (Argentina) dates.  Default: the whole record
            (raw telemetry: from 2009) to today.
        interval : str
            ``"daily"`` (validated pillow SWE), ``"sub_daily"`` (raw pillow
            telemetry) or ``"periodic"`` (snow-course surveys).
        include_flags : bool
            Add ``flag``: the ``Calificador`` qualifier (see
            :data:`DATA_FLAGS`; ``None`` when blank), ``"unvalidated"`` for
            a historical value served with ``Validado = false``, and
            ``"raw"`` on every ``"sub_daily"`` record.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``datetime`` in Argentina time for ``"sub_daily"``, + ``flag``).
            An id the portal does not know returns no records.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        SNIHError
            On an unsupported interval, an unknown variable, a variable the
            interval does not offer, a non-numeric station id or a portal
            failure.

        Example
        -------
        >>> client = SNIHClient()
        >>> records = client.get_data(
        ...     station_ids="19145",  # Toscas
        ...     begin_date="2015-06-01",
        ...     end_date="2015-09-30",
        ... )
        >>> records[0]["units"], records[0]["interval"]
        ('cm', 'daily')
        """
        interval = str(interval).lower()
        if interval not in _INTERVALS:
            raise SNIHError(
                f"Unsupported interval {interval!r} for SNIH — expected one of "
                f"{list(_INTERVALS)}."
            )
        names = _resolve_variables(variables, interval)
        if station_ids is not None:
            ids = [s.strip() for s in _coerce_list(station_ids)]
        elif bbox is not None:
            ids = [
                s["station_id"] for s in _filter_by_bbox(self._snow_stations(), bbox)
            ]
        else:
            raise ValueError("Provide station_ids or bbox.")
        for sid in ids:
            if not sid.isdigit():
                raise SNIHError(f"SNIH station codes are integers; got {sid!r}")
        begin = date.fromisoformat(_date_str(begin_date)) if begin_date else None
        end = date.fromisoformat(_date_str(end_date)) if end_date else None
        records: list[dict] = []
        for sid in ids:
            for var in names:
                records.extend(
                    self._series(sid, var, interval, begin, end, include_flags)
                )
        return records
