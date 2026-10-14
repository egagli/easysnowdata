"""
clients/rihmi/rihmi_client.py
=============================
Python client for the Russian snow-survey and snow-cover station archives
of ВНИИГМИ-МЦД (RIHMI-WDC, the All-Russian Research Institute of
Hydrometeorological Information – World Data Center, Obninsk, meteo.ru),
plus the public NSIDC copy of the Soviet snow-course record.

Two RIHMI datasets are covered:

1. **«Маршрутные снегомерные съемки»** (route snow surveys, i.e. snow
   courses) — http://meteo.ru/data/snow-surveys/.  517 Russian
   meteorological stations, 1966 to present.  A survey is made every 10 days
   through the cold season and every 5 days during intense melt, separately
   on a *field* route (2 km, depth every 20 m, density every 200 m), a
   *forest* route (1 km, depth every 10 m, density every 100 m) and, on
   request of the hydromet observatory only, in *ravines/gullies* (балки).
   Each record has 19 fields (see :data:`ROUTE_SURVEY_FIELDS`).  Database
   registration № 2013620279 (Булыгина, Разуваев, Александрова).
2. **«Характеристики снежного покрова на метеорологических станциях России
   и бывшего СССР»** (daily snow cover at met stations) —
   http://meteo.ru/data/snow-cover/.  620 stations of Russia and the former
   USSR; daily snow depth on a fixed stake plus the snow-cover degree of the
   station surroundings and three quality codes (see
   :data:`SNOW_COVER_FIELDS`).  Database registration № 2014621201.

Data access
-----------
**RIHMI data need a (free) login.**  Both datasets are served only through
Web АИСОРИ-М (http://aisori-m.meteo.ru/waisori/index.xhtml?idata=6 for route
surveys, ``idata=7`` for daily snow cover), a JSF/PrimeFaces web app whose
pages say «Пожалуйста, авторизуйтесь».  meteo.ru offers no bulk download,
FTP or API for these datasets, and no documented programmatic access, so this
client does **not** log in or scrape АИСОРИ.  Instead:

- ``RIHMIClient(data_dir=...)`` / ``get_data(files=...)`` parse files a user
  exported from АИСОРИ-М by hand (see *Manual export* below).
- ``interval="periodic"`` (route surveys) additionally reads the **public,
  no-login NSIDC copy**: *Former Soviet Union Hydrological Snow Surveys,
  1966–1996*, NSIDC G01170 v1 (Krenke 1998, updated 2004,
  doi:10.7265/N58C9T60), https://noaadata.apps.nsidc.org/NOAA/G01170/.  It
  is a different digitisation of the same Soviet snow-course observations
  (keyed at the Institute of Geography RAS under A. Krenke), 1,345 stations
  1966–1990 and ~300 of them 1991–1996.  It is fetched live and cached via
  ``easysnowdata.config.cache_dir("stations", "rihmi")``.
- Daily snow cover (``interval="daily"``) has **no** public no-login copy
  that this client knows of — it is available only from local АИСОРИ
  exports.  (GHCN-Daily carries daily SNWD for many of the same WMO stations
  as ``RSM000xxxxx``; use the ``ghcnd`` client for that.)

Manual export (АИСОРИ-М)
------------------------
Checked against the live site on 2026-10-14 (free account).

1. Register at http://aisori-m.meteo.ru/waisori/registration.xhtml (login,
   password, name, e-mail, country, city, organisation; no confirmation
   step), then log in through «Переход на страницу авторизации».
2. «Выбор данных» → «Раздел БД: Сутки» → «источник данных»:
   ``SNMAR - Маршрутные снегосъемки`` (route surveys) or
   ``SNOW - Снежный покров`` (daily snow cover).  Tick stations in the left
   list and move them right with «>», then «Дальше».  (In October 2026 the
   SNMAR source offered 68 of the 517 catalogue stations.)
3. Choose the delimiter («Пробел» or «Точка с запятой»), the year range
   («Год», От/До), press «Все» to take every parameter, then «Результат» and
   «Получить результат».
4. On the result page press «Загрузить»: a ``.zip`` holding ``wr<id>.txt``
   (the data), ``fld<id>.txt`` (columns) and ``statlist<id>.txt``.
5. Point ``RIHMIClient(data_dir="...")`` at the folder holding the zips (or
   their extracted files), or pass ``get_data(files=[...])``.

If credentials are ever automated, an auth provider (like
``easysnowdata/auth/nve.py``) would need the АИСОРИ username/password and the
JSF session flow above (``jsessionid`` cookie + ``jakarta.faces.ViewState``
per form post).

Record layouts and what is verified
-----------------------------------
- **NSIDC G01170** ``*.syn`` files — *verified live*.  One file per survey
  date (``YYYYMMDD.syn``), one line per station and route, 17
  whitespace-separated columns (see :data:`NSIDC_SYN_COLUMNS`).  ``-2`` means
  missing (documented); ``-1`` also occurs (~87 000 times in the ice-crust
  cover column, ~7 300 times in SWE, mostly with depth ≤ 4 cm) and is *not
  documented* — it is treated as missing.  Latitude/longitude in
  ``station.loc`` and the ``.syn`` files are **degrees+minutes packed as
  DDMM / DDDMM** (Moscow ``5545 3734`` = 55°45′N 37°34′E), not "value / 100"
  as the NSIDC user guide says.  Survey dates are every ~5 days (not only the
  10th/20th/last day as the guide says of the ``.trn`` files).  A handful of
  files list a station+route twice; the first line wins.
- **RIHMI daily snow cover** — the field order, widths and missing codes
  (``9999`` depth, ``99`` cover degree, ``Q1=9``) are documented on meteo.ru
  for the institute's ``IIIII.dat`` fixed-width files.
- **RIHMI route surveys** — the 19-field order and units come from the
  meteo.ru description, but *missing-value codes are not documented*; this
  client treats blank fields and ``9999``/``-9999`` as missing and nulls
  values outside physical bounds.
- **АИСОРИ-М export files** — *verified against real exports* of both
  datasets, in both delimiter variants (2026-10-14; trimmed copies are the
  test fixtures in ``tests/stations/fixtures/rihmi_aisori/``).  Values are
  right-aligned in fixed widths with a missing value written as blanks.  The
  ``;`` variant separates fields with ``;``; the space variant is **fixed
  width** (one space between fields), so it cannot be split on whitespace.
  ``fld<id>.txt`` lists each column's width and Russian name
  (:data:`AISORI_COLUMN_NAMES`), which is how a selection of only some
  parameters is read; its widths are not always exact (the route survey's two
  one-character code columns are printed two wide), so space-variant fields
  are matched to columns by right edge.  Route-survey exports carry no flag
  columns even with «Включать признаки» ticked; snow-cover exports always
  carry ``Q1``–``Q3``.  :func:`read_aisori_export` reads the downloaded zip
  directly.

Station catalogues
------------------
The two RIHMI station catalogues exist only as PDFs on meteo.ru.  They were
converted with ``pdftotext -layout`` plus a regex parser into the packaged
``route_survey_stations.csv`` (517 rows) and ``snow_cover_stations.csv``
(620 rows) next to this module (provenance is in each file's header).  PDF
quirks: coordinates printed in four styles (``71°58′``, ``65о54’``,
``8037``, symbol-font glyphs); ``61°419′`` and ``59°234′`` are taken as
41.9′ and 23.4′; two route stations have no elevation.  NSIDC's
``station.loc`` (1,345 stations, Latin transliterated names) is fetched
live.  Station ids are 5-digit WMO indexes; the union is 1,552 stations.

Status
------
``"Active"`` = listed in the route-survey catalogue (an ongoing dataset,
last updated on АИСОРИ-М in Jan 2025 through Oct 2024) or listed in the
snow-cover catalogue without a «закрыта»/«законсервирована» note.  All
other stations (NSIDC-only, or marked closed) are ``"Inactive"``.  This is
a catalogue-based judgement: whether a given station still reports is not
verifiable without a login.

Terms of use
------------
- NSIDC G01170: "freely, openly, and fully available to use without
  restrictions, provided that you cite the data" (NSIDC DIF
  ``Use_Constraints``); cite Krenke (1998), doi:10.7265/N58C9T60.
- RIHMI: the АИСОРИ-М footer says the data are placed on the institute's
  official site «в открытом доступе» (in open access); meteo.ru gives an
  official bibliographic reference per dataset (above) but publishes no
  licence or redistribution terms that could be found.  Treat RIHMI exports
  as citable but of unknown redistribution status.

Design principles
-----------------
- Metric-first: SWE and total water storage in cm (from mm), depths in cm,
  density in % (g cm⁻³ × 100), cover degrees in % (tenths × 10), layer
  thicknesses in mm, descriptive codes as their integer code.
- Every route-survey record carries ``route_type`` (``field``, ``forest``,
  ``ravine``, ``unknown``) and ``source`` (``aisori_export`` or
  ``nsidc_g01170``).  Records are emitted field → forest → ravine → unknown
  for each station/date, so a per-type pivot that keeps the first value
  prefers the field route.  ``route_types=`` filters them.
- Local АИСОРИ exports win over NSIDC on the same station/date/route.
- Physically implausible snow values are nulled (SWE/depth 0–15 m).
- ``interval`` is ``"periodic"`` (route surveys) or ``"daily"`` (snow
  cover); anything else raises :class:`RIHMIError`.
- NSIDC downloads use the shared ``_common.request_with_retries`` loop.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .._common import coerce_list as _coerce_list
from .._common import date_str as _date_str
from .._common import filter_by_bbox as _filter_by_bbox
from .._common import request_with_retries

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

NSIDC_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G01170"
NSIDC_DOI = "https://doi.org/10.7265/N58C9T60"

#: G01170 directories: 1966–1990 base product, then four 1991–1996 updates
#: (disjoint station sets; NSIDC never merged them).
NSIDC_DIRS: tuple[str, ...] = (
    "product",
    "update04012002",
    "update02172003",
    "update05152003",
    "update04052004",
)

ROUTE_SURVEY_PAGE = "http://meteo.ru/data/snow-surveys/"
SNOW_COVER_PAGE = "http://meteo.ru/data/snow-cover/"
AISORI_ROUTE_URL = "http://aisori-m.meteo.ru/waisori/index.xhtml?idata=6"
AISORI_SNOW_COVER_URL = "http://aisori-m.meteo.ru/waisori/index.xhtml?idata=7"

_HERE = Path(__file__).parent
_ROUTE_CATALOGUE = _HERE / "route_survey_stations.csv"
_SNOW_COVER_CATALOGUE = _HERE / "snow_cover_stations.csv"

_DEFAULT_TIMEOUT = 300
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

_ROUTE_SOURCE = (
    "RIHMI-WDC «Маршрутные снегомерные съемки» (АИСОРИ-М export, "
    f"{AISORI_ROUTE_URL}) or NSIDC G01170 ({NSIDC_DOI})"
)
_COVER_SOURCE = (
    "RIHMI-WDC «Характеристики снежного покрова» (АИСОРИ-М export, "
    f"{AISORI_SNOW_COVER_URL})"
)

#: Route-type codes (RIHMI field 4; NSIDC column 4).
ROUTE_TYPES: dict[int, str] = {1: "field", 2: "forest", 3: "ravine", 4: "unknown"}
_ROUTE_ORDER = {"field": 0, "forest": 1, "ravine": 2, "unknown": 3}

# Values ≥ this (in output cm) are implausible for snow depth / SWE.
_MAX_SNOW_CM = 1500.0

# ── Native record layouts ────────────────────────────────────────────────────

#: The 19 fields of a RIHMI route-survey record, in order (meteo.ru Table 1).
#: Field 18 changed codebook in 1984 (see :data:`SNOW_BEDDING_CODES`).
ROUTE_SURVEY_FIELDS: tuple[str, ...] = (
    "wmo_index",  # 1 Индекс ВМО
    "year",  # 2
    "month",  # 3
    "route_type",  # 4 1 поле, 2 лес, 3 балки
    "day",  # 5
    "COVER_SURROUNDINGS",  # 6 snow cover of station surroundings, 0–10
    "COVER_ROUTE",  # 7 snow cover of the route, 0–10
    "ICE_CRUST_COVER",  # 8 ice-crust cover of the route, 0–10
    "HS_MEAN",  # 9 mean depth on route, cm
    "HS_MAX",  # 10 max depth on route, cm
    "HS_MIN",  # 11 min depth on route, cm
    "DENSITY",  # 12 mean density, g/cm3
    "ICE_CRUST_THICKNESS",  # 13 mean ice-crust thickness, mm
    "SATURATED_LAYER",  # 14 water-saturated snow layer, mm
    "WATER_LAYER",  # 15 layer of pure water, mm
    "SWE",  # 16 water storage in snow, mm
    "WATER_TOTAL",  # 17 total water storage, mm
    "SNOW_BEDDING",  # 18 character of snow-cover bedding, code
    "SNOW_STRUCTURE",  # 19 character of the snow, code
)

#: The 9 fields of a RIHMI daily snow-cover record (``IIIII.dat`` layout:
#: cols 1-5 index, 7-10 year, 12-13 month, 15-16 day, 18-21 H, 23-24 S,
#: 26 Q1, 28 Q2, 30 Q3).
SNOW_COVER_FIELDS: tuple[str, ...] = (
    "wmo_index",
    "year",
    "month",
    "day",
    "H",
    "S",
    "Q1",
    "Q2",
    "Q3",
)

#: The 17 whitespace-separated columns of an NSIDC G01170 ``.syn`` line.
NSIDC_SYN_COLUMNS: tuple[str, ...] = (
    "wmo_index",
    "lat_ddmm",
    "lon_dddmm",
    "route_type",
    "COVER_SURROUNDINGS",
    "COVER_ROUTE",
    "ICE_CRUST_COVER",
    "HS_MEAN",
    "HS_MAX",
    "HS_MIN",
    "DENSITY",
    "ICE_CRUST_THICKNESS",
    "SATURATED_LAYER",
    "SWE",
    "SNOW_BEDDING",
    "SNOW_STRUCTURE",
    "qc_flag",
)

# ── Public variable / flag tables ────────────────────────────────────────────


def _var(
    name: str,
    vtype: str,
    units: str,
    output_units: str,
    scale: float,
    interval: str,
    description: str,
    notes: str,
    valid: tuple[float, float] | None,
    nsidc: bool = True,
) -> dict:
    return {
        "name": name,
        "type": vtype,
        "units": units,
        "output_units": output_units,
        "scale": scale,
        "interval": interval,
        "valid_range": valid,
        "in_nsidc": nsidc,
        "description": description,
        "notes": notes,
        "source": _ROUTE_SOURCE if interval == "periodic" else _COVER_SOURCE,
    }


#: Variables keyed by native name.  ``valid_range`` is in *native* units;
#: values outside it are returned as ``None``.  ``in_nsidc`` says whether
#: NSIDC G01170 carries the field (otherwise only АИСОРИ exports do).
VARIABLES: dict[str, dict] = {
    # — route surveys (periodic) —
    "SWE": _var(
        "Snow Water Equivalent (route mean)",
        "swe",
        "mm",
        "cm",
        0.1,
        "periodic",
        "Water storage in the snow (запас воды в снеге), route mean.",
        "RIHMI field 16 / NSIDC column 14; mm → cm. Computed from route-mean "
        "depth and density (snow-weighing balance at every 100–200 m).",
        (0.0, 15000.0),
    ),
    "HS_MEAN": _var(
        "Snow Depth (route mean)",
        "snwd",
        "cm",
        "cm",
        1.0,
        "periodic",
        "Mean snow depth along the survey route.",
        "RIHMI field 9 / NSIDC column 8. 100–200 rod readings per route.",
        (0.0, 1500.0),
    ),
    "HS_MAX": _var(
        "Snow Depth (route maximum)",
        "other",
        "cm",
        "cm",
        1.0,
        "periodic",
        "Largest snow depth along the survey route.",
        "RIHMI field 10 / NSIDC column 9. Typed 'other' so it is never "
        "mistaken for the route-mean depth.",
        (0.0, 1500.0),
    ),
    "HS_MIN": _var(
        "Snow Depth (route minimum)",
        "other",
        "cm",
        "cm",
        1.0,
        "periodic",
        "Smallest snow depth along the survey route.",
        "RIHMI field 11 / NSIDC column 10.",
        (0.0, 1500.0),
    ),
    "DENSITY": _var(
        "Snow Density (route mean)",
        "density",
        "g/cm3",
        "%",
        100.0,
        "periodic",
        "Mean bulk snow density along the route.",
        "RIHMI field 12 / NSIDC column 11; g/cm3 → % (× 100).",
        (0.0, 1.0),
    ),
    "WATER_TOTAL": _var(
        "Total Water Storage",
        "other",
        "mm",
        "cm",
        0.1,
        "periodic",
        "Total water storage (запас воды, общий): snow plus ponded water.",
        "RIHMI field 17 only (not in NSIDC); mm → cm.",
        (0.0, 15000.0),
        nsidc=False,
    ),
    "ICE_CRUST_THICKNESS": _var(
        "Ice Crust Thickness",
        "other",
        "mm",
        "mm",
        1.0,
        "periodic",
        "Mean thickness of the ice crust on the route.",
        "RIHMI field 13 / NSIDC column 12.",
        (0.0, 2000.0),
    ),
    "SATURATED_LAYER": _var(
        "Water-Saturated Snow Layer",
        "other",
        "mm",
        "mm",
        1.0,
        "periodic",
        "Thickness of the snow layer saturated with water.",
        "RIHMI field 14 / NSIDC column 13.",
        (0.0, 5000.0),
    ),
    "WATER_LAYER": _var(
        "Pure Water Layer",
        "other",
        "mm",
        "mm",
        1.0,
        "periodic",
        "Thickness of the layer of pure water under/in the snow.",
        "RIHMI field 15 only (not in NSIDC).",
        (0.0, 5000.0),
        nsidc=False,
    ),
    "COVER_SURROUNDINGS": _var(
        "Snow Cover of Station Surroundings",
        "other",
        "tenths",
        "%",
        10.0,
        "periodic",
        "Degree of snow cover of the station surroundings.",
        "RIHMI field 6 / NSIDC column 5; 0–10 scale → % (× 10).",
        (0.0, 10.0),
    ),
    "COVER_ROUTE": _var(
        "Snow Cover of the Route",
        "other",
        "tenths",
        "%",
        10.0,
        "periodic",
        "Degree of snow cover along the route.",
        "RIHMI field 7 / NSIDC column 6; 0–10 scale → % (× 10).",
        (0.0, 10.0),
    ),
    "ICE_CRUST_COVER": _var(
        "Ice Crust Cover of the Route",
        "other",
        "tenths",
        "%",
        10.0,
        "periodic",
        "Degree of ice-crust cover along the route.",
        "RIHMI field 8 / NSIDC column 7; 0–10 scale → % (× 10). NSIDC uses "
        "an undocumented -1 here very often; it is returned as None.",
        (0.0, 10.0),
    ),
    "SNOW_BEDDING": _var(
        "Snow Cover Bedding Code",
        "other",
        "code",
        "code",
        1.0,
        "periodic",
        "Character of snow-cover bedding (evenness, drifts, soil state).",
        "RIHMI field 18 / NSIDC column 15. Codebook changed in 1984; see "
        "SNOW_BEDDING_CODES.",
        (0.0, 9.0),
    ),
    "SNOW_STRUCTURE": _var(
        "Snow Character Code",
        "other",
        "code",
        "code",
        1.0,
        "periodic",
        "Character of the snow (fresh/old, wet, crust); see SNOW_STRUCTURE_CODES.",
        "RIHMI field 19 / NSIDC column 16.",
        (0.0, 9.0),
    ),
    # — daily snow cover —
    "H": _var(
        "Snow Depth (station stake)",
        "snwd",
        "cm",
        "cm",
        1.0,
        "daily",
        "Daily snow depth at the meteorological station (fixed stake).",
        "RIHMI snow-cover field 5. 9999 or Q1=9 → None. Only from АИСОРИ "
        "exports (no public copy).",
        (0.0, 1500.0),
        nsidc=False,
    ),
    "S": _var(
        "Snow Cover of Station Surroundings",
        "other",
        "tenths",
        "%",
        10.0,
        "daily",
        "Daily degree of snow cover of the station surroundings.",
        "RIHMI snow-cover field 6; 0–10 → % (× 10); 99 → None. Before "
        "Aug 1959 the field is a 0/1 code (≤50% / >50%) and is returned as None.",
        (0.0, 10.0),
        nsidc=False,
    ),
}

#: Standardized type → native names, priority order, per interval.
_TYPE_TO_VARS: dict[str, dict[str, list[str]]] = {}
for _name, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["interval"], {}).setdefault(
        _info["type"], []
    ).append(_name)

_DEFAULT_VARS: dict[str, list[str]] = {
    "periodic": ["SWE", "HS_MEAN"],
    "daily": ["H"],
}

#: Field 18 (SNOW_BEDDING) codebooks.
SNOW_BEDDING_CODES: dict[str, dict[int, str]] = {
    "1977-1983": {
        0: "Even, no drifts",
        1: "Moderately uneven (small drifts, no bare patches)",
        2: "Moderately uneven (small drifts with bare patches)",
        3: "Very uneven (large drifts), no bare patches",
        4: "Very uneven (large drifts) with bare patches",
        5: "Snow cover with thawed patches",
        6: "Snow only in places (1 tenth or less)",
    },
    "1984-": {
        0: "Even, on frozen soil, no drifts",
        1: "Even, on thawed soil, no drifts",
        2: "Even, no drifts, soil state unknown",
        3: "Uneven, on frozen soil, small drifts",
        4: "Uneven, on thawed soil, small drifts",
        5: "Uneven, soil state unknown, small drifts",
        6: "Very uneven, on frozen soil, large drifts",
        7: "Very uneven, on thawed soil, large drifts",
        8: "Very uneven, soil state unknown, large drifts",
        9: "Snow cover with thawed patches",
    },
}

#: Field 19 (SNOW_STRUCTURE) codebook.
SNOW_STRUCTURE_CODES: dict[int, str] = {
    0: "Fresh snow, powdery",
    1: "Fresh snow, fluffy",
    2: "Fresh snow, sticky",
    3: "Old snow, loose",
    4: "Old snow, dense",
    5: "Old snow, wet",
    6: "Snow crust not bonded to the snow below",
    7: "Dense snow with surface crust",
    8: "Wet snow with surface crust",
    9: "Snow saturated with water",
}

#: Flags.  NSIDC records carry ``"NSIDC:<qc>"`` (column 17); АИСОРИ daily
#: snow-cover records carry ``"Q1:<a>,Q2:<b>,Q3:<c>"``.  Route-survey
#: АИСОРИ records have no flag field (``flag`` is ``None``).
DATA_FLAGS: dict[str, str] = {
    "NSIDC:0": "NSIDC QC flag not computed",
    "NSIDC:1": "Raw density increased ×10 by NSIDC (e.g. .6 → .06)",
    "NSIDC:2": "Raw density missing decimal point, fixed by NSIDC (6 → .06)",
    "NSIDC:3": "Mean depth > 200 cm, reset by NSIDC from SWE and density",
    "NSIDC:4": "SWE > 400 mm, reset by NSIDC from depth and density",
    "NSIDC:5": "Suspect, out of range; error unknown so data unchanged",
    "Q1:0": "Snow depth correct",
    "Q1:1": "No snow",
    "Q1:2": "No snow at the station, but snow in the surroundings",
    "Q1:3": "Snow depth less than 0.5 cm",
    "Q1:9": "Not observed or depth rejected (value returned as None)",
    "Q2:0": "Depth passed the step-consistency check",
    "Q2:1": "Doubtful: |ΔH2-ΔH1|>30 cm with both steps >10 cm",
    "Q3:0": "Depth consistent with air temperature",
    "Q3:1": "Snow reported with Tmean > 5 °C and Tmin > 0 °C",
}

# ── Station catalogues ───────────────────────────────────────────────────────

_CLOSED_RE = re.compile(r"закрыт|законсерв", re.IGNORECASE)
_STATION_LOC_RE = re.compile(
    r"^\s*(\d{1,5})\s+(.*?)\s+(\d{3,4})\s+(\d{3,5})(?:\s+(-?\d+))?\s*$"
)


def _ddmm(value: str) -> float | None:
    """Convert packed degrees+minutes (``DDMM`` / ``DDDMM``) to decimal."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    sign = -1 if n < 0 else 1
    n = abs(n)
    deg = n // 100 + (n % 100) / 60.0
    if deg > 180:
        deg -= 360
    return round(sign * deg, 4)


def _read_catalogue(path: Path) -> dict[str, dict]:
    """Read a packaged RIHMI catalogue CSV (``#`` lines are provenance)."""
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    out: dict[str, dict] = {}
    for row in csv.DictReader(lines):
        sid = row["station_id"].strip()
        out[sid] = {
            "station_id": sid,
            "name": row["name"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "elevation_m": float(row["elevation_m"]) if row["elevation_m"] else None,
            "start_year": int(row["start_year"]) if row["start_year"] else None,
            "note": row["note"] or None,
        }
    return out


def _parse_station_loc(text: str) -> dict[str, dict]:
    """Parse NSIDC G01170 ``station.loc`` (id, name, DDMM lat, DDDMM lon, elev)."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        m = _STATION_LOC_RE.match(line)
        if not m:
            continue
        sid = m.group(1).zfill(5)
        if sid == "00000":
            continue
        out[sid] = {
            "station_id": sid,
            "name": m.group(2).strip(),
            "latitude": _ddmm(m.group(3)),
            "longitude": _ddmm(m.group(4)),
            "elevation_m": float(m.group(5)) if m.group(5) else None,
        }
    return out


# ── Value conversion ─────────────────────────────────────────────────────────


def _num(token: str | None) -> float | None:
    """Parse a native token; blanks and ±9999 are missing."""
    if token is None:
        return None
    text = token.strip().replace(",", ".")
    if not text:
        return None
    try:
        f = float(text)
    except ValueError:
        return None
    if f != f or abs(f) == 9999.0:
        return None
    return f


def _convert(variable: str, raw: float | None) -> float | None:
    """Native value → output units, nulling values outside ``valid_range``."""
    if raw is None:
        return None
    info = VARIABLES[variable]
    lo, hi = info["valid_range"] or (float("-inf"), float("inf"))
    if not (lo <= raw <= hi):
        return None
    value = round(raw * info["scale"], 3)
    if info["type"] in ("swe", "snwd") and not (0.0 <= value <= _MAX_SNOW_CM):
        return None
    return value


def _nsidc_num(token: str) -> float | None:
    """NSIDC ``.syn`` token → float; ``-2`` (documented) and ``-1`` → None."""
    try:
        f = float(token)
    except ValueError:
        return None
    if f < 0:
        return None
    return f


def _valid_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


# ── NSIDC .syn parsing ───────────────────────────────────────────────────────

_SYN_NAME_RE = re.compile(r"(?:^|/)(\d{8})\.syn$")


def _parse_syn_lines(
    day: str, text: str, wanted: set[str] | None
) -> Iterator[dict[str, Any]]:
    """Yield parsed rows of one ``YYYYMMDD.syn`` file for wanted stations."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != len(NSIDC_SYN_COLUMNS):
            continue
        sid = parts[0].zfill(5)
        if wanted is not None and sid not in wanted:
            continue
        try:
            route = ROUTE_TYPES.get(int(parts[3]), "unknown")
        except ValueError:
            route = "unknown"
        values = {
            col: _nsidc_num(parts[i])
            for i, col in enumerate(NSIDC_SYN_COLUMNS)
            if col in VARIABLES
        }
        yield {
            "station_id": sid,
            "date": day,
            "route_type": route,
            "values": values,
            "flag": f"NSIDC:{int(parts[16])}"
            if parts[16].lstrip("-").isdigit()
            else None,
            "source": "nsidc_g01170",
        }


def _iter_syn_tar(
    raw: bytes | str | os.PathLike,
    wanted: set[str] | None,
    begin: str | None,
    end: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield rows from a G01170 ``syn.tar`` (path or bytes) inside the window."""
    if isinstance(raw, (bytes, bytearray)):
        tf = tarfile.open(fileobj=io.BytesIO(raw))
    else:
        tf = tarfile.open(raw)
    with tf:
        for member in tf:
            m = _SYN_NAME_RE.search(member.name)
            if not m or not member.isfile():
                continue
            ymd = m.group(1)
            day = _valid_date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
            if day is None:
                continue
            if (begin and day < begin) or (end and day > end):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            text = fh.read().decode("ascii", errors="replace")
            yield from _parse_syn_lines(day, text, wanted)


# ── АИСОРИ export parsing (verified against real exports) ───────────────────

_INDEX_RE = re.compile(r"^\d{5}$")
_YEAR_RE = re.compile(r"^\d{4}$")

#: Column names as АИСОРИ-M writes them in an export's ``fld*.txt`` (and on
#: its result page) → this module's native field names.  Verified against
#: real exports of both datasets (2026-10-14).
AISORI_COLUMN_NAMES: dict[str, str] = {
    "Индекс ВМО": "wmo_index",
    "Год": "year",
    "Месяц": "month",
    "Тип маршрута": "route_type",
    "День": "day",
    "Степень покрытия окрестности станции снегом": "COVER_SURROUNDINGS",
    "Степень покрытия маршрута снегом": "COVER_ROUTE",
    "Степень покрытия маршрута ледяной коркой": "ICE_CRUST_COVER",
    "Средняя высота снежного покрова на маршруте (см)": "HS_MEAN",
    "Наибольшая высота снежного покрова на маршруте (см)": "HS_MAX",
    "Наименьшая высота снежного покрова на маршруте (см)": "HS_MIN",
    "Средняя плотность снега": "DENSITY",
    "Средняя толщина ледяной корки": "ICE_CRUST_THICKNESS",
    "Толщина слоя снега, насыщенного водой (мм)": "SATURATED_LAYER",
    "Толщина слоя чистой воды (мм)": "WATER_LAYER",
    "Запас воды в снеге (мм)": "SWE",
    "Запас воды, общий (мм)": "WATER_TOTAL",
    "Характер залегания снежного покрова": "SNOW_BEDDING",
    "Характер снежного покрова": "SNOW_STRUCTURE",
    # daily snow cover (SNOW)
    "Высота снежного покрова": "H",
    "Степень покрова окрестности станции снегом": "S",
    "Дополнительная информация о высоте снежного покрова": "Q1",
    "Признак качества по высоте снежного покрова": "Q2",
    "Дополнительная информация с учетом температуры воздуха": "Q3",
}

#: Column widths of a full АИСОРИ-M export, used when no ``fld*.txt`` is at
#: hand.  The space-delimited variant is fixed width: each field is
#: right-aligned in its width, fields are separated by one space, and a
#: missing value is all blanks — so it cannot be split on whitespace.
ROUTE_SURVEY_WIDTHS: tuple[int, ...] = (
    5, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 2, 4, 4, 1, 1,
)  # fmt: skip
SNOW_COVER_WIDTHS: tuple[int, ...] = (5, 4, 2, 2, 4, 2, 1, 1, 1)

_FLD_RE = re.compile(r"^\s*\d+\s+\d+\s+(\d+)(?:,\d+)?\s+(.+?)\s*$")


def parse_aisori_columns(text: str) -> list[tuple[str, int]]:
    """Parse an export's ``fld*.txt`` into ``[(native field, width), …]``.

    Each line is ``N  N  width[,decimals]  <Russian column name>``.
    Unknown column names are kept as the Russian name (and then ignored).
    """
    cols: list[tuple[str, int]] = []
    for line in text.splitlines():
        m = _FLD_RE.match(line)
        if m:
            name = m.group(2).strip()
            cols.append((AISORI_COLUMN_NAMES.get(name, name), int(m.group(1))))
    return cols


def _default_columns(n_fields: int) -> list[tuple[str, int]] | None:
    if n_fields == len(ROUTE_SURVEY_FIELDS):
        return list(zip(ROUTE_SURVEY_FIELDS, ROUTE_SURVEY_WIDTHS, strict=True))
    if n_fields == len(SNOW_COVER_FIELDS):
        return list(zip(SNOW_COVER_FIELDS, SNOW_COVER_WIDTHS, strict=True))
    return None


def _fixed_width(line: str, widths: list[int]) -> list[str] | None:
    """Split a space-variant export line into fields by column position.

    Fields are right-aligned in their widths with one space between them,
    and a missing value is blanks, so whitespace splitting would shift every
    later field.  The widths in ``fld*.txt`` are not always exact (the route
    survey's two one-character code columns are printed two wide), so each
    token is matched to the column whose right edge is nearest its own,
    keeping column order; a line that cannot be matched within a few
    characters returns ``None``.
    """
    edges, pos = [], 0
    for w in widths:
        pos += w
        edges.append(pos)
        pos += 1
    parts = [""] * len(widths)
    last = -1
    for m in re.finditer(r"\S+", line):
        end = m.end()
        best = None
        for j in range(last + 1, len(edges)):
            diff = abs(edges[j] - end)
            if best is None or diff < best[0]:
                best = (diff, j)
            if edges[j] > end + 3:
                break
        if best is None or best[0] > 3:
            return None
        last = best[1]
        parts[last] = m.group()
    return parts


def parse_aisori_text(
    text: str,
    kind: str | None = None,
    columns: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse an АИСОРИ-M export data file (``wr*.txt``) into native rows.

    Both delimiter choices on АИСОРИ-M's export form are handled: ``;``
    separated, and the space variant, which is fixed width (one space
    between right-aligned fields; a missing value is blanks).  A selection
    of only some parameters is handled when ``columns`` (from the export's
    ``fld*.txt``, see :func:`parse_aisori_columns`) is given;
    :func:`read_aisori_export` finds and applies that file for you.

    Parameters
    ----------
    text : str
        Data file contents.
    kind : {"route_survey", "snow_cover"} or None
        Record layout.  ``None`` infers it from the columns (a ``Тип
        маршрута`` column means route survey) or, without ``columns``, from
        the field count of a ``;`` line (19 → route survey, 9 → snow cover).
    columns : list of (field, width), optional
        The export's column list, in order.

    Returns
    -------
    list[dict]
        ``{station_id, date, route_type, values, flag, source, kind}`` rows,
        ``values`` mapping native variable names to native-unit floats
        (``None`` when missing).  Lines that are not data lines are skipped.

    Raises
    ------
    RIHMIError
        If ``kind`` is not one of the accepted values.

    Examples
    --------
    >>> line = "22113;2020; 1; 1;10;10;10;  ;  62;  77;  43;0.26;   ;   ;  ; 161; 161; 3; 2"
    >>> parse_aisori_text(line)[0]["values"]["SWE"]
    161.0
    """
    if kind not in (None, "route_survey", "snow_cover"):
        raise RIHMIError(f"Unknown export kind {kind!r}")
    if columns is not None and kind is None:
        names = {name for name, _ in columns}
        kind = "route_survey" if "route_type" in names else "snow_cover"
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line in text.splitlines():
        line = line.rstrip("\r\n")
        if not line.strip() or not _INDEX_RE.match(line[:5]):
            continue
        cols = columns
        if ";" in line:
            parts = [p.strip() for p in line.split(";")]
            if parts and parts[-1] == "":
                parts.pop()  # a trailing ";" (other tools add one)
            if cols is None:
                cols = _default_columns(len(parts))
        else:
            if cols is None:
                # Fixed width needs the layout: try the full ones by length.
                # Without fld*.txt, pick the full layout by line length.
                n = len(line.rstrip())
                for fields, widths in (
                    (ROUTE_SURVEY_FIELDS, ROUTE_SURVEY_WIDTHS),
                    (SNOW_COVER_FIELDS, SNOW_COVER_WIDTHS),
                ):
                    full = sum(widths) + len(widths) - 1
                    if full - 2 <= n <= full + 4:
                        cols = list(zip(fields, widths, strict=True))
                        break
            parts = _fixed_width(line, [w for _, w in cols]) if cols else None
        if not cols or parts is None or len(parts) != len(cols):
            skipped += 1
            continue
        f = {name: value for (name, _), value in zip(cols, parts, strict=True)}
        k = kind or ("route_survey" if "route_type" in f else "snow_cover")
        if not _YEAR_RE.match(f.get("year", "")):
            skipped += 1
            continue
        row = _route_row(f) if k == "route_survey" else _snow_cover_row(f)
        if row is None:
            skipped += 1
            continue
        row["kind"] = k
        rows.append(row)
    if skipped:
        logger.warning("RIHMI: skipped %d unparseable export line(s)", skipped)
    return rows


def read_aisori_export(path: str | os.PathLike) -> list[dict[str, Any]]:
    """
    Read one АИСОРИ-M export: the downloaded ``.zip`` or its ``wr*.txt``.

    The zip АИСОРИ-M serves holds ``wr<id>.txt`` (the data), ``fld<id>.txt``
    (column numbers, widths and names) and ``statlist<id>.txt`` (the
    stations).  The column file is used when present (inside the zip, or
    next to a ``wr*.txt``), so exports of a subset of parameters parse
    correctly.

    Parameters
    ----------
    path : str or PathLike
        A ``.zip`` export or a data ``.txt``/``.dat`` file.

    Returns
    -------
    list[dict]
        As :func:`parse_aisori_text`.

    Raises
    ------
    RIHMIError
        If the file cannot be read or the zip holds no data file.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                data = [n for n in names if Path(n).name.lower().startswith("wr")]
                if not data:
                    raise RIHMIError(f"{path.name}: no wr*.txt data file inside")
                rows: list[dict[str, Any]] = []
                for name in data:
                    stem = Path(name).name[2:]
                    fld = next(
                        (
                            n
                            for n in names
                            if Path(n).name.lower() == "fld" + stem.lower()
                        ),
                        None,
                    )
                    columns = (
                        parse_aisori_columns(_decode(zf.read(fld))) if fld else None
                    )
                    rows.extend(
                        parse_aisori_text(
                            _decode(zf.read(name)), columns=columns or None
                        )
                    )
                return rows
        raw = path.read_bytes()
        columns = None
        if path.name.lower().startswith("wr"):
            fld = path.with_name("fld" + path.name[2:])
            if fld.exists():
                columns = parse_aisori_columns(_decode(fld.read_bytes())) or None
        return parse_aisori_text(_decode(raw), columns=columns)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RIHMIError(f"Cannot read export file {path}: {exc}") from exc


def _route_row(f: dict[str, str]) -> dict[str, Any] | None:
    try:
        y, m, d = int(f["year"]), int(f["month"]), int(f["day"])
        route_code = int(f["route_type"])
    except (KeyError, ValueError):
        return None
    day = _valid_date(y, m, d)
    if day is None:
        return None
    return {
        "station_id": f["wmo_index"],
        "date": day,
        "route_type": ROUTE_TYPES.get(route_code, "unknown"),
        "values": {n: _num(f.get(n)) for n in ROUTE_SURVEY_FIELDS if n in VARIABLES},
        "flag": None,
        "source": "aisori_export",
    }


def _snow_cover_row(f: dict[str, str]) -> dict[str, Any] | None:
    try:
        y, m, d = int(f["year"]), int(f["month"]), int(f["day"])
    except (KeyError, ValueError):
        return None
    day = _valid_date(y, m, d)
    if day is None:
        return None
    h = _num(f.get("H"))
    s = _num(f.get("S"))
    q1 = (f.get("Q1") or "").strip()
    if q1 == "9":
        h = None
    if s is not None and s == 99:
        s = None
    if s is not None and (y, m) < (1959, 8):
        s = None  # 0/1 code (≤50% / >50%) before Aug 1959, not tenths
    q2 = (f.get("Q2") or "").strip()
    q3 = (f.get("Q3") or "").strip()
    flag = f"Q1:{q1 or '-'},Q2:{q2 or '-'},Q3:{q3 or '-'}"
    return {
        "station_id": f["wmo_index"],
        "date": day,
        "route_type": None,
        "values": {"H": h, "S": s},
        "flag": flag,
        "source": "aisori_export",
    }


# ── Client ───────────────────────────────────────────────────────────────────


class RIHMIClient:
    """
    Client for RIHMI-WDC route snow surveys and daily snow cover.

    Route surveys (``interval="periodic"``) come from local АИСОРИ-М exports
    and/or the public NSIDC G01170 archive (1966–1996, no credentials).
    Daily snow cover (``interval="daily"``) comes only from local exports.

    Parameters
    ----------
    data_dir : str or Path, optional
        Folder of АИСОРИ-М exports, searched recursively for the downloaded
        ``*.zip`` files and for extracted ``*.txt``/``*.dat``/``*.csv`` data
        files (an export's ``fld*``/``statlist*`` files are used, not read
        as data).  See the module docstring for the export steps.
    use_nsidc : bool
        Include NSIDC G01170 stations and route-survey data (default True).
    cache_dir : str or Path, optional
        Where the G01170 tar files are cached (~50 MB).  Defaults to
        ``easysnowdata.config.cache_dir("stations", "rihmi")``.
    base_url : str
        Root of the NSIDC G01170 archive.  Override for a mirror.
    timeout, max_retries, backoff : int
        HTTP settings for the NSIDC downloads.
    session : requests.Session, optional
        Pre-configured session.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike | None = None,
        use_nsidc: bool = True,
        cache_dir: str | os.PathLike | None = None,
        base_url: str = NSIDC_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.use_nsidc = use_nsidc
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._routes_cat = _read_catalogue(_ROUTE_CATALOGUE)
        self._cover_cat = _read_catalogue(_SNOW_COVER_CATALOGUE)
        self._nsidc_cat: dict[str, dict] | None = None
        self._local_cache: dict[str, list[dict]] = {}

    # ── NSIDC files ──────────────────────────────────────────────────────────

    def _cache_root(self) -> Path:
        if self._cache_dir is None:
            from easysnowdata.config import cache_dir  # noqa: PLC0415

            self._cache_dir = cache_dir("stations", "rihmi")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _nsidc_file(self, rel: str) -> Path:
        """Return the cached path of ``base_url/rel``, downloading once."""
        path = self._cache_root() / "g01170" / rel
        if path.exists() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = request_with_retries(
            self._session,
            f"{self.base_url}/{rel}",
            error_cls=RIHMIError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        return path

    def _nsidc_stations(self) -> dict[str, dict]:
        if self._nsidc_cat is None:
            if not self.use_nsidc:
                self._nsidc_cat = {}
            else:
                text = self._nsidc_file("station.loc").read_text(
                    encoding="ascii", errors="replace"
                )
                self._nsidc_cat = _parse_station_loc(text)
                logger.info("RIHMI: %d NSIDC G01170 stations", len(self._nsidc_cat))
        return self._nsidc_cat

    def _nsidc_rows(
        self, wanted: set[str], begin: str | None, end: str | None
    ) -> Iterator[dict[str, Any]]:
        for d in NSIDC_DIRS:
            # Base product is 1966–1990, updates 1991–1996.
            lo, hi = (
                ("1966-01-01", "1990-12-31")
                if d == "product"
                else (
                    "1991-01-01",
                    "1996-12-31",
                )
            )
            if (end and end < lo) or (begin and begin > hi):
                continue
            path = self._nsidc_file(f"{d}/syn.tar")
            yield from _iter_syn_tar(path, wanted, begin, end)

    # ── Local АИСОРИ exports ─────────────────────────────────────────────────

    def _local_files(self, files: Iterable | None) -> list[Path]:
        if files is not None:
            return [
                Path(f)
                for f in ([files] if isinstance(files, (str, os.PathLike)) else files)
            ]
        if self.data_dir is None:
            return []
        if not self.data_dir.is_dir():
            raise RIHMIError(f"data_dir {self.data_dir} is not a directory")
        return sorted(
            p
            for p in self.data_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".zip", ".txt", ".dat", ".csv")
            # an export's column list and station list are not data
            and not p.name.lower().startswith(("fld", "statlist"))
        )

    def _local_rows(self, files: Iterable | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self._local_files(files):
            key = str(path.resolve())
            if key not in self._local_cache:
                self._local_cache[key] = read_aisori_export(path)
            rows.extend(self._local_cache[key])
        return rows

    # ── Public API — stations ────────────────────────────────────────────────

    def _build_station(self, sid: str) -> dict | None:
        r = self._routes_cat.get(sid)
        c = self._cover_cat.get(sid)
        n = self._nsidc_stations().get(sid)
        base = r or c or n
        if base is None:
            return None
        closed = bool(c and c["note"] and _CLOSED_RE.search(c["note"]))
        active = r is not None or (c is not None and not closed)
        has_route = r is not None or n is not None
        return {
            "station_id": sid,
            "name": base["name"],
            "name_latin": n["name"] if n else None,
            "latitude": base["latitude"],
            "longitude": base["longitude"],
            "elevation_m": base["elevation_m"],
            "status": "Active" if active else "Inactive",
            "wmo_id": sid,
            "operator": "Roshydromet (archive: RIHMI-WDC, Obninsk)",
            "measurement_method": (
                "Manual snow survey (snow course): portable rod depth every "
                "10–20 m and snow-weighing balance every 100–200 m on field/"
                "forest routes; station stake for daily depth"
            ),
            "has_swe": has_route,
            "has_snwd": True,
            "in_route_survey_catalogue": r is not None,
            "in_snow_cover_catalogue": c is not None,
            "in_nsidc_g01170": n is not None,
            "route_survey_start_year": r["start_year"] if r else None,
            "snow_cover_start_year": c["start_year"] if c else None,
            "public_period": "1966–1996 (NSIDC G01170)" if n else None,
            "login_period": "1966–present (АИСОРИ-М)" if r else None,
            "note": (c or {}).get("note") or (r or {}).get("note"),
            "station_url": ROUTE_SURVEY_PAGE if r or n else SNOW_COVER_PAGE,
        }

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Every station in the RIHMI route-survey and snow-cover catalogues and
        (when ``use_nsidc``) NSIDC G01170's ``station.loc``.

        Parameters
        ----------
        active_only : bool
            Keep only catalogue-``"Active"`` stations (see the module
            docstring).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (WMO index), ``name`` (Cyrillic from RIHMI, else
            NSIDC's Latin), ``name_latin``, ``latitude``, ``longitude``,
            ``elevation_m``, ``status``, ``wmo_id``, ``operator``,
            ``measurement_method``, ``has_swe``, ``has_snwd``, catalogue
            membership flags, start years, ``public_period``,
            ``login_period``, ``note`` and ``station_url``.

        Raises
        ------
        RIHMIError
            If NSIDC's ``station.loc`` cannot be downloaded.
        """
        ids = set(self._routes_cat) | set(self._cover_cat) | set(self._nsidc_stations())
        out = [s for sid in sorted(ids) if (s := self._build_station(sid)) is not None]
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Station dict plus ``data_variables``.

        Parameters
        ----------
        station_id : str
            WMO index, e.g. ``"22113"`` (zero-padded to 5 digits).

        Returns
        -------
        dict
            The :meth:`get_all_stations` dict plus ``data_variables``, one
            entry per variable with ``name``, ``type``, ``interval``,
            ``units`` (emitted), ``description``, ``notes``,
            ``begin_date``, ``end_date`` and ``n_obs``.

        Notes
        -----
        Route-survey variables are listed when the station is in the RIHMI
        route catalogue or NSIDC (NSIDC-only stations list just the fields
        NSIDC carries); snow-cover variables when it is in the snow-cover
        catalogue.  ``begin_date`` is the catalogue start year (route
        catalogue: start of the dataset at that station; snow-cover
        catalogue: start of observations at the station), ``end_date`` is
        ``1996-12-31`` for NSIDC-only stations and ``None`` otherwise
        (ongoing behind a login).  ``n_obs`` is ``None`` (unknown without
        reading the data).

        Raises
        ------
        RIHMIError
            If the station is in none of the catalogues.
        """
        sid = str(station_id).zfill(5)
        sta = self._build_station(sid)
        if sta is None:
            raise RIHMIError(
                f"Station {station_id!r} not found in the RIHMI/NSIDC catalogues"
            )
        dvars: list[dict] = []
        for name, info in VARIABLES.items():
            if info["interval"] == "periodic":
                if sta["in_route_survey_catalogue"]:
                    begin = f"{sta['route_survey_start_year']}-01-01"
                    end = None
                elif sta["in_nsidc_g01170"] and info["in_nsidc"]:
                    begin, end = "1966-01-01", "1996-12-31"
                else:
                    continue
            else:
                if not sta["in_snow_cover_catalogue"]:
                    continue
                begin = (
                    f"{sta['snow_cover_start_year']}-01-01"
                    if sta["snow_cover_start_year"]
                    else None
                )
                end = None
            dvars.append(
                {
                    "name": name,
                    "type": info["type"],
                    "interval": info["interval"],
                    "units": info["output_units"],
                    "description": info["description"],
                    "notes": info["notes"],
                    "begin_date": begin,
                    "end_date": end,
                    "n_obs": None,
                }
            )
        sta["data_variables"] = dvars
        return sta

    # ── Public API — data ────────────────────────────────────────────────────

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "periodic",
        include_flags: bool = False,
        route_types: list[str] | str | None = None,
        files: Iterable[str | os.PathLike] | str | os.PathLike | None = None,
        source: str = "auto",
    ) -> list[dict]:
        """
        Standardized data fetch — a flat list of observation records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            WMO indexes (``"22113"``).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            Native names (``"SWE"``, ``"HS_MEAN"``, ``"H"``…) or standardized
            types (``"swe"``, ``"snwd"``, ``"density"``, ``"other"``) valid for
            the chosen interval.  ``None`` = snow only: ``SWE`` + ``HS_MEAN``
            (periodic) or ``H`` (daily).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds.
        interval : {"periodic", "daily"}
            ``"periodic"`` = route snow surveys; ``"daily"`` = station snow
            cover (local exports only).
        include_flags : bool
            Add ``flag`` (see :data:`DATA_FLAGS`).
        route_types : list[str] or str, optional
            Keep only these routes (``"field"``, ``"forest"``, ``"ravine"``,
            ``"unknown"``).  Periodic only.
        files : path or list of paths, optional
            АИСОРИ-М export files; overrides ``data_dir`` for this call.
        source : {"auto", "nsidc", "files"}
            ``"auto"`` merges local exports (preferred) with NSIDC;
            ``"nsidc"`` / ``"files"`` restrict to one.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval,
            route_type, source}`` (+ ``flag``).  NSIDC downloads the ~50 MB
            G01170 ``syn.tar`` files once into the cache.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        RIHMIError
            Unknown variable, interval, route type or source; daily data
            requested with no local export; download failure.

        Example
        -------
        >>> client = RIHMIClient()
        >>> records = client.get_data(
        ...     station_ids="22113",
        ...     variables=["swe"],
        ...     begin_date="1985-01-01",
        ...     end_date="1985-04-30",
        ... )
        >>> records[0]["units"], records[0]["route_type"]
        ('cm', 'field')
        """
        interval = interval.lower()
        if interval not in _DEFAULT_VARS:
            raise RIHMIError(
                f"Unsupported interval {interval!r} for RIHMI — use 'periodic' "
                "(route snow surveys) or 'daily' (station snow cover)."
            )
        if source not in ("auto", "nsidc", "files"):
            raise RIHMIError(
                f"Unknown source {source!r}; use 'auto', 'nsidc' or 'files'."
            )
        wanted_vars = _resolve_variables(variables, interval)
        routes = _resolve_routes(route_types) if interval == "periodic" else None

        if station_ids is not None:
            ids = [str(s).zfill(5) for s in _coerce_list(station_ids)]
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        wanted = set(ids)
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        kind = "route_survey" if interval == "periodic" else "snow_cover"

        local: list[dict] = []
        if source in ("auto", "files"):
            local = [
                r
                for r in self._local_rows(files)
                if r["kind"] == kind
                and r["station_id"] in wanted
                and not (begin and r["date"] < begin)
                and not (end and r["date"] > end)
            ]
        if interval == "daily" and source == "nsidc":
            raise RIHMIError(
                "NSIDC G01170 has no daily snow cover; use source='files'."
            )
        if interval == "daily" and not local and not self._local_files(files):
            raise RIHMIError(
                "Daily snow cover is only available from АИСОРИ-М exports "
                f"(login required: {AISORI_SNOW_COVER_URL}). Pass data_dir= or files=."
            )

        # Merge: local first, NSIDC fills gaps (same station/date/route).
        merged: dict[tuple, dict] = {}
        for row in local:
            merged.setdefault((row["station_id"], row["date"], row["route_type"]), row)
        if interval == "periodic" and source in ("auto", "nsidc") and self.use_nsidc:
            for row in self._nsidc_rows(wanted, begin, end):
                merged.setdefault(
                    (row["station_id"], row["date"], row["route_type"]), row
                )

        records: list[dict] = []
        for key in sorted(
            merged,
            key=lambda k: (k[0], k[1], _ROUTE_ORDER.get(k[2] or "", 9)),
        ):
            row = merged[key]
            if routes is not None and row["route_type"] not in routes:
                continue
            for var in wanted_vars:
                if var not in row["values"]:
                    continue
                info = VARIABLES[var]
                rec: dict[str, Any] = {
                    "station_id": row["station_id"],
                    "date": row["date"],
                    "variable": var,
                    "type": info["type"],
                    "value": _convert(var, row["values"][var]),
                    "units": info["output_units"],
                    "interval": interval,
                    "route_type": row["route_type"],
                    "source": row["source"],
                }
                if include_flags:
                    rec["flag"] = row["flag"]
                records.append(rec)
        return records


# ── Exception ────────────────────────────────────────────────────────────────


class RIHMIError(Exception):
    """Raised when RIHMI/NSIDC data cannot be fetched, parsed or resolved."""


# ── Private helpers ──────────────────────────────────────────────────────────


def _decode(raw: bytes) -> str:
    """Decode an export: UTF-8 (with BOM) first, then Windows-1251."""
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _resolve_variables(variables: list[str] | str | None, interval: str) -> list[str]:
    """Translate native names and/or standardized types for one interval."""
    if variables is None:
        return list(_DEFAULT_VARS[interval])
    raw = [variables] if isinstance(variables, str) else list(variables)
    if not raw:
        return list(_DEFAULT_VARS[interval])
    by_type = _TYPE_TO_VARS.get(interval, {})
    out: list[str] = []
    for v in raw:
        if v in VARIABLES:
            if VARIABLES[v]["interval"] != interval:
                raise RIHMIError(
                    f"Variable {v!r} belongs to interval "
                    f"{VARIABLES[v]['interval']!r}, not {interval!r}."
                )
            candidates = [v]
        elif v in by_type:
            candidates = by_type[v]
        else:
            raise RIHMIError(
                f"Unknown variable {v!r} for RIHMI ({interval}) — expected one of "
                f"{sorted(n for n, i in VARIABLES.items() if i['interval'] == interval)}"
                f" or a standardized type ({sorted(by_type)})."
            )
        out.extend(c for c in candidates if c not in out)
    return out


def _resolve_routes(route_types: list[str] | str | None) -> set[str] | None:
    if route_types is None:
        return None
    wanted = set(_coerce_list(route_types))
    unknown = wanted - set(ROUTE_TYPES.values())
    if unknown:
        raise RIHMIError(
            f"Unknown route type(s) {sorted(unknown)} — expected some of "
            f"{sorted(ROUTE_TYPES.values())}."
        )
    return wanted
