"""
clients/dwd/dwd_client.py
=========================
Python client for the Deutscher Wetterdienst (DWD) Climate Data Center
(CDC) open-data daily station archive — snow water equivalent and snow
depth for Germany.

Primary use case: daily manual snow water equivalent (``WASH_6``) at
~550 DWD and partner-network climate stations, plus daily snow depth
(``SH_TAG``) and new-snow depth (``NSH_TAG``) at the ~2 300 currently
reporting precipitation stations, including the mountain synoptic stations
(Zugspitze, Feldberg, Fichtelberg, Brocken, Großer Arber) that are *not* in
the SWE product.

Data source       : https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/
Format reference  : the dataset descriptions in each product directory,
                    ``water_equiv/DESCRIPTION_f62e277b-…_en.pdf`` (and the
                    German ``BESCHREIBUNG_…_de.pdf``, which is more complete)
                    and ``more_precip/DESCRIPTION_obsgermany-climate-daily-more_precip_en.pdf``
Licence           : CC BY 4.0 (stated in the "COPYRIGHT" section of both
                    dataset descriptions); cite as "Deutscher Wetterdienst".
Authentication    : none

Products used
-------------
- ``water_equiv/`` — "Daily observations of snow height and water
  equivalent for Germany".  Columns ``STATIONS_ID;MESS_DATUM;QN_6;ASH_6;
  SH_TAG;WASH_6;WAAS_6;eor``.  Source of ``WASH_6``, ``WAAS_6``, ``ASH_6``
  (and of ``SH_TAG`` for the few stations not in ``more_precip``).
- ``more_precip/`` — daily precipitation-station values.  Columns
  ``STATIONS_ID;MESS_DATUM;QN_6;RS;RSF;SH_TAG;NSH_TAG;eor``.  Source of
  ``SH_TAG``, ``NSH_TAG`` and ``RS``.

Each product has a ``recent/`` directory (rolling ~500 days up to
yesterday, rebuilt daily, quality control *not* finished) and a
``historical/`` directory (rebuilt annually, quality control finished).
Per-station zips are ``tageswerte_{Wa|RR}_{id5}_akt.zip`` (recent) and
``tageswerte_{Wa|RR}_{id5}_{begin}_{end}_hist.zip`` (historical — the
date range is in the name, so the directory listing is parsed to find
it).  Each zip holds one ``produkt_*.txt`` (semicolon CSV, ``-999`` =
missing) plus ``Metadaten_*`` files (instrument, station-move and
parameter history).  Station lists are fixed-width latin-1 text files
(``Wa_/RR_Tageswerte_Beschreibung_Stationen.txt``).

Merging recent + historical
---------------------------
The two directories overlap (in Oct 2026: historical to 2025-12-31,
recent from 2025-04-12).  Per DWD, ``historical/`` has completed quality
control and ``recent/`` has not, so the historical value is kept for
every date it covers and ``recent/`` only fills later dates.  Historical
zips are immutable per file name and are cached on disk under
``easysnowdata.config.cache_dir("stations", "dwd")``; recent zips are
always re-downloaded.

Measurement and cadence (from the German dataset description)
-------------------------------------------------------------
SWE is measured manually: a snow core of defined area is cut out
(*ausgestochen*), melted, and the melt-water column read in mm (station
metadata: "Schneesonde WS 43; Wasseräquivalentmessung, manuell" at
essentially every station).  DWD: "Wasseräquivalente [wurden] meistens
nicht an allen Tagen bestimmt …, sondern je nach gültigen
Messvorschriften und erst ab einer Mindestschneehöhe (oft 5 cm)" — SWE
is mostly not determined every day, but according to the observing rules
in force and only above a minimum snow depth (often 5 cm).  The
reference time is the 06 UTC morning observation (05:50 UTC).  Zero SWE
is never reported: days without a measurement are ``-999`` (→ ``None``).

Variables
---------
- ``WASH_6`` — water equivalent of the total snow depth (mm → cm).
- ``WAAS_6`` — water equivalent of the cored (*ausgestochene*) snow (mm →
  cm).  In 2025–26 identical to ``WASH_6`` on every row.
- ``ASH_6`` — height of the cored snow sample (cm).
- ``SH_TAG`` — daily snow depth (cm).
- ``NSH_TAG`` — new-snow depth (cm).
- ``RS`` — daily precipitation (mm).

Quirks
------
- water_equiv ``produkt`` files end at the station's last SWE day (the
  station list's ``bis_datum`` is that day too), so out of season the
  newest water_equiv row is months old while more_precip runs to
  yesterday.
- Every water_equiv station is also a more_precip station, and the two
  ``SH_TAG`` columns agreed on 86 681 of 86 683 shared 2025–26 rows.
- ``NSH_TAG`` was added to more_precip in January 2019 (CDC change log);
  older files lack the column, and the automated mountain synoptic
  stations (Zugspitze, Feldberg, Fichtelberg, Brocken) report none.
- The more_precip station list holds every precipitation station; some
  never report snow depth.

Quality flags
-------------
``QN_6`` is the row-level quality level (see :data:`DATA_FLAGS`);
``include_flags=True`` emits it as the record ``flag`` string.

Unverified
----------
- The ``RSF`` precipitation-form code in ``more_precip`` is not exposed:
  its code table is not given in the dataset description.
- The ``Abgabe`` column of the station list reads ``Frei`` ("free") or
  is blank; the meaning of a blank is not documented, but in Oct 2026 all
  39 blank-``Abgabe`` water_equiv stations had no data file at all.
- DWD docs give no explicit "active" definition; ``status`` here is
  derived from the station list's ``bis_datum``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily"
)

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: Product directory → file-name prefix and station-list file.
PRODUCTS: dict[str, dict[str, str]] = {
    "water_equiv": {
        "prefix": "Wa",
        "stations": "Wa_Tageswerte_Beschreibung_Stationen.txt",
        "title": "Daily observations of snow height and water equivalent",
    },
    "more_precip": {
        "prefix": "RR",
        "stations": "RR_Tageswerte_Beschreibung_Stationen.txt",
        "title": "Daily precipitation, snow depth and new snow",
    },
}

#: A station is Active when its station-list ``bis_datum`` is within this
#: many days of today.  Generous because SWE stops each spring: the
#: water_equiv list's ``bis_datum`` is the last SWE-season day (~April).
_ACTIVE_DAYS = 400

#: ``recent/`` holds roughly the last 500 days; skip it for windows that
#: end well before that.
_RECENT_SPAN_DAYS = 600

_MISSING = -999.0

# Plausibility bounds in output units (DESIGN.md: SWE/depth 0–15 m).
_MAX_PLAUSIBLE = {
    "swe": 1500.0,  # cm
    "snwd": 1500.0,  # cm
    "snowfall": 500.0,  # cm, one day
}

_SRC_WA = BASE_URL + "/water_equiv/"
_SRC_RR = BASE_URL + "/more_precip/"

# ── Public variable / flag tables ────────────────────────────────────────────

#: Variables this client knows, keyed by the native column name.
#: ``product`` lists the product(s) the column is read from, in priority
#: order; ``scale`` converts the stored value to ``output_units``.
VARIABLES: dict[str, dict] = {
    "WASH_6": {
        "name": "Snow Water Equivalent (total snow depth)",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "product": ["water_equiv"],
        "source": _SRC_WA,
        "description": (
            "Water equivalent of the total snow depth "
            "(Wasseräquivalent der Gesamtschneehöhe), 06 UTC observation."
        ),
        "notes": (
            "Manual snow-sampler measurement; stored in mm, returned in cm. "
            "Per DWD mostly not measured every day, and only above a minimum "
            "snow depth (often 5 cm). Zero is never reported — no measurement "
            "and no snow are both missing."
        ),
    },
    "WAAS_6": {
        "name": "Snow Water Equivalent (cored snow)",
        "type": "other",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "product": ["water_equiv"],
        "source": _SRC_WA,
        "description": (
            "Water equivalent of the cored snow sample "
            "(Wasseräquivalent ausgestochene Schneehöhe)."
        ),
        "notes": (
            "Stored in mm, returned in cm. Typed 'other' so that 'swe' "
            "resolves to WASH_6 only; in 2025-26 it equals WASH_6 on every row."
        ),
    },
    "ASH_6": {
        "name": "Height of Cored Snow",
        "type": "other",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "product": ["water_equiv"],
        "source": _SRC_WA,
        "description": (
            "Height of the snow sample cut out for the SWE measurement "
            "(Höhe ausgestochener Schnee)."
        ),
        "notes": "Usually equals SH_TAG on SWE days.",
    },
    "SH_TAG": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "product": ["more_precip", "water_equiv"],
        "source": _SRC_RR + " (fallback " + _SRC_WA + ")",
        "description": "Daily snow depth (Schneehöhe Tageswert), morning observation.",
        "notes": (
            "Read from more_precip (year-round, ~2 300 stations); stations "
            "absent from more_precip fall back to the water_equiv column. "
            "Manual snow stake at most stations, a few ultrasonic/laser sensors."
        ),
    },
    "NSH_TAG": {
        "name": "New Snow Depth",
        "type": "snowfall",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "product": ["more_precip"],
        "source": _SRC_RR,
        "description": "Depth of fresh snow (Neuschneehöhe) over the observation day.",
        "notes": "Stored and returned in cm.",
    },
    "RS": {
        "name": "Precipitation",
        "type": "precip",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "product": ["more_precip"],
        "source": _SRC_RR,
        "description": "Daily precipitation height.",
        "notes": "Stored and returned in mm.",
    },
}

#: Standardized type → native column(s), priority order.
_TYPE_TO_VARS: dict[str, list[str]] = {}
for _var, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_var)

#: Default variables when ``variables=None`` (DESIGN.md §3.4: snow only).
_DEFAULT_VARS: list[str] = ["WASH_6", "SH_TAG"]

#: ``QN_6`` quality levels (row-level), from the dataset descriptions.
#: Data up to 1980 reach at best QN 5, later data QN 10.
DATA_FLAGS: dict[str, str] = {
    "1": "Only formal control",
    "2": "Controlled with individually defined criteria",
    "3": "Automatic control and correction",
    "5": "Historic, subjective procedures",
    "7": "Second control done, before correction",
    "8": "Quality control outside ROUTINE",
    "9": "Not all parameters corrected",
    "10": "Quality control finished, all corrections finished",
}

# ── Parsers ──────────────────────────────────────────────────────────────────

_STATION_LINE = re.compile(
    r"^(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+(?:\.\d+)?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s(.*)$"
)

# Name and Bundesland are fixed 41-character fields after the longitude.
_NAME_W = 41
_STATE_W = 41


def _ymd(text: str) -> str:
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _pad_id(station_id: str | int) -> str:
    """Normalize a DWD station id to its 5-digit zero-padded form."""
    text = str(station_id).strip()
    if not text.isdigit():
        raise DWDError(f"Invalid DWD station id {station_id!r} — expected digits.")
    return text.zfill(5)


def _parse_station_list(text: str) -> dict[str, dict]:
    """Parse a ``*_Tageswerte_Beschreibung_Stationen.txt`` station list.

    Returns ``{id5: {station_id, name, latitude, longitude, elevation_m,
    state, release, begin_date, end_date}}``.  The two header lines and
    blank lines are skipped.  The numeric columns are whitespace-split
    (their widths differ between products); name and Bundesland are fixed
    41-character fields after the longitude.
    """
    out: dict[str, dict] = {}
    for line in text.splitlines():
        m = _STATION_LINE.match(line)
        if not m:
            continue
        sid, von, bis, elev, lat, lon, rest = m.groups()
        sid = sid.zfill(5)
        out[sid] = {
            "station_id": sid,
            "name": rest[:_NAME_W].strip(),
            "latitude": float(lat),
            "longitude": float(lon),
            "elevation_m": float(elev),
            "state": rest[_NAME_W : _NAME_W + _STATE_W].strip() or None,
            "release": rest[_NAME_W + _STATE_W :].strip() or None,
            "begin_date": _ymd(von),
            "end_date": _ymd(bis),
        }
    return out


_LISTING_RE = re.compile(
    r'href="(tageswerte_([A-Za-z]+)_(\d{5})_(?:(\d{8})_(\d{8})_hist|akt)\.zip)"'
)


def _parse_listing(html: str) -> dict[str, dict]:
    """Parse an Apache directory index into ``{id5: {file, begin, end}}``.

    ``begin``/``end`` (``YYYY-MM-DD``) come from historical file names and
    are ``None`` for recent (``_akt``) files.
    """
    out: dict[str, dict] = {}
    for m in _LISTING_RE.finditer(html):
        fname, _prefix, sid, b, e = m.groups()
        out[sid] = {
            "file": fname,
            "begin": _ymd(b) if b else None,
            "end": _ymd(e) if e else None,
        }
    return out


def _parse_produkt(raw: bytes) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Read the ``produkt_*.txt`` member of a station zip.

    Returns ``(columns, {YYYY-MM-DD: {column: float|None, "QN_6": str}})``.
    ``-999`` → ``None``.  Raises :class:`DWDError` on a bad archive.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise DWDError(f"Not a valid DWD station zip: {exc}") from exc
    members = [n for n in zf.namelist() if n.startswith("produkt")]
    if not members:
        raise DWDError("DWD station zip has no produkt_*.txt member")
    text = zf.read(members[0]).decode("latin-1")
    reader = csv.reader(io.StringIO(text), delimiter=";")
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration as exc:
        raise DWDError("Empty produkt file") from exc
    try:
        i_date = header.index("MESS_DATUM")
    except ValueError as exc:
        raise DWDError(f"produkt file lacks MESS_DATUM: {header}") from exc
    i_qn = header.index("QN_6") if "QN_6" in header else None
    value_cols = [
        (i, h)
        for i, h in enumerate(header)
        if h not in {"STATIONS_ID", "MESS_DATUM", "QN_6", "eor", ""}
    ]
    rows: dict[str, dict[str, Any]] = {}
    for rec in reader:
        if len(rec) <= i_date:
            continue
        day = rec[i_date].strip()
        if len(day) != 8 or not day.isdigit():
            continue
        row: dict[str, Any] = {}
        for i, h in value_cols:
            row[h] = _to_value(rec[i]) if i < len(rec) else None
        row["QN_6"] = rec[i_qn].strip() if i_qn is not None and i_qn < len(rec) else ""
        rows[_ymd(day)] = row
    return [h for _, h in value_cols], rows


def _to_value(text: str) -> float | None:
    try:
        v = float(text.strip())
    except ValueError:
        return None
    if v == _MISSING or v != v:
        return None
    return v


def _convert(var: str, stored: float | None) -> float | None:
    """Scale a stored value to output units, applying the snow clamps."""
    if stored is None:
        return None
    info = VARIABLES[var]
    value = round(stored * info["scale"], 3)
    bound = _MAX_PLAUSIBLE.get(info["type"])
    if bound is not None and not (0.0 <= value <= bound):
        return None
    return value


def _merge(
    hist: dict[str, dict[str, Any]] | None,
    recent: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Union of historical and recent rows; historical wins on overlap."""
    merged: dict[str, dict[str, Any]] = dict(recent or {})
    merged.update(hist or {})
    return merged


# ── Client ───────────────────────────────────────────────────────────────────


class DWDClient:
    """
    Client for DWD CDC daily snow water equivalent and snow depth.

    No credentials are needed.

    Parameters
    ----------
    base_url : str
        Root of the CDC ``observations_germany/climate/daily`` tree.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session or None
        Optional pre-configured session.
    cache : bool
        Cache historical station zips on disk (they never change for a
        given file name).  Recent zips are never cached.
    cache_path : str or Path or None
        Override the cache directory (default
        ``easysnowdata.config.cache_dir("stations", "dwd")``).
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
        cache: bool = True,
        cache_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.cache = cache
        self._cache_path = Path(cache_path) if cache_path else None
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._station_lists: dict[str, dict[str, dict]] = {}
        self._listings: dict[tuple[str, str], dict[str, dict]] = {}

    # ── Directory / station-list access ──────────────────────────────────────

    def _station_list(self, product: str) -> dict[str, dict]:
        if product not in self._station_lists:
            fname = PRODUCTS[product]["stations"]
            raw = self._get(f"{product}/recent/{fname}").content
            self._station_lists[product] = _parse_station_list(raw.decode("latin-1"))
            logger.info(
                "DWD %s: parsed %d stations", product, len(self._station_lists[product])
            )
        return self._station_lists[product]

    def _listing(self, product: str, kind: str) -> dict[str, dict]:
        key = (product, kind)
        if key not in self._listings:
            html = self._get(f"{product}/{kind}/").text
            self._listings[key] = _parse_listing(html)
        return self._listings[key]

    def _cache_dir(self, product: str) -> Path:
        if self._cache_path is not None:
            path = self._cache_path / product
            path.mkdir(parents=True, exist_ok=True)
            return path
        from easysnowdata.config import cache_dir  # noqa: PLC0415

        return cache_dir("stations", "dwd", product)

    def _fetch_zip(self, product: str, kind: str, fname: str) -> bytes:
        path = f"{product}/{kind}/{fname}"
        if kind == "historical" and self.cache:
            target = self._cache_dir(product) / fname
            if target.exists():
                return target.read_bytes()
            raw = self._get(path).content
            tmp = target.with_suffix(".part")
            tmp.write_bytes(raw)
            tmp.replace(target)
            return raw
        return self._get(path).content

    # ── Public API — stations ────────────────────────────────────────────────

    def get_stations(
        self,
        products: list[str] | tuple[str, ...] | str = ("water_equiv", "more_precip"),
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Stations in the given product station lists, merged by id.

        Parameters
        ----------
        products : list[str] or str
            Any of ``"water_equiv"`` (SWE stations) and ``"more_precip"``
            (precipitation stations reporting snow depth / new snow).
        active_only : bool
            Keep stations whose latest ``bis_datum`` is within ~400 days.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (5-digit), ``name``, ``latitude``, ``longitude``,
            ``elevation_m``, ``status``, ``state`` (Bundesland), ``operator``,
            ``measurement_method``, ``release``, ``has_swe``, ``has_snwd``,
            ``products`` (``{product: {begin_date, end_date}}``),
            ``begin_date``, ``end_date``, ``swe_begin_date``,
            ``swe_end_date`` and ``station_url`` (``None`` — DWD has no
            per-station page).
        """
        wanted = _coerce_list(products)
        unknown = set(wanted) - set(PRODUCTS)
        if unknown:
            raise DWDError(
                f"Unknown DWD product(s) {sorted(unknown)} — expected some of "
                f"{sorted(PRODUCTS)}."
            )
        merged: dict[str, dict] = {}
        for product in PRODUCTS:  # fixed order: water_equiv attributes win
            if product not in wanted:
                continue
            for sid, sta in self._station_list(product).items():
                entry = merged.get(sid)
                if entry is None:
                    entry = {
                        k: sta[k]
                        for k in (
                            "station_id",
                            "name",
                            "latitude",
                            "longitude",
                            "elevation_m",
                            "state",
                            "release",
                        )
                    }
                    entry["products"] = {}
                    merged[sid] = entry
                entry["products"][product] = {
                    "begin_date": sta["begin_date"],
                    "end_date": sta["end_date"],
                }
        out = [self._finish_station(e) for e in merged.values()]
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    @staticmethod
    def _finish_station(entry: dict) -> dict:
        prods = entry["products"]
        begins = [p["begin_date"] for p in prods.values()]
        ends = [p["end_date"] for p in prods.values()]
        end = max(ends)
        cutoff = (date.today() - timedelta(days=_ACTIVE_DAYS)).isoformat()
        wa = prods.get("water_equiv")
        entry.update(
            {
                "status": "Active" if end >= cutoff else "Inactive",
                "operator": "Deutscher Wetterdienst (DWD) or partner network",
                "measurement_method": (
                    "manual snow-sampler SWE and snow-stake depth"
                    if wa
                    else "snow depth (mostly manual snow stake)"
                ),
                "has_swe": wa is not None,
                # water_equiv rows always carry SH_TAG; more_precip lists
                # every precipitation station, and SH_TAG may be all
                # missing at some of them.
                "has_snwd": True,
                "begin_date": min(begins),
                "end_date": end,
                "swe_begin_date": wa["begin_date"] if wa else None,
                "swe_end_date": wa["end_date"] if wa else None,
                "station_url": None,
            }
        )
        return entry

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """All SWE (water_equiv) and snow-depth (more_precip) stations.

        See :meth:`get_stations` for the schema and for one product only.
        """
        return self.get_stations(active_only=active_only, bbox=bbox)

    def get_metadata(self, station_id: str | int) -> dict:
        """
        Metadata and variable inventory for one station.

        Returns the :meth:`get_stations` dict plus ``data_variables``: one
        entry per variable available from the station's products, with
        ``name``, ``type``, ``interval``, ``units`` (emitted),
        ``description``, ``notes``, ``begin_date``/``end_date`` (from the
        product station list: the period of record of the product, not
        of the individual column) and ``n_obs`` (``None``).

        Raises
        ------
        DWDError
            If the station is in neither station list.
        """
        sid = _pad_id(station_id)
        matches = [s for s in self.get_stations() if s["station_id"] == sid]
        if not matches:
            raise DWDError(f"Station {sid!r} not found in DWD station lists")
        sta = matches[0]
        dvars = []
        for var, info in VARIABLES.items():
            prod = next((p for p in info["product"] if p in sta["products"]), None)
            if prod is None:
                continue
            dvars.append(
                {
                    "name": var,
                    "type": info["type"],
                    "interval": "daily",
                    "units": info["output_units"],
                    "description": info["description"],
                    "notes": info["notes"] + f" Product: {prod}.",
                    "begin_date": sta["products"][prod]["begin_date"],
                    "end_date": sta["products"][prod]["end_date"],
                    "n_obs": None,
                }
            )
        sta["data_variables"] = dvars
        return sta

    # ── Public API — data ────────────────────────────────────────────────────

    def get_station_rows(
        self,
        station_id: str | int,
        product: str,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Merged historical + recent rows for one station and product.

        Returns ``{YYYY-MM-DD: {column: stored value or None, "QN_6": str}}``
        in native units.  Historical rows win on overlapping dates.  Files
        that cannot hold the window (by historical file name / recent span)
        are not downloaded.  Returns ``{}`` when the station has no file.
        """
        if product not in PRODUCTS:
            raise DWDError(f"Unknown DWD product {product!r}")
        sid = _pad_id(station_id)
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None

        hist_rows = recent_rows = None
        hist = self._listing(product, "historical").get(sid)
        if hist and not (
            (begin and hist["end"] and hist["end"] < begin)
            or (end and hist["begin"] and hist["begin"] > end)
        ):
            _, hist_rows = _parse_produkt(
                self._fetch_zip(product, "historical", hist["file"])
            )
        recent_cutoff = (date.today() - timedelta(days=_RECENT_SPAN_DAYS)).isoformat()
        hist_covers = bool(hist and hist["end"] and end and hist["end"] >= end)
        if not hist_covers and not (end and end < recent_cutoff):
            rec = self._listing(product, "recent").get(sid)
            if rec:
                _, recent_rows = _parse_produkt(
                    self._fetch_zip(product, "recent", rec["file"])
                )
        rows = _merge(hist_rows, recent_rows)
        return {
            d: r
            for d, r in rows.items()
            if (begin is None or d >= begin) and (end is None or d <= end)
        }

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
            DWD station id(s), with or without zero padding (``"00020"``,
            ``"20"``, ``20``).  Required unless ``bbox`` is provided.
        variables : list[str] or str or None
            Native columns (``"WASH_6"``) or standardized types
            (``"swe"``, ``"snwd"``, ``"snowfall"``, ``"precip"``).
            ``None`` means snow only: ``WASH_6`` and ``SH_TAG``.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds.
        interval : str
            Only ``"daily"`` is supported.
        include_flags : bool
            Add ``flag`` = the row's ``QN_6`` quality level.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            records (plus ``flag``), only for dates with a non-missing value.
            Stations whose files cannot be fetched are logged and skipped.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        DWDError
            On an unknown variable or interval.
        """
        if str(interval).lower() != "daily":
            raise DWDError(
                f"Unsupported interval {interval!r} for DWD — only 'daily' exists."
            )
        variables_ = _resolve_variables(variables)

        if station_ids is None and bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        elif station_ids is not None:
            ids = [_pad_id(s) for s in _coerce_list(station_ids)]
        else:
            raise ValueError("Provide station_ids or bbox.")

        records: list[dict] = []
        for sid in ids:
            cache: dict[str, dict[str, dict[str, Any]]] = {}
            for var in variables_:
                prod = self._product_for(sid, var)
                if prod is None:
                    continue
                if prod not in cache:
                    try:
                        cache[prod] = self.get_station_rows(
                            sid, prod, begin_date, end_date
                        )
                    except DWDError as exc:
                        logger.warning(
                            "DWD: skipping %s/%s for station %s: %s",
                            prod,
                            var,
                            sid,
                            exc,
                        )
                        cache[prod] = {}
                records.extend(
                    self._records(sid, var, cache[prod], include_flags=include_flags)
                )
        records.sort(key=lambda r: (r["station_id"], r["variable"], r["date"]))
        return records

    def _product_for(self, sid: str, var: str) -> str | None:
        """First product (in the variable's priority order) holding the station."""
        for prod in VARIABLES[var]["product"]:
            if sid in self._listing(prod, "recent") or sid in self._listing(
                prod, "historical"
            ):
                return prod
        return None

    @staticmethod
    def _records(
        sid: str,
        var: str,
        rows: dict[str, dict[str, Any]],
        *,
        include_flags: bool,
    ) -> list[dict]:
        info = VARIABLES[var]
        out: list[dict] = []
        for day in sorted(rows):
            row = rows[day]
            if var not in row or row[var] is None:
                continue
            rec: dict[str, Any] = {
                "station_id": sid,
                "date": day,
                "variable": var,
                "type": info["type"],
                "value": _convert(var, row[var]),
                "units": info["output_units"],
                "interval": "daily",
            }
            if include_flags:
                rec["flag"] = row.get("QN_6", "")
            out.append(rec)
        return out

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get(self, path: str) -> requests.Response:
        return request_with_retries(
            self._session,
            f"{self.base_url}/{path}",
            error_cls=DWDError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )


# ── Exception ────────────────────────────────────────────────────────────────


class DWDError(Exception):
    """Raised when a DWD CDC file cannot be fetched or parsed."""


# ── Private helpers ──────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Translate native columns and/or standardized types to native columns."""
    if variables is None:
        return list(_DEFAULT_VARS)
    raw = [variables] if isinstance(variables, str) else list(variables)
    if not raw:
        return list(_DEFAULT_VARS)
    out: list[str] = []
    for v in raw:
        if v in VARIABLES:
            candidates = [v]
        elif v in _TYPE_TO_VARS and v != "other":
            candidates = _TYPE_TO_VARS[v]
        else:
            # No silent fallback (DESIGN.md §3.6)
            raise DWDError(
                f"Unknown variable {v!r} for DWD — expected one of "
                f"{sorted(VARIABLES)} or a standardized type "
                f"({sorted(t for t in _TYPE_TO_VARS if t != 'other')})."
            )
        out.extend(c for c in candidates if c not in out)
    return out
