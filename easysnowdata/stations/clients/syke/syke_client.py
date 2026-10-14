"""
clients/syke/syke_client.py
===========================
Python client for the Finnish Environment Institute (Syke) hydrology OData
API ("Hydrologiarajapinta").

Primary use case: Finnish snow-course snow water equivalent (quantity ``S``,
"Lumilinjamittaus" / "Snow course measurement") — manual snow-tube surveys
along a fixed course, normally once a month from November to May.  The same
API also serves Syke's areal snow water equivalent (quantity ``SA``,
"Lumen aluevesiarvo"), a catchment-average SWE produced on the 1st and 16th
of each month; it is not a point measurement, so it is exposed separately
through :meth:`SYKEClient.get_areal_units` / :meth:`SYKEClient.get_areal_swe`.

Data source       : https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1/odata/
Schema            : ``$metadata`` of the same root (OData v3, WCF Data
                    Services; JSON via ``Accept: application/json``)
Authentication    : none
Licence           : Syke open data, Creative Commons Attribution 4.0 (CC BY
                    4.0) — https://www.syke.fi/en/environmental-data/use-license-and-responsibilities
                    and the avoindata.fi "Hydrologiarajapinta" listing.
                    Attribute "Finnish Environment Institute (Syke)".

Tables used
-----------
- ``Paikka``    — sites.  ``Suure_Id`` 8 = snow course, 7 = areal SWE unit.
  Coordinates come in three systems: ``KoordErTmPohj``/``KoordErTmIta``
  (ETRS-TM35FIN, EPSG:3067, used here), ``KoordYk*`` (KKJ uniform
  coordinate system, EPSG:2393) and ``KoordLat``/``KoordLong`` as
  ``DDMMSS`` text on the KKJ datum (~175 m from WGS84 — do not read it as
  WGS84).
  There is **no elevation** column, so ``elevation_m`` is always ``None``.
  ``Tila_Id`` is the site state (see :data:`SITE_STATES`).
- ``LumiPiste`` — snow-course values: ``Paikka_Id, Aika, Arvo (mm), Lippu_Id``.
  One value per survey: SWE only.  Snow depth and density are *not*
  published by this API.
- ``LumiAlue``  — areal SWE values: ``Paikka_Id, Aika, Arvo (mm)``.

Paging
------
The server returns at most 500 rows per response and appends
``odata.nextLink`` (``…&$skip=500``) when more remain; the client follows
it until exhausted.  ``$count`` is not supported (``$inlinecount=allpages``
is, and gives ``odata.count``).  OData v3 has no ``in`` operator, so
station ids are sent as ``or`` chains in chunks of 15 — the server rejects
filters over 100 parse nodes with HTTP 400.

Design principles
-----------------
- Metric-first: SWE is published in mm and returned in cm (× 0.1).
- ``station_id`` is the API's ``Paikka_Id`` (as a string); the official
  site number is ``site_number`` (``Nro``).
- ``status`` is derived from ``Tila``: "Jatkuva" (continuous) and
  "Määräaikainen" (fixed-term) are Active; "Lopetettu" (discontinued),
  "Satunnainen" (occasional) and "Ei poiminnassa" are Inactive.
- Values whose ``Lippu`` marks a model calculation (``Lumimallilaskenta``)
  rather than a measurement are nulled unless ``keep_modelled=True``.  None
  were present in October 2026 (every row was flag 96, "Mittaus").
- SWE outside 0–15 m is nulled.
- Snow courses support only ``interval="periodic"``; the areal product
  only ``"semi_monthly"``.
"""

from __future__ import annotations

import logging
from datetime import date
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

BASE_URL = "https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1/odata"

_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: ``Suure_Id`` of the quantities this client serves.
SUURE_SNOW_COURSE = 8  # "S"  — LumiPiste
SUURE_AREAL_SWE = 7  # "SA" — LumiAlue

#: Ids per ``or``-chain filter.  The server enforces an OData node-count
#: limit of 100 ("The node count limit of '100' has been exceeded"): 18 ids
#: plus two date bounds pass, 19 fail.  15 leaves headroom.
_ID_CHUNK = 15

#: Plausibility bound for SWE, cm (DESIGN.md: 0–15 m).
_MAX_SWE_CM = 1500.0

_SOURCE = "Finnish Environment Institute (Syke) Hydrologiarajapinta — " + BASE_URL

#: Native ``Tila`` (site state) table, ``Tila_Id`` → (Finnish name, English).
SITE_STATES: dict[int, tuple[str, str]] = {
    1: ("Jatkuva", "continuous"),
    2: ("Lopetettu", "discontinued"),
    3: ("Satunnainen", "occasional"),
    4: ("Ei poiminnassa", "not collected"),
    5: ("Määräaikainen", "fixed-term"),
}
_ACTIVE_STATES = frozenset({1, 5})

#: ``Laitteisto`` (equipment) entries seen on snow-course sites.
EQUIPMENT: dict[int, str] = {
    12: "Mittauksista laskettu (computed from measurements)",
    14: "Lumipuntari (snow-tube scale)",
    16: "Lumipuntari, pienet alueet (snow-tube scale, small catchments)",
    17: "Lumipuntari, Värriö (snow-tube scale, Värriö)",
}

# ── Public variable / flag tables ────────────────────────────────────────────

#: Point variables served by :meth:`SYKEClient.get_data`, keyed by the
#: native Syke quantity code (``Suure.Suurekoodi``).
VARIABLES: dict[str, dict] = {
    "S": {
        "name": "Snow Water Equivalent (snow course)",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "source": _SOURCE + "/LumiPiste",
        "description": (
            "Snow water equivalent from a manual snow-course survey "
            "(Lumilinjamittaus), table LumiPiste."
        ),
        "notes": (
            "Published in mm; returned in cm (× 0.1). Typically monthly, "
            "mid-month, November–May. Snow depth and density are not "
            "published by the API."
        ),
    },
}

#: Areal product served by :meth:`SYKEClient.get_areal_swe` (not a point).
AREAL_VARIABLES: dict[str, dict] = {
    "SA": {
        "name": "Areal Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "source": _SOURCE + "/LumiAlue",
        "description": (
            "Catchment-average snow water equivalent (Lumen aluevesiarvo), "
            "table LumiAlue."
        ),
        "notes": (
            "Published in mm; returned in cm. Values on the 1st and 16th of "
            "each month, year round (0 in summer). Derived product; the "
            "Paikka coordinates are the reference point of the unit, not a "
            "measurement site. The derivation method is not documented in "
            "the API."
        ),
    },
}

#: ``Lippu`` (flag) codes that occur, or may occur, on snow-course values,
#: keyed by ``Lippu_Id`` as a string (what ``flag`` carries).
DATA_FLAGS: dict[str, str] = {
    "96": "Mittaus — snow course measurement result",
    "110": "Lumimallilaskenta — value calculated with the snow model",
    "117": (
        "Lumimallilaskenta 2022 — value calculated with the snow-course "
        "model introduced in autumn 2022"
    ),
}

#: Flags marking modelled rather than measured values.
_MODELLED_FLAGS = frozenset({"110", "117"})

_TYPE_TO_VARS: dict[str, list[str]] = {}
for _code, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_code)

_DEFAULT_VARS: list[str] = ["S"]


class SYKEError(Exception):
    """Raised for Syke hydrology API failures and bad requests."""


# ── Parsers ──────────────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Map native codes or standardized types to native codes."""
    items = _coerce_list(variables) if variables is not None else []
    if not items:
        return list(_DEFAULT_VARS)
    out: list[str] = []
    for v in items:
        if v in VARIABLES:
            codes = [v]
        elif v.upper() in VARIABLES:
            codes = [v.upper()]
        elif v.lower() in _TYPE_TO_VARS:
            codes = _TYPE_TO_VARS[v.lower()]
        else:
            raise SYKEError(
                f"Unknown Syke variable {v!r} — expected one of "
                f"{sorted(VARIABLES)} or types {sorted(_TYPE_TO_VARS)}. "
                "Snow depth and density are not published by this API; "
                "areal SWE ('SA') is served by get_areal_swe()."
            )
        for c in codes:
            if c not in out:
                out.append(c)
    return out


def _dms_to_deg(text: Any) -> float | None:
    """Parse Syke's ``DDMMSS`` (``DDDMMSS``) coordinate text to degrees."""
    if text is None:
        return None
    s = str(text).strip()
    if not s.isdigit() or len(s) < 6:
        return None
    deg, mins, secs = int(s[:-4]), int(s[-4:-2]), int(s[-2:])
    if mins >= 60 or secs >= 60:
        return None
    return round(deg + mins / 60 + secs / 3600, 6)


_TRANSFORMERS: dict[str, Any] = {}


def _to_wgs84(crs: str, x: Any, y: Any) -> tuple[float | None, float | None]:
    """Transform ``(x, y)`` in ``crs`` to WGS84 ``(lat, lon)``; None if absent."""
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None, None
    if not xf or not yf:
        return None, None
    if crs not in _TRANSFORMERS:
        from pyproj import Transformer

        _TRANSFORMERS[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = _TRANSFORMERS[crs].transform(xf, yf)
    return round(lat, 6), round(lon, 6)


def _tm35fin_to_wgs84(northing: Any, easting: Any) -> tuple[float | None, float | None]:
    """Convert ETRS-TM35FIN (EPSG:3067) to WGS84 (lat, lon)."""
    return _to_wgs84("EPSG:3067", easting, northing)


def _site_coords(row: dict) -> tuple[float | None, float | None]:
    """Best WGS84 position for a ``Paikka`` row.

    Order: ETRS-TM35FIN (``KoordErTm*``); KKJ uniform coordinates
    (``KoordYk*``, EPSG:2393), which agree with it to ~2 m; and last the
    ``KoordLat``/``KoordLong`` DDMMSS text, which is on the old KKJ datum
    (EPSG:4123) — read as WGS84 it is off by ~175 m, so it is transformed.
    """
    lat, lon = _tm35fin_to_wgs84(row.get("KoordErTmPohj"), row.get("KoordErTmIta"))
    if lat is None:
        lat, lon = _to_wgs84("EPSG:2393", row.get("KoordYkIta"), row.get("KoordYkPohj"))
    if lat is None:
        dlat, dlon = _dms_to_deg(row.get("KoordLat")), _dms_to_deg(row.get("KoordLong"))
        if dlat is not None and dlon is not None:
            lat, lon = _to_wgs84("EPSG:4123", dlon, dlat)
    return lat, lon


def _parse_site(row: dict, owners: dict[int, str] | None = None) -> dict:
    """Turn a ``Paikka`` row into a station dict."""
    lat, lon = _site_coords(row)
    tila = row.get("Tila_Id")
    state = SITE_STATES.get(tila, (None, None))
    owner_id = row.get("Omistaja_Id")
    return {
        "station_id": str(row["Paikka_Id"]),
        "name": (row.get("Nimi") or "").strip(),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": None,
        "status": "Active" if tila in _ACTIVE_STATES else "Inactive",
        "status_native": state[0],
        "status_description": state[1],
        "site_number": (row.get("Nro") or "").strip() or None,
        "municipality": row.get("KuntaNimi"),
        "basin": row.get("PaaVesalNimi"),
        "sub_basin": row.get("VesalNimi"),
        "operator": (owners or {}).get(owner_id, "Finnish Environment Institute"),
        "owner_id": owner_id,
        "measurement_method": EQUIPMENT.get(row.get("Laitteisto_Id")),
        "equipment_id": row.get("Laitteisto_Id"),
        "etrs_tm35fin_n": row.get("KoordErTmPohj"),
        "etrs_tm35fin_e": row.get("KoordErTmIta"),
        "coordinate_accuracy_code": row.get("KoordTarkkuus"),
        "station_url": None,
    }


def _convert(raw: Any) -> float | None:
    """mm string/decimal → cm, nulling values outside 0–15 m."""
    v = _to_float(raw)
    if v is None:
        return None
    cm = round(v * 0.1, 3)
    if not (0.0 <= cm <= _MAX_SWE_CM):
        return None
    return cm


def _odata_dt(d: str) -> str:
    return f"datetime'{d}T00:00:00'"


def _build_filter(
    ids: list[str] | None, begin: str | None, end: str | None
) -> str | None:
    parts: list[str] = []
    if ids:
        parts.append("(" + " or ".join(f"Paikka_Id eq {int(i)}" for i in ids) + ")")
    if begin:
        parts.append(f"Aika ge {_odata_dt(begin)}")
    if end:
        parts.append(f"Aika le {_odata_dt(end)}")
    return " and ".join(parts) or None


# ── Client ───────────────────────────────────────────────────────────────────


class SYKEClient:
    """
    Client for the Syke hydrology OData API (snow courses and areal SWE).

    No credentials are needed.

    Parameters
    ----------
    base_url : str
        OData service root.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session or None
        Optional pre-configured session.
    keep_modelled : bool
        If False (default), snow-course values flagged as snow-model output
        (``Lippu_Id`` 110/117) are returned as ``None``.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
        keep_modelled: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.keep_modelled = keep_modelled
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "easysnowdata", "Accept": "application/json"}
        )
        self._sites_cache: dict[int, list[dict]] = {}
        self._owners_cache: dict[int, str] | None = None

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        resp = request_with_retries(
            self._session,
            url,
            params=params,
            error_cls=SYKEError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        try:
            return resp.json()
        except ValueError as exc:
            raise SYKEError(
                f"Non-JSON response from {url}: {resp.text[:200]!r}"
            ) from exc

    def _query(self, table: str, params: dict | None = None) -> list[dict]:
        """GET ``table`` and follow ``odata.nextLink`` to the end."""
        url: str | None = f"{self.base_url}/{table}"
        rows: list[dict] = []
        p = params
        while url:
            payload = self._get_json(url, p)
            if "value" not in payload:
                raise SYKEError(f"Unexpected payload from {url}: {str(payload)[:200]}")
            rows.extend(payload["value"])
            url = payload.get("odata.nextLink") or payload.get("@odata.nextLink")
            p = None  # nextLink already carries the query
        return rows

    # ── Sites ────────────────────────────────────────────────────────────────

    def _owners(self) -> dict[int, str]:
        if self._owners_cache is None:
            try:
                rows = self._query("Omistaja")
                self._owners_cache = {r["Omistaja_Id"]: r["Nimi"] for r in rows}
            except SYKEError as exc:
                logger.warning("Syke: could not read Omistaja table: %s", exc)
                self._owners_cache = {}
        return self._owners_cache

    def _sites(self, suure_id: int) -> list[dict]:
        if suure_id not in self._sites_cache:
            rows = self._query("Paikka", {"$filter": f"Suure_Id eq {suure_id}"})
            owners = self._owners()
            self._sites_cache[suure_id] = [_parse_site(r, owners) for r in rows]
            logger.info("Syke: %d sites for Suure_Id %d", len(rows), suure_id)
        return self._sites_cache[suure_id]

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        All Syke snow courses (``Suure_Id`` 8).

        Parameters
        ----------
        active_only : bool
            Keep sites whose ``Tila`` is continuous or fixed-term.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            ``station_id`` (Paikka_Id), ``name``, ``latitude``,
            ``longitude``, ``elevation_m`` (always ``None``), ``status``,
            ``status_native``, ``site_number``, ``municipality``, ``basin``,
            ``sub_basin``, ``operator``, ``measurement_method``,
            ETRS-TM35FIN coordinates and ``station_url`` (``None``).  The
            period of record is not in the site table; use
            :meth:`get_metadata`.

        Raises
        ------
        SYKEError
            On an API failure.
        """
        sites = [dict(s) for s in self._sites(SUURE_SNOW_COURSE)]
        if active_only:
            sites = [s for s in sites if s["status"] == "Active"]
        return _filter_by_bbox(sites, bbox)

    def get_areal_units(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Areal-SWE units (``Suure_Id`` 7).

        Parameters
        ----------
        active_only : bool
            Keep units whose ``Tila`` is continuous or fixed-term.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        list[dict]
            Same schema as :meth:`get_all_stations`.

        Raises
        ------
        SYKEError
            On an API failure.
        """
        sites = [dict(s) for s in self._sites(SUURE_AREAL_SWE)]
        if active_only:
            sites = [s for s in sites if s["status"] == "Active"]
        return _filter_by_bbox(sites, bbox)

    def _por(self, table: str, paikka_id: int) -> tuple[str | None, str | None, int]:
        """First date, last date and row count for one site."""
        flt = f"Paikka_Id eq {paikka_id}"
        first = self._get_json(
            f"{self.base_url}/{table}",
            {
                "$filter": flt,
                "$orderby": "Aika asc",
                "$top": "1",
                "$inlinecount": "allpages",
            },
        )
        rows = first.get("value") or []
        n = int(first.get("odata.count") or 0)
        if not rows:
            return None, None, 0
        last = self._get_json(
            f"{self.base_url}/{table}",
            {"$filter": flt, "$orderby": "Aika desc", "$top": "1"},
        )
        lrows = last.get("value") or rows
        return rows[0]["Aika"][:10], lrows[0]["Aika"][:10], n

    def get_metadata(self, station_id: str) -> dict:
        """
        Site metadata plus ``data_variables`` (with period of record).

        Works for snow courses and for areal-SWE units (``variable`` "SA",
        ``interval`` "semi_monthly").

        Parameters
        ----------
        station_id : str
            ``Paikka_Id`` of a snow course or areal-SWE unit.

        Returns
        -------
        dict
            The station dict plus ``kind`` (``"snow_course"`` / ``"areal"``)
            and ``data_variables``: one entry with ``name``, ``type``,
            ``interval``, ``units`` (emitted), ``description``, ``notes``,
            ``begin_date``, ``end_date`` and ``n_obs`` (from the data
            table).

        Raises
        ------
        SYKEError
            If the id is neither a snow course nor an areal-SWE unit, or on
            an API failure.
        """
        sid = str(station_id)
        for suure, table, reg, interval in (
            (SUURE_SNOW_COURSE, "LumiPiste", VARIABLES, "periodic"),
            (SUURE_AREAL_SWE, "LumiAlue", AREAL_VARIABLES, "semi_monthly"),
        ):
            match = [s for s in self._sites(suure) if s["station_id"] == sid]
            if not match:
                continue
            sta = dict(match[0])
            begin, end, n = self._por(table, int(sid))
            code, info = next(iter(reg.items()))
            sta["kind"] = "snow_course" if suure == SUURE_SNOW_COURSE else "areal"
            sta["data_variables"] = [
                {
                    "name": code,
                    "type": info["type"],
                    "interval": interval,
                    "units": info["output_units"],
                    "description": info["description"],
                    "notes": info["notes"],
                    "begin_date": begin,
                    "end_date": end,
                    "n_obs": n,
                }
            ]
            return sta
        raise SYKEError(f"Station {sid!r} not found among Syke snow sites")

    # ── Data ─────────────────────────────────────────────────────────────────

    def _fetch_values(
        self,
        table: str,
        ids: list[str],
        begin: str | None,
        end: str | None,
    ) -> list[dict]:
        rows: list[dict] = []
        for ids_chunk in _chunk(ids, _ID_CHUNK):
            params = {"$orderby": "Paikka_Id,Aika"}
            flt = _build_filter(ids_chunk, begin, end)
            if flt:
                params["$filter"] = flt
            rows.extend(self._query(table, params))
        return rows

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "periodic",
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Snow-course SWE as flat records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            Paikka_Id(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            ``"S"`` or ``"swe"``; ``None`` = ``["S"]``.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds (applied server side).
        interval : str
            Only ``"periodic"``.
        include_flags : bool
            Add ``flag`` = the ``Lippu_Id`` (see :data:`DATA_FLAGS`).

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``flag``).

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        SYKEError
            On an unknown variable, unsupported interval, non-integer
            station id or API failure.

        Example
        -------
        >>> client = SYKEClient()
        >>> stations = client.get_all_stations(active_only=True)
        >>> records = client.get_data(
        ...     station_ids=stations[0]["station_id"],
        ...     begin_date="2024-01-01",
        ...     end_date="2024-05-31",
        ... )
        >>> records[0]["units"]
        'cm'
        """
        if interval.lower() != "periodic":
            raise SYKEError(
                f"Unsupported interval {interval!r} for Syke snow courses — "
                "only 'periodic' exists (areal SWE: get_areal_swe())."
            )
        codes = _resolve_variables(variables)
        if station_ids is not None:
            ids = _coerce_list(station_ids)
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        for i in ids:
            if not i.strip().isdigit():
                raise SYKEError(f"Syke station ids are integers (Paikka_Id); got {i!r}")
        if not ids or "S" not in codes:
            return []
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        rows = self._fetch_values("LumiPiste", ids, begin, end)
        return [self._record(r, "S", "periodic", include_flags) for r in rows]

    def get_areal_swe(
        self,
        area_ids: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        active_only: bool = False,
    ) -> list[dict]:
        """
        Areal (catchment-average) SWE, ``interval="semi_monthly"``.

        Parameters
        ----------
        area_ids : list[str] or str or None
            Areal-unit ``Paikka_Id`` values (see :meth:`get_areal_units`).
            With neither ``area_ids`` nor ``bbox`` every unit is fetched
            (≈116 000 rows for the full record — pass dates).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds (applied server side).
        active_only : bool
            When resolving units from ``bbox`` / all units, keep active ones.

        Returns
        -------
        list[dict]
            ``{station_id, date, variable ("SA"), type, value (cm), units,
            interval ("semi_monthly")}`` records.

        Raises
        ------
        SYKEError
            On an API failure.
        """
        if area_ids is not None:
            ids = _coerce_list(area_ids)
        else:
            ids = [
                s["station_id"]
                for s in self.get_areal_units(active_only=active_only, bbox=bbox)
            ]
        if not ids:
            return []
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        rows = self._fetch_values("LumiAlue", ids, begin, end)
        return [self._record(r, "SA", "semi_monthly", False) for r in rows]

    def _record(self, row: dict, code: str, interval: str, include_flags: bool) -> dict:
        info = VARIABLES.get(code) or AREAL_VARIABLES[code]
        flag = row.get("Lippu_Id")
        flag_s = str(flag) if flag is not None else None
        value = _convert(row.get("Arvo"))
        if flag_s in _MODELLED_FLAGS and not self.keep_modelled:
            value = None
        rec = {
            "station_id": str(row["Paikka_Id"]),
            "date": str(row["Aika"])[:10],
            "variable": code,
            "type": info["type"],
            "value": value,
            "units": info["output_units"],
            "interval": interval,
        }
        if include_flags:
            rec["flag"] = flag_s
        return rec
