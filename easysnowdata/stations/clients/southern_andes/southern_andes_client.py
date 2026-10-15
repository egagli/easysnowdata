"""
clients/southern_andes/southern_andes_client.py
===============================================
Python client for the Southern Andes Daily Snow Depth Dataset (2010–2024),
a quality-controlled compilation of automatic snow-depth stations in Chile
and Argentina (Medina Mendoza & Caro, 2026, Zenodo; described in Caro et
al., 2026, ESSD 18, 7097–7123, https://doi.org/10.5194/essd-18-7097-2026).

Primary use case: daily snow depth across the Andes from 21° S to 54° S
(30–5,804 m), from the networks of the Dirección General de Aguas (DGA,
Chile), IANIGLA (Argentina), CEAZA (Chile) and CIEP (Chile).  It is the
only southern-hemisphere station dataset in easysnowdata.  There is **no
SWE**: every sensor is an ultrasonic depth gauge (Campbell SR50A, at most
DGA stations replaced by a Lufft SHM31).

Data source       : Zenodo concept record 19560127
                    (https://doi.org/10.5281/zenodo.19560127); the version
                    checked here is record 21577171 (published 2026-07-25):
                    ``Southern_Andes_Snow_Depth_Dataset_v4.2.csv`` (0.9 MB)
                    and ``Station_metadata_for_snow_depth_observations.csv``.
Format reference  : Caro et al. (2026), Sect. 3 (compilation and the
                    four-stage quality control) and the Zenodo description.
Authentication    : none
Licence           : CC BY 4.0 (Zenodo record).  Cite the dataset
                    (doi:10.5281/zenodo.21577171) and the paper.

Files used
----------
The latest version is resolved through the Zenodo API
(``/api/records/19560127/versions/latest``); pass ``record_id=`` to pin
one.  The two CSVs are downloaded once (md5-checked) into
``easysnowdata.config.cache_dir("stations", "southern_andes") / <record id>``
and read from there.  A new Zenodo version is picked up automatically.  If
Zenodo cannot be reached, the newest cached version is used.  ``path=``
accepts a folder holding the two CSVs instead.

- The data file is **wide**: a ``date`` column (``YYYY-MM-DD``, every day
  from 2010-06-11 to 2024-12-31) and one column per station, headed by the
  station id.  Values are the **daily mean snow depth in cm** (paper,
  Sect. 3.1 and the cm thresholds of Sect. 3.3), two decimals.  An empty
  cell is a missing day, including values the quality control removed.
- The metadata file has one row per station: ``code_internal`` (the id),
  ``name``, ``lat``, ``lon``, ``elevation`` (m), ``basin``, ``source``
  (the operating network), ``Snow Depth Sensor``, ``start date`` /
  ``end date`` (``YYYY/MM/DD``), ``potential days with data`` and
  ``effective days with data`` (``"1062 (94%)"``).

Known quirks (record 21577171)
------------------------------
- Three cells hold quoted text instead of a number (``","`` twice and
  ``",0.45"`` once, 2019–2020).  They are treated as missing days (logged
  at debug level); ``",0.45"`` is not guessed to be 0.45.
- Station ids mix DGA/CEAZA/CIEP numeric codes (``"5703009"``) and
  IANIGLA names (``"Capayan"``).  They are kept verbatim as strings.
- The paper reports **81** stations after quality control; the Zenodo
  files hold **69** (the 12 University of Chile stations of the paper are
  not in them).

Design principles
-----------------
- Metric-first: depth is published in cm and returned in cm.
- One variable (``SD``, type ``snwd``) at ``interval="daily"``; anything
  else raises :class:`SouthernAndesError`.
- The values are already quality controlled; this client nulls only
  depths outside 0–15 m, and none were in October 2026.
- No flags are published (``DATA_FLAGS`` is empty).
- ``status`` is "Active" when a station reported in the dataset's final
  year (2024 for record 21577171) and that year is at most two years
  before the current one: the dataset is a versioned compilation, so
  "Active" means "still reporting in the latest version", not "reporting
  today".
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .._common import coerce_list as _coerce_list
from .._common import date_str as _date_str
from .._common import download_file, request_with_retries
from .._common import filter_by_bbox as _filter_by_bbox

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ZENODO_API = "https://zenodo.org/api/records"
CONCEPT_RECID = "19560127"
DATASET_URL = "https://doi.org/10.5281/zenodo.19560127"
PAPER_URL = "https://doi.org/10.5194/essd-18-7097-2026"

#: File-name prefixes of the two CSVs in a record (the version suffix of
#: the data file, ``_v4.2``, changes between versions).
DATA_PREFIX = "Southern_Andes_Snow_Depth_Dataset"
METADATA_PREFIX = "Station_metadata"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4

#: Plausibility bound for snow depth, cm (DESIGN.md: 0–15 m).
_MAX_SNWD_CM = 1500.0

#: Years after the dataset's final year for which a station reporting in
#: that year still counts as Active.
_ACTIVE_LAG_YEARS = 2

_SOURCE = (
    "Southern Andes Daily Snow Depth Dataset (Medina Mendoza & Caro 2026) — "
    + DATASET_URL
)

#: ``source`` column → operating network (paper, Sect. 3.1).
NETWORKS: dict[str, str] = {
    "DGA": "Dirección General de Aguas (DGA), Ministerio de Obras Públicas, Chile",
    "IANIGLA": (
        "Instituto Argentino de Nivología, Glaciología y Ciencias Ambientales "
        "(IANIGLA-CONICET), Argentina"
    ),
    "CEAZA": "Centro de Estudios Avanzados en Zonas Áridas (CEAZA), Chile",
    "CIEP": "Centro de Investigación en Ecosistemas de la Patagonia (CIEP), Chile",
}

#: Original data portals per network, as the paper cites them (Sect. 3.1).
NETWORK_URLS: dict[str, str] = {
    "DGA": "https://snia.mop.gob.cl/BNAConsultas/reportes",
    "IANIGLA": "https://observatorioandino.com/estaciones/",
    "CEAZA": "https://www.ceazamet.cl/",
    "CIEP": "http://190.121.23.217/index_cdataaysen.php",
}

# ── Public variable / flag tables ────────────────────────────────────────────

VARIABLES: dict[str, dict] = {
    "SD": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "source": _SOURCE,
        "description": (
            "Daily mean snow depth from an automatic ultrasonic sensor "
            "(Campbell SR50A or Lufft SHM31), quality controlled by the "
            "dataset authors."
        ),
        "notes": (
            "Published and returned in cm. Quality control (Caro et al. 2026): "
            "baseline (zero-offset) correction, removal of isolated spikes "
            "with a rolling median/MAD filter, removal of values outside "
            "basin-specific physical limits, and a cross-check against "
            "precipitation and air temperature. Removed values are missing."
        ),
    },
}

#: The dataset carries no per-value flags; removed values are simply empty.
DATA_FLAGS: dict[str, str] = {}

_TYPE_TO_VARS: dict[str, list[str]] = {}
for _code, _info in VARIABLES.items():
    _TYPE_TO_VARS.setdefault(_info["type"], []).append(_code)

_DEFAULT_VARS: list[str] = ["SD"]


class SouthernAndesError(Exception):
    """Raised when the Southern Andes dataset cannot be fetched, read or queried."""


# ── Parsers ──────────────────────────────────────────────────────────────────


def _resolve_variables(variables: list[str] | str | None) -> list[str]:
    """Map native names or standardized types to native names."""
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
            raise SouthernAndesError(
                f"Unknown Southern Andes variable {v!r} — expected one of "
                f"{sorted(VARIABLES)} or types {sorted(_TYPE_TO_VARS)}. "
                "The dataset holds snow depth only (no SWE)."
            )
        for c in codes:
            if c not in out:
                out.append(c)
    return out


def _convert(raw: str | None) -> float | None:
    """Cell text → snow depth in cm, or ``None`` (empty, junk or implausible)."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        logger.debug("Southern Andes: non-numeric cell %r read as missing", raw)
        return None
    if not 0.0 <= value <= _MAX_SNWD_CM:
        return None
    return round(value, 3)


def _slash_date(text: str | None) -> str | None:
    """``"2021/12/01"`` → ``"2021-12-01"``."""
    text = (text or "").strip()
    return text.replace("/", "-") if text else None


def _days(text: str | None) -> int | None:
    """``"1062 (94%)"`` or ``"1127"`` → ``1062`` / ``1127``."""
    head = (text or "").strip().split(" ", 1)[0]
    return int(head) if head.isdigit() else None


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def parse_metadata(text: str) -> list[dict]:
    """Parse the station-metadata CSV into station dicts (no ``status`` yet)."""
    rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    out: list[dict] = []
    for r in rows:
        sid = (r.get("code_internal") or "").strip()
        if not sid:
            continue
        net = (r.get("source") or "").strip()
        elev = _float(r.get("elevation"))
        out.append(
            {
                "station_id": sid,
                "name": (r.get("name") or "").strip() or None,
                "latitude": _float(r.get("lat")),
                "longitude": _float(r.get("lon")),
                "elevation_m": elev,
                "network": net or None,
                "operator": NETWORKS.get(net, net or None),
                "country": "AR" if net == "IANIGLA" else "CL",
                "basin": (r.get("basin") or "").strip() or None,
                "sensor": (r.get("Snow Depth Sensor") or "").strip() or None,
                "measurement_method": "Automatic ultrasonic snow-depth sensor",
                "begin_date": _slash_date(r.get("start date")),
                "end_date": _slash_date(r.get("end date")),
                "potential_days": _days(r.get("potential days with data")),
                "effective_days": _days(r.get("effective days with data")),
                "has_swe": False,
                "has_snwd": True,
                "station_url": NETWORK_URLS.get(net) or DATASET_URL,
            }
        )
    return out


def parse_data(text: str) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    """Parse the wide data CSV.

    Returns
    -------
    (station_ids, columns)
        The station ids in header order, and per station a list of
        ``(date, cell text)`` for every numeric cell (empty and non-numeric
        cells are missing days).
    """
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    try:
        header = next(reader)
    except StopIteration:
        raise SouthernAndesError("Southern Andes data file is empty") from None
    if not header or header[0].strip().lower() != "date":
        raise SouthernAndesError(
            f"Southern Andes data file does not start with a 'date' column: "
            f"{header[:3]!r}"
        )
    ids = [h.strip() for h in header[1:]]
    columns: dict[str, list[tuple[str, str]]] = {sid: [] for sid in ids}
    junk = 0
    for row in reader:
        if not row or not row[0].strip():
            continue
        day = row[0].strip()
        for sid, cell in zip(ids, row[1:]):
            text = cell.strip()
            if not text:
                continue
            try:
                float(text)
            except ValueError:
                junk += 1  # e.g. '","' — not an observation
                continue
            columns[sid].append((day, text))
    if junk:
        logger.debug("Southern Andes: %d non-numeric cells read as missing", junk)
    return ids, columns


# ── Client ───────────────────────────────────────────────────────────────────


class SouthernAndesClient:
    """
    Client for the Southern Andes Daily Snow Depth Dataset (Zenodo).

    No credentials are needed.

    Parameters
    ----------
    path : str or Path, optional
        Folder holding the two CSVs (the data file and the station
        metadata); skips Zenodo entirely.
    record_id : str, optional
        Pin a Zenodo record (e.g. ``"21577171"``).  Default: the latest
        version of concept record 19560127.
    cache_dir : str or Path, optional
        Override ``easysnowdata.config.cache_dir("stations", "southern_andes")``.
    timeout, max_retries, backoff : int
        HTTP settings for the shared retry loop.
    session : requests.Session, optional
        Pre-configured session.
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        record_id: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self.record_id = record_id
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self.record: dict | None = None
        self._stations: list[dict] | None = None
        self._columns: dict[str, list[tuple[str, str]]] | None = None
        self._last_day: str | None = None

    # ── Files ────────────────────────────────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            from easysnowdata.config import cache_dir  # noqa: PLC0415

            self._cache_dir = cache_dir("stations", "southern_andes")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def resolve_record(self) -> dict:
        """Zenodo record JSON for the pinned or latest version."""
        if self.record_id:
            url = f"{ZENODO_API}/{self.record_id}"
        else:
            url = f"{ZENODO_API}/{CONCEPT_RECID}/versions/latest"
        resp = request_with_retries(
            self._session,
            url,
            error_cls=SouthernAndesError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )
        try:
            record = resp.json()
        except ValueError as exc:
            raise SouthernAndesError(f"Zenodo returned non-JSON for {url}") from exc
        self.record = record
        return record

    @staticmethod
    def _find(names: list[str], prefix: str) -> str | None:
        hits = sorted(n for n in names if n.startswith(prefix) and n.endswith(".csv"))
        return hits[-1] if hits else None

    def _local_files(self, folder: Path) -> tuple[Path, Path] | None:
        names = [p.name for p in folder.glob("*.csv")]
        data, meta = self._find(names, DATA_PREFIX), self._find(names, METADATA_PREFIX)
        if data and meta:
            return folder / data, folder / meta
        return None

    def _cached_files(self) -> tuple[Path, Path] | None:
        folders = sorted(
            (p for p in self.cache_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for folder in folders:
            found = self._local_files(folder)
            if found:
                return found
        return None

    def dataset_files(self) -> tuple[Path, Path]:
        """Local ``(data CSV, metadata CSV)``, downloading them on first use.

        Returns
        -------
        tuple[Path, Path]

        Raises
        ------
        SouthernAndesError
            If ``path=`` lacks the two CSVs, or nothing can be downloaded
            or found in the cache.
        """
        if self._path is not None:
            found = self._local_files(self._path) if self._path.is_dir() else None
            if not found:
                raise SouthernAndesError(
                    f"{self._path} must be a folder holding {DATA_PREFIX}*.csv "
                    f"and {METADATA_PREFIX}*.csv"
                )
            return found
        try:
            record = self.resolve_record()
        except SouthernAndesError as exc:
            cached = self._cached_files()
            if cached is None:
                raise
            logger.warning(
                "Southern Andes: Zenodo unreachable (%s); using cached %s",
                exc,
                cached[0].parent.name,
            )
            return cached
        files = {f.get("key", ""): f for f in record.get("files", [])}
        keys = [self._find(list(files), p) for p in (DATA_PREFIX, METADATA_PREFIX)]
        if not all(keys):
            raise SouthernAndesError(
                f"Zenodo record {record.get('id')} lacks the expected CSVs "
                f"({DATA_PREFIX}*.csv, {METADATA_PREFIX}*.csv); found {sorted(files)}"
            )
        folder = self.cache_dir / str(record.get("id", "unknown"))
        out: list[Path] = []
        for key in keys:
            info = files[key]
            dest = folder / key
            if not dest.exists():
                folder.mkdir(parents=True, exist_ok=True)
                url = info.get("links", {}).get("self") or (
                    f"{ZENODO_API}/{record.get('id', '')}/files/{key}/content"
                )
                logger.info("Southern Andes: downloading %s", key)
                download_file(
                    self._session,
                    url,
                    dest,
                    error_cls=SouthernAndesError,
                    size=int(info.get("size") or 0) or None,
                    md5=info.get("checksum"),
                    timeout=self.timeout,
                    progress=False,
                )
            out.append(dest)
        return out[0], out[1]

    def _load(self) -> None:
        if self._columns is not None:
            return
        data_path, meta_path = self.dataset_files()
        try:
            data_text = data_path.read_text(encoding="utf-8")
            meta_text = meta_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SouthernAndesError(
                f"Cannot read Southern Andes files: {exc}"
            ) from exc
        ids, columns = parse_data(data_text)
        stations = {s["station_id"]: s for s in parse_metadata(meta_text)}
        missing = [sid for sid in ids if sid not in stations]
        if missing:
            logger.warning(
                "Southern Andes: %d data columns have no metadata row: %s",
                len(missing),
                missing,
            )
        last_day = max((c[-1][0] for c in columns.values() if c), default=None)
        self._columns = columns
        self._last_day = last_day
        self._stations = [
            self._finish_station(stations[sid]) for sid in ids if sid in stations
        ]
        logger.info(
            "Southern Andes: %d stations, data to %s", len(self._stations), last_day
        )

    def _finish_station(self, sta: dict) -> dict:
        assert self._columns is not None
        cells = self._columns.get(sta["station_id"], [])
        first = cells[0][0] if cells else None
        last = cells[-1][0] if cells else None
        active = False
        if last and self._last_day:
            final_year = int(self._last_day[:4])
            active = (
                int(last[:4]) == final_year
                and date.today().year - final_year <= _ACTIVE_LAG_YEARS
            )
        return {
            **sta,
            "status": "Active" if active else "Inactive",
            "first_obs": first,
            "last_obs": last,
            "n_snwd": len(cells),
        }

    # ── Public API — stations ────────────────────────────────────────────────

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        networks: list[str] | str | None = None,
    ) -> list[dict]:
        """
        Every station in the dataset.

        Parameters
        ----------
        active_only : bool
            Keep stations that reported in the dataset's final year.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        networks : list[str] or str, optional
            Keep these networks (``"DGA"``, ``"IANIGLA"``, ``"CEAZA"``,
            ``"CIEP"``; see :data:`NETWORKS`).

        Returns
        -------
        list[dict]
            ``station_id``, ``name``, ``latitude``, ``longitude``,
            ``elevation_m``, ``status``, ``network`` (the ``source`` column),
            ``operator``, ``country``, ``basin``, ``sensor``,
            ``measurement_method``, ``begin_date`` / ``end_date`` (as the
            metadata file gives them), ``potential_days``,
            ``effective_days``, ``first_obs`` / ``last_obs`` / ``n_snwd``
            (from the data file), ``has_swe`` (always False), ``has_snwd``
            and ``station_url`` (the network's own portal).

        Raises
        ------
        SouthernAndesError
            On a download failure or an unknown network.
        """
        self._load()
        assert self._stations is not None
        out = [dict(s) for s in self._stations]
        if networks is not None:
            wanted = {n.upper() for n in _coerce_list(networks)}
            unknown = wanted - set(NETWORKS)
            if unknown:
                raise SouthernAndesError(
                    f"Unknown network(s) {sorted(unknown)}; expected {sorted(NETWORKS)}"
                )
            out = [s for s in out if s["network"] in wanted]
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    def get_metadata(self, station_id: str) -> dict:
        """
        Station metadata plus ``data_variables``.

        Parameters
        ----------
        station_id : str
            Id as in :meth:`get_all_stations` (``"5703009"``, ``"Capayan"``).

        Returns
        -------
        dict
            The station dict plus ``data_variables``: one entry (``SD``)
            with ``name``, ``type``, ``interval`` (``"daily"``), ``units``,
            ``description``, ``notes``, ``begin_date`` / ``end_date`` (first
            and last value in the data file) and ``n_obs``.

        Raises
        ------
        SouthernAndesError
            If the station is not in the dataset.
        """
        sid = str(station_id).strip()
        match = [s for s in self.get_all_stations() if s["station_id"] == sid]
        if not match:
            raise SouthernAndesError(
                f"Station {sid!r} not in the Southern Andes dataset"
            )
        sta = match[0]
        info = VARIABLES["SD"]
        sta["data_variables"] = [
            {
                "name": "SD",
                "type": info["type"],
                "interval": "daily",
                "units": info["output_units"],
                "description": info["description"],
                "notes": info["notes"],
                "begin_date": sta["first_obs"],
                "end_date": sta["last_obs"],
                "n_obs": sta["n_snwd"],
            }
        ]
        return sta

    # ── Public API — data ────────────────────────────────────────────────────

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
        Daily snow depth as flat records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            Station id(s).  Required unless ``bbox`` is given.
        variables : list[str] or str or None
            ``"SD"`` or ``"snwd"``; ``None`` = ``["SD"]``.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds.
        interval : str
            Only ``"daily"``.
        include_flags : bool
            Add ``flag`` (always ``None``: the dataset has no flags).

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``flag``), one per day with a value in the file (``value`` is
            ``None`` only for a depth outside 0–15 m).

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        SouthernAndesError
            On an unknown variable or station, an unsupported interval or a
            download failure.

        Example
        -------
        >>> client = SouthernAndesClient()
        >>> records = client.get_data(
        ...     station_ids="5703009",
        ...     begin_date="2023-06-01",
        ...     end_date="2023-09-30",
        ... )
        >>> records[0]["units"], records[0]["interval"]
        ('cm', 'daily')
        """
        if interval.lower() != "daily":
            raise SouthernAndesError(
                f"Unsupported interval {interval!r} for the Southern Andes "
                "dataset — only 'daily' exists."
            )
        codes = _resolve_variables(variables)
        if station_ids is not None:
            ids = [str(s).strip() for s in _coerce_list(station_ids)]
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        self._load()
        assert self._columns is not None
        unknown = [sid for sid in ids if sid not in self._columns]
        if unknown:
            raise SouthernAndesError(
                f"Station(s) {unknown} not in the Southern Andes dataset"
            )
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None
        info = VARIABLES["SD"]
        records: list[dict] = []
        for sid in ids:
            if "SD" not in codes:
                continue
            for day, cell in self._columns[sid]:
                if (begin and day < begin) or (end and day > end):
                    continue
                rec: dict[str, Any] = {
                    "station_id": sid,
                    "date": day,
                    "variable": "SD",
                    "type": info["type"],
                    "value": _convert(cell),
                    "units": info["output_units"],
                    "interval": "daily",
                }
                if include_flags:
                    rec["flag"] = None
                records.append(rec)
        return records
