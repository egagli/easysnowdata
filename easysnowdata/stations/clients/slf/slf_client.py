"""
clients/slf/slf_client.py
=========================
Client for open Swiss snow-station data from the WSL Institute for Snow and
Avalanche Research SLF.

SLF publishes several open station datasets; this client exposes four of
them behind one ``dataset=`` switch (an SLF dataset is a different set of
stations with a different cadence, not another route to the same data):

``"gcos"`` (default) — *GCOS SWE data from 11 stations in Switzerland*
    EnviDat ``gcos-swe-data``, DOI 10.16904/15, licence ODbL 1.0 + DbCL.
    Manual snow-pit SWE and snow depth, roughly every two weeks
    (1st and 15th of the month) at 11 SLF study plots, 1 350–2 540 m.
    Records start 1936 (Weissfluhjoch) / 1947 (most), end of the
    2024/25 season at the time of writing (newest value 2025-05-31).
    The data file lives in the EnviDat S3 bucket; its location is resolved
    at runtime from the CKAN ``package_show`` record.
``"new_snow"`` — *Dataset on new snow water equivalent*
    EnviDat ``dataset-on-new-snow-water-equivalent``, DOI
    10.16904/envidat.590, licence CC BY 4.0.  41 SLF study plots,
    2016-09-01 → 2022-08-31 (static).  Daily manual new-snow depth (HN),
    daily new-snow water equivalent (HNW), daily stake snow depth (HS) and
    biweekly profile snow depth / SWE.  Coordinates are given in CH1903/LV03
    and converted here to WGS84 with swisstopo's approximate formulas
    (≈1 m accuracy).  The zip is resolved through CKAN at runtime.
``"study_plot"`` — SLF manual measuring network (historical file service)
    https://measurement-data.slf.ch/study-plot/, DOI 10.16904/envidat.408,
    licence CC BY 4.0 (SLF Data Service terms).  135 study plots (100
    active), daily morning HS, HN_1D and (at ~100 plots) HNW_1D, from 1930
    to yesterday — no SWE of the snowpack.  Files are rebuilt daily.
``"imis"`` — SLF IMIS automatic stations (historical file service)
    https://measurement-data.slf.ch/imis/, DOI 10.16904/envidat.406,
    licence CC BY 4.0 (SLF Data Service terms).  ~180 snow/flowcapt
    stations, mostly 2 000–3 000 m.  Daily HS (median over the last 24 h)
    and SNOWPACK-*modelled* HN_1D from 1992; 30-minute HS and air
    temperature (``interval="sub_hourly"``) — no SWE.  Rebuilt daily.

No open daily SWE series exists in any of these (checked October 2026):
the only SWE is manual and biweekly (``gcos``, ``new_snow`` profiles).
IMIS and the study plots measure snow depth, and SNOWPACK model output
beyond HN_1D is not served openly.

Authentication : none
Terms of use   : https://www.slf.ch/en/services-and-products/slf-data-service/
                 (``study_plot``/``imis``: CC BY 4.0, cite the DOI, credit
                 "WSL Institute for Snow and Avalanche Research SLF" with a
                 link to www.slf.ch; do not overload SLF infrastructure).
                 EnviDat datasets carry the licence on their CKAN record.

Design notes
------------
- Metric-first: SWE (mm) is returned in cm; HS/HN stay in cm; HNW in mm;
  air temperature in °C.  Missing values (``-9999``, empty) → ``None``.
- Physically implausible snow values are nulled with scoped clamps
  (SWE / snow depth outside 0–15 m, new snow outside 0–5 m).  IMIS HS has
  small negative values (sensor noise around bare ground); they are nulled.
- ``HNW_1D`` (water equivalent of the day's new snow) is typed ``"other"``,
  not ``"precip"``: it excludes rain, so it must not be mistaken for total
  precipitation (same convention as GHCN-D ``WESF``).
- Dates: daily values are labelled with the date of the morning
  observation (the 24 h it closes end on that date).  ``study_plot`` stamps
  observations at 06:00 UTC, ``new_snow`` at 06:00 CET; IMIS daily values
  are stamped 00:00 UTC although SLF's README says "at 6 UTC" —
  the date label is used as-is.
- Station ids are the dataset's own codes: ``"5WJ"`` (gcos/new_snow),
  ``"5WJ0"`` (study_plot), ``"WFJ2"`` (imis).  The three-character GCOS and
  new-snow codes are the study-plot codes without the trailing ``0``.
- No quality flags are published; ``DATA_FLAGS`` is empty.
- Downloads are cached under ``easysnowdata.config.cache_dir("stations",
  "slf")`` and refreshed after ``cache_ttl_hours`` (static datasets never
  expire).  The live API (https://measurement-api.slf.ch, last 1–7 days)
  is not used: the file service already carries data to yesterday.
"""

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from .._common import coerce_list as _coerce_list
from .._common import date_str as _date_str
from .._common import filter_by_bbox as _filter_by_bbox
from .._common import request_with_retries
from .._common import to_float as _to_float

logger = logging.getLogger(__name__)


class SLFError(Exception):
    """Raised for SLF / EnviDat source failures and bad requests."""


# ── Constants ────────────────────────────────────────────────────────────────

CKAN_URL = "https://www.envidat.ch/api/action/package_show"
MEASUREMENT_DATA_URL = "https://measurement-data.slf.ch"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 4
_DEFAULT_TTL_HOURS = 24.0

# Fallback location of the GCOS files if the CKAN record cannot be parsed.
_GCOS_FALLBACK_BASE = (
    "https://zhc-c.s3.cloud.switch.ch/envidat-doi/10.16904_15/data/GCOS"
)

# GCOS is extended once a year after the season.  A station is Active when
# it reported within ``_GCOS_SEASON_DAYS`` of the file's newest value, and
# the file's newest value is no older than ``_GCOS_STALE_DAYS``.
_GCOS_SEASON_DAYS = 200
_GCOS_STALE_DAYS = 800

# Plausibility bounds (output units).
_MAX_PLAUSIBLE: dict[str, float] = {
    "swe": 1500.0,  # cm
    "snwd": 1500.0,  # cm
    "snowfall": 500.0,  # cm
}

_TERMS_URL = "https://www.slf.ch/en/services-and-products/slf-data-service/"

#: The datasets this client exposes.  ``variables`` maps each native
#: variable to the intervals it is available at in that dataset (first =
#: default) and, for ``new_snow``, the file inside the zip.
DATASETS: dict[str, dict[str, Any]] = {
    "gcos": {
        "title": "GCOS SWE data from 11 stations in Switzerland",
        "ckan_id": "gcos-swe-data",
        "doi": "10.16904/15",
        "license": "ODbL-1.0 (database) + DbCL-1.0 (contents)",
        "landing_url": "https://www.envidat.ch/dataset/gcos-swe-data",
        "update_frequency": "about yearly, after the snow season",
        "static": False,
        "default_variables": ["SWE", "HS"],
        "variables": {
            "SWE": {"intervals": ["periodic"], "column": "SWE(mm)"},
            "HS": {"intervals": ["periodic"], "column": "HS(cm)"},
        },
    },
    "new_snow": {
        "title": "Dataset on new snow water equivalent",
        "ckan_id": "dataset-on-new-snow-water-equivalent",
        "doi": "10.16904/envidat.590",
        "license": "CC-BY-4.0",
        "landing_url": (
            "https://www.envidat.ch/dataset/dataset-on-new-snow-water-equivalent"
        ),
        "update_frequency": "static (2016-09-01 to 2022-08-31)",
        "static": True,
        "default_variables": ["SWE", "HS"],
        "variables": {
            "SWE": {"intervals": ["periodic"], "file": "SWE/OBS-SWE-PROFILE.txt"},
            "HS": {"intervals": ["daily"], "file": "HS/OBS-HS-STAKE.txt"},
            "HS_PROFILE": {
                "intervals": ["periodic"],
                "file": "HS/OBS-HS-PROFILE.txt",
            },
            "HN_1D": {"intervals": ["daily"], "file": "HN/OBS-HN.txt"},
            "HNW_1D": {"intervals": ["daily"], "file": "HNW/OBS-HNW.txt"},
        },
    },
    "study_plot": {
        "title": "SLF manual measuring network (study plots)",
        "ckan_id": "manual-measuring-network",
        "doi": "10.16904/envidat.408",
        "license": "CC-BY-4.0",
        "landing_url": f"{MEASUREMENT_DATA_URL}/study-plot/",
        "update_frequency": "daily (file service rebuilt each day)",
        "static": False,
        "default_variables": ["HS"],
        "variables": {
            "HS": {"intervals": ["daily"]},
            "HN_1D": {"intervals": ["daily"]},
            "HNW_1D": {"intervals": ["daily"]},
        },
    },
    "imis": {
        "title": "SLF IMIS automatic stations",
        "ckan_id": "imis-measuring-network",
        "doi": "10.16904/envidat.406",
        "license": "CC-BY-4.0",
        "landing_url": f"{MEASUREMENT_DATA_URL}/imis/",
        "update_frequency": "daily (file service rebuilt each day)",
        "static": False,
        "default_variables": ["HS"],
        "variables": {
            "HS": {"intervals": ["daily", "sub_hourly"]},
            "HN_1D": {"intervals": ["daily"]},
            "TA_30MIN_MEAN": {"intervals": ["sub_hourly"]},
        },
    },
}

DEFAULT_DATASET = "gcos"

# ── Public variable / flag tables ────────────────────────────────────────────

#: Native variables across all SLF datasets.  Native names follow the SLF
#: file-service column names; ``scale`` converts native to output units.
VARIABLES: dict[str, dict] = {
    "SWE": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "output_units": "cm",
        "scale": 0.1,
        "description": "Water equivalent of the snowpack, from a manual snow pit.",
        "notes": (
            "Manual, about every two weeks ('periodic'). gcos: column "
            "'SWE(mm)'; new_snow: OBS-SWE-PROFILE. Returned in cm (× 0.1)."
        ),
        "source": "EnviDat (gcos-swe-data, dataset-on-new-snow-water-equivalent)",
    },
    "HS": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "description": "Total depth of snow on the ground.",
        "notes": (
            "gcos: at the snow pit (biweekly); new_snow: fixed stake (daily); "
            "study_plot: manual morning reading (daily); imis: ultrasonic "
            "sensor — daily = median over the last 24 h, sub_hourly = 30-min "
            "instantaneous. Negative values are nulled."
        ),
        "source": "SLF (EnviDat, measurement-data.slf.ch)",
    },
    "HS_PROFILE": {
        "name": "Snow Depth at Snow Pit",
        "type": "snwd",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "description": "Snow depth measured at the biweekly snow profile.",
        "notes": "new_snow only (OBS-HS-PROFILE); pairs with SWE.",
        "source": "EnviDat (dataset-on-new-snow-water-equivalent)",
    },
    "HN_1D": {
        "name": "24-hour New Snow",
        "type": "snowfall",
        "units": "cm",
        "output_units": "cm",
        "scale": 1.0,
        "description": "Depth of new snow over the last 24 hours.",
        "notes": (
            "Manual board measurement at study plots; at IMIS stations it is "
            "MODELLED by SNOWPACK, not measured."
        ),
        "source": "SLF (EnviDat, measurement-data.slf.ch)",
    },
    "HNW_1D": {
        "name": "24-hour New Snow Water Equivalent",
        "type": "other",
        "units": "mm",
        "output_units": "mm",
        "scale": 1.0,
        "description": "Water equivalent of the last 24 hours' new snow.",
        "notes": (
            "Melted new-snow core, manual. Typed 'other' rather than "
            "'precip' because rain is not included. Usually blank on days "
            "without new snow."
        ),
        "source": "SLF (EnviDat, measurement-data.slf.ch)",
    },
    "TA_30MIN_MEAN": {
        "name": "Air Temperature",
        "type": "temp",
        "units": "degC",
        "output_units": "degC",
        "scale": 1.0,
        "description": "Air temperature, mean over the last 30 minutes.",
        "notes": "IMIS only, interval 'sub_hourly'.",
        "source": "SLF measurement-data.slf.ch/imis",
    },
}

#: SLF publishes no per-value quality flags for these datasets.
DATA_FLAGS: dict[str, str] = {}


# ── Parsers ──────────────────────────────────────────────────────────────────


def lv03_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """CH1903/LV03 → WGS84 ``(lat, lon)`` with swisstopo's approximate formulas."""
    y = (easting - 600000.0) / 1e6
    x = (northing - 200000.0) / 1e6
    lon = (
        2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x * x - 0.0436 * y**3
    )
    lat = (
        16.9023892
        + 3.238272 * x
        - 0.270978 * y * y
        - 0.002528 * x * x
        - 0.0447 * y * y * x
        - 0.0140 * x**3
    )
    return round(lat * 100 / 36, 6), round(lon * 100 / 36, 6)


def _parse_gcos_metadata(text: str) -> dict[str, dict]:
    """Parse the GCOS ``metadata.txt`` (whitespace-separated, label with spaces).

    Lines look like ``1MR    Mürren (1MR) 1650 m    7.890192955   46.55818213	1951``.
    """
    out: dict[str, dict] = {}
    for line in text.splitlines()[1:]:
        m = re.match(
            r"^\s*(\w+)\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d{4})\s*$",
            line,
        )
        if not m:
            continue
        code, label, lon, lat, start = m.groups()
        em = re.search(r"(\d+)\s*m\s*$", label)
        name = re.sub(r"\s*\(\w+\)\s*\d+\s*m\s*$", "", label).strip()
        out[code] = {
            "station_id": code,
            "name": name,
            "latitude": float(lat),
            "longitude": float(lon),
            "elevation_m": float(em.group(1)) if em else None,
            "start_year": int(start),
        }
    return out


def _parse_gcos_csv(text: str) -> pd.DataFrame:
    """Parse ``SWE_GCOS.csv`` (``;``-separated) into a long frame."""
    sep = ";" if text.split("\n", 1)[0].count(";") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    need = {"STAT_ABK", "DATUM", "HS(cm)", "SWE(mm)"}
    if not need <= set(df.columns):
        raise SLFError(f"Unexpected GCOS columns {list(df.columns)}")
    df = df.rename(columns={"STAT_ABK": "station_code", "DATUM": "measure_date"})
    df["SWE"] = df.pop("SWE(mm)")
    df["HS"] = df.pop("HS(cm)")
    return df


def _parse_new_snow_table(text: str) -> pd.DataFrame:
    """Parse one wide ``OBS-*.txt`` table from the new-snow zip.

    Header block, then ``TABLE WITH DATA``, blank line, ``Time,SLF.1AD,...``.
    Returns a long frame ``station_code, measure_date, value`` (raw strings).
    """
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "TABLE WITH DATA")
    except StopIteration as exc:
        raise SLFError("new_snow table: 'TABLE WITH DATA' marker not found") from exc
    body = "\n".join(ln for ln in lines[start + 1 :] if ln.strip())
    wide = pd.read_csv(io.StringIO(body), dtype=str)
    long = wide.melt(id_vars="Time", var_name="station_code", value_name="value")
    long["station_code"] = long["station_code"].str.replace("SLF.", "", regex=False)
    return long.rename(columns={"Time": "measure_date"})


def _parse_new_snow_stations(text: str) -> dict[str, dict]:
    """Parse ``STATION-LIST.txt`` (LV03 easting/northing, altitude)."""
    out: dict[str, dict] = {}
    df = pd.read_csv(io.StringIO(text), dtype=str)
    for _, row in df.iterrows():
        code = str(row.iloc[0]).replace("SLF.", "").strip()
        e, n, z = (_to_float(row.iloc[i]) for i in (1, 2, 3))
        if e is None or n is None:
            continue
        lat, lon = lv03_to_wgs84(e, n)
        out[code] = {
            "station_id": code,
            "name": code,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": z,
            "easting_lv03": e,
            "northing_lv03": n,
        }
    return out


def _parse_measurement_csv(text: str) -> pd.DataFrame:
    """Parse a measurement-data.slf.ch CSV (``station_code,measure_date,...``)."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if "measure_date" not in df.columns:
        raise SLFError(f"Unexpected SLF CSV columns {list(df.columns)[:8]}")
    return df


def _convert(variable: str, raw: Any) -> float | None:
    """Native → output units, with missing handling and snow clamps."""
    value = _to_float(raw)
    if value is None:
        return None
    info = VARIABLES[variable]
    value = round(value * info["scale"], 3)
    bound = _MAX_PLAUSIBLE.get(info["type"])
    if bound is not None and not (0.0 <= value <= bound):
        return None
    if variable == "HNW_1D" and value < 0:
        return None
    return value


def _resolve_dataset(dataset: str) -> str:
    key = str(dataset).lower()
    if key not in DATASETS:
        raise SLFError(
            f"Unknown SLF dataset {dataset!r} — expected one of {sorted(DATASETS)}."
        )
    return key


def _resolve_variables(
    dataset: str, variables: list[str] | str | None, interval: str | None
) -> list[tuple[str, str]]:
    """Return ``[(native variable, interval)]`` for a request.

    ``variables`` may mix native names and standardized types.  ``None``
    means the dataset's snow defaults.  An explicitly named variable that
    the dataset does not serve at ``interval`` raises — no silent fallback.
    """
    spec = DATASETS[dataset]["variables"]
    if interval is not None and interval not in {
        i for v in spec.values() for i in v["intervals"]
    }:
        offered = sorted({i for v in spec.values() for i in v["intervals"]})
        raise SLFError(
            f"Unsupported interval {interval!r} for SLF dataset {dataset!r} — "
            f"expected one of {offered}."
        )

    def pick(var: str) -> str | None:
        ivs = spec[var]["intervals"]
        if interval is None:
            return ivs[0]
        return interval if interval in ivs else None

    if variables is None or (not isinstance(variables, str) and not list(variables)):
        if interval is None:
            return [
                (v, spec[v]["intervals"][0])
                for v in DATASETS[dataset]["default_variables"]
            ]
        out = [
            (v, interval)
            for v in spec
            if VARIABLES[v]["type"] in ("swe", "snwd")
            and interval in spec[v]["intervals"]
        ]
        if not out:
            raise SLFError(
                f"SLF dataset {dataset!r} has no snow variables at interval {interval!r}."
            )
        return out

    raw = [variables] if isinstance(variables, str) else list(variables)
    out: list[tuple[str, str]] = []
    for v in raw:
        if v in spec:
            candidates = [v]
        elif v in VARIABLES:
            raise SLFError(
                f"Variable {v!r} is not part of SLF dataset {dataset!r} — "
                f"it offers {sorted(spec)}."
            )
        else:
            candidates = [n for n in spec if VARIABLES[n]["type"] == v]
            if not candidates:
                raise SLFError(
                    f"Unknown variable {v!r} for SLF dataset {dataset!r} — "
                    f"expected one of {sorted(spec)} or a standardized type "
                    f"({sorted({VARIABLES[n]['type'] for n in spec})})."
                )
        chosen = [(c, pick(c)) for c in candidates if pick(c) is not None]
        if not chosen:
            raise SLFError(
                f"Variable {v!r} is not available at interval {interval!r} in "
                f"SLF dataset {dataset!r} (offered: "
                f"{ {c: spec[c]['intervals'] for c in candidates} })."
            )
        out.extend(c for c in chosen if c not in out)
    return out


# ── Client ───────────────────────────────────────────────────────────────────


class SLFClient:
    """
    Client for open SLF snow-station datasets (GCOS SWE, new-snow SWE,
    manual study plots, IMIS).

    No credentials are needed.

    Parameters
    ----------
    dataset : str
        ``"gcos"`` (default), ``"new_snow"``, ``"study_plot"`` or ``"imis"``;
        see :data:`DATASETS` and the module docstring.
    cache_dir : str or Path or None
        Where downloads are cached.  Defaults to
        ``easysnowdata.config.cache_dir("stations", "slf")``.
    cache_ttl_hours : float
        Re-download a cached file of a non-static dataset older than this.
    timeout, max_retries, backoff : int
        HTTP settings passed to the shared retry loop.
    session : requests.Session or None
        Optional pre-configured session.
    """

    def __init__(
        self,
        dataset: str = DEFAULT_DATASET,
        cache_dir: str | Path | None = None,
        cache_ttl_hours: float = _DEFAULT_TTL_HOURS,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self.dataset = _resolve_dataset(dataset)
        if cache_dir is None:
            from easysnowdata.config import cache_dir as _cache_dir  # noqa: PLC0415

            cache_dir = _cache_dir("stations", "slf")
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "easysnowdata"})
        self._ckan: dict[str, dict] = {}
        self._stations: dict[str, dict] | None = None
        self._frames: dict[str, pd.DataFrame] = {}
        self._zip: zipfile.ZipFile | None = None

    # ── HTTP / cache ─────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        return request_with_retries(
            self._session,
            url,
            params=params,
            error_cls=SLFError,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff=self.backoff,
        )

    def _cached(self, name: str, url: str, *, static: bool = False) -> bytes:
        """Return ``url``'s bytes, cached as ``cache_dir/<dataset>/<name>``."""
        path = self.cache_dir / self.dataset / name
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            if static or age_h < self.cache_ttl_hours:
                return path.read_bytes()
        body = self._get(url).content
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        return body

    def ckan_package(self, ckan_id: str | None = None) -> dict:
        """The EnviDat CKAN ``package_show`` record for the dataset."""
        ckan_id = ckan_id or DATASETS[self.dataset]["ckan_id"]
        if ckan_id not in self._ckan:
            resp = self._get(CKAN_URL, params={"id": ckan_id})
            try:
                payload = resp.json()
            except ValueError as exc:
                raise SLFError(
                    f"EnviDat CKAN returned non-JSON for {ckan_id!r}"
                ) from exc
            if not payload.get("success"):
                raise SLFError(f"EnviDat CKAN lookup failed for {ckan_id!r}: {payload}")
            self._ckan[ckan_id] = payload["result"]
        return self._ckan[ckan_id]

    # ── Dataset-specific loaders ─────────────────────────────────────────────

    def _gcos_base(self) -> str:
        """Resolve the GCOS S3 folder from the CKAN 'DATA ACCESS' resource.

        The resource URL is an EnviCloud browser link carrying ``bucket=`` and
        ``prefix=`` query parameters; the metadata resource is a direct link.
        """
        try:
            pkg = self.ckan_package()
        except SLFError as exc:
            logger.warning("SLF gcos: CKAN unavailable (%s); using fallback URL", exc)
            return _GCOS_FALLBACK_BASE
        for res in pkg.get("resources", []):
            url = res.get("url", "")
            if "bucket=" in url and "prefix=" in url:
                query = parse_qs(urlparse(url.replace("#/?", "?")).query)
                bucket = query.get("bucket", [""])[0].rstrip("/")
                prefix = query.get("prefix", [""])[0].strip("/")
                if bucket and prefix:
                    return f"{bucket}/{prefix}"
        for res in pkg.get("resources", []):
            url = res.get("url", "")
            if url.endswith("/metadata.txt"):
                return url.rsplit("/", 1)[0]
        logger.warning("SLF gcos: no data resource on CKAN record; using fallback URL")
        return _GCOS_FALLBACK_BASE

    def _new_snow_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = self._open_new_snow_zip()
        return self._zip

    def _open_new_snow_zip(self) -> zipfile.ZipFile:
        path = self.cache_dir / self.dataset / "new_snow_dataset_v1.zip"
        if not path.exists():
            pkg = self.ckan_package()
            urls = [
                r["url"]
                for r in pkg.get("resources", [])
                if str(r.get("url", "")).lower().endswith(".zip")
            ]
            if not urls:
                raise SLFError("new_snow: no .zip resource on the EnviDat record")
            name = urls[0].rsplit("/", 1)[-1]
            body = self._cached(name, urls[0], static=True)
        else:
            body = path.read_bytes()
        try:
            return zipfile.ZipFile(io.BytesIO(body))
        except zipfile.BadZipFile as exc:
            raise SLFError(f"new_snow: download is not a zip file: {exc}") from exc

    @staticmethod
    def _zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
        for name in zf.namelist():
            if name.endswith(suffix):
                return zf.read(name).decode("utf-8", errors="replace")
        raise SLFError(f"new_snow: {suffix!r} missing from the zip")

    def _gcos_frame(self) -> pd.DataFrame:
        if "gcos" not in self._frames:
            text = self._cached(
                "SWE_GCOS.csv", f"{self._gcos_base()}/SWE_GCOS.csv"
            ).decode("utf-8", errors="replace")
            self._frames["gcos"] = _parse_gcos_csv(text)
        return self._frames["gcos"]

    def _new_snow_frame(self, variable: str) -> pd.DataFrame:
        key = f"new_snow:{variable}"
        if key not in self._frames:
            zf = self._new_snow_zip()
            file = DATASETS["new_snow"]["variables"][variable]["file"]
            df = _parse_new_snow_table(self._zip_member(zf, file))
            self._frames[key] = df.rename(columns={"value": variable})
        return self._frames[key]

    def _measurement_frame(self, station_id: str, interval: str) -> pd.DataFrame:
        if self.dataset == "study_plot":
            name = f"{station_id}.csv"
            url = f"{MEASUREMENT_DATA_URL}/study-plot/data/by_station/{name}"
        elif interval == "daily":
            name = f"daily_snow_{station_id}.csv"
            url = f"{MEASUREMENT_DATA_URL}/imis/data/daily_snow_values/daily_snow_by_station/{name}"
        else:
            name = f"{station_id}.csv"
            url = f"{MEASUREMENT_DATA_URL}/imis/data/by_station/{name}"
        key = f"{self.dataset}:{interval}:{station_id}"
        if key not in self._frames:
            text = self._cached(name, url).decode("utf-8", errors="replace")
            self._frames[key] = _parse_measurement_csv(text)
        return self._frames[key]

    # ── Stations ─────────────────────────────────────────────────────────────

    def _station_table(self) -> dict[str, dict]:
        if self._stations is not None:
            return self._stations
        spec = DATASETS[self.dataset]
        common = {
            "operator": "WSL Institute for Snow and Avalanche Research SLF",
            "dataset": self.dataset,
            "doi": spec["doi"],
            "license": spec["license"],
        }
        stations: dict[str, dict] = {}
        if self.dataset == "gcos":
            meta = _parse_gcos_metadata(
                self._cached(
                    "metadata.txt", f"{self._gcos_base()}/metadata.txt"
                ).decode("utf-8", errors="replace")
            )
            df = self._gcos_frame()
            dates = pd.to_datetime(df["measure_date"], errors="coerce")
            span = dates.groupby(df["station_code"]).agg(["min", "max"])
            # The file is extended once a year, so "Active" = reported in the
            # file's latest season and that season is recent.
            newest = span["max"].max()
            cutoff = newest - pd.Timedelta(days=_GCOS_SEASON_DAYS)
            if newest < pd.Timestamp(date.today() - timedelta(days=_GCOS_STALE_DAYS)):
                cutoff = pd.Timestamp.max
            for code, sta in meta.items():
                first = span["min"].get(code)
                last = span["max"].get(code)
                stations[code] = {
                    **sta,
                    **common,
                    "status": "Active"
                    if last is not None and last >= cutoff
                    else "Inactive",
                    "measurement_method": "manual snow pit (SWE) and snow depth, ~biweekly",
                    "period_of_record": (
                        f"{first.date()}/{last.date()}" if first is not None else None
                    ),
                    "has_swe": True,
                    "has_snwd": True,
                    "station_url": spec["landing_url"],
                }
        elif self.dataset == "new_snow":
            zf = self._new_snow_zip()
            for code, sta in _parse_new_snow_stations(
                self._zip_member(zf, "STATION-LIST.txt")
            ).items():
                swe = self._new_snow_frame("SWE")
                has_swe = bool(
                    swe[(swe["station_code"] == code)]["SWE"]
                    .map(_to_float)
                    .notna()
                    .any()
                )
                stations[code] = {
                    **sta,
                    **common,
                    "status": "Inactive",
                    "measurement_method": (
                        "manual: daily new snow / new-snow SWE / stake HS; "
                        "biweekly snow-pit HS and SWE"
                    ),
                    "period_of_record": "2016-09-01/2022-08-31",
                    "has_swe": has_swe,
                    "has_snwd": True,
                    "station_url": spec["landing_url"],
                }
        else:
            folder = "study-plot" if self.dataset == "study_plot" else "imis"
            text = self._cached(
                "stations.csv", f"{MEASUREMENT_DATA_URL}/{folder}/stations.csv"
            ).decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
            for _, row in df.iterrows():
                stype = row.get("station_type", "") or None
                if self.dataset == "imis" and stype == "WIND":
                    continue  # wind stations measure no snow
                code = row["station_code"]
                active = str(row.get("active", "")).lower() == "true"
                stations[code] = {
                    "station_id": code,
                    "name": row.get("label") or code,
                    "latitude": _to_float(row.get("lat")),
                    "longitude": _to_float(row.get("lon")),
                    "elevation_m": _to_float(row.get("elevation")),
                    "status": "Active" if active else "Inactive",
                    **common,
                    "network": row.get("network") or None,
                    "station_type": stype,
                    "measurement_method": (
                        "manual daily morning observation"
                        if self.dataset == "study_plot"
                        else "automatic (ultrasonic snow depth); HN_1D modelled by SNOWPACK"
                    ),
                    "period_of_record": None,
                    "has_swe": False,
                    "has_snwd": True,
                    "station_url": spec["landing_url"],
                }
        self._stations = stations
        return stations

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        All stations of the selected dataset.

        Returns
        -------
        list[dict]
            ``station_id``, ``name``, ``latitude``, ``longitude``,
            ``elevation_m``, ``status``, ``operator``, ``dataset``, ``doi``,
            ``license``, ``measurement_method``, ``period_of_record``
            (``"first/last"`` where known), ``has_swe``, ``has_snwd``,
            ``station_url``; ``study_plot``/``imis`` add ``network`` and
            ``station_type``; ``new_snow`` adds LV03 coordinates.
        """
        out = [dict(s) for s in self._station_table().values()]
        if active_only:
            out = [s for s in out if s["status"] == "Active"]
        return _filter_by_bbox(out, bbox)

    # ── Data ─────────────────────────────────────────────────────────────────

    def _rows(
        self, station_id: str, variable: str, interval: str
    ) -> list[tuple[str, str | None, Any]]:
        """``(YYYY-MM-DD, iso datetime or None, raw value)`` rows, date-sorted."""
        if self.dataset == "gcos":
            df = self._gcos_frame()
            df = df[df["station_code"] == station_id]
            return [
                (str(d)[:10], None, v) for d, v in zip(df["measure_date"], df[variable])
            ]
        if self.dataset == "new_snow":
            df = self._new_snow_frame(variable)
            df = df[df["station_code"] == station_id]
            return [
                (str(d)[:10], None, v) for d, v in zip(df["measure_date"], df[variable])
            ]
        df = self._measurement_frame(station_id, interval)
        if variable not in df.columns:
            return []
        if interval == "sub_hourly":
            ts = pd.to_datetime(df["measure_date"], utc=True, errors="coerce")
            return [
                (t.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%dT%H:%M:%SZ"), v)
                for t, v in zip(ts, df[variable])
                if not pd.isna(t)
            ]
        return [
            (str(d)[:10], None, v) for d, v in zip(df["measure_date"], df[variable])
        ]

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str | None = None,
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Standardized data fetch — a flat list of observation records.

        Parameters
        ----------
        station_ids : list[str] or str or None
            Dataset station code(s), e.g. ``"5WJ"`` (gcos), ``"5WJ0"``
            (study_plot), ``"WFJ2"`` (imis).  Required unless ``bbox``.
        variables : list[str] or str or None
            Native names (``"SWE"``, ``"HS"``, ``"HN_1D"`` …) or
            standardized types (``"swe"``, ``"snwd"``).  ``None`` = snow
            only (SWE + snow depth where the dataset has them).
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or date, optional
            Inclusive ``YYYY-MM-DD`` bounds (files are fetched whole).
        interval : str or None
            ``None`` = each variable at its native cadence (``"periodic"``
            for biweekly snow pits, ``"daily"`` otherwise).  ``"periodic"``,
            ``"daily"`` or (imis) ``"sub_hourly"`` select explicitly;
            anything the dataset lacks raises :class:`SLFError`.
        include_flags : bool
            Adds ``flag`` (always ``None`` — SLF publishes no flags).

        Returns
        -------
        list[dict]
            ``{station_id, date, variable, type, value, units, interval}``
            (+ ``datetime`` for sub-hourly, + ``flag``).

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is given.
        SLFError
            On an unknown variable / interval, or a source failure.
        """
        wanted = _resolve_variables(self.dataset, variables, interval)
        if station_ids is not None:
            ids = _coerce_list(station_ids)
        elif bbox is not None:
            ids = [s["station_id"] for s in self.get_all_stations(bbox=bbox)]
        else:
            raise ValueError("Provide station_ids or bbox.")
        begin = _date_str(begin_date) if begin_date else None
        end = _date_str(end_date) if end_date else None

        records: list[dict] = []
        for sid in ids:
            for var, iv in wanted:
                try:
                    rows = self._rows(sid, var, iv)
                except SLFError as exc:
                    logger.warning(
                        "SLF %s: skipping %s/%s: %s", self.dataset, sid, var, exc
                    )
                    continue
                info = VARIABLES[var]
                for day, dt, raw in rows:
                    if begin and day < begin:
                        continue
                    if end and day > end:
                        continue
                    if _to_float(raw) is None:
                        continue  # absent observation, not a measured value
                    rec = {
                        "station_id": sid,
                        "date": day,
                        "variable": var,
                        "type": info["type"],
                        "value": _convert(var, raw),
                        "units": info["output_units"],
                        "interval": iv,
                    }
                    if dt is not None:
                        rec["datetime"] = dt
                    if include_flags:
                        rec["flag"] = None
                    records.append(rec)
        return records

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_metadata(self, station_id: str) -> dict:
        """
        Station dict plus ``data_variables`` for one station.

        ``begin_date``/``end_date``/``n_obs`` are computed from the data
        file (non-missing values) except for IMIS 30-minute variables,
        whose files are large; those report ``None``.

        Raises
        ------
        SLFError
            If the station is not in the dataset.
        """
        sta = self._station_table().get(station_id)
        if sta is None:
            raise SLFError(
                f"Station {station_id!r} not found in SLF dataset {self.dataset!r}"
            )
        out = dict(sta)
        dvars: list[dict] = []
        for var, spec in DATASETS[self.dataset]["variables"].items():
            for iv in spec["intervals"]:
                begin = end = n = None
                if iv != "sub_hourly":
                    try:
                        days = [
                            d
                            for d, _, v in self._rows(station_id, var, iv)
                            if _to_float(v) is not None
                        ]
                    except SLFError as exc:
                        logger.warning(
                            "SLF metadata: %s/%s unavailable: %s", station_id, var, exc
                        )
                        days = []
                    if days:
                        begin, end, n = min(days), max(days), len(days)
                    else:
                        n = 0
                    if n == 0:
                        continue
                info = VARIABLES[var]
                dvars.append(
                    {
                        "name": var,
                        "type": info["type"],
                        "interval": iv,
                        "units": info["output_units"],
                        "description": info["description"],
                        "notes": info["notes"],
                        "begin_date": begin,
                        "end_date": end,
                        "n_obs": n,
                    }
                )
        out["data_variables"] = dvars
        if self.dataset in ("study_plot", "imis") and dvars:
            begins = [d["begin_date"] for d in dvars if d["begin_date"]]
            ends = [d["end_date"] for d in dvars if d["end_date"]]
            if begins and ends:
                out["period_of_record"] = f"{min(begins)}/{max(ends)}"
        out["terms_url"] = _TERMS_URL
        return out
