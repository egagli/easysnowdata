# -*- coding: utf-8 -*-
"""
clients/_common.py
==================
Shared helpers for all data-source clients (DESIGN.md §3.1).

Everything here used to be re-implemented per client with slightly
different behaviour (four sentinel policies, five retry loops, three
``_to_float`` variants).  Clients import from this module instead of
defining their own copies.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

# ── Interval enum (DESIGN.md §3.3) ───────────────────────────────────────────

#: The one shared interval vocabulary.  Clients map native duration codes
#: to these values and back; values outside this set must never leak into
#: records or artifacts.
INTERVALS: frozenset[str] = frozenset({
    "periodic",
    "monthly",
    "semi_monthly",
    "daily",
    "sub_daily",
    "hourly",
    "sub_hourly",
    "instantaneous",
    "annual",
})

# ── Missing-value sentinels (DESIGN.md §3.6) ─────────────────────────────────

#: Numeric sentinels that mean "missing" across the sources.
MISSING_SENTINELS: frozenset[float] = frozenset({-9999.0, -999.0, -99999.0})

#: String tokens that mean "missing" in CSV/HTML payloads.
MISSING_TOKENS: frozenset[str] = frozenset({"", "na", "nan", "null", "none"})


# ── Small shared helpers ─────────────────────────────────────────────────────

def coerce_list(value: list | tuple | set | str | int) -> list[str]:
    """Coerce a scalar or iterable of ids/codes to a list of strings."""
    if isinstance(value, (str, int)):
        return [str(value)]
    return [str(v) for v in value]


def date_str(d: str | date | datetime) -> str:
    """Normalize a date-like object to a ``YYYY-MM-DD`` string."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def to_float(value: Any) -> float | None:
    """Parse a number that may carry commas, sentinels, or NaN.

    Returns ``None`` for missing tokens (``NA``/``NULL``/…), numeric
    missing sentinels (−9999 family), NaN, and unparseable input.
    """
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if text.lower() in MISSING_TOKENS:
        return None
    try:
        f = float(text)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in MISSING_SENTINELS:
        return None
    return f


def filter_by_bbox(
    stations: list[dict],
    bbox: tuple[float, float, float, float] | None,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
) -> list[dict]:
    """Keep stations inside ``(min_lon, min_lat, max_lon, max_lat)``.

    Stations without coordinates are dropped (they cannot be located).
    """
    if bbox is None:
        return stations
    min_lon, min_lat, max_lon, max_lat = bbox
    kept: list[dict] = []
    for sta in stations:
        lat, lon = sta.get(lat_key), sta.get(lon_key)
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            kept.append(sta)
    return kept


def chunk(items: list, size: int) -> Iterator[list]:
    """Yield successive ``size``-sized chunks of ``items``."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ── Shared HTTP retry loop (DESIGN.md §3.7) ──────────────────────────────────

def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    error_cls: type[Exception],
    timeout: int = 60,
    max_retries: int = 3,
    backoff: int = 4,
    method: str = "GET",
) -> requests.Response:
    """One retry policy for every client.

    - Network errors and HTTP 5xx retry with linear backoff
      (``backoff × attempt`` seconds).
    - HTTP 429 retries honouring ``Retry-After`` when present.
    - HTTP 400/404 raise immediately with the response body excerpt.
    - Anything else non-OK (401, 403, …) raises immediately.

    Raises ``error_cls`` (the calling client's ``{Client}Error``) in every
    failure case.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = session.request(
                method, url, params=params, timeout=timeout
            )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed (attempt %d/%d): %s",
                attempt, max_retries, exc,
            )
            if attempt == max_retries:
                raise error_cls(
                    f"Request to {url} failed after "
                    f"{max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff * attempt)
            continue

        if response.ok:
            return response

        if response.status_code == 400:
            raise error_cls(
                f"HTTP 400 Bad Request: {url} "
                f"(params={params!r}): {response.text[:500]}"
            )

        if response.status_code == 404:
            raise error_cls(
                f"HTTP 404 Not Found: {url} "
                f"(params={params!r}): {response.text[:300]}"
            )

        if response.status_code == 429:
            if attempt < max_retries:
                try:
                    delay = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    delay = float(backoff * attempt)
                logger.warning(
                    "HTTP 429 from %s (attempt %d/%d) — retrying in %.1fs",
                    url, attempt, max_retries, delay,
                )
                time.sleep(delay)
                continue
            raise error_cls(
                f"HTTP 429 Too Many Requests from {url} "
                f"after {max_retries} attempts"
            )

        if response.status_code >= 500:
            logger.warning(
                "HTTP %d from %s (attempt %d/%d) — retrying in %ds",
                response.status_code, url,
                attempt, max_retries, backoff * attempt,
            )
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            raise error_cls(
                f"HTTP {response.status_code} from {url} "
                f"after {max_retries} attempts"
            )

        raise error_cls(
            f"HTTP {response.status_code} from {url}: {response.text[:200]}"
        )

    raise error_cls(f"Exhausted retries for {url}")  # unreachable
