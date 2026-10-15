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
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Interval enum (DESIGN.md §3.3) ───────────────────────────────────────────

#: The one shared interval vocabulary.  Clients map native duration codes
#: to these values and back; values outside this set must never leak into
#: records or artifacts.
INTERVALS: frozenset[str] = frozenset(
    {
        "periodic",
        "monthly",
        "semi_monthly",
        "daily",
        "sub_daily",
        "hourly",
        "sub_hourly",
        "instantaneous",
        "annual",
    }
)

# ── Type vocabulary (DESIGN.md §3.2) ─────────────────────────────────────────

#: The one shared standardized-type vocabulary. Every ``VARIABLES``/``SENSORS``
#: entry's ``type`` must be one of these, and nothing outside this set may
#: reach a record or an artifact.
#:
#: ``snowfall`` is new snow as a *depth of snow* and is deliberately distinct
#: from ``precip``, a depth of *water*: they are not interconvertible, since
#: 5 cm of new snow is roughly 5 mm of water rather than 50 mm. ``snwd`` is
#: the snow already on the ground.
TYPES: frozenset[str] = frozenset(
    {
        "swe",
        "snwd",
        "snowfall",
        "temp",
        "temp_max",
        "temp_min",
        "precip",
        "rh",
        "wind_spd",
        "wind_gust",
        "wind_dir",
        "wind_run",
        "solar",
        "baro",
        "density",
        "snow_line",
        "soil_moisture",
        "other",
    }
)

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
        yield items[i : i + size]


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
    json: Any = None,
) -> requests.Response:
    """One retry policy for every client.

    ``json`` is sent as a JSON request body (for POST APIs such as ASP.NET
    page methods); ``params`` stay in the query string.

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
            if json is None:
                response = session.request(method, url, params=params, timeout=timeout)
            else:
                response = session.request(
                    method, url, params=params, json=json, timeout=timeout
                )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                raise error_cls(
                    f"Request to {url} failed after {max_retries} attempts: {exc}"
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
                f"HTTP 404 Not Found: {url} (params={params!r}): {response.text[:300]}"
            )

        if response.status_code == 429:
            if attempt < max_retries:
                try:
                    delay = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    delay = float(backoff * attempt)
                logger.warning(
                    "HTTP 429 from %s (attempt %d/%d) — retrying in %.1fs",
                    url,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise error_cls(
                f"HTTP 429 Too Many Requests from {url} after {max_retries} attempts"
            )

        if response.status_code >= 500:
            logger.warning(
                "HTTP %d from %s (attempt %d/%d) — retrying in %ds",
                response.status_code,
                url,
                attempt,
                max_retries,
                backoff * attempt,
            )
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            raise error_cls(
                f"HTTP {response.status_code} from {url} after {max_retries} attempts"
            )

        raise error_cls(
            f"HTTP {response.status_code} from {url}: {response.text[:200]}"
        )

    raise error_cls(f"Exhausted retries for {url}")  # unreachable


# ── Large-file download (ranged, parallel, resumable) ────────────────────────

#: Files at least this large are fetched in parallel byte ranges.
PARALLEL_DOWNLOAD_MIN_BYTES = 200 * 1024 * 1024


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    error_cls: type[Exception],
    size: int | None = None,
    md5: str | None = None,
    parts: int = 8,
    timeout: int = 120,
    max_retries: int = 5,
    backoff: int = 4,
    progress: bool = True,
) -> Path:
    """Download ``url`` to ``dest``, resuming and verifying when possible.

    Large files (``size`` ≥ :data:`PARALLEL_DOWNLOAD_MIN_BYTES`, with a
    server that honours ``Range``) are fetched as ``parts`` byte ranges in
    parallel, each written to ``dest.partNN``. An interrupted download
    resumes from what those part files already hold, so a dropped
    connection on a multi-gigabyte file costs one retry of one range, not
    the whole file. Smaller files stream in one request. The result is
    md5-checked when ``md5`` is given, then moved into place atomically.

    Parameters
    ----------
    session : requests.Session
        Session to issue the requests with.
    url : str
        File URL.
    dest : Path
        Final path. Nothing is written there until the file is complete
        and verified.
    error_cls : type[Exception]
        The calling client's error class, raised on any failure.
    size : int, optional
        Expected size in bytes (needed for the parallel path).
    md5 : str, optional
        Expected md5 hex digest; a ``"md5:"`` prefix is accepted.
    parts : int
        Number of parallel ranges for large files.
    timeout, max_retries, backoff
        Per-request timeout and retry policy for each range.
    progress : bool
        Show a progress bar in interactive sessions.

    Returns
    -------
    Path
        ``dest``.

    Raises
    ------
    error_cls
        On HTTP failure after retries, a size mismatch or an md5 mismatch.
    """
    import hashlib  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from tqdm.auto import tqdm  # noqa: PLC0415

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = md5.split(":", 1)[-1] if md5 else None
    parallel = size is not None and size >= PARALLEL_DOWNLOAD_MIN_BYTES and parts > 1
    n_parts = parts if parallel else 1
    if size:
        bounds = [
            (i * size // n_parts, (i + 1) * size // n_parts - 1) for i in range(n_parts)
        ]
    else:
        bounds = [(0, None)]
    part_paths = [dest.with_name(f"{dest.name}.part{i:02d}") for i in range(n_parts)]
    if parallel and isinstance(session, requests.Session):
        # One pooled connection per range, so parallel ranges don't churn
        # connections ("Connection pool is full" warnings).
        adapter = session.get_adapter(url)
        if getattr(adapter, "_pool_maxsize", n_parts) < n_parts:
            session.mount(
                url.split("/", 3)[0] + "//" + url.split("/", 3)[2] + "/",
                requests.adapters.HTTPAdapter(pool_maxsize=n_parts),
            )

    try:
        from easysnowdata.config import is_interactive  # noqa: PLC0415

        show = progress and is_interactive()
    except Exception:  # noqa: BLE001 — progress display is optional
        show = False
    have = sum(p.stat().st_size for p in part_paths if p.exists())
    bar = tqdm(
        total=size,
        initial=have,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        disable=not show,
    )

    def fetch(i: int) -> None:
        start, end = bounds[i]
        path = part_paths[i]
        want = None if end is None else end - start + 1
        for attempt in range(1, max_retries + 1):
            done = path.stat().st_size if path.exists() else 0
            if want is not None and done >= want:
                return
            headers = {}
            if parallel or done:
                last = "" if end is None else str(end)
                headers["Range"] = f"bytes={start + done}-{last}"
            try:
                with session.get(
                    url, headers=headers, stream=True, timeout=timeout
                ) as resp:
                    if resp.status_code == 200 and headers:
                        if parallel:
                            raise error_cls(f"{url} ignores Range requests")
                        done = 0  # server restarted from byte 0
                        path.unlink(missing_ok=True)
                    elif resp.status_code not in (200, 206):
                        raise error_cls(f"HTTP {resp.status_code} from {url}")
                    with open(path, "ab") as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            fh.write(chunk)
                            bar.update(len(chunk))
                if want is None or path.stat().st_size >= want:
                    return
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Range %d of %s failed (attempt %d/%d): %s",
                    i,
                    dest.name,
                    attempt,
                    max_retries,
                    exc,
                )
            time.sleep(backoff * attempt)
        raise error_cls(
            f"Download of {url} did not complete after {max_retries} attempts; "
            "run again to resume"
        )

    try:
        if n_parts == 1:
            fetch(0)
        else:
            with ThreadPoolExecutor(max_workers=n_parts) as pool:
                for future in [pool.submit(fetch, i) for i in range(n_parts)]:
                    future.result()
    finally:
        bar.close()

    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for path in part_paths:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            path.unlink()
    got = tmp.stat().st_size
    if size is not None and got != size:
        tmp.unlink()
        raise error_cls(f"{dest.name}: got {got} bytes, expected {size}")
    if expected:
        digest = hashlib.md5()  # noqa: S324 — integrity check, not security
        with open(tmp, "rb") as fh:
            for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            tmp.unlink()
            raise error_cls(f"{dest.name}: md5 mismatch (download corrupted)")
    tmp.replace(dest)
    return dest
