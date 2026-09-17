"""The one temporal input, ``time`` (design contract §2.2), and "today".

``parse_time`` accepts anything :func:`pandas.to_datetime` or STAC datetime
syntax accepts — ``"2023-10"``, ``("2023-10-01", "2024-06-30")``,
``"2023-10/2024-06"``, a ``slice``, a ``pandas.Period`` — and returns an
inclusive ``(start, end)`` pair of timestamps. Partial dates expand to the
period they name (``"2023-10"`` is the whole month). An open end means "now",
computed at call time, never at import (§2.9).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd

__all__ = ["parse_time", "to_stac_datetime", "today", "now"]

TimeLike = Any


def now() -> pd.Timestamp:
    """The current UTC time as a naive timestamp (computed when called)."""
    return pd.Timestamp(_dt.datetime.now(tz=_dt.UTC)).tz_localize(None)


def today() -> str:
    """Today's date as ``YYYY-MM-DD`` (computed when called)."""
    return now().strftime("%Y-%m-%d")


def _period_bounds(value: Any, *, end: bool) -> pd.Timestamp:
    """Timestamp for one endpoint; strings naming a period expand to its edge."""
    if value is None:
        raise ValueError("None is not a valid endpoint here.")
    if isinstance(value, pd.Period):
        return value.end_time.floor("s") if end else value.start_time
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1]
        try:
            period = pd.Period(text)
        except Exception:  # noqa: BLE001 — fall back to a plain parse
            period = None
        if period is not None and period.freqstr[0] in "YMDQ":
            # Only expand *partial* dates ("2023", "2023-10", "2023-10-01");
            # anything with a time component is taken literally.
            if "T" not in text and ":" not in text:
                return period.end_time.floor("s") if end else period.start_time
        stamp = pd.Timestamp(text)
    else:
        stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def parse_time(
    time: TimeLike, *, default_start: TimeLike = None
) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """Return ``(start, end)`` for any supported ``time`` input.

    Parameters
    ----------
    time
        ``None`` (open interval ending now), a string (``"2023-10"``,
        ``"2023-10-01"``, ``"2023-10/2024-06"``, ``"2023-10-01T00:00:00Z/.."``,
        ``"../2024-06"``), a two-item tuple/list (either item may be ``None``),
        a ``slice``, a ``pandas.Period`` or a single datetime-like.
    default_start
        Used when the start is open; ``None`` keeps it open (``None``).

    Returns
    -------
    (start, end)
        Naive UTC timestamps; ``end`` defaults to now, ``start`` may be ``None``.
    """
    start: Any
    end: Any
    if time is None:
        start, end = None, None
    elif isinstance(time, slice):
        start, end = time.start, time.stop
    elif isinstance(time, str) and "/" in time:
        left, right = time.split("/", 1)
        start = None if left.strip() in ("", "..") else left
        end = None if right.strip() in ("", "..") else right
    elif isinstance(time, (tuple, list)):
        if len(time) != 2:
            raise ValueError(f"time must be (start, end); got {len(time)} items.")
        start, end = time
    else:
        start, end = time, time

    start_ts = (
        _period_bounds(start if start is not None else default_start, end=False)
        if (start is not None or default_start is not None)
        else None
    )
    end_ts = _period_bounds(end, end=True) if end is not None else now()
    if start_ts is not None and start_ts > end_ts:
        raise ValueError(f"time start {start_ts} is after end {end_ts}.")
    return start_ts, end_ts


def to_stac_datetime(time: TimeLike) -> str:
    """Render ``time`` as a STAC ``datetime`` interval (``start/end``, RFC 3339 UTC)."""
    start, end = parse_time(time)
    left = ".." if start is None else start.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{left}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
