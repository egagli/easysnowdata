"""Vectorized water-year helpers.

This is the single copy of these functions (§12 Q16): the scalar
``utils.datetime_to_WY``/``datetime_to_DOWY`` that used to be applied with
``pd.Index.map`` and the ``global_snow_networks`` ``utils/utils.py``
implementations are both reconciled here. The names ``wy_start``/``wy_end``
from that repo are :func:`water_year_bounds`, ``wy_date_range`` is
:func:`water_year_range`, ``wy_length`` is :func:`water_year_length` and
``add_wy_coords`` is :func:`add_water_year_coords`; all of them also take a
*hemisphere*, which the originals did not.

A northern-hemisphere water year starts 1 October and is named for the
calendar year in which it *ends* (WY 2021 = 2020-10-01 … 2021-09-30); a
southern-hemisphere one starts 1 April and is named for the year it starts.
For aggregation prefer the pandas anchored offsets:
``swe_ds.resample(time="YS-OCT").max()``. Day-of-water-year has no pandas
primitive, hence :func:`day_of_water_year` and :func:`add_water_year_coords`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "water_year_start",
    "water_year",
    "day_of_water_year",
    "water_year_bounds",
    "water_year_range",
    "water_year_length",
    "add_water_year_coords",
    "START_MONTH",
]

START_MONTH = {"northern": 10, "southern": 4}


def _start_month(hemisphere: str) -> int:
    try:
        return START_MONTH[hemisphere.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"hemisphere must be 'northern' or 'southern', got {hemisphere!r}."
        ) from None


def _as_datetime64(times: Any) -> tuple[np.ndarray, Any]:
    """Return (datetime64[ns] ndarray, template) where template rebuilds the input type."""
    if isinstance(times, xr.DataArray):
        return pd.to_datetime(times.values).values.astype("datetime64[ns]"), times
    if isinstance(times, pd.Series):
        return pd.to_datetime(times).values.astype("datetime64[ns]"), times
    if isinstance(times, pd.Index):
        return pd.to_datetime(times).values.astype("datetime64[ns]"), times
    if np.ndim(times) == 0:
        return np.array(
            [pd.Timestamp(times).to_datetime64()], dtype="datetime64[ns]"
        ), None
    return pd.to_datetime(np.asarray(times)).values.astype(
        "datetime64[ns]"
    ), np.asarray(times)


def _wrap(values: np.ndarray, template: Any, name: str) -> Any:
    if template is None:
        return values[0].item()
    if isinstance(template, xr.DataArray):
        return xr.DataArray(
            values, dims=template.dims, coords=template.coords, name=name
        )
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=name)
    if isinstance(template, pd.Index):
        return pd.Index(values, name=name)
    return values


def water_year_start(times: Any, hemisphere: str = "northern") -> Any:
    """The first day of the water year containing each time (same container type as the input)."""
    dt64, template = _as_datetime64(times)
    month = _start_month(hemisphere)
    ts = pd.DatetimeIndex(dt64)
    start_year = np.where(ts.month < month, ts.year - 1, ts.year)
    starts = pd.to_datetime(
        {"year": start_year, "month": month, "day": 1}
    ).values.astype("datetime64[ns]")
    return (
        _wrap(starts, template, "water_year_start")
        if template is not None
        else pd.Timestamp(starts[0])
    )


def water_year(times: Any, hemisphere: str = "northern") -> Any:
    """The water year of each time as an integer (calendar year the WY ends in for the north)."""
    dt64, template = _as_datetime64(times)
    month = _start_month(hemisphere)
    ts = pd.DatetimeIndex(dt64)
    start_year = np.where(ts.month < month, ts.year - 1, ts.year)
    wy = start_year + (1 if hemisphere.lower() == "northern" else 0)
    return _wrap(wy.astype("int64"), template, "water_year")


def day_of_water_year(times: Any, hemisphere: str = "northern") -> Any:
    """Day of the water year, 1-indexed (1 October = 1 in the northern hemisphere)."""
    dt64, template = _as_datetime64(times)
    month = _start_month(hemisphere)
    ts = pd.DatetimeIndex(dt64)
    start_year = np.where(ts.month < month, ts.year - 1, ts.year)
    starts = pd.to_datetime(
        {"year": start_year, "month": month, "day": 1}
    ).values.astype("datetime64[ns]")
    days = (
        (dt64.astype("datetime64[D]") - starts.astype("datetime64[D]")).astype("int64")
    ) + 1
    return _wrap(days, template, "dowy")


def water_year_bounds(
    year: int, hemisphere: str = "northern"
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of water year *year*.

    >>> water_year_bounds(2024)
    (Timestamp('2023-10-01 00:00:00'), Timestamp('2024-09-30 00:00:00'))
    """
    month = _start_month(hemisphere)
    start_year = int(year) - 1 if hemisphere.lower() == "northern" else int(year)
    start = pd.Timestamp(start_year, month, 1)
    return start, start + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def water_year_range(year: int, hemisphere: str = "northern") -> pd.DatetimeIndex:
    """Daily ``DatetimeIndex`` spanning water year *year* (365 or 366 days)."""
    start, end = water_year_bounds(year, hemisphere)
    return pd.date_range(start, end, freq="D")


def water_year_length(year: int, hemisphere: str = "northern") -> int:
    """Number of days in water year *year* — 366 when 29 February falls inside it."""
    start, end = water_year_bounds(year, hemisphere)
    return int((end - start).days) + 1


def add_water_year_coords(
    obj: xr.Dataset | xr.DataArray,
    hemisphere: str = "northern",
    *,
    dim: str = "time",
    names: tuple[str, str] = ("water_year", "dowy"),
) -> xr.Dataset | xr.DataArray:
    """Attach ``water_year`` and ``dowy`` coordinates along *dim*.

    Both are plain integer coordinates, so ``obj.groupby("water_year")`` and
    ``obj.swap_dims({"time": "dowy"})`` work directly.
    """
    if dim not in obj.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input.")
    times_da = obj[dim]
    wy_name, dowy_name = names
    return obj.assign_coords(
        {
            wy_name: (dim, np.asarray(water_year(times_da, hemisphere).values)),
            dowy_name: (
                dim,
                np.asarray(day_of_water_year(times_da, hemisphere).values),
            ),
        }
    )
