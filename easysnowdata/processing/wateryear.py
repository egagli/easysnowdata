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

The two namings line up: water year *N* is, in both hemispheres, the snow
season that peaks in calendar year *N*. The southern convention is the one
the Andes SWE reanalysis uses (Cortés and Margulis 2017; its Dryad record,
doi:10.5061/dryad.ngf1vhj0s: "WY1985 corresponds to April 1st, 1985 through
March 31st, 1986").

A ``(station, time)`` Dataset can mix hemispheres. ``hemisphere="auto"``
in :func:`add_water_year_coords` gives each station its own water year from
its latitude (:func:`station_hemispheres`).
"""

from __future__ import annotations

import logging
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
    "station_hemispheres",
    "START_MONTH",
]

START_MONTH = {"northern": 10, "southern": 4}

_logger = logging.getLogger(__name__)


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


def station_hemispheres(latitudes: Any) -> np.ndarray:
    """``"southern"`` where the latitude is negative, else ``"northern"``.

    A missing latitude counts as northern; :func:`add_water_year_coords`
    warns about those.
    """
    lat = np.asarray(latitudes, dtype="float64")
    return np.where(lat < 0, "southern", "northern")


def add_water_year_coords(
    obj: xr.Dataset | xr.DataArray,
    hemisphere: str = "northern",
    *,
    dim: str = "time",
    names: tuple[str, str] = ("water_year", "dowy"),
    station_dim: str = "station",
    latitude: str = "latitude",
) -> xr.Dataset | xr.DataArray:
    """Attach ``water_year`` and ``dowy`` coordinates along *dim*.

    Both are plain integer coordinates, so ``obj.groupby("water_year")`` and
    ``obj.swap_dims({"time": "dowy"})`` work directly.

    *hemisphere* is ``"northern"``, ``"southern"`` or ``"auto"``. ``"auto"``
    reads each station's hemisphere from the *latitude* coordinate on
    *station_dim* and records it as a ``hemisphere`` coordinate there:

    * every station in one hemisphere: the coordinates stay 1-D on *dim*,
      exactly as with that hemisphere named;
    * stations in both: the coordinates become 2-D, ``(station_dim, dim)``,
      so a southern station's April–March season is one water year while a
      northern one's is October–September. ``groupby("water_year")`` still
      works and compares like with like (water year *N* peaks in calendar
      year *N* in both hemispheres); ``swap_dims`` needs one station first,
      e.g. ``ds.sel(station=code).swap_dims({"time": "dowy"})``.

    Without a *latitude* coordinate ``"auto"`` falls back to northern with a
    warning.
    """
    if dim not in obj.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input.")
    times_da = obj[dim]
    wy_name, dowy_name = names
    if str(hemisphere).lower() != "auto":
        return obj.assign_coords(
            {
                wy_name: (dim, np.asarray(water_year(times_da, hemisphere).values)),
                dowy_name: (
                    dim,
                    np.asarray(day_of_water_year(times_da, hemisphere).values),
                ),
            }
        )
    if station_dim not in obj.dims or latitude not in obj.coords:
        _logger.warning(
            "hemisphere='auto' needs a %r coordinate on %r; using the northern "
            "water year (1 October). Pass hemisphere= to choose.",
            latitude,
            station_dim,
        )
        return add_water_year_coords(obj, "northern", dim=dim, names=names)
    lat = np.asarray(obj[latitude].values, dtype="float64")
    if np.isnan(lat).any():
        missing = np.asarray(obj[station_dim].values)[np.isnan(lat)]
        _logger.warning(
            "No latitude for %d station(s) (%s); they get the northern water year.",
            len(missing),
            ", ".join(str(m) for m in missing[:10]),
        )
    hems = station_hemispheres(lat)
    obj = obj.assign_coords(hemisphere=(station_dim, hems.astype("str")))
    kinds = sorted(set(hems.tolist()))
    if len(kinds) == 1:
        return add_water_year_coords(obj, kinds[0], dim=dim, names=names)
    per_hem = {
        h: (
            np.asarray(water_year(times_da, h).values),
            np.asarray(day_of_water_year(times_da, h).values),
        )
        for h in kinds
    }
    wy = np.stack([per_hem[h][0] for h in hems])
    dowy = np.stack([per_hem[h][1] for h in hems])
    return obj.assign_coords(
        {wy_name: ((station_dim, dim), wy), dowy_name: ((station_dim, dim), dowy)}
    )
