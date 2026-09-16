"""SAR backscatter helpers: dB conversion and Sentinel-1 border-noise removal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "linear_to_db",
    "db_to_linear",
    "remove_border_noise",
    "S1_BORDER_NOISE_CUTOFF",
]

#: ESA's IPF 2.90 (2018-03-13) fixed border noise; earlier scenes need a threshold.
S1_BORDER_NOISE_CUTOFF = "2018-03-14"


def linear_to_db(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """``10 log10(x)``; non-positive values become NaN. Sets ``units="dB"``."""
    positive = obj.where(obj > 0)
    out = 10 * np.log10(positive)
    _set_units(out, "dB")
    return out


def db_to_linear(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """``10 ** (x / 10)``. Sets ``units="linear power"``."""
    out = 10 ** (obj / 10)
    _set_units(out, "linear power")
    return out


def _set_units(obj: xr.DataArray | xr.Dataset, units: str) -> None:
    obj.attrs["units"] = units
    if isinstance(obj, xr.Dataset):
        for var in obj.data_vars.values():
            var.attrs["units"] = units


def remove_border_noise(
    obj: xr.DataArray | xr.Dataset,
    threshold: float = 0.001,
    *,
    cutoff: str | pd.Timestamp = S1_BORDER_NOISE_CUTOFF,
    dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Mask falsely low backscatter (border noise) in linear-power data.

    Scenes before *cutoff* keep values above *threshold*; later scenes keep
    values above 0. CRS metadata is preserved.
    """
    if dim not in obj.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input.")
    is_old = obj[dim] < np.datetime64(pd.Timestamp(cutoff))
    limit = xr.where(is_old, threshold, 0.0)
    return obj.where(obj > limit)
