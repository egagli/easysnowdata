"""Quality masks: Sentinel-2 SCL classes, HLS Fmask bits, nodata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

from easysnowdata.processing.categorical import meaning_key
from easysnowdata.processing.optical import SCL_CLASSES

__all__ = [
    "SCL_NAMES",
    "DEFAULT_SCL_REMOVE",
    "FMASK_BITS",
    "FMASK_AEROSOL_LEVELS",
    "DEFAULT_FMASK_REMOVE",
    "scl_mask",
    "apply_scl_mask",
    "fmask_bit",
    "fmask_aerosol_level",
    "fmask_mask",
    "apply_fmask",
    "mask_nodata",
]

#: Short SCL names → class values (the long names are in ``SCL_CLASSES``).
SCL_NAMES: dict[str, int] = {
    "no_data": 0,
    "saturated_defective": 1,
    "topo_shadows": 2,
    "cloud_shadows": 3,
    "vegetation": 4,
    "not_vegetated": 5,
    "water": 6,
    "unclassified": 7,
    "cloud_medium": 8,
    "cloud_high": 9,
    "thin_cirrus": 10,
    "snow_ice": 11,
}
#: What the legacy ``Sentinel2.mask_data()`` removed by default.
DEFAULT_SCL_REMOVE: tuple[str, ...] = (
    "no_data",
    "saturated_defective",
    "topo_shadows",
    "cloud_shadows",
    "cloud_medium",
    "cloud_high",
    "thin_cirrus",
)

#: HLS v2.0 Fmask bit positions (bits 6–7 hold the aerosol level).
FMASK_BITS: dict[str, int] = {
    "cirrus": 0,
    "cloud": 1,
    "adjacent": 2,
    "cloud_shadow": 3,
    "snow_ice": 4,
    "water": 5,
}
FMASK_AEROSOL_LEVELS: dict[str, int] = {
    "climatology": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}
DEFAULT_FMASK_REMOVE: tuple[str, ...] = ("cirrus", "cloud", "adjacent", "cloud_shadow")


def _scl_values(classes: Iterable[str | int]) -> list[int]:
    values = []
    long_names = {
        meaning_key(name).lower(): value for value, (name, _) in SCL_CLASSES.items()
    }
    for c in classes:
        if isinstance(c, (int, np.integer)):
            values.append(int(c))
            continue
        key = meaning_key(c).lower()
        if key in SCL_NAMES:
            values.append(SCL_NAMES[key])
        elif key in long_names:
            values.append(long_names[key])
        else:
            raise ValueError(f"Unknown SCL class {c!r}; known: {list(SCL_NAMES)}.")
    return values


def scl_mask(
    scl: xr.DataArray, remove: Sequence[str | int] = DEFAULT_SCL_REMOVE
) -> xr.DataArray:
    """Boolean mask that is ``True`` where a pixel is *kept* (its SCL class is not in *remove*).

    Classes may be short names (``"cloud_high"``), long names or integers.
    """
    return ~scl.isin(_scl_values(remove))


def apply_scl_mask(
    obj: xr.Dataset | xr.DataArray,
    scl: xr.DataArray | None = None,
    remove: Sequence[str | int] = DEFAULT_SCL_REMOVE,
) -> xr.Dataset | xr.DataArray:
    """Mask *obj* to NaN where the SCL class is in *remove* (``scl`` defaults to ``obj["scl"]``)."""
    if scl is None:
        if not isinstance(obj, xr.Dataset) or "scl" not in obj:
            raise ValueError("Pass scl= or a Dataset with an 'scl' variable.")
        scl = obj["scl"]
    return obj.where(scl_mask(scl, remove))


def fmask_bit(fmask: xr.DataArray, bit: int) -> xr.DataArray:
    """The value (0/1) of *bit* in each Fmask pixel."""
    return (fmask.astype("uint8") >> bit) & 1


def fmask_aerosol_level(fmask: xr.DataArray) -> xr.DataArray:
    """The aerosol level 0–3 encoded in Fmask bits 6–7."""
    return (fmask.astype("uint8") >> 6) & 3


def fmask_mask(
    fmask: xr.DataArray,
    remove: Sequence[str] = DEFAULT_FMASK_REMOVE,
    aerosol_remove: Sequence[str] = (),
) -> xr.DataArray:
    """Boolean mask that is ``True`` where a pixel is *kept*.

    *remove* names Fmask bits (:data:`FMASK_BITS`); *aerosol_remove* names
    aerosol levels (:data:`FMASK_AEROSOL_LEVELS`). Fill pixels (255) are removed.
    """
    bad = fmask == 255
    for name in remove:
        if name not in FMASK_BITS:
            raise ValueError(f"Unknown Fmask flag {name!r}; known: {list(FMASK_BITS)}.")
        bad = bad | (fmask_bit(fmask, FMASK_BITS[name]) == 1)
    if aerosol_remove:
        levels = []
        for name in aerosol_remove:
            if name not in FMASK_AEROSOL_LEVELS:
                raise ValueError(
                    f"Unknown aerosol level {name!r}; known: {list(FMASK_AEROSOL_LEVELS)}."
                )
            levels.append(FMASK_AEROSOL_LEVELS[name])
        bad = bad | fmask_aerosol_level(fmask).isin(levels)
    return ~bad


def apply_fmask(
    obj: xr.Dataset | xr.DataArray,
    fmask: xr.DataArray | None = None,
    remove: Sequence[str] = DEFAULT_FMASK_REMOVE,
    aerosol_remove: Sequence[str] = (),
) -> xr.Dataset | xr.DataArray:
    """Mask *obj* to NaN where Fmask flags are set (``fmask`` defaults to ``obj["Fmask"]``).

    Note: Fmask cloud detection is unreliable over snow and ice.
    """
    if fmask is None:
        if not isinstance(obj, xr.Dataset) or "Fmask" not in obj:
            raise ValueError("Pass fmask= or a Dataset with an 'Fmask' variable.")
        fmask = obj["Fmask"]
    return obj.where(fmask_mask(fmask, remove, aerosol_remove))


def mask_nodata(
    obj: xr.Dataset | xr.DataArray, nodata: float | int | None = None
) -> xr.Dataset | xr.DataArray:
    """Replace the nodata value with NaN (converting integers to float).

    *nodata* defaults to each variable's ``nodata`` attr, then ``rio.nodata``.
    Variables without a known nodata value are returned unchanged.
    """
    if isinstance(obj, xr.Dataset):
        out = obj.copy()
        for name in obj.data_vars:
            out[name] = mask_nodata(obj[name], nodata)
        return out
    value = nodata
    if value is None:
        value = obj.attrs.get("nodata")
    if value is None:
        value = getattr(getattr(obj, "rio", None), "nodata", None)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return obj
    masked = obj.where(obj != value)
    masked.attrs = {k: v for k, v in obj.attrs.items() if k != "nodata"}
    return masked
