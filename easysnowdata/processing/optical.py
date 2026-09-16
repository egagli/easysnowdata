"""Optical processing: Sentinel-2 baseline harmonization, scale/offset,
spectral indices, RGB composites and contrast stretches. Pure functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "SCL_CLASSES",
    "S2_BASELINE_CUTOFF",
    "S2_BASELINE_OFFSET",
    "S2_REFLECTANCE_BANDS",
    "harmonize_s2_baseline",
    "scale_offset",
    "normalized_difference",
    "ndsi",
    "ndvi",
    "ndwi",
    "ndbi",
    "evi",
    "rgb",
    "stretch_percentile",
    "stretch_clahe",
]

#: Sen2Cor scene classification: value → (meaning, color).
#: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
SCL_CLASSES: dict[int, tuple[str, str]] = {
    0: ("No Data (Missing data)", "#000000"),
    1: ("Saturated or defective pixel", "#ff0000"),
    2: ("Topographic casted shadows", "#2f2f2f"),
    3: ("Cloud shadows", "#643200"),
    4: ("Vegetation", "#00a000"),
    5: ("Not-vegetated", "#ffe65a"),
    6: ("Water", "#0000ff"),
    7: ("Unclassified", "#808080"),
    8: ("Cloud medium probability", "#c0c0c0"),
    9: ("Cloud high probability", "#ffffff"),
    10: ("Thin cirrus", "#64c8ff"),
    11: ("Snow or ice", "#ff96ff"),
}

#: ESA processing baseline 04.00: from this date L2A digital numbers carry a
#: +1000 offset (BOA_ADD_OFFSET) that must be undone to compare with older data.
S2_BASELINE_CUTOFF = "2022-01-25"
S2_BASELINE_OFFSET = 1000
S2_REFLECTANCE_BANDS = (
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09", "swir16", "swir22",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
)  # fmt: skip


def harmonize_s2_baseline(
    obj: xr.Dataset | xr.DataArray,
    *,
    cutoff: str | pd.Timestamp = S2_BASELINE_CUTOFF,
    offset: int | float = S2_BASELINE_OFFSET,
    bands: Sequence[str] | None = None,
    dim: str = "time",
) -> xr.Dataset | xr.DataArray:
    """Undo the post-2022-01-25 Sentinel-2 L2A offset so all dates share one baseline.

    Values acquired on or after *cutoff* become ``clip(min=offset) - offset``;
    earlier values are untouched. On a Dataset only *bands* (default: the
    reflectance band names present) are changed; the SCL and other layers are
    left alone. Where the catalog already provides ``raster:bands`` offsets
    (Earth Search), prefer :func:`scale_offset` instead of this date rule.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    if dim not in obj.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input.")
    is_new = obj[dim] >= np.datetime64(cutoff_ts)

    def _fix(da: xr.DataArray) -> xr.DataArray:
        fixed = da.clip(min=offset) - offset
        return xr.where(is_new, fixed, da, keep_attrs=True).transpose(*da.dims)

    if isinstance(obj, xr.DataArray):
        return _fix(obj)
    names = [b for b in (bands or S2_REFLECTANCE_BANDS) if b in obj.data_vars]
    out = obj.copy()
    for name in names:
        out[name] = _fix(obj[name])
    return out


_SCALE_KEYS = ("scale", "scale_factor")
_OFFSET_KEYS = ("offset", "add_offset")


def _coerce(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scale_offset(
    obj: xr.Dataset | xr.DataArray,
    scale: float | Mapping[str, float] | None = None,
    offset: float | Mapping[str, float] | None = None,
    *,
    nodata_to_nan: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Apply ``value * scale + offset`` and return float data.

    *scale* / *offset* may be scalars, per-variable mappings, or ``None`` to
    read ``scale``/``scale_factor`` and ``offset``/``add_offset`` from each
    variable's attrs (odc-stac deliberately ignores ``raster:bands`` scale and
    offset, so this is where they get applied). A variable's ``nodata`` attr
    (or ``rio.nodata``) is masked to NaN first when *nodata_to_nan*. The scale
    and offset attrs are removed from the result so it cannot be applied twice.
    """
    if isinstance(obj, xr.Dataset):
        out = obj.copy()
        for name in obj.data_vars:
            s = scale.get(name) if isinstance(scale, Mapping) else scale
            o = offset.get(name) if isinstance(offset, Mapping) else offset
            out[name] = scale_offset(obj[name], s, o, nodata_to_nan=nodata_to_nan)
        return out

    attrs = dict(obj.attrs)
    s = (
        _coerce(scale)
        if scale is not None
        else next((_coerce(attrs[k]) for k in _SCALE_KEYS if k in attrs), None)
    )
    o = (
        _coerce(offset)
        if offset is not None
        else next((_coerce(attrs[k]) for k in _OFFSET_KEYS if k in attrs), None)
    )
    da = obj
    if nodata_to_nan:
        nodata = attrs.get("nodata", getattr(getattr(obj, "rio", None), "nodata", None))
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            da = da.where(da != nodata)
    if s is None and o is None and not nodata_to_nan:
        return da
    result = da.astype("float32") if not np.issubdtype(da.dtype, np.floating) else da
    if s is not None and s != 1:
        result = result * s
    if o is not None and o != 0:
        result = result + o
    result.attrs = {
        k: v for k, v in attrs.items() if k not in (*_SCALE_KEYS, *_OFFSET_KEYS)
    }
    if nodata_to_nan:
        result.attrs.pop("nodata", None)
    return result


def normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """``(a - b) / (a + b)`` as float, NaN where the denominator is zero."""
    a_f = a.astype("float32") if not np.issubdtype(a.dtype, np.floating) else a
    b_f = b.astype("float32") if not np.issubdtype(b.dtype, np.floating) else b
    denominator = a_f + b_f
    return ((a_f - b_f) / denominator.where(denominator != 0)).rename(None)


def _band(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        raise KeyError(
            f"Band {name!r} not in dataset; available: {list(ds.data_vars)}."
        )
    return ds[name]


def ndsi(ds: xr.Dataset, green: str = "green", swir: str = "swir16") -> xr.DataArray:
    """Normalized Difference Snow Index ``(green - swir16) / (green + swir16)``."""
    return normalized_difference(_band(ds, green), _band(ds, swir)).assign_attrs(
        long_name="NDSI"
    )


def ndvi(ds: xr.Dataset, nir: str = "nir", red: str = "red") -> xr.DataArray:
    """Normalized Difference Vegetation Index ``(nir - red) / (nir + red)``."""
    return normalized_difference(_band(ds, nir), _band(ds, red)).assign_attrs(
        long_name="NDVI"
    )


def ndwi(ds: xr.Dataset, green: str = "green", nir: str = "nir") -> xr.DataArray:
    """Normalized Difference Water Index (McFeeters) ``(green - nir) / (green + nir)``."""
    return normalized_difference(_band(ds, green), _band(ds, nir)).assign_attrs(
        long_name="NDWI"
    )


def ndbi(ds: xr.Dataset, nir: str = "nir", swir: str = "swir22") -> xr.DataArray:
    """Normalized Difference Built-up Index as used here: ``(nir - swir22) / (nir + swir22)``."""
    return normalized_difference(_band(ds, nir), _band(ds, swir)).assign_attrs(
        long_name="NDBI"
    )


def evi(
    ds: xr.Dataset, nir: str = "nir", red: str = "red", blue: str = "blue"
) -> xr.DataArray:
    """Enhanced Vegetation Index ``2.5 (nir - red) / (nir + 6 red - 7.5 blue + 1)`` (reflectance 0–1)."""
    n, r, b = (_band(ds, k).astype("float32") for k in (nir, red, blue))
    return (
        (2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1))
        .rename(None)
        .assign_attrs(long_name="EVI")
    )


def rgb(
    ds: xr.Dataset,
    bands: Sequence[str] = ("red", "green", "blue"),
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    dim: str = "band",
) -> xr.DataArray:
    """Stack three bands into a ``band``-dimensioned float composite.

    With *vmin*/*vmax* the values are linearly scaled to 0–1 and clipped.
    """
    if len(bands) != 3:
        raise ValueError("rgb() needs exactly three band names.")
    da = xr.concat(
        [_band(ds, b).astype("float32") for b in bands],
        dim=pd.Index(list(bands), name=dim),
    )
    if vmin is not None or vmax is not None:
        lo = 0.0 if vmin is None else float(vmin)
        hi = float(vmax) if vmax is not None else float(da.max())
        da = ((da - lo) / (hi - lo)).clip(0, 1)
    return da.transpose(dim, ...)


def stretch_percentile(
    composite: xr.DataArray,
    lower: float = 2,
    upper: float = 98,
    *,
    dim: str = "band",
) -> xr.DataArray:
    """Percentile contrast stretch per band, clipped to 0–1.

    Percentiles are computed over all non-band dimensions (Dask arrays are
    rechunked along them). Bands with no spread stay 0.
    """
    other = [d for d in composite.dims if d != dim]
    data = (
        composite.chunk({d: -1 for d in other})
        if composite.chunks is not None
        else composite
    )
    q = data.quantile([lower / 100, upper / 100], dim=other, skipna=True)
    lo, hi = q.isel(quantile=0, drop=True), q.isel(quantile=1, drop=True)
    span = (hi - lo).where(hi > lo)
    return (
        ((composite - lo) / span)
        .clip(0, 1)
        .fillna(0)
        .transpose(*composite.dims)
        .assign_attrs(composite.attrs)
    )


def stretch_clahe(
    composite: xr.DataArray,
    *,
    clip_limit: float = 0.03,
    nbins: int = 256,
    kernel_size: int | None = None,
    dim: str = "band",
) -> xr.DataArray:
    """Contrast Limited Adaptive Histogram Equalization per band (scikit-image).

    Input should be 0–1 floats (see :func:`rgb`); NaNs are filled with 0 for
    the equalisation and restored afterwards. Computes eagerly.
    """
    from skimage import exposure  # noqa: PLC0415

    values = np.asarray(composite.transpose(dim, ...).values, dtype="float64")
    nan_mask = np.isnan(values)
    filled = np.clip(np.nan_to_num(values, nan=0.0), 0, 1)
    out = np.empty_like(filled)
    for i in range(filled.shape[0]):
        out[i] = exposure.equalize_adapthist(
            filled[i], clip_limit=clip_limit, nbins=nbins, kernel_size=kernel_size
        )
    out[nan_mask] = np.nan
    result = xr.DataArray(
        out.astype("float32"),
        dims=composite.transpose(dim, ...).dims,
        coords=composite.transpose(dim, ...).coords,
        attrs=composite.attrs,
    )
    return result.transpose(*composite.dims)
