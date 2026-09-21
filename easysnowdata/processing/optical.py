"""Optical processing: Sentinel-2 baseline harmonization, metadata-driven
scale/offset, and the PlanetScope UDM2 mask decoder. Pure functions.

Band arithmetic is deliberately *not* wrapped here. A normalized difference is
one line of xarray, and hiding it behind ``ndsi(ds)`` obscured which bands
were being used and what happened to negative reflectance::

    ndsi = (s2["green"] - s2["swir16"]) / (s2["green"] + s2["swir16"])
    rgb = s2[["red", "green", "blue"]].to_array("band").clip(0, 0.3) / 0.3
    rgb.isel(time=0).plot.imshow(rgb="band")

The gallery examples spell these out every time they are used.
"""

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
    "UDM2_BANDS",
    "UDM2_BINARY_BANDS",
    "UDM1_BITS",
    "decode_udm2",
    "udm1_bit",
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


#: PlanetScope UDM2 band order (Planet's "Usable Data Mask" specification).
#: Bands 1-6 are 0/1 masks, band 7 is a percentage confidence and band 8 is
#: the legacy UDM1 bit field (bit 0 = blackfill / no data).
UDM2_BANDS: tuple[str, ...] = (
    "clear",
    "snow",
    "shadow",
    "light_haze",
    "heavy_haze",
    "cloud",
    "confidence",
    "unusable",
)
UDM2_BINARY_BANDS = UDM2_BANDS[:6]
#: Bit meanings of the UDM1 field carried in band 8.
UDM1_BITS: dict[str, int] = {
    "blackfill": 0,
    "cloud_udm1": 1,
    "missing_blue": 2,
    "missing_green": 3,
    "missing_red": 4,
    "missing_rededge": 5,
    "missing_nir": 6,
}


def decode_udm2(
    udm2: xr.DataArray | xr.Dataset, *, band_dim: str = "band"
) -> xr.Dataset:
    """Decode a PlanetScope UDM2 mask into named variables.

    Parameters
    ----------
    udm2
        The 8-band UDM2 raster as a DataArray with a band dimension (values
        1–8 or 0-based), or a Dataset whose variables are already named
        ``band_1`` … ``band_8``.
    band_dim
        Name of the band dimension on a DataArray input.

    Returns
    -------
    xarray.Dataset
        ``clear``, ``snow``, ``shadow``, ``light_haze``, ``heavy_haze`` and
        ``cloud`` as ``uint8`` 0/1 masks with CF flag attributes,
        ``confidence`` as a percentage, and ``unusable`` as the raw UDM1 bit
        field. The snow band is the one that is directly useful here.

    Notes
    -----
    Band order follows Planet's UDM2 specification; the file itself carries no
    band names, so a raster with fewer than eight bands raises.
    """
    if isinstance(udm2, xr.Dataset):
        layers = [udm2[name] for name in list(udm2.data_vars)[: len(UDM2_BANDS)]]
    else:
        if band_dim not in udm2.dims:
            raise ValueError(
                f"{band_dim!r} is not a dimension of the UDM2 raster; pass band_dim=."
            )
        layers = [
            udm2.isel({band_dim: i}, drop=True) for i in range(udm2.sizes[band_dim])
        ]
    if len(layers) < len(UDM2_BANDS):
        raise ValueError(
            f"A UDM2 raster has {len(UDM2_BANDS)} bands ({', '.join(UDM2_BANDS)}); "
            f"got {len(layers)}."
        )
    out = {}
    for name, layer in zip(UDM2_BANDS, layers, strict=False):
        layer = layer.rename(name)
        if name in UDM2_BINARY_BANDS:
            layer = layer.astype("uint8")
            layer.attrs = {
                "long_name": f"UDM2 {name.replace('_', ' ')} mask",
                "flag_values": [0, 1],
                "flag_meanings": f"not_{name} {name}",
                "flag_colors": "#00000000 #1f78b4",
            }
        elif name == "confidence":
            layer.attrs = {"long_name": "UDM2 classification confidence", "units": "%"}
        else:
            layer.attrs = {
                "long_name": "UDM1 unusable-data bit field",
                "bit_meanings": " ".join(UDM1_BITS),
            }
        out[name] = layer
    dataset = xr.Dataset(out)
    dataset.attrs["udm2_band_order"] = " ".join(UDM2_BANDS)
    return dataset


def udm1_bit(unusable: xr.DataArray, flag: str) -> xr.DataArray:
    """The value (0/1) of one UDM1 bit (``"blackfill"``, ``"cloud_udm1"``, …)."""
    if flag not in UDM1_BITS:
        raise ValueError(f"Unknown UDM1 flag {flag!r}; known: {list(UDM1_BITS)}.")
    return (unusable.astype("uint8") >> UDM1_BITS[flag]) & 1
