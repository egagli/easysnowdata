"""Snow-product processing: class tables and the SNODAS flat-file reader.

Pure functions (design contract §2.6). The class tables are the MODIS and
VIIRS sentinel values that share a byte with the NDSI percentage. Thresholding
that byte is left to the user, in the open, because the choice of threshold
and of what to do with cloud, night and water is the analysis::

    ndsi_da = snow_ds["CGF_NDSI_Snow_Cover"]
    valid_da = ndsi_da <= 100                       # 200+ are the sentinels below
    binary_da = (ndsi_da >= 40).where(valid_da)     # NaN where the pixel says nothing
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "repair_fill_values",
    "NDSI_FLAGS",
    "MOD10A2_CLASSES",
    "ndsi_flag_attrs",
    "parse_snodas_header",
    "snodas_members",
    "snodas_array",
    "SNODAS_PRODUCTS",
]

#: Sentinel values that share the byte with the NDSI percentage (0–100) in the
#: MODIS ``MOD10A1``/``MOD10A1F`` and VIIRS ``VNP10A1``/``VNP10A1F`` products.
NDSI_FLAGS: dict[int, tuple[str, str]] = {
    200: ("missing_data", "#4d4d4d"),
    201: ("no_decision", "#8c8c8c"),
    211: ("night", "#000000"),
    237: ("inland_water", "#2b83ba"),
    239: ("ocean", "#1f4e79"),
    250: ("cloud", "#d9d9d9"),
    251: ("cloud", "#d9d9d9"),
    252: ("cloud", "#d9d9d9"),
    254: ("detector_saturated", "#fdae61"),
    255: ("fill", "#ffffff"),
}
#: The 8-day maximum-snow-extent classes of ``MOD10A2`` / ``MYD10A2``.
MOD10A2_CLASSES: dict[int, tuple[str, str]] = {
    0: ("missing_data", "#4d4d4d"),
    1: ("no_decision", "#8c8c8c"),
    11: ("night", "#000000"),
    25: ("no_snow", "#a6611a"),
    37: ("lake", "#2b83ba"),
    39: ("ocean", "#1f4e79"),
    50: ("cloud", "#d9d9d9"),
    100: ("lake_ice", "#92c5de"),
    200: ("snow", "#2166ac"),
    254: ("detector_saturated", "#fdae61"),
    255: ("fill", "#ffffff"),
}


def ndsi_flag_attrs(long_name: str = "NDSI snow cover") -> dict[str, Any]:
    """CF attributes for an NDSI byte: percentages plus the sentinel meanings."""
    values = sorted(NDSI_FLAGS)
    seen: dict[int, str] = {}
    for value in values:
        seen[value] = NDSI_FLAGS[value][0]
    return {
        "long_name": long_name,
        "units": "%",
        "valid_range": [0, 100],
        "flag_values": list(seen),
        "flag_meanings": " ".join(seen.values()),
        "flag_colors": " ".join(NDSI_FLAGS[v][1] for v in seen),
        "description": (
            "0–100 is percent NDSI snow cover; values above 100 are the sentinel "
            "meanings in flag_values/flag_meanings"
        ),
    }


# ── SNODAS (NSIDC G02158) ─────────────────────────────────────────────────────

#: SNODAS product code → (variable name, units, description). The code is the
#: ``ssmv1`` field of the file name; ``11034`` is SWE and ``11036`` depth.
SNODAS_PRODUCTS: dict[str, tuple[str, str, str]] = {
    "1034": ("SWE", "m", "snow water equivalent, total of snow layers"),
    "1036": ("snow_depth", "m", "snow layer thickness, total of snow layers"),
    "1038": ("snowpack_average_temperature", "K", "snowpack average temperature"),
    "1039": ("blowing_snow_sublimation", "m", "sublimation of blowing snow"),
    "1044": ("snow_melt", "m", "snow melt runoff at the base of the snowpack"),
    "1050": ("snowpack_sublimation", "m", "sublimation from the snowpack"),
    "0025SlL01": ("snowfall", "kg m-2", "solid precipitation"),
    "0025SlL00": ("rainfall", "kg m-2", "liquid precipitation"),
}

_HEADER_NUMBERS = (
    "Number of columns",
    "Number of rows",
    "Data bytes per pixel",
    "No data value",
    "Minimum x-axis coordinate",
    "Maximum x-axis coordinate",
    "Minimum y-axis coordinate",
    "Maximum y-axis coordinate",
    "X-axis resolution",
    "Y-axis resolution",
    "Benchmark x-axis coordinate",
    "Benchmark y-axis coordinate",
)


def parse_snodas_header(text: str) -> dict[str, Any]:
    """Parse a NOHRSC ``.txt`` header into a dict.

    Keys are lower-cased with underscores (``number_of_columns``,
    ``minimum_x_axis_coordinate``…); the numeric fields are floats or ints.
    The scaling factor hidden in ``Data units`` (``"Meters / 1000.000000"``)
    comes back as ``scale_factor``.
    """
    header: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        target = key.lower().replace(" ", "_").replace("-", "_")
        if key in _HEADER_NUMBERS:
            number = float(value)
            header[target] = int(number) if number.is_integer() else number
        else:
            header[target] = value
    units = header.get("data_units", "")
    match = re.search(r"/\s*([0-9.]+)", units)
    header["scale_factor"] = 1.0 / float(match.group(1)) if match else 1.0
    header["units"] = units.split("/")[0].strip().lower() if units else ""
    if header["units"] == "meters":
        header["units"] = "m"
    return header


def snodas_members(tar_bytes: bytes | io.BufferedIOBase) -> dict[str, dict[str, bytes]]:
    """Group a SNODAS day's tar into ``{stem: {"dat": ..., "txt": ...}}``.

    The members are gzipped inside the tar; both are decompressed here so the
    caller only deals with bytes.
    """
    handle = (
        io.BytesIO(tar_bytes)
        if isinstance(tar_bytes, (bytes, bytearray))
        else tar_bytes
    )
    out: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=handle, mode="r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            if name.endswith(".dat.gz"):
                stem, kind = name[: -len(".dat.gz")], "dat"
            elif name.endswith(".txt.gz"):
                stem, kind = name[: -len(".txt.gz")], "txt"
            elif name.endswith(".dat"):
                stem, kind = name[: -len(".dat")], "dat"
            elif name.endswith(".txt"):
                stem, kind = name[: -len(".txt")], "txt"
            else:
                continue
            payload = tar.extractfile(member).read()
            if name.endswith(".gz"):
                payload = gzip.decompress(payload)
            out.setdefault(stem, {})[kind] = payload
    return out


def snodas_array(
    data: bytes, header: Mapping[str, Any], *, name: str | None = None
) -> xr.DataArray:
    """Build a georeferenced DataArray from one SNODAS ``.dat`` and its header.

    The flat file is big-endian 16-bit integers in row-major order on the
    header's geographic grid; the nodata sentinel is masked and the scale
    factor from ``Data units`` applied, so the result is metres (or the
    product's own unit) as floats.
    """
    rows = int(header["number_of_rows"])
    columns = int(header["number_of_columns"])
    dtype = ">i2" if int(header.get("data_bytes_per_pixel", 2)) == 2 else ">i4"
    values = np.frombuffer(data, dtype=dtype)
    if values.size != rows * columns:
        raise ValueError(
            f"SNODAS payload has {values.size} values, expected {rows * columns} "
            f"({rows} rows × {columns} columns)."
        )
    grid = values.reshape(rows, columns).astype("float32")
    nodata = float(header.get("no_data_value", -9999))
    grid = np.where(grid == nodata, np.nan, grid) * float(
        header.get("scale_factor", 1.0)
    )

    res_x = float(header["x_axis_resolution"])
    res_y = float(header["y_axis_resolution"])
    west = float(header["minimum_x_axis_coordinate"])
    north = float(header["maximum_y_axis_coordinate"])
    longitude = west + (np.arange(columns) + 0.5) * res_x
    latitude = north - (np.arange(rows) + 0.5) * res_y
    da = xr.DataArray(
        grid,
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name=name,
    )
    da.attrs = {
        "long_name": header.get("description", ""),
        "units": header.get("units", ""),
        "scale_factor_applied": float(header.get("scale_factor", 1.0)),
        "_FillValue_source": nodata,
    }
    return da.rio.write_crs("EPSG:4326")


def repair_fill_values(
    da: xr.DataArray, *, last_class: int = 3, fill: int = 255
) -> xr.DataArray:
    """Put the upstream 256/265 fill values back to 255 and return uint8.

    The published rasters use nodata 256 (and 265 in places), which does not
    fit in a byte, so GDAL widens the whole array to uint32. Everything above
    *last_class* is fill, so it is rewritten to *fill*.
    """
    return da.where(da <= last_class, fill).astype("uint8")
