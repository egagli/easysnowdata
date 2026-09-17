"""VIIRS snow cover: VNP10A1 and the cloud-gap-filled VNP10A1F (375 m).

New in the rewrite and Eric's first priority (plan §12 Q11): VIIRS is the
successor to MODIS snow cover as Terra winds down, at 375 m instead of 500 m
and with the same NDSI byte convention, so
:func:`easysnowdata.processing.binary_snow` works on both.

One source: NSIDC via ``earthaccess`` (``NSIDC_CPRD``, cloud-hosted). The
granules are HDF-EOS5 (``.h5``), which — unlike the HDF4 MODIS granules —
GDAL and h5netcdf can both read, so they are opened through the HDF5 driver
after a download to the cache. The grid comes from ``StructMetadata.0``,
which carries the sinusoidal corner coordinates.

::

    import easysnowdata as esd
    granules = esd.snow.viirs.search(aoi, "2023-03", product="VNP10A1F")
    snow = esd.snow.viirs.load(aoi, "2023-03", product="VNP10A1F")
    binary = esd.processing.binary_snow(snow["CGF_NDSI_Snow_Cover"], product="VNP10A1F")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract
from easysnowdata.processing import snow as snow_processing

__all__ = [
    "PRODUCT",
    "PRODUCTS",
    "DEFAULT_VARIABLES",
    "GRID_GROUP",
    "SINUSOIDAL",
    "search",
    "load",
    "read_struct_metadata",
]

_logger = logging.getLogger(__name__)

#: Supported VIIRS snow products → their main data field.
PRODUCTS: dict[str, str] = {
    "VNP10A1": "NDSI_Snow_Cover",
    "VNP10A1F": "CGF_NDSI_Snow_Cover",
    "VJ110A1": "NDSI_Snow_Cover",  # NOAA-20 sibling
    "VJ110A1F": "CGF_NDSI_Snow_Cover",
}
DEFAULT_VARIABLES = {name: (field,) for name, field in PRODUCTS.items()}
#: Where the data fields live inside the HDF-EOS5 file.
GRID_GROUP = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields"
#: The MODIS/VIIRS sinusoidal grid (a sphere of radius 6371007.181 m).
SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"

VIIRS_VERSION = "2"
_NSIDC_DOCS = "https://nsidc.org/data/vnp10a1f/versions/2"
_GRANULE_DATE = re.compile(r"\.A(\d{4})(\d{3})\.")
_STRUCT_UL = re.compile(r"UpperLeftPointMtrs=\(([-0-9.eE+]+),([-0-9.eE+]+)\)")
_STRUCT_LR = re.compile(r"LowerRightMtrs=\(([-0-9.eE+]+),([-0-9.eE+]+)\)")
_STRUCT_XDIM = re.compile(r"XDim=(\d+)")
_STRUCT_YDIM = re.compile(r"YDim=(\d+)")

_NDSI_FLAGS = {
    "flag_values": tuple(sorted(snow_processing.NDSI_FLAGS)),
    "flag_meanings": tuple(
        snow_processing.NDSI_FLAGS[v][0] for v in sorted(snow_processing.NDSI_FLAGS)
    ),
    "flag_colors": tuple(
        snow_processing.NDSI_FLAGS[v][1] for v in sorted(snow_processing.NDSI_FLAGS)
    ),
}

PRODUCT = Product(
    id="viirs-snow",
    theme="snow",
    title="VIIRS snow cover (VNP10A1, VNP10A1F)",
    description=(
        "Suomi-NPP VIIRS daily NDSI snow cover and the cloud-gap-filled daily "
        "product at 375 m, version 2, with the NOAA-20 (VJ1) siblings. The "
        "successor to the MODIS snow products as Terra winds down."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="earthdata",
            location="VNP10A1 / VNP10A1F (and the VJ1 NOAA-20 siblings)",
            requires=("earthdata",),
            resolution_m=375,
            temporal="2012-01/present",
            latency="~1 day",
            notes=(
                "HDF-EOS5 granules, cloud-hosted at NSIDC_CPRD; read through the HDF5 "
                "driver with the grid taken from StructMetadata.0"
            ),
            title="NSIDC (Earthdata)",
            health=Probe(
                "VIIRS snow cover VNP10A1F (NASA NSIDC)",
                partial(
                    health.earthdata_search,
                    "VNP10A1F",
                    temporal=("2023-01-01", "2023-01-07"),
                ),
            ),
        ),
    ),
    variables=(
        Variable(
            "NDSI_Snow_Cover",
            units="%",
            dtype="uint8",
            nodata=255,
            long_name="NDSI snow cover",
            **_NDSI_FLAGS,
        ),
        Variable(
            "CGF_NDSI_Snow_Cover",
            units="%",
            dtype="uint8",
            nodata=255,
            long_name="cloud-gap-filled NDSI snow cover",
            **_NDSI_FLAGS,
        ),
        Variable(
            "Cloud_Persistence",
            units="days",
            dtype="uint8",
            nodata=255,
            long_name="consecutive days of cloud cover behind the gap-filled value",
        ),
        Variable(
            "Basic_QA",
            dtype="uint8",
            nodata=255,
            long_name="basic quality assessment",
            flag_values=(0, 1, 2, 3, 211, 239, 255),
            flag_meanings=("best", "good", "ok", "poor", "night", "ocean", "unusable"),
            flag_colors=(
                "#1a9641",
                "#a6d96a",
                "#ffffbf",
                "#fdae61",
                "#000000",
                "#1f4e79",
                "#ffffff",
            ),
        ),
    ),
    citation=(
        "Riggs, G. A., Hall, D. K. and Román, M. O. (2019). VIIRS/NPP Snow Cover Daily "
        "L3 Global 375m SIN Grid, Version 2 (VNP10A1); Cloud Gap Filled (VNP10A1F). "
        "NASA NSIDC DAAC."
    ),
    license="NASA Earthdata (free registration)",
    doi="10.5067/VIIRS/VNP10A1F.002",
    references=(_NSIDC_DOCS,),
    loader="easysnowdata.snow.viirs.load",
    examples=("snow/plot_viirs_snow.py",),
    tags=("snow cover", "ndsi", "viirs", "suomi-npp"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _product(name: str) -> str:
    key = str(name).upper()
    if key not in PRODUCTS:
        raise ValueError(f"product must be one of {list(PRODUCTS)}, got {name!r}.")
    return key


def _variables(product: str, variables: str | Sequence[str] | None) -> list[str]:
    if variables is None:
        return list(DEFAULT_VARIABLES[product])
    return [variables] if isinstance(variables, str) else list(variables)


def granule_date(name: str) -> pd.Timestamp | None:
    """The acquisition date encoded in a granule name (``.AYYYYDDD.``)."""
    match = _GRANULE_DATE.search(str(name))
    if not match:
        return None
    year, doy = int(match.group(1)), int(match.group(2))
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def read_struct_metadata(path: Path | str) -> dict[str, float]:
    """Corner coordinates and shape from an HDF-EOS5 ``StructMetadata.0``.

    Returns ``upper_left_x``, ``upper_left_y``, ``lower_right_x``,
    ``lower_right_y``, ``columns`` and ``rows`` — enough to build the
    sinusoidal grid the granule sits on, which the file otherwise does not
    carry as CF coordinates.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as handle:
        raw = handle["HDFEOS INFORMATION/StructMetadata.0"][()]
    text = raw.decode() if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)
    upper_left = _STRUCT_UL.search(text)
    lower_right = _STRUCT_LR.search(text)
    columns = _STRUCT_XDIM.search(text)
    rows = _STRUCT_YDIM.search(text)
    if not (upper_left and lower_right and columns and rows):
        raise ValueError(f"{path}: StructMetadata.0 has no grid definition.")
    return {
        "upper_left_x": float(upper_left.group(1)),
        "upper_left_y": float(upper_left.group(2)),
        "lower_right_x": float(lower_right.group(1)),
        "lower_right_y": float(lower_right.group(2)),
        "columns": int(columns.group(1)),
        "rows": int(rows.group(1)),
    }


def _georeference(ds: xr.Dataset, grid: dict[str, float]) -> xr.Dataset:
    """Attach sinusoidal coordinates to a granule opened without them."""
    columns, rows = int(grid["columns"]), int(grid["rows"])
    res_x = (grid["lower_right_x"] - grid["upper_left_x"]) / columns
    res_y = (grid["lower_right_y"] - grid["upper_left_y"]) / rows
    x = grid["upper_left_x"] + (np.arange(columns) + 0.5) * res_x
    y = grid["upper_left_y"] + (np.arange(rows) + 0.5) * res_y
    dims = [d for d in ds.dims if d not in ("x", "y")]
    if len(dims) == 2:  # the phony dims h5netcdf invents
        ds = ds.rename({dims[0]: "y", dims[1]: "x"})
    ds = ds.assign_coords(x=x, y=y)
    return ds.rio.write_crs(SINUSOIDAL)


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None,
    time: Any = None,
    *,
    product: str = "VNP10A1F",
    source: str | None = None,
    max_items: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Search for VIIRS snow granules and return them as a ``GeoDataFrame``."""
    src = resolve_source(PRODUCT, source)
    name = _product(product)
    ensure_source(PRODUCT, src)
    granules = providers.earthdata.search(
        name, aoi, time, version=VIIRS_VERSION, **kwargs
    )
    if max_items is not None:
        granules = granules[:max_items]
    gdf = providers.earthdata.granules_to_geodataframe(granules)
    if len(gdf):
        gdf["date"] = [granule_date(i) for i in gdf["id"]]
    gdf.attrs = {"source": src.id, "product": name}
    return gdf


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    product: str = "VNP10A1F",
    variables: str | Sequence[str] | None = None,
    source: str | None = None,
    granules: Any = None,
    mask: bool = False,
    chunks: Any = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Load VIIRS snow cover as a lazy ``xarray.Dataset`` (``time``, ``y``, ``x``).

    Parameters
    ----------
    aoi, time
        The usual spatial and temporal inputs.
    product
        ``"VNP10A1F"`` (cloud-gap-filled, default), ``"VNP10A1"``, or a
        NOAA-20 sibling (``"VJ110A1"``, ``"VJ110A1F"``).
    variables
        Data fields to read; the product's main field by default. The others
        are ``Cloud_Persistence``, ``Basic_QA`` and
        ``Algorithm_Bit_Flags_QA``.
    mask
        ``True`` NaN-masks the sentinel values; the default keeps the raw
        byte with its CF flags (see
        :func:`easysnowdata.processing.binary_snow`).
    """
    src = resolve_source(PRODUCT, source)
    name = _product(product)
    fields = _variables(name, variables)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    if granules is None:
        granules = providers.earthdata.search(name, aoi, time, version=VIIRS_VERSION)
    elif isinstance(granules, gpd.GeoDataFrame):
        raise TypeError(
            "Pass the earthaccess granules themselves (what providers.earthdata.search "
            "returns), not the GeoDataFrame from snow.viirs.search()."
        )
    if not len(granules):
        raise ValueError(f"No {name} granules at NSIDC for this AOI and time.")
    paths = providers.earthdata.download(granules, name)

    per_date: dict[pd.Timestamp, list[xr.Dataset]] = {}
    for path in paths:
        when = granule_date(Path(path).name)
        if when is None:  # pragma: no cover — defensive
            _logger.warning("Skipping %s: no date in the granule name.", path)
            continue
        granule = _open_granule(path, fields, chunks, parsed, **kwargs)
        per_date.setdefault(when, []).append(granule)

    datasets = []
    for when in sorted(per_date):
        tiles = per_date[when]
        merged = tiles[0] if len(tiles) == 1 else _merge_tiles(tiles)
        datasets.append(merged.expand_dims(time=[when]))
    combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    combined = combined.sortby("time")

    combined = contract.apply_variables(
        combined,
        [v for v in PRODUCT.variables if v.name in combined.data_vars],
        mask=True if mask else False,
    )
    return contract.finalize(
        combined,
        PRODUCT,
        src,
        variables=(),
        source_url=_NSIDC_DOCS,
        attrs={"viirs_product": name, "version": VIIRS_VERSION},
    )


def _open_granule(
    path: Path, fields: Sequence[str], chunks: Any, parsed: Any, **kwargs: Any
) -> xr.Dataset:
    """Open one HDF-EOS5 granule and georeference it from ``StructMetadata.0``."""
    ds = xr.open_dataset(
        str(path),
        engine="h5netcdf",
        group=GRID_GROUP,
        phony_dims="sort",
        chunks={} if chunks is True else chunks,
        mask_and_scale=False,
        **kwargs,
    )
    missing = [f for f in fields if f not in ds.data_vars]
    if missing:
        raise KeyError(
            f"{Path(path).name} has no {missing}; it holds {list(ds.data_vars)}."
        )
    ds = ds[list(fields)]
    ds = _georeference(ds, read_struct_metadata(path))
    if parsed is not None and parsed.clip and not parsed.is_global:
        bounds = parsed.to_crs(ds.rio.crs).total_bounds
        ds = ds.rio.clip_box(*bounds, crs=ds.rio.crs, auto_expand=True)
    return ds


def _merge_tiles(tiles: Sequence[xr.Dataset]) -> xr.Dataset:
    """Mosaic same-date VIIRS tiles onto one grid."""
    from rioxarray.merge import merge_arrays  # noqa: PLC0415

    return xr.Dataset(
        {
            field: merge_arrays([tile[field] for tile in tiles])
            for field in tiles[0].data_vars
        }
    )
