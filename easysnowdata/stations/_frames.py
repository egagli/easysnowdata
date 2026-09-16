"""Turn the clients' dict records into the package's return types. No I/O.

Two conversions live here:

* :func:`stations_to_geodataframe` — a client's ``get_all_stations()`` list
  into the EPSG:4326 ``GeoDataFrame`` :func:`easysnowdata.stations.inventory`
  returns, indexed by the globally unique station code;
* :func:`records_to_dataset` — a client's ``get_data()`` list into the
  ``(station, time)`` ``Dataset`` :func:`easysnowdata.stations.load` returns,
  one data variable per standardized type, station metadata as non-dimension
  coordinates and water-year coordinates on ``time``.

The station × time grid is sparse in the everyday sense — most stations do
not span most of the record — and it is materialized densely with NaN, which
is what xarray, Dask and every downstream ``groupby`` want. Nothing here
needs a sparse array backend, and an ``xvec`` geometry coordinate is
deliberately not used (see the module docstring of
:mod:`easysnowdata.stations`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from easysnowdata.processing import wateryear
from easysnowdata.stations import networks

__all__ = [
    "INVENTORY_COLUMNS",
    "records_to_dataset",
    "records_to_frame",
    "stations_to_geodataframe",
]

_logger = logging.getLogger(__name__)

#: Columns every inventory carries, in order. Client-specific extras follow.
INVENTORY_COLUMNS = (
    "name",
    "network",
    "network_code",
    "latitude",
    "longitude",
    "elevation_m",
    "state",
    "status",
    "is_active",
    "operator",
    "data_provider",
    "station_url",
)

#: Station-dict keys that are already the inventory's own columns.
_SKIP_EXTRAS = frozenset(INVENTORY_COLUMNS) | {"code", "station_id", "geometry"}


def _first(station: dict, *keys: str) -> Any:
    for key in keys:
        value = station.get(key)
        if value not in (None, ""):
            return value
    return None


def stations_to_geodataframe(
    stations: Sequence[dict],
    network: str,
    *,
    extras: bool = True,
) -> gpd.GeoDataFrame:
    """One client's station dicts as an EPSG:4326 frame indexed by code.

    Parameters
    ----------
    stations
        What ``client.get_all_stations()`` returned.
    network
        Which network they came from (``"awdb"``, ``"cdec"``, …).
    extras
        Keep the source-specific keys alongside the standard columns.
    """
    net = networks.get(network)
    rows: list[dict[str, Any]] = []
    for station in stations:
        station_id = _first(station, net.id_key, "station_id")
        if station_id is None:
            _logger.debug("Dropping a %s station with no id: %r", network, station)
            continue
        status = station.get("status")
        row: dict[str, Any] = {
            "code": net.code(str(station_id)),
            "station_id": str(station_id),
            "name": station.get("name") or "",
            "network": network,
            "network_code": _first(station, "networkCode", "network_code"),
            "latitude": _as_float(station.get("latitude")),
            "longitude": _as_float(station.get("longitude")),
            "elevation_m": _as_float(station.get("elevation_m")),
            "state": _first(station, "stateCode", "state"),
            "status": status,
            "is_active": None if status is None else str(status).lower() == "active",
            "operator": station.get("operator"),
            "data_provider": station.get("data_provider") or net.data_provider,
            "station_url": station.get("station_url") or None,
        }
        if extras:
            for key, value in station.items():
                if key in _SKIP_EXTRAS or key in row:
                    continue
                row[key] = value if _is_scalar(value) else str(value)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["code", "station_id", *INVENTORY_COLUMNS])
    if rows:
        frame = pd.DataFrame(rows)
        ordered = ["code", "station_id", *INVENTORY_COLUMNS]
        frame = frame[[*ordered, *[c for c in frame.columns if c not in ordered]]]
    geometry = gpd.points_from_xy(
        frame.get("longitude", pd.Series(dtype="float64")),
        frame.get("latitude", pd.Series(dtype="float64")),
    )
    gdf = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
    return gdf.set_index("code")


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def records_to_frame(records: Iterable[dict], network: str) -> pd.DataFrame:
    """The flat records as a tidy DataFrame with a ``code`` and a ``time`` column.

    Sub-daily records carry ``datetime``; daily ones only ``date``. Both
    become one ``time`` column, so a mixed-interval request still lines up.
    """
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return pd.DataFrame(
            columns=["code", "station_id", "time", "variable", "type", "value", "units"]
        )
    when = frame["datetime"] if "datetime" in frame.columns else frame["date"]
    if "datetime" in frame.columns:
        when = when.where(when.notna() & (when != ""), frame["date"])
    frame["time"] = pd.to_datetime(when, format="mixed", utc=False, errors="coerce")
    frame["code"] = [networks.to_code(str(sid), network) for sid in frame["station_id"]]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _harmonize_units(frame: pd.DataFrame, type_name: str) -> tuple[pd.DataFrame, str]:
    """Bring one type's values onto the canonical unit; returns (frame, units)."""
    target = networks.canonical_units(type_name)
    units = [u for u in frame["units"].dropna().unique().tolist() if u]
    if not units:
        return frame, target
    if not target:
        target = units[0]
    if set(units) == {target}:
        return frame, target
    out = frame.copy()
    for unit in units:
        if unit == target:
            continue
        factor = networks.unit_factor(unit, target)
        _logger.info(
            "Converting %s values from %s to %s (factor %g) so the variable "
            "carries one unit.",
            type_name,
            unit,
            target,
            factor,
        )
        rows = out["units"] == unit
        out.loc[rows, "value"] = out.loc[rows, "value"] * factor
    out["units"] = target
    return out, target


def records_to_dataset(
    records: Iterable[dict] | dict[str, Iterable[dict]],
    network: str | None = None,
    *,
    metadata: gpd.GeoDataFrame | None = None,
    hemisphere: str = "northern",
    stations: Sequence[str] | None = None,
) -> xr.Dataset:
    """The clients' records as a ``(station, time)`` Dataset.

    Parameters
    ----------
    records
        One network's records (with *network* naming it), or a
        ``{network_id: records}`` mapping to merge several networks.
    network
        Which network *records* came from, when it is a plain list.
    metadata
        Station frame (from :func:`easysnowdata.stations.inventory`) whose
        columns become non-dimension coordinates on ``station``.
    hemisphere
        Which water year to attach — ``"northern"`` (1 October) or
        ``"southern"`` (1 April).
    stations
        Codes to keep on the ``station`` axis even when they returned no data,
        so a request for five stations always comes back with five.

    Returns
    -------
    xarray.Dataset
        One data variable per standardized type (``swe``, ``snwd``, …) with
        dims ``(station, time)``, ``water_year`` and ``dowy`` coordinates on
        ``time``, and the station metadata on ``station``.
    """
    if isinstance(records, dict):
        by_network: dict[str, Any] = dict(records)
    else:
        if network is None:
            raise ValueError(
                "records_to_dataset() needs network= when records is a plain "
                "list; pass {network: records} to merge several networks."
            )
        by_network = {network: list(records)}
    frames = [records_to_frame(recs, net) for net, recs in by_network.items()]
    frames = [f for f in frames if not f.empty]
    tidy = (
        pd.concat(frames, ignore_index=True)
        if frames
        else records_to_frame([], next(iter(by_network), "awdb"))
    )

    codes = list(dict.fromkeys(stations or [])) or sorted(
        tidy["code"].unique().tolist()
    )
    data_vars: dict[str, Any] = {}
    var_attrs: dict[str, dict[str, Any]] = {}
    times = pd.DatetimeIndex(sorted(tidy["time"].dropna().unique()), name="time")
    for type_name, group in tidy.groupby("type", sort=True):
        group, units = _harmonize_units(group, str(type_name))
        # A station can serve several native variables of one type (CDEC's
        # SNO ADJ and raw SWE both mean `swe`); the client orders them by its
        # own preference, so the first non-null wins.
        pivot = group.pivot_table(
            index="code",
            columns="time",
            values="value",
            aggfunc="first",
            dropna=False,
        )
        pivot = pivot.reindex(index=codes, columns=times)
        data_vars[str(type_name)] = (("station", "time"), pivot.to_numpy("float64"))
        natives = sorted({str(v) for v in group["variable"].dropna().unique()})
        intervals = sorted({str(v) for v in group["interval"].dropna().unique()})
        var_attrs[str(type_name)] = {
            "units": units,
            "long_name": networks.TYPES.get(str(type_name), ("", str(type_name)))[1],
            "native_variables": " ".join(natives),
            "interval": " ".join(intervals),
        }

    ds = xr.Dataset(
        data_vars,
        coords={"station": np.array(codes, dtype="str"), "time": times},
    )
    for name, attrs in var_attrs.items():
        ds[name].attrs.update(attrs)
    ds = _attach_metadata(ds, metadata, by_network)
    if ds.sizes.get("time", 0):
        ds = wateryear.add_water_year_coords(ds, hemisphere)
    return ds


def _attach_metadata(
    ds: xr.Dataset,
    metadata: gpd.GeoDataFrame | None,
    by_network: dict[str, Any],
) -> xr.Dataset:
    """Station metadata as non-dimension coordinates on ``station``."""
    codes = [str(c) for c in ds["station"].values]
    if metadata is None or metadata.empty:
        # Even without an inventory, say which network each station came from.
        lookup = {
            networks.to_code(str(sid), net): net
            for net, recs in by_network.items()
            for sid in {r.get("station_id") for r in recs}
            if sid is not None
        }
        if lookup:
            ds = ds.assign_coords(
                network=("station", [lookup.get(c, "") for c in codes])
            )
        return ds
    frame = metadata.reindex(codes)
    for column in frame.columns:
        if column == "geometry":
            continue
        values = frame[column]
        if values.isna().all():
            continue
        # Nothing in a coordinate may be a Python object we cannot serialize:
        # numbers and booleans stay, everything else becomes a string.
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            data = values.to_numpy()
        else:
            data = np.array(
                [("" if pd.isna(v) else str(v)) for v in values], dtype="str"
            )
        ds = ds.assign_coords({column: ("station", data)})
    return ds
