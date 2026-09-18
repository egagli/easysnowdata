"""SNOTEL and CCSS station data — a shim over :mod:`easysnowdata.stations`.

.. deprecated:: 0.1.0
    Use :func:`easysnowdata.stations.inventory` and
    :func:`easysnowdata.stations.load` (or
    :func:`easysnowdata.stations.archive.load`). ``StationCollection`` is kept
    for one minor release and removed in 0.3.0.

The data no longer comes from the frozen ``egagli/snotel_ccss_stations``
archive: that route is gone, with no transition period (§12 Q9). Every call
below now reaches the live AWDB and CDEC clients, or the daily archive
``global_snow_networks`` publishes, through the Phase 3 adapter. What that
changes, and what it does not:

* **Station codes are unchanged.** ``679_WA_SNTL`` still means Paradise; the
  adapter maps it to the AWDB triplet ``679:WA:SNTL`` on the way out.
* **Variable names are unchanged.** ``WTEQ``, ``SNWD``, ``PRCPSA``, ``TAVG``,
  ``TMIN`` and ``TMAX`` still name the columns, mapped to the standardized
  types the networks speak (``swe``, ``snwd``, ``precip``, ``temp``,
  ``temp_min``, ``temp_max``).
* **Units are unchanged.** The old CSVs held metres and millimetres of water;
  the networks emit centimetres and millimetres. The shim converts back, so
  the numbers a notebook plots are the same ones it plotted before.
* **The station list is larger and current.** It comes from the live
  inventory rather than a 2024 snapshot, so stations added since then appear
  and ``endDate`` reflects today.
* **SNOTEL air temperature is a newer vintage, and that is a fix rather than
  a regression.** ``TAVG``, ``TMIN`` and ``TMAX`` map to the same AWDB
  elements the frozen archive used, but their *values* differ for roughly
  2004-2024 at the stations NRCS bias-corrected: the frozen CSVs hold the
  uncorrected series, about 1.1 °C warm, because that archive's updater only
  ever rewrote the last ten days and never re-fetched history. Verified
  against both AWDB REST and the CUAHSI WaterOneFlow service the frozen
  archive itself used, which agree with each other to 0.02 °C. At Paradise
  the offset spans 2004-2024 and is zero either side; 303:CO and 457:CO show
  the same shape; 642:WA shows none. Reading live means you get the corrected
  values, so a Paradise winter mean moves from -1.55 °C to -2.95 °C.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from easysnowdata import stations as _stations
from easysnowdata._deprecation import deprecated
from easysnowdata.temporal import today

__all__ = ["StationCollection"]

_logger = logging.getLogger(__name__)

_SINCE = "0.1.0"
_REMOVE_IN = "0.3.0"

#: The six names the old CSVs used -> the standardized type the networks use.
VARIABLE_TYPES: dict[str, str] = {
    "WTEQ": "swe",
    "SNWD": "snwd",
    "PRCPSA": "precip",
    "TAVG": "temp",
    "TMIN": "temp_min",
    "TMAX": "temp_max",
}

#: Multiply the adapter's value by this to get the old CSVs' unit.
#: SWE, snow depth and precipitation were metres of water / metres of snow;
#: the networks emit centimetres (swe, snwd) and millimetres (precip).
_TO_LEGACY_UNITS: dict[str, float] = {
    "WTEQ": 0.01,  # cm -> m
    "SNWD": 0.01,  # cm -> m
    "PRCPSA": 0.001,  # mm -> m
    "TAVG": 1.0,  # already °C
    "TMIN": 1.0,
    "TMAX": 1.0,
}

#: The networks this class has always meant by "SNOTEL and CCSS".
_NETWORKS = ("awdb", "cdec")

DEFAULT_VARIABLES = tuple(VARIABLE_TYPES)


def _legacy_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename the inventory's columns to the ones the old GeoJSON carried."""
    out = gdf.copy()
    if "network_code" in out.columns:
        # The old file said "SNOTEL" or "CCSS"; the inventory keeps the
        # network's own code (SNTL, MSNT, CCSS, SNOW, …).
        out["network"] = [
            "SNOTEL"
            if str(code).startswith("SNTL")
            else ("CCSS" if net == "cdec" else str(code))
            for code, net in zip(out["network_code"], out["network"], strict=True)
        ]
    renames = {"begin_date": "beginDate", "end_date": "endDate"}
    out = out.rename(columns={k: v for k, v in renames.items() if k in out.columns})
    if "daily_or_better" in out.columns:
        # `csvData` meant "this station has a data file in the archive", which
        # is what `daily_or_better` decides now.
        out["csvData"] = out["daily_or_better"].fillna(False).astype(bool)
    return out


class StationCollection:
    """A collection of SNOTEL and CCSS automatic weather stations.

    .. deprecated:: 0.1.0
        Use :func:`easysnowdata.stations.inventory` and
        :func:`easysnowdata.stations.load`. Removed in 0.2.0.

    Parameters
    ----------
    data_available : bool, optional
        If ``True`` (default), only include stations with a daily record —
        which is what "has a CSV file" meant in the old archive.
    sortby_dist_to_geom : GeoDataFrame or tuple or shapely geometry, optional
        If provided, stations are sorted by distance to this geometry and a
        ``dist_km`` column is added to ``all_stations``.
    **kwargs
        Passed to ``geopandas.read_file`` when the station inventory is read
        (e.g. ``rows=10``).

    Attributes
    ----------
    all_stations : geopandas.GeoDataFrame
        All stations matching the filter criteria, indexed by station code.
    stations : geopandas.GeoDataFrame or None
        The subset selected by the most recent :meth:`choose_stations` call.
    data : pandas.DataFrame or xarray.Dataset or None
        Data returned by the most recent :meth:`get_data` call.
    entire_data_archive : xarray.Dataset or None
        Full dataset returned by :meth:`get_entire_data_archive`.

    Examples
    --------
    >>> sc = StationCollection()                                  # doctest: +SKIP
    >>> sc.get_data(stations="679_WA_SNTL", variables=["WTEQ"],
    ...             start_date="2020-10-01", end_date="2021-09-30")  # doctest: +SKIP

    Notes
    -----
    Available variables: ``WTEQ`` (SWE), ``SNWD`` (snow depth),
    ``PRCPSA`` (accumulated precipitation), ``TAVG``, ``TMIN``, ``TMAX``.
    Values are in metres and °C, as they were in the frozen archive.
    """

    @deprecated(
        "easysnowdata.stations.inventory and easysnowdata.stations.load",
        since=_SINCE,
        remove_in=_REMOVE_IN,
        name="easysnowdata.automatic_weather_stations.StationCollection",
    )
    def __init__(
        self,
        data_available: bool = True,
        sortby_dist_to_geom: gpd.GeoDataFrame | tuple | None = None,
        **kwargs: Any,
    ) -> None:
        self.data_available = data_available
        self.sortby_dist_to_geom = sortby_dist_to_geom
        self.read_file_kwargs = kwargs

        self.all_stations: gpd.GeoDataFrame | None = None
        self.stations: gpd.GeoDataFrame | None = None
        self.data: pd.DataFrame | xr.Dataset | None = None
        self.entire_data_archive: xr.Dataset | None = None

        for variable in VARIABLE_TYPES:
            setattr(self, variable, None)

        self.get_all_stations()

    def get_all_stations(self) -> None:
        """Fetch station metadata and populate ``all_stations``.

        Returns
        -------
        None
            Sets ``self.all_stations``.
        """
        gdf = _stations.inventory(
            networks=_NETWORKS,
            daily_only=self.data_available,
            source="archive",
            **self.read_file_kwargs,
        )
        # The frozen file listed every SNOTEL station before every CCSS one;
        # the live inventory is sorted by name across all networks. Restore
        # the old grouping so code that indexed positionally still works.
        gdf = gdf.sort_values(["network", gdf.index.name or "code"], kind="stable")
        gdf = _legacy_columns(gdf)

        if self.sortby_dist_to_geom is not None:
            from easysnowdata.aoi import parse_aoi  # noqa: PLC0415

            _logger.info("Sorting stations by distance to provided geometry.")
            target = parse_aoi(self.sortby_dist_to_geom).footprint
            proj = "EPSG:32611"
            gdf["dist_km"] = (
                gdf.to_crs(proj).distance(target.to_crs(proj).geometry.iloc[0]) / 1000
            )
            gdf = gdf.sort_values("dist_km")

        self.all_stations = gdf
        _logger.info("Loaded %d stations into all_stations.", len(gdf))

    def choose_stations(self, stations_input: gpd.GeoDataFrame | str | list) -> None:
        """Select a subset of stations by code, list of codes, or GeoDataFrame.

        Returns
        -------
        None
            Sets ``self.stations``.
        """
        if isinstance(stations_input, str):
            self.stations = self.all_stations.loc[[stations_input]]
        elif isinstance(stations_input, list):
            self.stations = self.all_stations.loc[stations_input]
        else:
            self.stations = stations_input

    def get_data(
        self,
        stations: gpd.GeoDataFrame | str | list = "679_WA_SNTL",
        variables: str | list | None = None,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Fetch data for the given stations and variables.

        Dispatches to :meth:`get_single_station_data` or
        :meth:`get_multiple_station_data` based on how many stations are
        selected, exactly as before.

        Parameters
        ----------
        stations : str, list of str, or GeoDataFrame, optional
            Station code(s) to fetch. Default ``"679_WA_SNTL"`` (Paradise, WA).
        variables : str or list of str, optional
            Defaults to all six variables for a single station, or ``WTEQ``
            for several.
        start_date, end_date : str, optional
            ISO date strings. *end_date* defaults to today.
        **kwargs
            Accepted for compatibility. ``dtype`` is applied to the result;
            other ``pandas.read_csv`` arguments no longer have a CSV to act on
            and are ignored with a warning.

        Returns
        -------
        None
            Sets ``self.data``.
        """
        self.choose_stations(stations)
        if len(self.stations) == 1:
            self.get_single_station_data(
                variables=variables, start_date=start_date, end_date=end_date, **kwargs
            )
        else:
            if variables is None:
                _logger.info(
                    "Multiple stations chosen with variables=None — defaulting to WTEQ."
                )
            self.get_multiple_station_data(
                variables=variables or "WTEQ",
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )

    def get_single_station_data(
        self,
        variables: list[str] | None = None,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Fetch variables for the currently selected single station.

        Returns
        -------
        None
            Sets ``self.data`` to a :class:`pandas.DataFrame` indexed by
            ``datetime``, one column per variable, in metres and °C.
        """
        wanted = _variables(variables, default=list(VARIABLE_TYPES))
        ds = self._load(wanted, start_date, end_date)
        index = pd.DatetimeIndex(ds["time"].values, name="datetime")
        # A variable the station does not serve is an empty column, which is
        # what the frozen CSVs did — they carried all six headers regardless.
        frame = pd.DataFrame(
            {
                name: (
                    ds[name].isel(station=0).to_series().to_numpy()
                    if name in ds.data_vars
                    else float("nan")
                )
                for name in wanted
            },
            index=index,
        )
        self.data = _apply_dtype(frame, kwargs)
        _logger.info("Loaded data for station %s.", self.stations.index[0])

    def get_multiple_station_data(
        self,
        variables: str | list[str] = "WTEQ",
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Fetch one or more variables for all currently selected stations.

        Returns
        -------
        None
            Sets ``self.data`` to an :class:`xarray.Dataset` with water-year
            coordinates ``WY`` and ``DOWY``.
        """
        wanted = _variables(variables, default=["WTEQ"])
        ds = self._load(wanted, start_date, end_date)
        self.data = _apply_dtype(_to_legacy_dataset(ds, wanted, self.stations), kwargs)
        for name in wanted:
            setattr(self, name, self.data[name].to_pandas().T)
        _logger.info("Loaded %s for %d stations.", wanted, len(self.stations))

    def _load(
        self, variables: list[str], start_date: str, end_date: str | None
    ) -> xr.Dataset:
        """The adapter call behind every ``get_*_data`` method, in legacy units."""
        ds = _stations.load(
            self.stations,
            variables=[VARIABLE_TYPES[name] for name in variables],
            time=(start_date, end_date or today()),
        )
        return _to_legacy_units(ds, variables)

    def get_entire_data_archive(
        self, refresh: bool = True, temp_dir: str = "/tmp/", **kwargs: Any
    ) -> xr.Dataset:
        """Every station's whole daily record.

        .. deprecated:: 0.1.0
            Use :func:`easysnowdata.stations.archive.load`.

        Parameters
        ----------
        refresh, temp_dir
            Accepted for compatibility and ignored: the archive is cached by
            the package (``EASYSNOWDATA_CACHE_DIR`` moves the cache).
        **kwargs
            ``dtype`` is applied to the result; anything else is ignored.

        Returns
        -------
        xarray.Dataset
            ``WTEQ`` and ``SNWD`` for every station, with ``WY`` and ``DOWY``
            coordinates. The archive holds no precipitation or temperature;
            use :meth:`get_data` for those.
        """
        ds = _stations.archive.load(self.all_stations)
        ds = _to_legacy_units(ds, ["WTEQ", "SNWD"])
        out = _to_legacy_dataset(ds, ["WTEQ", "SNWD"], self.all_stations)
        self.entire_data_archive = _apply_dtype(out, kwargs)
        _logger.info("Full archive loaded (%d stations).", out.sizes["station"])
        return self.entire_data_archive


_TEMPERATURE_WARNED = False

#: The variables the NRCS bias correction moved.
_CORRECTED_TEMPERATURES = ("TAVG", "TMIN", "TMAX")


def _warn_about_temperature_vintage() -> None:
    """Say once that SNOTEL air temperature is the corrected series now."""
    global _TEMPERATURE_WARNED  # noqa: PLW0603 — warn once per process
    if _TEMPERATURE_WARNED:
        return
    _TEMPERATURE_WARNED = True
    message = (
        "SNOTEL air temperature (TAVG, TMIN, TMAX) is read live from AWDB, "
        "so it is the NRCS bias-corrected series. The frozen "
        "snotel_ccss_stations archive held the uncorrected values for "
        "roughly 2004-2024 at the stations that were corrected, about "
        "1.1 degC warm, because its updater never re-fetched history. "
        "Numbers from that era will not reproduce against the old CSVs, and "
        "the new ones are the right ones. CCSS stations are unaffected."
    )
    warnings.warn(message, UserWarning, stacklevel=3)
    _logger.warning("%s", message)


def _variables(variables: Any, *, default: list[str]) -> list[str]:
    if variables is None:
        return list(default)
    wanted = [variables] if isinstance(variables, str) else list(variables)
    unknown = [v for v in wanted if v not in VARIABLE_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown variable(s) {', '.join(unknown)}; this class serves "
            f"{', '.join(VARIABLE_TYPES)}. easysnowdata.stations.load() serves "
            "the full vocabulary."
        )
    if any(name in _CORRECTED_TEMPERATURES for name in wanted):
        _warn_about_temperature_vintage()
    return wanted


def _to_legacy_units(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Rename the standardized types back to the old names, in the old units."""
    out = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    for name in variables:
        type_name = VARIABLE_TYPES[name]
        if type_name not in ds.data_vars:
            continue
        factor = _TO_LEGACY_UNITS[name]
        values = ds[type_name] * factor if factor != 1.0 else ds[type_name]
        values.attrs = {**ds[type_name].attrs, "units": _legacy_units(name)}
        out[name] = values
    return out


def _legacy_units(name: str) -> str:
    return "degC" if name.startswith("T") else "m"


def _to_legacy_dataset(
    ds: xr.Dataset, variables: list[str], stations: gpd.GeoDataFrame | None
) -> xr.Dataset:
    """The old multi-station shape: ``WY``/``DOWY`` coords and metadata columns."""
    out = ds[[v for v in variables if v in ds.data_vars]]
    if "water_year" in out.coords:
        out = out.assign_coords(WY=out["water_year"], DOWY=out["dowy"])
    if stations is not None:
        for column in stations.columns:
            if column == "geometry" or column in out.coords:
                continue
            values = stations[column].reindex([str(c) for c in out["station"].values])
            out = out.assign_coords({column: ("station", values.to_numpy())})
    return out


def _apply_dtype(obj: Any, kwargs: dict[str, Any]) -> Any:
    """Honour ``dtype=`` the way ``pandas.read_csv`` did; warn about the rest."""
    dtype = kwargs.pop("dtype", None)
    if kwargs:
        _logger.warning(
            "Ignoring %s: these were pandas.read_csv arguments, and the data "
            "no longer comes from a CSV. See easysnowdata.stations.load().",
            ", ".join(sorted(kwargs)),
        )
    if dtype is None:
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.astype(dtype)
    mapping = dtype if isinstance(dtype, dict) else dict.fromkeys(obj.data_vars, dtype)
    return obj.assign(
        {name: obj[name].astype(kind) for name, kind in mapping.items() if name in obj}
    )
