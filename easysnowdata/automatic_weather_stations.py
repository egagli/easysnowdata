"""Access SNOTEL and CCSS automatic weather station data.

Data are hosted on the companion repository
`egagli/snotel_ccss_stations <https://github.com/egagli/snotel_ccss_stations>`_
and retrieved as individual CSV files or as a single compressed archive.

References
----------
- SNOTEL: https://www.nrcs.usda.gov/wps/portal/wcc/home/quicklinks/imap
- CCSS: https://cdec.water.ca.gov/snow/current/snow/
"""

from __future__ import annotations

import datetime
import glob
import logging
import pathlib
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    datetime_to_DOWY,
    datetime_to_WY,
)

__all__ = ["StationCollection"]

_logger = logging.getLogger(__name__)

_STATION_GEOJSON_URL = (
    "https://github.com/egagli/snotel_ccss_stations/raw/main/all_stations.geojson"
)
_STATION_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/"
)
_ARCHIVE_URL = (
    "https://github.com/egagli/snotel_ccss_stations/raw/main/data/all_station_data.tar.lzma"
)


class StationCollection:
    """A collection of SNOTEL and CCSS automatic weather stations.

    Retrieves station metadata and time-series data from the
    `egagli/snotel_ccss_stations <https://github.com/egagli/snotel_ccss_stations>`_
    GitHub repository. Outputs are pandas DataFrames (single station) or
    xarray Datasets (multiple stations).

    Parameters
    ----------
    data_available : bool, optional
        If ``True`` (default), only include stations that have CSV data files.
    sortby_dist_to_geom : GeoDataFrame or tuple or shapely geometry, optional
        If provided, stations are sorted by distance to this geometry and a
        ``dist_km`` column is added to ``all_stations``.

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
    Single-station retrieval (returns a DataFrame):

    >>> sc = StationCollection()
    >>> sc.get_data(stations="679_WA_SNTL", variables=["WTEQ", "SNWD"],
    ...             start_date="2020-10-01", end_date="2021-09-30")
    >>> sc.data.head()

    Multi-station retrieval (returns an xarray Dataset):

    >>> sc = StationCollection()
    >>> sc.get_data(stations=["679_WA_SNTL", "680_WA_SNTL"],
    ...             variables=["WTEQ"],
    ...             start_date="2022-01-01", end_date="2022-03-31")
    >>> sc.data

    Notes
    -----
    Available variables: ``WTEQ`` (SWE), ``SNWD`` (snow depth),
    ``PRCPSA`` (accumulated precipitation), ``TAVG``, ``TMIN``, ``TMAX``.
    """

    def __init__(
        self,
        data_available: bool = True,
        sortby_dist_to_geom: gpd.GeoDataFrame | tuple | None = None,
    ) -> None:
        self.data_available = data_available
        self.sortby_dist_to_geom = sortby_dist_to_geom

        self.all_stations: gpd.GeoDataFrame | None = None
        self.stations: gpd.GeoDataFrame | None = None
        self.data: pd.DataFrame | xr.Dataset | None = None
        self.entire_data_archive: xr.Dataset | None = None

        # Per-variable DataFrames populated by get_multiple_station_data
        self.TAVG: pd.DataFrame | None = None
        self.TMIN: pd.DataFrame | None = None
        self.TMAX: pd.DataFrame | None = None
        self.SNWD: pd.DataFrame | None = None
        self.WTEQ: pd.DataFrame | None = None
        self.PRCPSA: pd.DataFrame | None = None

        self.get_all_stations()

    def get_all_stations(self) -> None:
        """Fetch all station metadata from GitHub and populate ``all_stations``.

        Optionally filters to stations with data files and sorts by distance
        to ``sortby_dist_to_geom`` if provided.

        Returns
        -------
        None
            Sets ``self.all_stations``.
        """
        all_stations_gdf = gpd.read_file(_STATION_GEOJSON_URL).set_index("code")

        if self.data_available:
            all_stations_gdf = all_stations_gdf[all_stations_gdf["csvData"]]

        if self.sortby_dist_to_geom is not None:
            _logger.info("Sorting stations by distance to provided geometry.")
            geom_gdf = convert_bbox_to_geodataframe(self.sortby_dist_to_geom)
            proj = "EPSG:32611"
            all_stations_gdf["dist_km"] = (
                all_stations_gdf.to_crs(proj).distance(
                    geom_gdf.to_crs(proj).geometry.iloc[0]
                )
                / 1000
            )
            all_stations_gdf = all_stations_gdf.sort_values("dist_km")

        self.all_stations = all_stations_gdf
        _logger.info(
            "Loaded %d stations into all_stations.", len(self.all_stations)
        )

    def choose_stations(
        self, stations_input: gpd.GeoDataFrame | str | list
    ) -> None:
        """Select a subset of stations by code string, list of codes, or GeoDataFrame.

        Parameters
        ----------
        stations_input : str, list of str, or geopandas.GeoDataFrame
            Station code(s) to select (e.g. ``"679_WA_SNTL"`` or
            ``["679_WA_SNTL", "680_WA_SNTL"]``), or a GeoDataFrame already
            filtered from ``all_stations``.

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
    ) -> None:
        """Fetch data for the given stations and variables.

        Dispatches to :meth:`get_single_station_data` or
        :meth:`get_multiple_station_data` based on the number of stations
        selected.

        Parameters
        ----------
        stations : str, list of str, or GeoDataFrame, optional
            Station code(s) to fetch. Default is ``"679_WA_SNTL"``
            (Paradise, WA SNOTEL).
        variables : str or list of str, optional
            Variable(s) to fetch. Defaults to all variables for a single
            station, or ``WTEQ`` for multiple stations.
        start_date : str, optional
            ISO date string ``"YYYY-MM-DD"``. Default is ``"1900-01-01"``.
        end_date : str, optional
            ISO date string. Default is today's date.

        Returns
        -------
        None
            Sets ``self.data``.
        """
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")

        self.choose_stations(stations)

        if len(self.stations) == 1:
            self.get_single_station_data(
                variables=variables, start_date=start_date, end_date=end_date
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
            )

    def get_single_station_data(
        self,
        variables: list[str] | None = None,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
    ) -> None:
        """Fetch all (or selected) variables for the currently selected single station.

        Parameters
        ----------
        variables : list of str, optional
            Variable columns to keep. Defaults to all available variables.
        start_date : str, optional
            ISO date string. Default ``"1900-01-01"``.
        end_date : str, optional
            ISO date string. Defaults to today.

        Returns
        -------
        None
            Sets ``self.data`` to a :class:`pandas.DataFrame`.
        """
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if variables is None:
            variables = ["WTEQ", "SNWD", "PRCPSA", "TAVG", "TMIN", "TMAX"]

        station_code = self.stations.index[0]
        url = f"{_STATION_DATA_BASE_URL}{station_code}.csv"
        df = pd.read_csv(url, index_col="datetime", parse_dates=True)

        drop_cols = [c for c in df.columns if c not in variables]
        self.data = df.drop(columns=drop_cols).loc[start_date:end_date]
        _logger.info("Loaded data for station %s.", station_code)

    def get_multiple_station_data(
        self,
        variables: str | list[str] = "WTEQ",
        start_date: str = "1900-01-01",
        end_date: str | None = None,
    ) -> None:
        """Fetch one or more variables for all currently selected stations.

        Parameters
        ----------
        variables : str or list of str, optional
            Variable(s) to retrieve. Default is ``"WTEQ"``.
        start_date : str, optional
            ISO date string. Default ``"1900-01-01"``.
        end_date : str, optional
            ISO date string. Defaults to today.

        Returns
        -------
        None
            Sets ``self.data`` to an :class:`xarray.Dataset` with water-year
            coordinates ``WY`` and ``DOWY``.
        """
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if isinstance(variables, str):
            variables = [variables]

        dataarrays = []
        for variable in variables:
            station_dict: dict[str, pd.Series] = {}
            for station in tqdm.tqdm(self.stations.index, desc=variable):
                try:
                    url = f"{_STATION_DATA_BASE_URL}{station}.csv"
                    tmp = pd.read_csv(
                        url, index_col="datetime", parse_dates=True
                    )[variable]
                    station_dict[station] = tmp
                except Exception as exc:
                    _logger.warning("Failed to retrieve %s for %s: %s", variable, station, exc)

            station_df = pd.DataFrame.from_dict(station_dict).loc[start_date:end_date]
            setattr(self, variable, station_df)

            da = (
                station_df.to_xarray()
                .to_dataarray(dim="station")
                .rename(variable)
                .rename({"datetime": "time"})
            )
            dataarrays.append(da)

        ds = xr.merge(dataarrays)

        for col in self.stations.columns:
            ds = ds.assign_coords({col: ("station", self.stations[col])})

        ds["time"] = pd.to_datetime(ds.time)
        ds.coords["WY"] = ("time", pd.to_datetime(ds.time).map(datetime_to_WY))
        ds.coords["DOWY"] = ("time", pd.to_datetime(ds.time).map(datetime_to_DOWY))

        self.data = ds
        _logger.info(
            "Loaded %s for %d stations.", variables, len(self.stations)
        )

    def get_entire_data_archive(
        self, refresh: bool = True, temp_dir: str = "/tmp/"
    ) -> xr.Dataset:
        """Download, decompress, and assemble the full station data archive.

        Parameters
        ----------
        refresh : bool, optional
            Re-download the archive even if it already exists locally.
            Default is ``True``.
        temp_dir : str, optional
            Local directory for the downloaded archive. Default is ``"/tmp/"``.

        Returns
        -------
        xarray.Dataset
            All variables for all stations with ``WY`` and ``DOWY`` coordinates.
            Also stored as ``self.entire_data_archive``.

        Notes
        -----
        The compressed archive is ~several hundred MB; allow a few minutes for
        download and decompression on first run.
        """
        compressed_path = pathlib.Path(temp_dir, "all_station_data.tar.lzma")
        decompressed_dir = pathlib.Path(temp_dir, "data")

        if not compressed_path.exists() or refresh:
            _logger.info("Downloading archive to %s …", compressed_path)
            subprocess.run(
                ["wget", "-q", "-P", temp_dir, _ARCHIVE_URL], check=True
            )

        if not decompressed_dir.exists() or refresh:
            _logger.info("Decompressing archive …")
            subprocess.run(
                ["tar", "--lzma", "-xf", str(compressed_path), "-C", temp_dir],
                check=True,
            )

        _logger.info("Building xarray.Dataset from decompressed CSVs …")
        datasets = []
        for csv_file in glob.glob(str(decompressed_dir / "*.csv")):
            station_name = pathlib.Path(csv_file).stem
            df = (
                pd.read_csv(csv_file, parse_dates=True)
                .rename(columns={"datetime": "time"})
                .set_index("time")
                .sort_index()
            )
            station_ds = df.to_xarray().assign_coords(station=station_name)
            for col in self.all_stations.columns:
                station_ds.coords[col] = self.all_stations.loc[station_name, col]
            datasets.append(station_ds)

        ds = xr.concat(datasets, dim="station", coords="all")
        ds["time"] = pd.to_datetime(ds.time)
        ds.coords["WY"] = ("time", pd.to_datetime(ds.time).map(datetime_to_WY))
        ds.coords["DOWY"] = ("time", pd.to_datetime(ds.time).map(datetime_to_DOWY))

        self.entire_data_archive = ds
        _logger.info("Full archive loaded (%d stations).", len(datasets))
        return ds
