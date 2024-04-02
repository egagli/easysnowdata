import glob
import logging
import os
import pathlib
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

import datetime
today = datetime.datetime.now().strftime('%Y-%m-%d')

from easysnowdata.utils import (
    convert_bbox_to_geodataframe,
    datetime_to_DOWY,
    datetime_to_WY,
)

# based on https://github.com/egagli/snotel_ccss_stations/blob/main/example_usage.ipynb


class StationCollection:
    def __init__(
        self,
        data_available: bool = True,
        #snotel_stations: bool = True,
        #ccss_stations: bool = True,
        sortby_dist_to_geom=None,
    ):

        self.data_available = data_available
        #self.snotel_stations = snotel_stations
        #self.ccss_stations = ccss_stations
        self.sortby_dist_to_geom = sortby_dist_to_geom

        self.all_stations = None
        self.stations = None

        self.TAVG = None
        self.TMIN = None
        self.TMAX = None
        self.SNWD = None
        self.WTEQ = None
        self.PRCPSA = None

        self.data = None

        self.entire_data_archive = None

        self.get_all_stations()


    def get_all_stations(self):
        """
        Fetches all weather stations from a GeoJSON file hosted on the snotel_ccss_stations GitHub repository.
        The function allows filtering based on data availability and station network type. Optionally, a geometry
        can be passed in to calculate the distance of each station to this geometry and sort the stations by this distance.

        Parameters:
        data_available (bool): If True, only stations with available data are included.
        snotel_stations (bool): If True, stations from the SNOTEL network are included.
        ccss_stations (bool): If True, stations from the CCSS network are included.
        sortby_dist_to_geom (GeoDataFrame, tuple, Polygon): An optional geometry to calculate distances to.
                                                   Can be a GeoDataFrame, a tuple representing a point (longitude, latitude),
                                                   or a Shapely Polygon.

        Returns:
        GeoDataFrame: A GeoDataFrame containing the details of the stations. Optionally adds a 'dist_km' column, representing distance from each station to the input geometry.
        """
        # Read the GeoJSON file
        all_stations_gdf = gpd.read_file(
            "https://github.com/egagli/snotel_ccss_stations/raw/main/all_stations.geojson"
        ).set_index("code")

        # # Filter based on data availability
        if self.data_available:
            all_stations_gdf = all_stations_gdf[all_stations_gdf["csvData"]]

        # # Filter out SNOTEL stations if not required
        # if not self.snotel_stations:
        #     all_stations_gdf = all_stations_gdf[all_stations_gdf["network"] != "SNOTEL"]

        # # Filter out CCSS stations if not required
        # if not self.ccss_stations:
        #     all_stations_gdf = all_stations_gdf[all_stations_gdf["network"] != "CCSS"]

        # If a geometry is passed in, calculate the distance to this geometry for each station
        if self.sortby_dist_to_geom is not None:
            print(f"Sorting by distance to given geometry. See dist_km column.")
            geom_gdf = convert_bbox_to_geodataframe(self.sortby_dist_to_geom)
            proj = "EPSG:32611"
            all_stations_gdf["dist_km"] = (
                all_stations_gdf.to_crs(proj).distance(
                    geom_gdf.to_crs(proj).geometry[0]
                )
                / 1000
            )
            all_stations_gdf = all_stations_gdf.sort_values("dist_km")

        self.all_stations = all_stations_gdf
        print(
            f"Geodataframe with all stations has been added to the Station object. Please use the .all_stations attribute to access."
        )
        print(
            f"Use the .get_data(stations=geodataframe/string/list,variables=string/list,start_date=str,end_date=str) method to fetch data for specific stations and variables."
        )

    def choose_stations(self, stations_gdf):
        """
        Choose the stations to be used for further analysis.

        Parameters:
        - stations_gdf: gpd.GeoDataFrame, str, or list
            The name(s) or index(es) of the stations to be chosen.

        Returns:
        None
        """
        if type(stations_gdf) is str:
            stations_gdf = self.all_stations.loc[[stations_gdf]]
        if type(stations_gdf) is list:
            stations_gdf = self.all_stations.loc[stations_gdf]

        self.stations = stations_gdf

    def get_data(self, stations="679_WA_SNTL", variables=None, start_date='1900-01-01', end_date=today):
        """
        Retrieves data for the specified stations and variables. Access retrieved data using the .{variable} or .data attribute.

        Parameters:
        - stations (geodataframe, str, or list): The stations to retrieve data for. Defaults to '679_WA_SNTL'.
        - variables (str or list): The variables to retrieve data for. Defaults to None.
        - start_date (str): The start date for the data. Defaults to '1900-01-01'.
        - end_date (str): The end date for the data. Defaults to today's date.
        
        Returns:
        None

        If only one station is chosen and variables is None, all variables for that station will be fetched.
        If only one station is chosen and variables is specified, only the specified variables will be fetched for that station.
        If multiple stations are chosen and variables is None, WTEQ (Water Equivalent) data will be fetched for all stations.
        If multiple stations are chosen and variables is specified, only the specified variables will be fetched for all stations.

        Example usage:
        >>> stations = easysnowdata.automatic_weather_stations.Stations()
        >>> stations.get_data(stations='679_WA_SNTL', variables=['WTEQ', 'SNWD'])
        Retrieves SWE and snow depth data for the station '679_WA_SNTL'.

        >>> bbox_gdf = gpd.read_file('https://github.com/egagli/sar_snowmelt_timing/raw/main/input/shapefiles/mt_rainier.geojson')
        >>> stations = easysnowdata.automatic_weather_stations.Stations(sortby_dist_to_geom=bbox_gdf)
        >>> stations.get_data(stations=stations.all_stations.iloc[0:4])
        Retrieves all variables for the 4 stations closest to the input geometry.
        """

        self.choose_stations(stations)

        if len(self.stations) == 1:
            if variables is None:
                print(
                    f"Only one station chosen with variables=None. Default behavior fetches all variables for this station."
                )
                self.get_single_station_data(start_date=start_date, end_date=end_date)
            else:
                self.get_single_station_data(variables=variables, start_date=start_date, end_date=end_date)
        else:
            if variables is None:
                print(
                    f"Multiple stations chosen with variables=None. Default behavior fetches WTEQ for all stations."
                )
                self.get_multiple_station_data(start_date=start_date, end_date=end_date)
            else:
                self.get_multiple_station_data(variables=variables, start_date=start_date, end_date=end_date)

    def get_single_station_data(
        self, variables=["WTEQ", "SNWD", "PRCPSA", "TAVG", "TMIN", "TMAX"], start_date='1900-01-01', end_date=today
    ):
        """
        Retrieves data for a single weather station.

        Args:
            variables (list, optional): List of variables to include in the data. Defaults to ['WTEQ','SNWD','PRCPSA','TAVG','TMIN','TMAX'].

        Returns:
            None
        """

        single_station_df = pd.read_csv(
            f"https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{self.stations.index.values[0]}.csv",
            index_col="datetime",
            parse_dates=True,
        )

        columns_to_drop = [
            col for col in single_station_df.columns if col not in variables
        ]
        single_station_df = single_station_df.drop(columns=columns_to_drop).loc[start_date:end_date]
        self.data = single_station_df
        print(
            f"Dataframe has been added to the Station object. Please use the .data attribute to access."
        )

    def get_multiple_station_data(self, variables="WTEQ", start_date='1900-01-01', end_date=today):
        """
        Fetches the data for the specified variable(s) from the snotel_ccss_stations GitHub repository.

        Parameters:
        variables (str or list): The variable(s) to fetch. Default is 'WTEQ' (water equivalent of snow on the ground).
                                 Can be a single variable as a string or multiple variables as a list.

        Returns:
        DataFrame: A DataFrame containing the data for the specified variable(s).

        """

        dataarrays = []

        if isinstance(variables, str):
            variables = [variables]

        for variable in variables:

            self.station_dict = {}

            for station in tqdm.tqdm(self.stations.index):
                try:
                    tmp = pd.read_csv(
                        f"https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station}.csv",
                        index_col="datetime",
                        parse_dates=True,
                    )[variable]
                    self.station_dict[station] = tmp
                except:
                    print(f"failed to retrieve {station}")

            station_data_df = pd.DataFrame.from_dict(self.station_dict).loc[start_date:end_date]

            setattr(self, f"{variable}", station_data_df)
            print(
                f"{variable} dataframe has been added to the Station object. Please use the .{variable} attribute to access the dataframe."
            )

            station_data_da = (
                station_data_df.to_xarray()
                .to_dataarray(dim="station")
                .rename(f"{variable}")
                .rename({"datetime": "time"})
            )

            dataarrays.append(station_data_da)

        all_stations_ds = xr.merge(dataarrays)

        for col in self.stations.columns:
            all_stations_ds = all_stations_ds.assign_coords(
                {f"{col}": ("station", self.stations[f"{col}"])}
            )

        all_stations_ds["time"] = pd.to_datetime(all_stations_ds.time)

        water_year = pd.to_datetime(all_stations_ds.time).map(datetime_to_WY)
        day_of_water_year = pd.to_datetime(all_stations_ds.time).map(datetime_to_DOWY)

        all_stations_ds.coords["WY"] = ("time", water_year)
        all_stations_ds.coords["DOWY"] = ("time", day_of_water_year)

        self.data = all_stations_ds
        print(
            f"Full {variables} dataset has been added to the station object. Please use the .data attribute to access the dataset."
        )

    def get_entire_data_archive(self, refresh: bool = True, temp_dir: str = "/tmp/") -> xr.Dataset:
        """
        Downloads, decompresses and processes automatic weather station data into an xarray Dataset.

        This function downloads a compressed file containing weather station data from a specific URL,
        decompresses the file, and processes the data into an xarray Dataset. The data is organized by station,
        with each station's data stored in a separate CSV file. The function also adds additional coordinates
        to the Dataset from a provided GeoDataFrame. Set the refresh parameter to True to redownload cached data.

        Parameters:
        all_stations_gdf (GeoDataFrame): A GeoDataFrame containing additional data for each station.
        refresh (bool, optional): If True, the compressed data file will be redownloaded. Defaults to True.
        temp_dir (str, optional): The directory where the compressed data file will be downloaded and decompressed.
                                Defaults to '/tmp/'.

        Returns:
        xarray.Dataset: An xarray Dataset containing the processed weather station data.
        """

        github_tar_file_path = "https://github.com/egagli/snotel_ccss_stations/raw/main/data/all_station_data.tar.lzma"
        compressed_file_path = pathlib.Path(temp_dir, "all_station_data.tar.lzma")
        decompressed_dir_path = pathlib.Path(temp_dir, "data")

        if not compressed_file_path.exists() or refresh:
            print(
                f"Downloading compressed data to a temporary directory ({compressed_file_path})..."
            )
            subprocess.run(
                ["wget", "-q", "-P", temp_dir, github_tar_file_path], check=True
            )

        if not decompressed_dir_path.exists() or refresh:
            print(f"Decompressing data...")
            subprocess.run(
                ["tar", "--lzma", "-xf", str(compressed_file_path), "-C", temp_dir],
                check=True,
            )

        print(f"Creating xarray.Dataset from the uncompressed data...")
        list_of_csv_files = glob.glob(str(decompressed_dir_path / "*.csv"))

        datasets = []
        for csv_file in list_of_csv_files:

            logging.info(f"Working on {csv_file}...")
            # Extract station name from the csv file name
            station_name = csv_file.split("/")[-1].split(".")[0]

            # Load the CSV data into a pandas DataFrame
            station_df = (
                pd.read_csv(csv_file, parse_dates=True)
                .rename(columns={"datetime": "time"})
                .set_index("time")
                .sort_index()
            )

            # Convert the DataFrame into an xarray DataSet and add station coordinate
            station_ds = station_df.to_xarray()
            station_ds.assign_coords(station=station_name)

            # Add other coordinates from all_stations_gdf
            for col in self.all_stations.columns:
                station_ds.coords[col] = self.all_stations.loc[station_name, col]

            datasets.append(station_ds)

        logging.info(f"Combining all dataarrays into one dataset...")
        all_stations_ds = xr.concat(datasets, dim="station", coords="all")
        all_stations_ds["time"] = pd.to_datetime(all_stations_ds.time)

        water_year = pd.to_datetime(all_stations_ds.time).map(datetime_to_WY)
        day_of_water_year = pd.to_datetime(all_stations_ds.time).map(datetime_to_DOWY)

        all_stations_ds.coords["WY"] = ("time", water_year)
        all_stations_ds.coords["DOWY"] = ("time", day_of_water_year)

        self.entire_data_archive = all_stations_ds

        print(
            f"Done! Entire archive dataset has been added to the station object. Please use the .entire_data_archive attribute to access."
        )
