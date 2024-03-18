import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
import xarray as xr
import logging
import subprocess
import pathlib

from easysnowdata.utils import convert_bbox_to_geodataframe


#based on https://github.com/egagli/snotel_ccss_stations/blob/main/example_usage.ipynb


def get_all_stations(data_available: bool = True, snotel_stations: bool = True, ccss_stations: bool = True, sortby_dist_to_geom = None) -> gpd.GeoDataFrame:    
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
    all_stations_gdf = gpd.read_file('https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson').set_index('code')

    # Filter based on data availability
    if data_available:
        all_stations_gdf = all_stations_gdf[all_stations_gdf['csvData']]

    # Filter out SNOTEL stations if not required
    if not snotel_stations:
        all_stations_gdf = all_stations_gdf[all_stations_gdf['network'] != 'SNOTEL']

    # Filter out CCSS stations if not required
    if not ccss_stations:
        all_stations_gdf = all_stations_gdf[all_stations_gdf['network'] != 'CCSS']

    # If a geometry is passed in, calculate the distance to this geometry for each station
    if sortby_dist_to_geom is not None:
        geom_gdf = convert_bbox_to_geodataframe(sortby_dist_to_geom)
        proj = 'EPSG:32611'
        all_stations_gdf['dist_km'] = all_stations_gdf.to_crs(proj).distance(geom_gdf.to_crs(proj).geometry[0])/1000
        all_stations_gdf = all_stations_gdf.sort_values('dist_km')

    return all_stations_gdf


def get_station_data(station_id):

    # certain variable or all

    #if string...
        
    #if list of strings
    #station_list = ['356_CA_SNTL','BLK']

    #station_dict = {}

    #for station in station_list:
    #    tmp = pd.read_csv(f'https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station}.csv',index_col='datetime',parse_dates=True)['WTEQ']
    #    station_dict[station] = tmp

    #stations_swe_df = pd.DataFrame.from_dict(station_dict)

    #if dataframe


    # if all stations
    
    # time 
    # station_dict = {}

    # for station in tqdm.tqdm(all_stations_gdf.index):
    #     try:
    #         tmp = pd.read_csv(f'https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station}.csv',index_col='datetime',parse_dates=True)['WTEQ']
    #         station_dict[station] = tmp
    #     except:
    #         print(f'failed to retrieve {station}')

    station_data_df = pd.read_csv(f'https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/{station_id}.csv',index_col='datetime', parse_dates=True)


    return station_data_df

def get_all_stations_all_data(all_stations_gdf: gpd.GeoDataFrame, temp_dir: str = '/tmp/') -> xr.Dataset:
    """
    Downloads, decompresses and processes automatic weather station data into an xarray Dataset.

    This function downloads a compressed file containing weather station data from a specific URL, 
    decompresses the file, and processes the data into an xarray Dataset. The data is organized by station, 
    with each station's data stored in a separate CSV file. The function also adds additional coordinates 
    to the Dataset from a provided GeoDataFrame.

    Parameters:
    all_stations_gdf (GeoDataFrame): A GeoDataFrame containing additional data for each station.
    temp_dir (str, optional): The directory where the compressed data file will be downloaded and decompressed. 
                              Defaults to '/tmp/'.

    Returns:
    xarray.Dataset: An xarray Dataset containing the processed weather station data.
    """

    github_tar_file_path = "https://github.com/egagli/snotel_ccss_stations/raw/main/data/all_station_data.tar.lzma"
    compressed_file_path = pathlib.Path(temp_dir, 'all_station_data.tar.lzma')
    decompressed_dir_path = pathlib.Path(temp_dir, 'data')

    if not compressed_file_path.exists():
        logging.info(f'Downloading compressed data to a temporary directory ({compressed_file_path})...')
        subprocess.run(['wget', '-q', '-P', temp_dir, github_tar_file_path], check=True)
    
    if not decompressed_dir_path.exists():
        logging.info(f'Decompressing data...')
        subprocess.run(['tar', '--lzma', '-xf', str(compressed_file_path), '-C', temp_dir], check=True)

    logging.info(f'Creating xarray.Dataset from the uncompressed data...')
    list_of_csv_files = glob.glob(str(decompressed_dir_path / '*.csv'))

    datasets = []
    for csv_file in list_of_csv_files:

        logging.info(f'Working on {csv_file}...')
        # Extract station name from the csv file name
        station_name = csv_file.split('/')[-1].split('.')[0]

        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_file, parse_dates=True).rename(columns={'datetime':'time'}).set_index('time').sort_index()

        # Convert the DataFrame into an xarray DataSet and add station coordinate
        ds = df.to_xarray()
        ds.coords['station'] = station_name

        # Add other coordinates from all_stations_gdf
        for col in all_stations_gdf.columns:
            ds.coords[col] = all_stations_gdf.loc[station_name, col]

        datasets.append(ds)

    logging.info(f'Combining all dataarrays into one dataset...')
    ds = xr.concat(datasets,dim='station')

    #logging.info(f'Removing temporary directory...')
    #subprocess.run(['rm', '-rf', temp_dir], check=True)

    logging.info(f'Done!')

    return ds