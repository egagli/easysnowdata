import pandas as pd
import geopandas as gpd
import numpy as np

from easysnowdata.utils import convert_bbox_to_geodataframe


#based on https://github.com/egagli/snotel_ccss_stations/blob/main/example_usage.ipynb


def get_all_stations(data_available=True, snotel_stations=True, ccss_stations=True, sortby_dist_to_geom=None):
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

