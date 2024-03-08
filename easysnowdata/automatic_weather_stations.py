import pandas as pd
import geopandas as gpd
import numpy as np



#based on https://github.com/egagli/snotel_ccss_stations/blob/main/example_usage.ipynb


def get_all_stations(data_available=True): # add snotel_stations=True, ccss_stations=True, etc.

    all_stations_gdf = gpd.read_file('https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson').set_index('code')

    if data_available:
        all_stations_gdf = all_stations_gdf[all_stations_gdf['csvData']==True]

    return all_stations_gdf


#def get_nearest_stations()


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

