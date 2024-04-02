# #!/usr/bin/env python

# """Tests for `easysnowdata` package."""


# import unittest

# from easysnowdata import automatic_weather_stations
# import pandas as pd
# import geopandas as gpd


# class TestAutomaticWeatherStations(unittest.TestCase):
#     """Tests for `easysnowdata` package."""
    
#     @classmethod
#     def setUpClass(self):
#         """Set up test fixtures, if any."""
#         bbox_gdf = gpd.read_file('https://github.com/egagli/sar_snowmelt_timing/raw/main/input/shapefiles/mt_rainier.geojson')
#         self.StationCollectionSNOTEL = automatic_weather_stations.StationCollection(snotel_stations=True, ccss_stations=False)
#         self.StationCollectionCCSS = automatic_weather_stations.StationCollection(snotel_stations=False, ccss_stations=True)
#         self.StationCollectionAll = automatic_weather_stations.StationCollection(snotel_stations=True, ccss_stations=True)
#         self.StationCollectionAllOrdered = automatic_weather_stations.StationCollection(snotel_stations=True, ccss_stations=False, sortby_dist_to_geom=bbox_gdf)

#     @classmethod
#     def tearDownClass(self):
#         """Tear down test fixtures, if any."""
#         self.StationCollectionSNOTEL = None
#         self.StationCollectionCCSS = None
#         self.StationCollectionAll = None
#         self.StationCollectionAllOrdered = None

        
#     def test_get_all_stations(self):
#         """Test get_all_stations method in automatic_weather_stations class."""
#         self.StationCollection.get_all_stations()
#         self.assertIsInstance(self.StationCollectionAll.stations, gpd.GeoDataFrame)
#         self.assertGreater(len(self.StationCollectionAll.stations), 960)
        
#     def test_choose_station(self):
#         """Test choose_station method in automatic_weather_stations class."""
#         self.StationCollection.choose_station()
#         self.assertIsInstance(self.StationCollection.station, dict)
        
#     def test_get_data(self):
#         """Test get_data method in automatic_weather_stations class."""
        
#         self.StationCollection.get_data()
#         self.assertIsInstance(self.StationCollection.data, pd.DataFrame)
        
#     def test_get_single_station_data(self):
#         """Test get_single_station_data method in automatic_weather_stations class."""
#         self.StationCollection.get_single_station_data()
#         self.assertIsInstance(self.StationCollection.data, pd.DataFrame)
        
#     def test_multiple_station_data(self):
#         """Test multiple_station_data method in automatic_weather_stations class."""
#         self.StationCollection.multiple_station_data()
#         self.assertIsInstance(self.StationCollection.data, pd.DataFrame)
        
#     def test_get_entire_data_archive(self):
#         """Test get_entire_data_archive method in automatic_weather_stations class."""
#         self.StationCollection.get_entire_data_archive()
#         self.assertIsInstance(self.StationCollection.data, pd.DataFrame)
        
