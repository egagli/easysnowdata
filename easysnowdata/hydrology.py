import geopandas as gpd
import ee
import json
import xarray as xr

from easysnowdata.utils import convert_bbox_to_geodataframe

#ee.Authenticate() need to figure out https://developers.google.com/earth-engine/guides/auth
#ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def get_huc_geometries(bbox_input=(-180,-90,180,90),huc_level='02'):
    """
    Retrieves Hydrologic Unit Code (HUC) geometries within a specified bounding box and HUC level.

    This function queries the USGS Water Boundary Dataset (WBD) for HUC geometries. It can retrieve
    HUC geometries at different levels (e.g., HUC 02, 04, 06, 08, 10, 12) for a specified region
    defined by a bounding box. If no bounding box is provided, it retrieves HUC geometries for the
    entire United States.

    Parameters:
    - bbox_input (tuple of float, optional): A tuple representing the bounding box in the format
      (min_lon, min_lat, max_lon, max_lat). Defaults to (-180, -90, 180, 90) which represents the
      entire world.
    - huc_level (str, optional): The HUC level to retrieve geometries for. Valid levels are '02',
      '04', '06', '08', '10', '12'. Defaults to '02'.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the retrieved HUC geometries along with associated
      attributes such as name, area in square kilometers, states, TNMID, and geometry.
    """

    # Check if the default bounding box is used
    if isinstance(bbox_input, tuple) and (bbox_input == (-180, -90, 180, 90)):
        print(f'No bounding box input provided, retrieving all HUC{huc_level} geometries. This will take a moment...')
    else:
        print(f'Retrieving HUC{huc_level} geometries for the region of interest...')

    # Convert bounding box to feature collection to use as region for querying HUC geometries
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
    bbox_json = bbox_gdf.to_json()
    featureCollection = ee.FeatureCollection(json.loads(bbox_json))

    # Search Earth Engine USGS WBD collection for HUC geometries
    huc_gdf = ee.data.listFeatures({
        'assetId': f'USGS/WBD/2017/HUC{huc_level}',
        'region': featureCollection.geometry().getInfo(),
        'fileFormat': 'GEOPANDAS_GEODATAFRAME'
    })

    # Add crs to geodataframe and select relevant columns
    huc_gdf.crs = 'EPSG:4326'
    huc_gdf = huc_gdf[['name',f'huc{huc_level.lstrip("0")}','areasqkm','states','tnmid','geometry']]

    return huc_gdf


def get_era5(bbox_input=(-180,-90,180,90)):
  """
  Retrieves ERA5 data for a given bounding box.

  Parameters:
  bbox_input (tuple): Bounding box coordinates in the format (min_lon, min_lat, max_lon, max_lat).
             Default value is (-180, -90, 180, 90) which represents the global bounding box.

  Returns:
  era5 (xarray dataset): ERA5 dataset for the specified bounding box.
  """

  # Check if the default bounding box is used
  if isinstance(bbox_input, tuple) and (bbox_input == (-180, -90, 180, 90)):
    print(f'No bounding box input provided, retrieving global ERA5 data...')
    clip_flag = False
  else:
    print(f'Retrieving ERA5 data for the region of interest...')
    clip_flag = True

  bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

  era5 = xr.open_zarr( #https://cloud.google.com/storage/docs/public-datasets/era5
      "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
      chunks={'time': 48},
      consolidated=True,
  )
  era5.rio.write_crs("EPSG:4326", inplace=True)
  era5.coords['longitude'] = (era5.coords['longitude'] + 180) % 360 - 180
  era5 = era5.sortby(era5.longitude)

  if clip_flag:
    era5 = era5.rio.clip(bbox_gdf.geometry,all_touched=True)

  # ar_full_37_1h = xr.open_zarr( #https://github.com/google-research/arco-era5
  #   'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
  #   chunks=None,
  #   storage_options=dict(token='anon'),
  # )

  return era5

#huc map, from gee?

#hydroatlas? https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroATLAS_v1_Basins_level03

#maybe seperate climate module,or use https://github.com/hyriver/pydaymet

#https://github.com/OpenTopography/OT_3DEP_Workflows/blob/main/notebooks/03_3DEP_Generate_DEM_USGS_HUCs.ipynb
# from opentopo....
# Now exectute the call to the USGS Watershed Boundary Dataset REST API. There are several possibilities here if the following step does not execute properly. There is a print statement implemented that should provide some indication of which it is.

# If you receive the error Error with Service Call or Error loading JSON output, this most likely indicates that the region you selected does coincide with a USGS hydrologic unit from that particular service.

# If you recieve the error {'message': 'Endpoint request timed out'}, this likely indicates the Watershed Boundary Dataset service may be down. To check, click this link (https://stats.uptimerobot.com/gxzRZFARLZ/783928857). If you are greeted with the message ScienceBase is down., the service is temporarily down and it is not possible to query the service at this time.

# If the cell executes successfully, the interesecting watershed boundary geometry will be printed.

# #url for 12-digit HUCs is ../MapServer/6/
# #url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/7/query?'   #14-Digit HUC
# url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?'   #12-Digit HUC

# #the parameters here will query the map server for the appropriate HU boundary
# params = dict(geometry=user_AOI,geometryType='esriGeometryEnvelope',inSR='4326',
#               spatialRel='esriSpatialRelIntersects',f='geojson')

# #Execute REST API call.  
# try:
#     r = requests.get(url,params=params)
# except:
#     print('Error with Service Call. This could mean that there is no hydrologic unit polygon where you selected.')

# #load API JSON output into a variable
# try:
#     wbd_geojson = json.loads(r.content)
#     print(wbd_geojson)   
# except:
#     print('Error loading JSON output')

# #To write out a JSON file...
# with open('WBD_API_Query.geojson', 'w') as outfile:
#     json.dump(wbd_geojson, outfile)