import re
import pandas as pd
import geopandas as gpd
import ee
import json
import xarray as xr
import numpy as np
import earthaccess
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Union
import shapely
import tempfile
import zipfile
import requests
import os


from easysnowdata.utils import convert_bbox_to_geodataframe

# ee.Authenticate() need to figure out https://developers.google.com/earth-engine/guides/auth
# ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def get_huc_geometries(
        bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None, 
        huc_level: str = "02",
) -> gpd.GeoDataFrame:
    """
    Retrieves Hydrologic Unit Code (HUC) geometries within a specified bounding box and HUC level.

    This function queries the USGS Water Boundary Dataset (WBD) for HUC geometries. It can retrieve
    HUC geometries at different levels for a specified region defined by a bounding box. If no 
    bounding box is provided, it retrieves HUC geometries for the entire United States.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire US dataset is returned.
    huc_level : str, optional
        The HUC level to retrieve geometries for. Valid levels are '02', '04', '06', '08', '10', '12'.
        Default is '02'.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the retrieved HUC geometries along with associated attributes
        such as name, area in square kilometers, states, TNMID, and geometry.

    Examples
    --------
    Get HUC geometries for a specific region at HUC level 08...

    >>> huc_data = get_huc_geometries(bbox_input=(-121.94, 46.72, -121.54, 46.99), huc_level="08")
    >>> huc_data.plot()

    Notes
    -----
    This function requires an active Earth Engine session. Make sure to authenticate 
    with Earth Engine before using this function.

    Data citation: 
    Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, 
    Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): 
    U.S. Geological Survey Techniques and Methods 11-A3, 54 p., 
    https://doi.org/10.3133/tm11A3
    """

    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    # Convert bounding box to feature collection to use as region for querying HUC geometries
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)
    bbox_json = bbox_gdf.to_json()
    featureCollection = ee.FeatureCollection(json.loads(bbox_json))

    # Search Earth Engine USGS WBD collection for HUC geometries
    huc_gdf = ee.data.listFeatures(
        {
            "assetId": f"USGS/WBD/2017/HUC{huc_level}",
            "region": featureCollection.geometry().getInfo(),
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        }
    )

    # Add crs to geodataframe and select relevant columns
    huc_gdf.crs = "EPSG:4326"
    huc_gdf = huc_gdf[
        [
            "name",
            f'huc{huc_level.lstrip("0")}',
            "areasqkm",
            "states",
            "tnmid",
            "geometry",
        ]
    ]

    huc_gdf.attrs = {"Data citation": "Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): U.S. Geological Survey Techniques and Methods 11-A3, 54 p., https://doi.org/10.3133/tm11A3"}
    
    return huc_gdf


def get_hydroBASINS(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    regions: str | list = "all",
    level: int = 4,
) -> gpd.GeoDataFrame:
    """
    Retrieves HydroBASINS sub-basin boundaries at specified hierarchical level.

    This function downloads and loads vectorized polygon layers depicting sub-basin boundaries
    from the HydroBASINS database. It provides consistently sized and hierarchically nested 
    sub-basins at different scales, supported by Pfafstetter coding for catchment topology analysis.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire dataset is returned.
    regions : str or list, optional
        Regions to download. Can be 'all' or list of region names. Valid regions are:
        'Africa', 'Arctic', 'Asia', 'Australia', 'Europe', 'Greenland', 'North America',
        'South America', 'Siberia'. Default is 'all'.
    level : int, optional
        The hierarchical level (1-12) of sub-basin delineation. Higher levels represent
        finer subdivisions. Default is 4.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the HydroBASINS sub-basin boundaries with associated attributes.

    Examples
    --------
    Get level 4 sub-basins for all regions...
    
    >>> basins = get_hydrobasins()
    >>> basins.plot()

    Get level 6 sub-basins for North America...
    
    >>> na_basins = get_hydrobasins(regions=['North America'], level=6)
    >>> na_basins.plot()

    Notes
    -----
    Data citation:
    Lehner, B., Grill G. (2013). Global river hydrography and network routing: baseline data 
    and new approaches to study the world's large river systems. Hydrological Processes, 
    27(15): 2171–2186. https://doi.org/10.1002/hyp.9740
    """
    
    HYDROBASINS_LEVELS = {
        1: 'lev01', 2: 'lev02', 3: 'lev03', 4: 'lev04', 
        5: 'lev05', 6: 'lev06', 7: 'lev07', 8: 'lev08',
        9: 'lev09', 10: 'lev10', 11: 'lev11', 12: 'lev12'
    }

    HYDROBASINS_REGIONS = {
        'Africa': 'af', 'Arctic': 'ar', 'Asia': 'as', 
        'Australia': 'au', 'Europe': 'eu', 'Greenland': 'gr',
        'North America': 'na', 'South America': 'sa', 'Siberia': 'si'
    }

    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None

    # Handle regions parameter
    if regions == 'all':
        regions_to_process = HYDROBASINS_REGIONS
    else:
        regions_to_process = {region: HYDROBASINS_REGIONS[region] for region in regions}

    level_code = HYDROBASINS_LEVELS[level]
    region_gdfs = []

    for region_name, region_code in regions_to_process.items():
        
        print(f'Getting geometries for {region_name}...')
        url = f"https://data.hydrosheds.org/file/hydrobasins/standard/hybas_{region_code}_lev01-12_v1c.zip"
        
        if region_name == 'Africa': 
            # Special handling for Africa due to streaming issues
            print("Africa takes a bit longer because we have to temporarily save the file due to read issue...")
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, f"hybas_{region_code}_lev01-12_v1c.zip")
                
                response = requests.get(url, stream=True)
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                shp_path = os.path.join(temp_dir, f'hybas_{region_code}_{level_code}_v1c.shp')
                region_gdf = gpd.read_file(shp_path)
            
        else:
            
            region_gdf = gpd.read_file("zip+" + url, layer=f'hybas_{region_code}_{level_code}_v1c')
        
        region_gdfs.append(region_gdf)

    basins_gdf = pd.concat(region_gdfs)
    
    # Clip to bbox if provided
    if bbox_gdf is not None:
        basins_gdf = basins_gdf.clip(bbox_gdf)

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = "Lehner, B., Grill G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. Hydrological Processes, 27(15): 2171–2186. https://doi.org/10.1002/hyp.9740"

    return basins_gdf


def get_grdc_major_river_basins_of_the_world(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """
    Retrieves GRDC Major River Basins of the World dataset.

    This function downloads and loads the Global Runoff Data Centre's (GRDC) Major River Basins 
    dataset, which contains 520 river/lake basins considered major in size or hydro-political 
    importance. The basins include both exorheic drainage (flowing to oceans) and endorheic 
    drainage (inland sinks/lakes) systems.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the GRDC major river basins with associated attributes.

    Examples
    --------
    Get all major river basins...

    >>> basins = get_grdc_basins()
    >>> basins.plot()

    Get basins for a specific region...

    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> regional_basins = get_grdc_basins(bbox_input=bbox)
    >>> regional_basins.plot()

    Notes
    -----
    This dataset incorporates data from HydroSHEDS database which is © World Wildlife Fund, Inc. 
    (2006-2013) and has been used under license.

    Data citation:
    GRDC (2020): GRDC Major River Basins. Global Runoff Data Centre. 2nd, rev. ed. 
    Koblenz: Federal Institute of Hydrology (BfG).
    """
    
    url = "https://datacatalogfiles.worldbank.org/ddh-published/0041426/DR0051689/major_basins_of_the_world_0_0_0.zip"
    
    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    
    # Load the data
    basins_gdf = gpd.read_file("zip+" + url)
    
    # Clip to bbox if provided
    if bbox_gdf is not None:
        basins_gdf = basins_gdf.clip(bbox_gdf)
    else:
        print("No spatial subsetting because bbox_input was not provided.")

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = "GRDC (2020): GRDC Major River Basins. Global Runoff Data Centre. 2nd, rev. ed. Koblenz: Federal Institute of Hydrology (BfG)."
    
    return basins_gdf


def get_grdc_wmo_basins(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """
    Retrieves WMO Basins and Sub-Basins dataset.

    This function downloads and loads the Global Runoff Data Centre's (GRDC) WMO Basins 
    and Sub-Basins dataset. It contains 515 WMO Basins representing hydrographic regions
    including river/lake basins with both exorheic drainage (flowing to oceans) and 
    endorheic drainage (inland sinks/lakes).

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the WMO Basins and Sub-Basins with associated attributes.

    Examples
    --------
    Get all WMO basins...

    >>> basins = get_wmo_basins_and_subbasins()
    >>> basins.plot()

    Get basins for a specific region...

    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> regional_basins = get_wmo_basins_and_subbasins(bbox_input=bbox)
    >>> regional_basins.plot()

    Notes
    -----
    This dataset incorporates data from the HydroSHEDS database which is © World Wildlife Fund, Inc. 
    (2006-2013) and has been used under license.

    WMO basins and sub-basins are attributed with:
    - WMOBB: identifier of hydrographic region
    - WMOBB_NAME: name of hydrographic region
    - WMOBB_BASIN: name of river/lake basin, coastal region or island
    - WMOBB_SUBBASIN: name of river/lake basin forming a separate sub-basin
    - WMOBB_DESCRIPTION: description of hydrographic region
    - REGNUM: number of the WMO Region (Regional Association)
    - REGNAME: name of the WMO Region (Regional Association)
    - WMO306_MoC_NUM: reference to Manual on Codes, 2-digit basin code
    - WMO306_MoC_REFERENCE: reference to Manual on Codes, name of basin/sub-basin
    - SUMSUBAREA: approximate of drainage area (in square km)

    Data citation:
    GRDC (2020): WMO Basins and Sub-Basins / Global Runoff Data Centre, GRDC. 3rd, rev. ext. ed. 
    Koblenz, Germany: Federal Institute of Hydrology (BfG).
    """
    
    url = "https://grdc.bafg.de/downloads/wmobb_json.zip/wmobb_basins.json"
    
    # Convert bbox to GeoDataFrame if provided
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    
    basins_gdf = gpd.read_file("zip+" + url)
    
    # Clip to bbox if provided
    if bbox_gdf is not None:
        basins_gdf = basins_gdf.clip(bbox_gdf)
    else:
        print("No spatial subsetting because bbox_input was not provided.")

    # Add citation to attributes
    basins_gdf.attrs["data_citation"] = "GRDC (2020): WMO Basins and Sub-Basins / Global Runoff Data Centre, GRDC. 3rd, rev. ext. ed. Koblenz, Germany: Federal Institute of Hydrology (BfG)."
    
    return basins_gdf

def get_era5(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    version: str = "ERA5",
    cadence: str = "HOURLY",
    source: str = "auto",  # "auto", "GEE", or "GCS"
    start_date: str | None = None,
    end_date: str | None = None,
    variables: str | list | None = None,
    initialize_ee: bool = True,
) -> xr.Dataset:
    """
    Retrieves ERA5 reanalysis data using optimal source selection.
    
    By default, this function uses Google Earth Engine for most requests, but automatically 
    switches to the high-resolution ARCO-ERA5 Zarr dataset from Google Cloud Storage for
    hourly ERA5 data due to its superior performance and coverage for that specific 
    combination. Please note, these datasets may be different from the original ERA5 data
    hosted on the Copernicus Climate Data Store (CDS).
    
    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame or tuple or shapely.Geometry, optional
        The spatial bounding box for subsetting. If None, returns global data.
    version : str, optional
        Version of ERA5 data. Options are 'ERA5' or 'ERA5_LAND'. Default is 'ERA5'.
    cadence : str, optional
        Temporal resolution. Options are 'HOURLY', 'DAILY', or 'MONTHLY'. Default is 'HOURLY'.
    source : str, optional
        Data source to use: "auto" (smart selection), "GEE" (Google Earth Engine), or 
        "GCS" (Google Cloud Storage). Default is "auto", which uses GCS for ERA5 hourly data
        and GEE for everything else.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, uses earliest available date.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses latest available date.
    variables : str or list, optional
        Variable(s) to select. If None, returns all variables. Only applicable for GEE source.
    initialize_ee : bool, optional
        Whether to initialize Earth Engine. Default is True. Only applicable for GEE source.
        
    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing ERA5 reanalysis data for the specified region.
        
    Examples
    --------
    Get hourly ERA5 data (automatically uses ARCO-ERA5 from GCS):
    
    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>> era5_ds = get_era5(bbox_input=bbox)  # Uses GCS for hourly ERA5
    >>> era5_ds["2m_temperature"].sel(time="2020-05-26").mean(dim="time").plot()
    
    Get monthly ERA5 data (uses Google Earth Engine):
    
    >>> era5_gee = get_era5(
    ...     bbox_input=bbox,
    ...     cadence="MONTHLY", 
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     variables=["temperature_2m"]
    ... )  # Uses GEE for monthly data
    >>> era5_gee["temperature_2m"].plot()
    
    Force using GEE for hourly ERA5 data:
    
    >>> era5_hourly_gee = get_era5(
    ...     bbox_input=bbox,
    ...     source="GEE",
    ...     start_date="2020-01-01",
    ...     end_date="2020-01-02"
    ... )  # Explicitly uses GEE for hourly data
    
    Notes
    -----
    - The function automatically selects the optimal data source based on your request
    - Hourly ERA5 data comes from ARCO-ERA5 on Google Cloud Storage by default
    - All other combinations use Google Earth Engine
    - You can override the automatic source selection by explicitly setting the source parameter
    - Please note, these data are not the original ERA5 data but have been processed and optimized for cloud access. Each dataset will also have an assosciated latency different from the original dataset. The most up-to-date information can be found at: https://cds.climate.copernicus.eu/datasets

    
    Data citations:
    - GEE+GCS: Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
    - GCS: Carver, Robert W, and Merose, Alex. (2023): ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset. 22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1, https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842
    """
    # Determine the appropriate source based on parameters
    effective_source = source.upper()
    
    if effective_source == "AUTO":
        if version == "ERA5" and cadence == "HOURLY":
            effective_source = "GCS"  # Use ARCO dataset for hourly ERA5
        else:
            effective_source = "GEE"  # Default to GEE for all other combinations
    
    # Convert bbox to GeoDataFrame format for consistent handling
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input) if bbox_input is not None else None
    
    # Option 1: Google Cloud Storage (GCS) - ARCO-ERA5 Zarr dataset
    if effective_source == "GCS":
        # Verify we're using ERA5 hourly (the only supported option for GCS)
        if version != "ERA5" or cadence != "HOURLY":
            raise ValueError(f"GCS source only supports ERA5 hourly data, not {version} {cadence}")
            
        era5_ds = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
            chunks=None,
            storage_options=dict(token='anon'),
        )
        
        # Apply time filtering if specified
        if start_date is not None and end_date is not None:
            era5_ds = era5_ds.sel(time=slice(start_date, end_date))
        else:
            era5_ds = era5_ds.sel(time=slice(era5_ds.attrs['valid_time_start'], 
                                            era5_ds.attrs['valid_time_stop']))
            
        # Set CRS and normalize longitude coordinates
        era5_ds.rio.write_crs("EPSG:4326", inplace=True)
        era5_ds = era5_ds.assign_coords(
            longitude=(((era5_ds.longitude + 180) % 360) - 180)
        ).sortby('longitude')
        
        # Add coordinate attributes
        era5_ds["longitude"].attrs["long_name"] = "longitude"
        era5_ds["longitude"].attrs["units"] = "degrees_east"
        
        # Apply spatial subsetting if specified
        if bbox_gdf is not None:
            era5_ds = era5_ds.rio.clip_box(*bbox_gdf.total_bounds, crs=bbox_gdf.crs)
            
        # Add metadata
        era5_ds.attrs["data_citation"] = (
            "Carver, Robert W, and Merose, Alex. (2023): ARCO-ERA5: An Analysis-Ready "
            "Cloud-Optimized Reanalysis Dataset. 22nd Conf. on AI for Env. Science, "
            "Denver, CO, Amer. Meteo. Soc, 4A.1, "
            "https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842"
        )
        era5_ds.attrs["source"] = "Google Cloud Storage (ARCO-ERA5)"
        era5_ds.attrs["version"] = version
        era5_ds.attrs["cadence"] = cadence
        
        return era5_ds
        
    # Option 2: Google Earth Engine (GEE)
    elif effective_source == "GEE":
        # Initialize Earth Engine if requested
        if initialize_ee:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            print("Earth Engine initialization skipped. Please ensure EE is initialized.")
        
        # Collection name mapping
        collection_mapping = {
            ("ERA5_LAND", "HOURLY"): "ECMWF/ERA5_LAND/HOURLY",
            ("ERA5_LAND", "DAILY"): "ECMWF/ERA5_LAND/DAILY_AGGR",
            ("ERA5_LAND", "MONTHLY"): "ECMWF/ERA5_LAND/MONTHLY_AGGR",
            ("ERA5", "HOURLY"): "ECMWF/ERA5/HOURLY",
            ("ERA5", "DAILY"): "ECMWF/ERA5/DAILY",
            ("ERA5", "MONTHLY"): "ECMWF/ERA5/MONTHLY"
        }
        
        # Get collection name
        collection_key = (version, cadence)
        if collection_key not in collection_mapping:
            raise ValueError(f"Invalid combination of version '{version}' and cadence '{cadence}'")
        
        collection_name = collection_mapping[collection_key]
        
        # Initialize image collection
        image_collection = ee.ImageCollection(collection_name)
        
        # Apply date filtering if specified
        if start_date is not None and end_date is not None:
            end_date = end_date + "T23:59:59"  # Include full end date
            image_collection = image_collection.filterDate(start_date, end_date)
        
        # Apply variable selection if specified
        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]
            image_collection = image_collection.select(variables)
        
        # Get projection from first image
        image = image_collection.first()
        projection = image.select(0).projection()
        
        # Prepare geometry for GEE
        geometry = None
        if bbox_gdf is not None:
            geometry = tuple(bbox_gdf.total_bounds)
        
        # Load dataset
        ds = xr.open_dataset(
            image_collection,
            engine='ee',
            geometry=geometry,
            projection=projection,
            chunks=None
        )
        
        # Clean up dimensions and coordinate names
        ds = (ds
              .transpose('time', 'lat', 'lon')
              .rename({'lat': 'latitude', 'lon': 'longitude'})
              .rio.set_spatial_dims(x_dim='longitude', y_dim='latitude'))
        
        # Add metadata
        ds.attrs['data_citation'] = (
            "Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. "
            "Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049."
        )
        ds.attrs['version'] = version
        ds.attrs['cadence'] = cadence
        ds.attrs["source"] = "Google Earth Engine"
        
        return ds
        
    else:
        raise ValueError("Source must be 'auto', 'GEE' (Google Earth Engine), or 'GCS' (Google Cloud Storage)")


def get_ucla_snow_reanalysis(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
                             variable: str = 'SWE_Post',
                             stats: str = 'mean',
                             start_date: str = '1984-10-01',
                             end_date: str = '2021-09-30',
) -> xr.DataArray:
    """
    Fetches the Margulis UCLA snow reanalysis product for a specified bounding box and time range.

    This function retrieves snow reanalysis data from the UCLA dataset, allowing users to specify
    the type of snow data variable, statistical measure, and the temporal range for the data retrieval.
    The data is then clipped to the specified bounding box and returned as an xarray DataArray.

    Parameters
    ----------
    bbox_input : geopandas.GeoDataFrame, tuple, or Shapely Geometry, optional
        The bounding box for spatial subsetting. If None, the entire dataset is returned.
    variable : str, optional
        The type of snow data variable to retrieve. Options include 'SWE_Post' (Snow Water Equivalent),
        'SCA_Post' (Snow Cover Area), and 'SD_Post' (Snow Depth). Default is 'SWE_Post'.
    stats : str, optional
        The ensemble statistic. Options are 'mean', 'std' (standard deviation),
        'median', '25pct' (25th percentile), and '75pct' (75th percentile). Default is 'mean'.
    start_date : str, optional
        The start date for the data retrieval in 'YYYY-MM-DD' format. Default is '1984-10-01'.
    end_date : str, optional
        The end date for the data retrieval in 'YYYY-MM-DD' format. Default is '2021-09-30'.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the requested snow reanalysis data, clipped to the specified bounding box.

    Examples
    --------
    Get mean Snow Water Equivalent data for a specific region and time period...

    >>> swe_reanalysis_da = easysnowdata.hydroclimatology.get_ucla_snow_reanalysis(bbox_input=(-121.94, 46.72, -121.54, 46.99), 
    ...                                     variable='SWE_Post', 
    ...                                     start_date='2000-01-01', 
    ...                                     end_date='2000-12-31')
    >>> snow_reanalysis_da.isel(time=slice(0, 365, 30)).plot.imshow(col="time",col_wrap=5,cmap="Blues",vmin=0,vmax=3)

    Notes
    -----
    Data citation:

    Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2
    """

    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    search = earthaccess.search_data(
                short_name="WUS_UCLA_SR",
                cloud_hosted=True,
                bounding_box=tuple(bbox_gdf.total_bounds),
                temporal=(start_date, end_date),
            )
    
    files = earthaccess.open(search) # cant disable progress bar yet https://github.com/nsidc/earthaccess/issues/612
    snow_reanalysis_ds = xr.open_mfdataset(files).transpose()

    url = files[0].path
    date_pattern = r'\d{4}\.\d{2}\.\d{2}'
    WY_start_date = pd.to_datetime(re.search(date_pattern, url).group())

    snow_reanalysis_ds.coords['time'] = ("Day", pd.date_range(WY_start_date, periods=snow_reanalysis_ds.sizes['Day']))
    snow_reanalysis_ds = snow_reanalysis_ds.swap_dims({'Day':'time'})

    snow_reanalysis_ds = snow_reanalysis_ds.sel(time=slice(start_date, end_date))

    stats_dictionary = {'mean':0, 'std':1, 'median':2, '25pct':2, '75pct':3}
    stats_index = stats_dictionary[stats]

    snow_reanalysis_da = snow_reanalysis_ds[variable].sel(Stats=stats_index)
    snow_reanalysis_da = snow_reanalysis_da.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude")
    snow_reanalysis_da = snow_reanalysis_da.rio.write_crs(bbox_gdf.crs)
    snow_reanalysis_da = snow_reanalysis_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs)


    snow_reanalysis_da.attrs["data_citation"] = "Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis. (WUS_UCLA_SR, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PP7T2GBI52I2"
    
    return snow_reanalysis_da


def get_koppen_geiger_classes(
        bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
        resolution: str = "0.1 degree",
) -> xr.DataArray:
    """
    Retrieves Köppen-Geiger climate classification data for a given bounding box and resolution.

    This function fetches global Köppen-Geiger climate classification data from a high-resolution dataset
    based on constrained CMIP6 projections. It allows for optional spatial subsetting and provides
    multiple resolution options. The returned DataArray includes a custom plotting function as an attribute.

    Parameters
    ----------
    bbox_input:
        The bounding box for spatial subsetting. If None, the entire global dataset is returned.
    resolution:
        The spatial resolution of the data. Options are "1 degree", "0.5 degree", "0.1 degree", or "1 km".
        Default is "0.1 degree".

    Returns
    -------
    xarray.DataArray
        A DataArray containing the Köppen-Geiger climate classification data, with class information,
        color map, data citation, and a custom plotting function included as attributes.

    Examples
    --------
    Get Köppen-Geiger climate classification data for the entire globe with a 1-degree resolution, use custom plotting function:
    >>> koppen_data = get_koppen_geiger_classes(bbox_input=None, resolution="1 degree")
    >>> koppen_data.attrs['example_plot'](koppen_data)
    Get Köppen-Geiger climate classification data for a specific region with a 1 km resolution, plot using xarray's built-in plotting function
    >>> koppen_geiger_da = get_koppen_geiger_classes(bbox_input=(-121.94224976, 46.72842173, -121.54136001, 46.99728203), resolution="1 km")
    >>> koppen_data.plot(cmap=koppen_data.attrs["cmap"])

    Notes
    -----
    Data citation:

    Beck, H.E., McVicar, T.R., Vergopolan, N. et al. High-resolution (1 km) Köppen-Geiger maps
    for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023).
    https://doi.org/10.1038/s41597-023-02549-6
    """

    def get_class_info():
        classes = {
            1: {"name": "Af", "description": "Tropical, rainforest", "color": [0, 0, 255]},
            2: {"name": "Am", "description": "Tropical, monsoon", "color": [0, 120, 255]},
            3: {"name": "Aw", "description": "Tropical, savannah", "color": [70, 170, 250]},
            4: {"name": "BWh", "description": "Arid, desert, hot", "color": [255, 0, 0]},
            5: {"name": "BWk", "description": "Arid, desert, cold", "color": [255, 150, 150]},
            6: {"name": "BSh", "description": "Arid, steppe, hot", "color": [245, 165, 0]},
            7: {"name": "BSk", "description": "Arid, steppe, cold", "color": [255, 220, 100]},
            8: {"name": "Csa", "description": "Temperate, dry summer, hot summer", "color": [255, 255, 0]},
            9: {"name": "Csb", "description": "Temperate, dry summer, warm summer", "color": [200, 200, 0]},
            10: {"name": "Csc", "description": "Temperate, dry summer, cold summer", "color": [150, 150, 0]},
            11: {"name": "Cwa", "description": "Temperate, dry winter, hot summer", "color": [150, 255, 150]},
            12: {"name": "Cwb", "description": "Temperate, dry winter, warm summer", "color": [100, 200, 100]},
            13: {"name": "Cwc", "description": "Temperate, dry winter, cold summer", "color": [50, 150, 50]},
            14: {"name": "Cfa", "description": "Temperate, no dry season, hot summer", "color": [200, 255, 80]},
            15: {"name": "Cfb", "description": "Temperate, no dry season, warm summer", "color": [100, 255, 80]},
            16: {"name": "Cfc", "description": "Temperate, no dry season, cold summer", "color": [50, 200, 0]},
            17: {"name": "Dsa", "description": "Cold, dry summer, hot summer", "color": [255, 0, 255]},
            18: {"name": "Dsb", "description": "Cold, dry summer, warm summer", "color": [200, 0, 200]},
            19: {"name": "Dsc", "description": "Cold, dry summer, cold summer", "color": [150, 50, 150]},
            20: {"name": "Dsd", "description": "Cold, dry summer, very cold winter", "color": [150, 100, 150]},
            21: {"name": "Dwa", "description": "Cold, dry winter, hot summer", "color": [170, 175, 255]},
            22: {"name": "Dwb", "description": "Cold, dry winter, warm summer", "color": [90, 120, 220]},
            23: {"name": "Dwc", "description": "Cold, dry winter, cold summer", "color": [75, 80, 180]},
            24: {"name": "Dwd", "description": "Cold, dry winter, very cold winter", "color": [50, 0, 135]},
            25: {"name": "Dfa", "description": "Cold, no dry season, hot summer", "color": [0, 255, 255]},
            26: {"name": "Dfb", "description": "Cold, no dry season, warm summer", "color": [55, 200, 255]},
            27: {"name": "Dfc", "description": "Cold, no dry season, cold summer", "color": [0, 125, 125]},
            28: {"name": "Dfd", "description": "Cold, no dry season, very cold winter", "color": [0, 70, 95]},
            29: {"name": "ET", "description": "Polar, tundra", "color": [178, 178, 178]},
            30: {"name": "EF", "description": "Polar, frost", "color": [102, 102, 102]}
        }
        return classes


    def get_class_cmap(classes):
        colors = {k: [c/255 for c in v["color"]] for k, v in classes.items()}
        return matplotlib.colors.ListedColormap([colors[i] for i in range(1, 31)])
    

    def plot_classes(self, ax=None, figsize=(8, 10), cbar_orientation='horizontal'):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        else:
            f = ax.get_figure()

        bounds = np.arange(0.5, 31.5, 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, self.attrs["cmap"].N)

        im = self.plot(ax=ax, cmap=self.attrs["cmap"], norm=norm, add_colorbar=False)

        ax.set_aspect("equal")

        cbar = f.colorbar(im, ax=ax, orientation=cbar_orientation, aspect=30, pad=0.08)

        cbar.set_ticks(np.arange(1, 31))
        cbar.set_ticklabels([f"{v['name']}: {v['description']}" for k, v in self.attrs["class_info"].items()], fontsize=8)

        if cbar_orientation == 'horizontal':
            plt.setp(cbar.ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
        else:
            plt.setp(cbar.ax.get_yticklabels(), rotation=0, ha='right')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Köppen-Geiger climate classification")
        f.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)

        return f, ax
    
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    resolution_dict = {"1 degree": "1p0", "0.5 degree": "0p5", "0.1 degree": "0p1", "1 km": "0p00833333"}
    resolution = resolution_dict[resolution]

    koppen_geiger_da = rxr.open_rasterio(f"zip+https://figshare.com/ndownloader/files/45057352/koppen_geiger_tif.zip/1991_2020/koppen_geiger_{resolution}.tif").squeeze()

    koppen_geiger_da = koppen_geiger_da.rio.clip_box(*bbox_gdf.total_bounds,crs=bbox_gdf.crs)
        

    koppen_geiger_da.attrs["class_info"] = get_class_info()
    koppen_geiger_da.attrs["cmap"] = get_class_cmap(koppen_geiger_da.attrs["class_info"])
    koppen_geiger_da.attrs["data_citation"] = "Beck, H.E., McVicar, T.R., Vergopolan, N. et al. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023). https://doi.org/10.1038/s41597-023-02549-6"

    koppen_geiger_da.attrs['example_plot'] = plot_classes

    return koppen_geiger_da




















# huc map, from gee?

# hydroatlas? https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroATLAS_v1_Basins_level03

# maybe seperate climate module,or use https://github.com/hyriver/pydaymet

# https://github.com/OpenTopography/OT_3DEP_Workflows/blob/main/notebooks/03_3DEP_Generate_DEM_USGS_HUCs.ipynb
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
