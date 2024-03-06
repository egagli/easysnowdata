import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import shapely
import dask

from easysnowdata.utils import convert_bbox_to_geodataframe




def get_forest_cover_fraction(bbox_input) -> xr.DataArray:
    """
    Fetches ~100m forest cover fraction (FCF) data for a given bounding box.

    The data is obtained from the Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe dataset, available at https://zenodo.org/records/3939050. The specific layer used is the Tree-CoverFraction-layer, which provides the fractional cover (%) for the forest class.

    Citation:
    Marcel Buchhorn, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, & Steffen Fritz. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3939050

    Parameters:
    bbox_input (GeoDataFrame or tuple or Shapely geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.

    Returns:
    DataArray: FCF data.
    """    
    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    fcf = rxr.open_rasterio('https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif', chunks=True, mask_and_scale=True)
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    fcf = fcf.rio.clip_box(xmin, ymin, xmax, ymax).squeeze()
    
    return fcf

def get_seasonal_snow_classification(bbox_input, return_classes_dict=False) -> xr.DataArray:
    """
    Fetches 10arcsec (~300m) Sturm & Liston 2021 seasonal snow classification data for a given bounding box.

    "This data set consists of global, seasonal snow classifications—e.g., tundra, boreal forest, maritime, ephemeral, prairie, montane forest, and ice—determined from air temperature, precipitation, and wind speed climatologies." This is the 10 arcsec (~300m) product in EPSG:4326. The data is available on NSIDC at http://dx.doi.org/10.5067/99FTCYYYLAQ0. 

    Citation:
    Liston, G. E. and M. Sturm. (2021). Global Seasonal-Snow Classification, Version 1 [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.5067/99FTCYYYLAQ0. Date Accessed 03-06-2024.

    Parameters:
    bbox_input (GeoDataFrame or tuple or Shapely geometry): GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    return_classes_dict (bool): If True, instead returns a dictionary of snow classes. Default is False.

    Returns:
    DataArray: Snow class data.
    """    
    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    snow_classification = rxr.open_rasterio('https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif', chunks=True, mask_and_scale=True)
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    snow_classification = snow_classification.rio.clip_box(xmin, ymin, xmax, ymax).squeeze().rio.write_nodata(9, encoded=True)
    
    if return_classes_dict:
        classes_dict = {
            1: "Tundra",
            2: "Boreal Forest",
            3: "Maritime",
            4: "Ephemeral (includes no snow)",
            5: "Prairie",
            6: "Montane Forest",
            7: "Ice (glaciers and ice sheets)",
            8: "Ocean",
            9: "Fill"
        }
        return classes_dict

    return snow_classification


