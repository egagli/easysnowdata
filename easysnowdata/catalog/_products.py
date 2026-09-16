"""Catalog entries for the products easysnowdata serves today.

Phase 1 records what the existing modules do (locations, credentials, class
tables, citations, health probes) with ``loader`` pointing at the current
functions and classes. Phase 2 migrates each loader onto ``providers/`` and
moves its entry next to the new module; the ids and source ids here are the
contract that migration keeps.
"""

from __future__ import annotations

from functools import partial

from easysnowdata.auth.planetary_computer import STAC_URL as PC_STAC
from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable

EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1"
CMR_LPCLOUD_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"


def _flags(table: dict[int, tuple[str, str]]) -> dict[str, tuple]:
    """Turn ``{value: (meaning, color)}`` into Variable flag fields."""
    return {
        "flag_values": tuple(table),
        "flag_meanings": tuple(m for m, _ in table.values()),
        "flag_colors": tuple(c for _, c in table.values()),
    }


SCL_CLASSES = {
    0: ("No Data (Missing data)", "#000000"),
    1: ("Saturated or defective pixel", "#ff0000"),
    2: ("Topographic casted shadows", "#2f2f2f"),
    3: ("Cloud shadows", "#643200"),
    4: ("Vegetation", "#00a000"),
    5: ("Not-vegetated", "#ffe65a"),
    6: ("Water", "#0000ff"),
    7: ("Unclassified", "#808080"),
    8: ("Cloud medium probability", "#c0c0c0"),
    9: ("Cloud high probability", "#ffffff"),
    10: ("Thin cirrus", "#64c8ff"),
    11: ("Snow or ice", "#ff96ff"),
}

SNOW_CLASSIFICATION_CLASSES = {
    1: ("Tundra", "#a100c8"),
    2: ("Boreal Forest", "#00a0fe"),
    3: ("Maritime", "#fe0000"),
    4: ("Ephemeral (includes no snow)", "#e7dc32"),
    5: ("Prairie", "#f08328"),
    6: ("Montane Forest", "#00dc00"),
    7: ("Ice (glaciers and ice sheets)", "#aaaaaa"),
    8: ("Ocean", "#0000ff"),
    9: ("Fill", "#ffffff"),
}

MOUNTAIN_SNOW_CLASSES = {
    0: ("Mountains with little-to-no snow", "#030303"),
    1: ("Indeterminate due to clouds", "#755F4A"),
    2: ("Mountains with ephemeral snow", "#792B8E"),
    3: ("Mountains with seasonal snow", "#679ACF"),
    255: ("Fill", "#ffffff"),
}

SNOW_CLASSES = {
    0: ("Little-to-no snow", "#030303"),
    1: ("Indeterminate due to clouds", "#755F4A"),
    2: ("Ephemeral snow", "#792B8E"),
    3: ("Seasonal snow", "#679ACF"),
    255: ("Fill", "#ffffff"),
}

WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#FFBB22"),
    30: ("Grassland", "#FFFF4C"),
    40: ("Cropland", "#F096FF"),
    50: ("Built-up", "#FA0000"),
    60: ("Bare / sparse vegetation", "#B4B4B4"),
    70: ("Snow and ice", "#F0F0F0"),
    80: ("Permanent water bodies", "#0064C8"),
    90: ("Herbaceous wetland", "#0096A0"),
    95: ("Mangroves", "#00CF75"),
    100: ("Moss and lichen", "#FAE6A0"),
}

MOD10A2_CLASSES = {
    0: ("missing data", "#006400"),
    1: ("no decision", "#FFBB22"),
    11: ("night", "#FFFF4C"),
    25: ("no snow", "#F096FF"),
    37: ("lake", "#FA0000"),
    39: ("ocean / sparse vegetation", "#B4B4B4"),
    50: ("cloud", "#F0F0F0"),
    100: ("lake ice", "#0064C8"),
    200: ("snow", "#0096A0"),
    254: ("detector saturated", "#00CF75"),
    255: ("fill", "#FAE6A0"),
}

_GITHUB_STATIONS = "https://github.com/egagli/snotel_ccss_stations"
_URL_HYDROATLAS = (
    "https://ndownloader.figshare.com/files/20082137/BasinATLAS_Data_v10.gdb.zip"
)
_URL_GRDC_MAJOR = "https://datacatalogfiles.worldbank.org/ddh-published/0041426/DR0051689/major_basins_of_the_world_0_0_0.zip"
_URL_GRDC_WMO = "https://grdc.bafg.de/downloads/wmobb_json.zip"
_URL_KOPPEN = "https://ndownloader.figshare.com/files/61012822/koppen_geiger_tif.zip"
_URL_SNOWCLASS = "https://uwcryo.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
_URL_FOREST = "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif"
_URL_MTNSNOW = "https://zenodo.org/records/2626737/files/MODIS_mtnsnow_classes.zip"
_URL_ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_GEE_SNODAS = "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"


PRODUCTS: tuple[Product, ...] = (
    Product(
        id="snotel-ccss-stations",
        theme="stations",
        title="SNOTEL and CCSS station observations",
        description=(
            "Daily SWE, snow depth, precipitation and temperature from NRCS SNOTEL and "
            "California CCSS stations, served from the frozen egagli/snotel_ccss_stations "
            "archive. Superseded by the live network clients in Phase 3 (§9)."
        ),
        sources=(
            Source(
                id="github-archive",
                provider="vector_http",
                location=_GITHUB_STATIONS,
                extent="western US",
                temporal="varies by station / 2024",
                latency="frozen archive",
                notes="GeoJSON station list plus one CSV per station",
                title="GitHub archive",
                health=(
                    Probe(
                        "SNOTEL/CCSS station list (GitHub)",
                        partial(
                            health.http_first_byte,
                            f"{_GITHUB_STATIONS}/raw/main/all_stations.geojson",
                        ),
                    ),
                    Probe(
                        "SNOTEL/CCSS station CSV (GitHub)",
                        partial(
                            health.http_first_byte,
                            "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/679_WA_SNTL.csv",
                        ),
                    ),
                ),
            ),
        ),
        variables=(
            Variable("WTEQ", units="m", long_name="snow water equivalent"),
            Variable("SNWD", units="m", long_name="snow depth"),
            Variable("PRCPSA", units="m", long_name="precipitation accumulation"),
            Variable("TAVG", units="degC", long_name="mean daily air temperature"),
        ),
        citation="USDA NRCS National Water and Climate Center (SNOTEL); California Cooperative Snow Surveys (CCSS).",
        license="Public domain (US government data)",
        loader="easysnowdata.automatic_weather_stations.StationCollection",
        tags=("swe", "snow depth", "stations", "snotel"),
    ),
    Product(
        id="hydrobasins",
        theme="hydro",
        title="HydroBASINS / BasinATLAS",
        description="Global nested sub-basin polygons (Pfafstetter levels 1–12) with BasinATLAS attributes.",
        sources=(
            Source(
                id="figshare-basinatlas",
                provider="vector_http",
                location=_URL_HYDROATLAS,
                extent="global",
                temporal="static (v1.0, 2019)",
                notes="2.7 GB file geodatabase; read with mask= pushdown",
                title="figshare BasinATLAS gdb",
                health=Probe(
                    "HydroATLAS basins (figshare)",
                    partial(health.http_first_byte, _URL_HYDROATLAS),
                ),
            ),
        ),
        citation=(
            "Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental "
            "sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6, 283."
        ),
        license="CC BY 4.0",
        doi="10.1038/s41597-019-0300-6",
        loader="easysnowdata.hydroclimatology.get_hydroBASINS",
        tags=("basins", "watersheds"),
    ),
    Product(
        id="grdc-major-river-basins",
        theme="hydro",
        title="GRDC major river basins of the world",
        description="The 405 major river basins of the world (GRDC, 2020 edition via the World Bank data catalog).",
        sources=(
            Source(
                id="world-bank",
                provider="vector_http",
                location=_URL_GRDC_MAJOR,
                temporal="static (2020)",
                title="World Bank zip",
                health=Probe(
                    "GRDC major river basins (World Bank)",
                    partial(health.http_first_byte, _URL_GRDC_MAJOR),
                ),
            ),
        ),
        citation="GRDC (2020). Major River Basins of the World / Global Runoff Data Centre, 2nd rev. ext. ed. Koblenz: BfG.",
        license="Open (attribution)",
        loader="easysnowdata.hydroclimatology.get_grdc_major_river_basins_of_the_world",
        tags=("basins",),
    ),
    Product(
        id="grdc-wmo-basins",
        theme="hydro",
        title="GRDC / WMO basins and sub-basins",
        description="WMO basins and sub-basins (GRDC, 3rd ed.), fetched once into the cache because the server rejects HEAD.",
        sources=(
            Source(
                id="grdc",
                provider="vector_http",
                location=_URL_GRDC_WMO,
                temporal="static (2020)",
                notes="grdc.bafg.de answers HEAD with 400; GET-first",
                title="GRDC zip",
                health=Probe(
                    "GRDC WMO basins", partial(health.http_first_byte, _URL_GRDC_WMO)
                ),
            ),
        ),
        citation="GRDC (2020). WMO Basins and Sub-Basins / Global Runoff Data Centre, 3rd rev. ext. ed. Koblenz: BfG.",
        license="Open (attribution)",
        loader="easysnowdata.hydroclimatology.get_grdc_wmo_basins",
        tags=("basins",),
    ),
    Product(
        id="huc",
        theme="hydro",
        title="USGS Watershed Boundary Dataset (HUC)",
        description="Hydrologic unit boundaries (HUC2–HUC12) for the United States.",
        sources=(
            Source(
                id="gee",
                provider="gee",
                location="USGS/WBD/2017/HUC{level}",
                requires=("earthengine",),
                extent="United States",
                temporal="static (2017)",
                notes="credential-free WBD REST route planned (§4)",
                title="Earth Engine",
                health=Probe(
                    "HUC geometries (GEE/USGS WBD)",
                    partial(
                        health.gee_asset, "USGS/WBD/2017/HUC08", "feature_collection"
                    ),
                ),
            ),
        ),
        citation="U.S. Geological Survey and U.S. Department of Agriculture, Natural Resources Conservation Service. Watershed Boundary Dataset.",
        license="Public domain (US government data)",
        loader="easysnowdata.hydroclimatology.get_huc_geometries",
        tags=("basins", "watersheds", "huc"),
    ),
    Product(
        id="era5",
        theme="climate",
        title="ERA5 / ERA5-Land reanalysis",
        description="ECMWF ERA5 hourly (ARCO-ERA5 on GCS) and ERA5 / ERA5-Land aggregates (Earth Engine).",
        sources=(
            Source(
                id="arco-era5-gcs",
                provider="zarr_cloud",
                location=_URL_ARCO_ERA5,
                resolution_m=27750,
                temporal="1940/present (ERA5T)",
                latency="~5 days",
                notes="Zarr v2, consolidated, anonymous; hourly ERA5 only",
                title="ARCO-ERA5 (GCS)",
                health=Probe(
                    "ARCO-ERA5 (GCS anonymous)",
                    partial(health.zarr_metadata, _URL_ARCO_ERA5),
                ),
            ),
            Source(
                id="gee",
                provider="gee",
                location="ECMWF/ERA5_LAND/HOURLY (and ERA5/ERA5_LAND daily/monthly aggregates)",
                requires=("earthengine",),
                resolution_m=11132,
                temporal="1950/present",
                latency="~1 week",
                notes="the only ERA5-Land route today",
                title="Earth Engine",
                health=Probe(
                    "ERA5 (Google Earth Engine)",
                    partial(
                        health.gee_asset,
                        "ECMWF/ERA5_LAND/HOURLY",
                        start="2020-01-01",
                        end="2020-01-02",
                    ),
                ),
            ),
        ),
        citation="Hersbach, H., et al. (2020). The ERA5 global reanalysis. QJRMS 146, 1999–2049. Muñoz-Sabater, J., et al. (2021). ERA5-Land. ESSD 13, 4349–4383.",
        license="Copernicus licence",
        doi="10.1002/qj.3803",
        loader="easysnowdata.hydroclimatology.get_era5",
        tags=("reanalysis", "temperature", "precipitation", "snow depth"),
    ),
    Product(
        id="snodas",
        theme="snow",
        title="SNODAS snow water equivalent and snow depth",
        description="NOHRSC SNODAS daily 1 km SWE and snow depth over CONUS (Climate Engine re-hosting on Earth Engine).",
        sources=(
            Source(
                id="gee-climate-engine",
                provider="gee",
                location=_GEE_SNODAS,
                requires=("earthengine",),
                resolution_m=1000,
                extent="CONUS",
                temporal="2003-10/present",
                latency="~1 day",
                notes="credential-free NSIDC G02158 route is priority 2 for Phase 2",
                title="Earth Engine (Climate Engine)",
                health=Probe(
                    "SNODAS (GEE/Climate Engine)",
                    partial(
                        health.gee_asset,
                        _GEE_SNODAS,
                        start="2020-01-01",
                        end="2020-01-02",
                    ),
                ),
            ),
        ),
        variables=(
            Variable("SWE", units="mm", long_name="snow water equivalent"),
            Variable("Snow_Depth", units="mm", long_name="snow depth"),
        ),
        citation="National Operational Hydrologic Remote Sensing Center (2004). Snow Data Assimilation System (SNODAS) Data Products at NSIDC, Version 1.",
        license="Public domain (US government data)",
        doi="10.7265/N5TB14TC",
        loader="easysnowdata.hydroclimatology.get_snodas",
        tags=("swe", "snow depth"),
    ),
    Product(
        id="ucla-snow-reanalysis",
        theme="snow",
        title="UCLA Western US snow reanalysis",
        description="Daily 480 m posterior SWE and snow-covered area for the western US, water years 1985–2021 (WUS_UCLA_SR v1).",
        sources=(
            Source(
                id="nsidc",
                provider="earthdata",
                location="WUS_UCLA_SR",
                requires=("earthdata",),
                resolution_m=480,
                extent="western US",
                temporal="1984-10/2021-09",
                latency="static",
                notes="one NetCDF-4 per water year and tile; no DMR++ sidecars",
                title="NSIDC (Earthdata)",
                health=Probe(
                    "UCLA Snow Reanalysis (NASA NSIDC)",
                    partial(health.earthdata_search, "WUS_UCLA_SR"),
                ),
            ),
        ),
        variables=(
            Variable(
                "SWE_Post", units="m", long_name="posterior snow water equivalent"
            ),
            Variable(
                "SCA_Post", units="1", long_name="posterior snow-covered area fraction"
            ),
        ),
        citation="Fang, Y., Liu, Y. & Margulis, S. A. (2022). Western United States UCLA Daily Snow Reanalysis, Version 1. NSIDC.",
        license="NASA Earthdata (free registration)",
        doi="10.5067/PP7T2GBI52I2",
        loader="easysnowdata.hydroclimatology.get_ucla_snow_reanalysis",
        tags=("swe", "reanalysis"),
    ),
    Product(
        id="koppen-geiger",
        theme="climate",
        title="Köppen-Geiger climate classification (Beck et al. 2023)",
        description="Present-day and projected Köppen-Geiger classes at 1 km to 1°, from the 2023 update.",
        sources=(
            Source(
                id="figshare",
                provider="raster_http",
                location=_URL_KOPPEN,
                resolution_m=1000,
                temporal="1991-2020 (and other periods)",
                notes="file 61012822 is the current release; 45057352 was v1",
                title="figshare zip",
                health=Probe(
                    "Köppen-Geiger classification (figshare)",
                    partial(health.http_first_byte, _URL_KOPPEN),
                ),
            ),
        ),
        citation="Beck, H. E., et al. (2023). High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Scientific Data 10, 724.",
        license="CC BY 4.0",
        doi="10.1038/s41597-023-02549-6",
        loader="easysnowdata.hydroclimatology.get_koppen_geiger_classes",
        tags=("climate classification",),
    ),
    Product(
        id="forest-cover-fraction",
        theme="land",
        title="Forest cover fraction (CGLS-LC100 2019)",
        description="Copernicus Global Land Service 100 m tree-cover fraction, epoch 2019 (collection 3).",
        sources=(
            Source(
                id="zenodo",
                provider="raster_http",
                location=_URL_FOREST,
                resolution_m=100,
                temporal="static (2019)",
                title="Zenodo GeoTIFF",
                health=Probe(
                    "Forest cover fraction (Zenodo)",
                    partial(health.http_first_byte, _URL_FOREST),
                ),
            ),
        ),
        variables=(
            Variable("tree_cover_fraction", units="%", dtype="uint8", nodata=255),
        ),
        citation="Buchhorn, M., et al. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe (V3.0.1). Zenodo.",
        license="CC BY 4.0",
        doi="10.5281/zenodo.3939050",
        loader="easysnowdata.remote_sensing.get_forest_cover_fraction",
        tags=("forest", "land cover"),
    ),
    Product(
        id="snow-classification",
        theme="snow",
        title="Sturm & Liston seasonal snow classification",
        description="Global seasonal snow classes (tundra, boreal forest, maritime, ephemeral, prairie, montane forest, ice) at 10 arc-seconds.",
        sources=(
            Source(
                id="hosted-cog",
                provider="raster_http",
                location=_URL_SNOWCLASS,
                resolution_m=300,
                temporal="static (2021)",
                notes="COG on the uwcryo Azure blob; location likely to change (Zenodo/GitHub release). NSIDC-0768 with Earthdata Login is the planned default (§12 Q8)",
                title="hosted COG (Azure)",
                health=Probe(
                    "Sturm & Liston snow classification (Azure)",
                    partial(health.http_first_byte, _URL_SNOWCLASS),
                ),
            ),
        ),
        variables=(
            Variable(
                "snow_class",
                dtype="uint8",
                nodata=9,
                long_name="seasonal snow class",
                **_flags(SNOW_CLASSIFICATION_CLASSES),
            ),
        ),
        citation="Liston, G. E. and Sturm, M. (2021). Global Seasonal-Snow Classification, Version 1. NSIDC.",
        license="NASA Earthdata (free registration)",
        doi="10.5067/99FTCYYYLAQ0",
        loader="easysnowdata.remote_sensing.get_seasonal_snow_classification",
        tags=("snow class",),
    ),
    Product(
        id="mountain-snow-mask",
        theme="snow",
        title="Wrzesien global seasonal mountain snow mask",
        description="MODIS-derived masks of mountains with seasonal, ephemeral or little snow (Wrzesien et al. 2019), plus the all-terrain snow classes.",
        sources=(
            Source(
                id="zenodo",
                provider="raster_http",
                location=_URL_MTNSNOW,
                resolution_m=500,
                temporal="static (2000-2016 climatology)",
                notes="zipped GeoTIFF; nodata written as 256/265 upstream",
                title="Zenodo zip",
                health=Probe(
                    "Mountain snow mask (Zenodo)",
                    partial(health.http_first_byte, _URL_MTNSNOW),
                ),
            ),
        ),
        variables=(
            Variable(
                "mountain_snow",
                dtype="uint8",
                nodata=255,
                long_name="mountain snow class",
                **_flags(MOUNTAIN_SNOW_CLASSES),
            ),
            Variable(
                "snow",
                dtype="uint8",
                nodata=255,
                long_name="snow class",
                **_flags(SNOW_CLASSES),
            ),
        ),
        citation="Wrzesien, M., Pavelsky, T., Durand, M., Lundquist, J., & Dozier, J. (2019). Global Seasonal Mountain Snow Mask from MODIS MOD10A2. Zenodo.",
        license="CC BY 4.0",
        doi="10.5281/zenodo.2626737",
        loader="easysnowdata.remote_sensing.get_seasonal_mountain_snow_mask",
        tags=("snow mask", "mountains"),
    ),
    Product(
        id="esa-worldcover",
        theme="land",
        title="ESA WorldCover land cover",
        description="Global 10 m land cover for 2020 (v100) and 2021 (v200).",
        sources=(
            Source(
                id="planetary-computer",
                provider="stac",
                location="esa-worldcover",
                resolution_m=10,
                temporal="2020, 2021",
                notes="AWS esa-worldcover bucket is an unsigned alternative (§B.7)",
                title="Planetary Computer",
                health=Probe(
                    "ESA WorldCover (Planetary Computer)",
                    partial(health.stac_search, PC_STAC, "esa-worldcover", sign=True),
                ),
            ),
        ),
        variables=(
            Variable(
                "map",
                dtype="uint8",
                nodata=0,
                long_name="land cover class",
                **_flags(WORLDCOVER_CLASSES),
            ),
        ),
        citation="Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.",
        license="CC BY 4.0",
        doi="10.5281/zenodo.7254221",
        loader="easysnowdata.remote_sensing.get_esa_worldcover",
        tags=("land cover",),
    ),
    Product(
        id="nlcd",
        theme="land",
        title="National Land Cover Database (NLCD)",
        description="USGS NLCD 2021 release: land cover, impervious surface and science products for CONUS at 30 m.",
        sources=(
            Source(
                id="gee",
                provider="gee",
                location="USGS/NLCD_RELEASES/2021_REL/NLCD",
                requires=("earthengine",),
                resolution_m=30,
                extent="CONUS",
                temporal="2001-2021 epochs",
                notes="class tables come from the asset properties at run time",
                title="Earth Engine",
                health=Probe(
                    "NLCD (GEE/USGS)",
                    partial(health.gee_asset, "USGS/NLCD_RELEASES/2021_REL/NLCD"),
                ),
            ),
        ),
        citation="Dewitz, J. (2023). National Land Cover Database (NLCD) 2021 Products. U.S. Geological Survey data release.",
        license="Public domain (US government data)",
        doi="10.5066/P9JZ7AO3",
        loader="easysnowdata.remote_sensing.get_nlcd_landcover",
        tags=("land cover",),
    ),
    Product(
        id="copernicus-dem",
        theme="terrain",
        title="Copernicus DEM GLO-30 / GLO-90",
        description="Global 30 m and 90 m digital surface model from the Copernicus DEM (WorldDEM heritage).",
        sources=(
            Source(
                id="planetary-computer",
                provider="stac",
                location="cop-dem-glo-30 / cop-dem-glo-90",
                resolution_m=30,
                temporal="static (2021 release)",
                notes="AWS copernicus-dem-30m / Earth Search are unsigned alternatives; the 2024_1 release is only on GEE and CDSE (§B.8)",
                title="Planetary Computer",
                health=Probe(
                    "Copernicus DEM (Planetary Computer)",
                    partial(health.stac_search, PC_STAC, "cop-dem-glo-30", sign=True),
                ),
            ),
        ),
        variables=(
            Variable(
                "data", units="m", dtype="float32", nodata=-32767, long_name="elevation"
            ),
        ),
        citation="European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model. Distributed by OpenTopography.",
        license="Copernicus DEM licence (free, attribution)",
        doi="10.5069/G9028PQB",
        loader="easysnowdata.topography.get_copernicus_dem",
        tags=("dem", "elevation"),
    ),
    Product(
        id="chili",
        theme="terrain",
        title="CHILI (Continuous Heat-Insolation Load Index)",
        description="Topographic heat-load index from ALOS AW3D30 (Theobald et al. 2015), 90 m, 70°N–70°S.",
        sources=(
            Source(
                id="gee",
                provider="gee",
                location="CSP/ERGo/1_0/Global/ALOS_CHILI",
                requires=("earthengine",),
                resolution_m=90,
                extent="70°N–70°S",
                temporal="static",
                notes="a DEM-computed heat-load index is the planned credential-free route",
                title="Earth Engine",
                health=Probe(
                    "CHILI (GEE/CSP ERGo)",
                    partial(
                        health.gee_asset, "CSP/ERGo/1_0/Global/ALOS_CHILI", "image"
                    ),
                ),
            ),
        ),
        variables=(
            Variable(
                "constant", units="1", long_name="heat-insolation load index (0-1)"
            ),
        ),
        citation="Theobald, D. M., Harrison-Atlas, D., Monahan, W. B., Albano, C. M. (2015). Ecologically-Relevant Maps of Landforms and Physiographic Diversity for Climate Adaptation Planning. PLoS ONE 10(12): e0143619.",
        license="CC BY 4.0",
        doi="10.1371/journal.pone.0143619",
        loader="easysnowdata.topography.get_chili",
        tags=("terrain", "insolation"),
    ),
    Product(
        id="sentinel-2-l2a",
        theme="optical",
        title="Sentinel-2 Level-2A surface reflectance",
        description="Sentinel-2 MSI bottom-of-atmosphere reflectance with the Sen2Cor scene classification layer, 10–60 m.",
        sources=(
            Source(
                id="planetary-computer",
                provider="stac",
                location="sentinel-2-l2a",
                resolution_m=10,
                temporal="2015-06/present",
                latency="~1-2 days",
                notes="raw ESA values; the post-2022-01-25 baseline offset must be undone by the client",
                title="Planetary Computer",
                health=Probe(
                    "Sentinel-2 L2A (Planetary Computer)",
                    partial(
                        health.stac_search,
                        PC_STAC,
                        "sentinel-2-l2a",
                        datetime_range="2024-07-01/2024-07-31",
                        sign=True,
                    ),
                ),
            ),
            Source(
                id="earth-search",
                provider="stac",
                location="sentinel-2-l2a / sentinel-2-c1-l2a",
                resolution_m=10,
                temporal="2015-06/present",
                latency="~1 day",
                notes="items carry raster:bands scale/offset; Element 84 already harmonizes sentinel-2-l2a",
                title="Earth Search (AWS)",
                health=Probe(
                    "Sentinel-2 L2A (Earth Search)",
                    partial(
                        health.stac_search,
                        EARTH_SEARCH_STAC,
                        "sentinel-2-l2a",
                        datetime_range="2024-07-01/2024-07-31",
                    ),
                ),
            ),
        ),
        variables=(
            Variable("blue", units="1", dtype="uint16", nodata=0),
            Variable("green", units="1", dtype="uint16", nodata=0),
            Variable("red", units="1", dtype="uint16", nodata=0),
            Variable("nir", units="1", dtype="uint16", nodata=0),
            Variable("swir16", units="1", dtype="uint16", nodata=0),
            Variable("swir22", units="1", dtype="uint16", nodata=0),
            Variable(
                "scl",
                dtype="uint8",
                nodata=0,
                long_name="scene classification layer",
                **_flags(SCL_CLASSES),
            ),
        ),
        citation="European Space Agency. Copernicus Sentinel-2 MSI Level-2A products.",
        license="Copernicus Sentinel data licence",
        loader="easysnowdata.remote_sensing.Sentinel2",
        tags=("optical", "reflectance", "ndsi"),
    ),
    Product(
        id="sentinel-1-rtc",
        theme="sar",
        title="Sentinel-1 radiometrically terrain-corrected backscatter",
        description="Sentinel-1 C-band gamma0 backscatter (VV, VH) terrain-corrected with the Copernicus GLO-30 DEM, 10 m.",
        sources=(
            Source(
                id="planetary-computer",
                provider="stac",
                location="sentinel-1-rtc",
                resolution_m=10,
                temporal="2014-10/present",
                latency="~2 days",
                notes="OPERA RTC-S1 (EDL) and GEE OPERA/RTC/L2_V1/S1 are planned alternatives (§B.1)",
                title="Planetary Computer",
                health=Probe(
                    "Sentinel-1 RTC (Planetary Computer)",
                    partial(
                        health.stac_search,
                        PC_STAC,
                        "sentinel-1-rtc",
                        datetime_range="2024-07-01/2024-07-31",
                        sign=True,
                    ),
                ),
            ),
        ),
        variables=(
            Variable(
                "vv",
                units="linear power",
                dtype="float32",
                nodata=-32768,
                long_name="gamma0 VV",
            ),
            Variable(
                "vh",
                units="linear power",
                dtype="float32",
                nodata=-32768,
                long_name="gamma0 VH",
            ),
        ),
        citation="European Space Agency. Copernicus Sentinel-1 GRD; RTC processing by Catalyst for Microsoft Planetary Computer.",
        license="Copernicus Sentinel data licence",
        loader="easysnowdata.remote_sensing.Sentinel1",
        tags=("sar", "backscatter"),
    ),
    Product(
        id="hls",
        theme="optical",
        title="Harmonized Landsat Sentinel-2 (HLS) v2.0",
        description="NASA HLS L30 (Landsat 8/9) and S30 (Sentinel-2) 30 m surface reflectance with Fmask, harmonized to a common grid.",
        sources=(
            Source(
                id="lpcloud-cmr-stac",
                provider="stac",
                location="HLSL30_2.0 / HLSS30_2.0",
                requires=("earthdata",),
                resolution_m=30,
                temporal="2013-04/present",
                latency="~2-3 days",
                notes="search is open; COG reads need Earthdata Login (netrc + cookie jar, or a bearer token)",
                title="CMR-STAC LPCLOUD",
                health=Probe(
                    "HLS L30 (CMR-STAC LPCLOUD)",
                    partial(
                        health.stac_search,
                        CMR_LPCLOUD_STAC,
                        "HLSL30_2.0",
                        datetime_range="2024-07-01T00:00:00Z/2024-07-31T23:59:59Z",
                    ),
                    requires=(),
                ),
            ),
        ),
        variables=(
            Variable("red", units="1", dtype="int16", nodata=-9999),
            Variable("nir08", units="1", dtype="int16", nodata=-9999),
            Variable("swir16", units="1", dtype="int16", nodata=-9999),
            Variable("Fmask", dtype="uint8", nodata=255, long_name="quality bits"),
        ),
        citation="Masek, J., et al. (2021). HLS Operational Land Imager Surface Reflectance and TOA Brightness Daily Global 30m v2.0; HLS Sentinel-2 MSI Surface Reflectance Daily Global 30m v2.0. NASA EOSDIS LP DAAC.",
        license="NASA Earthdata (free registration)",
        doi="10.5067/HLS/HLSL30.002",
        loader="easysnowdata.remote_sensing.HLS",
        tags=("optical", "reflectance", "landsat", "sentinel-2"),
    ),
    Product(
        id="modis-snow",
        theme="snow",
        title="MODIS snow cover (MOD10A1, MOD10A2, MOD10A1F)",
        description="Terra MODIS daily NDSI snow cover, 8-day maximum snow extent and the cloud-gap-filled daily product, 500 m.",
        sources=(
            Source(
                id="planetary-computer",
                provider="stac",
                location="modis-10A1-061 / modis-10A2-061",
                resolution_m=500,
                temporal="2000-02/~2025",
                notes="COG mirror; archiving reportedly stopped in 2025 — an access-route failure, not a product failure",
                title="Planetary Computer",
                health=Probe(
                    "MODIS snow cover MOD10A1 (Planetary Computer)",
                    partial(
                        health.stac_search,
                        PC_STAC,
                        "modis-10A1-061",
                        datetime_range="2023-01-01/2023-01-31",
                        sign=True,
                    ),
                ),
            ),
            Source(
                id="nsidc",
                provider="earthdata",
                location="MOD10A1F (and MOD10A1 / MOD10A2 via earthaccess)",
                requires=("earthdata",),
                resolution_m=500,
                temporal="2000-02/present",
                latency="~1 day",
                notes="HDF-EOS2 granules: downloaded to the cache; GDAL needs the HDF4 driver (conda-forge libgdal-hdf4)",
                title="NSIDC (Earthdata)",
                health=Probe(
                    "MODIS snow cover MOD10A1F (NASA NSIDC)",
                    partial(
                        health.earthdata_search,
                        "MOD10A1F",
                        temporal=("2023-01-01", "2023-01-07"),
                    ),
                ),
            ),
        ),
        variables=(
            Variable(
                "NDSI_Snow_Cover",
                units="%",
                dtype="uint8",
                nodata=255,
                long_name="NDSI snow cover",
            ),
            Variable(
                "Maximum_Snow_Extent",
                dtype="uint8",
                nodata=255,
                long_name="8-day maximum snow extent",
                **_flags(MOD10A2_CLASSES),
            ),
            Variable(
                "CGF_NDSI_Snow_Cover",
                units="%",
                dtype="uint8",
                nodata=255,
                long_name="cloud-gap-filled NDSI snow cover",
            ),
        ),
        citation="Hall, D. K. and Riggs, G. A. (2021). MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid, Version 61 (MOD10A1); 8-Day (MOD10A2); Cloud-Gap-Filled (MOD10A1F). NSIDC DAAC.",
        license="NASA Earthdata (free registration)",
        doi="10.5067/MODIS/MOD10A1.061",
        loader="easysnowdata.remote_sensing.MODIS_snow",
        tags=("snow cover", "ndsi", "modis"),
    ),
)
