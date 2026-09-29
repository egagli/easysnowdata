"""Snow: MODIS and VIIRS snow cover, SNODAS and UCLA snow water equivalent and depth, the Sturm & Liston snow classes, and the Wrzesien mountain snow mask.

Static products
---------------
:mod:`~easysnowdata.snow.snow_classification`
    Sturm & Liston seasonal snow classes (NSIDC-0768, or the hosted COG).
:mod:`~easysnowdata.snow.mountain_snow_mask`
    Wrzesien mountain snow mask and its clouds layer (Zenodo).

Time series
-----------
:mod:`~easysnowdata.snow.modis`
    MODIS NDSI snow cover, 500 m: MOD10A1/MYD10A1 (daily), MOD10A2/MYD10A2
    (8-day maximum extent), and MOD10A1F/MYD10A1F (cloud-gap-filled), from
    NSIDC; MOD/MYD10A1 and 10A2 also from Planetary Computer.
:mod:`~easysnowdata.snow.viirs`
    VIIRS NDSI snow cover, 375 m: VNP10A1 and VNP10A1F, and the NOAA-20
    VJ110A1 and VJ110A1F, from NSIDC.
:mod:`~easysnowdata.snow.snodas`
    SNODAS snow water equivalent, snow depth, and the other model states,
    daily, 1 km, CONUS, 2003-10 onward; no account needed.
:mod:`~easysnowdata.snow.ucla_sr`
    UCLA snow reanalysis: daily 480 m SWE, snow depth, and snow-covered area
    over the western US (water years 1985-2021) and High Mountain Asia
    (2000-2017).

Class tables, NDSI flag attributes, and the SNODAS flat-file reader are pure
functions in :mod:`easysnowdata.processing.snow`; a binary-snow threshold is one
line of xarray and is left to you.

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.snow.snow_classification.load(aoi, source="hosted-cog")
    esd.snow.mountain_snow_mask.load(aoi)
    esd.snow.snodas.load(aoi, time="2024-03")
"""

from __future__ import annotations

from easysnowdata.snow import (
    modis,
    mountain_snow_mask,
    snodas,
    snow_classification,
    ucla_sr,
    viirs,
)

__all__ = [
    "modis",
    "mountain_snow_mask",
    "snodas",
    "snow_classification",
    "ucla_sr",
    "viirs",
]
