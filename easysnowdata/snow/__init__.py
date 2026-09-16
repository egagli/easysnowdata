"""Snow: classifications, masks, cover and snow water equivalent.

Static products
---------------
:mod:`~easysnowdata.snow.snow_classification`
    Sturm & Liston seasonal snow classes (NSIDC-0768, hosted COG).
:mod:`~easysnowdata.snow.mountain_snow_mask`
    Wrzesien mountain snow mask and its clouds layer (Zenodo).

Time series
-----------
:mod:`~easysnowdata.snow.modis`, :mod:`~easysnowdata.snow.viirs`
    MODIS and VIIRS snow cover.
:mod:`~easysnowdata.snow.snodas`
    SNODAS snow water equivalent and snow depth.
:mod:`~easysnowdata.snow.ucla_sr`
    UCLA snow reanalysis.

Class tables, binary-snow thresholding and the SNODAS flat-file reader are
pure functions in :mod:`easysnowdata.processing.snow`.

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
