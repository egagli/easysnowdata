"""Snow: classifications, masks, cover and snow water equivalent.

Products
--------
:mod:`~easysnowdata.snow.snow_classification`
    Sturm & Liston seasonal snow classes (NSIDC-0768, hosted COG).
:mod:`~easysnowdata.snow.mountain_snow_mask`
    Wrzesien mountain snow mask and its clouds layer (Zenodo).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.snow.snow_classification.load(aoi, source="hosted-cog")
    esd.snow.mountain_snow_mask.load(aoi)
"""

from __future__ import annotations

from easysnowdata.snow import mountain_snow_mask, snow_classification

__all__ = ["mountain_snow_mask", "snow_classification"]
