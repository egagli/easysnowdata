"""Land cover and vegetation.

Products
--------
:mod:`~easysnowdata.land.landcover`
    ESA WorldCover 10 m land cover (Planetary Computer, AWS Open Data).
:mod:`~easysnowdata.land.nlcd`
    National Land Cover Database: Annual NLCD and the 2021 release.
:mod:`~easysnowdata.land.forest_cover`
    CGLS-LC100 tree-cover fraction (Zenodo, Earth Engine).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.land.landcover.load(aoi)
    esd.land.forest_cover.load(aoi)
"""

from __future__ import annotations

from easysnowdata.land import forest_cover, landcover, nlcd

__all__ = ["forest_cover", "landcover", "nlcd"]
