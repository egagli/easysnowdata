"""Terrain: digital elevation models and topographic indices.

Products
--------
:mod:`~easysnowdata.terrain.dem`
    Five DEMs behind one ``load``: Copernicus DEM GLO-30/90 (the default),
    NASADEM, SRTM GL1, USGS 3DEP (10 m, US) and ALOS World 3D. Planetary
    Computer and Earth Search need no account; Earth Engine is an
    alternative route to each and the only one to SRTM proper.
:mod:`~easysnowdata.terrain.chili`
    CHILI, the continuous heat-insolation load index (Earth Engine).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.terrain.dem.load(aoi)                            # Copernicus GLO-30
    esd.terrain.dem.load(aoi, product="3dep")            # 10 m over the US
    esd.terrain.dem.compare()                            # how the five differ
    esd.terrain.chili.load(aoi)
"""

from __future__ import annotations

from easysnowdata.terrain import chili, dem

__all__ = ["chili", "dem"]
