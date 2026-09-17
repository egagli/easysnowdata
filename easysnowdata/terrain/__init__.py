"""Terrain: digital elevation models and topographic indices.

Products
--------
:mod:`~easysnowdata.terrain.dem`
    Copernicus DEM GLO-30 / GLO-90 (Planetary Computer, Earth Search).
:mod:`~easysnowdata.terrain.chili`
    CHILI, the continuous heat-insolation load index (Earth Engine).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.terrain.dem.load(aoi)
    esd.terrain.chili.load(aoi)
"""

from __future__ import annotations

from easysnowdata.terrain import chili, dem

__all__ = ["chili", "dem"]
