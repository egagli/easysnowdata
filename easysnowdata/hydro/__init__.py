"""Hydrography: basin and watershed boundaries.

Products
--------
:mod:`~easysnowdata.hydro.basins`
    USGS HUC units (WBD REST, Earth Engine), HydroBASINS / BasinATLAS
    (figshare, HydroSHEDS regional zips, Earth Engine), GRDC major river
    basins and GRDC / WMO basins.

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.hydro.basins.huc(aoi, level=12)
    esd.hydro.basins.hydrobasins(aoi, level=5)
"""

from __future__ import annotations

from easysnowdata.hydro import basins

__all__ = ["basins"]
