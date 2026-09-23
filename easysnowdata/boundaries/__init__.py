"""Boundaries: countries, states, counties, admin units, mountain ranges, glaciers.

Products
--------
:mod:`~easysnowdata.boundaries.admin`
    Countries and states/provinces (Natural Earth), US states and counties
    (US Census), and any administrative level of any country (geoBoundaries).
:mod:`~easysnowdata.boundaries.natural_earth`
    Any Natural Earth vector layer by name: lakes, rivers, coastline,
    glaciated areas, populated places.
:mod:`~easysnowdata.boundaries.mountains`
    GMBA Mountain Inventory v2 mountain ranges.
:mod:`~easysnowdata.boundaries.glaciers`
    Randolph Glacier Inventory outlines, version 7.0 or 6.0.

Every loader returns a :class:`geopandas.GeoDataFrame` in EPSG:4326 of the
features that intersect the AOI (whole features, not cut at its edge), and any
of them is itself an AOI for the other loaders.

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    esd.boundaries.admin.countries()                         # the world, 1:110m
    esd.boundaries.admin.states(aoi)                         # Washington (Census)
    esd.boundaries.admin.counties(aoi)                       # Pierce, Lewis …
    esd.boundaries.admin.admin(country="NOR", level=2)       # Norwegian kommuner
    esd.boundaries.natural_earth.load(aoi, layer="lakes")
    esd.boundaries.mountains.load(aoi)                       # Mount Rainier Massif …
    esd.boundaries.glaciers.load(aoi)                        # RGI 7.0
    esd.boundaries.glaciers.load(aoi, version="6.0", source="oggm-mirror")
"""

from __future__ import annotations

from easysnowdata.boundaries import admin, glaciers, mountains, natural_earth

__all__ = ["admin", "glaciers", "mountains", "natural_earth"]
