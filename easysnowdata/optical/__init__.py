"""Optical (passive) imagery: Sentinel-2 L2A, HLS L30 and S30, and PlanetScope.

Visible through shortwave infrared, plus HLS L30's two thermal bands. The name
pairs with :mod:`easysnowdata.sar` (active microwave). The Sentinel-2 scene
classification and HLS Fmask masks live in :mod:`easysnowdata.processing.masks`;
the Sentinel-2 baseline harmonization, scale/offset, and the PlanetScope UDM2
decoder in :mod:`easysnowdata.processing.optical`. Band arithmetic is written
out, not wrapped. The loaders here only add product knowledge: collection ids,
band aliases, nodata, scaling, and citations.
"""

from __future__ import annotations

from easysnowdata.optical import hls, planetscope, sentinel2

__all__ = ["hls", "planetscope", "sentinel2"]
