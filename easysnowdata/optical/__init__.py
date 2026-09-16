"""Optical (passive, visible through SWIR and thermal) imagery.

The name pairs with :mod:`easysnowdata.sar` (active microwave) and covers
every sensor in the theme, including thermal-only and hyperspectral products
that are not multispectral (plan §3.1). Band math, masks and composites live
in :mod:`easysnowdata.processing.optical`; the loaders here only add product
knowledge: collection ids, band aliases, nodata, scaling and citations.
"""

from __future__ import annotations

from easysnowdata.optical import hls, sentinel2

__all__ = ["hls", "sentinel2"]
