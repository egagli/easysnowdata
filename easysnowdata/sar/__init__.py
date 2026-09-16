"""Synthetic aperture radar (active microwave) products.

Pairs with :mod:`easysnowdata.optical` (passive optical). Backscatter
processing — dB conversion, border noise, terrain geometry and the local
incidence angle — lives in :mod:`easysnowdata.processing.sar`.
"""

from __future__ import annotations

from easysnowdata.sar import sentinel1

__all__ = ["sentinel1"]
