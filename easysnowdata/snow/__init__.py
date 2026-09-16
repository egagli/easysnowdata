"""Snow products.

Time series (Phase 2b): MODIS and VIIRS snow cover, SNODAS, and the UCLA
snow reanalysis. Class tables, binary-snow thresholding and the SNODAS
flat-file reader are pure functions in :mod:`easysnowdata.processing.snow`.
"""

from __future__ import annotations

from easysnowdata.snow import modis, snodas, viirs

__all__ = ["modis", "snodas", "viirs"]
