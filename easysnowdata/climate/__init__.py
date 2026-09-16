"""Climate and reanalysis products: ERA5 / ERA5-Land and the Köppen-Geiger classification.

Each module exposes module-level ``search()`` / ``load()`` functions with a
``source=`` argument (design contract §2.12) and registers its catalog entry
on import.
"""

from __future__ import annotations

from easysnowdata.climate import era5

__all__ = ["era5"]
