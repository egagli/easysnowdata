"""Climate products: ERA5 / ERA5-Land, the Köppen-Geiger classification, and climate-mode indices.

Each module exposes module-level ``search()`` / ``load()`` functions with a
``source=`` argument (design contract §2.12) and registers its catalog entry
on import.
"""

from __future__ import annotations

from easysnowdata.climate import era5, indices, koppen_geiger

__all__ = ["era5", "indices", "koppen_geiger"]
