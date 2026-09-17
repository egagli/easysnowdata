"""The product catalog: one declarative entry per product (§3.3).

::

    import easysnowdata as esd
    esd.catalog.list(theme="snow")          # DataFrame of products
    esd.catalog.describe("copernicus-dem")  # sources, credentials, resolution, citation
    esd.catalog.search("swe")               # free-text search
    esd.catalog.get("era5").sources         # the Source dataclasses

The same registry drives ``esd.catalog.health`` (the weekly probes and the
README status table) and, in Phase 4, the generated docs pages.
"""

from __future__ import annotations

from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable, validate
from easysnowdata.catalog._registry import (
    describe,
    get,
    products,
    register,
    search,
    themes,
    validate_all,
)
from easysnowdata.catalog._registry import (
    list_products as list,  # noqa: A001 — public API name
)

__all__ = [
    "Variable",
    "Probe",
    "Source",
    "Product",
    "register",
    "products",
    "get",
    "list",
    "search",
    "describe",
    "themes",
    "validate",
    "validate_all",
    "health",
]
