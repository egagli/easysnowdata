"""Helpers shared by the loaders in this theme (output contract §2.5).

Phase 2 migrates the products theme by theme in parallel sessions, so each
theme package carries its own copy of these few helpers rather than racing for
one shared module. The merge session folds the copies into a single helper
(``easysnowdata._product``) once every theme has landed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr
from odc.geo.xr import assign_crs  # noqa: F401 — registers the .odc accessor
from pyproj import CRS

from easysnowdata import catalog
from easysnowdata.catalog._models import Product, Source, Variable
from easysnowdata.processing.categorical import set_flags

__all__ = [
    "DEFAULT",
    "resolve",
    "attrs_for",
    "standardize",
    "apply_nodata",
    "set_variable_flags",
    "variable",
]

#: Sentinel for "the package default" on keyword arguments whose ``None`` means
#: something else (``chunks=None`` loads eagerly, as in odc-stac/rioxarray).
DEFAULT: Any = "easysnowdata-default"

_GEOGRAPHIC = {"x": "longitude", "y": "latitude", "lon": "longitude", "lat": "latitude"}
_PROJECTED = {"longitude": "x", "latitude": "y", "lon": "x", "lat": "y"}


def resolve(product_id: str, source_id: str | None = None) -> tuple[Product, Source]:
    """Return ``(product, source)`` from the catalog; ``None`` picks the default source."""
    product = catalog.get(product_id)
    return product, product.source(source_id)


def variable(product: Product, name: str) -> Variable:
    """Return the catalog :class:`Variable` called *name*."""
    for var in product.variables:
        if var.name == name:
            return var
    raise KeyError(f"{product.id} has no variable {name!r}.")


def attrs_for(product: Product, source: Source, **extra: Any) -> dict[str, Any]:
    """The attrs every product carries (§2.5), plus *extra* (``None`` values dropped)."""
    from easysnowdata import __version__  # noqa: PLC0415 — avoids a circular import

    attrs: dict[str, Any] = {
        "source": source.id,
        "source_url": source.location,
        "product_id": product.id,
        "data_citation": product.citation,
        "license": product.license,
        "easysnowdata_version": __version__,
    }
    if product.doi:
        attrs["doi"] = product.doi
    attrs.update({k: v for k, v in extra.items() if v is not None})
    return attrs


def standardize(obj: xr.DataArray, crs: Any = None) -> xr.DataArray:
    """Name the spatial dims per §2.5 and write the CRS with both accessors.

    Geographic grids get ``latitude``/``longitude``, projected grids ``y``/``x``;
    the CRS is written once so that both ``.rio.crs`` and ``.odc.crs`` read it.
    """
    current = obj.rio.crs
    target = crs if crs is not None else current
    if target is None:
        raise ValueError("The data carries no CRS and none was given.")
    target_crs = CRS.from_user_input(str(target))
    names = _GEOGRAPHIC if target_crs.is_geographic else _PROJECTED
    rename = {
        old: new
        for old, new in names.items()
        if old in obj.dims and new not in obj.dims
    }
    if rename:
        obj = obj.rename(rename)
    obj = assign_crs(obj, str(target_crs))
    return obj.rio.write_crs(str(target_crs))


def apply_nodata(
    da: xr.DataArray, nodata: float | int | None, *, mask: bool
) -> xr.DataArray:
    """Apply the nodata policy (§2.5).

    ``mask=False`` keeps the source sentinel and sets ``rio.nodata`` (what
    categorical products do by default); ``mask=True`` replaces it with NaN and
    keeps it as ``rio.encoded_nodata`` (the default for continuous products).
    """
    if nodata is None:
        return da
    if not mask:
        return da.rio.write_nodata(nodata, encoded=False)
    out = da.astype("float32") if np.issubdtype(da.dtype, np.integer) else da
    out = out.where(out != nodata)
    return out.rio.write_nodata(nodata, encoded=True)


def set_variable_flags(da: xr.DataArray, product: Product, name: str) -> xr.DataArray:
    """Attach the catalog's CF flag attributes for variable *name* (§2.5)."""
    var = variable(product, name)
    if not var.categorical:
        return da
    return set_flags(
        da,
        var.flag_values,
        var.flag_meanings,
        var.flag_colors or None,
        long_name=var.long_name or None,
    )
