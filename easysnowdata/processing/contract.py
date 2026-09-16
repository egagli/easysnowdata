"""Output-contract helpers (design contract §2.5), shared by every theme loader.

Pure functions: they rename dimensions, write the CRS with both the ``rio``
and ``odc`` accessors, mask nodata the way the contract prescribes
(categorical keeps the sentinel with ``rio.nodata`` set; continuous is
NaN-masked with ``encoded_nodata`` preserved) and stamp the standard
provenance attributes (``source``, ``source_url``, ``product_id``,
``data_citation``, ``license``, ``easysnowdata_version``). No I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr
from pyproj import CRS

__all__ = [
    "DEFAULT",
    "PROVENANCE_KEYS",
    "geographic_dims",
    "write_crs",
    "mask_continuous",
    "set_categorical_nodata",
    "provenance",
    "apply_variables",
    "finalize",
]

#: Sentinel for "the package default" on a keyword whose ``None`` means something
#: else — ``chunks=None`` loads eagerly, as in odc-stac and rioxarray (§12.2 D).
DEFAULT: Any = "easysnowdata-default"

PROVENANCE_KEYS = (
    "source",
    "source_url",
    "product_id",
    "data_citation",
    "license",
    "easysnowdata_version",
)


def _crs(crs: Any) -> CRS:
    return crs if isinstance(crs, CRS) else CRS.from_user_input(crs)


def _spatial_dims(obj: xr.Dataset | xr.DataArray) -> tuple[str, str] | None:
    """``(x_dim, y_dim)`` from the known naming conventions, else ``None``."""
    dims = set(obj.dims)
    for x, y in (("x", "y"), ("longitude", "latitude"), ("lon", "lat")):
        if x in dims and y in dims:
            return x, y
    return None


def geographic_dims(
    obj: xr.Dataset | xr.DataArray, crs: Any | None = None
) -> xr.Dataset | xr.DataArray:
    """Rename spatial dims to the contract: ``latitude``/``longitude`` for a
    geographic CRS, ``y``/``x`` for a projected one."""
    crs_obj = _crs(crs) if crs is not None else _detect_crs(obj)
    if crs_obj is None:
        return obj
    dims = _spatial_dims(obj)
    if dims is None:
        return obj
    x, y = dims
    if crs_obj.is_geographic:
        mapping = {x: "longitude", y: "latitude"}
    else:
        mapping = {x: "x", y: "y"}
    mapping = {k: v for k, v in mapping.items() if k != v}
    return obj.rename(mapping) if mapping else obj


def _detect_crs(obj: xr.Dataset | xr.DataArray) -> CRS | None:
    try:
        rio_crs = obj.rio.crs
    except Exception:  # noqa: BLE001 — no spatial dims yet
        rio_crs = None
    if rio_crs is not None:
        return CRS.from_user_input(rio_crs)
    return None


def write_crs(obj: xr.Dataset | xr.DataArray, crs: Any) -> xr.Dataset | xr.DataArray:
    """Write *crs* so that both ``obj.rio.crs`` and ``obj.odc.crs`` read it.

    The spatial dims are set from their names first (``x``/``y`` or
    ``longitude``/``latitude``), and the dims are renamed to the contract's
    convention for the CRS kind.
    """
    import odc.geo.xr  # noqa: F401, PLC0415 — registers the .odc accessor

    if crs is None:
        raise ValueError("The data carries no CRS and none was given.")
    crs_obj = _crs(crs)
    obj = geographic_dims(obj, crs_obj)
    dims = _spatial_dims(obj)
    if dims is not None:
        x, y = dims
        obj = obj.rio.set_spatial_dims(x_dim=x, y_dim=y, inplace=False)
    obj = obj.rio.write_crs(crs_obj, inplace=False)
    try:
        ok = obj.odc.crs is not None
    except Exception:  # noqa: BLE001 — odc could not read the rio metadata
        ok = False
    if not ok:
        obj = obj.odc.assign_crs(str(crs_obj))
    return obj


def mask_continuous(
    da: xr.DataArray, nodata: float | int | None = None
) -> xr.DataArray:
    """NaN-mask a continuous variable and keep the encoded nodata value.

    *nodata* defaults to ``da.rio.nodata`` / ``da.attrs["nodata"]`` /
    ``da.rio.encoded_nodata``. Integer data become ``float32``.
    """
    if nodata is None:
        nodata = da.attrs.get("nodata")
    if nodata is None:
        nodata = da.rio.nodata if _has_rio(da) else None
    if nodata is None and _has_rio(da):
        nodata = da.rio.encoded_nodata
    out = da if np.issubdtype(da.dtype, np.floating) else da.astype("float32")
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        out = out.where(da != nodata)
    out.attrs = {k: v for k, v in da.attrs.items() if k != "nodata"}
    out.encoding = dict(da.encoding)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        out.encoding["_FillValue"] = nodata
    if _has_rio(out):
        out = out.rio.write_nodata(np.nan, inplace=False)
    return out


def set_categorical_nodata(da: xr.DataArray, nodata: int | float) -> xr.DataArray:
    """Keep the source sentinel and record it as ``rio.nodata`` (and ``attrs['nodata']``)."""
    out = da.copy(deep=False)
    if _has_rio(out):
        out = out.rio.write_nodata(nodata, inplace=False)
    out.attrs["nodata"] = nodata
    return out


def _has_rio(obj: Any) -> bool:
    try:
        obj.rio.x_dim  # noqa: B018 — probes whether spatial dims are set
        return True
    except Exception:  # noqa: BLE001
        return False


def provenance(
    product: Any,
    source: Any,
    *,
    source_url: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The standard provenance attrs for a catalog *product* loaded via *source*."""
    from easysnowdata import __version__  # noqa: PLC0415

    if isinstance(source, str):
        source_title, source_id = source, source
    else:
        source_id = getattr(source, "id", str(source))
        source_title = getattr(source, "title", None) or source_id
    attrs: dict[str, Any] = {
        # `source` is the id, so it can be handed straight back to load(source=...);
        # `source_title` is the human-readable name for display.
        "source": source_id,
        "source_id": source_id,
        "source_title": source_title,
        "source_url": source_url or _default_source_url(source),
        "product_id": product.id,
        "title": product.title,
        "data_citation": product.citation,
        "license": product.license,
        "easysnowdata_version": __version__,
    }
    if getattr(product, "doi", None):
        attrs["doi"] = product.doi
    attrs.update({k: v for k, v in extra.items() if v is not None})
    return attrs


def _default_source_url(source: Any) -> str:
    location = getattr(source, "location", "")
    if isinstance(location, str) and "://" in location:
        return location
    return ""


def apply_variables(
    ds: xr.Dataset,
    variables: Iterable[Any],
    *,
    mask: bool | None = None,
) -> xr.Dataset:
    """Apply the catalog ``Variable`` definitions present in *ds*.

    Categorical variables get their CF flag attrs and keep the sentinel
    (``rio.nodata`` set); continuous variables are NaN-masked with the encoded
    nodata preserved. ``mask=False`` skips the NaN masking (the raw product);
    ``mask=True`` also NaN-masks categorical variables.
    """
    out = ds.copy()
    for var in variables:
        if var.name not in out.data_vars:
            continue
        da = out[var.name]
        cf = var.cf_attrs()
        if var.dtype and not var.categorical and mask is False:
            pass
        nodata = var.nodata if var.nodata is not None else da.attrs.get("nodata")
        if var.categorical:
            if nodata is not None:
                da = set_categorical_nodata(da, nodata)
            if mask is True and nodata is not None:
                da = mask_continuous(da, nodata)
        elif mask is not False:
            da = mask_continuous(da, nodata)
        elif nodata is not None:
            da = set_categorical_nodata(da, nodata)
        da.attrs.update(cf)
        out[var.name] = da
    return out


def finalize(
    obj: xr.Dataset | xr.DataArray,
    product: Any,
    source: Any,
    *,
    crs: Any | None = None,
    variables: Iterable[Any] | None = None,
    mask: bool | None = None,
    source_url: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    time_dim: str = "time",
) -> xr.Dataset | xr.DataArray:
    """Bring a loaded object onto the output contract.

    Steps: write *crs* (default: the object's ``rio.crs``) with both
    accessors and rename dims for the CRS kind; sort and order dims as
    ``(time, y, x)``; apply the product's ``Variable`` definitions (nodata
    policy, CF flags); stamp provenance and *attrs*.
    """
    crs_obj = _crs(crs) if crs is not None else _detect_crs(obj)
    if crs_obj is not None:
        obj = write_crs(obj, crs_obj)
    dims = _spatial_dims(obj)
    if dims is not None:
        x, y = dims
        order = [d for d in obj.dims if d not in (x, y, time_dim)]
        if time_dim in obj.dims:
            order = [time_dim, *order]
        obj = obj.transpose(*order, y, x)
    if isinstance(obj, xr.Dataset):
        var_defs = list(variables if variables is not None else product.variables)
        obj = apply_variables(obj, var_defs, mask=mask)
    obj.attrs.update(provenance(product, source, source_url=source_url))
    if attrs:
        obj.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return obj
