"""CHILI — Continuous Heat-Insolation Load Index (§4.5).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    chili_da = esd.terrain.chili.load(aoi)                     # native 0-255 values
    index_da = esd.terrain.chili.load(aoi, normalize="index")  # the 0-1 index

CHILI is a topographic index of the combined effect of solar radiation and
surface temperature computed from ALOS World 3D-30m (Theobald et al. 2015),
served at ~90 m between 70°N and 70°S. The Earth Engine asset stores the 0-1
index scaled to 8 bits, so values come back on a 0-255 scale.

Unlike the pre-0.2 ``topography.get_chili`` this loader does **not** rescale by
default: min-max normalization inside the AOI made values depend on the box
that was requested, so the same pixel changed value with the window
(``normalize="minmax"`` keeps that behaviour for one release).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = ["PRODUCT_ID", "ASSET_ID", "load"]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "chili"
ASSET_ID = "CSP/ERGo/1_0/Global/ALOS_CHILI"
#: The Earth Engine band; the asset holds the index scaled to 0-255.
BAND = "constant"
INDEX_SCALE = 255.0


def _normalized(da: xr.DataArray, normalize: bool | str) -> tuple[xr.DataArray, str]:
    if normalize is False or normalize is None:
        return da, "1 (0-255, the asset's 8-bit scaling of the 0-1 index)"
    mode = "minmax" if normalize is True else str(normalize)
    if mode == "index":
        return da / INDEX_SCALE, "1 (0-1 heat-insolation load index)"
    if mode == "minmax":
        low_da, high_da = da.min(), da.max()
        return (da - low_da) / (high_da - low_da), "1 (min-max rescaled within the AOI)"
    raise ValueError(
        f"normalize must be False, 'index' or 'minmax' (True), got {normalize!r}."
    )


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    normalize: bool | str = False,
    chunks: Any = contract.DEFAULT,
    mask: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Load CHILI for *aoi* from Earth Engine at the asset's native grid.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. The grid is the
        smallest block of native pixels covering it, so values are never
        resampled.
    source
        Only ``"gee"`` today; a DEM-computed heat-load index is the planned
        credential-free route.
    normalize
        ``False`` (default) returns the native 0-255 values, ``"index"``
        divides by 255 to give the 0-1 index, ``"minmax"`` (or ``True``)
        rescales within the AOI — what the pre-0.2 ``get_chili`` did.
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        Kept for symmetry with the other loaders. Earth Engine already returns
        pixels outside the asset as NaN, so there is no sentinel to mask.
    **kwargs
        Passed to ``xarray.open_dataset(engine="ee")``.

    Returns
    -------
    xarray.DataArray
        ``chili``, dims ``latitude``/``longitude`` (or ``y``/``x`` when the
        asset's native grid is projected), lazy unless ``chunks=None``.

    Raises
    ------
    easysnowdata.auth.CredentialError
        When Earth Engine is not configured (``esd.auth.login("earthengine")``).
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    ee = providers.gee.ee()
    image = ee.Image(ASSET_ID)
    grid = providers.gee.grid_params(image, aoi)
    params: dict[str, Any] = {"grid": grid}
    if chunks is not contract.DEFAULT:
        params["chunks"] = chunks
    ds = providers.gee.open_dataset(ee.ImageCollection(image), aoi, **params, **kwargs)
    da = ds[BAND]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)  # a single static image
    da = contract.write_crs(da, grid["crs"])
    da, units = _normalized(da, normalize)
    da = da.rename("chili")
    da.attrs.update(
        contract.provenance(
            product,
            src,
            source_url=f"https://developers.google.com/earth-engine/datasets/catalog/{ASSET_ID.replace('/', '_')}",
            long_name="continuous heat-insolation load index",
            units=units,
            normalize=str(normalize),
            extent="70°N-70°S",
        )
    )
    if chunks is None:
        da = da.compute()
    return da


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="terrain",
    title="CHILI (Continuous Heat-Insolation Load Index)",
    description=(
        "Topographic heat-load index from ALOS AW3D30 (Theobald et al. 2015), "
        "90 m, 70°N-70°S. Native values are the 0-1 index scaled to 0-255; "
        "warm slopes are above 0.767 and cool slopes below 0.448."
    ),
    sources=(
        Source(
            id="gee",
            provider="gee",
            location=ASSET_ID,
            requires=("earthengine",),
            resolution_m=90,
            extent="70°N-70°S",
            temporal="static",
            notes=(
                "native values by default (the AOI-relative rescaling of "
                "topography.get_chili is opt-in); a DEM-computed heat-load "
                "index is the planned credential-free route"
            ),
            title="Earth Engine",
            health=Probe(
                "CHILI (GEE/CSP ERGo)",
                partial(health.gee_asset, ASSET_ID, "image"),
            ),
        ),
    ),
    variables=(
        Variable(
            "chili",
            units="1",
            dtype="float32",
            long_name="continuous heat-insolation load index",
        ),
    ),
    citation=(
        "Theobald, D. M., Harrison-Atlas, D., Monahan, W. B., Albano, C. M. (2015). "
        "Ecologically-Relevant Maps of Landforms and Physiographic Diversity for "
        "Climate Adaptation Planning. PLoS ONE 10(12): e0143619."
    ),
    license="CC BY 4.0",
    doi="10.1371/journal.pone.0143619",
    loader="easysnowdata.terrain.chili.load",
    examples=("terrain/plot_chili.py",),
    references=(
        "https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_CHILI",
    ),
    tags=("terrain", "insolation", "heat load"),
)

catalog.register(PRODUCT, replace=True)
