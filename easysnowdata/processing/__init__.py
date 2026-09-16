"""Pure processing functions (design contract §2.6): xarray in, xarray out, no I/O.

* :mod:`~easysnowdata.processing.masks` — Sentinel-2 SCL and HLS Fmask masks,
  nodata masking
* :mod:`~easysnowdata.processing.optical` — baseline harmonization,
  scale/offset, spectral indices, RGB composites and stretches
* :mod:`~easysnowdata.processing.sar` — dB conversion, border-noise removal
* :mod:`~easysnowdata.processing.wateryear` — vectorized water-year helpers
* :mod:`~easysnowdata.processing.categorical` — CF flag attributes for
  categorical products (the contract's replacement for ``class_info`` dicts)
"""

from __future__ import annotations

from easysnowdata.processing import categorical, masks, optical, sar, wateryear
from easysnowdata.processing.categorical import (
    flag_mask,
    flags,
    from_class_info,
    set_flags,
)
from easysnowdata.processing.masks import (
    DEFAULT_SCL_REMOVE,
    FMASK_BITS,
    apply_fmask,
    apply_scl_mask,
    fmask_aerosol_level,
    fmask_bit,
    fmask_mask,
    mask_nodata,
    scl_mask,
)
from easysnowdata.processing.optical import (
    SCL_CLASSES,
    evi,
    harmonize_s2_baseline,
    ndbi,
    ndsi,
    ndvi,
    ndwi,
    normalized_difference,
    rgb,
    scale_offset,
    stretch_clahe,
    stretch_percentile,
)
from easysnowdata.processing.sar import db_to_linear, linear_to_db, remove_border_noise
from easysnowdata.processing.wateryear import (
    add_water_year_coords,
    day_of_water_year,
    water_year,
    water_year_start,
)

__all__ = [
    "categorical",
    "masks",
    "optical",
    "sar",
    "wateryear",
    "DEFAULT_SCL_REMOVE",
    "FMASK_BITS",
    "SCL_CLASSES",
    "add_water_year_coords",
    "apply_fmask",
    "apply_scl_mask",
    "day_of_water_year",
    "db_to_linear",
    "evi",
    "flag_mask",
    "flags",
    "fmask_aerosol_level",
    "fmask_bit",
    "fmask_mask",
    "from_class_info",
    "harmonize_s2_baseline",
    "linear_to_db",
    "mask_nodata",
    "ndbi",
    "ndsi",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "remove_border_noise",
    "rgb",
    "scale_offset",
    "scl_mask",
    "set_flags",
    "stretch_clahe",
    "stretch_percentile",
    "water_year",
    "water_year_start",
]
