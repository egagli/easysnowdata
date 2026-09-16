"""Pure processing functions (design contract §2.6): xarray in, xarray out, no I/O.

* :mod:`~easysnowdata.processing.masks` — Sentinel-2 SCL and HLS Fmask masks,
  nodata masking
* :mod:`~easysnowdata.processing.optical` — baseline harmonization,
  scale/offset, spectral indices, RGB composites and stretches
* :mod:`~easysnowdata.processing.sar` — dB conversion, border-noise removal,
  slope/aspect and the local incidence angle
* :mod:`~easysnowdata.processing.snow` — snow-product class tables, binary
  snow and the SNODAS flat-file reader
* :mod:`~easysnowdata.processing.wateryear` — vectorized water-year helpers
* :mod:`~easysnowdata.processing.categorical` — CF flag attributes for
  categorical products (the contract's replacement for ``class_info`` dicts)
* :mod:`~easysnowdata.processing.contract` — the output contract (§2.5):
  CRS via both accessors, dim names, nodata policy, provenance attrs
"""

from __future__ import annotations

from easysnowdata.processing import (
    categorical,
    contract,
    masks,
    optical,
    sar,
    snow,
    wateryear,
)
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
    UDM1_BITS,
    UDM2_BANDS,
    UDM2_BINARY_BANDS,
    decode_udm2,
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
    udm1_bit,
)
from easysnowdata.processing.sar import (
    db_to_linear,
    linear_to_db,
    local_incidence_angle,
    look_azimuth,
    remove_border_noise,
    slope_aspect,
)
from easysnowdata.processing.snow import (
    MOD10A2_CLASSES,
    NDSI_FLAGS,
    binary_snow,
    ndsi_flag_attrs,
    parse_snodas_header,
)
from easysnowdata.processing.wateryear import (
    add_water_year_coords,
    day_of_water_year,
    water_year,
    water_year_start,
)

__all__ = [
    "categorical",
    "contract",
    "masks",
    "optical",
    "sar",
    "snow",
    "wateryear",
    "DEFAULT_SCL_REMOVE",
    "FMASK_BITS",
    "SCL_CLASSES",
    "UDM1_BITS",
    "UDM2_BANDS",
    "UDM2_BINARY_BANDS",
    "MOD10A2_CLASSES",
    "NDSI_FLAGS",
    "add_water_year_coords",
    "apply_fmask",
    "binary_snow",
    "apply_scl_mask",
    "day_of_water_year",
    "db_to_linear",
    "decode_udm2",
    "evi",
    "flag_mask",
    "flags",
    "fmask_aerosol_level",
    "fmask_bit",
    "fmask_mask",
    "from_class_info",
    "harmonize_s2_baseline",
    "linear_to_db",
    "local_incidence_angle",
    "look_azimuth",
    "mask_nodata",
    "ndbi",
    "ndsi",
    "ndvi",
    "ndwi",
    "ndsi_flag_attrs",
    "normalized_difference",
    "parse_snodas_header",
    "remove_border_noise",
    "rgb",
    "scale_offset",
    "scl_mask",
    "set_flags",
    "slope_aspect",
    "stretch_clahe",
    "stretch_percentile",
    "udm1_bit",
    "water_year",
    "water_year_start",
]
