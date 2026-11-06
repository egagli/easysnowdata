# esd-requires: earthdata
"""
Seasonal snow classification (Sturm and Liston)
===============================================

Sturm and Liston's global seasonal snow classification (NSIDC-0768, 2021)
assigns each place one of six snow classes — tundra, boreal forest, maritime,
ephemeral, prairie, montane forest — plus ice and ocean, from climatologies of
air temperature, precipitation and wind speed. It is a static map at 10 arcsec
(about 300 m) with 30 arcsec, 2.5 arcmin and 0.5 degree versions.

Two routes. ``source="nsidc"`` (the default) fetches a whole GeoTIFF from the
NSIDC archive behind Earthdata Login and caches it; the 10 arcsec global file
is 8.4 GB, so the coarser grids are the sensible choice there.
``source="hosted-cog"`` reads the 10 arcsec global map as a cloud-optimized
GeoTIFF with range requests and no credentials, and only the pixels you ask
for are transferred.

The figures show the 10 arcsec classes from the hosted COG next to the
2.5 arcmin grid from NSIDC over the central Cascades, and the agreement
between them.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-123.0, 46.0, -120.5, 48.0)  # the central Cascades, Puget Sound to Yakima

box = esd.parse_aoi(aoi)
print(box)

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("snow-classification").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The 10 arcsec map through the credential-free COG: maritime snow on the
# Cascade crest, montane forest on the flanks, ephemeral in the lowlands.
# The colours and legend come from the CF flag attributes the loader attaches.
# Both grids are geographic; the box is loaded with a 5 km margin (more than
# half a 2.5 arcmin cell) so the UTM grids the maps use are fully covered.
fine_da = esd.snow.snow_classification.load(
    box.buffer(5000), source="hosted-cog"
).compute()
print(fine_da.attrs["flag_meanings"])

# %%
# The 2.5 arcmin grid from NSIDC (a 37 MB download, cached after the first
# call) beside it: the same classes, one pixel where the COG has 225.
coarse_da = esd.snow.snow_classification.load(
    box.buffer(5000), resolution="2.5arcmin"
).compute()

# Drawn on the AOI's UTM zone at each product's own resolution (nearest, so
# the classes survive); the comparison below stays on the native grids.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
esd.plotting.categorical(
    fine_da.odc.reproject(
        box.to_geobox(crs="utm", resolution=300), resampling="nearest"
    ),
    ax=axes[0],
    title="hosted COG, 10 arcsec",
)
esd.plotting.categorical(
    coarse_da.odc.reproject(
        box.to_geobox(crs="utm", resolution=4500), resampling="nearest"
    ),
    ax=axes[1],
    title="NSIDC-0768, 2.5 arcmin",
)
fig.tight_layout()

# %%
# How well the coarse grid agrees with the fine one: sample the 10 arcsec map
# at each 2.5 arcmin cell centre and count matches.
sampled_da = fine_da.sel(
    latitude=coarse_da["latitude"], longitude=coarse_da["longitude"], method="nearest"
)
valid_da = (coarse_da != 9) & (sampled_da.values != 9)
agree = (coarse_da.values == sampled_da.values) & valid_da.values
print(
    f"2.5 arcmin cells whose centre pixel at 10 arcsec has the same class: "
    f"{int(agree.sum())} of {int(valid_da.sum())} ({100 * agree.sum() / valid_da.sum():.0f} %)"
)
for value, name in zip(
    fine_da.attrs["flag_values"], fine_da.attrs["flag_meanings"].split()
):
    share = float((fine_da == value).mean()) * 100
    if share:
        print(f"{name.replace('_', ' '):32} {share:5.1f} % of the 10 arcsec box")
print(f"pixels compared: {int(np.prod(coarse_da.shape))}")
