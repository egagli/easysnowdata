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

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("snow-classification").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The 10 arcsec map through the credential-free COG: maritime snow on the
# Cascade crest, montane forest on the flanks, ephemeral in the lowlands.
# The colours and legend come from the CF flag attributes the loader attaches.
fine = esd.snow.snow_classification.load(aoi, source="hosted-cog").compute()
print(fine.attrs["flag_meanings"])

# %%
# The 2.5 arcmin grid from NSIDC (a 37 MB download, cached after the first
# call) beside it: the same classes, one pixel where the COG has 225.
coarse = esd.snow.snow_classification.load(aoi, resolution="2.5arcmin").compute()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
esd.plotting.categorical(fine, ax=axes[0], title="hosted COG, 10 arcsec")
esd.plotting.categorical(coarse, ax=axes[1], title="NSIDC-0768, 2.5 arcmin")
fig.tight_layout()

# %%
# How well the coarse grid agrees with the fine one: sample the 10 arcsec map
# at each 2.5 arcmin cell centre and count matches.
sampled = fine.sel(
    latitude=coarse["latitude"], longitude=coarse["longitude"], method="nearest"
)
valid = (coarse != 9) & (sampled.values != 9)
agree = (coarse.values == sampled.values) & valid.values
print(
    f"2.5 arcmin cells whose centre pixel at 10 arcsec has the same class: "
    f"{int(agree.sum())} of {int(valid.sum())} ({100 * agree.sum() / valid.sum():.0f} %)"
)
for value, name in zip(fine.attrs["flag_values"], fine.attrs["flag_meanings"].split()):
    share = float((fine == value).mean()) * 100
    if share:
        print(f"{name.replace('_', ' '):32} {share:5.1f} % of the 10 arcsec box")
print(f"pixels compared: {int(np.prod(coarse.shape))}")
