"""
VIIRS snow cover, the successor to MODIS
========================================

VIIRS carries the snow-cover record forward at 375 m as Terra winds down. The
byte convention is the same as MODIS, so the same thresholding function works
on both and the two can be compared directly over a winter.

Both products need an Earthdata account; the search and the read go through
the same ``earthdata`` provider.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)
when = "2023-03-01/2023-03-08"

# %%
# The cloud-gap-filled daily product. ``Cloud_Persistence`` says how old the
# gap-filled value is, which is the number to check before trusting a pixel.
viirs = esd.snow.viirs.load(
    aoi,
    when,
    product="VNP10A1F",
    variables=["CGF_NDSI_Snow_Cover", "Cloud_Persistence"],
)
print(viirs)

snow = esd.processing.binary_snow(
    viirs["CGF_NDSI_Snow_Cover"], product="VNP10A1F", threshold=40
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
snow.isel(time=0).plot.imshow(ax=axes[0], cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("VIIRS binary snow, 375 m")
viirs["Cloud_Persistence"].isel(time=0).plot.imshow(ax=axes[1], cmap="magma_r")
axes[1].set_title("Days since the last clear view")
for ax in axes:
    ax.set_aspect("equal")
fig.tight_layout()

# %%
# The same week from MODIS, at 500 m. The two records overlap from 2012, which
# is what makes a continuity check possible before Terra stops.
modis = esd.snow.modis.load(aoi, when, product="MOD10A1F")
modis_snow = esd.processing.binary_snow(
    modis["CGF_NDSI_Snow_Cover"], product="MOD10A1F", threshold=40
)

fig, ax = plt.subplots(figsize=(8, 3.5))
for label, series in (
    ("VIIRS 375 m", snow.mean(dim=["y", "x"]).compute()),
    ("MODIS 500 m", modis_snow.mean(dim=["y", "x"]).compute()),
):
    series.plot(ax=ax, marker="o", label=label)
ax.set_ylabel("snow-covered fraction")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Snow-covered fraction, VIIRS against MODIS")
fig.tight_layout()

print(
    "mean difference:",
    float(
        np.nanmean(snow.mean(dim=["y", "x"]).values)
        - np.nanmean(modis_snow.mean(dim=["y", "x"]).values)
    ),
)
