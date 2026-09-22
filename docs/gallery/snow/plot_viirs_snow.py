# esd-requires: earthdata
"""
VIIRS snow cover (VNP10A1F)
===========================

The VIIRS snow products from Suomi-NPP (VNP) and NOAA-20 (VJ1) at 375 m on
the same sinusoidal grid as MODIS, version 2: the daily ``VNP10A1`` NDSI snow
cover and the cloud-gap-filled daily ``VNP10A1F``, which carries the last
clear observation forward and records how old it is in ``Cloud_Persistence``.
The byte convention is MODIS's: 0-100 is the NDSI, values above 100 are the
sentinels named in the array's CF flags.

One route, ``source="nsidc"``, behind Earthdata Login. The granules are
HDF-EOS5 files, downloaded once into the cache and opened through the HDF5
driver with the grid taken from their ``StructMetadata``.

The figures show a week of the gap-filled product over Mount Rainier: the
NDSI and the age of the observation behind it on one day, VIIRS beside the
MODIS gap-filled product on the same day, and the two basin snow fractions on
one calendar axis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA
week = "2023-03-01/2023-03-07"
day = "2023-03-04"

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the product comes from, and what the route asks for.
for src in esd.catalog.get("viirs-snow").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The gap-filled product with its persistence field.
viirs = esd.snow.viirs.load(
    aoi,
    week,
    product="VNP10A1F",
    variables=["CGF_NDSI_Snow_Cover", "Cloud_Persistence"],
)
print(viirs)
ndsi = viirs["CGF_NDSI_Snow_Cover"].compute()
age = viirs["Cloud_Persistence"].compute()

# %%
# The NDSI on one day, and how many consecutive cloudy days the value behind
# each pixel has been carried through. In this week some pixels on the
# mountain had not been seen clearly for over a month, which is the number to
# check before trusting a gap-filled value.
scene = ndsi.sel(time=day)
valid = scene <= 100
print(
    f"oldest observation carried forward on {day}: {int(age.sel(time=day).max())} days"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.map(
    scene.where(valid),
    ax=axes[0],
    cmap="Blues",
    vmin=0,
    vmax=100,
    title=f"VNP10A1F NDSI, {day}",
    cbar_label="NDSI [%]",
)
esd.plotting.map(
    age.sel(time=day),
    ax=axes[1],
    cmap="magma_r",
    vmin=0,
    vmax=35,
    title=f"cloud persistence, {day}",
    cbar_label="cloud persistence [days]",
)
fig.tight_layout()

# %%
# The MODIS gap-filled product for the same week, at 500 m. Four VIIRS pixels
# span three MODIS pixels, on the same sinusoidal grid, so the two can be
# compared over any box without reprojecting either.
modis = esd.snow.modis.load(aoi, week, product="MOD10A1F")
modis_ndsi = modis["CGF_NDSI_Snow_Cover"].compute()


def binary(ndsi_bytes):
    """Snow at NDSI ≥ 40 %, NaN where the byte is a sentinel."""
    snow = (ndsi_bytes >= 40).where(ndsi_bytes <= 100)
    return esd.processing.set_flags(
        snow,
        [0, 1],
        ["no snow", "snow"],
        ["#a6611a", "#2166ac"],
        long_name="snow (NDSI ≥ 40 %)",
    )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.categorical(binary(scene), ax=axes[0], title=f"VIIRS 375 m, {day}")
esd.plotting.categorical(
    binary(modis_ndsi.sel(time=day)), ax=axes[1], title=f"MODIS 500 m, {day}"
)
fig.tight_layout()

# %%
# The snow-covered fraction of the box from both sensors on one calendar axis.
fractions = xr.concat(
    [
        binary(ndsi).mean(dim=["y", "x"]),
        binary(modis_ndsi).mean(dim=["y", "x"]),
    ],
    dim=pd.Index(["VIIRS VNP10A1F, 375 m", "MODIS MOD10A1F, 500 m"], name="sensor"),
)
fractions.attrs.update({"long_name": "snow-covered fraction of the box", "units": "1"})

ax = esd.plotting.timeseries(fractions, marker="o")
ax.set_ylim(0, 1)
ax.figure.tight_layout()

difference = fractions.sel(sensor="VIIRS VNP10A1F, 375 m") - fractions.sel(
    sensor="MODIS MOD10A1F, 500 m"
)
print(
    f"mean VIIRS minus MODIS snow fraction over the week: {float(difference.mean()):+.3f}"
)
print(f"largest daily gap: {float(np.abs(difference).max()):.3f}")
