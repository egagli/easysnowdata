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

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=375)  # every VIIRS map below is drawn on it
print(box)
print(grid)

# %%
# Where the product comes from, and what the route asks for.
for src in esd.catalog.get("viirs-snow").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The gap-filled product with its persistence field. The loader keeps the
# native sinusoidal grid, which is sheared at this longitude (north is not
# up), so the box is loaded with a 1 km margin and both fields are drawn on
# the AOI's UTM grid at 375 m, resampled nearest-neighbour so the coded bytes
# and day counts survive.
viirs_ds = esd.snow.viirs.load(
    box.buffer(1000),
    week,
    product="VNP10A1F",
    variables=["CGF_NDSI_Snow_Cover", "Cloud_Persistence"],
)
print(viirs_ds)
ndsi_da = (
    viirs_ds["CGF_NDSI_Snow_Cover"].odc.reproject(grid, resampling="nearest").compute()
)
age_da = (
    viirs_ds["Cloud_Persistence"].odc.reproject(grid, resampling="nearest").compute()
)

# %%
# The NDSI on one day, and how many consecutive cloudy days the value behind
# each pixel has been carried through. In this week some pixels on the
# mountain had not been seen clearly for over a month, which is the number to
# check before trusting a gap-filled value.
scene_da = ndsi_da.sel(time=day)
valid_da = scene_da <= 100
print(
    f"oldest observation carried forward on {day}: {int(age_da.sel(time=day).max())} days"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.map(
    scene_da.where(valid_da),
    ax=axes[0],
    cmap="Blues",
    vmin=0,
    vmax=100,
    title=f"VNP10A1F NDSI, {day}",
    cbar_label="NDSI [%]",
)
esd.plotting.map(
    age_da.sel(time=day),
    ax=axes[1],
    cmap="magma_r",
    vmin=0,
    vmax=35,
    title=f"cloud persistence, {day}",
    cbar_label="cloud persistence [days]",
)
fig.tight_layout()

# %%
# The MODIS gap-filled product for the same week, at 500 m, on the same
# sinusoidal grid (four VIIRS pixels span three MODIS pixels). It is drawn on
# the same UTM zone at its own 500 m, and the box fractions below come from
# the two UTM grids.
modis_ds = esd.snow.modis.load(box.buffer(1000), week, product="MOD10A1F")
modis_ndsi_da = (
    modis_ds["CGF_NDSI_Snow_Cover"]
    .odc.reproject(box.to_geobox(crs="utm", resolution=500), resampling="nearest")
    .compute()
)


def binary(ndsi_bytes_da):
    """Snow at NDSI ≥ 40 %, NaN where the byte is a sentinel."""
    snow_da = (ndsi_bytes_da >= 40).where(ndsi_bytes_da <= 100)
    return esd.processing.set_flags(
        snow_da,
        [0, 1],
        ["no snow", "snow"],
        ["#a6611a", "#2166ac"],
        long_name="snow (NDSI ≥ 40 %)",
    )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.categorical(binary(scene_da), ax=axes[0], title=f"VIIRS 375 m, {day}")
esd.plotting.categorical(
    binary(modis_ndsi_da.sel(time=day)), ax=axes[1], title=f"MODIS 500 m, {day}"
)
fig.tight_layout()

# %%
# The snow-covered fraction of the box from both sensors on one calendar axis.
fractions_da = xr.concat(
    [
        binary(ndsi_da).mean(dim=["y", "x"]),
        binary(modis_ndsi_da).mean(dim=["y", "x"]),
    ],
    dim=pd.Index(["VIIRS VNP10A1F, 375 m", "MODIS MOD10A1F, 500 m"], name="sensor"),
)
fractions_da.attrs.update(
    {"long_name": "snow-covered fraction of the box", "units": "1"}
)

ax = esd.plotting.timeseries(fractions_da, marker="o")
ax.set_ylim(0, 1)
ax.figure.tight_layout()

difference_da = fractions_da.sel(sensor="VIIRS VNP10A1F, 375 m") - fractions_da.sel(
    sensor="MODIS MOD10A1F, 500 m"
)
print(
    f"mean VIIRS minus MODIS snow fraction over the week: {float(difference_da.mean()):+.3f}"
)
print(f"largest daily gap: {float(np.abs(difference_da).max()):.3f}")
