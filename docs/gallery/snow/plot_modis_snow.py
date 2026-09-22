# esd-requires: earthdata
"""
MODIS snow cover (MOD10A1, MOD10A1F, MOD10A2)
=============================================

The MODIS snow products from Terra (MOD) and Aqua (MYD) at 500 m on the
sinusoidal grid, Collection 6.1: the daily ``MOD10A1`` NDSI snow cover, the
cloud-gap-filled daily ``MOD10A1F``, and the 8-day maximum snow extent
``MOD10A2``. Since Collection 6 the daily product reports the NDSI itself
(0-100) rather than a fixed-threshold snow map, so the threshold is yours;
values above 100 are sentinels (cloud, night, water, fill) carried in the CF
``flag_values`` / ``flag_meanings`` attributes of the array.

Two routes. ``source="nsidc"`` (the default) downloads the HDF-EOS2 granules
through Earthdata Login and needs a GDAL with the HDF4 driver; it is the only
route with the cloud-gap-filled product and the Aqua siblings.
``source="planetary-computer"`` reads the ``MOD10A1`` / ``MOD10A2`` COG
mirrors with no credentials.

The figures show a clear day of NDSI and its binary snow map, the daily
clear-sky and snow fractions of the box, an 8-day maximum extent, the
cloud-gap-filled product on a day the raw product saw only cloud, and the
difference between the two routes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA
week = "2023-03-15/2023-03-22"
clear_day = "2023-03-18"
cloudy_day = "2023-03-19"

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=500)  # every map below is drawn on this grid
print(box)
print(grid)

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("modis-snow").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# A week of the daily product from the credential-free mirror. The loader
# keeps the raw byte with its CF flags rather than deciding what you want, and
# keeps the native sinusoidal grid, which at this longitude is sheared: north
# is not up and a lat/lon box is a parallelogram on it. The maps are therefore
# drawn on the AOI's UTM grid, with nearest-neighbour resampling so the coded
# bytes survive; the box is loaded with a 1 km margin so that grid is fully
# covered. The native array is kept for the byte comparison at the end.
daily = esd.snow.modis.load(
    box.buffer(1000), week, product="MOD10A1", source="planetary-computer"
)
native = daily["NDSI_Snow_Cover"].compute()
ndsi = native.odc.reproject(grid, resampling="nearest")
print(ndsi.attrs["flag_values"])
print(ndsi.attrs["flag_meanings"])

# %%
# The NDSI on a clear day, and the binary snow map at the heritage threshold
# of 40 %. ``valid`` drops the sentinels, so cloud and water are NaN, not 0.
scene = ndsi.sel(time=clear_day)
valid = scene <= 100
snow = (scene >= 40).where(valid)
snow = esd.processing.set_flags(
    snow,
    [0, 1],
    ["no snow", "snow"],
    ["#a6611a", "#2166ac"],
    long_name="snow (NDSI ≥ 40 %)",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.map(
    scene.where(valid),
    ax=axes[0],
    cmap="Blues",
    vmin=0,
    vmax=100,
    title=f"NDSI, {clear_day}",
)
esd.plotting.categorical(snow, ax=axes[1], title=f"binary snow, {clear_day}")
fig.tight_layout()

# %%
# How much of the box each day could see, and how much of what it saw was
# snow. Cloud is the story in March in the Cascades: one day saw nothing.
clear = (ndsi <= 100).mean(dim=["y", "x"])
snowy = ((ndsi >= 40) & (ndsi <= 100)).sum(dim=["y", "x"]) / (ndsi <= 100).sum(
    dim=["y", "x"]
)
fractions = xr.concat(
    [clear, snowy],
    dim=pd.Index(
        ["clear-sky fraction", "snow fraction of clear pixels"], name="quantity"
    ),
)
fractions.attrs.update({"long_name": "fraction of the box", "units": "1"})

ax = esd.plotting.timeseries(fractions, marker="o")
ax.set_ylim(0, 1.05)
ax.figure.tight_layout()

# %%
# The 8-day maximum snow extent is a classified product, so it is drawn from
# its class table. Its periods start on fixed days of the year (1, 9, 17, ...),
# so the composite dated 14 March covers 14-21 March.
extent = esd.snow.modis.load(
    box.buffer(1000), week, product="MOD10A2", source="planetary-computer"
)
composite = (
    extent["Maximum_Snow_Extent"].isel(time=0).odc.reproject(grid, resampling="nearest")
)
print(f"composites in the window: {extent['time'].values.astype('datetime64[D]')}")

ax = esd.plotting.categorical(composite, title="MOD10A2 maximum snow extent")
ax.figure.tight_layout()

# %%
# The cloud-gap-filled product from NSIDC carries the last clear observation
# forward through cloudy days. On the day the raw product saw only cloud it
# still has a snow map.
short = f"{cloudy_day}/{cloudy_day}"
filled = esd.snow.modis.load(box.buffer(1000), short, product="MOD10A1F")
cgf = (
    filled["CGF_NDSI_Snow_Cover"]
    .sel(time=cloudy_day)
    .odc.reproject(grid, resampling="nearest")
    .compute()
)
raw = ndsi.sel(time=cloudy_day)
print(f"raw MOD10A1 valid pixels on {cloudy_day}: {int((raw <= 100).sum())}")
print(f"MOD10A1F valid pixels on {cloudy_day}: {int((cgf <= 100).sum())}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.map(
    raw.where(raw <= 100),
    ax=axes[0],
    cmap="Blues",
    vmin=0,
    vmax=100,
    title=f"MOD10A1, {cloudy_day}",
    cbar_label="NDSI [%]",
)
esd.plotting.map(
    cgf.where(cgf <= 100),
    ax=axes[1],
    cmap="Blues",
    vmin=0,
    vmax=100,
    title=f"MOD10A1F, {cloudy_day}",
    cbar_label="NDSI [%]",
)
fig.tight_layout()

# %%
# The two routes, same granule: NSIDC's HDF-EOS2 file against the mirror's
# COG on the clear day. On the native sinusoidal grid they land on the same
# pixels to within a few micrometres, so the bytes can be compared directly
# and should be identical; the comparison uses the native arrays and only the
# difference is reprojected for drawing.
archive = esd.snow.modis.load(
    box.buffer(1000), f"{clear_day}/{clear_day}", product="MOD10A1"
)
nsidc = archive["NDSI_Snow_Cover"].sel(time=clear_day).compute()
mirror = native.sel(time=clear_day)
nsidc = nsidc.reindex_like(mirror, method="nearest", tolerance=1)
identical = float((nsidc.values == mirror.values).mean()) * 100
print(f"bytes identical between NSIDC and Planetary Computer: {identical:.2f} %")

diff = (nsidc.astype("int16") - mirror.astype("int16")).where(
    (mirror <= 100) & (nsidc <= 100)
)
diff.attrs.update({"long_name": "NDSI, NSIDC minus Planetary Computer", "units": "%"})
ax = esd.plotting.map(
    diff.odc.reproject(grid, resampling="nearest"),
    cmap="RdBu",
    vmin=-10,
    vmax=10,
    title=f"NSIDC minus Planetary Computer, {clear_day}",
    cbar_label="NDSI difference [%]",
)
ax.figure.tight_layout()
print(f"largest absolute difference: {float(np.nanmax(np.abs(diff.values))):.0f} %")
