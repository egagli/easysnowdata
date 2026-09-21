# esd-requires: earthdata
"""
HLS surface reflectance (Landsat and Sentinel-2)
================================================

NASA's Harmonized Landsat Sentinel-2 (HLS) v2.0 puts Landsat 8/9 (the L30
product) and Sentinel-2 (S30) on one 30 m grid with a common atmospheric
correction, band-pass adjustment and BRDF normalisation, so the two
constellations read as one 2–3 day revisit. Each granule carries the Fmask
quality layer and the sun and view angles.

The default route is NASA's CMR-STAC ``LPCLOUD`` catalog, the archive of
record: the search is open, but the COG reads need an Earthdata Login (a
``~/.netrc`` entry or ``EARTHDATA_TOKEN``). ``source="planetary-computer"``
reads Microsoft's credential-free mirror of the same files.

The figures show a Sentinel-2 day and a Landsat day one day apart as
true-colour composites, as NDSI and as Fmask classes, and the last cell checks
the mirror against the archive of record.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)  # the Nisqually glacier side of Rainier

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("hls").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# Search both products at once. The frame says which product and which
# satellite each granule came from, read from STAC properties rather than
# from the per-granule XML the old class scraped.
items = esd.optical.hls.search(aoi, "2023-08-15/2023-08-16", cloud_cover=20)
print(items[["datetime", "product", "platform", "eo:cloud_cover"]].to_string())

# %%
# Load both days into one Dataset stacked on ``time``, with a ``product``
# coordinate. No Fmask mask is applied here, deliberately: the cell after the
# figure shows why.
hls = esd.optical.hls.load(
    aoi, items=items, bands=["blue", "green", "red", "swir16", "Fmask"]
).compute()
print(hls["time"].values, hls["product"].values)

ndsi = (hls["green"] - hls["swir16"]) / (hls["green"] + hls["swir16"])
ndsi.attrs = {"long_name": "NDSI (green − SWIR 1.6 µm)"}

# %%
# Fmask is a bit field (cirrus, cloud, adjacent, shadow, snow/ice, water, and
# two aerosol bits), not a class map. To draw it, one class per pixel is
# picked in priority order and given CF flag attributes so the legend comes
# from the array itself.
bits = esd.processing.masks.FMASK_BITS
fmask = hls["Fmask"]
classes = np.zeros(fmask.shape, dtype="uint8")
for value, name in [
    (1, "water"),
    (2, "snow_ice"),
    (3, "adjacent"),
    (4, "cloud_shadow"),
    (5, "cirrus"),
    (6, "cloud"),
]:
    classes[esd.processing.masks.fmask_bit(fmask, bits[name]).values == 1] = value
fmask_class = esd.processing.set_flags(
    fmask.copy(data=classes),
    values=range(7),
    meanings=["clear", "water", "snow or ice", "adjacent to cloud", "cloud shadow", "cirrus", "cloud"],
    colors=["#d9d9d9", "#0000ff", "#ff96ff", "#c0c0c0", "#643200", "#64c8ff", "#ffffff"],
    long_name="Fmask class",
)  # fmt: skip

# %%
# One row per day: Sentinel-2 on the 15th, Landsat on the 16th. The composite
# uses a fixed 0–0.5 reflectance stretch, stated rather than fitted, so the two
# rows are comparable.
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for row, step in enumerate(range(hls.sizes["time"])):
    scene = hls.isel(time=step)
    when = str(scene["time"].values)[:10]
    label = f"{str(scene['product'].values)}, {when}"
    rgb = scene[["red", "green", "blue"]].to_array("band").clip(0, 0.5) / 0.5
    rgb.plot.imshow(rgb="band", ax=axes[row, 0], add_labels=False)
    axes[row, 0].set_title(f"True colour (0–0.5 reflectance), {label}")
    esd.plotting.finish_map(axes[row, 0], hls.rio.crs)
    esd.plotting.map(
        ndsi.isel(time=step), ax=axes[row, 1], cmap="Blues", vmin=-0.5, vmax=1.0,
        title=f"NDSI, {label}", cbar_label="NDSI",
    )  # fmt: skip
    esd.plotting.categorical(
        fmask_class.isel(time=step), ax=axes[row, 2], title=f"Fmask, {label}"
    )
for ax in axes[:, 1:].ravel():
    ax.set_ylabel("")  # the panels share their latitude axis
fig.tight_layout()

# %%
# Why masking is opt-in. The composites show two cloud-free days, yet Fmask
# reports cloud, cloud shadow and cloud-adjacent pixels on the Landsat day,
# over bright rock and the sunlit snow beside it, and calls shaded snow
# "water" on the Sentinel-2 day. Fmask was tuned for vegetated lowlands; over
# snow and ice it is a hint, not a mask. ``mask="fmask-default"`` would drop
# the flagged pixels below, so use the snow/ice bit (or NDSI) and mask clouds
# only where you have checked what Fmask does.
cloudy = (
    sum(
        esd.processing.masks.fmask_bit(fmask, bits[name]) == 1
        for name in ("cloud", "cloud_shadow", "adjacent", "cirrus")
    )
    > 0
)
water = esd.processing.masks.fmask_bit(fmask, bits["water"]) == 1
for step in range(hls.sizes["time"]):
    valid = fmask.isel(time=step) != esd.optical.hls.FMASK_NODATA
    n = float(valid.sum())
    print(
        f"{str(hls['product'].values[step])} {str(hls['time'].values[step])[:10]}: "
        f"{float(cloudy.isel(time=step).sum()) / n:.0%} of pixels flagged cloud, "
        f"shadow or adjacent; {float(water.isel(time=step).sum()) / n:.0%} water"
    )

# %%
# The credential-free mirror serves the same granules under its own item ids,
# so it is searched again rather than handed the CMR items. Reading the green
# band of both days from Planetary Computer and differencing against the
# archive of record shows they are the same files: the difference is zero
# everywhere.
mirror = esd.optical.hls.load(
    aoi,
    "2023-08-15/2023-08-16",
    bands=["green"],
    source="planetary-computer",
    cloud_cover=20,
).compute()
difference = hls["green"].values - mirror["green"].values
print(mirror.attrs["source"], mirror.attrs["collections"])
print(f"max |CMR − mirror| in green reflectance: {np.nanmax(np.abs(difference)):.4f}")
