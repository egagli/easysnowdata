"""
Sentinel-2 snow index, and the two catalogs side by side
========================================================

Sentinel-2 L2A comes from Planetary Computer by default and from Earth Search
with ``source="earth-search"``. Both are credential-free; the loader puts
every scene on the pre-2022 baseline, so the two agree pixel for pixel.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)  # the Nisqually glacier side of Rainier

# %%
# Search first: the result is a GeoDataFrame you can filter before loading.
items = esd.optical.sentinel2.search(aoi, "2023-08-01/2023-08-12", cloud_cover=40)
print(items[["datetime", "eo:cloud_cover", "collection"]].head())

# %%
# Load the clearest scene, masked with the default SCL classes, and take NDSI.
clearest = items.sort_values("eo:cloud_cover").iloc[:1]
s2 = esd.optical.sentinel2.load(
    aoi, items=clearest, bands=["green", "swir16", "red", "scl"], mask="scl-default"
)
ndsi = esd.processing.ndsi(s2).isel(time=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ndsi.plot.imshow(ax=axes[0], cmap="Blues", vmin=-0.5, vmax=1.0)
axes[0].set_title("NDSI")
esd.plotting.categorical(
    s2["scl"].isel(time=0), ax=axes[1], title="Scene classification"
)
for ax in axes:
    ax.set_aspect("equal")
fig.tight_layout()

# %%
# The same day from both catalogs. Earth Search's ``sentinel-2-l2a`` pixels
# already carry the baseline correction and Planetary Computer's do not, so
# the loader reads each catalog's metadata rather than applying one date rule
# everywhere; the medians below agree to well under a percent of reflectance.
window = dict(bands=["red"], resolution=60, crs="EPSG:32610")
pc = esd.optical.sentinel2.load(aoi, "2023-08-10/2023-08-11", **window)
es = esd.optical.sentinel2.load(
    aoi, "2023-08-10/2023-08-11", source="earth-search", **window
)
for name, ds in (("planetary-computer", pc), ("earth-search", es)):
    median = float(np.nanmedian(ds["red"].isel(time=0).compute().values))
    print(f"{name}: median red reflectance {median:.4f}")
