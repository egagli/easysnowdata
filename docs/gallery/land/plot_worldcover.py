"""
ESA WorldCover land cover
=========================

ESA WorldCover is a global 10 m land cover map in eleven classes, made from
Sentinel-1 and Sentinel-2 for 2020 (``version="v100"``) and 2021
(``version="v200"``, the default and the last one the project produced). The
classes come back as CF ``flag_values`` / ``flag_meanings`` / ``flag_colors``
attributes, so ``esd.plotting.categorical`` draws the map in the product's own
palette with a legend of the classes present.

Both routes are credential-free: the Planetary Computer STAC collection
(``source="planetary-computer"``, the default) and the unsigned AWS Open Data
bucket (``source="aws-open-data"``), which reads the 3°×3° tiles directly.

The first figure is the 2021 map on a UTM grid, with the area of each class.
The second puts 2020 next to 2021: ESA's own guidance is that the two versions
were made with different algorithms and are not a change product, and the
fraction of pixels that flip between them says why.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

# %%
# Where the product comes from: two routes, neither needs an account.
for src in esd.catalog.get("esa-worldcover").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The 2021 map on the AOI's UTM zone at the native 10 m, so a pixel is
# 100 m² and class areas are a count. Tree cover fills the valleys and stops
# at the treeline; above it WorldCover sees bare rock, snow and ice, and a
# ring of grassland in the subalpine meadows between.
landcover = esd.land.landcover.load(aoi, crs="utm", grid_resolution=10, chunks=None)
print(landcover)
esd.plotting.categorical(landcover, title="ESA WorldCover v200 (2021)")

classes = esd.processing.categorical.flags(landcover)
values, counts = np.unique(landcover.values, return_counts=True)
area = pd.Series(counts * 100 / 1e6, index=values).rename("area [km²]")
table = classes.set_index("value")["meaning"].str.replace("_", " ").to_frame()
table = table.join(area, how="inner").sort_values("area [km²]", ascending=False)
print(table.round(1).to_string())

# %%
# 2020 (v100) against 2021 (v200). The versions differ in algorithm as well
# as in year, so the pixels that change class between them are mostly a
# change of method, not of land. The largest flips are between neighbouring
# classes: tree cover against grassland, in both directions, along the
# treeline, and the alpine classes (moss and lichen, bare ground, snow and
# ice) against each other on the mountain, where which one wins depends on
# the summer scenes each version saw.
earlier = esd.land.landcover.load(
    aoi, version="v100", crs="utm", grid_resolution=10, chunks=None
)
names = classes.set_index("value")["meaning"].str.replace("_", " ")
flips = pd.DataFrame({"v100": earlier.values.ravel(), "v200": landcover.values.ravel()})
flips = flips[flips["v100"] != flips["v200"]]
print(
    f"pixels that change class between v100 and v200: {len(flips) / landcover.size:.1%}"
)
top = flips.value_counts().head(6).rename("area [km²]") * 100 / 1e6
top.index = [f"{names[a]} -> {names[b]}" for a, b in top.index]
print(top.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
esd.plotting.categorical(
    earlier, ax=axes[0], legend=False, title="WorldCover v100 (2020)"
)
esd.plotting.categorical(landcover, ax=axes[1], title="WorldCover v200 (2021)")
