"""
Forest cover fraction (CGLS-LC100)
==================================

The Copernicus Global Land Service LC100 collection 3 gives, for every 100 m
pixel, the fraction of the ground covered by tree canopy, as a percentage.
Epoch 2019 is the last the service produced. Where a land-cover class says
"tree cover" or not, this says how much, which is what an optical snow product
needs: snow under a closed canopy is snow a satellite cannot see.

The default route (``source="zenodo"``) reads one global GeoTIFF on Zenodo
with range requests and needs no account. ``source="gee"`` is the Earth
Engine image collection, which adds the 2015-2018 epochs and the other LC100
cover fractions (``esd.auth.login("earthengine")``).

The first figure is the map. The second puts tree cover against elevation
from the Copernicus DEM, also credential-free, and finds the treeline: the
elevation band where the canopy fraction falls to zero.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the product comes from: the Zenodo GeoTIFF needs nothing, the Earth
# Engine collection needs an account.
for src in esd.catalog.get("forest-cover-fraction").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The 2019 tree-cover fraction. The 255 sentinel the product uses outside its
# footprint is masked to NaN and kept as the encoded nodata.
forest = esd.land.forest_cover.load(aoi, chunks=None)
print(forest)
print(
    f"pixels without a value (255 in the source): {float(forest.isnull().mean()):.1%}"
)
esd.plotting.map(
    forest,
    cmap="Greens",
    vmin=0,
    vmax=100,
    title="CGLS-LC100 tree cover fraction, 2019",
)

# %%
# Tree cover against elevation. The Copernicus DEM is 1 arc-second on the same
# geographic grid family, so it is sampled onto the 100 m forest grid with a
# bilinear interpolation; the pixels are then binned by 100 m of elevation.
dem = esd.terrain.dem.load(aoi, chunks=None).interp_like(forest, method="linear")
edges = np.arange(500, 3100, 100)
centres = edges[:-1] + 50
elevation = dem.values.ravel()
cover = forest.values.ravel()
valid = np.isfinite(elevation) & np.isfinite(cover)
which = np.digitize(elevation[valid], edges) - 1
median, low, high = (np.full(len(centres), np.nan) for _ in range(3))
for i in range(len(centres)):
    values = cover[valid][which == i]
    if len(values) >= 20:
        low[i], median[i], high[i] = np.percentile(values, [25, 50, 75])

bare = centres[np.nan_to_num(median, nan=100) < 5]
treeline = float(bare.min())
print(f"lowest 100 m band with median tree cover below 5 %: {treeline:.0f} m")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_betweenx(centres, low, high, color="#74c476", alpha=0.4, label="25-75 %")
ax.plot(median, centres, color="#006d2c", linewidth=2, label="median")
ax.axhline(treeline, color="0.3", linestyle="--", linewidth=1)
ax.text(100, treeline, f" treeline ≈ {treeline:.0f} m", va="bottom", ha="right")
ax.set_xlabel(esd.plotting.label(forest))
ax.set_ylabel(esd.plotting.label(dem, name="elevation"))
ax.set_xlim(0, 100)
ax.set_ylim(edges[0], float(np.nanmax(centres[np.isfinite(median)])) + 150)
ax.set_title("Tree cover by elevation band")
ax.legend(frameon=False, loc="upper right")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()

# %%
# The Earth Engine route, for the earlier epochs:
#
# .. code-block:: python
#
#     epochs = esd.land.forest_cover.load(aoi, source="gee", time="2015/2019")
#     epochs.mean(dim=("y", "x")).plot()
