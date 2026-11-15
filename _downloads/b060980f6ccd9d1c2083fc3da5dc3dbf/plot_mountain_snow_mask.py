"""
Mountain snow mask (Wrzesien et al.)
====================================

Wrzesien et al. (2019) classified the MODIS MOD10A2 snow-cover record for
2000-2016 at 30 arcsec into little-to-no, ephemeral and seasonal snow, and
intersected it with a mountain definition based on GTOPO30 relief. The result
is three layers: ``mountain_snow`` (the classes over mountains only),
``snow`` (the same classes over all terrain) and ``clouds`` (how often the
composite was indeterminate because of cloud).

One route, ``source="zenodo"``, with no credentials: one zipped GeoTIFF per
layer, fetched once into the easysnowdata cache.

The figures show the mountain and all-terrain layers side by side across the
central Cascades and Puget lowlands, and the cloud-indeterminacy layer.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-123.0, 46.0, -120.5, 48.0)  # the central Cascades, Puget Sound to Yakima

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=500)  # every map below is drawn on this grid
print(box)
print(grid)

# %%
# Where the product comes from, and what the route asks for.
for src in esd.catalog.get("mountain-snow-mask").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The two class layers. In the mountain layer, ``Fill`` (255) is every pixel
# that is not mountain by the relief criterion, so the lowlands drop out and
# the mask says where seasonal snow is also mountain snow. The all-terrain
# layer keeps them and classifies the Puget lowlands as ephemeral.
# The layers are geographic; the box is loaded with a 2 km margin and each
# is drawn on the AOI's UTM grid, resampled nearest so the classes survive.
mountain_da = esd.snow.mountain_snow_mask.load(box.buffer(2000), layer="mountain_snow")
terrain_da = esd.snow.mountain_snow_mask.load(box.buffer(2000), layer="snow")
print(mountain_da.attrs["flag_meanings"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
esd.plotting.categorical(
    mountain_da.odc.reproject(grid, resampling="nearest"),
    ax=axes[0],
    title="mountains only",
)
esd.plotting.categorical(
    terrain_da.odc.reproject(grid, resampling="nearest"),
    ax=axes[1],
    title="all terrain",
)
fig.tight_layout()

values, counts = np.unique(mountain_da.values, return_counts=True)
for value, count in zip(values, counts):
    print(f"class {value:3d}: {count:6d} pixels")

# %%
# The clouds layer is the indeterminacy level (0-6) behind the classes; the
# upstream record does not define the scale further, so it is kept as an
# integer without a class table. Rainier's high terrain is where the MOD10A2
# composites were most often indeterminate.
clouds_da = esd.snow.mountain_snow_mask.load(box.buffer(2000), layer="clouds")
ax = esd.plotting.map(
    clouds_da.odc.reproject(grid, resampling="nearest"),
    cmap="magma_r",
    vmin=0,
    vmax=6,
    title="cloud indeterminacy, 2000-2016",
    cbar_label="indeterminacy level [0-6]",
)
ax.figure.tight_layout()
