# esd-requires: earthengine
"""
NLCD land cover, annual (USGS)
==============================

The National Land Cover Database maps the conterminous United States at 30 m
in sixteen classes. The official release is frozen at the 2001-2021 epochs;
Annual NLCD (collection 1) redid the whole record as one map per year from
1985 to 2024, so land cover can be followed through time instead of compared
between two epochs.

Both routes are on Earth Engine and need ``esd.auth.login("earthengine")``:
``source="gee-annual"`` (default) is the community mirror of Annual NLCD,
``source="gee"`` the official 2021 release with its science products and
impervious-surface descriptors. The class table is read from the asset at run
time and attached as CF flags, so ``esd.plotting.categorical`` draws the
USGS palette.

The first figure is the first and the last year of the record side by side.
The second follows the area of the main classes through all forty years: the
Cascades' evergreen forest is a patchwork of clear-cuts and regrowth, and the
annual record shows the shrub and grass that appear where a stand was cut and
the forest that returns twenty years later.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=30)  # every map below is drawn on this grid
print(box)
print(grid)

# %%
# Where the product comes from: two Earth Engine assets, both need an account.
for src in esd.catalog.get("nlcd").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The whole annual record for the box: one 30 m map per year on the asset's
# native Albers grid, 40 years in one lazy array. Selecting a year gives a
# 2-D map that keeps its date as a scalar coordinate.
# The box is loaded with a 1 km margin so the AOI's UTM grid the maps use is
# fully covered; the class areas below are counted on the native Albers pixels.
landcover_da = esd.land.nlcd.load(box.buffer(1000), time="1985/2024")
print(landcover_da)
first_da, last_da = landcover_da.isel(time=0), landcover_da.isel(time=-1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
esd.plotting.categorical(
    first_da.odc.reproject(grid, resampling="nearest"),
    ax=axes[0],
    legend=False,
    title="Annual NLCD 1985",
)
esd.plotting.categorical(
    last_da.odc.reproject(grid, resampling="nearest"),
    ax=axes[1],
    title="Annual NLCD 2024",
)

# %%
# Class area through time, as the change since 1985. A pixel is 900 m², so a
# class count per year is an area. Most of the box is Mount Rainier National
# Park, which is not logged, but the national forest around it was clear-cut
# heavily before the 1990s: the shrub/scrub of those cuts shrinks year by year
# as the stands close back into evergreen forest, and the two curves are
# close to mirror images. Perennial ice/snow does not move at all.
classes_df = esd.processing.categorical.flags(landcover_da).set_index("value")
follow = {
    42: "Evergreen forest",
    52: "Shrub/scrub",
    71: "Grassland/herbaceous",
    12: "Perennial ice/snow",
}
fig, ax = plt.subplots(figsize=(9, 4.2))
for value, name in follow.items():
    area_km2_da = (landcover_da == value).sum(dim=("y", "x")) * 900 / 1e6
    print(
        f"{name:22} {float(area_km2_da.isel(time=0)):7.1f} km² in 1985, "
        f"{float(area_km2_da.isel(time=-1)):7.1f} km² in 2024"
    )
    ax.plot(
        landcover_da["time"].values,
        (area_km2_da - area_km2_da.isel(time=0)).values,
        label=name,
        color=classes_df.loc[value, "color"],
        linewidth=1.8,
    )
ax.axhline(0, color="0.3", linewidth=0.8)
ax.set_ylabel("area change since 1985 [km²]")
ax.set_title("Annual NLCD class area, change since 1985")
ax.grid(True, color="0.85", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
