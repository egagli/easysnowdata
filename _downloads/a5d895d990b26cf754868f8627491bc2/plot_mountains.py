"""
Mountain ranges: GMBA Mountain Inventory v2
===========================================

The Global Mountain Biodiversity Assessment inventory (Snethlage et al. 2022)
names 8,327 mountain ranges and nests them in a hierarchy up to ten levels
deep. ``esd.boundaries.mountains.load`` reads it in three subsets:

* ``subset="basic"`` (default) — the smallest non-overlapping units;
* ``subset="300"`` — 291 non-overlapping major systems;
* ``subset="all"`` — every range at every level, overlapping; ``level=`` picks one;

and two extents: ``"standard"`` follows the GMBA mountain definition,
``"broad"`` reaches into the surrounding terrain. Each is one zipped shapefile
on EarthEnv, fetched once into the cache; no account is needed.

The figures show the basic units and the Cascade Range over the Natural Earth
hillshade, the hierarchy above Mount Rainier, and, per range, how much of it
the Wrzesien mask classifies as seasonal mountain snow.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import easysnowdata as esd

aoi = (-123.0, 46.0, -120.5, 48.0)  # the central Cascades, Puget Sound to Yakima
rainier = (-121.94, 46.72, -121.54, 46.99)

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=1000)
print(box)

for src in esd.catalog.get("gmba-mountains").sources:
    print(f"{src.id:10} {src.title:10} {', '.join(src.requires) or 'no account'}")

# %%
# The basic units intersecting the AOI, the one major system they belong to,
# and the standard against the broad extent of that system. Every frame starts
# with ``name``, ``gmba_id``, ``level`` and ``path``.
basic_gdf = esd.boundaries.mountains.load(aoi)
major_gdf = esd.boundaries.mountains.load(aoi, subset="300")
broad_gdf = esd.boundaries.mountains.load(aoi, subset="300", extent="broad")
print(f"{len(basic_gdf)} basic units; major systems: {', '.join(major_gdf['name'])}")
print(basic_gdf[["name", "level"]].sort_values("name").to_string(index=False))

hillshade_da = esd.terrain.hillshade.load(box.buffer(40_000), style="shaded-relief")
ax = esd.plotting.map(
    hillshade_da.odc.reproject(grid, resampling="bilinear"),
    cmap="gray",
    vmin=0,
    vmax=255,
    colorbar=False,
    title="GMBA basic units (colour), the Cascade Range standard and broad",
    figsize=(10, 8),
)
esd.plotting.add_outline(
    ax,
    basic_gdf,
    column="name",
    cmap="tab20",
    alpha=0.45,
    edgecolor="0.3",
    linewidth=0.4,
)
esd.plotting.add_outline(ax, major_gdf, edgecolor="black", linewidth=2.0)
esd.plotting.add_outline(
    ax, broad_gdf, edgecolor="black", linewidth=1.2, linestyle="--"
)
ax.figure.tight_layout()

# %%
# The hierarchy above Rainier: ``subset="all"`` holds every level, so the
# ranges containing the mountain run from the continent down to the massif.
# ``level=`` keeps one of them; level 4 is the Cascade Range.
nested_gdf = esd.boundaries.mountains.load(rainier, subset="all")
ladder_gdf = nested_gdf[nested_gdf.contains(esd.parse_aoi(rainier).geometry.centroid)]
print(ladder_gdf.sort_values("level")[["level", "name"]].to_string(index=False))
print(esd.boundaries.mountains.load(rainier, subset="all", level=4)["name"].tolist())

# %%
# Where the snow is: the fraction of each basic unit that the Wrzesien mask
# classifies as seasonal mountain snow (MODIS 2000-2016). The ranges along the
# crest are mostly seasonal; the foothills to the west and east are not.
mask_da = esd.snow.mountain_snow_mask.load(box.buffer(5000), chunks=None)
lon, lat = np.meshgrid(mask_da["longitude"].values, mask_da["latitude"].values)
pixels_df = pd.DataFrame(
    {"lon": lon.ravel(), "lat": lat.ravel(), "class": mask_da.values.ravel()}
)
pixels_gdf = gpd.GeoDataFrame(
    pixels_df,
    geometry=gpd.points_from_xy(pixels_df["lon"], pixels_df["lat"]),
    crs="EPSG:4326",
)
joined_gdf = pixels_gdf.sjoin(basic_gdf[["name", "geometry"]], predicate="within")
seasonal_df = (
    joined_gdf.groupby("name")["class"]
    .apply(lambda classes: float((classes == 3).mean()))
    .rename("seasonal_fraction")
    .reset_index()
)
ranges_gdf = basic_gdf.merge(seasonal_df, on="name")
print(
    ranges_gdf.sort_values("seasonal_fraction", ascending=False)[
        ["name", "seasonal_fraction"]
    ].to_string(index=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
esd.plotting.map(
    hillshade_da.odc.reproject(grid, resampling="bilinear"),
    ax=ax,
    cmap="gray",
    vmin=0,
    vmax=255,
    colorbar=False,
    title="Seasonal mountain snow per GMBA unit (Wrzesien et al. 2019)",
)
esd.plotting.add_outline(
    ax,
    ranges_gdf,
    column="seasonal_fraction",
    cmap="Blues",
    vmin=0,
    vmax=1,
    alpha=0.8,
    edgecolor="0.3",
    linewidth=0.4,
    legend=True,
    legend_kwds={"label": "fraction of pixels with seasonal snow", "shrink": 0.7},
)
fig.tight_layout()
