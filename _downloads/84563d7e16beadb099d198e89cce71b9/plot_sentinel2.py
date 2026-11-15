"""
Sentinel-2 L2A surface reflectance
==================================

Sentinel-2 MSI Level-2A is bottom-of-atmosphere reflectance at 10–60 m, every
five days since mid-2015, with Sen2Cor's scene classification layer (SCL)
alongside the bands. It is the workhorse for mapping snow-covered area at the
scale of a single glacier.

Two credential-free routes serve it. ``source="planetary-computer"`` (the
default) reads Microsoft's ``sentinel-2-l2a`` collection;
``source="earth-search"`` reads Element 84's copy on AWS, which also holds the
Collection-1 reprocessing (``collection="sentinel-2-c1-l2a"``). Whichever
catalog a scene comes from, the loader scales the digital numbers to
reflectance and puts every acquisition on the pre-2022 radiometric baseline.

The figures show one August scene as a true-colour composite, as NDSI and as
the SCL, then the same day from both catalogs with the difference between them.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)  # the Nisqually glacier side of Rainier

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("sentinel-2-l2a").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# Search first: the result is a GeoDataFrame of STAC items you can filter
# before loading. Planetary Computer holds the original processing of each
# acquisition and ESA's 2024 reprocessing of it (two ``s2:processing_baseline``
# values per tile per day), so a day can return twice the items you expect.
# ``load`` groups by solar day, which folds them into one time step.
items_gdf = esd.optical.sentinel2.search(aoi, "2023-08-15", cloud_cover=40)
print(items_gdf[["datetime", "eo:cloud_cover", "s2:processing_baseline"]].to_string())

# %%
# Load the scene at its native 10 m, masked with the default SCL classes
# (no data, saturated, shadows, clouds, cirrus). Band arithmetic stays in the
# open: NDSI is one line of xarray.
s2_ds = esd.optical.sentinel2.load(
    aoi,
    items=items_gdf,
    bands=["blue", "green", "red", "swir16", "scl"],
    mask="scl-default",
).compute()

ndsi_da = (s2_ds["green"] - s2_ds["swir16"]) / (s2_ds["green"] + s2_ds["swir16"])
ndsi_da.attrs = {"long_name": "NDSI (green − SWIR 1.6 µm)"}
ndsi_da

# %%
# A true-colour composite is the three bands stacked on a ``band`` dimension
# and stretched to a fixed reflectance range (0–0.5 here, so snow saturates to
# white and rock keeps its texture). xarray draws it directly.
rgb_da = s2_ds[["red", "green", "blue"]].to_array("band").isel(time=0)
rgb_da = rgb_da.clip(0, 0.5) / 0.5

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
rgb_da.plot.imshow(rgb="band", ax=axes[0], add_labels=False)
axes[0].set_title("True colour (0–0.5 reflectance), 2023-08-15")
esd.plotting.finish_map(axes[0], s2_ds.rio.crs)
esd.plotting.map(
    ndsi_da,
    ax=axes[1],
    cmap="Blues",
    vmin=-0.5,
    vmax=1.0,
    title="NDSI",
    cbar_label="NDSI",
)
esd.plotting.categorical(s2_ds["scl"], ax=axes[2], title="Scene classification (SCL)")
for ax in axes[1:]:
    ax.set_ylabel("")  # the panels share their latitude axis
fig.tight_layout()

# %%
# The same day from both catalogs. Since processing baseline 04.00 (25 January
# 2022) ESA adds 1000 to every L2A digital number. Earth Search's items say so
# in ``raster:bands`` and its pixels already have the offset removed
# (``earthsearch:boa_offset_applied``); Planetary Computer's items carry the
# raw values, so the loader undoes the offset by acquisition date. Without that
# step the two catalogs would disagree by 0.1 in reflectance everywhere, which
# in a SWIR band is the difference between snow and rock.
window = dict(bands=["green", "swir16"], resolution=20)
pc_ds = esd.optical.sentinel2.load(aoi, "2023-08-15", **window).compute()
es_ds = esd.optical.sentinel2.load(
    aoi, "2023-08-15", source="earth-search", **window
).compute()

for name, catalog_ds in (("planetary-computer", pc_ds), ("earth-search", es_ds)):
    median = float(np.nanmedian(catalog_ds["green"].values))
    print(f"{name:20} median green reflectance {median:.4f}")

# %%
# NDSI from each catalog, and the difference in green reflectance between
# them. The medians agree to a thousandth and the snow maps are the same map,
# so the offset handling is right; what remains is a pixel-level scatter of a
# few hundredths that follows the terrain's texture. It is not a shift (the
# 10 m grids coincide exactly) and not the offset (which would be a uniform
# 0.1): the two archives hold different processings of the same acquisition.
# The acquisition timestamps differ by a few minutes (tile time versus
# datatake time), so the time coordinate is dropped before subtracting.
ndsi_pc_da = (
    (pc_ds["green"] - pc_ds["swir16"]) / (pc_ds["green"] + pc_ds["swir16"])
).isel(time=0)
ndsi_es_da = (
    (es_ds["green"] - es_ds["swir16"]) / (es_ds["green"] + es_ds["swir16"])
).isel(time=0)
diff_da = pc_ds["green"].isel(time=0, drop=True) - es_ds["green"].isel(
    time=0, drop=True
)
diff_da.attrs = {"long_name": "green reflectance, Planetary Computer − Earth Search"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
esd.plotting.map(
    ndsi_pc_da, ax=axes[0], cmap="Blues", vmin=-0.5, vmax=1.0,
    title="NDSI, Planetary Computer", cbar_label="NDSI",
)  # fmt: skip
esd.plotting.map(
    ndsi_es_da, ax=axes[1], cmap="Blues", vmin=-0.5, vmax=1.0,
    title="NDSI, Earth Search", cbar_label="NDSI",
)  # fmt: skip
esd.plotting.map(
    diff_da, ax=axes[2], cmap="RdBu_r", vmin=-0.05, vmax=0.05,
    title="Green reflectance difference", cbar_label="PC − Earth Search",
)  # fmt: skip
for ax in axes[1:]:
    ax.set_ylabel("")
fig.tight_layout()

ok = np.isfinite(diff_da.values)
correlation = np.corrcoef(pc_ds["green"].values[0][ok], es_ds["green"].values[0][ok])[
    0, 1
]
print(
    f"green reflectance: correlation {correlation:.3f}, "
    f"median |difference| {float(np.nanmedian(np.abs(diff_da.values))):.4f}"
)
