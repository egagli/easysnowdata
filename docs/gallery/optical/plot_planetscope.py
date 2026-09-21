# esd-requires: planet
"""
PlanetScope surface reflectance and UDM2 (Planet)
=================================================

PlanetScope is Planet Labs' constellation of small satellites: 3 m pixels,
four or eight bands, and a near-daily revisit, which is what catches a melt
event between two Sentinel-2 overpasses. It is the one commercial source in
the package. Every scene ships with the UDM2 usable-data mask, whose layers
(clear, snow, shadow, haze, cloud) are Planet's own classification.

Both routes need a Planet account with data access. ``source="orders-api"``
(the default) searches with the Data API and then **orders** the scenes,
clipped to the AOI and optionally harmonised to Sentinel-2, so Planet charges
for the clipped area. ``source="data-api"`` activates and streams one whole
scene and charges its full area. Searching is free on either route; ordering
spends quota, so the ordering cell below is left for you to run by hand, and
``load`` never orders on its own.

PlanetScope has no shortwave-infrared band, so there is no NDSI; the UDM2
snow layer is the snow product. The figures that need a delivery are written
out but not executed; the free comparison at the end draws the same box from
Sentinel-2.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)  # the Nisqually glacier side of Rainier
when = "2023-08-14/2023-08-16"

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("planetscope").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# Search: free, and it only returns scenes the account may download. Asking
# for the surface-reflectance and UDM2 assets up front keeps out scenes that
# could not be ordered as ``analytic_sr`` anyway.
scenes = esd.optical.planetscope.search(
    aoi, when, cloud_cover=20, asset_types=["ortho_analytic_4b_sr", "ortho_udm2"]
)
print(len(scenes), "scenes")
print(
    scenes[["acquired", "cloud_percent", "clear_percent", "snow_ice_percent"]]
    .head(10)
    .to_string()
)

# %%
# .. warning::
#
#    **This cell spends Planet quota.** It is the only step that does, which
#    is why it is a separate, explicit call and why ``load`` refuses to order
#    for you. The ``clip`` tool makes Planet deliver COGs cut to the AOI, so
#    the charge is for the box, not for the whole strip; ``harmonize``
#    puts the radiometry on Sentinel-2's scale.
#
# .. code-block:: python
#
#     order = esd.optical.planetscope.order(
#         aoi,
#         items=scenes.head(1),
#         bundle="analytic_sr",
#         harmonize="Sentinel-2",
#     )
#     ps = esd.optical.planetscope.load(aoi, order=order)

# %%
# With a delivery in hand the bands are named, scaled to reflectance, and the
# UDM2 raster is decoded into its layers as 0/1 masks with CF flag attributes,
# so ``ps["snow"]`` plots as a categorical map directly. The composite uses
# the same fixed 0–0.5 stretch as the Sentinel-2 example, so the two can be
# compared side by side.
#
# .. code-block:: python
#
#     scene = ps.isel(time=0)
#     rgb = scene[["red", "green", "blue"]].to_array("band").clip(0, 0.5) / 0.5
#     print(f"snow fraction: {float(esd.optical.planetscope.snow_fraction(ps)[0]):.2f}")
#
#     fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
#     rgb.plot.imshow(rgb="band", ax=axes[0], add_labels=False)
#     axes[0].set_title("PlanetScope true colour (0–0.5 reflectance), 3 m")
#     esd.plotting.finish_map(axes[0], ps.rio.crs)
#     esd.plotting.categorical(scene["snow"], ax=axes[1], title="UDM2 snow")
#     esd.plotting.categorical(scene["clear"], ax=axes[2], title="UDM2 clear")
#     fig.tight_layout()

# %%
# The free comparison: the same box from Sentinel-2, which needs no account
# and no quota, as a true-colour composite and as NDSI. Run the Planet cells
# above to put the 3 m and 10 m views side by side.
s2 = esd.optical.sentinel2.load(
    aoi, when, bands=["blue", "green", "red", "swir16", "scl"], mask="scl-default"
).compute()
scene = s2.isel(time=0)

rgb = scene[["red", "green", "blue"]].to_array("band").clip(0, 0.5) / 0.5
ndsi = (scene["green"] - scene["swir16"]) / (scene["green"] + scene["swir16"])
ndsi.attrs = {"long_name": "NDSI (green − SWIR 1.6 µm)"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
rgb.plot.imshow(rgb="band", ax=axes[0], add_labels=False)
axes[0].set_title(f"Sentinel-2 true colour, {str(scene['time'].values)[:10]}")
esd.plotting.finish_map(axes[0], s2.rio.crs)
esd.plotting.map(
    ndsi, ax=axes[1], cmap="Blues", vmin=-0.5, vmax=1.0,
    title="Sentinel-2 NDSI, same box", cbar_label="NDSI",
)  # fmt: skip
fig.tight_layout()
