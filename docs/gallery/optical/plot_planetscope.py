# esd-requires: planet
"""
PlanetScope beside Sentinel-2, and the UDM2 snow band
=====================================================

PlanetScope is 3 m and near-daily, which is what makes it useful for catching
a melt event between Sentinel-2 overpasses. It is also commercial: an order
spends the account's quota, so the ordering cell below is **manual** and does
not run when the gallery is built.

Searching is free, so that part runs anywhere Planet credentials exist.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)
when = "2023-07-01/2023-07-03"

# %%
# Search: free, and it only returns scenes the account may download.
scenes = esd.optical.planetscope.search(
    aoi, when, cloud_cover=20, asset_types=["ortho_analytic_4b_sr", "ortho_udm2"]
)
print(scenes[["acquired", "cloud_percent", "clear_percent", "snow_ice_percent"]])

# %%
# .. warning::
#
#    **This cell spends Planet quota.** It is the only step that does, and it
#    is why ``load`` never orders by itself. Uncomment it to run it by hand.
#    The ``clip`` tool means Planet charges for the AOI, not for the whole
#    600 MB strip; ``harmonize`` puts the radiometry on Sentinel-2's scale.
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
# With a delivery in hand, the bands are named, the UDM2 mask is decoded into
# its layers, and the snow band is directly usable:
#
# .. code-block:: python
#
#     ndsi_like = esd.processing.normalized_difference(ps["green"], ps["nir"])
#     snow = ps["snow"]                       # UDM2 snow mask, 0/1
#     print(float(esd.optical.planetscope.snow_fraction(ps).isel(time=0)))
#
#     fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
#     esd.plotting.rgb(
#         esd.processing.stretch_percentile(
#             esd.processing.rgb(ps.isel(time=0), ("red", "green", "blue"))
#         ),
#         ax=axes[0],
#         title="PlanetScope 3 m",
#     )
#     ndsi_like.isel(time=0).plot.imshow(ax=axes[1], cmap="Blues")
#     esd.plotting.categorical(snow.isel(time=0), ax=axes[2], title="UDM2 snow")

# %%
# The free comparison: the same box from Sentinel-2, which needs no account
# and no quota. Run the Planet cells above to put the 3 m and 10 m views side
# by side.
s2 = esd.optical.sentinel2.load(
    aoi, when, bands=["green", "swir16", "red", "blue", "scl"], mask="scl-default"
)
ndsi = esd.processing.ndsi(s2)
if ndsi.sizes["time"]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ndsi.isel(time=0).plot.imshow(ax=ax, cmap="Blues", vmin=-0.5, vmax=1.0)
    ax.set_title("Sentinel-2 NDSI, same box")
    ax.set_aspect("equal")
    fig.tight_layout()
