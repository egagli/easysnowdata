"""
Watersheds around Mount Rainier, and the basin they drain to
============================================================

Three nested views of the same drainage, all credential-free: HUC12
subwatersheds from the USGS Watershed Boundary Dataset, the HUC8 subbasins
they sit inside, and the GRDC major river basin that contains the lot.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

# %%
# The WBD REST service answers directly, so no Earth Engine account is needed
# for HUC geometries. The loader pages the service in small blocks — a single
# unpaged HUC12 query over a 2° box times out and comes back as HTML.
subbasins = esd.hydro.basins.huc(aoi, level=8)
subwatersheds = esd.hydro.basins.huc(aoi, level=12)

fig, ax = plt.subplots(figsize=(7, 6))
subbasins.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
subwatersheds.plot(ax=ax, column="name", cmap="tab20", alpha=0.6, edgecolor="white")
for _, row in subbasins.iterrows():
    ax.annotate(row["name"], row.geometry.centroid.coords[0], ha="center", fontsize=9)
ax.set_xlim(aoi[0], aoi[2])
ax.set_ylim(aoi[1], aoi[3])
ax.set_title("USGS HUC8 subbasins and HUC12 subwatersheds")
fig.tight_layout()

# %%
# Zooming out: the GRDC major river basins are 405 whole-basin polygons for
# the world, so the one that intersects Rainier comes back uncut. This is the
# layer to reach for when a station or a pixel needs a basin name.
basin = esd.hydro.basins.grdc_major(aoi)
# The shipped area columns (ACRES, LAEA_HA) are zero or nonsense in this
# release of the layer, so take the area from the geometry instead.
area = basin.to_crs("ESRI:102008").area / 1e6
print(f"{basin.iloc[0]['NAME']} basin: {area.iloc[0]:,.0f} km2")

fig, ax = plt.subplots(figsize=(7, 6))
basin.plot(ax=ax, facecolor="#cfe3f7", edgecolor="#1f4e79", linewidth=1.2)
subbasins.plot(ax=ax, facecolor="none", edgecolor="firebrick", linewidth=1.5)
ax.set_title(f"{basin.iloc[0]['NAME']} basin, with the HUC8s above in red")
ax.set_aspect("equal")
fig.tight_layout()
