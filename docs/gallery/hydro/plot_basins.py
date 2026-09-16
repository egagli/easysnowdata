"""
Watersheds around Mount Rainier
===============================

HUC12 subwatersheds from the USGS Watershed Boundary Dataset, drawn over the
HUC8 subbasins they nest inside. Both come from the public REST service, so
this example needs no credentials at all.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

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
