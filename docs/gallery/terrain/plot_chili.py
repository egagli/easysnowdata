"""
CHILI heat-load index on Mount Rainier
======================================

CHILI combines slope, aspect and latitude into one index of the heat load a
pixel receives. Cool, north-facing slopes hold snow later in the season than
the warm slopes across the valley.

Needs Earth Engine credentials (``esd.auth.login("earthengine")``).
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

# normalize="index" undoes the asset's 8-bit scaling: 0 is coolest, 1 warmest.
chili = esd.terrain.chili.load(aoi, normalize="index")

fig, ax = plt.subplots(figsize=(7, 5))
chili.plot.imshow(
    ax=ax, cmap="RdYlBu_r", vmin=0, vmax=1, cbar_kwargs={"label": "CHILI [0-1]"}
)
ax.set_title("Continuous Heat-Insolation Load Index")
ax.set_aspect("equal")
fig.tight_layout()
