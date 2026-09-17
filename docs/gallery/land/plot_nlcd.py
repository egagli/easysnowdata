# esd-requires: earthengine
"""
Annual NLCD land cover for a Cascades basin
===========================================

Annual NLCD carries one map per year from 1985 to 2024, so land cover can be
followed through time instead of pinned to the frozen 2021 release. Here the
newest year is drawn from the class table the Earth Engine asset ships.

Needs Earth Engine credentials (``esd.auth.login("earthengine")``).
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

landcover = esd.land.nlcd.load(aoi)

ax = esd.plotting.categorical(landcover, figsize=(8, 5))
ax.set_title(f"Annual NLCD land cover ({str(landcover.time.values)[:4]})")
plt.tight_layout()
