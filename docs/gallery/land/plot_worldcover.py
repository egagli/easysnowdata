"""
ESA WorldCover around Mount Rainier
===================================

Ten-metre land cover with the classes drawn straight from the CF flag
attributes the loader attaches. The same map is on the Planetary Computer and,
unsigned, in the public AWS bucket (``source="aws-open-data"``).
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

landcover = esd.land.landcover.load(aoi, version="v200")

ax = esd.plotting.categorical(landcover, figsize=(8, 5))
ax.set_title("ESA WorldCover v200 (2021)")
plt.tight_layout()
