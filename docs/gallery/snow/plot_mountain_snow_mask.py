"""
Where mountain snow is seasonal
===============================

Wrzesien et al. separate mountains with seasonal snow from mountains with
ephemeral snow, which is the distinction that decides whether a basin stores
water as snow through the winter.

The Zenodo archive is cached on the first call, so re-running this is fast.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

mountain_snow = esd.snow.mountain_snow_mask.load(aoi, layer="mountain_snow")

ax = esd.plotting.categorical(mountain_snow, figsize=(8, 5))
ax.set_title("Global seasonal mountain snow mask (Wrzesien et al. 2019)")
plt.tight_layout()
