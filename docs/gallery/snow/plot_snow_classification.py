"""
Seasonal snow classes around Mount Rainier
==========================================

Sturm & Liston's classification says what kind of snowpack a place gets:
maritime on the Cascade crest, montane forest on the flanks, ephemeral in the
lowlands. The classes are drawn from the CF flag attributes the loader
attaches.

The default source is NSIDC-0768 and needs an Earthdata Login;
``source="hosted-cog"`` reads the same 300 m map with no credentials.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

snow_class = esd.snow.snow_classification.load(aoi, source="hosted-cog")

ax = esd.plotting.categorical(snow_class, figsize=(8, 5))
ax.set_title("Seasonal snow classification (Sturm & Liston 2021)")
plt.tight_layout()
