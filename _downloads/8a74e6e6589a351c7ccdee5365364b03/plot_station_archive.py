"""
Every daily snow station at once
================================

The archive route reads what ``global_snow_networks`` pre-downloads: a
normalized inventory of every station across the five networks, and one daily
SWE / snow-depth CSV per station whose daily record has been probe-verified.
One ~28 MB download instead of five API sweeps, and it needs no credentials —
not even NVE's, because the Norwegian stations are already in the bundle.

Roughly 1 550 stations by 47 000 days if you ask for all of it, so pass
``time=``, ``aoi=`` or ``stations=`` unless you really want every cell.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

# %%
# The inventory is one HTTP request and carries the probe-verified
# ``daily_or_better`` verdict, rather than what each network advertises.
inv = esd.stations.archive.inventory(daily_only=True)
print(inv["network"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(10, 5))
inv.plot(ax=ax, column="network", legend=True, markersize=4)
ax.set_title(f"{len(inv)} probe-verified daily snow stations")
fig.tight_layout()

# %%
# One water year everywhere. The series are cut to the window before the
# dense grid is built, so this is far cheaper than loading the whole archive.
obs = esd.stations.archive.load(time="2023-10/2024-09")
print(obs)

# %%
# Peak SWE per station, against elevation, coloured by network.
peak = obs["swe"].max(dim="time")
fig, ax = plt.subplots(figsize=(8, 5))
for network in np.unique(obs["network"].values):
    rows = obs["network"] == network
    ax.scatter(
        obs["elevation_m"][rows], peak[rows], s=10, alpha=0.6, label=str(network)
    )
ax.set_xlabel("elevation (m)")
ax.set_ylabel(f"peak SWE, WY2024 ({obs['swe'].attrs['units']})")
ax.set_title("Peak snow water equivalent against elevation")
ax.legend()
fig.tight_layout()

# %%
# For anything the archive does not hold — precipitation, temperature, wind,
# quality flags, sub-daily data, or today's observation — go to the networks
# themselves with :func:`easysnowdata.stations.load`.
