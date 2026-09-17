"""
SNOTEL snow water equivalent at Mount Rainier
=============================================

The AWDB network — SNOTEL and SNOLITE telemetry sites, SCAN, and the manual
snow courses — through ``easysnowdata.stations``. No credentials.

``inventory()`` answers "which stations are here" as a GeoDataFrame, and
``load()`` turns the ones you pick into a ``(station, time)`` Dataset with
water-year coordinates already attached.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-122.1, 46.6, -121.3, 47.1)  # Mount Rainier, WA

# %%
# Every AWDB station in the box whose daily record has been probe-verified.
inv = esd.stations.inventory(aoi, networks="awdb", daily_only=True)
print(inv[["name", "network_code", "elevation_m", "state"]].to_string())

# %%
# One water year of SWE and snow depth for all of them. Values are
# centimetres, the unit the networks themselves report.
obs = esd.stations.load(inv, variables=["swe", "snwd"], time="2023-10/2024-09")
print(obs)

# %%
# ``dowy`` is a plain integer coordinate, so it can be swapped in as the axis
# to compare stations on a common water-year calendar.
fig, ax = plt.subplots(figsize=(9, 4.5))
for station in obs["station"].values:
    series = obs["swe"].sel(station=station)
    ax.plot(
        obs["dowy"],
        series,
        label=f"{str(obs['name'].sel(station=station).values)} "
        f"({float(obs['elevation_m'].sel(station=station)):.0f} m)",
    )
ax.set_xlabel("day of water year (1 = 1 October)")
ax.set_ylabel(f"SWE ({obs['swe'].attrs['units']})")
ax.set_title("SNOTEL SWE around Mount Rainier, water year 2024")
ax.legend(fontsize=8)
fig.tight_layout()
