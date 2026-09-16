"""
British Columbia automated snow weather stations
================================================

BC's ASWS network reports hourly and daily, and the province also runs manual
snow-survey courses. The client serves both, plus the wind, humidity and
barometric variables the ASWS sites carry — more than the snow archive keeps.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-120.5, 49.5, -119.0, 51.0)  # BC interior

# %%
# The automated sites in the box.
inv = esd.stations.inventory(aoi, networks="databc", daily_only=True)
print(inv[["name", "elevation_m", "status"]].head(10).to_string())

# %%
# SWE and snow depth through one winter.
obs = esd.stations.load(inv.head(4), variables=["swe", "snwd"], time="2023-10/2024-06")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for station in obs["station"].values:
    name = str(obs["name"].sel(station=station).values)
    axes[0].plot(obs["time"], obs["swe"].sel(station=station), label=name)
    axes[1].plot(obs["time"], obs["snwd"].sel(station=station), label=name)
axes[0].set_ylabel(f"SWE ({obs['swe'].attrs['units']})")
axes[1].set_ylabel(f"snow depth ({obs['snwd'].attrs['units']})")
axes[0].set_title("BC automated snow weather stations, winter 2023-24")
axes[0].legend(fontsize=8)
fig.tight_layout()

# %%
# The snow variables are only part of what these sites report. Ask the client
# directly for anything the adapter does not surface — flags, a native
# variable name, or an interval other than daily:
#
# .. code-block:: python
#
#     from easysnowdata.stations.clients import DataBCClient
#
#     records = DataBCClient().get_data(
#         station_ids=["1A01P"], variables=["temp", "wind_spd"],
#         begin_date="2024-03-01", end_date="2024-03-07", interval="hourly",
#     )
