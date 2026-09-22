"""
Yukon snow stations (AquaCache)
===============================

The Yukon Snow Survey Network mixes automated snow weather stations that
report SWE and snow depth daily with manual snow courses surveyed monthly
through late winter and spring, and the territory's water-data database (AquaCache) also
mirrors Environment and Climate Change Canada meteorological stations. All of
them come through the inventory; ``daily_only`` keeps the automated sites.

There is one route, ``source="yukon"``, the open Yukon Water Data REST API.
No account is needed. Station codes are AquaCache location codes such as
``09AA-M1``; the ``M`` sites are the automated meteorological stations and
the ``SC`` sites are courses.

The figures show every station the network lists, coloured by kind, and one
water year of SWE and snow depth at two automated sites.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

print(f"easysnowdata {esd.__version__}")

# %%
# Where the product comes from.
for src in esd.catalog.get("yukon-stations").sources:
    print(f"{src.id:20} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# No area of interest: the whole network is small enough to list at once.
everything = esd.stations.inventory(networks="yukon")
everything["kind"] = everything["daily_or_better"].map(
    {True: "automated (daily)", False: "snow course"}
)
print(everything["kind"].value_counts().to_string())

# %%
# A few of the courses the survey operates sit across the border in northern
# British Columbia and Alaska, so ``state`` is not always ``YT``.
print(everything["state"].value_counts().to_string())
ax = esd.plotting.points(
    everything,
    column="kind",
    legend_label="Yukon site kind",
    title="Yukon Snow Survey Network",
)

# %%
# One water year of SWE and snow depth at two automated sites.
obs = esd.stations.load(
    ["09AA-M1", "09BA-M7"], variables=["swe", "snwd"], time="2023-10/2024-09"
)
print(obs)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
esd.plotting.timeseries(obs["swe"], ax=axes[0], title="Water year 2024")
esd.plotting.timeseries(obs["snwd"], ax=axes[1], title="", legend=False)
fig.tight_layout()

# %%
# A subarctic snowpack holds little water: peak SWE here is around 15 cm,
# an order of magnitude below a Cascades SNOTEL site. Snow depth flattens out
# in March while SWE keeps rising for another month, and the whole pack goes
# in about two weeks at the end of April.
peak_snwd = obs["snwd"].idxmax(dim="time").dt.strftime("%Y-%m-%d")
peak_swe = obs["swe"].idxmax(dim="time").dt.strftime("%Y-%m-%d")
for station in obs["station"].values:
    print(
        f"{str(obs['name'].sel(station=station).values):34}"
        f" deepest {str(peak_snwd.sel(station=station).values)}"
        f"  peak SWE {str(peak_swe.sel(station=station).values)}"
    )
