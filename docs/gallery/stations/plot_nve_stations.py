# esd-requires: nve
"""
Norwegian snow stations (NVE HydAPI)
====================================

The Norwegian Water Resources and Energy Directorate (NVE) runs snow pillows
that report SWE and snow depth daily and hourly, plus snow-depth stations at
its climate and hydrometric sites, all served through HydAPI. Parameter 2003
is SWE and 2002 is snow depth; ``daily_only`` keeps the stations that carry
either at daily resolution.

There is one route, ``source="nve"``, HydAPI v1, and it is the one network of
the five that needs a credential: a free API key in ``NVE_API_KEY`` (request
one at https://hydapi.nve.no/). The key is read through the ``nve`` auth
provider, so a missing one raises ``CredentialError`` with the setup steps
before any request is made. The station inventory does not need it, because
it comes from the published archive.

The figures show the daily stations across Norway, coloured by which snow
variable they report, and one water year of SWE and snow depth at two pillows.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

# %%
# Where the product comes from, and whether the key is configured.
for src in esd.catalog.get("nve-stations").sources:
    print(f"{src.id:20} {src.title:45} {', '.join(src.requires) or 'no account'}")
print(esd.auth.status().loc["nve"])

# %%
# Every NVE station with a probe-verified daily SWE or snow-depth record. This
# reads the published inventory, so no key is involved yet.
inv = esd.stations.inventory(networks="nve", daily_only=True)
inv["reports"] = inv["has_daily_swe"].map({True: "SWE and depth", False: "depth only"})
print(f"{len(inv)} NVE snow stations with a daily record")
print(inv["reports"].value_counts().to_string())

# %%
# Not all of them are on the mainland. One is on Svalbard, and HydAPI also
# serves the four snow stations NVE helps run in Nepal's Langtang and Mustang
# valleys, so a box around Norway misses them. The map is the mainland.
mainland = (4.5, 57.5, 31.5, 72.0)
in_norway = esd.stations.inventory(mainland, networks="nve", daily_only=True)
in_norway["reports"] = inv["reports"]
elsewhere = inv.loc[~inv.index.isin(in_norway.index)]
print(elsewhere[["name", "latitude", "longitude"]].to_string())
ax = esd.plotting.points(
    in_norway,
    column="reports",
    legend_label="daily snow variables",
    title="NVE stations with a daily snow record",
)

# %%
# One water year of SWE and snow depth at two long-running pillows. This is
# the call that needs the key; the codes are NVE's dotted station numbers.
obs = esd.stations.load(
    ["12.142.0", "121.2.0"], variables=["swe", "snwd"], time="2023-10/2024-09"
)
print(obs)

# %%
# The archive inventory carries no elevation for NVE stations, so the legend
# shows names alone where the other networks' examples show ``name (elev m)``.
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
esd.plotting.timeseries(obs["swe"], ax=axes[0], title="Water year 2024")
esd.plotting.timeseries(obs["snwd"], ax=axes[1], title="", legend=False)
fig.tight_layout()

# %%
# Norway's melt season runs later than the western United States at the same
# elevation, and the snow-depth stations without a pillow are still useful:
# depth alone tracks the timing of accumulation and melt-out, if not the water.
