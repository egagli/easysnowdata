# esd-requires: nve
"""
All five station networks in one call
=====================================

``easysnowdata.stations`` reads five public snow-station networks: the USDA
NRCS AWDB (SNOTEL and snow courses), California's CDEC, British Columbia's
DataBC, Norway's NVE HydAPI and the Yukon's AquaCache. Each has its own
catalog entry and client, but ``inventory()`` and ``load()`` take station
codes from any of them and fan the request out to the right client.

``inventory()`` defaults to ``source="archive"``, the published inventory
from ``global_snow_networks``: one request for all five networks, no
credentials. ``source="clients"`` asks the five APIs instead. ``load()``
always goes to the live clients, so a list that includes a Norwegian station
needs the NVE key in ``NVE_API_KEY``; the other four need no account.

The figures show every daily station in the world coloured by network, and
one water year of SWE at one station from each network on a common time axis.
"""

import easysnowdata as esd

# %%
# The five products and the routes behind them. Only NVE needs a credential.
for product_id in sorted(esd.stations.PRODUCT_IDS.values()):
    for src in esd.catalog.get(product_id).sources:
        print(
            f"{product_id:16} {src.id:8} {src.title:40} "
            f"{', '.join(src.requires) or 'no account'}"
        )

# %%
# Every station with a probe-verified daily record, across all five networks,
# in one request. AWDB is two thirds of the total.
inv = esd.stations.inventory(daily_only=True)
print(inv["network"].value_counts().to_string())

# %%
# The map spans from Alaska to Svalbard, and to the four stations NVE serves
# in Nepal. A degree of longitude is short at 70°N, so the axes are drawn with
# a latitude-corrected aspect, and the scale bar is right at the central latitude.
ax = esd.plotting.points(
    inv,
    column="network",
    markersize=6,
    figsize=(12, 5),
    title=f"{len(inv)} daily snow stations in five networks",
)

# %%
# One station from each network, by code alone. AWDB triplets, NVE's dotted
# numbers and Yukon's hyphenated codes give their network away by shape; a
# CDEC or DataBC code is looked up in the inventory. The five requests go to
# five clients and come back on one ``(station, time)`` grid.
codes = ["679_WA_SNTL", "CSL", "1A01P", "12.142.0", "09AA-M1"]
obs = esd.stations.load(codes, variables="swe", time="2023-10/2024-09")
print(obs["network"].to_series().to_string())

# %%
# One water year of SWE, in centimetres everywhere: the clients convert AWDB's
# inches and DataBC's millimetres before the values are merged.
ax = esd.plotting.timeseries(
    obs["swe"], title="Water year 2024, one station per network"
)

# %%
# With several networks in one Dataset, ``attrs["source"]`` joins the source
# ids, and the per-station truth is the ``network`` coordinate.
print(obs.attrs["source"])
print(obs.attrs["source_title"])
print(obs["swe"].attrs)

# %%
# For every daily station at once, :func:`easysnowdata.stations.archive.load`
# reads the pre-downloaded bundle instead of calling five APIs.
