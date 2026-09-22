"""
British Columbia snow stations (DataBC)
=======================================

British Columbia runs an automated snow weather station (ASWS) network that
reports SWE, snow depth, precipitation, temperature, wind, humidity and
barometric pressure hourly, and a manual snow survey (MSS) network of courses
measured around the first of the month. Both are in the inventory; only the
automated sites carry a daily record.

There is one route, ``source="databc"``: station geometry from the BC Data
Catalogue WFS and observations from the province's snow-data CSV service. No
account is needed. Station codes are BC location codes such as ``1A01P``,
where the trailing ``P`` marks an automated pillow site.

The figures show the stations of a piece of the southern interior coloured by
kind, and one winter of SWE and snow depth at the automated sites there.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-120.5, 49.5, -119.0, 51.0)

box = esd.parse_aoi(aoi)
print(box, box.utm_crs)  # the map below is drawn in the AOI's UTM zone

# %%
# Where the product comes from.
for src in esd.catalog.get("databc-stations").sources:
    print(f"{src.id:20} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# Everything in the box, with the automated sites marked.
everything = esd.stations.inventory(aoi, networks="databc")
everything["kind"] = everything["daily_or_better"].map(
    {True: "automated (ASWS)", False: "snow course (MSS)"}
)
print(everything["kind"].value_counts().to_string())
print(everything[["name", "kind", "elevation_m", "earliest_record_date"]].to_string())

# %%
# The courses outnumber the automated sites almost four to one here, and
# several pairs share a name: the automated site was put beside the course it
# now supplements (``2F10`` and ``2F10P`` are both Silver Star Mountain).
ax = esd.plotting.points(
    everything.to_crs(box.utm_crs),
    column="kind",
    legend_label="BC site kind",
    title="BC snow stations, southern interior",
)

# %%
# SWE and snow depth through one water year at the automated sites. The daily
# snow-depth value is the 16:00 UTC reading, the morning observation in BC.
daily = everything[everything["daily_or_better"].fillna(False).astype(bool)]
obs = esd.stations.load(daily, variables=["swe", "snwd"], time="2023-10/2024-09")
print(obs["swe"].attrs)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
esd.plotting.timeseries(obs["swe"], ax=axes[0], title="Water year 2024")
esd.plotting.timeseries(obs["snwd"], ax=axes[1], title="", legend=False)
fig.tight_layout()

# %%
# Two stations are in the legend and draw nothing. Ottomite Mountain is
# inactive, and McCulloch's record begins in October 2024, after this water
# year ends. The inventory's ``is_active`` and ``earliest_record_date`` say so
# before anything is loaded; the count of observations says so after.
print(obs["swe"].count(dim="time").to_series().to_string())

# %%
# The snow variables are only part of what the automated sites report. For
# wind, humidity or pressure, an hourly interval, or the quality flags, ask
# the client directly:
#
# .. code-block:: python
#
#     from easysnowdata.stations.clients import DataBCClient
#
#     records = DataBCClient().get_data(
#         station_ids=["1A01P"], variables=["temp", "wind_spd"],
#         begin_date="2024-03-01", end_date="2024-03-07", interval="hourly",
#     )
