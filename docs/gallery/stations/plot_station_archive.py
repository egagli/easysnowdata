"""
Daily snow-station archive (global_snow_networks)
=================================================

``global_snow_networks`` pre-downloads daily SWE and snow depth for every
station across the five networks whose daily record its probe has verified,
and publishes two things: a normalized inventory of every station, periodic
snow courses included, and one CSV per daily station bundled into a single
archive of about 28 MB. Both are rebuilt daily.

Two routes: ``source="github-tarball"`` (default) downloads the bundle once
into the package cache and reads every wanted CSV out of it, which is the way
to get all of it; ``source="github-csv"`` fetches one CSV per station, cheaper
for a handful. Neither needs an account, not even for the Norwegian stations,
which are already in the bundle. The archive holds SWE and snow depth only.

The figures show every daily station coloured by the length of its record,
one water year of peak SWE against elevation for all of them, and one
station's complete daily record.
"""

import matplotlib.pyplot as plt
import pandas as pd

import easysnowdata as esd

print(f"easysnowdata {esd.__version__}")

# %%
# Where the product comes from.
for src in esd.catalog.get("snow-station-archive").sources:
    print(f"{src.id:20} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The inventory is one HTTP request, and ``daily_or_better`` is the probe's
# verdict: a station counts as daily only once daily values have actually been
# retrieved from it. Most records start around 1980; the oldest are the NWS
# Cooperative Observer sites AWDB mirrors, with snow depth from the 1890s.
inv = esd.stations.archive.inventory(daily_only=True)
print(inv["network"].value_counts().to_string())
span = inv["latest_record_date"] - inv["earliest_record_date"]
inv["record_length"] = pd.cut(
    span.dt.days / 365.25,
    [0, 15, 30, 45, 200],
    labels=["under 15 years", "15 to 30 years", "30 to 45 years", "over 45 years"],
)
inv["record_length"] = (
    inv["record_length"].cat.add_categories("no dates").fillna("no dates")
)
oldest = inv.nsmallest(3, "earliest_record_date")
print(oldest[["name", "network", "earliest_record_date"]].to_string())

# %%
# Every probe-verified daily station, by length of record. The largest class is
# over 45 years: SNOTEL and the CCSS pillows were built out around 1980 and
# their daily archives begin there. An ordered categorical keeps the legend in
# that order. Two hundred stations, nearly all CDEC, carry the daily flag but
# no record dates yet.
print(inv["record_length"].value_counts().to_string())
ax = esd.plotting.points(
    inv,
    column="record_length",
    legend_label="record length",
    markersize=6,
    figsize=(12, 5),
    title=f"{len(inv)} daily snow stations",
)

# %%
# One water year everywhere. Each station's series is cut to the window before
# the dense grid is built, so this is much cheaper than the whole archive.
obs = esd.stations.archive.load(time="2023-10/2024-09")
print(obs)

# %%
# The archive passes the networks' values through unfiltered. A few stations
# report a peak of ten metres or more of water in a single day, which is a
# sensor fault rather than snow; find them before they set the axis.
peak = obs["swe"].max(dim="time")
suspect = peak.where(peak > 300, drop=True)
print(suspect.to_series().round(0).to_string())

# %%
# Peak SWE against elevation, one point per station, coloured by network, with
# the faulty values clipped off the top. The same peak sits a kilometre lower
# in the maritime Cascades and coast ranges than in the Sierra Nevada or the
# continental interior.
fig, ax = plt.subplots(figsize=(8, 5))
for network in sorted(set(obs["network"].values.tolist())):
    rows = obs["network"] == network
    ax.scatter(
        obs["elevation_m"][rows], peak[rows], s=10, alpha=0.6, label=str(network)
    )
ax.set_xlabel(esd.plotting.label(obs["elevation_m"], name="elevation", units="m"))
ax.set_ylabel(esd.plotting.label(obs["swe"], name="peak SWE, water year 2024"))
ax.set_title("Peak snow water equivalent against elevation")
ax.set_ylim(0, 300)
ax.grid(True, color="0.85", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(title="network", fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()

# %%
# One station's whole record, from the per-station CSV route: a single
# request and no bundle. Paradise on Mount Rainier begins in October 1980,
# which is where most of the archive begins.
paradise = esd.stations.archive.load("679_WA_SNTL", source="github-csv")
ax = esd.plotting.timeseries(
    paradise["swe"].squeeze("station"),
    title="Paradise (1570 m): the whole daily record",
)

# %%
# For anything the archive does not hold (precipitation, temperature, wind,
# quality flags, sub-daily data, or today's observation) go to the networks
# themselves with :func:`easysnowdata.stations.load`.
