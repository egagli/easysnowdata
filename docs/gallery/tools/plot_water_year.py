"""
Water years and day of water year
=================================

A snow season straddles the calendar year, so snow hydrology counts in
*water years*: in the northern hemisphere WY2024 runs from 1 October 2023 to
30 September 2024 and is named for the year it ends in; in the southern
hemisphere the year starts 1 April and is named for the year it starts.
``esd.processing.wateryear`` holds the vectorised helpers for both conventions,
and every station loader attaches ``water_year`` and ``dowy`` (day of water
year) coordinates so a series can be grouped, overlaid or aligned by season
without any date arithmetic in user code.

The example takes one SNOTEL station from the credential-free station archive
and shows the same six seasons three ways: on calendar dates, overlaid on one
October-to-September axis with ``esd.plotting.timeseries(...,
by_water_year=True)``, and as one peak-SWE bar per water year.
"""

import matplotlib.pyplot as plt
import pandas as pd

import easysnowdata as esd
from easysnowdata.processing import wateryear

# %%
# The scalar helpers. ``water_year`` names the year, ``day_of_water_year``
# counts from 1 on the first day of the season, ``water_year_start`` gives
# that day, and ``water_year_range`` is the season as a daily DatetimeIndex.
# WY2024 has 366 days because 29 February 2024 falls inside it.
print("water_year('2024-03-15')        ", wateryear.water_year("2024-03-15"))
print("day_of_water_year('2024-03-15') ", wateryear.day_of_water_year("2024-03-15"))
print("day_of_water_year('2023-10-01') ", wateryear.day_of_water_year("2023-10-01"))
print("water_year_start('2024-03-15')  ", wateryear.water_year_start("2024-03-15"))
wy2024 = wateryear.water_year_range(2024)
print(
    f"water_year_range(2024): {wy2024[0].date()} to {wy2024[-1].date()}, {len(wy2024)} days"
)

# %%
# The southern-hemisphere convention: the season starts 1 April, and the year
# is named for the calendar year it *starts* in, so 15 March 2024 belongs to
# WY2023 and is its 350th day.
print(
    "southern water_year('2024-03-15')       ",
    wateryear.water_year("2024-03-15", hemisphere="southern"),
)
print(
    "southern day_of_water_year('2024-03-15')",
    wateryear.day_of_water_year("2024-03-15", hemisphere="southern"),
)
south = wateryear.water_year_range(2024, hemisphere="southern")
print(f"southern water_year_range(2024): {south[0].date()} to {south[-1].date()}")

# %%
# The same functions accept a DatetimeIndex, a Series or a DataArray and
# return the same container type, so they vectorise over a whole record.
days = pd.date_range("2023-09-29", "2023-10-02", freq="D")
print(
    pd.DataFrame(
        {
            "water_year": wateryear.water_year(days),
            "dowy": wateryear.day_of_water_year(days),
        },
        index=days,
    )
)

# %%
# A real series: six seasons of daily SWE at Paradise (SNOTEL 679, Mount
# Rainier) from the station archive. The loader has already called
# ``add_water_year_coords``, so ``water_year`` and ``dowy`` ride along the
# ``time`` dimension; calling it again is harmless and shows what it adds.
obs = esd.stations.archive.load(["679_WA_SNTL"], time="2018-10/2024-09")
obs = wateryear.add_water_year_coords(obs)
swe = obs["swe"].sel(station="679_WA_SNTL")
print(swe.coords["water_year"].values[:3], "...", swe.coords["dowy"].values[:3], "...")

# %%
# On calendar dates first: the seasons are easy to count, hard to compare.
ax = esd.plotting.timeseries(swe, title=f"{swe['name'].item()} SNOTEL, daily SWE")
ax.figure.tight_layout()

# %%
# Overlaid by water year: each season becomes one line on a shared
# October-to-September axis, labelled WY2019 to WY2024, so the timing of
# accumulation and melt-out lines up across years.
ax = esd.plotting.timeseries(
    swe, by_water_year=True, title=f"{swe['name'].item()} SNOTEL, SWE by water year"
)
ax.figure.tight_layout()

# %%
# Because ``water_year`` is a plain integer coordinate, ``groupby`` works
# directly. Peak SWE and the day it was reached, per season.
peak = swe.groupby("water_year").max()
peak_dowy = swe.groupby("water_year").map(lambda s: s["dowy"][int(s.argmax("time"))])
peak_date = swe.groupby("water_year").map(lambda s: s["time"][int(s.argmax("time"))])
summary = pd.DataFrame(
    {
        "peak SWE [cm]": peak.values,
        "day of water year": peak_dowy.values,
        "date": pd.DatetimeIndex(peak_date.values).date,
    },
    index=[f"WY{int(y)}" for y in peak["water_year"].values],
)
print(summary.to_string())

# %%
# For the yearly maximum alone there is a pandas idiom that needs no helper at
# all: an anchored yearly offset. ``"YS-OCT"`` is "year start, October", so
# ``resample(time="YS-OCT")`` bins from each 1 October to the next 30
# September; its labels are the *start* dates, which is why the coordinate
# version reads more naturally as WY2019, WY2020 ...
by_offset = swe.resample(time="YS-OCT").max()
print(by_offset.to_series().rename("peak SWE [cm]").to_string())
assert (by_offset.values == peak.values).all()

# %%
# The peaks as bars. A year like WY2022 that peaked late shows up in the
# table above, not here: bar charts hide timing, which is what the overlay
# was for.
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(summary.index, summary["peak SWE [cm]"], color="#3b6ea8", width=0.7)
for label, row in summary.iterrows():
    ax.annotate(
        row["date"].strftime("%-d %b"),
        (label, row["peak SWE [cm]"]),
        ha="center",
        va="bottom",
        fontsize=8,
        xytext=(0, 2),
        textcoords="offset points",
    )
ax.set_ylabel("peak SWE [cm]")
ax.set_ylim(0, summary["peak SWE [cm]"].max() * 1.12)
ax.set_title(f"{swe['name'].item()} SNOTEL, peak SWE per water year")
ax.grid(True, axis="y", color="0.85", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
