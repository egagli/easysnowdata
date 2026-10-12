"""
Climate-mode indices: PDO, ENSO, AO (NOAA NCEI / CPC)
=====================================================

Climate-mode indices are monthly series, one number per month, that summarize a
large-scale pattern: the Pacific Decadal Oscillation (PDO), the El Niño-Southern
Oscillation (ENSO), the Arctic and North Atlantic Oscillations (AO, NAO), the
Pacific/North American pattern (PNA) and the Atlantic Multidecadal Oscillation
(AMO). They are the usual first explanation for why one winter's snowpack is
not like the next.

The same name covers several different series, so ``source="noaa"`` (the
default) reads each index from the office that maintains it: the PDO and AMO
from NCEI's ERSSTv5 files, ENSO, AO, NAO and PNA from CPC, and MEI.v2 from
PSL. For ENSO that means **RONI**, the relative Oceanic Niño Index, which CPC
made its official index in February 2026; the legacy ONI is still published.
``source="psl"`` reads PSL's correlation-page copies, for reproducing older
work. No account is needed for either.

The first figure is the PDO, RONI and AO since 1950. The second shows why the
source matters: two "PDO" series and the two ENSO indices, side by side. The
third ties the indices to snow, with April 1 snow water equivalent at the
Paradise SNOTEL station on Mount Rainier against each winter's mean RONI and
PDO.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import easysnowdata as esd

print(f"easysnowdata {esd.__version__}")

# Warm/cool poles for an index's sign, and a gray for neutral winters.
WARM, COOL, NEUTRAL = "#e34948", "#2a78d6", "#7a7a75"
# Two routes or two series compared on one axis: the first two categorical hues.
FIRST, SECOND = "#2a78d6", "#eb6834"

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("climate-indices").sources:
    print(f"{src.id:6} {src.title:40} {', '.join(src.requires) or 'no account'}")

# %%
# What the default route offers and which file each index comes from. Nothing
# is downloaded until ``load``.
print(esd.climate.indices.search()[["provider", "cadence", "url"]].to_string())

# %%
# Every index since 1950, as one monthly Dataset. A 3-month-season index
# (RONI, ONI) is stamped at its centre month, so the value for January is the
# December-February mean. Each variable's ``latest`` attribute says where its
# file ends; CPC's ENSO files run a month behind the others.
idx_ds = esd.climate.indices.load(time="1950/..")
print(idx_ds)
print({name: idx_ds[name].attrs["latest"] for name in idx_ds.data_vars})

# %%
# The PDO, RONI and AO since 1950, shaded warm above zero and cool below. The
# PDO's long phases stand out (negative 1950-1976, positive 1977-1998, and
# mostly negative since), as do the big El Niños of 1982-83, 1997-98, 2015-16
# and 2023-24.
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
for ax, name in zip(axes, ["pdo", "roni", "ao"], strict=True):
    series = idx_ds[name].to_series().dropna()
    esd.plotting.timeseries(
        idx_ds,
        ax=ax,
        variable=name,
        legend=False,
        title=idx_ds[name].attrs["long_name"],
        color="0.35",
        linewidth=0.5,
    )
    ax.fill_between(series.index, series, 0, where=series > 0, color=WARM, lw=0)
    ax.fill_between(series.index, series, 0, where=series < 0, color=COOL, lw=0)
    ax.axhline(0, color="0.2", linewidth=0.6)
    ax.set_xlabel("")
    ax.set_ylabel(f"{name.upper()} [{idx_ds[name].attrs['units']}]")
fig.tight_layout()

# %%
# Two routes to "the PDO". PSL's ``pdo.data`` is a different series from
# NCEI's, built a different way, and it stopped at 2025-08 (``load`` logs a
# warning when a file has gone stale). The two agree on the phases but not on
# the month: they correlate at about 0.9 and can differ by more than one unit.
psl_ds = esd.climate.indices.load(time="1950/..", indices="pdo", source="psl")
both_df = pd.DataFrame(
    {
        "NCEI ERSSTv5": idx_ds["pdo"].to_series(),
        "PSL pdo.data": psl_ds["pdo"].to_series(),
    }
).dropna()
difference = both_df["NCEI ERSSTv5"] - both_df["PSL pdo.data"]
print(f"r = {both_df.corr().iloc[0, 1]:.2f} over {len(both_df)} months")
print(
    f"largest monthly difference {difference.abs().max():.2f} ({difference.abs().idxmax():%Y-%m})"
)
print("PSL file ends", psl_ds["pdo"].attrs["latest"])

# %%
# Two ENSO indices from the same office. The ONI measures Niño 3.4 against a
# fixed climatology, so a warming ocean pushes it up; RONI subtracts the
# tropical-mean anomaly first. They now disagree by about half a degree, and
# they can disagree about the phase itself: a winter that is "El Niño" by one
# can be neutral by the other. Classify winters from the index you name, and
# say which.
recent = slice("2000-01-01", None)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=False)
axes[0].plot(
    both_df.index, both_df["NCEI ERSSTv5"], color=FIRST, lw=1, label="NCEI ERSSTv5"
)
axes[0].plot(
    both_df.index, both_df["PSL pdo.data"], color=SECOND, lw=1, label="PSL pdo.data"
)
axes[0].set_title("PDO: two series with one name")
axes[0].set_ylabel("PDO [1]")
for name, color in (("oni", SECOND), ("roni", FIRST)):
    series = idx_ds[name].sel(time=recent).to_series().dropna()
    axes[1].plot(series.index, series, color=color, lw=1.2, label=name.upper())
axes[1].axhspan(-0.5, 0.5, color="#f0efec", zorder=0, label="neutral band (±0.5 °C)")
axes[1].set_title("ENSO: the legacy ONI and CPC's official RONI")
axes[1].set_ylabel("anomaly [°C]")
for ax in axes:
    ax.axhline(0, color="0.2", linewidth=0.6)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
fig.tight_layout()

oni_phase = esd.climate.indices.enso_phase(idx_ds["oni"])
roni_phase = esd.climate.indices.enso_phase(idx_ds["roni"])
phases_df = pd.DataFrame({"ONI": oni_phase, "RONI": roni_phase}).dropna()
winters_df = phases_df[phases_df.index.month == 1]
disagree_df = winters_df[winters_df["ONI"] != winters_df["RONI"]]
print(
    f"{len(disagree_df)} of {len(winters_df)} DJF seasons since 1950 get a different phase:"
)
print(disagree_df.set_axis(disagree_df.index.year).to_string())

# %%
# Now the snow. ``seasonal_mean`` averages each index over November-March and
# labels the result by water year, so a winter is one row. April 1 SWE comes
# from the Paradise SNOTEL station (679_WA_SNTL) through
# ``esd.stations``; ``enso_phase`` labels each winter by RONI with CPC's rule
# (five overlapping seasons beyond ±0.5 °C), read at DJF.
winter_df = esd.climate.indices.seasonal_mean(idx_ds[["roni", "pdo", "pna", "ao"]])
paradise_ds = esd.stations.load(
    ["679_WA_SNTL"], variables=["swe"], time="1980-10-01/2026-06-30"
)
swe = paradise_ds["swe"].isel(station=0).to_series()
april1 = swe[(swe.index.month == 4) & (swe.index.day == 1)].dropna()
april1.index = april1.index.year
djf_phase = roni_phase[roni_phase.index.month == 1]
djf_phase.index = djf_phase.index.year
snow_df = winter_df.join(april1.rename("swe_cm"), how="inner").join(
    djf_phase.rename("phase")
)
print(f"{len(snow_df)} water years, {snow_df.index.min()}-{snow_df.index.max()}")

# %%
# April 1 SWE against the winter's mean RONI and PDO. Both lean the same way:
# cool winters (La Niña, negative PDO) leave more snow at Paradise. What is not
# obvious is that the PDO's correlation is as strong as ENSO's, and so is the
# PNA's, so "it was an El Niño year" is only part of the story. One station
# and 44 winters is a small sample: the correlations are about -0.4,
# real but loose, and the spread within each phase is wide.
styles = {
    "El Niño": dict(color=WARM, marker="^"),
    "neutral": dict(color=NEUTRAL, marker="o"),
    "La Niña": dict(color=COOL, marker="v"),
}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, name, xlabel in (
    (axes[0], "roni", "Nov-Mar mean RONI [°C]"),
    (axes[1], "pdo", "Nov-Mar mean PDO [1]"),
):
    for phase, style in styles.items():
        rows = snow_df[snow_df["phase"] == phase]
        ax.scatter(
            rows[name],
            rows["swe_cm"],
            s=42,
            edgecolor="white",
            linewidth=0.8,
            label=f"{phase} ({len(rows)})",
            **style,
        )
    slope, intercept = np.polyfit(snow_df[name], snow_df["swe_cm"], 1)
    xs = np.linspace(snow_df[name].min(), snow_df[name].max(), 2)
    ax.plot(xs, intercept + slope * xs, color="0.3", lw=1, ls="--")
    r = snow_df[[name, "swe_cm"]].corr().iloc[0, 1]
    ax.set_title(f"r = {r:.2f}", loc="left", fontsize=10)
    ax.axvline(0, color="0.75", lw=0.6)
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("April 1 SWE at Paradise [cm]")
axes[0].legend(title="ENSO phase (RONI, DJF)", fontsize=8, frameon=False)
fig.suptitle(
    "Winter climate modes and April 1 snowpack, Paradise SNOTEL, Mount Rainier"
)
fig.tight_layout()

# %%
# The numbers behind the figure: each index's correlation with April 1 SWE, and
# the mean April 1 SWE by ENSO phase.
print(
    snow_df.drop(columns="phase").corr()["swe_cm"].drop("swe_cm").round(2).to_string()
)
print(snow_df.groupby("phase")["swe_cm"].agg(["mean", "count"]).round(1).to_string())
