# esd-requires: earthengine
"""
SNODAS snow water equivalent and depth (NOHRSC)
===============================================

SNODAS is the NOHRSC Snow Data Assimilation System: a daily 1 km model of
snow water equivalent, snow depth, melt, sublimation and snowpack
temperature over the contiguous United States, assimilating station, airborne
and satellite observations since October 2003.

Two routes carry it. ``source="nsidc"`` (the default) reads the NSIDC G02158
archive, one tar of flat-binary grids per day, with no account.
``source="gee-climate-engine"`` reads Climate Engine's mirror on Earth Engine,
which is lazy and server-side subset but needs Earth Engine credentials and
carries only SWE and snow depth.

The figures show one day of SWE and snow depth over Mount Rainier, the
difference between the two routes on that day, and a week of basin-mean SWE
from both on one calendar axis.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA
when = "2024-03-10/2024-03-16"
day = "2024-03-15"

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("snodas").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# ``search`` lists the days a request covers and where each one lives without
# downloading anything: the archive is one file per day at a predictable URL.
print(esd.snow.snodas.search(aoi, when).to_string(index=False))

snodas = esd.snow.snodas.load(aoi, when)
print(snodas)

# %%
# One day of SWE and snow depth. SNODAS never melts perennial ice out, so SWE
# grows without bound over glaciers and saturates the 16-bit field at
# 32.767 m: the pinned pixels on the summit are Rainier's ice cap, not a
# reader artefact, and they are passed through untouched.
swe = snodas["SWE"].sel(time=day)
depth = snodas["snow_depth"].sel(time=day)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
esd.plotting.map(swe, ax=axes[0], cmap="Blues", vmin=0, vmax=3, title=f"SWE, {day}")
esd.plotting.map(
    depth, ax=axes[1], cmap="Purples", vmin=0, vmax=8, title=f"snow depth, {day}"
)
fig.tight_layout()

saturated = int((swe > 30).sum())
seasonal = snodas["SWE"].where(snodas["SWE"] < 30)
print(f"pixels saturated by the glacier artefact on {day}: {saturated}")
print(
    f"median SWE without them: {float(seasonal.sel(time=day).compute().median()):.2f} m"
)

# %%
# The same week from the Earth Engine mirror. Both routes land on the same
# 1/120 degree grid, but their coordinate values differ in the twelfth decimal,
# so snap the mirror onto the archive's coordinates before subtracting.
mirror = esd.snow.snodas.load(aoi, when, source="gee-climate-engine")
mirror = mirror.reindex(
    latitude=snodas["latitude"], longitude=snodas["longitude"], method="nearest"
)
# SNODAS stores SWE as integer millimetres, and the two routes divide by 1000
# in float32 independently, so what is left is the rounding of that division.
diff = ((mirror["SWE"].sel(time=day) - swe) * 1000).compute()
diff.attrs.update({"long_name": "SWE, Earth Engine minus NSIDC", "units": "mm"})
print(f"largest absolute difference on {day}: {float(np.abs(diff).max()):.4f} mm")

ax = esd.plotting.map(
    diff,
    cmap="RdBu",
    vmin=-0.01,
    vmax=0.01,
    title=f"Earth Engine minus NSIDC, {day}",
    cbar_label="SWE difference [mm]",
)
ax.figure.tight_layout()

# %%
# Basin-mean SWE from both routes on one calendar axis, glaciers masked. The
# two lines sit on top of each other, which is the point.
archive_mean = seasonal.mean(dim=["latitude", "longitude"]).compute()
mirror_mean = (
    mirror["SWE"].where(mirror["SWE"] < 30).mean(dim=["latitude", "longitude"])
).compute()
for series in (archive_mean, mirror_mean):
    series.attrs.update(
        {"long_name": "basin-mean SWE, seasonal snow only", "units": "m"}
    )

ax = esd.plotting.timeseries(archive_mean, marker="o", label="NSIDC G02158")
esd.plotting.timeseries(
    mirror_mean,
    ax=ax,
    marker="x",
    linestyle="--",
    label="Earth Engine (Climate Engine)",
)
ax.set_ylim(bottom=0)
ax.figure.tight_layout()
