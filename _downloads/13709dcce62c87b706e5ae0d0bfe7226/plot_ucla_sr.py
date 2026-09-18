# esd-requires: earthdata
"""
UCLA snow reanalysis, and when virtualization pays off
======================================================

The UCLA reanalysis is one NetCDF-4 granule per water year and 1° tile, with
no DMR++ sidecars. Reading a single water year straight through
``earthaccess.open`` is fastest; over many water years the one-off HDF5
metadata scan that ``virtualize=True`` performs is cheaper than reopening
every file, and the references it writes are cached for next time. That is
what ``virtualize="auto"`` decides for you.

Needs an Earthdata account.
"""

import time as timing

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)

# %%
# One water year, the plain path.
started = timing.perf_counter()
swe = esd.snow.ucla_sr.load(aoi, "2019-10-01/2020-09-30")
print(
    f"one water year: virtualized={swe.attrs['virtualized']}, "
    f"access={swe.attrs['access']}, {timing.perf_counter() - started:.1f}s to open"
)

basin_mean = swe.mean(dim=["latitude", "longitude"]).compute()

fig, ax = plt.subplots(figsize=(9, 3.5))
basin_mean.plot(ax=ax, color="tab:blue")
ax.set_ylabel("basin-mean SWE (m)")
ax.set_title("UCLA reanalysis, water year 2020")
fig.tight_layout()

# %%
# The ensemble statistics: the reanalysis carries a spread, not just a mean.
# (Phase 0 fixed the old mapping, where the median and the 25th percentile
# pointed at the same slice.)
common = dict(aoi=aoi, time="2020-03-01/2020-03-31")
median = esd.snow.ucla_sr.load(**common, stats="median")
q25 = esd.snow.ucla_sr.load(**common, stats="25pct")
q75 = esd.snow.ucla_sr.load(**common, stats="75pct")

fig, ax = plt.subplots(figsize=(9, 3.5))
for label, series in (("25th", q25), ("median", median), ("75th", q75)):
    series.mean(dim=["latitude", "longitude"]).compute().plot(ax=ax, label=label)
ax.legend()
ax.set_ylabel("SWE (m)")
ax.set_title("Posterior ensemble spread, March 2020")
fig.tight_layout()

# %%
# A long series: past a handful of granules ``virtualize="auto"`` turns the
# reference path on. Run this twice to see the cache do its job.
started = timing.perf_counter()
long_series = esd.snow.ucla_sr.load(aoi, "2015-10-01/2021-09-30")
print(
    f"six water years: virtualized={long_series.attrs['virtualized']}, "
    f"{timing.perf_counter() - started:.1f}s to open"
)
print(f"reference cache: {esd.config.cache_dir('virtual', 'ucla_sr')}")

# %%
# The High Mountain Asia sibling is the same reanalysis over a different
# domain, and the same call reaches it:
#
# .. code-block:: python
#
#     hma = esd.snow.ucla_sr.load((80.0, 30.0, 81.0, 31.0), "2000-10/2001-09",
#                                 region="hma")
peak = float(np.nanmax(basin_mean.values))
print(f"peak basin-mean SWE in WY2020: {peak:.2f} m")
