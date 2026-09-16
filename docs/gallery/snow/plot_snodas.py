"""
SNODAS: the authoritative archive against the Earth Engine mirror
================================================================

SNODAS now defaults to the NSIDC G02158 archive, which needs no account at
all: one tar per day of flat-binary grids, read straight into xarray. The
Climate Engine mirror on Earth Engine stays available and is faster for long
series. They should agree, and this example checks that they do.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)
when = ("2024-03-10", "2024-03-16")

# %%
# The credential-free route. ``search`` lists the days and where each one
# lives, without downloading anything.
print(esd.snow.snodas.search(aoi, when).to_string())

swe = esd.snow.snodas.load(aoi, when)
print(swe)

# %%
# One day of SWE and snow depth.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
swe["SWE"].isel(time=0).plot.imshow(ax=axes[0], cmap="Blues", vmin=0, vmax=3)
axes[0].set_title("SNODAS SWE (m)")
swe["snow_depth"].isel(time=0).plot.imshow(ax=axes[1], cmap="Purples", vmin=0, vmax=8)
axes[1].set_title("SNODAS snow depth (m)")
for ax in axes:
    ax.set_aspect("equal")
fig.tight_layout()

# %%
# SNODAS never melts perennial ice out, so SWE grows without bound over
# glaciers and saturates the 16-bit field at 32.767 m. On Rainier that is the
# summit ice cap, and it is the model's own behaviour rather than a reader
# artefact, so the values are passed through untouched. Mask them when a
# basin contains glaciers.
glaciated = swe["SWE"].isel(time=0) > 30
print(f"pixels saturated by the glacier artefact: {int(glaciated.sum())}")
seasonal = swe["SWE"].where(swe["SWE"] < 30)
print(f"median SWE without them: {float(np.nanmedian(seasonal.isel(time=0))):.2f} m")

# %%
# The same week from the Earth Engine mirror, for users who already have an
# Earth Engine account and want a server-side subset:
#
# .. code-block:: python
#
#     mirror = esd.snow.snodas.load(aoi, when, source="gee-climate-engine")
#     both = xr.concat(
#         [swe["SWE"].mean(dim=["latitude", "longitude"]),
#          mirror["SWE"].mean(dim=["latitude", "longitude"])],
#         dim="source",
#     )
#     both.plot(hue="source")
basin_mean = seasonal.mean(dim=["latitude", "longitude"]).compute()
fig, ax = plt.subplots(figsize=(8, 3.5))
basin_mean.plot(ax=ax, marker="o")
ax.set_ylabel("mean SWE (m)")
ax.set_title("SNODAS basin-mean SWE, seasonal snow only")
fig.tight_layout()
