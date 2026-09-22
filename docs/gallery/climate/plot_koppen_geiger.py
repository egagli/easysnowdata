"""
Köppen-Geiger climate classification (Beck et al.)
==================================================

The Köppen-Geiger scheme sorts the world into 30 climate classes from monthly
temperature and precipitation: tropical (A), arid (B), temperate (C), cold (D)
and polar (E), each subdivided by seasonality and summer warmth. Beck et al.
(2023) rebuilt the maps at 1 km from station-constrained climatologies for five
historical 30-year periods, and projected them to 2041-2070 and 2071-2099 under
seven CMIP6 scenarios.

One route, the paper's figshare archive, needs no account: a single 1.3 GB zip
that is read in place, one GeoTIFF at a time. ``load`` takes ``period=``,
``scenario=`` and ``resolution=`` (1°, 0.5°, 0.1° or 1 km); ``search`` lists
the archive's layout without downloading it.

The classes come back as ``uint8`` values with CF ``flag_values``,
``flag_meanings`` and ``flag_colors`` attributes, so ``esd.plotting.categorical``
draws the paper's own legend. The first figure is the Pacific Northwest at the
default 0.1°; the second is the Mount Rainier box at 1 km, present day beside
the SSP5-8.5 end-of-century projection, with a table of what changed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import easysnowdata as esd

print(f"easysnowdata {esd.__version__}")

# %%
# Where the product comes from, and what the route asks for.
for src in esd.catalog.get("koppen-geiger").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# What the archive holds, without downloading it: one row per period,
# scenario and resolution.
inventory = esd.climate.koppen_geiger.search()
print(f"{len(inventory)} rasters in the archive")
print(inventory[inventory["resolution"] == "1 km"].head(8).to_string(index=False))

# %%
# The Pacific Northwest at the default 0.1° for the default period, 1991-2020.
# The legend lists only the classes that occur in the array; the value 0 is
# ocean and is kept as the nodata sentinel, so it draws as nothing.
pnw = esd.climate.koppen_geiger.load((-125.0, 42.0, -116.0, 49.5))
print(pnw)
ax = esd.plotting.categorical(pnw, title="Köppen-Geiger classes, 1991-2020, 0.1°")
ax.figure.tight_layout()

# %%
# Mount Rainier at 1 km: the present period beside the end of the century under
# SSP5-8.5. ``all_classes`` puts every class in the legend, so the same legend
# serves both panels, in two columns to keep it shorter than the map.
# The 1 km grid is geographic; the box is loaded with a 2 km margin and both
# periods are drawn on its UTM zone at 1 km, nearest so the classes survive.
box = esd.parse_aoi((-122.6, 46.4, -120.9, 47.3))
grid = box.to_geobox(crs="utm", resolution=1000)
present = esd.climate.koppen_geiger.load(box.buffer(2000), resolution="1 km")
future = esd.climate.koppen_geiger.load(
    box.buffer(2000), period="2071_2099", scenario="ssp585", resolution="1 km"
)
print(present.attrs["archive_member"], "->", future.attrs["archive_member"])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
esd.plotting.categorical(
    present.odc.reproject(grid, resampling="nearest"),
    ax=axes[0],
    legend=False,
    title="1991-2020",
)
esd.plotting.categorical(
    future.odc.reproject(grid, resampling="nearest"),
    ax=axes[1],
    title="2071-2099, SSP5-8.5",
    legend_kwargs={"all_classes": True, "ncol": 2, "fontsize": 7},
)
fig.tight_layout()

# %%
# How much of the box changes class, and to what. ``esd.processing.flags``
# reads the CF flag table back out of the attrs. Land pixels per class: the
# cold dry-summer belts (Dsc, Dsb) and the summit tundra (ET) shrink, and the
# hot-summer Csa, absent today, takes over most of what was warm-summer Csb.
# This is a 1 km grid in degrees, so the counts are pixels, not km².
symbols = esd.processing.flags(present).set_index("value")["meaning"]
land = (present.values != 0) & (future.values != 0)
table = (
    pd.DataFrame(
        {
            "1991-2020": pd.Series(present.values[land].ravel()).value_counts(),
            "2071-2099 SSP5-8.5": pd.Series(future.values[land].ravel()).value_counts(),
        }
    )
    .fillna(0)
    .astype(int)
)
table.index = [symbols[v] for v in table.index]
table["change"] = table["2071-2099 SSP5-8.5"] - table["1991-2020"]
print(table.sort_values("change").to_string())
changed = np.mean(present.values[land] != future.values[land])
print(f"{changed:.0%} of the land pixels change class")
