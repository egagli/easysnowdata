# esd-requires: earthengine
"""
CHILI heat-load index (CSP ERGo)
================================

The Continuous Heat-Insolation Load Index (Theobald et al. 2015) folds slope,
aspect and latitude into one number for how much solar heat a pixel receives:
0 is the coolest north-facing slope, 1 the warmest south-west-facing one. It
is computed from the ALOS World 3D-30m surface model and served at about 90 m
between 70°N and 70°S. For snow work it is the quickest proxy for which slopes
melt out first.

The only route today is Earth Engine (``source="gee"``, needs
``esd.auth.login("earthengine")``); a DEM-computed heat-load index is the
planned credential-free alternative.

The first figure maps the 0-1 index. The second splits it at the thresholds
the product's authors use for cool (< 0.448) and warm (> 0.767) slopes and
shows the distribution the thresholds cut through. The asset stores the index
scaled to 8 bits and the loader returns those 0-255 values unless asked
otherwise; ``normalize="index"`` divides by 255 and, unlike the pre-0.2
loader, never rescales to the AOI, so a pixel keeps its value whatever box it
is requested in.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the product comes from, and what it needs.
for src in esd.catalog.get("chili").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The 0-1 index on the asset's own grid. Warm colours are warm slopes: the
# south and west flanks of the volcano, and the south-facing walls of every
# valley in the box.
chili = esd.terrain.chili.load(aoi, normalize="index", chunks=None)
print(chili)
esd.plotting.map(
    chili,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1,
    cbar_label="CHILI [0-1]",
    title="Continuous heat-insolation load index",
)

# %%
# Cool, neutral and warm slopes at the published thresholds. Aspect decides
# the class here: the two walls of one valley fall on opposite sides of the
# neutral band, and the summit cone is cool all round because it is steep and
# high enough for slope to beat aspect.
COOL, WARM = 0.448, 0.767
classes = (chili >= COOL).astype("uint8") + (chili > WARM).astype("uint8")
classes = esd.processing.categorical.set_flags(
    classes.where(chili.notnull(), 255).astype("uint8"),
    [0, 1, 2],
    ["cool (< 0.448)", "neutral", "warm (> 0.767)"],
    ["#4575b4", "#ffffbf", "#d73027"],
    long_name="heat-load class",
)
classes = classes.rio.write_crs(chili.rio.crs).rio.write_nodata(255)
fractions = {
    name: float((classes == value).mean())
    for value, name in zip([0, 1, 2], ["cool", "neutral", "warm"])
}
print({k: f"{v:.1%}" for k, v in fractions.items()})

fig, (ax_map, ax_hist) = plt.subplots(
    1, 2, figsize=(13, 5), width_ratios=[1.35, 1], layout="constrained"
)
esd.plotting.categorical(classes, ax=ax_map, title="Heat-load classes")
# Bins of five 8-bit steps, aligned to the asset's own quantisation.
ax_hist.hist(
    chili.values.ravel(), bins=(np.arange(0, 261, 5) - 0.5) / 255, color="0.55"
)
for threshold, label in ((COOL, "cool"), (WARM, "warm")):
    ax_hist.axvline(threshold, color="k", linewidth=1)
    ax_hist.text(
        threshold, ax_hist.get_ylim()[1] * 0.97, f" {label} {threshold}", va="top"
    )
ax_hist.set_xlabel("CHILI [0-1]")
ax_hist.set_ylabel("pixels")
ax_hist.set_title("Distribution inside the box")
for side in ("top", "right"):
    ax_hist.spines[side].set_visible(False)
