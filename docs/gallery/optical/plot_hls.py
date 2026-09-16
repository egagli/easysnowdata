"""
HLS: Landsat and Sentinel-2 on one time axis
============================================

HLS v2.0 stacks Landsat 8/9 (L30) and Sentinel-2 (S30) on a common 30 m grid.
The loader returns both products in one Dataset with a ``product`` coordinate,
reading the scene metadata from STAC properties. The default route is NASA's
CMR-STAC (Earthdata Login for the reads); the Planetary Computer mirror needs
no account.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.80, 46.82, -121.72, 46.88)

# %%
# Search first. The frame carries the cloud cover and which satellite flew.
items = esd.optical.hls.search(aoi, "2023-08-01/2023-08-20", cloud_cover=50)
print(items[["datetime", "product", "platform", "eo:cloud_cover"]].to_string())

# %%
# Load both products, masked with the default Fmask flags. Fmask is unreliable
# over snow and ice, which is why masking is opt-in rather than automatic.
hls = esd.optical.hls.load(
    aoi,
    "2023-08-01/2023-08-20",
    bands=["green", "swir16", "Fmask"],
    mask="fmask-default",
    cloud_cover=50,
)
ndsi = esd.processing.ndsi(hls)

fig, axes = plt.subplots(1, min(3, ndsi.sizes["time"]), figsize=(12, 4), sharey=True)
for ax, step in zip(
    [axes] if ndsi.sizes["time"] == 1 else axes, range(ndsi.sizes["time"])
):
    ndsi.isel(time=step).plot.imshow(
        ax=ax, cmap="Blues", vmin=-0.5, vmax=1.0, add_colorbar=False
    )
    when = str(ndsi["time"].values[step])[:10]
    ax.set_title(f"{when} ({str(hls['product'].values[step])})")
    ax.set_aspect("equal")
fig.suptitle("HLS NDSI")
fig.tight_layout()

# %%
# The credential-free mirror serves the same scenes for users without an
# Earthdata account.
mirror = esd.optical.hls.load(
    aoi,
    "2023-08-01/2023-08-06",
    bands=["green"],
    products="S30",
    source="planetary-computer",
    resolution=120,
)
print(mirror.attrs["source"], mirror.attrs["collections"])
