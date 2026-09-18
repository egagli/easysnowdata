"""
MODIS snow cover, raw bytes and binary snow
===========================================

MODIS snow products pack a percentage and a set of sentinel values into one
byte, so the loader keeps the raw byte with CF flags rather than guessing what
you want. ``processing.binary_snow`` turns it into a 0/1 mask at whatever NDSI
threshold you choose, leaving cloud and night as NaN.

NSIDC is the default route (the archive of record, and the only one with the
cloud-gap-filled product); Planetary Computer stays as a credential-free
mirror of the two daily and 8-day products.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)

# %%
# The 8-day maximum snow extent from the credential-free mirror.
extent = esd.snow.modis.load(
    aoi, "2023-01-01/2023-02-01", product="MOD10A2", source="planetary-computer"
)
print(extent["Maximum_Snow_Extent"].attrs["flag_meanings"])

fig, ax = plt.subplots(figsize=(6, 5))
esd.plotting.categorical(
    extent["Maximum_Snow_Extent"].isel(time=0), ax=ax, title="MOD10A2 maximum extent"
)
fig.tight_layout()

# %%
# The same scene as a binary mask, and how the snow-covered fraction of the
# box moves through the winter.
binary = esd.processing.binary_snow(extent["Maximum_Snow_Extent"], product="MOD10A2")
fraction = binary.mean(dim=["y", "x"]).compute()

fig, ax = plt.subplots(figsize=(8, 3.5))
fraction.plot(ax=ax, marker="o", color="tab:blue")
ax.set_ylabel("snow-covered fraction")
ax.set_ylim(0, 1)
ax.set_title("MOD10A2 snow-covered fraction, Mount Rainier")
fig.tight_layout()

# %%
# With an Earthdata account, the cloud-gap-filled daily product fills the holes
# that clouds leave in MOD10A1, and the NDSI threshold is yours to choose:
#
# .. code-block:: python
#
#     cgf = esd.snow.modis.load(aoi, "2023-03", product="MOD10A1F")
#     snow = esd.processing.binary_snow(
#         cgf["CGF_NDSI_Snow_Cover"], product="MOD10A1F", threshold=20
#     )
print(f"gap-filled products available from NSIDC: {sorted(esd.snow.modis.PRODUCTS)}")
print(
    f"usable pixels in the mirror scene: {int(np.isfinite(binary.isel(time=0)).sum())}"
)
