"""
Hillshade: Natural Earth shaded relief as a basemap
===================================================

Natural Earth publishes global grayscale shaded relief in five styles, from a
plain hillshade to "Gray Earth" with hypsometric tints, ocean-bottom relief
and drainages, at 1 arcmin (the 1:10m scale) or 2 arcmin (1:50m).
``esd.terrain.hillshade.load`` fetches one style once into the easysnowdata
cache and clips it to the AOI; with no AOI it returns the globe.

The values are cartographic brightness (0-255), not a physical quantity, so
the layer belongs under other data rather than in an analysis. For a
hillshade at DEM resolution, shade a DEM instead (see the DEM example).

The figures show the default style over the central Cascades, the Wrzesien
mountain snow mask drawn over it, and the whole globe in the Robinson
projection the global snowmelt runoff onset maps use.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-123.0, 46.0, -120.5, 48.0)  # the central Cascades, Puget Sound to Yakima

box = esd.parse_aoi(aoi)
grid = box.to_geobox(crs="utm", resolution=1000)  # the maps below share this grid
print(box)

# %%
# One route, no account; the style and scale choose the archive.
for src in esd.catalog.get("hillshade").sources:
    print(f"{src.id:15} {src.title:25} {', '.join(src.requires) or 'no account'}")
for style, (description, stems) in esd.terrain.hillshade.STYLES.items():
    print(f"{style:28} {', '.join(stems):8} {description}")

# %%
# The default style, Gray Earth with ocean bottom and drainages at 1 arcmin
# (~1.85 km), resampled onto the AOI's 1 km UTM grid. Puget Sound, the
# Columbia and the Cascade volcanoes are all legible at this scale.
hillshade_da = esd.terrain.hillshade.load(box.buffer(5000))
print(hillshade_da)

utm_hillshade_da = hillshade_da.odc.reproject(grid, resampling="bilinear")
ax = esd.plotting.map(
    utm_hillshade_da,
    cmap="gray",
    vmin=0,
    vmax=255,
    colorbar=False,
    title="Natural Earth Gray Earth, 1:10m",
)
ax.figure.tight_layout()

# %%
# As a basemap: the plain ``shaded-relief`` style under the mountain snow
# mask's seasonal and ephemeral classes, half transparent so the relief shows
# through where the classes are.
relief_da = esd.terrain.hillshade.load(box.buffer(5000), style="shaded-relief")
mask_da = esd.snow.mountain_snow_mask.load(box.buffer(5000), layer="mountain_snow")
utm_mask_da = mask_da.odc.reproject(grid, resampling="nearest")

fig, ax = plt.subplots(figsize=(8, 6))
esd.plotting.map(
    relief_da.odc.reproject(grid, resampling="bilinear"),
    ax=ax,
    cmap="gray",
    vmin=0,
    vmax=255,
    colorbar=False,
    scalebar=False,
    graticule=False,
)
esd.plotting.categorical(
    utm_mask_da.where(utm_mask_da != 255),  # Fill (not mountain) shows the relief
    ax=ax,
    alpha=0.55,
    title="mountain snow classes over shaded relief",
)
fig.tight_layout()

# %%
# The whole globe at 1:50m, reprojected to Robinson (ESRI:54030) at 20 km.
# This is the step the runoff-onset notebook does with ``gdalwarp``; for a
# figure, reprojecting in memory is enough. ``rio.reproject`` (GDAL's warper)
# rather than ``odc.reproject``, because odc cannot trace the outline of a
# grid that reaches the poles in Robinson; pixels outside the projection's
# outline are filled with 255 and masked.
world_da = esd.terrain.hillshade.load(scale="50m", style="gray-earth-ocean")
robinson_da = world_da.rio.reproject("ESRI:54030", resolution=20_000, nodata=255)
robinson_da = robinson_da.where(robinson_da != 255)
print(robinson_da.sizes)

ax = esd.plotting.map(
    robinson_da,
    cmap="gray",
    vmin=0,
    vmax=255,
    colorbar=False,
    scalebar=False,
    graticule=False,  # the edge labels would fall outside the Robinson outline
    title="Natural Earth Gray Earth with ocean bottom, Robinson",
    figsize=(11, 6),
)
ax.figure.tight_layout()
