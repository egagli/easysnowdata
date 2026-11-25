# esd-requires: earthdata
"""
Glacier outlines: Randolph Glacier Inventory 7.0 and 6.0
========================================================

The Randolph Glacier Inventory (RGI) is the global inventory of glacier
outlines outside the ice sheets. ``esd.boundaries.glaciers.load`` reads either
version:

* ``version="7.0"`` (default; RGI 7.0 Consortium 2023) — outlines targeted at
  the year 2000, as glaciers (``product="glaciers"``) or glacier complexes
  (``product="complexes"``, contiguous ice as one polygon);
* ``version="6.0"`` (RGI Consortium 2017) — the inventory most work up to 2023
  used (OGGM and many mass-balance products), kept for comparison with it.

Both are distributed per first-order region. The loader finds the regions the
AOI touches, fetches each regional zip once into the cache, and reads the AOI.
NSIDC (``source="nsidc"``, the default) needs an Earthdata Login;
``source="oggm-mirror"`` is OGGM's credential-free mirror of the original
GLIMS files, for 6.0 only. The first columns are the same for both versions:
``rgi_id``, ``name``, ``area_km2`` and ``o1region``.

The figures show the 19 regions, Mount Rainier's glaciers over a hillshade of
the 30 m Copernicus DEM, and the Great Aletsch Glacier, where 7.0 redrew the
6.0 outlines; the text compares the versions and the Natural Earth map layer.
"""

from matplotlib.colors import LightSource

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

box = esd.parse_aoi(aoi)
utm = box.utm_crs
print(box)

for src in esd.catalog.get("rgi-glaciers").sources:
    print(f"{src.id:12} {src.title:28} {', '.join(src.requires) or 'no account'}")

# %%
# The first-order regions, from RGI 7.0's region file. ``regions(aoi)`` is how
# the loader decides which regional archives an AOI needs; Rainier sits in
# region 2, Western Canada and USA. (Region 20, the Antarctic mainland, has an
# outline but no glacier file.)
regions_gdf = esd.boundaries.glaciers.regions()
print(esd.boundaries.glaciers.regions(aoi)[["o1region", "name"]].to_string(index=False))

robinson = "ESRI:54030"
world_gdf = esd.boundaries.admin.countries()
ax = world_gdf.to_crs(robinson).plot(
    color="0.9", edgecolor="white", linewidth=0.3, figsize=(11, 6)
)
regions_gdf.to_crs(robinson).plot(
    ax=ax, column="o1region", cmap="tab20", alpha=0.55, edgecolor="0.3", linewidth=0.5
)
for _, region in (
    regions_gdf.dissolve("o1region").reset_index().to_crs(robinson).iterrows()
):
    point = region.geometry.representative_point()
    ax.annotate(
        f"{region['o1region']:02d}", (point.x, point.y), ha="center", fontsize=8
    )
esd.plotting.finish_map(ax, robinson, scalebar=False, graticule=False)
ax.set_title("RGI 7.0 first-order regions")
ax.figure.tight_layout()

# %%
# Rainier's glaciers from RGI 7.0, coloured by area, with the glacier
# complexes outlined: where ice is continuous across a divide the complex
# product keeps it as one polygon.
glaciers_gdf = esd.boundaries.glaciers.load(aoi)
complexes_gdf = esd.boundaries.glaciers.load(aoi, product="complexes")
print(
    f"RGI 7.0: {len(glaciers_gdf)} glaciers, {len(complexes_gdf)} complexes, "
    f"{glaciers_gdf['area_km2'].sum():.1f} km2"
)
largest_df = glaciers_gdf.nlargest(8, "area_km2")[
    ["rgi_id", "name", "area_km2", "zmin_m", "zmax_m", "aspect_deg"]
]
print(largest_df.to_string(index=False))

grid = box.to_geobox(crs="utm", resolution=30)
dem_da = esd.terrain.dem.load(aoi, crs="utm", grid_resolution=30, chunks=None)
shade = LightSource(azdeg=315, altdeg=45).hillshade(
    dem_da.values, vert_exag=1, dx=30, dy=30
)
shade_da = dem_da.copy(data=shade)
shade_da.attrs = {"long_name": "hillshade"}

ax = esd.plotting.map(
    shade_da,
    cmap="gray",
    vmin=0,
    vmax=1,
    colorbar=False,
    figsize=(9, 7),
    title="RGI 7.0 glaciers (colour: area) and complexes (black)",
)
esd.plotting.add_outline(
    ax,
    glaciers_gdf,
    column="area_km2",
    cmap="viridis",
    alpha=0.7,
    edgecolor="none",
    legend=True,
    legend_kwds={"label": "glacier area [km2]", "shrink": 0.7},
)
esd.plotting.add_outline(ax, complexes_gdf, edgecolor="black", linewidth=0.6)
ax.figure.tight_layout()

# %%
# The same box in RGI 6.0, here from the credential-free OGGM mirror (the
# NSIDC copy is the same file). Around Rainier, 7.0 did not redraw anything:
# every one of its outlines is a 6.0 outline (``is_rgi6``), and ``src_date``
# shows where they come from — USGS topographic mapping of 1959 and 1970, not
# the year 2000 that 7.0 targets. 7.0 only drops nine unnamed patches of
# 0.01 km2, at the inventory's size threshold. Check ``src_date`` before using
# RGI outlines as a year-2000 glacier extent.
rgi6_gdf = esd.boundaries.glaciers.load(aoi, version="6.0", source="oggm-mirror")
print(
    f"RGI 6.0: {len(rgi6_gdf)} glaciers, {rgi6_gdf['area_km2'].sum():.1f} km2; "
    f"RGI 7.0: {len(glaciers_gdf)} glaciers, {glaciers_gdf['area_km2'].sum():.1f} km2"
)
print(glaciers_gdf["is_rgi6"].value_counts().to_dict())
print(glaciers_gdf["src_date"].str[:4].value_counts().to_dict())

# %%
# Where 7.0 did redraw, the outlines change. Around the Great Aletsch Glacier
# (RGI region 11) every 7.0 outline is new, from 2003 imagery. The totals
# agree to within a few tenths of a percent, but the edges shift, most visibly
# around the small cirque glaciers and nunataks.
aletsch = (7.95, 46.38, 8.12, 46.56)
aletsch7_gdf = esd.boundaries.glaciers.load(aletsch)
aletsch6_gdf = esd.boundaries.glaciers.load(
    aletsch, version="6.0", source="oggm-mirror"
)
print(
    f"Aletsch box: 6.0 {len(aletsch6_gdf)} glaciers, "
    f"{aletsch6_gdf['area_km2'].sum():.1f} km2; 7.0 {len(aletsch7_gdf)} glaciers, "
    f"{aletsch7_gdf['area_km2'].sum():.1f} km2; "
    f"new outlines in 7.0: {int((aletsch7_gdf['is_rgi6'] == 0).sum())}"
)

aletsch_box = esd.parse_aoi(aletsch)
aletsch_utm = aletsch_box.utm_crs
aletsch_dem_da = esd.terrain.dem.load(
    aletsch, crs="utm", grid_resolution=30, chunks=None
)
aletsch_shade_da = aletsch_dem_da.copy(
    data=LightSource(azdeg=315, altdeg=45).hillshade(
        aletsch_dem_da.values, vert_exag=1, dx=30, dy=30
    )
)
aletsch_shade_da.attrs = {"long_name": "hillshade"}
ax = esd.plotting.map(
    aletsch_shade_da,
    cmap="gray",
    vmin=0,
    vmax=1,
    colorbar=False,
    figsize=(8, 8),
    title="Great Aletsch: RGI 6.0 (orange) against the redrawn 7.0 (blue)",
)
esd.plotting.add_outline(ax, aletsch6_gdf, edgecolor="#e6550d", linewidth=1.4)
esd.plotting.add_outline(ax, aletsch7_gdf, edgecolor="#08519c", linewidth=0.9)
ax.figure.tight_layout()

# %%
# Natural Earth's ``glaciated_areas`` layer is a map layer, generalized for
# 1:10m: around Rainier it is one blob, larger than the glaciers it stands for.
# Use it for small-scale maps and the RGI for anything measured.
ice_gdf = esd.boundaries.natural_earth.load(aoi, layer="glaciated_areas")
rainier_ice_gdf = ice_gdf.clip(box.geometry)
print(
    f"Natural Earth 1:10m glaciated area in the box: "
    f"{rainier_ice_gdf.to_crs(utm).area.sum() / 1e6:.1f} km2, against "
    f"{glaciers_gdf.clip(box.geometry).to_crs(utm).area.sum() / 1e6:.1f} km2 in RGI 7.0"
)
