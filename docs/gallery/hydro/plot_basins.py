# esd-requires: earthengine
"""
Basins: HUC, HydroBASINS and GRDC
=================================

Four basin products answer the same question, "which drainage is this in?",
at different scales and for different parts of the world. The USGS Watershed
Boundary Dataset (HUC) nests the United States from 2-digit regions to
16-digit sub-watersheds. HydroBASINS is the global equivalent, Pfafstetter
levels 1 to 12. The GRDC layers are coarse and whole-world: 405 major river
basins, and the WMO basins and sub-basins with their region numbers.

Every loader returns a GeoDataFrame in EPSG:4326 holding the features that
*intersect* the AOI, whole and uncut. HUC defaults to the public USGS ArcGIS
REST service and needs no account; ``source="gee"`` reads the 2017 snapshot on
Earth Engine instead. HydroBASINS defaults to the 2.7 GB BasinATLAS
geodatabase (with its several hundred attributes, read with mask pushdown),
or, as here, ``source="hydrosheds"`` fetches one ~300 MB regional zip with the
geometry and Pfafstetter fields only. The GRDC layers are single zips. Nothing
executed on this page needs credentials.

The first figure fills the HUC12 sub-watersheds inside the HUC8 subbasins,
then lays the HUC8 divides over HydroBASINS level 8: two independent
delineations of the same drainage. The second nests level 6 inside the WMO
basins and steps out to the GRDC major basin the whole AOI drains to.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

box = esd.parse_aoi(aoi)
utm = box.utm_crs  # the local maps are drawn in the AOI's UTM zone
print(box, utm)

# %%
# Where each product comes from, and what the route asks for.
for product_id in ("huc", "hydrobasins", "grdc-major-river-basins", "grdc-wmo-basins"):
    for src in esd.catalog.get(product_id).sources:
        print(
            f"{product_id:24} {src.id:20} {src.title:28} {', '.join(src.requires) or 'no account'}"
        )

# %%
# HUC8 subbasins and HUC12 sub-watersheds from the REST service. The loader
# pages the service in small blocks, because one unpaged HUC12 query over a
# large box times out and comes back as HTML rather than GeoJSON.
huc8 = esd.hydro.basins.huc(aoi, level=8)
huc12 = esd.hydro.basins.huc(aoi, level=12)
print(huc8[["name", "huc8", "areasqkm", "states"]].to_string(index=False))
print(f"{len(huc12)} HUC12 sub-watersheds intersect the box")

# %%
# The same call with ``source="gee"`` returns the same columns from the 2017
# WBD snapshot on Earth Engine, for users who already hold an account:
#
# .. code-block:: python
#
#     huc8_gee = esd.hydro.basins.huc(aoi, level=8, source="gee")
#
# The geometries differ slightly from the REST service's, which is current
# and rounds coordinates to six decimals.

# %%
# HydroBASINS, here from the HydroATLAS copy on Earth Engine. The HydroSHEDS
# regional zip (``source="hydrosheds"``, no account, cached after the first
# download) serves the same polygons, but that server refuses cloud-runner
# addresses such as the one this page is built on, so the Earth Engine route
# is used for the build; on your own machine either works. ``PFAF_ID`` is the
# Pfafstetter code: every digit is one level of nesting, so a level-8 basin's
# level-6 parent is its first six digits. Level 8 here is 1 500 to 2 600 km²
# per basin, the size of a HUC8.
hybas_source = "hydrosheds"
try:
    hybas8 = esd.hydro.basins.hydrobasins(aoi, level=8, source=hybas_source)
except Exception as exc:  # the HydroSHEDS server blocks some networks
    print(
        f"HydroSHEDS route unavailable here ({type(exc).__name__}); using Earth Engine"
    )
    hybas8 = esd.hydro.basins.hydrobasins(aoi, level=8, source="gee")
print(hybas8[["HYBAS_ID", "PFAF_ID", "SUB_AREA", "UP_AREA"]].to_string(index=False))

# %%
# Left: HUC12s coloured by name inside the HUC8 outlines. Right: the HUC8
# divides over HydroBASINS level 8. The two products were drawn from
# different elevation models by different rules, and their main divides still
# agree closely; the units are not one-to-one, though: 78210029 covers
# 1 520 km² of the 2 655 km² Upper Cowlitz HUC8, the rest falling in level-8
# units that do not touch the box and so were not returned. Vector layers
# draw with geopandas, here reprojected to the AOI's UTM zone; ``finish_map``
# adds the basemap, graticule and scale bar.
huc8_utm, huc12_utm, hybas8_utm = (g.to_crs(utm) for g in (huc8, huc12, hybas8))
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
huc12_utm.plot(
    ax=axes[0],
    column="name",
    cmap="tab20",
    alpha=0.55,
    edgecolor="white",
    linewidth=0.5,
)
huc8_utm.boundary.plot(ax=axes[0], color="black", linewidth=1.6)
for _, row in huc8_utm.iterrows():
    axes[0].annotate(
        row["name"],
        row.geometry.centroid.coords[0],
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
axes[0].set_title("HUC12 sub-watersheds inside the HUC8 subbasins (USGS WBD)")

palette8 = dict(zip(hybas8["PFAF_ID"], ("#fdc086", "#beaed4", "#7fc97f"), strict=True))
hybas8_utm.plot(
    ax=axes[1],
    color=hybas8_utm["PFAF_ID"].map(palette8),
    alpha=0.55,
    edgecolor="white",
    linewidth=0.8,
)
huc8_utm.boundary.plot(ax=axes[1], color="black", linewidth=1.4, linestyle="--")
axes[1].legend(
    handles=[
        *[
            Patch(facecolor=c, alpha=0.55, label=f"HydroBASINS {k}")
            for k, c in palette8.items()
        ],
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.4,
            linestyle="--",
            label="HUC8 (USGS WBD)",
        ),
    ],
    fontsize=8,
    loc="lower right",
)
axes[1].set_title("HUC8 divides over HydroBASINS level 8")
x0, y0, x1, y1 = box.to_crs(utm).total_bounds
for ax in axes:
    ax.set_xlim(x0 - 30_000, x1 + 30_000)
    ax.set_ylim(y0 - 30_000, y1 + 30_000)
    esd.plotting.finish_map(ax, utm, basemap=True)
fig.tight_layout()

# %%
# The GRDC layers. The WMO basins carry region numbers and names; the major
# river basins are one polygon each. The shipped area columns of the major
# basins layer (``ACRES``, ``LAEA_HA``) are zero in this release, so the area
# below is taken from the geometry in an equal-area projection.
wmo = esd.hydro.basins.grdc_wmo(aoi)
print(wmo[["WMOBB", "WMOBB_NAME", "REGNAME", "SUM_SUB_AREA"]].to_string(index=False))
major = esd.hydro.basins.grdc_major(aoi)
area_km2 = major.to_crs("ESRI:102008").area.iloc[0] / 1e6
print(
    f"{major.iloc[0]['NAME']} basin: shipped ACRES={major.iloc[0]['ACRES']}, geometry {area_km2:,.0f} km2"
)

# %%
# Left: level-6 HydroBASINS (fill) inside the WMO basins (outline: the
# Columbia, and the North Pacific coastal basins), with the AOI in red. Right:
# the GRDC major basin, the Columbia, with the same AOI as a dot, to show the
# scale jump from a sub-watershed to a continental basin.
hybas6 = esd.hydro.basins.hydrobasins(aoi, level=6, source=hybas_source)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
palette6 = dict(zip(hybas6["PFAF_ID"], ("#66c2a5", "#e6f598", "#8da0cb"), strict=True))
hybas6.plot(
    ax=axes[0],
    color=hybas6["PFAF_ID"].map(palette6),
    alpha=0.6,
    edgecolor="white",
    linewidth=0.8,
)
wmo.boundary.plot(ax=axes[0], color="black", linewidth=1.4)
esd.aoi.parse_aoi(aoi).footprint.boundary.plot(ax=axes[0], color="red", linewidth=1.5)
handles = [
    Patch(facecolor=c, alpha=0.6, label=f"PFAF {k}") for k, c in palette6.items()
]
handles += [
    Line2D([], [], color="black", linewidth=1.4, label="GRDC / WMO basins"),
    Line2D([], [], color="red", linewidth=1.5, label="AOI"),
]
axes[0].legend(handles=handles, fontsize=8, loc="upper right")
x0, y0, x1, y1 = hybas6.total_bounds
axes[0].set_xlim(x0 - 0.2, x1 + 0.2)
axes[0].set_ylim(y0 - 0.2, y1 + 0.2)
axes[0].set_title("HydroBASINS level 6 inside the GRDC / WMO basins")

major.plot(
    ax=axes[1], facecolor="#cfe3f7", edgecolor="#1f4e79", linewidth=1.2, alpha=0.8
)
axes[1].plot(
    *esd.aoi.parse_aoi(aoi).geometry.centroid.coords[0],
    marker="o",
    color="red",
    markersize=6,
)
axes[1].set_title(f"GRDC major basin: {major.iloc[0]['NAME']}, {area_km2:,.0f} km²")
for ax in axes:
    esd.plotting.finish_map(ax, hybas6.crs, basemap=True)
fig.tight_layout()
