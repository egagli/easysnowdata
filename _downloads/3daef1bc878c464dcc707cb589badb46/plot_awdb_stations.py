"""
SNOTEL and snow courses (USDA NRCS AWDB)
========================================

The USDA NRCS Air and Water Database holds the SNOTEL and SNOLITE telemetry
sites, the SCAN soil-climate network and the manual snow courses of the
western United States and parts of western Canada. SNOTEL sites report SWE,
snow depth, precipitation and temperature every day (hourly for most);
snow courses are measured by hand around the first of the month through the
winter. Both kinds sit in the inventory, and ``daily_only`` keeps the first.

There is one route, ``source="awdb"``, the AWDB REST API v1. It needs no
account. The API answers in inches and degrees Fahrenheit; the client converts
to centimetres and degrees Celsius before anything reaches the Dataset.

The figures show the stations around a volcano in the Cascades, coloured by
kind of site; one water year of SWE and snow depth at the daily sites; and
five water years of SWE at one site on a single October-to-September axis.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-122.1, 46.6, -121.3, 47.1)

box = esd.parse_aoi(aoi)
print(box, box.utm_crs)  # the map below is drawn in the AOI's UTM zone

# %%
# Where the product comes from.
for src in esd.catalog.get("awdb-stations").sources:
    print(f"{src.id:20} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# Every AWDB station in the box, then only those whose daily record the
# archive pipeline has actually retrieved. The ``network_code`` column is
# AWDB's own kind-of-site code: ``SNTL`` for SNOTEL, ``SNOW`` for a course.
everything_gdf = esd.stations.inventory(aoi, networks="awdb")
daily_gdf = esd.stations.inventory(aoi, networks="awdb", daily_only=True)
print(f"{len(everything_gdf)} AWDB stations, {len(daily_gdf)} with a daily record")
print(everything_gdf["network_code"].value_counts().to_string())

# %%
# The snow courses sit at the same passes as the SNOTEL sites, often a few
# hundred metres apart: many of them predate the telemetry and were kept.
ax = esd.plotting.points(
    everything_gdf.to_crs(box.utm_crs),
    column="network_code",
    legend_label="AWDB site kind",
    title="AWDB stations around Mount Rainier",
)

# %%
# One water year of SWE and snow depth at the daily sites, in centimetres.
obs_ds = esd.stations.load(daily_gdf, variables=["swe", "snwd"], time="2023-10/2024-09")
print(obs_ds)

# %%
# The legend labels come from the ``name`` and ``elevation_m`` coordinates,
# so the series read as stations rather than as codes.
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
esd.plotting.timeseries(obs_ds["swe"], ax=axes[0], title="Water year 2024")
esd.plotting.timeseries(obs_ds["snwd"], ax=axes[1], title="", legend=False)
fig.tight_layout()

# %%
# Snow depth peaks weeks before SWE does: the pack keeps taking on water as it
# settles and densifies, so the deepest day and the day of peak SWE are not
# the same day.
peak_snwd_da = obs_ds["snwd"].idxmax(dim="time").dt.strftime("%Y-%m-%d")
peak_swe_da = obs_ds["swe"].idxmax(dim="time").dt.strftime("%Y-%m-%d")
for station in obs_ds["station"].values:
    print(
        f"{str(obs_ds['name'].sel(station=station).values):16}"
        f" deepest {str(peak_snwd_da.sel(station=station).values)}"
        f"  peak SWE {str(peak_swe_da.sel(station=station).values)}"
    )

# %%
# Five water years at Paradise, overlaid on one October-to-September axis.
paradise_ds = esd.stations.load("679_WA_SNTL", variables="swe", time="2019-10/2024-09")
ax = esd.plotting.timeseries(
    paradise_ds["swe"],
    by_water_year=True,
    title=f"{str(paradise_ds['name'].values[0])}"
    f" ({float(paradise_ds['elevation_m'].values[0]):.0f} m): five water years",
)
