"""
Russian route snow surveys (RIHMI-WDC)
======================================

Russia's weather stations survey snow along 1–2 km routes in the field and in
forest every 10 days through the cold season (every 5 days during melt),
measuring depth, density and SWE: the largest SWE record in Eurasia. The
Russian Research Institute of Hydrometeorological Information (RIHMI-WDC)
holds it from 1966 to the present, but its own portal (AISORI-M) needs a free login
and has no API. The open copy is NSIDC's G01170 mirror, **1966–1996**, about
1,350 stations.

``RIHMIClient`` reads the NSIDC mirror with no account, and can also read
zips a user has exported from AISORI-M by hand (``data_dir=``; the client
docstring lists the export steps). Every survey
record is ``"periodic"`` and says which route it came from (``route_type``).
It is not part of ``esd.stations.inventory()`` or ``load()``.

The figures show the stations in the public mirror, and the field and forest
SWE at Yakutsk through the 1980s.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

import easysnowdata as esd
from easysnowdata.stations.clients import RIHMIClient

client = RIHMIClient()

# %%
# Stations in the public mirror, coloured by whether RIHMI still lists them in
# its current route-survey catalogue (data after 1996 needs an AISORI-M login).
stations = [s for s in client.get_all_stations() if s["in_nsidc_g01170"]]
stations_gdf = gpd.GeoDataFrame(
    pd.DataFrame(stations).set_index("station_id"),
    geometry=gpd.points_from_xy(
        [s["longitude"] for s in stations], [s["latitude"] for s in stations]
    ),
    crs="EPSG:4326",
)
stations_gdf["catalogue"] = stations_gdf["in_route_survey_catalogue"].map(
    {True: "still surveyed (login for data after 1996)", False: "public mirror only"}
)
print(stations_gdf["catalogue"].value_counts().to_string())

ax = esd.plotting.points(
    stations_gdf.cx[15:180, :],
    column="catalogue",
    legend_label="RIHMI route surveys",
    title="Route snow surveys in NSIDC G01170",
    markersize=3,
    figsize=(12, 6),
)

# %%
# Field and forest routes at Yakutsk (WMO 24959), 1980–1990. Forest routes
# hold more snow: less wind, less sublimation.
records = client.get_data(
    "24959", variables="swe", begin_date="1980-07-01", end_date="1990-06-30"
)
swe_df = pd.DataFrame(records).assign(date=lambda df: pd.to_datetime(df["date"]))
print(swe_df.groupby("route_type").size().to_string())
fig, ax = plt.subplots(figsize=(10, 4))
for route, group in swe_df.groupby("route_type"):
    ax.plot(group["date"], group["value"], ".", label=f"{route} route")
ax.set(ylabel="SWE (cm)", title="Route-survey SWE at Yakutsk")
ax.legend()
