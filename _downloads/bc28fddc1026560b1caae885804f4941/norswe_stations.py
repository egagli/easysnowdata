"""
Northern Hemisphere historical SWE (NorSWE)
===========================================

NorSWE (Mortimer & Vionnet, 2025) is a harmonised, quality-controlled archive
of in-situ SWE across North America, Russia, Finland, Norway and Switzerland,
1979–2021: snow courses, snow pillows, gamma sensors and NOAA NOHRSC airborne
gamma flight lines, about 10,000 sites in all. It is a **frozen** archive (v3
ends in 2021) and most of it duplicates sources that have their own clients
(SNOTEL and NRCS courses in AWDB, CanSWE, NVE, SYKE, SLF, RIHMI-WDC). What it
adds is the airborne-gamma lines, the northeastern U.S. snow courses (NRCC,
Maine, New Hampshire) and one consistent QC across all of it, which makes it
a good validation set for gridded SWE.

``NorSWEClient`` builds a local store on first use: it downloads the 2.4 GB
zip from Zenodo in parallel byte ranges (resumable), unzips the 21.6 GB
netCDF, converts it to a **31 MB** Zarr store and deletes both. That takes
about an hour, mostly the download, and needs about 22 GB of free disk while
it runs (set ``EASYSNOWDATA_CACHE_DIR`` to use a bigger disk). After that a
new client lists every site in seconds.

Because of that first download, this example is **not run** when the docs
are built; the code below is what to run once locally.
"""

import geopandas as gpd
import pandas as pd

import easysnowdata as esd
from easysnowdata.stations.clients import NorSWEClient

client = NorSWEClient()  # the first call to the client builds the local store

# %%
# Every site, with its source prefix, country and method. ``mmask`` marks
# mountain sites, where the QC limits are higher.
stations = client.get_all_stations()
stations_gdf = gpd.GeoDataFrame(
    pd.DataFrame(stations).set_index("station_id"),
    geometry=gpd.points_from_xy(
        [s["longitude"] for s in stations], [s["latitude"] for s in stations]
    ),
    crs="EPSG:4326",
)
print(pd.crosstab(stations_gdf["id_prefix"], stations_gdf["method_short"]))

# %%
# Map by method: the airborne-gamma lines fill the U.S. Midwest and the
# Canadian Prairies, where there are few ground sites.
ax = esd.plotting.points(
    stations_gdf,
    column="method_short",
    legend_label="measurement method",
    title="NorSWE sites",
    markersize=2,
    figsize=(12, 6),
)

# %%
# Airborne-gamma lines are flown a few times a season, so their records are
# ``"periodic"``. The flag carries NOHRSC's soil-moisture method code.
gamma = client.get_all_stations(methods="airborne_gamma", bbox=(-100, 44, -94, 49))
records = client.get_data(
    [s["station_id"] for s in gamma[:3]], variables="swe", include_flags=True
)
print(pd.DataFrame(records).head().to_string())

# %%
# The same daily SNOTEL record as AWDB serves, to 2021: Paradise, Washington.
paradise = pd.DataFrame(
    client.get_data("SNOTEL-679", variables="swe", begin_date="2015-10-01")
)
paradise.assign(date=pd.to_datetime(paradise["date"])).plot(
    x="date", y="value", legend=False, title="Paradise SNOTEL in NorSWE"
)
