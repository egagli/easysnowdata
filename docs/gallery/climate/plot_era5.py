# esd-requires: earthengine
"""
ERA5 reanalysis (ARCO-ERA5 and Earth Engine)
============================================

ERA5 is ECMWF's global reanalysis: hourly fields on a 0.25° grid from 1940 to
about a week ago, with the last three months served as the preliminary ERA5T.
ERA5-Land re-runs the land surface at 0.1° with the same forcing. Both are the
usual source of temperature and precipitation for a basin without a station.

Two routes. The default, ``arco-era5-gcs``, is Google's analysis-ready Zarr
copy of hourly ERA5 on Cloud Storage: anonymous, no account, lazy. Everything
else, ERA5-Land and the daily and monthly aggregates, lives on Earth Engine
(``source="gee"``) and needs Earth Engine credentials.

The figures put one week of 2 m temperature from both routes on one calendar
axis, then map the ERA5-Land grid at its coldest hour to show what a
reanalysis cell is: this 40 km box selects a single 0.25° ERA5 cell, and its
temperature is that of the cell's average elevation, not of a summit.
"""

import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier
week = "2023-03-01/2023-03-07"

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the product comes from, and what each route asks for.
for src in esd.catalog.get("era5").sources:
    print(f"{src.id:22} {src.title:45} {', '.join(src.requires) or 'no account'}")

# %%
# The credential-free route: one week of hourly 2 m temperature and snow depth
# from ARCO-ERA5. The result is Dask-backed with ``time``, ``latitude``,
# ``longitude`` dims, and the subset holds the native cells whose centres fall
# in the box: at 0.25° that is a single cell here.
era5 = esd.climate.era5.load(aoi, week, variables=["2m_temperature", "snow_depth"])
print(era5)

# %%
# The same week from ERA5-Land on Earth Engine. ERA5-Land names the variable
# ``temperature_2m`` and resolves the box into 4 x 5 cells at 0.1°.
land = esd.climate.era5.load(
    aoi, week, variables=["temperature_2m"], source="gee", version="ERA5_LAND"
)
print(land)


# %%
# Domain means in degrees Celsius, on one axis. ``timeseries`` labels the y
# axis from the array's ``long_name`` and ``units`` attrs, so set them after
# the conversion.
def domain_mean_celsius(da):
    out = (da.mean(dim=["latitude", "longitude"]) - 273.15).compute()
    out.attrs = {"long_name": "2 m temperature", "units": "°C"}
    return out


t2m = xr.concat(
    [
        domain_mean_celsius(era5["2m_temperature"]),
        domain_mean_celsius(land["temperature_2m"]),
    ],
    dim=pd.Index(
        ["ERA5, ARCO-ERA5 on GCS (0.25°)", "ERA5-Land, Earth Engine (0.1°)"],
        name="source",
    ),
)
ax = esd.plotting.timeseries(
    t2m, title="Mount Rainier AOI, domain-mean 2 m temperature"
)
ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
ax.figure.tight_layout()

# %%
# ERA5-Land runs a degree or two colder here because its 0.1° cells sit higher
# on average than the one 0.25° ERA5 cell the box selects; neither is a summit
# temperature. The map of the coldest hour shows the twenty ERA5-Land cells,
# the single ERA5 cell outlined, and the summit marked.
coldest = pd.Timestamp(t2m.sel(source=t2m["source"][1]).idxmin("time").item())
frame = (land["temperature_2m"].sel(time=coldest) - 273.15).compute()
frame.attrs = {"long_name": "ERA5-Land 2 m temperature", "units": "°C"}
ax = esd.plotting.map(
    frame,
    cmap="coolwarm",
    title=f"ERA5-Land 2 m temperature, {coldest:%Y-%m-%d %H:%M} UTC",
)
lon, lat = float(era5["longitude"].item()), float(era5["latitude"].item())
ax.add_patch(
    Rectangle(
        (lon - 0.125, lat - 0.125),
        0.25,
        0.25,
        fill=False,
        edgecolor="black",
        linewidth=2,
        label="the ERA5 cell the box selects (0.25°)",
    )
)
ax.plot(
    -121.7603,
    46.8523,
    marker="^",
    color="black",
    markersize=9,
    linestyle="none",
    label="summit, 4392 m",
)
ax.legend(fontsize=8, loc="upper right")
ax.figure.tight_layout()

# %%
# Provenance travels with the data: which route served it, the citation and
# licence, and where final ERA5 ends and preliminary ERA5T begins in the ARCO
# store. A request that reaches past ``valid_time_stop`` gets ``era5t_from``.
for key in (
    "source",
    "product_id",
    "license",
    "valid_time_stop",
    "valid_time_stop_era5t",
):
    print(f"{key}: {era5.attrs[key]}")
