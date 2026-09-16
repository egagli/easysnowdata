"""
ERA5 hourly temperature over Mount Rainier
==========================================

Hourly 2 m temperature from ARCO-ERA5 on Google Cloud Storage: no credentials,
one week behind real time through ERA5T. The same call reaches ERA5-Land and
the daily / monthly aggregates with ``source="gee"``.
"""

import matplotlib.pyplot as plt

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier

# %%
# Load one week of hourly 2 m temperature. The result is Dask-backed, with
# ``time``/``latitude``/``longitude`` dims because the grid is geographic.
era5 = esd.climate.era5.load(
    aoi, "2023-03-01/2023-03-07", variables=["2m_temperature", "snow_depth"]
)
print(era5)

# %%
# The domain average through the week, in degrees Celsius.
t2m = (era5["2m_temperature"].mean(dim=["latitude", "longitude"]) - 273.15).compute()

fig, ax = plt.subplots(figsize=(9, 3.5))
t2m.plot(ax=ax, color="firebrick")
ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
ax.set_ylabel("2 m temperature (°C)")
ax.set_title("ERA5 hourly 2 m temperature, Mount Rainier")
fig.tight_layout()

# %%
# Provenance travels with the data: which route served it, the citation, the
# licence, and where the final ERA5 archive ends and preliminary ERA5T begins.
for key in ("source", "source_url", "product_id", "license", "valid_time_stop"):
    print(f"{key}: {era5.attrs[key]}")
