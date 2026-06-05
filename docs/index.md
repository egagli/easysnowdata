# easysnowdata

[![PyPI](https://img.shields.io/pypi/v/easysnowdata.svg)](https://pypi.python.org/pypi/easysnowdata)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/easysnowdata.svg)](https://anaconda.org/conda-forge/easysnowdata)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
[![CI](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml/badge.svg)](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python package to easily retrieve data relevant to snow science.**

`easysnowdata` unifies access to a wide range of snow-relevant geospatial
datasets under a consistent API that returns **xarray** objects. The emphasis
is on minimising downloads and local computation by leveraging cloud-optimised
data formats (Zarr, COGs, STAC) wherever possible.

![easysnowdata gallery](https://github.com/user-attachments/assets/5b2c83a4-b732-4c35-86fd-1bccb954c286)

---

## Modules at a glance

| Module | Contents |
|--------|----------|
| [`automatic_weather_stations`](automatic_weather_stations.md) | SNOTEL & CCSS station metadata + multi-variable time series |
| [`hydroclimatology`](hydroclimatology.md) | ERA5, SNODAS, UCLA snow reanalysis, watershed/basin geometries, Köppen-Geiger |
| [`remote_sensing`](remote_sensing.md) | Sentinel-1, Sentinel-2, HLS, MODIS snow, ESA WorldCover, forest cover, snow classification |
| [`topography`](topography.md) | Copernicus DEM (30 m / 90 m), CHILI topographic index |
| [`utils`](utils.md) | Shared helpers: bbox conversion, water-year utilities, STAC configs |

---

## Installation

=== "pip"

    ```bash
    pip install easysnowdata
    ```

=== "conda / mamba"

    ```bash
    conda install -c conda-forge easysnowdata
    # or
    mamba install -c conda-forge easysnowdata
    ```

=== "Development (pixi)"

    ```bash
    git clone https://github.com/egagli/easysnowdata.git
    cd easysnowdata
    pixi install
    pixi run test-fast   # credential-free tests
    ```

---

## Five-minute quickstart

```python
import easysnowdata

bbox = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier, WA

# Automatic weather station data
sc = easysnowdata.automatic_weather_stations.StationCollection()
sc.get_data(stations="679_WA_SNTL", variables=["WTEQ", "SNWD"],
            start_date="2023-10-01", end_date="2024-06-30")
sc.data.plot()

# Copernicus DEM
dem = easysnowdata.topography.get_copernicus_dem(bbox_input=bbox, resolution=30)
dem.plot()

# ERA5 hourly (anonymous GCS access — no credentials needed)
era5 = easysnowdata.hydroclimatology.get_era5(
    bbox_input=bbox, source="GCS",
    start_date="2023-01-01", end_date="2023-01-31"
)
era5["2m_temperature"].mean("time").plot()

# Seasonal snow classification
snow_class = easysnowdata.remote_sensing.get_seasonal_snow_classification(bbox)
snow_class.attrs["example_plot"](snow_class)
```

---

## Services requiring account setup

Some data sources require free accounts:

| Service | Env vars needed | Sign-up link |
|---------|----------------|--------------|
| Google Earth Engine | `EARTHENGINE_TOKEN` | [earthengine.google.com](https://earthengine.google.com) |
| NASA EarthData | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) |

Planetary Computer and anonymous GCS (ERA5) require no credentials.

---

## Citing

If you use easysnowdata in your research, please cite the Zenodo archive:

```
Gagliano, E. (2024). easysnowdata (Version 0.0.21) [Software].
Zenodo. https://doi.org/10.5281/zenodo.14741502
```
