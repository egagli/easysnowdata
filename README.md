# easysnowdata

[![PyPI](https://img.shields.io/pypi/v/easysnowdata.svg)](https://pypi.python.org/pypi/easysnowdata)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/easysnowdata.svg)](https://anaconda.org/conda-forge/easysnowdata)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
[![CI](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml/badge.svg)](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml)

**A Python package to easily retrieve data relevant to snow science.**

`easysnowdata` unifies access to a wide range of snow-relevant geospatial
datasets — weather stations, satellite imagery, climate reanalysis, DEMs, and
more — under a consistent API that returns xarray objects. The emphasis is on
minimising downloads and local computation by leveraging cloud-optimised data
formats wherever possible.

## Gallery

![easysnowdata](https://github.com/user-attachments/assets/5b2c83a4-b732-4c35-86fd-1bccb954c286)

## Data Source Status

<!-- DATA_STATUS_START -->
_Last updated: 2026-06-08 09:40 UTC_  
_⚠️ = skipped (credentials not available in this run)_

| Data Source | Latest (Jun 8) | Jun 5 |
| :---------- | :------: | :------: |
| SNOTEL/CCSS station list (GitHub) | ✅ | ✅ |
| SNOTEL/CCSS station CSV (GitHub) | ✅ | ✅ |
| HydroATLAS basins (figshare) | ✅ | ✅ |
| GRDC major river basins (World Bank) | ✅ | ✅ |
| GRDC WMO basins | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> |
| Köppen-Geiger classification (figshare) | ✅ | ✅ |
| Sturm & Liston snow classification (Azure) | <abbr title="RuntimeError: Unreachable: HTTPSConnectionPool(host='snowmelt.blob.core.windows.">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTPSConnectionPool(host='snowmelt.blob.core.windows.">❌</abbr> |
| Forest cover fraction (Zenodo) | ✅ | ✅ |
| Mountain snow mask (Zenodo) | ✅ | ✅ |
| ARCO-ERA5 (GCS anonymous) | ✅ | ✅ |
| Copernicus DEM (Planetary Computer) | ✅ | ✅ |
| ESA WorldCover (Planetary Computer) | ✅ | ✅ |
| HUC geometries (GEE/USGS WBD) | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> |
| SNODAS (GEE/Climate Engine) | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> |
| ERA5 (Google Earth Engine) | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> |
| CHILI (GEE/CSP ERGo) | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> |
| NLCD (GEE/USGS) | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> | <abbr title="RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_d">❌</abbr> |
| UCLA Snow Reanalysis (NASA NSIDC) | ⚠️ | ⚠️ |
<!-- DATA_STATUS_END -->

## Installation

```bash
pip install easysnowdata
```
```bash
conda install -c conda-forge easysnowdata
```
```bash
mamba install -c conda-forge easysnowdata
```

### Development install (with [pixi](https://pixi.sh))

```bash
git clone https://github.com/egagli/easysnowdata.git
cd easysnowdata
pixi install          # sets up the environment
pixi run test-fast    # run credential-free tests
pixi run test         # run all tests (requires API secrets)
pixi run docs-serve   # preview the docs locally
```

### Services that require account setup

Some data sources need free accounts and credentials passed as environment variables:

| Service | Env vars | Sign-up |
|---------|----------|---------|
| Google Earth Engine | `EARTHENGINE_TOKEN` | [earthengine.google.com](https://earthengine.google.com) |
| NASA EarthData | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | [urs.earthengine.nasa.gov](https://urs.earthdata.nasa.gov) |

Planetary Computer and anonymous GCS access require no credentials.

## Modules

| Module | What it provides |
|--------|-----------------|
| `automatic_weather_stations` | SNOTEL & CCSS station metadata + time-series data |
| `hydroclimatology` | ERA5, SNODAS, UCLA snow reanalysis, HUC boundaries, HydroATLAS, GRDC basins, Köppen-Geiger |
| `remote_sensing` | Sentinel-1, Sentinel-2, HLS, MODIS snow, ESA WorldCover, forest cover, snow classification |
| `topography` | Copernicus DEM (30 m / 90 m), CHILI topographic index |
| `utils` | Shared helpers: bbox conversion, water-year utilities, STAC config |

## Quick Start

```python
import easysnowdata

# ── Automatic weather stations ─────────────────────────────────────────────
sc = easysnowdata.automatic_weather_stations.StationCollection()
sc.get_data(stations="679_WA_SNTL", variables=["WTEQ", "SNWD"],
            start_date="2023-10-01", end_date="2024-06-30")
sc.data.plot()                        # pandas DataFrame for one station

# ── Topography ─────────────────────────────────────────────────────────────
bbox = (-121.94, 46.72, -121.54, 46.99)   # Mount Rainier, WA
dem = easysnowdata.topography.get_copernicus_dem(bbox_input=bbox, resolution=30)
dem.plot()                               # xarray DataArray

# ── Hydroclimatology ───────────────────────────────────────────────────────
era5 = easysnowdata.hydroclimatology.get_era5(
    bbox_input=bbox, source="GCS",
    start_date="2023-01-01", end_date="2023-01-31"
)
era5["2m_temperature"].mean("time").plot()

# ── Remote sensing ─────────────────────────────────────────────────────────
snow_class = easysnowdata.remote_sensing.get_seasonal_snow_classification(bbox)
snow_class.attrs["example_plot"](snow_class)
```

## Documentation

Full API reference and example notebooks: <https://egagli.github.io/easysnowdata>

## Contributing

Contributions welcome! See [CONTRIBUTING](docs/contributing.md) for guidelines.

## Citing

If you use easysnowdata in your research, please cite the Zenodo archive:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
