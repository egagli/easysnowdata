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

<!-- GALLERY_START -->
<!-- Built by scripts/make_montage.py from the thumbnails of the scheduled
     docs build and published with the site, so this picture is always the
     figures the package currently produces. Do not paste an upload here. -->
[![easysnowdata example gallery](https://egagli.github.io/easysnowdata/_static/gallery.webp)](https://egagli.github.io/easysnowdata/auto_examples/index.html)

One executed script per product — [browse them](https://egagli.github.io/easysnowdata/auto_examples/index.html),
each with the code, the figure and a downloadable notebook.
<!-- GALLERY_END -->

## Data Source Status

Every route of every product is probed weekly. A failure opens an issue
labelled [`data-source`](https://github.com/egagli/easysnowdata/issues?q=label%3Adata-source)
and a recovery closes it; latency and DMR++ readiness are on the
[status page](https://egagli.github.io/easysnowdata/status.html).

<!-- DATA_STATUS_START -->
_Last updated: 2026-09-17 21:39 UTC_  
_⚠️ = skipped (credentials not available in this run). Latency and virtualization probes are on the [status page](https://egagli.github.io/easysnowdata/status.html)._

| Data Source | Latest (Sep 17) | Sep 14 | Sep 7 | Aug 31 |
| :---------- | :------: | :------: | :------: | :------: |
| AWDB stations (NRCS REST API) | ✅ | — | — | — |
| CDEC stations (JSON data servlet) | ✅ | — | — | — |
| BC snow stations (DataBC WFS) | ✅ | — | — | — |
| NVE stations (HydAPI) | ⚠️ | — | — | — |
| Yukon stations (AquaCache API) | ✅ | — | — | — |
| SNOTEL/CCSS station list (GitHub) | ✅ | ✅ | ✅ | ✅ |
| Snow station archive tarball (global_snow_networks) | ✅ | — | — | — |
| SNOTEL/CCSS station CSV (GitHub) | ✅ | ✅ | ✅ | ✅ |
| ARCO-ERA5 (GCS anonymous) | ✅ | ✅ | ✅ | ✅ |
| ERA5 (Google Earth Engine) | ✅ | ✅ | ✅ | ✅ |
| Köppen-Geiger classification (figshare) | ✅ | ✅ | ✅ | ✅ |
| HUC geometries (USGS WBD REST) | ✅ | — | — | — |
| HUC geometries (GEE/USGS WBD) | ✅ | ✅ | ✅ | ✅ |
| HydroATLAS basins (figshare) | ✅ | ✅ | ✅ | ✅ |
| HydroBASINS (HydroSHEDS regional zip) | ✅ | — | — | — |
| HydroBASINS (GEE/HydroATLAS) | ✅ | — | — | — |
| GRDC major river basins (World Bank) | ✅ | ✅ | ✅ | ✅ |
| GRDC WMO basins | <abbr title="ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection r">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> |
| MODIS snow cover MOD10A1F (NASA NSIDC) | ✅ | — | — | — |
| MODIS snow cover MOD10A1 (Planetary Computer) | ✅ | — | — | — |
| Mountain snow mask (Zenodo) | ✅ | ✅ | ✅ | ✅ |
| SNODAS (NSIDC G02158) | ✅ | — | — | — |
| SNODAS (GEE/Climate Engine) | ✅ | ✅ | ✅ | ✅ |
| Sturm & Liston snow classification (NSIDC-0768) | ✅ | — | — | — |
| Sturm & Liston snow classification (Azure) | ✅ | ✅ | ✅ | ✅ |
| UCLA Snow Reanalysis (NASA NSIDC) | ✅ | ⚠️ | ⚠️ | ⚠️ |
| HMA Snow Reanalysis (NASA NSIDC) | ✅ | — | — | — |
| VIIRS snow cover VNP10A1F (NASA NSIDC) | ✅ | — | — | — |
| Forest cover fraction (Zenodo) | ✅ | ✅ | ✅ | ✅ |
| Forest cover fraction (GEE/CGLS-LC100) | ✅ | — | — | — |
| ESA WorldCover (Planetary Computer) | ✅ | ✅ | ✅ | ✅ |
| ESA WorldCover (AWS bucket) | ✅ | — | — | — |
| Annual NLCD (GEE community asset) | ✅ | — | — | — |
| NLCD (GEE/USGS) | ✅ | ✅ | ✅ | ✅ |
| HLS L30 (CMR-STAC LPCLOUD) | ✅ | — | — | — |
| HLS S30 (Planetary Computer) | ✅ | — | — | — |
| PlanetScope (Planet Data API) | ⚠️ | — | — | — |
| Sentinel-2 L2A (Planetary Computer) | ✅ | — | — | — |
| Sentinel-2 L2A (Earth Search) | ✅ | — | — | — |
| Sentinel-1 RTC (Planetary Computer) | ✅ | — | — | — |
| Sentinel-1 RTC OPERA (CMR-STAC ASF) | ✅ | — | — | — |
| Sentinel-1 RTC OPERA (Earth Engine) | ✅ | — | — | — |
| Sentinel-1 static layers (CMR-STAC ASF) | ✅ | — | — | — |
| Copernicus DEM for the incidence angle (Planetary Computer) | ✅ | — | — | — |
| Sentinel-1 GRD angle band (Earth Engine) | ✅ | — | — | — |
| CHILI (GEE/CSP ERGo) | ✅ | ✅ | ✅ | ✅ |
| Copernicus DEM (Planetary Computer) | ✅ | ✅ | ✅ | ✅ |
| Copernicus DEM (Earth Search) | ✅ | — | — | — |
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
pixi run test-unit    # offline tests (no network, no credentials)
pixi run test-live    # live tests against the data providers (credentialed ones skip without secrets)
pixi run docs-serve   # preview the docs locally
```

### Services that require account setup

Some data sources need free accounts and credentials passed as environment variables:

| Service | Env vars | Sign-up |
|---------|----------|---------|
| Google Earth Engine | `EARTHENGINE_TOKEN` (or `~/.config/earthengine/credentials` from `ee.Authenticate()`) | [earthengine.google.com](https://earthengine.google.com) |
| NASA Earthdata | `EARTHDATA_TOKEN` (recommended), or `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`, or a `~/.netrc` entry from `earthaccess.login(persist=True)` | [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) |

Planetary Computer and anonymous GCS access require no credentials.

## What is in it

<!-- CATALOG_START -->
28 products across 8 themes, each with one or more access routes:

| theme | products | open without an account |
| --- | --- | --- |
| **climate** | [`era5`](https://egagli.github.io/easysnowdata/catalog/era5.html), [`koppen-geiger`](https://egagli.github.io/easysnowdata/catalog/koppen-geiger.html) | 2 of 2 |
| **hydro** | [`grdc-major-river-basins`](https://egagli.github.io/easysnowdata/catalog/grdc-major-river-basins.html), [`grdc-wmo-basins`](https://egagli.github.io/easysnowdata/catalog/grdc-wmo-basins.html), [`huc`](https://egagli.github.io/easysnowdata/catalog/huc.html), [`hydrobasins`](https://egagli.github.io/easysnowdata/catalog/hydrobasins.html) | 4 of 4 |
| **land** | [`esa-worldcover`](https://egagli.github.io/easysnowdata/catalog/esa-worldcover.html), [`forest-cover-fraction`](https://egagli.github.io/easysnowdata/catalog/forest-cover-fraction.html), [`nlcd`](https://egagli.github.io/easysnowdata/catalog/nlcd.html) | 2 of 3 |
| **optical** | [`hls`](https://egagli.github.io/easysnowdata/catalog/hls.html), [`planetscope`](https://egagli.github.io/easysnowdata/catalog/planetscope.html), [`sentinel-2-l2a`](https://egagli.github.io/easysnowdata/catalog/sentinel-2-l2a.html) | 2 of 3 |
| **sar** | [`sentinel-1-local-incidence-angle`](https://egagli.github.io/easysnowdata/catalog/sentinel-1-local-incidence-angle.html), [`sentinel-1-rtc`](https://egagli.github.io/easysnowdata/catalog/sentinel-1-rtc.html) | 2 of 2 |
| **snow** | [`modis-snow`](https://egagli.github.io/easysnowdata/catalog/modis-snow.html), [`mountain-snow-mask`](https://egagli.github.io/easysnowdata/catalog/mountain-snow-mask.html), [`snodas`](https://egagli.github.io/easysnowdata/catalog/snodas.html), [`snow-classification`](https://egagli.github.io/easysnowdata/catalog/snow-classification.html), [`ucla-snow-reanalysis`](https://egagli.github.io/easysnowdata/catalog/ucla-snow-reanalysis.html), [`viirs-snow`](https://egagli.github.io/easysnowdata/catalog/viirs-snow.html) | 4 of 6 |
| **stations** | [`awdb-stations`](https://egagli.github.io/easysnowdata/catalog/awdb-stations.html), [`cdec-stations`](https://egagli.github.io/easysnowdata/catalog/cdec-stations.html), [`databc-stations`](https://egagli.github.io/easysnowdata/catalog/databc-stations.html), [`nve-stations`](https://egagli.github.io/easysnowdata/catalog/nve-stations.html), [`snow-station-archive`](https://egagli.github.io/easysnowdata/catalog/snow-station-archive.html), [`yukon-stations`](https://egagli.github.io/easysnowdata/catalog/yukon-stations.html) | 5 of 6 |
| **terrain** | [`chili`](https://egagli.github.io/easysnowdata/catalog/chili.html), [`copernicus-dem`](https://egagli.github.io/easysnowdata/catalog/copernicus-dem.html) | 1 of 2 |
<!-- CATALOG_END -->

Every product's routes, resolution, credentials, licence and health are on its
own page: <https://egagli.github.io/easysnowdata/catalog/>.

## Quick Start

```python
import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)          # Mount Rainier; any AOI form works

# Snow stations: which are here, then one water year of observations
inv = esd.stations.inventory(aoi, daily_only=True)
obs = esd.stations.load(inv, variables=["swe", "snwd"], time="2023-10/2024-09")

# Terrain, SAR and snow water equivalent — lazy, Dask-backed, CRS attached
dem = esd.terrain.dem.load(aoi)                             # Copernicus GLO-30
s1 = esd.sar.sentinel1.load(aoi, "2024-03", units="dB")     # Sentinel-1 RTC
swe = esd.snow.snodas.load(aoi, "2024-03")                  # SNODAS, no account

# Optical, masked and turned into a snow index
s2 = esd.optical.sentinel2.load(aoi, "2024-03", mask="scl-default")
ndsi = esd.processing.ndsi(s2)

# Categorical products carry CF flag attrs, so the legend draws itself
esd.plotting.categorical(esd.land.landcover.load(aoi))

# What is available, and what it needs
esd.catalog.search("swe")
esd.catalog.describe("snodas")
esd.auth.status()
```

The pre-0.1 API (`easysnowdata.remote_sensing.get_*`,
`automatic_weather_stations.StationCollection`, …) still works and emits a
`DeprecationWarning` naming its replacement. It is removed one minor release
after 0.1.

## Documentation

Full API reference and example notebooks: <https://egagli.github.io/easysnowdata>

## Contributing

Contributions welcome! See [CONTRIBUTING](docs/contributing.md) for guidelines.

## Citing

If you use easysnowdata in your research, please cite the Zenodo archive:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
