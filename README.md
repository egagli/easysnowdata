# easysnowdata

[![PyPI](https://img.shields.io/pypi/v/easysnowdata.svg)](https://pypi.python.org/pypi/easysnowdata)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/easysnowdata.svg)](https://anaconda.org/conda-forge/easysnowdata)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
[![CI](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml/badge.svg)](https://github.com/egagli/easysnowdata/actions/workflows/ci.yml)

**A Python package to easily retrieve data relevant to snow science.**

`easysnowdata` unifies access to a wide range of snow-relevant geospatial
datasets — weather stations, satellite imagery, snow products, climate
reanalysis, DEMs and a hillshade, basins, and boundaries from countries and
counties to mountain ranges and glacier outlines — under a consistent API that
returns xarray objects (and GeoDataFrames for vector products). The emphasis is
on minimising downloads and local computation by leveraging cloud-optimised
data formats wherever possible.

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

What a probe cannot see — a reprocessed collection, a republished file, a new
release (the next Census year, a new RGI version) — is caught by a weekly
[upstream watch](WATCHLIST.toml) that opens one digest issue labelled
[`upstream-watch`](https://github.com/egagli/easysnowdata/issues?q=label%3Aupstream-watch).

<!-- DATA_STATUS_START -->
_Last updated: 2026-09-23 22:33 UTC_  
_⚠️ = skipped (credentials not available in this run). Latency and virtualization probes are on the [status page](https://egagli.github.io/easysnowdata/status.html)._

### Stations (`esd.stations`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| AWDB stations (NRCS REST API) | ✅ | ✅ | ✅ | — |
| CDEC stations (JSON data servlet) | ✅ | ✅ | ✅ | — |
| BC snow stations (DataBC WFS) | ✅ | ✅ | ✅ | — |
| NVE stations (HydAPI) | ✅ | <abbr title="RuntimeError: Unreachable: HTTP 401">❌</abbr> | ⚠️ | — |
| Yukon stations (AquaCache API) | ✅ | ✅ | ✅ | — |
| Snow station Zarr archive (global_snow_networks) | ✅ | — | — | — |
| Snow station inventory (global_snow_networks) | ✅ | ✅ | ✅ | ✅ |
| Snow station archive tarball (global_snow_networks) | ✅ | ✅ | ✅ | — |
| Snow station CSV (global_snow_networks) | ✅ | ✅ | ✅ | ✅ |

### Snow (`esd.snow`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| MODIS snow cover MOD10A1F (NASA NSIDC) | ✅ | ✅ | ✅ | — |
| MODIS snow cover MOD10A1 (Planetary Computer) | ✅ | ✅ | ✅ | — |
| Mountain snow mask (Zenodo) | ✅ | ✅ | ✅ | ✅ |
| SNODAS (NSIDC G02158) | ✅ | ✅ | ✅ | — |
| SNODAS (GEE/Climate Engine) | ✅ | ✅ | ✅ | ✅ |
| Sturm & Liston snow classification (NSIDC-0768) | ✅ | ✅ | ✅ | — |
| Sturm & Liston snow classification (Azure) | ✅ | ✅ | ✅ | ✅ |
| UCLA Snow Reanalysis (NASA NSIDC) | <abbr title="RuntimeError: WUS_UCLA_SR: no granules found.">❌</abbr> | ✅ | ✅ | ⚠️ |
| HMA Snow Reanalysis (NASA NSIDC) | <abbr title="RuntimeError: HMA_SR_D: no granules found.">❌</abbr> | ✅ | ✅ | — |
| VIIRS snow cover VNP10A1F (NASA NSIDC) | ✅ | ✅ | ✅ | — |

### SAR (`esd.sar`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| Sentinel-1 RTC (Planetary Computer) | ✅ | ✅ | ✅ | — |
| Sentinel-1 RTC OPERA (CMR-STAC ASF) | <abbr title="APIError: {&quot;errors&quot;:[&quot;Oops! Something has gone wrong. We have been alerted and a">❌</abbr> | ✅ | ✅ | — |
| Sentinel-1 RTC OPERA (Earth Engine) | ✅ | ✅ | ✅ | — |
| Sentinel-1 static layers (CMR-STAC ASF) | <abbr title="APIError: {&quot;errors&quot;:[&quot;Oops! Something has gone wrong. We have been alerted and a">❌</abbr> | ✅ | ✅ | — |
| Copernicus DEM for the incidence angle (Planetary Computer) | ✅ | ✅ | ✅ | — |
| Sentinel-1 GRD angle band (Earth Engine) | ✅ | ✅ | ✅ | — |

### Optical imagery (`esd.optical`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| HLS L30 (CMR-STAC LPCLOUD) | <abbr title="APIError: {&quot;errors&quot;:[&quot;Oops! Something has gone wrong. We have been alerted and a">❌</abbr> | ✅ | ✅ | — |
| HLS S30 (Planetary Computer) | ✅ | ✅ | ✅ | — |
| PlanetScope (Planet Data API) | ✅ | <abbr title="DeprecationWarning: Auth.value has been deprecated.">❌</abbr> | ⚠️ | — |
| Sentinel-2 L2A (Planetary Computer) | ✅ | ✅ | ✅ | — |
| Sentinel-2 L2A (Earth Search) | ✅ | ✅ | ✅ | — |

### Terrain (`esd.terrain`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| CHILI (GEE/CSP ERGo) | ✅ | ✅ | ✅ | ✅ |
| Copernicus DEM (Planetary Computer) | ✅ | ✅ | ✅ | ✅ |
| Copernicus DEM (Earth Search) | ✅ | ✅ | ✅ | — |
| Copernicus DEM (Earth Engine) | ✅ | — | — | — |
| NASADEM (Planetary Computer) | ✅ | — | — | — |
| NASADEM (Earth Engine) | ✅ | — | — | — |
| SRTM GL1 (Earth Engine) | ✅ | — | — | — |
| 3DEP seamless (Planetary Computer) | ✅ | — | — | — |
| 3DEP 10 m (Earth Engine) | ✅ | — | — | — |
| ALOS World 3D (Planetary Computer) | ✅ | — | — | — |
| ALOS World 3D (Earth Engine) | ✅ | — | — | — |
| Natural Earth hillshade (S3) | ✅ | — | — | — |

### Land cover (`esd.land`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| Forest cover fraction (Zenodo) | ✅ | ✅ | ✅ | ✅ |
| Forest cover fraction (GEE/CGLS-LC100) | ✅ | ✅ | ✅ | — |
| ESA WorldCover (Planetary Computer) | ✅ | ✅ | ✅ | ✅ |
| ESA WorldCover (AWS bucket) | ✅ | ✅ | ✅ | — |
| Annual NLCD (GEE community asset) | ✅ | ✅ | ✅ | — |
| NLCD (GEE/USGS) | ✅ | ✅ | ✅ | ✅ |

### Hydrography (`esd.hydro`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| HUC geometries (USGS WBD REST) | ✅ | ✅ | ✅ | — |
| HUC geometries (GEE/USGS WBD) | ✅ | ✅ | ✅ | ✅ |
| HydroATLAS basins (figshare) | ✅ | ✅ | ✅ | ✅ |
| HydroBASINS (HydroSHEDS regional zip) | ✅ | <abbr title="RuntimeError: Unreachable: HTTP 403">❌</abbr> | ✅ | — |
| HydroBASINS (GEE/HydroATLAS) | ✅ | ✅ | ✅ | — |
| GRDC major river basins (World Bank) | ✅ | ✅ | ✅ | ✅ |
| GRDC WMO basins | ✅ | ✅ | <abbr title="ConnectionError: (&#x27;Connection aborted.&#x27;, ConnectionResetError(104, &#x27;Connection r">❌</abbr> | <abbr title="RuntimeError: Unreachable: HTTP 404">❌</abbr> |

### Boundaries (`esd.boundaries`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| Natural Earth countries (naciscdn) | ✅ | — | — | — |
| geoBoundaries countries (ADM0, gbOpen) | ✅ | — | — | — |
| Natural Earth states and provinces (naciscdn) | ✅ | — | — | — |
| US Census states (cartographic boundaries) | ✅ | — | — | — |
| geoBoundaries states and provinces (ADM1, gbOpen) | ✅ | — | — | — |
| US Census counties (cartographic boundaries) | ✅ | — | — | — |
| geoBoundaries admin units (ADM2, gbOpen) | ✅ | — | — | — |
| RGI 7.0 glacier outlines (NSIDC) | ✅ | — | — | — |
| RGI 6.0 glacier outlines (NSIDC) | ✅ | — | — | — |
| RGI 6.0 glacier outlines (OGGM mirror) | ✅ | — | — | — |
| GMBA mountains (EarthEnv) | ✅ | — | — | — |
| Natural Earth vector layers (naciscdn) | ✅ | — | — | — |

### Climate (`esd.climate`)

| Data Source | Latest (Sep 23) | Sep 21 | Sep 17 | Sep 14 |
| :---------- | :------: | :------: | :------: | :------: |
| ARCO-ERA5 (GCS anonymous) | ✅ | ✅ | ✅ | ✅ |
| ERA5 (Google Earth Engine) | ✅ | ✅ | ✅ | ✅ |
| Köppen-Geiger classification (figshare) | ✅ | ✅ | ✅ | ✅ |
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
pixi install                          # sets up the environments
pixi run -e test-py313 test-unit      # offline tests (no network, no credentials)
pixi run -e test-py313 test-live      # live tests against the data providers (credentialed ones skip without secrets)
pixi run -e docs docs-serve           # preview the docs locally
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
40 products across 9 themes, each with one or more access routes:

| theme | products | open without an account |
| --- | --- | --- |
| **stations** | [`awdb-stations`](https://egagli.github.io/easysnowdata/catalog/awdb-stations.html), [`cdec-stations`](https://egagli.github.io/easysnowdata/catalog/cdec-stations.html), [`databc-stations`](https://egagli.github.io/easysnowdata/catalog/databc-stations.html), [`nve-stations`](https://egagli.github.io/easysnowdata/catalog/nve-stations.html), [`snow-station-archive`](https://egagli.github.io/easysnowdata/catalog/snow-station-archive.html), [`yukon-stations`](https://egagli.github.io/easysnowdata/catalog/yukon-stations.html) | 5 of 6 |
| **snow** | [`modis-snow`](https://egagli.github.io/easysnowdata/catalog/modis-snow.html), [`mountain-snow-mask`](https://egagli.github.io/easysnowdata/catalog/mountain-snow-mask.html), [`snodas`](https://egagli.github.io/easysnowdata/catalog/snodas.html), [`snow-classification`](https://egagli.github.io/easysnowdata/catalog/snow-classification.html), [`ucla-snow-reanalysis`](https://egagli.github.io/easysnowdata/catalog/ucla-snow-reanalysis.html), [`viirs-snow`](https://egagli.github.io/easysnowdata/catalog/viirs-snow.html) | 4 of 6 |
| **sar** | [`sentinel-1-local-incidence-angle`](https://egagli.github.io/easysnowdata/catalog/sentinel-1-local-incidence-angle.html), [`sentinel-1-rtc`](https://egagli.github.io/easysnowdata/catalog/sentinel-1-rtc.html) | 2 of 2 |
| **optical** | [`hls`](https://egagli.github.io/easysnowdata/catalog/hls.html), [`planetscope`](https://egagli.github.io/easysnowdata/catalog/planetscope.html), [`sentinel-2-l2a`](https://egagli.github.io/easysnowdata/catalog/sentinel-2-l2a.html) | 2 of 3 |
| **terrain** | [`3dep`](https://egagli.github.io/easysnowdata/catalog/3dep.html), [`alos-dem`](https://egagli.github.io/easysnowdata/catalog/alos-dem.html), [`chili`](https://egagli.github.io/easysnowdata/catalog/chili.html), [`copernicus-dem`](https://egagli.github.io/easysnowdata/catalog/copernicus-dem.html), [`hillshade`](https://egagli.github.io/easysnowdata/catalog/hillshade.html), [`nasadem`](https://egagli.github.io/easysnowdata/catalog/nasadem.html), [`srtm`](https://egagli.github.io/easysnowdata/catalog/srtm.html) | 5 of 7 |
| **land** | [`esa-worldcover`](https://egagli.github.io/easysnowdata/catalog/esa-worldcover.html), [`forest-cover-fraction`](https://egagli.github.io/easysnowdata/catalog/forest-cover-fraction.html), [`nlcd`](https://egagli.github.io/easysnowdata/catalog/nlcd.html) | 2 of 3 |
| **hydro** | [`grdc-major-river-basins`](https://egagli.github.io/easysnowdata/catalog/grdc-major-river-basins.html), [`grdc-wmo-basins`](https://egagli.github.io/easysnowdata/catalog/grdc-wmo-basins.html), [`huc`](https://egagli.github.io/easysnowdata/catalog/huc.html), [`hydrobasins`](https://egagli.github.io/easysnowdata/catalog/hydrobasins.html) | 4 of 4 |
| **boundaries** | [`admin-boundaries`](https://egagli.github.io/easysnowdata/catalog/admin-boundaries.html), [`countries`](https://egagli.github.io/easysnowdata/catalog/countries.html), [`gmba-mountains`](https://egagli.github.io/easysnowdata/catalog/gmba-mountains.html), [`natural-earth-vectors`](https://egagli.github.io/easysnowdata/catalog/natural-earth-vectors.html), [`rgi-glaciers`](https://egagli.github.io/easysnowdata/catalog/rgi-glaciers.html), [`states-provinces`](https://egagli.github.io/easysnowdata/catalog/states-provinces.html), [`us-counties`](https://egagli.github.io/easysnowdata/catalog/us-counties.html) | 7 of 7 |
| **climate** | [`era5`](https://egagli.github.io/easysnowdata/catalog/era5.html), [`koppen-geiger`](https://egagli.github.io/easysnowdata/catalog/koppen-geiger.html) | 2 of 2 |
<!-- CATALOG_END -->

Every product's routes, resolution, credentials, licence and health are on its
own page: <https://egagli.github.io/easysnowdata/catalog/>.

## Quick Start

```python
import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)          # Mount Rainier; any AOI form works

# Snow stations: which are here, then one water year of observations
inv_gdf = esd.stations.inventory(aoi, daily_only=True)
obs_ds = esd.stations.load(inv_gdf, variables=["swe", "snwd"], time="2023-10/2024-09")

# Terrain, SAR and snow water equivalent — lazy, Dask-backed, CRS attached
dem_da = esd.terrain.dem.load(aoi)                          # Copernicus GLO-30 (default)
dem_da = esd.terrain.dem.load(aoi, product="3dep")          # or NASADEM, SRTM, 3DEP, ALOS
s1_ds = esd.sar.sentinel1.load(aoi, "2024-03", units="dB")  # Sentinel-1 RTC
snodas_ds = esd.snow.snodas.load(aoi, "2024-03")            # SNODAS, no account

# Boundaries as GeoDataFrames: states, counties, mountain ranges, glaciers
wa_gdf = esd.boundaries.admin.states(aoi)                   # Washington (US Census)
ranges_gdf = esd.boundaries.mountains.load(aoi)             # GMBA Mountain Inventory v2
glaciers_gdf = esd.boundaries.glaciers.load(aoi)            # RGI 7.0; version="6.0" too

# Optical, masked, and a snow index written out rather than hidden in a helper
s2_ds = esd.optical.sentinel2.load(aoi, "2024-03", mask="scl-default")
ndsi_da = (s2_ds["green"] - s2_ds["swir16"]) / (s2_ds["green"] + s2_ds["swir16"])

# Maps with equal aspect, a scale bar and a graticule; legends from CF flags
ax = esd.plotting.map(dem_da, cmap="terrain")
esd.plotting.add_outline(ax, glaciers_gdf)                  # vectors in the map's CRS
esd.plotting.categorical(esd.land.landcover.load(aoi))
esd.plotting.timeseries(obs_ds["swe"])                      # calendar dates, units in [ ]

# What is available, and what it needs
esd.catalog.search("swe")
esd.catalog.describe("snodas")
esd.auth.status()
```

Coming from 0.0.x? The old module API (`remote_sensing.get_*`,
`automatic_weather_stations.StationCollection`, …) was deprecated in 0.2 and
removed in 0.3; the [migration guide](https://egagli.github.io/easysnowdata/migration.html)
maps every old name to its replacement.

## Documentation

Executed example gallery, data catalog and API reference: <https://egagli.github.io/easysnowdata>

## Contributing

Contributions welcome! See [CONTRIBUTING](docs/contributing.md) for guidelines.

## Citing

If you use easysnowdata in your research, please cite the Zenodo archive:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14741502.svg)](https://doi.org/10.5281/zenodo.14741502)
