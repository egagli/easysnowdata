# Data products, sources, access patterns, and examples

_Companion to [`REVAMP_PLAN.md`](REVAMP_PLAN.md). Started 2026-09-15 from the existing package,
the open issues (#1, #3, #5, #8, #9, #10, #11), and a live audit of endpoints that day. This
file is meant to be kept current: when a product is added to the catalog, its row here gets a
link; when a source changes, the comparison here is where the reasoning lives. Access facts are
dated; a row without a date was not checked live. Nothing collected so far is dropped — an idea
that is not a data product (an access pattern, a tool, a reference) still gets a row in §E with a
note on whether and how it should be incorporated._

_2026-09-23: added §F (plans for boundaries, cloud products and cameras), the Natural Earth
hillshade that now ships as `esd.terrain.hillshade`, and their rows in §A, §C, §D and §E._

Contents:

- **A. Products in the rewrite** — what the new package ships, in three tiers
- **B. Products with more than one source** — the differences, and which one is the default
- **C. Evaluation of every product idea collected so far** — why each matters, how hard it is, verdict
- **D. Example gallery** — the one-script-per-product examples and the multi-source how-tos
- **E. Link register** — every link from the issues and idea dumps, what it is, why it matters, incorporate?
- **F. Plans for new modules** — boundaries (F.1), cloud products (F.2), cameras and station imagery (F.3)

---

## A. Products in the rewrite

Legend: 🟢 exists today and is migrated · 🆕 new in the rewrite · ⏭ cheap follow-on once the
`providers` layer exists (a catalog entry and a few lines) · creds: none / EDL (Earthdata
Login) / GEE (Earth Engine) / key.

### Tier 1 — ships with 0.1 (existing products modernized, plus Eric's three priorities and the Sentinel-1 ask)

| Theme | Product | Default source (creds) | Other sources | Notes |
| --- | --- | --- | --- | --- |
| sar | 🟢 Sentinel-1 RTC backscatter | Planetary Computer `sentinel-1-rtc` (none) | OPERA RTC-S1 via CMR-STAC/ASF (EDL); GEE `OPERA/RTC/L2_V1/S1` (GEE) | see B.1 |
| sar | 🆕 Sentinel-1 local incidence angle + layover/shadow mask | OPERA RTC-S1-STATIC (EDL) | computed from any DEM (none); GEE `S1_GRD` angle + DEM (GEE, legacy) | see B.1 |
| optical | 🟢 Sentinel-2 L2A | Planetary Computer `sentinel-2-l2a` (none) | Earth Search `sentinel-2-l2a` / `-c1-l2a` (none); CDSE/EOPF Zarr (CDSE creds, experimental) | see B.2 |
| optical | 🟢 HLS L30/S30 v2.0 — **priority 3** | CMR-STAC LPCLOUD `HLSL30_2.0`/`HLSS30_2.0` (EDL) | Planetary Computer `hls2-l30`/`hls2-s30` (none, ids unverified) | see B.3 |
| snow | 🟢 MODIS snow cover MOD10A1 / MOD10A2 / MOD10A1F (+ MYD) | NSIDC via `earthaccess` (EDL) | Planetary Computer `modis-10A*-061` (none, archiving reportedly stopped 2025); GEE `MODIS/061/MOD10A1` (GEE) | see B.4 |
| snow | 🆕 VIIRS snow cover VNP10A1 / VNP10A1F — **priority 1** | NSIDC via `earthaccess` (EDL) | GEE `NASA/VIIRS/002/VNP10A1F` (GEE, unverified) | successor to MODIS after Terra's end; 375 m |
| snow | 🟢 SNODAS SWE / snow depth — **priority 2** | NSIDC G02158 daily tarballs (none) 🆕 | GEE `projects/climate-engine/snodas/daily` (GEE) | see B.5 |
| snow | 🟢 UCLA Western US snow reanalysis | NSIDC `WUS_UCLA_SR` v1 (EDL) | sibling 🆕 `HMA_SR_D` v1 (EDL) | `virtualize="auto"` option for long series (see D); fix `stats=` index bug; explicit login |
| snow | 🟢 Sturm & Liston seasonal snow classification | NSIDC-0768 (EDL) | hosted COG on the `uwcryo` blob (none; **location likely to change**) | see B.6 |
| snow | 🟢 Wrzesien mountain snow mask (+ clouds layer) | Zenodo 2626737 zip (none) | — | `pooch` cache |
| land | 🟢 ESA WorldCover v100/v200 | Planetary Computer `esa-worldcover` (none) | AWS `esa-worldcover` bucket (none) | see B.7 |
| land | 🟢 NLCD | GEE `USGS/NLCD_RELEASES/2021_REL/NLCD` (GEE) | 🆕 Annual NLCD community asset (GEE) | see B.7 |
| land | 🟢 Forest cover fraction (CGLS-LC100 2019) | Zenodo 3939050 GeoTIFF (none) | GEE `COPERNICUS/Landcover/100m/Proba-V-C3/Global` (GEE) | no newer epoch exists |
| terrain | 🟢 Copernicus DEM GLO-30/90 | Planetary Computer `cop-dem-glo-*` (none) | AWS `copernicus-dem-30m/90m` (none); Earth Search `cop-dem-glo-*` (none); GEE `COPERNICUS/DEM/GLO30_2024_1` (GEE) | see B.8 |
| terrain | 🟢 CHILI | GEE `CSP/ERGo/1_0/Global/ALOS_CHILI` (GEE) | computed heat-load index from a DEM (none, later) | stop AOI-relative rescaling by default |
| terrain | 🆕 Hillshade basemap (Natural Earth shaded relief) | Natural Earth S3 bucket, one zip per style and scale (none) | — | **shipped 2026-09-23** as `esd.terrain.hillshade.load`: five styles (plain shaded relief to Gray Earth with ocean bottom and drainages), 1:10m (1′) or 1:50m (2′), EPSG:4326 uint8, cached once, no reprojection; the layer from the runoff-onset `download_and_preprocess_hillshade.ipynb` |
| climate | 🟢 ERA5 hourly | ARCO-ERA5 on GCS (none) | Earthmover Icechunk ERA5 (none); WeatherBench2 (none); NCAR `nsf-ncar-era5` (none); GEE (GEE) | see B.9 |
| climate | 🟢 ERA5 / ERA5-Land daily & monthly | GEE `ECMWF/ERA5*` (GEE) | CDS ARCO Zarr data lake (CDS token, beta); DestinE EDH (token) | see B.9 |
| climate | 🟢 Köppen-Geiger (Beck 2023) | figshare file 61012822 (none) | — | switch from stale v1 file; expose `period=` |
| hydro | 🟢 HUC boundaries | USGS WBD ArcGIS REST / HyRiver `pynhd` (none) 🆕 default | GEE `USGS/WBD/2017/HUC*` (GEE) | see B.10 |
| hydro | 🟢 HydroBASINS / BasinATLAS | figshare BasinATLAS gdb (none) | HydroSHEDS per-region zips (none); GEE `WWF/HydroATLAS/v1/Basins/level*` (GEE) | see B.10 |
| hydro | 🟢 GRDC major river basins; GRDC/WMO basins | World Bank zip; `grdc.bafg.de` zip (none) | — | GET-first fetch (server rejects HEAD) |
| stations | 🟢→🆕 Station SWE / snow depth / met | `global_snow_networks` clients: NRCS AWDB REST, CDEC, BC DataBC, NVE HydAPI (key), Yukon AquaCache (none/key) | pre-downloaded daily archive (none) | replaces the frozen `snotel_ccss_stations` CSVs |

### Tier 2 — cheap follow-ons (catalog entry + provider call; each a small PR)

| Theme | Product | Source (creds) | Why |
| --- | --- | --- | --- |
| optical | Landsat Collection 2 Level-2 | Earth Search / PC `landsat-c2-l2` (none); USGS `landsatlook` STAC (requester-pays) | pre-2015 snow cover; the USGS catalog also has `landsat-c2l3-fsca` fractional snow cover |
| optical | VIIRS daily surface reflectance VNP09GA | GEE `NASA/VIIRS/002/VNP09GA` (GEE); LPCLOUD (EDL) | the reflectance behind VIIRS snow; issue #11 |
| optical | 🆕 PlanetScope PSScene 3 m (4/8-band, SR, UDM2) and SkySat 50 cm | Planet Data API search + Orders API clip via the `planet` SDK 3.x (Planet account: OAuth2 session or API key; E&R quota) | Eric's ask (plan §12 Q17): near-daily 3 m snow-covered area and snow-disappearance dates at hillslope scale, validation chips for MODIS/VIIRS/S2 products; the UDM2 mask has a snow band; a rough `PlanetData` class already exists in `docs/examples/sandbox.ipynb`; see B.11 |
| snow | ASO lidar snow depth / SWE 50 m | NSIDC `ASO_50M_SD`, `ASO_50M_SWE` (EDL, cloud) | the validation dataset for any SWE product; also the subject of P21 |
| snow | AMSR daily SWE | NSIDC `AU_DySno` (EDL, cloud) | only passive-microwave SWE in the set; coarse but global and daily |
| snow | ICESat-2 ATL06 / ATL08 | NSIDC v007 (EDL, cloud) | snow depth by differencing; `icepyx`/`sliderule` exist — wrap, don't rewrite |
| snow | Sentinel-2 / HLS snow cover (issue #9) | computed in `processing` | NDSI + SCL/Fmask first; let-it-snow / Theia style later (links in E.3) |
| terrain | 3DEP 1/3″ seamless; 3DEP lidar | `s3://prd-tnm` (none); PC `3dep-seamless` / `3dep-lidar` group; GEE `3dep` tag | US 10 m DEM; `py3dep` already a (currently unused) dependency |
| terrain | NASADEM | LPCLOUD `NASADEM_HGT_001` (EDL); PC `nasadem`; GEE `NASA/NASADEM_HGT/001` | 30 m, void-filled SRTM heritage |
| terrain | slope / aspect / hillshade / heat-load index | computed from any DEM | needed by the DEM-based incidence-angle fallback; McCune & Keon heat load (E.1) replaces GEE CHILI; `processing.sar.slope_aspect` already exists, and a DEM-resolution `hillshade(dem_da)` beside it would complement the 1–2 km Natural Earth basemap now in `terrain.hillshade` |
| land | Dynamic World; Hansen GFC 2025 v1.13 | GEE (GEE) | annual/near-real-time land cover; forest loss year |
| climate | Daymet V4R1 | ORNL `Daymet_Daily_V4R1` (EDL, cloud, **has DMR++ sidecars**); PC `daymet-daily-na` Zarr (none) | 1 km daily North America met, the standard forcing for snow models |
| climate | gridMET; PRISM; NLDAS-2 | GEE `IDAHO_EPSCOR/GRIDMET`, `OREGONSTATE/PRISM/AN81d`, `NASA/NLDAS/FORA0125_H002` (GEE); PC `gridmet` Zarr; PRISM web service (none) | CONUS met forcings and normals |
| climate | NLDAS-3 (beta); CONUS404 | `s3://nasa-waterinsight` kerchunk/Icechunk (none); USGS HyTEST Zarr on OSN (none) | already virtualized/Zarr — source entries only |
| climate | GPM IMERG daily | GES DISC `GPM_3IMERGDF` v07 (EDL, cloud, **has DMR++**) | global precipitation where no gauge exists |
| climate | NOAA PSL climate indices | text files (none) | ENSO/PDO/AO context for interannual snow variability; trivial to add as a small table loader |
| stations | Canada MSC GeoMet; Swiss SLF IMIS; Synoptic/MesoWest | OGC API / REST (none / free token) | networks already scoped in `global_snow_networks` issues #2–#22 |
| boundaries | 🆕 Country, state/province, county and admin-level boundaries; mountain ranges | Natural Earth (none), US Census cartographic boundaries (none), geoBoundaries API (none), GMBA Mountain Inventory v2 on EarthEnv (none) | a new `esd.boundaries` theme; map outlines, AOIs by name, per-state and per-range summaries; see **F.1** |
| climate | 🆕 Cloud frequency climatology (EarthEnv, Wilson & Jetz 2016) | `data.earthenv.org/cloud/MODCF_*.tif` (none; CC BY-NC 4.0) | 1 km MODIS cloud frequency 2000–2014: mean annual, monthly, inter-/intra-annual variability, seasonality; where optical snow products go blind; see **F.2** |
| climate | Cloud cover from ERA5 | ARCO-ERA5 `total_cloud_cover`, `low_/medium_/high_cloud_cover` (none) | **works today** through `esd.climate.era5.load(..., variables=["total_cloud_cover"])` (verified in the store 2026-09-23); needs only a gallery example; see F.2 |
| stations | 🆕 Station photos and camera links | NRCS `siteimages/{id}.jpg` (none), BC AQUARIUS portal and NuPoint satellite cameras (none), from the vendored clients | expose the `station_image_url` / `station_camera_url` fields `global_snow_networks` already renders; see **F.3** |

### Tier 3 — on the shelf (see C for why)

GOES LST (#8), SWOT, RADARSAT-1, PALSAR-2, NISAR GCOV, HRRR Zarr (service ending),
Sentinel-3 SYN, SMAP, MODIS albedo (MCD43), SnowEx campaign data, stream gauges,
census population/TIGER tracts (P5), GIBS/Worldview, declassified imagery, EOPF Zarr (watch),
per-scene cloud masks (MOD35/VIIRS CLDMSK swaths, GOES ABI clear-sky mask; F.2), third-party
webcams (ski areas, DOT road cameras, PhenoCam; F.3). geoBoundaries moved from here to F.1.

---

## B. Products with more than one source

Each comparison ends with the default and the reason. "Verified" means checked live on
2026-09-15; Planetary Computer was in a maintenance window that day, so PC facts are from its
task repository, not a live `/collections` call.

### B.1 Sentinel-1 radiometrically terrain-corrected backscatter

| | Planetary Computer `sentinel-1-rtc` | OPERA RTC-S1 (ASF DAAC) | GEE `OPERA/RTC/L2_V1/S1` |
| --- | --- | --- | --- |
| Resolution / grid | 10 m, UTM MGRS-like scenes | 30 m, UTM, **burst**-based granules | 30 m, OPERA product re-served |
| Coverage | global, 2014-10 → present | global land, 2016-04 → present | same as OPERA |
| Processing | Catalyst (PC), gamma0, Copernicus GLO-30 | JPL OPERA, gamma0, Copernicus GLO-30, with per-burst **static layers** (local/ellipsoidal incidence angle, layover-shadow mask, number of looks, area-normalization factors) | no static layers |
| Format / access | COG via STAC, SAS-signed hrefs, `odc-stac` | single-band COG per polarization + HDF5 metadata; CMR-STAC `cloudstac/ASF` (`OPERA_L2_RTC-S1_V1_1`) or `earthaccess` `OPERA_L2_RTC-S1_V1`; HTTPS via `datapool.asf.alaska.edu` or in-region S3 | `xee` |
| Credentials | none | Earthdata Login | Earth Engine |
| Gotchas | overviews' resampling method unverified (matters when reading at 80 m for speckle); PC stops archiving products without notice (MOD10A2 precedent); PC maintenance windows take the whole API down | burst geometry means many small granules per AOI; `stac.asf.alaska.edu` does **not** carry OPERA; no PC/Earth Search mirror | GEE quotas |
| Verified | ids from `planetary-computer-tasks`; API 503 during check | yes | catalog page fetched |

**Default:** Planetary Computer, because it is credential-free, 10 m, and reaches back to 2014.
**OPERA** is the source for the incidence-angle and mask layers and for users who need an
official NASA product or the 2016-onward burst archive. Other incidence-angle routes collected
in issue #10 (GEE `S1_GRD` angle band + DEM, Sentinel Hub, HyP3 on-demand RTC) are in E.4;
the pure-DEM computation stays as the credential-free fallback. Sentinel-1 latency across
providers is already measured live by Eric's next-overpass tool; link to it from the catalog page.

### B.2 Sentinel-2 L2A

| | Planetary Computer `sentinel-2-l2a` | Earth Search `sentinel-2-l2a` | Earth Search `sentinel-2-c1-l2a` (+ `sentinel-2-pre-c1-l2a`) | CDSE / EOPF Zarr |
| --- | --- | --- | --- | --- |
| Baseline handling | raw ESA values; **post-2022-01-25 offset must be undone by the client** (current `harmonize_to_old`) | Element 84 already harmonizes | Collection-1 reprocessing; items carry `raster:bands` `offset: -0.1` so the client applies the offset from metadata | native Zarr per product (sample service; operational late 2026/2027) |
| STAC metadata | historically no `raster:bands` → needs our `stac_cfg` | `raster:bands` + `eo:bands.common_name` (verified) → no `stac_cfg` | same (verified) | EOPF STAC, in flux |
| Ingestion lag | a 2026-04 PC discussion reports delays | S3 `us-west-2`, no signing | same | — |
| Credentials | none (signing) | none | none | CDSE account + S3 keys |
| Verified | ids only | yes | yes | sample STAC fetched |

**Default:** Planetary Computer (existing users, one catalog for S1+S2+DEM+WorldCover), with
Earth Search a first-class alternative and the existing PC-vs-Earth-Search notebook kept as the
comparison example. The baseline offset should be applied from `raster:bands` where present and
from the hard-coded date rule only for PC. `xcube-stac` (E.5) is an alternative loader over the
same catalogs; `odc-stac` covers our needs, so it is a reference, not a dependency.

### B.3 HLS v2.0

| | NASA CMR-STAC LPCLOUD | Planetary Computer `hls2-l30` / `hls2-s30` |
| --- | --- | --- |
| Ids | `HLSL30_2.0`, `HLSS30_2.0` (+ `*_VI_2.0` vegetation indices) — verified on both `/stac` and `/cloudstac` roots | dataset page linked in #11 (`planetarycomputer.microsoft.com/dataset/hls2-l30`); collection ids and freshness unverified during the PC outage |
| Access | COG behind Earthdata Login; GDAL needs netrc + cookie jar, or `earthaccess.open` | SAS-signed COG, no account |
| Freshness | authoritative, same-day as LP DAAC | mirror; lag unknown |
| Credentials | EDL | none |

**Default:** CMR-STAC LPCLOUD (authoritative, and the Fmask/angle bands are certainly there);
add PC as `source="planetary-computer"` once its ids are confirmed, mainly for users without an
Earthdata account. Replace the per-granule XML metadata scrape with STAC properties.

### B.4 MODIS snow cover

| | NSIDC via `earthaccess` | Planetary Computer `modis-10A1-061` / `modis-10A2-061` | GEE `MODIS/061/MOD10A1` |
| --- | --- | --- | --- |
| Status | MOD10A1, MOD10A2, MOD10A1F, MYD10A1(F), VNP10A1(F) all **cloud-hosted** under `NSIDC_CPRD` (verified) | Eric's note: archiving of MOD10A2 stopped ~2025-06; unverifiable during the check | present; MOD10A1F not in GEE |
| Format | HDF-EOS2 (HDF4): download, open GDAL subdatasets, date from `.AYYYYDDD.`; not streamable through `fsspec` file objects; **no DMR++ sidecars, so `earthaccess.virtualize()` cannot help** | COG via STAC | `xee` |
| Credentials | EDL | none | GEE |
| Successor | VNP10A1 / VNP10A1F v2 (375 m) same route | none | `NASA/VIIRS/002/VNP10A1F` (unverified) |

**Default:** NSIDC via `earthaccess`, because it is the archive of record and the only route that
carries the cloud-gap-filled and VIIRS products; PC kept as an optional historical source. This
is the worked example of "a collection disappearing from a cloud catalog is an access-route
failure, not a product failure" already noted in the best-practices inbox. The MODIS/VIIRS
catalog groups on PC and GEE (E.1) are the browse pages for what else exists.

### B.5 SNODAS

| | NSIDC G02158 direct | GEE `projects/climate-engine/snodas/daily` |
| --- | --- | --- |
| What | authoritative NOHRSC product, masked (CONUS) and unmasked grids, 2003-10 → present | Climate Engine's community re-hosting (SWE, Snow_Depth), ~1-day lag; documented at gee-community-catalog (E.1) |
| Format | one `.tar` per day of `.dat.gz` + `.txt` header pairs (flat binary, header gives grid); not range-readable, so no virtualization | analysis-ready ImageCollection via `xee` |
| Credentials | none | GEE |
| Cost | one tar per day requested (a few MB); needs a small reader | fast, lazy, spatially subset server-side |

**Default:** NSIDC direct (authoritative, credential-free); GEE kept as the fast path for long
time series. Document that they should agree and provide a gallery example that checks it.

### B.6 Sturm & Liston seasonal snow classification

| | NSIDC-0768 | Hosted COG (today: `uwcryo` Azure blob) |
| --- | --- | --- |
| Formats | GeoTIFF / netCDF / ASCII at 10″, 30″, 2.5′, 0.5° | the 10″ GeoTIFF only |
| Access | HTTPS behind Earthdata Login; not cloud-hosted (0 CMR granules) | anonymous range reads (verified) |
| Longevity | NSIDC | account funding ends 2026-10-26; custodianship ends 2026-12-31 — **stays here for now (Eric, 2026-09-15), location likely to change** |

**Default (decided):** NSIDC with Earthdata Login; `source="hosted-cog"` as the credential-free
option pointing at the current blob. The catalog entry carries a "location likely to change"
note; moving it (Zenodo record with a DOI, or a GitHub release asset) is a one-line URL change
plus a health-probe run.

### B.7 Land cover

| | ESA WorldCover (PC / AWS) | NLCD 2021_REL (GEE) | Annual NLCD (GEE community) | Dynamic World (GEE) |
| --- | --- | --- | --- | --- |
| Years | 2020 (v100), 2021 (v200); project ended, successor is Copernicus LCFM on CDSE | 2001–2021 epochs + science products | 1985–2024 annual (Collection 1.x) | 2015 → today, near-real-time |
| Extent / res | global 10 m | CONUS 30 m | CONUS 30 m | global 10 m |
| Credentials | none | GEE | GEE | GEE |

**Defaults:** WorldCover v200 for global static land cover (PC, with the AWS bucket as an
unsigned alternative); Annual NLCD for CONUS time series with `2021_REL` for the science
products; Dynamic World listed as a source for "what is the land cover *now*".

### B.8 DEMs

| | Copernicus GLO-30 (PC) | Copernicus GLO-30 (AWS Open Data / Earth Search) | Copernicus GLO-30 2024_1 (GEE) | 3DEP 1/3″ (S3 / PC) | NASADEM (LPCLOUD / PC / GEE) |
| --- | --- | --- | --- | --- | --- |
| Release | 2021 | 2021 (objects dated 2022) | **2024_1** (newest) | current | 2020 |
| Res / extent | 30 m global | same | same | 10 m US | 30 m ±60° |
| Credentials | none | none (unsigned) | GEE | none | EDL / none / GEE |

**Default:** Copernicus GLO-30 from PC; AWS/Earth Search as the no-signing fallback; note the
2024_1 release is only on GEE/CDSE. 3DEP is the US 10 m upgrade path (PC `3dep-lidar` group and
GEE `3dep` tag from #11 are in E.1). No public Zarr or Icechunk DEM mirror exists (checked).

### B.9 ERA5 family

| | ARCO-ERA5 (GCS) | Earthmover Icechunk ERA5 | WeatherBench2 (GCS / Icechunk) | NCAR `nsf-ncar-era5` (AWS) | GEE `ECMWF/ERA5*` | CDS ARCO Zarr (beta) | DestinE EDH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Content | hourly ERA5 (37 levels + single levels), 1940 → final 2026-05-31, **ERA5T to 2026-09-09** (verified 2026-09-15) | ERA5 1940–2025, quarterly updates, NRT paid; no ERA5-Land, no snow depth | 1959–2023, **includes `snow_depth`** | NetCDF monthly files, 3–4 month lag | ERA5 and **ERA5-Land**, hourly/daily/monthly aggregates | ERA5 single levels and **ERA5-Land hourly** (the ECMWF ARCO access notebook in #11 is the worked example) | ERA5-Land Zarr v3, 0.1°, monthly |
| Format | Zarr v2, consolidated | Icechunk | Zarr / Icechunk | NetCDF | `xee` | Zarr (tokenized) | Zarr v3 (token) |
| Credentials | none | none | none | none | GEE | CDS token | EDH token |

**Defaults:** ARCO-ERA5 for hourly ERA5; GEE for ERA5-Land and aggregates; CDS ARCO added when
it leaves beta so ERA5-Land no longer requires Earth Engine. Earthmover and WeatherBench2 are
listed as sources for reproducibility (versioned Icechunk snapshots) and for `snow_depth`.

### B.10 Basins

| | USGS WBD REST / HyRiver | GEE `USGS/WBD/2017/HUC*` | BasinATLAS gdb (figshare) | HydroSHEDS per-region zips | GEE `WWF/HydroATLAS/v1/Basins` |
| --- | --- | --- | --- | --- | --- |
| Extent | US | US | global, levels 1–12, with attributes | global, per continent, levels 1–12 | global |
| Size | server-side query | server-side | 2.7 GB download, `mask=` pushdown | ~100 MB per region | server-side |
| Credentials | none | GEE | none | none | GEE |

**Defaults:** WBD REST for HUCs (drops the GEE requirement for a public dataset); BasinATLAS for
attributes with HydroSHEDS regional zips as the lighter route; GEE as alternatives.

### B.11 PlanetScope / SkySat (Planet Labs)

Added 2026-09-15 (plan §12 Q17). What exists locally: the untracked `docs/examples/sandbox.ipynb`
holds a `PlanetData` class written against the Data API v1 with `requests` (quick-search with
geometry / `acquired` / `cloud_cover` filters → per-item asset activation and polling →
`rioxarray.open_rasterio` on the activated full-scene GeoTIFF → `clip_box` → `xr.concat` over
`time`, plus a folium preview that overlays `https://tiles.planet.com/data/v1/{item_type}/{item_id}/{z}/{x}/{y}.png`),
and `docs/examples/planet_data/` holds the six `PSScene` `ortho_analytic_4b` strips it fetched for
2023-07-01 over the Rainier AOI (4-band uint16 blue/green/red/nir, EPSG:32610, 3 m, 256-px tiles
with 3/9/27 overviews, nodata 0, 12–13 k × 8.7–9.4 k px, 555–675 MB each — 3.6 GB to clip a
0.4° × 0.27° box). That is the argument for the default below.

| | Data API (search) + direct asset read | **Orders API with `clip`** | Subscriptions API | Sentinel Hub / Planet Insights Platform | Basemaps (Mosaics API) |
| --- | --- | --- | --- | --- | --- |
| What you get | item metadata; activated full-scene assets (`ortho_analytic_4b`, `_8b`, `_sr`, `ortho_udm2`, `ortho_visual`) as signed URLs | the same assets **cut to the AOI**, optionally `harmonize`d (to Sentinel-2 or Dove-Classic) and `composite`d, delivered as COGs to a zip or a cloud bucket | recurring delivery of new scenes or **Analysis-Ready PlanetScope** (daily, harmonized) / Planetary Variables to a bucket | STAC-style Catalog API + processing API over PlanetScope, S2, Landsat via `sentinelhub-py`; the SDK plans to absorb it | monthly/quarterly mosaics incl. NICFI tropics; quad tiles |
| AOI clipping | client-side after streaming the whole strip (the sandbox route) | server-side | server-side | server-side (processing API) | quad-level |
| Quota / cost | activation of a full scene charges the full scene area against an E&R quota; streaming 600 MB per scene | charged for the clipped area only | subscription-level | processing units | basemap licence |
| Credentials | Planet account (OAuth2 session, API key, or M2M client) | same | same | Sentinel Hub OAuth client | same as Data API |
| Fits the design contract | `search_*` returns a GeoDataFrame — yes; `load` lazy over signed URLs — yes but wasteful | `search` → `order` → `load` from delivered COGs (lazy, `odc-geo` grid) — yes; an order is an explicit, quota-spending step, so `load(..., order=True)` or a separate `order()` call | not an on-demand loader | different auth stack; later | different product |
| Verified 2026-09-15 | SDK 3.6.0 on PyPI/conda-forge; env-var names from the SDK source; the sandbox route worked in 2024-10 (files on disk) | SDK README lists Orders as supported; tool names from the Planet docs 🔍 re-check against the SDK guide when implementing | SDK README lists Subscriptions | SDK README notes unification work | SDK README lists Mosaics |

**Default:** Data API for search and quick-look tiles, **Orders API `clip` for anything that is
loaded as an array**; `source="data-api"` keeps the direct-read route for single-scene checks.
UDM2 decoding (`clear`, `snow`, `shadow`, `light haze`, `heavy haze`, `cloud`, `confidence`,
`unusable` bands 🔍 confirm band order) goes to `processing.optical.decode_udm2`. Item types to
support first: `PSScene` (3 m, 2016→, 4- and 8-band), then `SkySatCollect` (50 cm); RapidEye
(`REOrthoTile`, ended 2020) only if a historical need appears. Tests: recorded Data API JSON for
the unit tier; the live smoke test searches only (free); ordering stays a manual gallery step.
Health probe: authenticated `GET https://api.planet.com/data/v1/`. Licence: Planet imagery is
not redistributable — no real-scene crops in git without checking the E&R terms `[needs Eric]`.

---

## C. Evaluation of every product idea collected so far

Sources of the list: issue #11 (Eric's idea dump, 2024–2026), issues #8/#9/#10, the 2026-09-15
audit of cloud-hosted snow datasets, and the network scoping in `global_snow_networks`
(issues #2–#22 there). Verdicts: **rewrite** = Tier 1 · **follow-on** = Tier 2 · **shelf** = Tier 3,
keep the row and its links, revisit · **wrap** = point users at an existing package rather than
re-implement · **reference** = not a product; kept in §E as an access pattern or tool to learn from.

| Idea | What it is | Why it matters for snow science | Access (host · format · creds) | Effort | Verdict |
| --- | --- | --- | --- | --- | --- |
| OPERA RTC-S1 + static layers | NASA/JPL Sentinel-1 RTC, 30 m bursts, with incidence-angle and mask layers | official RTC with the geometry layers wet-snow and melt-timing work needs; replaces the GEE incidence computation | ASF · COG · EDL | medium (CMR-STAC + burst mosaicking) | **rewrite** |
| VIIRS snow VNP10A1 / VNP10A1F | 375 m daily snow cover, cloud-gap-filled variant | MODIS Terra ended; VIIRS is the continuity product for every MOD10-based method | NSIDC · HDF5 · EDL | low once MODIS route exists | **rewrite** (priority 1) |
| SNODAS from NSIDC | authoritative daily SWE/depth grids | removes the GEE dependency from the most-requested SWE product | NSIDC · `.dat.gz` tarballs · none | medium (small reader) | **rewrite** (priority 2) |
| HLS modernization | Landsat + Sentinel-2 harmonized 30 m | the only 30 m harmonized optical record; issues #5, #6 | LPCLOUD · COG · EDL | low | **rewrite** (priority 3) |
| Landsat C2 L2 (+ `landsat-c2l3-fsca`) | 30 m optical back to 1982; USGS fractional snow cover | pre-Sentinel snow cover; fSCA is a ready-made snow product | Earth Search/PC · COG · none; USGS STAC requester-pays | low | **follow-on** |
| VIIRS surface reflectance VNP09GA | daily 500 m/1 km reflectance | the reflectance behind VIIRS snow; VIIRS-era NDSI | GEE · `xee` · GEE; LPCLOUD · EDL | low | **follow-on** |
| ASO lidar SD/SWE | airborne lidar snow depth and SWE, 50 m | the reference dataset for validating SWE/depth | NSIDC · GeoTIFF · EDL (cloud) | low | **follow-on** |
| AMSR `AU_DySno` | passive-microwave daily SWE | only global daily SWE observation | NSIDC · HDF5 · EDL | low | **follow-on** |
| ICESat-2 ATL06/ATL08 | laser altimetry heights | snow depth by differencing | NSIDC · HDF5 · EDL | medium; `icepyx`/`sliderule` exist | **wrap** |
| S2/HLS snow cover (#9) | NDSI-threshold and let-it-snow style snow maps | turns imagery into a snow product; the most-asked feature | computed | medium | **follow-on** (NDSI first; algorithm links in E.3) |
| 3DEP 1/3″ and lidar | US 10 m DEM, point clouds | finer terrain for small basins; `py3dep` already a dependency | S3/PC · COG/EPT · none | low | **follow-on** |
| NASADEM | 30 m void-filled SRTM | alternative DEM where GLO-30 has artefacts | PC/GEE | low | **shipped 0.3** as `product="nasadem"`, with SRTM GL1 (`"srtm"`, GEE) and ALOS World 3D (`"alos-dem"`) beside it |
| Insolation proxies (CHILI, heat load) | topographic radiation indices | aspect control on melt timing (P4's CHILI classes) | GEE; or computed from a DEM with McCune & Keon 2002 | low | **rewrite** (CHILI exists) + **follow-on** (local computation) |
| Dynamic World | 10 m near-real-time land cover | current-year land cover for masking | GEE | low | **follow-on** |
| Hansen GFC 2025 | tree cover and loss year | forest change affects snow interception | GEE | low | **follow-on** |
| Daymet V4R1 | 1 km daily met, North America | standard snow-model forcing | ORNL cloud (EDL, DMR++) / PC Zarr | low | **follow-on** |
| gridMET, PRISM, NLDAS-2 | CONUS met forcings and normals | temperature/precip context for melt | GEE / PC / PRISM web service | low | **follow-on** |
| NLDAS-3 (beta) | 1 km North America forcing, 2001–2023 | successor to NLDAS-2 | `s3://nasa-waterinsight` NetCDF + kerchunk Parquet + Icechunk (anonymous, verified) | low (already virtualized) | **follow-on** |
| CONUS404 | 4 km WRF reanalysis, 40+ years | high-resolution hydroclimate forcing (linked in #11) | USGS HyTEST Zarr on OSN (anonymous, verified) | low (already Zarr) | **follow-on** as a Zarr source |
| GPM IMERG | global daily precipitation | precipitation where gauges are absent | GES DISC · NetCDF · EDL (DMR++) | low | **follow-on** |
| NOAA PSL climate indices | ENSO/PDO/AO time series | interannual context; used in the P4 anomaly figures | PSL text files · none | trivial | **follow-on** |
| Canada MSC GeoMet, SLF IMIS, Synoptic | national/regional station networks with open APIs | extends the station inventory beyond the five current clients | OGC API / REST · none or free token | medium each (new client) | **follow-on** (in `global_snow_networks` scope) |
| GOES LST / ABI (#8) | geostationary land surface temperature and imagery, 5–15 min | melt-refreeze diurnal cycle; the only sub-hourly view | AWS `noaa-goes` · NetCDF · none; PC GOES group; GOES LST on Azure blob | high (orthorectification, volume) | **shelf / wrap** `goes-ortho` + `goes2go` (E.2) |
| PALSAR-2 ScanSAR | L-band backscatter (LIN + mask bands) | L-band melt detection, NISAR proxy (P6); code snippet already in the old notebook | GEE · `xee` | low via GEE | **shelf** (until the NISAR work needs it; then trivial) |
| NISAR GCOV | L-band RTC | Eric's postdoc focus | ASF · HDF5 · EDL | medium; wait for operational data | **shelf** |
| SWOT | surface water elevation, river/lake | lakes/reservoirs downstream of snowmelt; PO.DAAC S3 tutorials are a good access-pattern reference | PO.DAAC · NetCDF · EDL | medium | **shelf** (P5/P22 relevance later) |
| RADARSAT-1 | historical C-band SAR 1995–2013 | pre-Sentinel-1 SAR record | ASF · EDL | medium | **shelf** |
| Planet (PlanetScope, SkySat) | commercial 3 m daily imagery (PSScene, 4/8-band + UDM2), 50 cm SkySat, basemaps | near-daily 3 m snow-covered area and snow-disappearance timing at hillslope scale; validation chips for MODIS/VIIRS/S2 snow products (P2 lists a near-daily binary snow mask as the fix for MODIS's resolution); basemap-only API quirks already in the best-practices inbox | Planet Data + Orders APIs via the `planet` SDK 3.x · COG · Planet account (OAuth2/API key), E&R quota, non-redistributable licence | medium (SDK does search/order/download; our part is the AOI→order→COG→xarray glue, UDM2 decoding, and the `planet` auth provider) | **follow-on** (Tier 2, B.11) — asked for 2026-09-15, plan §12 Q17 |
| HRRR | 3 km NWP | high-resolution forcings | `s3://hrrrzarr` — **service ends October 2026** per the AWS registry | — | **shelf** (do not build on the Zarr; native GRIB on `noaa-hrrr-bdp-pds` remains) |
| Sentinel-3 SYNERGY VG1/V10 | 300 m / 1 km vegetation products | coarse snow-free/vegetation context; PC NetCDF example notebooks | PC · NetCDF · none | medium | **shelf** |
| SMAP `SPL4SMGP` | soil moisture | soil-moisture pulses used in P12 validation | NSIDC · EDL | low | **shelf** (niche) |
| MODIS albedo MCD43A3/A4 | 500 m albedo / NBAR | snow albedo, melt energy | LPCLOUD · HDF · EDL | low | **shelf** |
| SnowEx campaign data | snow pits, GPR, lidar | method development | NSIDC · mixed · EDL | — | **shelf** |
| Stream gauges (hydrocloud, USGS NWIS) | discharge | runoff onset validation (P2–P5) | REST · none | — | **wrap** HyRiver `pygeohydro`; hydrocloud station list as reference |
| MetPy, metloom, SynopticPy | met-station and synoptic tooling | overlap with the station clients; MetPy for unit-aware met calculations | — | — | **reference** (metloom is a peer of the AWDB client; MetPy units could back `processing`) |
| geoBoundaries, US census, TIGER, `censusdata` | admin boundaries, population | P5 basin-population work | GEE / PC | low | **boundaries → follow-on** (F.1, 2026-09-23); population and tracts stay **shelf** (P5-specific) |
| Natural Earth vectors (countries, states/provinces, lakes, rivers, glaciated areas, populated places) | 1:10m / 1:50m / 1:110m cartographic vectors | outlines and labels for every map; country/state AOIs by name | `naciscdn.org` zipped shapefiles · none · public domain | low | **follow-on** (F.1 phase A) |
| US Census cartographic boundaries | states, counties (500k / 5m / 20m) | the authoritative US outlines; replaces the personal `eric.clst.org` mirror of the 2010 files | `www2.census.gov/geo/tiger/GENZ{year}` zips · none · public domain | low | **follow-on** (F.1 phase A) |
| GMBA Mountain Inventory v2 | 8 000+ mountain-range polygons, hierarchical (Snethlage et al. 2022) | "which range is this", per-range statistics; the mountain definition to pair with the Wrzesien mask | EarthEnv zip · none · CC BY 4.0 🔍 | low | **follow-on** (F.1 phase B) |
| RGI 7.0 glacier outlines | global glacier polygons | glacier masks (the SNODAS glacier artefact, snow-vs-ice) | NSIDC-0770 · EDL 🔍 | low | **follow-on** (F.1 phase C, or `snow` theme) |
| EPA Level III/IV ecoregions; RESOLVE Ecoregions 2017; PAD-US | ecological regions; protected areas | stratifying results; station and camera land ownership | EPA S3 zip (verified); others 🔍 | low | **follow-on** (F.1 phase C) |
| Overture Maps `divisions` | admin boundaries as GeoParquet, monthly releases | the cloud-native route with bbox pushdown | S3/Azure GeoParquet · none · ODbL | low–medium (release pinning) | **reference → later source** (F.1) |
| EarthEnv cloud frequency (Wilson & Jetz 2016) | 1 km MODIS cloud climatology, 2000–2014 | where and when optical snow mapping loses days; campaign and sensor planning; context for the Wrzesien clouds layer | `data.earthenv.org` GeoTIFF (0.7–0.8 GB, striped, no overviews) · none · **CC BY-NC 4.0** | low | **follow-on** (F.2) |
| ERA5 cloud cover | hourly total/low/medium/high cloud fraction, 0.25° | cloud as a melt-energy term (longwave, shortwave) | ARCO-ERA5 · none | none (already loadable) | **docs** (gallery example, F.2) |
| MODIS/VIIRS gridded cloud flags (MOD09GA `state_1km`, VNP09GA QF) | daily per-pixel cloud state, 1 km | the inputs to EarthEnv; per-day cloud masks aligned with the snow products | GEE / LPCLOUD · EDL | low (VNP09GA already Tier 2) | **follow-on** (decode in `processing.masks`, F.2) |
| MOD35 / VIIRS CLDMSK L2; GOES ABI ACM; PATMOS-x, CLARA-A3, ISCCP-H | swath cloud masks; geostationary clear-sky mask; AVHRR-era cloud climate records | sub-daily cloud (GOES), 40-year cloud trends | LAADS · EDL; AWS `noaa-goes`, `noaa-cdr-patmosx-…` · none | medium–high (swath/geostationary geometry) | **shelf** (F.2) |
| Sentinel-2 Cloud Score+ | per-pixel S2 cloud score | a better S2 mask over snow than SCL | GEE `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` · GEE | low | **follow-on** as `mask="cloud-score-plus"` (F.2) |
| Station photos and camera links | NRCS site photos, BC ASWS photos and satellite cameras | what a site looks like (canopy, exposure) and what it looks like **now** | vendored clients · none | low | **follow-on** (F.3) |
| Webcams: ski areas, DOT road cameras, NPS, FAA WeatherCams, PhenoCam, Windy webcams | third-party still-image cameras near snow | visual snow-on/snow-off checks and timelapses | mixed: open APIs (PhenoCam, Caltrans CWWP2, WSDOT with access code, NPS with key, Windy with key) vs vendor pages | medium each; terms of use vary | **shelf / reference** (F.3) |
| GIBS / Worldview | tile services and browser | browse imagery, quicklooks for catalog pages | — | — | **reference** (use GIBS tiles for docs thumbnails, not analysis) |
| Declassified imagery (USGS), keyhole viewer | 1960s–80s film | historical snow extent — a curiosity | USGS EROS | high | **shelf** |
| UCS satellite database, CelesTrak | satellite catalog and TLEs | orbit/overpass computation (Eric's next-overpass tool) | CSV/TLE · none | — | **reference** (belongs with the overpass tool) |
| EOPF Sentinel Zarr | ESA's native Zarr Sentinel products | future-proofing S1/S2 access | CDSE · Zarr · CDSE creds; sample STAC | watch | **shelf** (revisit when operational, late 2026/2027; toolkit links in E.5) |
| xcube-stac | STAC → xcube datacube | alternative loader over PC/CDSE | — | — | **reference** (odc-stac covers it) |
| StreamJoy, leafmap/maplibre GEE, contextily+GEE, uxarray, xeofs cookbooks | animation, web-map and analysis tooling | docs polish and downstream analysis, not data access | — | — | **reference** (tooling knowledge → best-practices repo) |

---

## D. Example gallery

One ~30-line script per Tier 1 product under `examples/<theme>/plot_<product>.py` (executed by
sphinx-gallery), plus multi-source and access-pattern how-tos under `examples/howto/`:

- `plot_s1_sources.py` — the same AOI and month from Planetary Computer and OPERA RTC-S1,
  with the OPERA incidence angle alongside (B.1).
- `plot_s2_catalogs.py` — the existing PC-vs-Earth-Search comparison, rewritten on the new API (B.2).
- `plot_modis_viirs_continuity.py` — MOD10A1F and VNP10A1F over one winter (B.4).
- `plot_snodas_sources.py` — NSIDC vs GEE SNODAS agreement check (B.5).
- `plot_ucla_sr_virtualized.py` — **the virtualization example**: 36 water years of UCLA SR
  SWE for one pillow location, `virtualize="auto"` vs the plain `open_mfdataset` path, timing
  both, showing the reference cache being reused on the second run, and printing which access
  mode (`direct` in us-west-2, `indirect` elsewhere) was chosen.
- `plot_station_vs_sar.py` — SNOTEL SWE against Sentinel-1 backscatter for one pillow.
- `plot_snow_cover_vs_snodas.py` — MODIS snow cover vs SNODAS SWE over a basin.
- `plot_ccss_2023.py` — the CCSS percent-of-normal story from the old station notebook.
- `plot_dowy_max_swe_trend.py` — the day-of-max-SWE trend from the old station notebook, using
  the vectorized water-year helpers.
- `plot_credentials_and_region.py` — what `esd.auth.status()` and `esd.config.region()` report,
  and how a `CredentialError` names the alternative source.
- `plot_planetscope_vs_s2.py` — one PlanetScope order clipped to the AOI next to the same-day
  Sentinel-2 scene, NDSI from both and the UDM2 snow band alongside (B.11); needs Planet
  credentials and spends quota, so it lives in the credentialed gallery subset and is executed
  only by the scheduled docs build.

- `terrain/plot_hillshade.py` — **shipped 2026-09-23**: the default Natural Earth style over the
  central Cascades, the Wrzesien mountain snow classes drawn over plain shaded relief, and the
  globe in Robinson (the runoff-onset notebook's `gdalwarp` step, in memory).
- `boundaries/plot_boundaries.py` (F.1) — Washington's outline and its counties over the
  hillshade, the GMBA ranges intersecting the AOI, and `parse_aoi` on a state polygon.
- `climate/plot_cloud_frequency.py` (F.2) — EarthEnv mean annual and monthly cloud frequency over
  the Cascades next to ERA5 `total_cloud_cover` for the same months, and the fraction of
  MOD10A1 days lost to cloud.
- `stations/plot_station_photos.py` (F.3) — the SNOTEL Paradise site photo beside its SWE record.

Each how-to doubles as a live integration test of two sources agreeing, and its thumbnail
feeds the README gallery montage.

---

## E. Link register

Every link collected in the issues and idea dumps, grouped by what it is. "Incorporate?" says
whether it becomes a catalog entry (**product**), shapes how a loader is written (**access
pattern**), is used in the docs (**docs**), is a paper or spec we cite (**reference**), or belongs
in the best-practices wiki rather than this package (**wiki**). Nothing is dropped; rows marked
"no" stay here so the idea is not lost.

### E.1 Data products and catalog pages (issue #11)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR-2_Level2_2_ScanSAR | PALSAR-2 ScanSAR L-band on GEE (LIN + mask bands) | L-band melt detection; NISAR proxy (P6); snippet already in the old notebook | **product**, Tier 3 (trivial via `providers.gee` when needed) |
| https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP09GA | VIIRS daily surface reflectance on GEE | VIIRS-era reflectance/NDSI | **product**, Tier 2 |
| https://planetarycomputer.microsoft.com/dataset/group/modis | MODIS product group on PC | browse page for what PC still archives; the MOD10 rows were partly dropped in 2025 | **docs** (catalog page link) + health probe of PC MODIS collections |
| https://developers.google.com/earth-engine/datasets/catalog/modis | MODIS on GEE | alternative MODIS route with GEE creds | **product** source for MOD10A1 (B.4) |
| https://podaac.github.io/tutorials/quarto_text/SWOT.html ; https://podaac.github.io/tutorials/notebooks/datasets/SWOTHR_s3Access.html | SWOT overview and in-region S3 access tutorial | downstream water; the S3 tutorial is a model for our `access="direct"` path | **product** Tier 3; **access pattern** yes (direct-S3 credential flow) |
| https://asf.alaska.edu/datasets/daac/radarsat-1/ | RADARSAT-1 at ASF | pre-2014 C-band SAR | **product**, Tier 3 |
| https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_CHILI | CHILI on GEE | existing product | **product**, Tier 1 |
| https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0143619 | Theobald et al. 2015, CHILI paper | citation for the CHILI catalog entry | **reference** (citation) |
| https://onlinelibrary.wiley.com/doi/10.1111/j.1654-1103.2002.tb02087.x | McCune & Keon 2002, heat-load equations | lets us compute a heat-load index from any DEM without GEE | **reference** → `processing.terrain.heat_load_index` (Tier 2) |
| https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001 | NASADEM on GEE | alternative DEM | **product**, Tier 2 (GEE source) |
| https://planetarycomputer.microsoft.com/dataset/group/3dep-lidar | 3DEP lidar products on PC | US 1 m–10 m terrain from lidar | **product**, Tier 2 |
| https://developers.google.com/earth-engine/datasets/tags/3dep | 3DEP on GEE | same, GEE route | **product**, Tier 2 (GEE source) |
| https://github.com/planetlabs/notebooks/tree/master ; https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/Data-API/planet_python_client_introduction.ipynb ; https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks | Planet API notebooks | commercial imagery access; licensing limits use | **product**, Tier 2 (B.11); **access pattern** reference (per-mosaic probe gotcha already in the wiki) |
| https://github.com/planetlabs/planet-client-python ; https://planet-sdk-for-python.readthedocs.io/en/latest/ ; https://planet-sdk-for-python.readthedocs.io/en/latest/auth/auth-overview/ | `planet` SDK 3.x repository, docs, client-authentication guide | the search/order/download client and the auth stack `providers.planet` and `auth.planet` delegate to | **access pattern** yes (B.11, plan §5) |
| https://developers.planet.com/docs/data/psscene/ ; https://developers.planet.com/docs/data/analysis-ready-ps/ ; https://developers.planet.com/docs/subscriptions/imagery-subs/ ; https://developers.planet.com/docs/subscriptions/pvs-subs/ ; https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/Data-API/search_and_preview_quickstart.ipynb | PSScene item/asset spec; Analysis-Ready PlanetScope; imagery and Planetary Variables subscriptions; search-and-preview quickstart (the links at the top of the sandbox `PlanetData` cell) | asset names and UDM2 definition for the catalog entry; ARPS/PV are the Tier 3 subscription products | **docs** (catalog page) + **reference** |
| https://www.hydrocloud.org/station-list.html | hydrocloud stream-gauge station list | runoff validation for P2–P5 | **wrap** HyRiver; **reference** |
| https://github.com/Unidata/MetPy | MetPy | unit-aware meteorological calculations | **reference**; possible backing for units in `processing` |
| https://github.com/M3Works/metloom | metloom | peer station client (SNOTEL, CDEC, Mesowest) | **reference** for `stations` API design; not a dependency |
| https://psl.noaa.gov/data/climateindices/list/ | NOAA PSL climate indices | ENSO/PDO/AO context | **product**, Tier 2 (small table loader) |
| https://developers.google.com/earth-engine/datasets/catalog/WM_geoLab_geoBoundaries_600_ADM1 | geoBoundaries ADM1 on GEE | admin boundaries for basin/population work | **product**, Tier 3 (P5) |
| https://planetarycomputer.microsoft.com/dataset/us-census | US Census on PC | population for P5 | **product**, Tier 3 (P5) |
| https://github.com/jtleider/censusdata | `censusdata` package | Census API client | **reference**/wrap |
| https://developers.google.com/earth-engine/datasets/catalog/TIGER_2010_Tracts_DP1 ; https://developers.google.com/earth-engine/datasets/catalog/TIGER_2010_Blocks | TIGER tracts/blocks on GEE | population-weighted basins | **product**, Tier 3 (P5) |
| https://prism.oregonstate.edu/downloads/ ; https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf ; https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_ANd | PRISM downloads, web service spec, GEE daily normals | CONUS climate normals and daily met | **product**, Tier 2 (GEE first; web service as a second source) |
| https://gee-community-catalog.org/projects/snodas/ | SNODAS community asset documentation | current GEE route; note "check for authoritative" → answered: NSIDC direct is authoritative (B.5) | **product** source, Tier 1 |
| https://planetarycomputer.microsoft.com/dataset/sentinel-3-synergy-vg1-l2-netcdf ; https://planetarycomputer.microsoft.com/dataset/sentinel-3-synergy-v10-l2-netcdf | Sentinel-3 SYN VG1/V10 on PC | coarse vegetation products with example notebooks | **product**, Tier 3 |
| https://blaylockbk.github.io/SynopticPy/meteogram/ | SynopticPy | Synoptic/MesoWest station API client (free token) | **reference** for a future Synoptic station client (Tier 2) |
| https://www.usgs.gov/data/conus404-four-kilometer-long-term-regional-hydroclimate-reanalysis-over-conterminous-united | CONUS404 | 4 km hydroclimate reanalysis; Zarr on HyTEST/OSN | **product**, Tier 2 (Zarr source) |
| https://planetarycomputer.microsoft.com/dataset/hls2-l30 | HLS L30 on PC | credential-free HLS mirror | **product** source (B.3) |
| https://github.com/ecmwf-training/dss-notebooks/blob/main/datasets/reanalysis-era5-land/arco-access.ipynb | ECMWF ERA5-Land ARCO access notebook | the worked example for the CDS ARCO Zarr route (B.9) | **access pattern** yes → `source="cds-arco"` |

### E.2 GOES (issue #8)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf | GOES-R beginner's guide | product naming, scan sectors, projection | **reference** |
| https://registry.opendata.aws/noaa-goes/ | GOES 16/17/18 on AWS (NetCDF, public) | the raw source | **product**, Tier 3 |
| https://planetarycomputer.microsoft.com/dataset/group/goes | GOES products on PC | STAC-cataloged alternative | **product** source, Tier 3 |
| https://planetarycomputer.microsoft.com/dataset/storage/goes-lst | GOES LST on Azure blob | the LST product Eric cares about | **product**, Tier 3 |
| https://nbviewer.org/github/oceanhackweek/ohw-tutorials/blob/OHW20/10-satellite-data-access/goes-cmp-netcdf-zarr.ipynb | NetCDF vs Zarr read comparison for GOES | shows why a Zarr/virtual layer matters for many small NetCDFs | **access pattern**, informs any future GOES loader |
| https://nbviewer.org/github/awslabs/amazon-asdi/blob/main/examples/dask/notebooks/goes16_dask.ipynb | GOES with Dask on Fargate | scaling pattern | **reference** |
| https://github.com/HamedAlemo/visualize-goes16/blob/main/visualize_GOES16_from_AWS.ipynb | GOES visualization from S3 | quicklook pattern | **docs** reference |
| https://spestana.github.io/goes-ortho/index.html ; https://github.com/spestana/goes-ortho/blob/main/src/goes_ortho/Downloader.py#L5 ; https://spestana.github.io/goes-ortho/_modules/orthorectify.html#ortho_zarr | Steven Pestana's `goes-ortho` (download + orthorectify to Zarr) | the orthorectification we should not re-implement | **wrap** (`goes-ortho` as the GOES loader, if GOES is ever added) |
| https://github.com/pydata/xarray/issues/7574 | `open_mfdataset` + fsspec + dask issue | a known sharp edge for many-file NetCDF; virtualization sidesteps it | **reference** (why `virtualize()` exists) |
| https://goes2go.readthedocs.io/en/latest/index.html ; https://goes2go.readthedocs.io/en/latest/_modules/goes2go/data.html#goes_timerange | `goes2go` file discovery | time-range → file list logic | **wrap**/reference |

### E.3 Sentinel-2 / HLS snow cover algorithms (issue #9)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow ; https://github.com/orfeotoolbox/let-it-snow/tree/master | LIS (let-it-snow) snow cover algorithm (Theia) | operational S2/L8 snow cover; the algorithm to port after NDSI | **follow-on** algorithm for `snow.snow_cover` |
| https://essd.copernicus.org/articles/11/493/2019/ | Gascoin et al. 2019, Theia Snow collection | method paper and validation | **reference** (citation) |
| https://www.mdpi.com/2072-4292/13/10/1957 | revised snow/cloud discrimination (Gran Paradiso) | improves snow-vs-cloud separation, our main masking problem | **reference**; candidate rule for `processing.snow` |
| https://ieeexplore.ieee.org/document/9554366 | evaluation of an operational snow cover classifier | accuracy expectations | **reference** |
| https://github.com/chiararik/rLIS_VLab | R implementation of LIS | second reference implementation | **reference** |

### E.4 Sentinel-1 local incidence angle (issue #10)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://github.com/egagli/generate_sentinel1_local_incidence_angle_maps | Eric's LIA-map generator | may contain per-orbit logic worth keeping | **fold in** what OPERA static + DEM computation lack |
| https://gis.stackexchange.com/questions/352602/getting-local-incidence-angle-from-sentinel-1-grd-image-collection-in-google-ear | GEE recipe (basis of the current code) | the equations in use today | **reference**; kept as the GEE fallback route |
| https://github.com/palubad/LC-SLIAC/blob/master/javascript_codes/LC-SLIAC_global.js | LC-SLIAC land-cover-specific LIA correction | radiometric correction beyond geometry | **reference**; possible `processing.sar` extension |
| https://developers.google.com/earth-engine/guides/sentinel1 | GEE Sentinel-1 guide | GRD preprocessing details | **reference** |
| https://docs.sentinel-hub.com/api/latest/data/sentinel-1-grd/examples/ ; https://docs.sentinel-hub.com/api/latest/data/sentinel-1-grd/#available-bands-and-data ; https://docs.sentinel-hub.com/api/latest/api/catalog/examples/ ; https://sentinelhub-py.readthedocs.io/en/latest/examples/data_collections.html | Sentinel Hub S1 GRD API (incl. local incidence angle band) | a commercial route that serves LIA directly | **shelf** (paid); reference |
| https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/ | ASF HyP3 RTC product guide | on-demand RTC with incidence-angle layer option | **reference**; HyP3 as a possible on-demand source later |

### E.5 Access patterns, tooling and learning resources (issue #11)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://nasa-gibs.github.io/gibs-api-docs/ ; https://worldview.earthdata.nasa.gov/ | GIBS tile API and Worldview | quicklooks for catalog pages | **docs** (thumbnails), not analysis |
| https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-1 ; https://keyhole.engelsjk.com/ | declassified imagery archive and viewer | historical snow extent curiosity | **shelf**; reference |
| https://www.ucsusa.org/resources/satellite-database ; https://celestrak.org/ | satellite catalog; TLEs | overpass prediction | **reference** (Eric's next-overpass tool) |
| https://discourse.pangeo.io/t/streamjoy-enjoy-animating-images-into-gifs-and-mp4s-in-parallel/4122/5 | StreamJoy animations | docs/gallery animations | **wiki** (visualization page) |
| https://leafmap.org/maplibre/google_earth_engine/ | leafmap + MapLibre + GEE | interactive maps in docs | **wiki** |
| https://developers.google.com/earth-engine/tutorials/community/data-converters | GEE data converters | moving results in/out of GEE | **wiki** (gee-and-xee page) |
| https://contextily.readthedocs.io/en/latest/friends_gee.html | contextily + GEE basemaps | plotting convenience | **wiki** |
| https://projectpythia.org/metpy-cookbook/ ; https://projectpythia.org/metpy-cookbook/notebooks/synoptic/hpa-vorticity-advection/ ; https://cookbooks.projectpythia.org/ | Pythia cookbooks | teaching material style; docs structure model | **docs** (how-to style) |
| https://github.com/xcube-dev/xcube-stac/blob/main/examples/notebooks/sentinel_2_planetary_computer.ipynb ; https://github.com/xcube-dev/xcube-stac/blob/main/examples/notebooks/sentinel_3_cdse.ipynb | xcube-stac over PC and CDSE | alternative STAC loader; CDSE access example | **reference** (CDSE auth pattern useful for EOPF later) |
| https://github.com/earth-mover/aifs-demo/blob/main/run-aifs-earthmover.ipynb | AIFS forecast via Earthmover | Icechunk-backed forecast access pattern | **wiki**; reference for Icechunk source entries |
| https://projectpythia.org/unstructured-grid-viz-cookbook/ ; https://projectpythia.org/eofs-cookbook/notebooks/climate-modes-xeofs/ | uxarray; xeofs | analysis downstream of the data (P16 EOF work) | **wiki** (already has an EOF page) |
| https://www.youtube.com/watch?v=ddR5OEF4-yM ; https://github.com/EOPF-TOOLKIT ; https://eopf-toolkit.github.io/eopf-101/01_about_eopf/13_overview_eopf_datasets.html ; …/02_about_eopf_zarr/24_zarr_struct_S2L2A.html ; …/22_zarr_structure_S1GRD.html ; …/23_S1_basic_operations.html ; …/25_zarr_struct_S3.html ; …/03_about_chunking/32_zarr_chunking_strat.html ; …/06_eopf_zarr_in_action/65_create_overviews.html ; …/66_use_overviews.html | EOPF Sentinel Zarr explorer, toolkit and eopf-101 chapters | the future native-Zarr route for S1/S2/S3; chunking and overview chapters are directly relevant to how we would read it | **shelf** product (operational late 2026/2027); **access pattern** to revisit; toolkit links → **wiki** |
| https://github.com/google/Xee/issues/345 ; https://github.com/google/Xee/issues/174 ; https://github.com/google/Xee/issues/302 | xee + Dask performance issues | why `providers.gee` must set the high-volume endpoint, chunking, and worker re-init | **access pattern** yes (informs `providers.gee`) |

### E.6 Ideas without links (issues #1, #3, #5)

| Issue | Idea | Incorporate? |
| --- | --- | --- |
| #1 | tests for all modules; datetime integrity for stations | **yes** — §6 of the plan (three test tiers); station datetime checks become offline tests on fixtures |
| #3 | `clip_to_bbox` option on every loader | **yes** — `clip=` in the AOI contract (§2.1) |
| #5 | HLS reads fail under an explicit Dask client (GDAL netrc not on workers) | **yes** — `auth.earthdata.env()` propagated via `odc.stac.configure_rio(client=...)` (§5) |

### E.7 Boundaries, clouds, cameras and hillshade (added 2026-09-23)

| Link | What it is | Why it matters | Incorporate? |
| --- | --- | --- | --- |
| https://www.earthenv.org/cloud | EarthEnv global 1 km cloud frequency (Wilson & Jetz 2016, PLoS Biol 14: e1002415, doi:10.1371/journal.pbio.1002415) | the cloud climatology for F.2; files at `https://data.earthenv.org/cloud/MODCF_<layer>.tif` | **product** (F.2) |
| https://www.earthenv.org/mountains ; https://data.earthenv.org/mountains/standard/GMBA_Inventory_v2.0_standard.zip | GMBA Mountain Inventory v2 on the same host (zip verified 2026-09-23) | mountain-range polygons; same fetch pattern as the cloud files | **product** (F.1 phase B) |
| https://naturalearth.s3.amazonaws.com/10m_raster/GRAY_HR_SR_OB_DR.zip ; https://www.naturalearthdata.com/downloads/10m-raster-data/10m-gray-earth/ | Natural Earth Gray Earth shaded relief (and its siblings) | the hillshade basemap | **product**, shipped as `terrain.hillshade` |
| https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip ; https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip ; https://naciscdn.org/naturalearth/10m/physical/ne_10m_glaciated_areas.zip | Natural Earth countries, states/provinces, glaciated areas (verified) | map outlines worldwide | **product** (F.1 phase A) |
| http://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json | a personal GeoJSON mirror of the Census **2010** 1:5m state boundaries | the URL in use today; convenient but third-party and 16 years old | **no** — use the Census source it mirrors (next row) |
| https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_20m.zip (also `_5m`, `_500k`, `county_*`) | US Census cartographic boundary files (verified) | authoritative US states and counties | **product** (F.1 phase A) |
| https://www.geoboundaries.org/api/current/gbOpen/USA/ADM1/ | geoBoundaries API (verified): JSON pointing to per-country ADM0–ADM5 GeoJSON, CC BY 4.0 | admin levels anywhere in the world | **product** (F.1 phase B) |
| https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3.zip | EPA Level III ecoregions (verified) | US stratification | **product** (F.1 phase C) |
| https://noaa-cdr-patmosx-radiances-and-clouds-pds.s3.amazonaws.com/index.html | PATMOS-x AVHRR cloud climate data record on AWS (bucket reachable) | a 1979-onward cloud record | **shelf** (F.2) |
| `/home/eric/repos/global_snow_networks/scripts/create_all_stations_geojson.py` (`BC_CAMERA_URLS`, `awdb_image_url`) ; `scripts/generate_live_map.py` | where the live map's camera links and photos come from | the starting point for F.3 | **access pattern** (F.3) |
| https://www2.gov.bc.ca/gov/content/environment/air-land-water/water/water-science-data/water-data-tools/snow-survey-data/snow-station-satellite-cameras | BC snow-station satellite camera index | the authoritative list behind `BC_CAMERA_URLS` | **reference** (F.3) |

---

## F. Plans for new modules

### F.1 Boundaries: a new `esd.boundaries` theme

**What Eric asked (2026-09-23):** country and state boundaries, as in

```python
states_gdf = gpd.read_file("http://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json")
world_gdf = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
```

**Where it goes: a new theme subpackage, `easysnowdata/boundaries/`.** One theme is one
subpackage (`catalog._models.KNOWN_THEMES`), and boundaries do not fit an existing one. `hydro`
is watersheds, and a country is not a basin; `terrain` and `land` are rasters. A theme of its
own also gives boundaries their own catalog section, gallery folder and API page. Adding it
takes an entry in `KNOWN_THEMES` / `THEME_TITLES` (after `hydro`), `docs/api/boundaries.rst`,
`docs/gallery/boundaries/`, and the `esd.__init__` import. "`reference`" is the alternative
name if the theme should also hold non-political outlines (mountains, ecoregions, glaciers).
`boundaries` reads better at the call site, and mountain ranges are boundaries too.

**Nothing new is needed underneath.** `providers.vector_http.read` already pushes an AOI down
as `mask=`/`bbox=`, `providers.raster_http.fetch` caches a zip once with a named User-Agent, and
`hydro/basins.py` is the template: a GeoDataFrame in EPSG:4326 of the features that *intersect*
the AOI (whole features, not cut at the edge), with provenance in `.attrs`.

**Proposed API**

```python
import easysnowdata as esd

world_gdf = esd.boundaries.countries()                           # Natural Earth 1:110m, whole world
wa_gdf = esd.boundaries.states(aoi)                              # US → Census; elsewhere → Natural Earth admin-1
bc_gdf = esd.boundaries.states(country="CAN", name="British Columbia")
counties_gdf = esd.boundaries.counties(aoi, state="WA")          # Census, US only
adm2_gdf = esd.boundaries.admin(aoi, level=2, country="NOR")     # geoBoundaries ADM0–ADM5, any country
ranges_gdf = esd.boundaries.mountains(aoi)                       # GMBA v2 (phase B)
lakes_gdf = esd.boundaries.natural_earth("lakes", scale="10m")   # any Natural Earth vector layer
esd.hydro.basins.huc(esd.boundaries.states(name="Washington"), level=4)  # a boundary is an AOI
```

- `scale=` follows Natural Earth (`"110m"` default for `countries()` with no AOI, `"10m"` when an
  AOI is given, `"50m"` between). For Census it maps to `"20m"`/`"5m"`/`"500k"`.
- `name=` / `iso3=` / `country=` filter by attribute after the spatial read. Every returned
  frame carries the same small set of normalized columns: `name`, `iso3`, `admin_level`,
  `source_id`, then the source's own columns.
- `source=` works as everywhere else. Each function is one catalog product
  (`natural-earth-countries`, `us-census-boundaries`, `geoboundaries`, `gmba-mountains`, …) with
  a health probe (`http_first_byte` on the zip; a GET of the geoBoundaries API).
- A small plotting helper, `esd.plotting.add_outline(ax, gdf, crs=None, **style)`, draws
  outlines in the map's CRS. With `terrain.hillshade` it gives a complete context map in three lines.

**Sources and defaults**

| | Natural Earth | US Census cartographic boundaries | geoBoundaries (gbOpen) | GADM 4.1 | Overture `divisions` |
| --- | --- | --- | --- | --- | --- |
| Coverage | world: admin-0 (110m/50m/10m), admin-1 (50m/10m) | US: nation, states, counties, … (500k/5m/20m), yearly | world, per country ADM0–ADM5; CGAZ global composite | world ADM0–ADM5 | world, monthly releases |
| Format / access | zipped shapefile on `naciscdn.org` (verified) | zipped shapefile, `GENZ{year}/shp/cb_{year}_us_{layer}_{res}.zip` (2024 verified) | API → GeoJSON (verified); also GEE `WM/geoLab/geoBoundaries` | GeoPackage download | GeoParquet on S3/Azure (bbox pushdown) |
| Licence | public domain | public domain | CC BY 4.0 | **no redistribution, non-commercial** | ODbL |
| Role | **default** for countries and non-US admin-1 | **default** for US states and counties | **default** for `admin(level=…)` | reference only (licence) | later cloud-native source, once releases can be pinned |

**Two things to watch.** (1) Natural Earth draws disputed borders from one point of view by
default, and publishes per-country point-of-view variants (`ne_10m_admin_0_countries_<iso>`).
The catalog page should say so, and `pov=` can expose the variants. (2) The `eric.clst.org`
URL is a personal mirror of the **2010** Census 1:5m states. Use the Census source it mirrors,
which is current and stable.

**Datasets worth including, beyond countries and states** (ordered by snow relevance):

1. **GMBA Mountain Inventory v2** (Snethlage et al. 2022): mountain-range polygons at several
   hierarchy levels, hosted on EarthEnv (the same host as F.2's cloud files). Useful for "which
   range", per-range summaries, and pairing a mountain definition with the Wrzesien mask.
   **Phase B.**
2. **Glacier outlines**: RGI 7.0 (NSIDC-0770 🔍) for analysis, and Natural Earth
   `glaciated_areas` (verified) for maps. They mask the SNODAS glacier artefact and separate
   snow from ice. Decide whether they belong under `snow` instead. **Phase C.**
3. **US counties and states from Census**: **phase A**, next to Natural Earth.
4. **geoBoundaries ADM1–ADM2**: sub-national units outside the US (Norwegian fylker and
   kommuner for the NVE stations, Canadian provinces for BC/Yukon). **Phase B.**
5. **Ecoregions**: EPA Level III/IV for the US (verified), RESOLVE 2017 globally 🔍. For
   stratifying results. **Phase C.**
6. **Protected areas**: PAD-US for US parks, wilderness and national forests 🔍 (station and
   camera land ownership, access). WDPA globally is non-commercial, so reference only.
   **Phase C.**
7. **Natural Earth physical and cultural layers** (lakes, rivers, coastline, populated
   places), through the generic `natural_earth(layer)`, for map context and labels. **Phase A.**
8. Lakes and rivers for analysis (HydroLAKES, HydroRIVERS) belong in `hydro`, not here.

**Phasing.** A: the theme, `countries`, `states`, `counties`, `natural_earth`, `add_outline`,
`plot_boundaries.py`. B: `admin(level=…)` via geoBoundaries, `mountains()` via GMBA. C:
ecoregions, protected areas, glacier outlines, and an Overture source. Tests follow the static
pattern: tiny zipped-shapefile / GeoJSON fixtures in `tests/static/make_fixtures.py`, a
recorded geoBoundaries API response, and one live smoke test per source.

**Open questions for Eric:** the theme name (`boundaries` or `reference`); whether glacier
outlines live here or in `snow`; whether `states()` should switch to Census automatically for
US AOIs (proposed) or always default to Natural Earth for consistency.

### F.2 Cloud products

Cloud matters to snow work three ways. It is why optical snow products have gaps (the whole
reason for MOD10A1F/VNP10A1F cloud-gap filling). Snow/cloud confusion is the main masking
problem (E.3). And cloud is an energy-balance term in melt. The package already touches
cloud in several places: the MOD10A1/VNP10A1 cloud class, VNP10A1F `Cloud_Persistence`,
S2 SCL and HLS Fmask cloud classes (`processing.masks`), and the Wrzesien mask's `clouds`
indeterminacy layer. What is missing is a **climatology** and a **per-day gridded cloud mask**.

| | EarthEnv cloud frequency | ERA5 cloud cover | MOD09GA / VNP09GA cloud flags | MOD35 / VIIRS CLDMSK L2 | GOES ABI clear-sky mask (ACM) | S2 Cloud Score+ | AVHRR records (PATMOS-x, CLARA-A3, ISCCP-H) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kind | climatology (MODIS 2000–2014, twice daily) | reanalysis, hourly | daily gridded QA bits | swath cloud mask | 5–15 min geostationary mask | per-pixel S2 score | 40-year cloud climate records |
| Resolution | 30″ (~1 km), EPSG:4326 | 0.25° | 1 km / 750 m | 1 km / 750 m swath | 2 km | 10 m | 0.1°–0.25° |
| Access | `data.earthenv.org/cloud/MODCF_<layer>.tif` | ARCO-ERA5 variables `total_cloud_cover`, `low_/medium_/high_cloud_cover` (verified 2026-09-23) | GEE / LPCLOUD (EDL) | LAADS (EDL) | AWS `noaa-goes` | GEE | AWS `noaa-cdr-patmosx-…`, CM SAF, NCEI |
| Licence | **CC BY-NC 4.0** | Copernicus | open | open | open | open (GEE) | open |
| Effort | low | **none**; loadable today | low (bit decoding) | medium (swath gridding) | high (see GOES, E.2) | low (GEE source) | medium |
| Verdict | **follow-on** | **docs** | **follow-on** | shelf | shelf | **follow-on** | shelf |

**EarthEnv (the link Eric flagged): `esd.climate.cloud_frequency.load(aoi, layer="meanannual" | "monthly" | "interannual_sd" | "intraannual_sd" | "seasonality_concentration" | …, month=None)`.**

- Layers: mean annual; 12 monthly means (`MODCF_monthlymean_01` … `_12`); inter- and
  intra-annual variability; 1° spatial variability; seasonality (concentration, θ); and a
  cloud-forest prediction.
- Checked 2026-09-23:
  - 43200 × 21600 uint16, nodata 65535, deflate.
  - **Striped (43200 × 1 blocks), with no tiles or overviews**, so a remote AOI read
    decompresses whole rows.
  - GDAL's `/vsicurl` refuses the host ("Range downloading not supported") **unless
    `CPL_VSIL_CURL_USE_HEAD=NO`**; with that set, range reads work.
- Values are cloud frequency × 100 🔍 (confirm the scaling against the paper before setting
  `scale_factor`).
- The loader should:
  - read remotely with that GDAL option for small AOIs;
  - fetch the whole 0.7–0.8 GB file into the cache for large or repeated use, as the
    Wrzesien loader does;
  - stack `layer="monthly"` into a `month` dimension.
- Put the **non-commercial licence** in the catalog entry and on the page.
- Citation: Wilson, A. M., & Jetz, W. (2016). *PLoS Biology* 14(3): e1002415.

**ERA5 cloud cover** needs no code, only a gallery example, because it is already in the
store `climate.era5` reads. **Per-day masks**: decode MOD09GA `state_1km` and VNP09GA QF cloud
bits in `processing.masks` beside Fmask/SCL. These are the inputs EarthEnv was built from, on
the same grid family as the snow products, and VNP09GA is already Tier 2. **Cloud Score+**
becomes an S2 `mask=` option through the GEE route. The rest stays on the shelf with GOES
(§E.2).

### F.3 Cameras and station imagery

**What exists today (traced 2026-09-23)**:

- **Live camera links (`station_camera_url`): only 8 BC stations, from a hand-curated table.**
  - The table lives only in `global_snow_networks`: `BC_CAMERA_URLS` in
    `scripts/create_all_stations_geojson.py`, applied per BC location id.
  - Each URL is a NuPoint Systems photo-slider page,
    `https://pvs.nupointsystems.com/api/photo-slider-by-nsn?pass=<opaque token>#images-1`. The
    tokens were copied by hand from the gov.bc.ca satellite-camera index (E.7). These are not
    an API and not a scrape.
  - Every other network hard-codes `None`.
  - The BC WFS layer also returns a `CAMERA_URL` attribute. The vendored `databc_client`
    stores it as `camera_url`, but gsn uses it only as a fallback for the *photo* field.
  - `scripts/generate_live_map.py` renders the camera as a link ("🛰 View live satellite
    camera") in the popup and side panel. It is never embedded.
- **Still site photos (`station_image_url`): NRCS SNOTEL/SNTLT (988 stations) and BC ASWS (102).**
  - **NRCS** is a URL template with no request made: `https://www.wcc.nrcs.usda.gov/siteimages/{stationId}.jpg`.
    It is built in `easysnowdata/stations/clients/awdb/awdb_client.py` (`_enrich_awdb_station`)
    and again, duplicated, in gsn's `awdb_image_url()`.
  - **BC** is scraped from the AQUARIUS portal by `DataBCClient.get_station_image_url()`:
    1. accept the disclaimer (a CSRF token, then a POST);
    2. read the numeric location id from the location page;
    3. regex `/Data/GetFileById/{n}` out of the summary page.
  - **Yukon** returns `None`, because AquaCache has no imagery and the explorer is behind
    Cloudflare. **CDEC and NVE** return `None`; photos for them are an open TODO in gsn's
    `docs/UNIFICATION_PLAN.md`.
  - The live map shows the photo as an `<img>` with a credit line.
- **No ski-area, road or other webcams exist anywhere in either repo.** The map's Sentinel-2
  chip is satellite context, not a camera.

**Plan.**

1. **Expose what the clients already know.** `esd.stations.inventory()` gains
   `station_image_url` and `station_camera_url` columns, from the clients for NRCS and BC. The
   BC camera table moves from gsn into the databc client (or a small data file next to it), so
   gsn and easysnowdata read one copy. The duplicate `awdb_image_url()` goes the same way.
   Add `esd.stations.photo(code)`, which returns the image (a PIL image or bytes) with its
   credit, and a gallery example.
2. **CDEC and NVE photos**: investigate as gsn's TODO says. CDEC station pages carry
   photographs 🔍.
3. **Third-party webcams** go in a later `esd.stations.cameras(aoi)` that returns a
   GeoDataFrame of camera points (name, operator, image URL, update cadence, terms), not image
   archives. The candidates, from most to least open:
   - **PhenoCam**, which has an API and a documented archive, is used in the literature for
     snow-on/snow-off, and has mountain sites 🔍;
   - **Caltrans CWWP2** CCTV JSON per district 🔍;
   - **WSDOT** Traveler Information API `HighwayCameras` (free access code; the Snoqualmie and
     Stevens pass cameras) 🔍;
   - **NPS API** `webcams` endpoint (free key; for example the Paradise cameras at Rainier) 🔍;
   - **FAA WeatherCams**, which is dense in Alaska's mountains, with no documented API 🔍;
   - **Windy Webcams API**, an aggregator that includes many ski-area cameras, with a free key
     and terms-limited use 🔍;
   - **ski resorts themselves**, which use vendor pages (Roundshot, Panomax) with no common
     API and terms that usually forbid scraping, so they are **reference only**.
4. Licensing is the constraint, not code. Camera images are mostly not redistributable, so the
   package returns URLs and fetches on demand. Nothing is cached into git or the station archive.

