# Data products, sources, and examples

_Companion to [`REVAMP_PLAN.md`](REVAMP_PLAN.md). Started 2026-09-15 from the existing package,
issue #11's idea dump, issues #8/#9/#10, and a live audit of endpoints that day. This file is
meant to be kept current: when a product is added to the catalog, its row here gets a link; when
a source changes, the comparison here is where the reasoning lives. Access facts are dated; a
row without a date was not checked live._

Contents:

- **A. Products in the rewrite** — what the new package ships, in three tiers
- **B. Products with more than one source** — the differences, and which one is the default
- **C. Evaluation of every idea collected so far** — why each matters, how hard it is, verdict
- **D. Example gallery** — the one-script-per-product examples and the multi-source how-tos

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
| optical | 🟢 HLS L30/S30 v2.0 — **priority 3** | CMR-STAC LPCLOUD `HLSL30_2.0`/`HLSS30_2.0` (EDL) | Planetary Computer `hls2-*` (none, ids unverified) | see B.3 |
| snow | 🟢 MODIS snow cover MOD10A1 / MOD10A2 / MOD10A1F (+ MYD) | NSIDC via `earthaccess` (EDL) | Planetary Computer `modis-10A*-061` (none, archiving reportedly stopped 2025) ; GEE `MODIS/061/MOD10A1` (GEE) | see B.4 |
| snow | 🆕 VIIRS snow cover VNP10A1 / VNP10A1F — **priority 1** | NSIDC via `earthaccess` (EDL) | GEE `NASA/VIIRS/002/VNP10A1F` (GEE, unverified) | successor to MODIS after Terra's end; 375 m |
| snow | 🟢 SNODAS SWE / snow depth — **priority 2** | NSIDC G02158 daily tarballs (none) 🆕 | GEE `projects/climate-engine/snodas/daily` (GEE) | see B.5 |
| snow | 🟢 UCLA Western US snow reanalysis | NSIDC `WUS_UCLA_SR` v1 (EDL) | — ; sibling 🆕 `HMA_SR_D` v1 (EDL) | fix `stats=` index bug; explicit login |
| snow | 🟢 Sturm & Liston seasonal snow classification | NSIDC-0768 (EDL) | hosted COG (none; needs a new home) | see B.6 |
| snow | 🟢 Wrzesien mountain snow mask (+ clouds layer) | Zenodo 2626737 zip (none) | — | `pooch` cache |
| land | 🟢 ESA WorldCover v100/v200 | Planetary Computer `esa-worldcover` (none) | AWS `esa-worldcover` bucket (none) | see B.7 |
| land | 🟢 NLCD | GEE `USGS/NLCD_RELEASES/2021_REL/NLCD` (GEE) | 🆕 Annual NLCD community asset (GEE) | see B.7 |
| land | 🟢 Forest cover fraction (CGLS-LC100 2019) | Zenodo 3939050 GeoTIFF (none) | GEE `COPERNICUS/Landcover/100m/Proba-V-C3/Global` (GEE) | no newer epoch exists |
| terrain | 🟢 Copernicus DEM GLO-30/90 | Planetary Computer `cop-dem-glo-*` (none) | AWS `copernicus-dem-30m/90m` (none); Earth Search `cop-dem-glo-*` (none); GEE `COPERNICUS/DEM/GLO30_2024_1` (GEE) | see B.8 |
| terrain | 🟢 CHILI | GEE `CSP/ERGo/1_0/Global/ALOS_CHILI` (GEE) | computed heat-load index from a DEM (none, later) | stop AOI-relative rescaling by default |
| climate | 🟢 ERA5 hourly | ARCO-ERA5 on GCS (none) | Earthmover Icechunk ERA5 (none); NCAR `nsf-ncar-era5` (none); GEE (GEE) | see B.9 |
| climate | 🟢 ERA5 / ERA5-Land daily & monthly | GEE `ECMWF/ERA5*` (GEE) | CDS ARCO Zarr data lake (CDS token, beta) | see B.9 |
| climate | 🟢 Köppen-Geiger (Beck 2023) | figshare file 61012822 (none) | — | switch from stale v1 file; expose `period=` |
| hydro | 🟢 HUC boundaries | USGS WBD ArcGIS REST / HyRiver `pynhd` (none) 🆕 default | GEE `USGS/WBD/2017/HUC*` (GEE) | see B.10 |
| hydro | 🟢 HydroBASINS / BasinATLAS | figshare BasinATLAS gdb (none) | HydroSHEDS per-region zips (none); GEE `WWF/HydroATLAS/v1/Basins/level*` (GEE) | see B.10 |
| hydro | 🟢 GRDC major river basins; GRDC/WMO basins | World Bank zip; `grdc.bafg.de` zip (none) | — | GET-first fetch (server rejects HEAD) |
| stations | 🟢→🆕 Station SWE / snow depth / met | `global_snow_networks` clients: NRCS AWDB REST, CDEC, BC DataBC, NVE HydAPI (key), Yukon AquaCache (none/key) | pre-downloaded daily archive (none) | replaces the frozen `snotel_ccss_stations` CSVs |

### Tier 2 — cheap follow-ons (catalog entry + provider call; each a small PR)

| Theme | Product | Source (creds) | Why |
| --- | --- | --- | --- |
| optical | Landsat Collection 2 Level-2 | Earth Search / PC `landsat-c2-l2` (none); USGS `landsatlook` STAC (requester-pays) | pre-2015 snow cover; the USGS catalog also has `landsat-c2l3-fsca` fractional snow cover |
| snow | ASO lidar snow depth / SWE 50 m | NSIDC `ASO_50M_SD`, `ASO_50M_SWE` (EDL, cloud) | the validation dataset for any SWE product; also the subject of P21 |
| snow | AMSR daily SWE | NSIDC `AU_DySno` (EDL, cloud) | only passive-microwave SWE in the set; coarse but global and daily |
| snow | ICESat-2 ATL06 / ATL08 | NSIDC v007 (EDL, cloud) | snow depth by differencing; `icepyx`/`sliderule` exist — wrap, don't rewrite |
| snow | Sentinel-2 / HLS snow cover (issue #9) | computed in `processing` | NDSI + SCL/Fmask first; let-it-snow / Theia style later |
| terrain | 3DEP 1/3″ seamless; 3DEP lidar | `s3://prd-tnm` (none); PC `3dep-seamless`; EPT STAC | US 10 m DEM; `py3dep` already a (currently unused) dependency |
| terrain | NASADEM | LPCLOUD `NASADEM_HGT_001` (EDL); PC `nasadem`; GEE | 30 m, void-filled SRTM heritage |
| terrain | slope / aspect / hillshade | computed from any DEM | needed by the DEM-based incidence-angle fallback anyway |
| land | Dynamic World; Hansen GFC 2025 v1.13 | GEE (GEE) | annual/near-real-time land cover; forest loss year |
| climate | Daymet V4R1 | ORNL `Daymet_Daily_V4R1` (EDL, cloud); PC `daymet-daily-na` Zarr (none) | 1 km daily North America met, the standard forcing for snow models |
| climate | gridMET; PRISM; NLDAS-2 | GEE `IDAHO_EPSCOR/GRIDMET`, `OREGONSTATE/PRISM/AN81d`, `NASA/NLDAS/FORA0125_H002` (GEE); PC `gridmet` Zarr | CONUS met forcings and normals |
| climate | GPM IMERG daily | GES DISC `GPM_3IMERGDF` v07 (EDL, cloud) | global precipitation where no gauge exists |
| stations | Canada MSC GeoMet; Swiss SLF IMIS | OGC API / REST (none) | networks already scoped in `global_snow_networks` issues #2–#22 |

### Tier 3 — on the shelf (see C for why)

GOES LST (#8), SWOT, RADARSAT-1, PALSAR-2, NISAR GCOV, Planet, CONUS404, HRRR, Sentinel-3 SYN,
SMAP, MODIS albedo (MCD43), SnowEx campaign data, stream gauges, geoBoundaries, census, GIBS,
NOAA climate indices, EOPF Zarr.

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
official NASA product or the 2016-onward burst archive. Sentinel-1 latency across providers is
already measured live by Eric's next-overpass tool; link to it from the catalog page.

### B.2 Sentinel-2 L2A

| | Planetary Computer `sentinel-2-l2a` | Earth Search `sentinel-2-l2a` | Earth Search `sentinel-2-c1-l2a` (+ `sentinel-2-pre-c1-l2a`) | CDSE / EOPF Zarr |
| --- | --- | --- | --- | --- |
| Baseline handling | raw ESA values; **post-2022-01-25 offset must be undone by the client** (current `harmonize_to_old`) | Element 84 already harmonizes | Collection-1 reprocessing; items carry `raster:bands` `offset: -0.1` so the client applies the offset from metadata | native Zarr per product (experimental) |
| STAC metadata | historically no `raster:bands` → needs our `stac_cfg` | `raster:bands` + `eo:bands.common_name` (verified) → no `stac_cfg` | same (verified) | EOPF STAC, in flux |
| Ingestion lag | a 2026-04 PC discussion reports delays | S3 `us-west-2`, no signing | same | — |
| Credentials | none (signing) | none | none | CDSE account + S3 keys |
| Verified | ids only | yes | yes | not checked |

**Default:** Planetary Computer (existing users, one catalog for S1+S2+DEM+WorldCover), with
Earth Search a first-class alternative and the existing PC-vs-Earth-Search notebook kept as the
comparison example. The baseline offset should be applied from `raster:bands` where present and
from the hard-coded date rule only for PC.

### B.3 HLS v2.0

| | NASA CMR-STAC LPCLOUD | Planetary Computer `hls2-l30` / `hls2-s30` |
| --- | --- | --- |
| Ids | `HLSL30_2.0`, `HLSS30_2.0` (+ `*_VI_2.0` vegetation indices) — verified on both `/stac` and `/cloudstac` roots | folder `hls2` exists in `planetary-computer-tasks`; collection ids and freshness unverified |
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
| Format | HDF-EOS2 (HDF4): download, open GDAL subdatasets, date from `.AYYYYDDD.`; not streamable through `fsspec` file objects | COG via STAC | `xee` |
| Credentials | EDL | none | GEE |
| Successor | VNP10A1 / VNP10A1F v2 (375 m) same route | none | `NASA/VIIRS/002/VNP10A1F` (unverified) |

**Default:** NSIDC via `earthaccess`, because it is the archive of record and the only route that
carries the cloud-gap-filled and VIIRS products; PC kept as an optional historical source. This
is the worked example of "a collection disappearing from a cloud catalog is an access-route
failure, not a product failure" already noted in the best-practices inbox.

### B.5 SNODAS

| | NSIDC G02158 direct | GEE `projects/climate-engine/snodas/daily` |
| --- | --- | --- |
| What | authoritative NOHRSC product, masked (CONUS) and unmasked grids, 2003-10 → present | Climate Engine's community re-hosting (SWE, Snow_Depth), ~1-day lag |
| Format | one `.tar` per day of `.dat.gz` + `.txt` header pairs (flat binary, header gives grid) | analysis-ready ImageCollection via `xee` |
| Credentials | none | GEE |
| Cost | one tar per day requested (a few MB); needs a small reader | fast, lazy, spatially subset server-side |

**Default:** NSIDC direct (authoritative, credential-free); GEE kept as the fast path for long
time series. Document that they should agree and provide a gallery example that checks it.

### B.6 Sturm & Liston seasonal snow classification

| | NSIDC-0768 | Hosted COG (today: `uwcryo` Azure blob) |
| --- | --- | --- |
| Formats | GeoTIFF / netCDF / ASCII at 10″, 30″, 2.5′, 0.5° | the 10″ GeoTIFF only |
| Access | HTTPS behind Earthdata Login; not cloud-hosted (0 CMR granules) | anonymous range reads (verified) |
| Longevity | NSIDC | account funding ends 2026-10-26; custodianship ends 2026-12-31 |

**Default (decided):** NSIDC with Earthdata Login; `source="hosted-cog"` as the credential-free
option once the COG has a new home (Zenodo DOI preferred; GitHub release asset as fallback).

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
2024_1 release is only on GEE/CDSE. 3DEP is the US 10 m upgrade path.

### B.9 ERA5 family

| | ARCO-ERA5 (GCS) | Earthmover Icechunk ERA5 | NCAR `nsf-ncar-era5` (AWS) | GEE `ECMWF/ERA5*` | CDS ARCO Zarr (beta) |
| --- | --- | --- | --- | --- | --- |
| Content | hourly ERA5 (37 levels + single levels), 1940 → final 2026-05-31, **ERA5T to 2026-09-09** (verified 2026-09-15) | ERA5 1940–2025, quarterly updates, NRT paid | NetCDF monthly files, 3–4 month lag | ERA5 and **ERA5-Land**, hourly/daily/monthly aggregates | ERA5 single levels and **ERA5-Land hourly** |
| Format | Zarr v2, consolidated | Icechunk | NetCDF | `xee` | Zarr (tokenized) |
| Credentials | none | none | none | GEE | CDS token |
| Note | old `1959-2022` stores still present; AWS `era5-pds` deprecated | anonymous read | — | only GEE route to ERA5-Land today | first ERA5-Land Zarr outside GEE |

Also confirmed 2026-09-15: **WeatherBench2** ERA5 on GCS (1959–2023, anonymous, includes
`snow_depth`; Icechunk copy at `data.icechunk.cloud`) and **DestinE Earth Data Hub** ERA5-Land
Zarr v3 (0.1°, monthly updates, token) — both candidates for `source=` once there is demand.

**Defaults:** ARCO-ERA5 for hourly ERA5; GEE for ERA5-Land and aggregates; CDS ARCO added when
it leaves beta so ERA5-Land no longer requires Earth Engine.

### B.10 Basins

| | USGS WBD REST / HyRiver | GEE `USGS/WBD/2017/HUC*` | BasinATLAS gdb (figshare) | HydroSHEDS per-region zips | GEE `WWF/HydroATLAS/v1/Basins` |
| --- | --- | --- | --- | --- | --- |
| Extent | US | US | global, levels 1–12, with attributes | global, per continent, levels 1–12 | global |
| Size | server-side query | server-side | 2.7 GB download, `mask=` pushdown | ~100 MB per region | server-side |
| Credentials | none | GEE | none | none | GEE |

**Defaults:** WBD REST for HUCs (drops the GEE requirement for a public dataset); BasinATLAS for
attributes with HydroSHEDS regional zips as the lighter route; GEE as alternatives.

---

## C. Evaluation of every idea collected so far

Sources of the list: issue #11 (Eric's idea dump, 2024–2026), issues #8/#9/#10, the 2026-09-15
audit of cloud-hosted snow datasets, and the network scoping in `global_snow_networks`
(issues #2–#22 there). Verdicts: **rewrite** = Tier 1 · **follow-on** = Tier 2 · **shelf** = Tier 3
· **wrap** = point users at an existing package rather than re-implement.

| Idea | What it is | Why it matters for snow science | Access (host · format · creds) | Effort | Verdict |
| --- | --- | --- | --- | --- | --- |
| OPERA RTC-S1 + static layers | NASA/JPL Sentinel-1 RTC, 30 m bursts, with incidence-angle and mask layers | official RTC with the geometry layers wet-snow and melt-timing work needs; replaces the GEE incidence computation | ASF · COG · EDL | medium (CMR-STAC + burst mosaicking) | **rewrite** |
| VIIRS snow VNP10A1 / VNP10A1F | 375 m daily snow cover, cloud-gap-filled variant | MODIS Terra ended; VIIRS is the continuity product for every MOD10-based method | NSIDC · HDF5 · EDL | low once MODIS route exists | **rewrite** (priority 1) |
| SNODAS from NSIDC | authoritative daily SWE/depth grids | removes the GEE dependency from the most-requested SWE product | NSIDC · `.dat.gz` tarballs · none | medium (small reader) | **rewrite** (priority 2) |
| HLS modernization | Landsat + Sentinel-2 harmonized 30 m | the only 30 m harmonized optical record; issues #5, #6 | LPCLOUD · COG · EDL | low | **rewrite** (priority 3) |
| Landsat C2 L2 (+ `landsat-c2l3-fsca`) | 30 m optical back to 1982; USGS fractional snow cover | pre-Sentinel snow cover; fSCA is a ready-made snow product | Earth Search/PC · COG · none; USGS STAC requester-pays | low | **follow-on** |
| ASO lidar SD/SWE | airborne lidar snow depth and SWE, 50 m | the reference dataset for validating SWE/depth | NSIDC · GeoTIFF · EDL (cloud) | low | **follow-on** |
| AMSR `AU_DySno` | passive-microwave daily SWE | only global daily SWE observation | NSIDC · HDF5 · EDL | low | **follow-on** |
| ICESat-2 ATL06/ATL08 | laser altimetry heights | snow depth by differencing | NSIDC · HDF5 · EDL | medium; `icepyx`/`sliderule` exist | **wrap** |
| S2/HLS snow cover (#9) | NDSI-threshold and let-it-snow style snow maps | turns imagery into a snow product; the most-asked feature | computed | medium | **follow-on** (NDSI first) |
| 3DEP 1/3″ and lidar | US 10 m DEM, point clouds | finer terrain for small basins; `py3dep` already a dependency | S3/PC · COG/EPT · none | low | **follow-on** |
| NASADEM | 30 m void-filled SRTM | alternative DEM where GLO-30 has artefacts | LPCLOUD/PC/GEE | low | **follow-on** |
| Dynamic World | 10 m near-real-time land cover | current-year land cover for masking | GEE | low | **follow-on** |
| Hansen GFC 2025 | tree cover and loss year | forest change affects snow interception | GEE | low | **follow-on** |
| Daymet V4R1 | 1 km daily met, North America | standard snow-model forcing | ORNL cloud (EDL) / PC Zarr | low | **follow-on** |
| gridMET, PRISM, NLDAS-2 | CONUS met forcings | temperature/precip context for melt | GEE / PC | low | **follow-on** |
| GPM IMERG | global daily precipitation | precipitation where gauges are absent | GES DISC · NetCDF · EDL | low | **follow-on** |
| Canada MSC GeoMet, SLF IMIS | national station networks with open APIs | extends the station inventory beyond the five current clients | OGC API / REST · none | medium each (new client) | **follow-on** (in `global_snow_networks` scope) |
| GOES LST (#8) | geostationary land surface temperature, 5–15 min | melt-refreeze diurnal cycle; Steven Pestana's `goes-ortho` orthorectifies | AWS · NetCDF · none | high (orthorectification, volume) | **shelf / wrap** `goes-ortho` |
| PALSAR-2 ScanSAR | L-band backscatter | L-band melt detection, NISAR proxy (P6) | GEE · `xee` | low via GEE | **shelf** (until NISAR work needs it) |
| NISAR GCOV | L-band RTC | Eric's postdoc focus | ASF · HDF5 · EDL | medium; wait for operational data | **shelf** |
| SWOT | surface water elevation | lakes/rivers, not snow | PO.DAAC | — | **shelf** |
| RADARSAT-1 | historical C-band | pre-2014 SAR | ASF | — | **shelf** |
| Planet | commercial 3 m imagery | licensed; basemap-only API quirks recorded in best-practices inbox | Planet API · key | — | **shelf** |
| CONUS404 | 4 km WRF reanalysis | high-resolution forcings | USGS HyTEST Zarr on OSN (`usgs.osn.mghpcc.org/hytest/conus404/…`, anonymous, verified) | low (already Zarr) | **follow-on** as a Zarr source |
| HRRR | 3 km NWP | high-resolution forcings | `s3://hrrrzarr` — **service ends October 2026** per the AWS registry | — | **skip** |
| NLDAS-3 (beta) | 1 km North America forcing, 2001–2023 | successor to NLDAS-2 | `s3://nasa-waterinsight` NetCDF + kerchunk Parquet + Icechunk (anonymous, verified) | low (already virtualized) | **follow-on** |
| Sentinel-3 SYN | 300 m optical | coarse snow albedo/cover | PC NetCDF | — | **shelf** |
| SMAP `SPL4SMGP` | soil moisture | soil-moisture pulses used in P12 validation | NSIDC · EDL | low | **shelf** (niche) |
| MODIS albedo MCD43A3/A4 | 500 m albedo / NBAR | snow albedo, melt energy | LPCLOUD · HDF · EDL | low | **shelf** |
| SnowEx campaign data | snow pits, GPR, lidar | method development | NSIDC · mixed · EDL | — | **shelf** |
| Stream gauges (hydrocloud, USGS NWIS) | discharge | runoff onset validation (P2–P5) | REST · none | — | **wrap** HyRiver `pygeohydro` |
| MetPy, metloom | met station clients | overlap with the station clients | — | — | **shelf** (metloom is a peer of the AWDB client) |
| geoBoundaries, census, TIGER | admin boundaries, population | P5 basin-population work | GEE / PC | low | **shelf** (P5-specific) |
| NOAA climate indices | ENSO/PDO series | interannual context | PSL text files | trivial | **shelf** |
| GIBS / Worldview | tile services | browse imagery, not analysis | — | — | **shelf** |
| EOPF Sentinel Zarr | ESA's native Zarr Sentinel products | future-proofing S1/S2 access | CDSE · Zarr · CDSE creds | watch | **shelf** (revisit when STAC is stable) |
| xcube-stac | STAC → xcube datacube | alternative loader | — | — | **skip** (odc-stac covers it) |
| StreamJoy, leafmap/maplibre GEE | animation and web-map tooling | docs polish | — | — | belongs in the best-practices repo, not here |

---

## D. Example gallery

One ~30-line script per Tier 1 product under `examples/<theme>/plot_<product>.py` (executed by
sphinx-gallery), plus multi-source how-tos under `examples/howto/`:

- `plot_s1_sources.py` — the same AOI and month from Planetary Computer and OPERA RTC-S1,
  with the OPERA incidence angle alongside (B.1).
- `plot_s2_catalogs.py` — the existing PC-vs-Earth-Search comparison, rewritten on the new API (B.2).
- `plot_modis_viirs_continuity.py` — MOD10A1F and VNP10A1F over one winter (B.4).
- `plot_snodas_sources.py` — NSIDC vs GEE SNODAS agreement check (B.5).
- `plot_station_vs_sar.py` — SNOTEL SWE against Sentinel-1 backscatter for one pillow.
- `plot_snow_cover_vs_snodas.py` — MODIS snow cover vs SNODAS SWE over a basin.
- `plot_ccss_2023.py` — the CCSS percent-of-normal story from the old station notebook.
- `plot_dowy_max_swe_trend.py` — the day-of-max-SWE trend from the old station notebook, using
  the vectorized water-year helpers.

Each how-to doubles as a live integration test of two sources agreeing, and its thumbnail
feeds the README gallery montage.
