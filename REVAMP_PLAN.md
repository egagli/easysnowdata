# easysnowdata revamp plan

_Drafted 2026-09-15 from a full read of this repository (code, tests, docs, workflows, issues,
CI history), the companion repos `global_snow_networks` and `snotel_ccss_stations`, and the two
knowledge bases (`geospatial_data_and_visualization_best_practices`, `all_project_memory`).
Items marked **[verify]** rest on memory or second-hand notes and need a live check before they
are acted on. Items marked **[decision]** are collected in §12 for Eric._

---

## 0. Summary

easysnowdata is a ~6 000-line, five-module package with twenty public entry points, published to
PyPI and conda-forge, with a docs site, a weekly data-source health check, and a modest live test
suite. It works, but it has grown by accretion: the module boundaries follow *where code was
first written* rather than *what the data is*; the big classes do search, load, mask, scale, plot
and print in their constructors; credentials are handled four different ways; every test needs
the network; the docs are six hand-executed notebooks totalling 67 MB in git; and several data
routes are stale or about to break (one already breaks CI).

The proposal is a **staged rewrite behind a new package layout**, not a big-bang:

1. **Stabilize** (days): fix the red CI, split offline from live tests, move CI to pixi.
2. **Foundations** (weeks): `aoi` (odc-geo `GeoBox`), `auth` (one provider registry), `catalog`
   (declarative product registry that also drives docs and health checks), `providers` (thin
   STAC / Earth Engine / Earthdata / HTTP-COG / Zarr adapters), logging, no import-time side
   effects.
3. **Migrate products theme by theme** onto the foundations, modernizing each source as it moves
   and writing one offline test file and one gallery example per product. Old names keep working
   through deprecation shims for one release cycle.
4. **Stations**: fold `global_snow_networks/clients` into `easysnowdata.stations` and make
   `global_snow_networks` a consumer.
5. **Docs & automation**: auto-generated gallery and per-product catalog pages; the health check
   grows into a health + upstream-change watch that opens issues instead of silently updating a
   table.
6. **1.0**: freeze the API, refresh the conda-forge feedstock, update the Zenodo record.

The rest of this document is the evidence (§1), the design contract (§2–§3), a source-by-source
modernization table (§4), and detailed designs for credentials (§5), testing (§6), docs (§7),
monitoring (§8), the stations merge (§9), and packaging (§10), followed by the phased roadmap
(§11) and the decisions that need Eric (§12).

---

## 1. Where the repository stands (audit)

### 1.1 Shape of the code

| Module | Lines | Public API | Notes |
| --- | ---: | --- | --- |
| `remote_sensing.py` | 3 487 | 5 functions, 4 classes | Mixes imagery (S1, S2, HLS, MODIS) with static derived layers (snow class, mountain snow mask, forest cover, WorldCover, NLCD). 55 `print()` calls. |
| `hydroclimatology.py` | 1 266 | 8 functions | Mixes reanalysis (ERA5, SNODAS, UCLA SR), vector basins (HUC, HydroBASINS, GRDC ×2) and a climate classification (Köppen-Geiger). |
| `utils.py` | 621 | 12 helpers | Credentials, EE init, bbox parsing, xee grid helper, STAC band configs as YAML strings, water-year helpers, HLS XML parser. |
| `automatic_weather_stations.py` | 417 | 1 class | Reads the frozen `snotel_ccss_stations` CSVs; shells out to `wget` and `tar`. |
| `topography.py` | 181 | 2 functions | Copernicus DEM (PC), CHILI (GEE). |

Twenty entry points cover twenty-one distinct data products from eight hosting routes: Planetary
Computer STAC, Earth Search STAC, NASA CMR-STAC (LPCLOUD), `earthaccess`/CMR, Google Earth Engine
(via `xee` and `ee.data.listFeatures`), anonymous GCS Zarr, plain HTTPS GeoTIFF/zip (Zenodo,
figshare, World Bank, GRDC, a UW Azure blob), and GitHub raw CSV.

### 1.2 Things that are wrong today, concretely

**Broken or about to break**

- **CI has been red on every run since at least June 2026.** The one failing test is
  `get_grdc_wmo_basins`: `https://grdc.bafg.de/downloads/wmobb_json.zip` returns 404. The weekly
  health check has flagged the same URL for at least four weeks, so the monitoring works but
  nothing turns a red row into action (§8).
- The **Sturm & Liston snow classification is served from `uwcryo.blob.core.windows.net`**, a UW
  Azure account whose CloudBank award expires 2026-10-26 and whose custodianship ends with Eric's
  UW appointment on 2026-12-31 (per `all_project_memory/_meta/compute-and-infrastructure.md`).
  This is the only product the package hosts itself and it needs a new home **[decision]**.
- **MODIS snow via Planetary Computer**: Eric's own note (best-practices inbox, item 31) records
  that PC stopped archiving MOD10A2 around June 2025 after Terra's decommissioning; the package
  still defaults `MODIS_snow` to PC. **[verify]** exact status of `modis-10A1-061` / `modis-10A2-061`.
- `HLS` searches `https://cmr.earthdata.nasa.gov/stac/LPCLOUD` **[verify]** whether NASA has moved
  cloud collections to the `cloudstac` root and renamed `HLSL30_2.0` → `HLSL30.v2.0`. Issue #6 was
  an earlier symptom of collection metadata drifting under a hard-coded `stac_cfg`.
- `get_nlcd_landcover` hard-codes `USGS/NLCD_RELEASES/2021_REL/NLCD` **[verify]** whether the
  Annual NLCD release has superseded it.
- `automatic_weather_stations` reads the `snotel_ccss_stations` repo, which Eric describes as
  frozen and superseded by `global_snow_networks` (`_meta/software.md`). The underlying `ulmo`
  SOAP client that built those CSVs no longer installs (best-practices inbox, item 18).

**Design smells that make the above hard to fix**

- **Import-time side effects**: `remote_sensing` enters a global `rasterio.Env` (with a cookie
  jar path under `~`), calls `odc.stac.configure_rio`, and sets `xr.set_options(keep_attrs=True)`
  for the user's whole session; `today` is evaluated once at import and baked into default
  arguments. Importing the package changes the behaviour of unrelated user code.
- **Constructors that do everything**: `Sentinel2(...)`, `Sentinel1(...)`, `HLS(...)`,
  `MODIS_snow(...)` search, load, remove nodata, harmonize, scale, fetch metadata and print in
  `__init__`. There is no way to search without loading, to load without post-processing, or to
  test any step in isolation.
- **Non-serializable attrs**: every categorical product stores a `matplotlib` colormap and a
  bound `example_plot` function in `.attrs` (25 occurrences). `to_netcdf()` / `to_zarr()` on
  these objects fails, and the same ~60-line legend-plotting function is copied five times.
- **Inconsistent output conventions**: dims are `y/x`, `lat/lon`, or `latitude/longitude`
  depending on the function; nodata is sometimes a sentinel with `rio.nodata`, sometimes NaN with
  `encoded_nodata`, sometimes both (`mask_nodata` flag); returns are `DataArray` or `Dataset`
  without a rule; CRS is written via `rio` in some places and inferred by `odc` in others.
- **Credentials handled four ways**: `@requires_*` decorators that only *detect* credentials;
  `initialize_earthengine()` called inside each GEE function with an `initialize_ee` flag;
  `authenticate_all()` living in `remote_sensing`; and GDAL netrc/cookie configuration at import.
  `scripts/check_data_sources.py` re-implements Earth Engine token parsing a second time. The
  `conftest.py` skip logic accepts only username/password while the package also accepts
  `EARTHDATA_TOKEN`, which is why the UCLA test and the UCLA health row have been "skipped" for
  weeks even though CI has an Earthdata token.
- **Subprocess and `/tmp` usage**: `wget`/`tar` for the station archive, `earthaccess.download`
  into `/tmp/local_folder` for MOD10A1F.
- **Declared but unused dependencies**: `contextily`, `folium`, `mapclassify`, `py3dep` are
  required by `pyproject.toml` but never imported by the package (they are notebook conveniences).
  `pyyaml` and `requests` are imported but not declared. The hard dependency on
  `earthengine-api` + `xee` forces Google auth libraries on every user.

**Tests and docs**

- 111 tests; roughly 35 are offline (bbox parsing, water year, token parsing, decorators). The
  other ~75 make live requests to eight providers and take 2.5 minutes on Linux, longer on
  Windows/macOS. There are no tests of masking, scaling, harmonization, dB conversion, class
  tables, the LIA math, the HLS Fmask bit logic, or the station data assembly.
- Docs: `mkdocs-material` + `mkdocstrings` + `mkdocs-jupyter` (`execute: false`). Six notebooks,
  67 MB committed, executed by hand; the remote-sensing notebook alone is 40 MB and covers nine
  products. The API reference is one page per module, so `remote_sensing.md` renders 3 500 lines
  of docstrings. The README "Gallery" is a single GIF hosted as a GitHub user attachment.
- `docs/contributing.md` still describes `setup.py develop`, `flake8`, `tox`, and Python 3.8.

### 1.3 Open issues, mapped to this plan

| Issue | Ask | Where it lands |
| --- | --- | --- |
| #1 develop tests | offline unit tests, datetime integrity for stations | §6 |
| #3 `clip_to_bbox` argument | option to not clip | §2 (AOI contract: `clip=` on every loader) |
| #5 HLS fails with a Dask client | GDAL netrc/cookies not propagated to workers | §5 (auth as a context manager that configures `rasterio.Env` per read, plus `odc.stac.configure_rio(client=...)`) |
| #8 GOES | new provider (AWS NetCDF/Zarr) | §4 backlog; out of scope for the rewrite proper |
| #9 S2/HLS snow cover beyond NDSI | let-it-snow / Theia style algorithms | §4 `snow.snow_cover` backlog |
| #10 Sentinel-1 local incidence angle | now exists via GEE; OPERA static layers supersede it | §4 SAR |
| #11 product wish-list | PALSAR-2, VIIRS, SWOT, 3DEP, NASADEM, PRISM, CONUS404, Sentinel-3, EOPF Zarr, ... | §4 backlog, prioritized **[decision]** |
| #6, #7, #12, #17 (closed) | band alias drift, `write_nodata(encoded=True)` bug, py3.10 import, kwargs pass-through | regressions to pin with tests in §6 |

---

## 2. Design contract

These are the rules every function in the new layout follows. They are short on purpose; the
point is that a contributor (or an agent) can add a product without re-deciding them.

1. **One spatial input, `aoi`.** Accepts a `(west, south, east, north)` tuple in EPSG:4326, a
   shapely geometry, a `GeoDataFrame`/`GeoSeries` in any CRS, or an `odc.geo.GeoBox`. Internally
   everything becomes a `GeoBox` (target grid) plus a `GeoDataFrame` (footprint). `clip=True`
   by default; `clip=False` returns the covering tiles/granules (closes #3).
2. **One temporal input, `time`.** A string or pair accepted by `pandas.to_datetime` / STAC
   datetime syntax (`"2023-10"`, `("2023-10-01", "2024-06-30")`, `"2023-10/2024-06"`).
3. **Lazy by default.** Loaders return Dask-backed `xarray` objects (or lazily-read
   `GeoDataFrame`s where the format allows) and never call `.compute()`. `chunks=` is passed
   through; the default is the source's native chunking where known.
4. **Search and load are separate.** `search_*` returns a `GeoDataFrame` of items/granules
   (STAC-GeoParquet columns where available) that can be inspected, filtered, and handed to
   `load_*`. Nothing prints.
5. **Output conventions.** Dims are `time`, `y`, `x` **[decision]**; CRS is attached with both
   `.rio` and `.odc` accessors (they read the same metadata); nodata is stored as the source's
   sentinel with `rio.nodata` set, and `mask=True` (default for float products, off for
   categorical) converts to NaN with `encoded_nodata` preserved. Categorical products carry
   CF-style `flag_values`, `flag_meanings`, plus `flag_colors` and `long_name` as plain strings;
   nothing in `.attrs` is a Python object. Every product carries `source`, `source_url`,
   `product_id`, `data_citation`, `license`, and `easysnowdata_version` attrs. Units are metric.
6. **Processing is pure.** Masking, scaling, baseline harmonization, dB conversion, indices,
   RGB stretches, and the LIA computation are standalone functions in `easysnowdata.processing`
   that take and return xarray objects and have no I/O. Loaders may *call* them via keyword
   options but the user can always get the raw product.
7. **Plotting is separate and optional.** `easysnowdata.plotting` reads the CF flag attrs to draw
   legends and registers named colormaps. No callables in attrs.
8. **Credentials are lazy and uniform.** A product declares `requires=("earthdata",)`; the
   provider is initialized on first use; failure raises `CredentialError` with the exact setup
   text for that provider (§5).
9. **No import-time side effects.** GDAL/rasterio configuration lives in context managers that
   wrap reads; `xr.set_options` is never called globally; "today" is computed at call time.
10. **Logging, not printing.** `logging.getLogger("easysnowdata")`, INFO for progress, DEBUG for
    URLs, WARNING for fallbacks. A `verbose=` flag is not needed; users set the log level.
11. **Every product has**: a catalog entry (§3.3), at least one offline test on a fixture, a live
    smoke test, a health probe, and a gallery example. The catalog entry is what generates the
    docs page and the health row, so the four stay in sync.
12. **Multiple routes to the same product are first-class.** A product can list several
    `sources` (e.g. Copernicus DEM on Planetary Computer and on AWS Open Data). One is the
    default; the docs page shows the comparison; `source=` selects. Adding a route does not
    change the return contract.

---

## 3. Proposed package layout

### 3.1 Option A (recommended): organize by *what the data is*, with providers underneath

```
easysnowdata/
├── __init__.py            # re-exports: AOI, catalog, auth, and the theme subpackages
├── aoi.py                 # AOI parsing → GeoBox + GeoDataFrame; UTM estimation; grid helpers
├── auth/                  # §5 — providers registry, login(), status(), CredentialError
├── catalog/               # §3.3 — product registry (dataclasses), health probes, docs generators
├── providers/             # thin, generic adapters; no snow knowledge
│   ├── stac.py            #   pystac-client + odc-stac for PC, Earth Search, CMR-STAC, ASF STAC
│   ├── earthdata.py       #   earthaccess search/open/download, GDAL env for EDL-protected COGs
│   ├── gee.py             #   ee.Initialize (once), xee open at native grid, listFeatures → gdf
│   ├── raster_http.py     #   COG / zipped GeoTIFF over HTTPS with optional pooch cache
│   ├── zarr_cloud.py      #   anonymous GCS/S3/Azure Zarr (ARCO-ERA5 and friends)
│   └── vector_http.py     #   remote GDB/GeoJSON/GeoParquet via pyogrio, bbox/mask pushdown
├── processing/            # pure functions: masks, scaling, indices, rgb, db, lia, water year
├── plotting/              # categorical legend, rgb quicklook, colormap registry
├── stations/              # §9 — AWDB, CDEC, DataBC, NVE, Yukon clients + archive reader
├── sar/                   # sentinel1 (PC RTC, OPERA RTC, OPERA static/LIA); nisar, palsar2 later
├── optical/               # sentinel2, hls, landsat, modis (reflectance), viirs
├── snow/                  # snow_cover (MOD10/VNP10, S2/HLS snow), snodas, ucla_sr,
│                          # snow_classification (Sturm & Liston), mountain_snow_mask (Wrzesien)
├── land/                  # landcover (WorldCover, NLCD, Dynamic World), forest_cover
├── terrain/               # dem (Copernicus, 3DEP, NASADEM), indices (CHILI, slope, aspect)
├── climate/               # era5, koppen_geiger; daymet/prism later
└── hydro/                 # basins (HUC, HydroBASINS, GRDC major, GRDC/WMO)
```

Why this shape:

- The user's mental model is "I need a DEM / snow cover / SWE for this box", not "I need the
  remote-sensing module". Derived products stop hiding inside `remote_sensing`.
- `providers/` absorbs the generic ~80% of every loader (open a STAC catalog, sign, search,
  `odc.stac.load`; or open an EE collection at native grid), so each product module shrinks to
  the ~20% that is product knowledge: collection IDs, band aliases, nodata, scale, class tables,
  citation. That is the fix for "functions too specifically tailored": the *generic* entry
  points (`providers.stac.load("sentinel-2-l2a", aoi, time, bands=...)`) are public too, so a
  user can reach any collection on any supported catalog without waiting for a wrapper.
- `processing/` and `plotting/` become testable without network.

### 3.2 Option B: minimal renames

Keep five top-level modules but rename and re-home: `stations`, `imagery` (S1, S2, HLS, MODIS,
VIIRS), `snow` (everything snow-specific), `land_terrain` (land cover, forest, DEM, CHILI),
`climate_hydro` (ERA5, Köppen, basins). Less churn, but `imagery` still mixes optical and SAR,
and there is no natural home for providers/processing. Not recommended, listed for
completeness. **[decision]**

### 3.3 The catalog: one declarative entry per product

```python
@dataclass(frozen=True)
class Source:
    id: str                      # "planetary-computer", "aws-open-data", "nsidc", ...
    provider: str                # key into providers/: "stac", "earthdata", "gee", "raster_http", ...
    location: str                # collection id, EE asset id, URL template, Zarr path
    requires: tuple[str, ...]    # auth providers: (), ("earthdata",), ("earthengine",)
    resolution_m: float | None
    extent: str                  # "global", "CONUS", "western US", ...
    temporal: str | None         # "2014-10/present", "static (2019)", ...
    latency: str | None          # "~2 days", "annual", ...
    notes: str = ""              # what differs from the other sources of this product
    health: Callable[[], None] | None = None   # minimal live probe (§8)

@dataclass(frozen=True)
class Product:
    id: str                      # "copernicus-dem"
    theme: str                   # "terrain"
    title: str
    description: str
    variables: tuple[Variable, ...]     # name, units, dtype, nodata, flags (for categorical)
    sources: tuple[Source, ...]         # first is default
    citation: str
    license: str
    doi: str | None
    references: tuple[str, ...]
    loader: str                  # dotted path of the load function
```

The same registry:

- renders `docs/catalog/<product>.md` (description, sources table with credentials/resolution/
  latency/notes, variables table, citation, health badge, links to gallery examples that use it);
- drives `easysnowdata.catalog.list()` / `.describe("copernicus-dem")` / `.search("swe")` in
  Python;
- drives the health check and its README table (§8);
- is validated by an offline test (every product has a loader that imports, a citation, a
  license, at least one health probe, and every categorical variable has matching
  `flag_values`/`flag_meanings`/`flag_colors` lengths).

### 3.4 API sketch

```python
import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)            # Mount Rainier; any AOI form works

# Catalog
esd.catalog.list(theme="snow")
esd.catalog.describe("sentinel-1-rtc")             # sources, creds, resolution, notes, citation

# Search/load split, lazy, source selectable
items = esd.sar.sentinel1.search(aoi, time="2023-10/2024-06")             # GeoDataFrame
s1 = esd.sar.sentinel1.load(aoi, time="2023-10/2024-06",
                            source="opera-rtc-s1", bands=["VV", "VH"], units="dB")
lia = esd.sar.sentinel1.local_incidence_angle(aoi)                        # OPERA static layer,
                                                                          # GEE/DEM fallback
s2 = esd.optical.sentinel2.load(aoi, time="2024-05", mask="scl-default", harmonize=True)
ndsi = esd.processing.normalized_difference(s2, "green", "swir16")

swe = esd.snow.snodas.load(aoi, time="2024-03", variables=["SWE"], source="nsidc")
dem = esd.terrain.dem.load(aoi, source="copernicus-glo30")                # or "aws-cop30", "3dep"
basins = esd.hydro.basins.load(aoi, level=8, source="usgs-wbd")           # no GEE needed

# Stations (from global_snow_networks clients)
inv = esd.stations.inventory(aoi=aoi, daily_only=True)                     # GeoDataFrame
obs = esd.stations.load(inv.index[:5], variables=["swe", "snwd"],
                        time="2023-10/2024-06")                            # xr.Dataset (station, time)

# Plotting helpers read CF flag attrs; nothing lives in .attrs but strings/numbers
esd.plotting.categorical(esd.snow.snow_classification.load(aoi))
```

Backward compatibility: keep `easysnowdata.remote_sensing`, `hydroclimatology`, `topography`,
`automatic_weather_stations`, `utils` as thin shim modules for one minor release that call the
new functions and emit `DeprecationWarning`. The old classes become factory functions returning
the loaded `Dataset` plus a `.metadata` GeoDataFrame in attrs-free form. **[decision]** whether
to ship shims at all or make 0.1.0 a clean break (the best-practices wiki already tells users to
pin versions and expect signature changes).

---

## 4. Data sources: audit and modernization

Legend for the "status" column: ✅ fine as is · ⚠️ works but stale/suboptimal · ❌ broken ·
🔍 **[verify]** before deciding. The **[verify]** rows are being checked against live endpoints
and current changelogs; this table will be updated when that check completes.

### 4.1 SAR

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Sentinel-1 RTC | PC `sentinel-1-rtc` (10 m, 2014→, no creds, `groupby="sat:absolute_orbit"`, hand-rolled border-noise fix) | ✅ works; single route | Keep as default source. Add **OPERA RTC-S1** (ASF DAAC, 30 m, 2016→ by burst, EDL creds, searchable via ASF STAC / CMR / `earthaccess`, COGs) as a second source with a docs comparison (resolution, geometry, coverage start, credentials, burst vs scene, latency) 🔍. Expose `source=`. Move border-noise / bad-scene logic to `processing.sar`. |
| Local incidence angle | Computed in GEE from `COPERNICUS/S1_GRD` `angle` band + GLO-30 DEM, per relative orbit (issue #10) | ⚠️ works, GEE-only, slow, 350 lines | Primary: **OPERA RTC-S1-STATIC** layers (`local_incidence_angle`, `layover_shadow_mask`, `number_of_looks`) 🔍. Secondary: pure-xarray `processing.sar.local_incidence_angle(dem, incidence, heading)` using the same equations on any DEM (no GEE) — unit-testable on a synthetic slope. Keep the GEE route only if the OPERA static layers do not cover the AOI. Also fold in `generate_sentinel1_local_incidence_angle_maps` if it has anything the two above lack. |
| NISAR, PALSAR-2 | not implemented (#11) | — | Backlog. NISAR GCOV via ASF/`earthaccess` once operational data flows; PALSAR-2 ScanSAR via GEE `JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`. |

### 4.2 Optical imagery

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Sentinel-2 L2A | PC `sentinel-2-l2a` (default) or Earth Search `sentinel-2-l2a` / `sentinel-2-c1-l2a`; own YAML `stac_cfg`; baseline harmonization; SCL masking; RGB percentile/CLAHE | ✅ works | Keep both catalogs as sources. Check whether the raster extension on both catalogs now supplies nodata/scale so the hand-written `stac_cfg` can shrink 🔍. Add **Copernicus Data Space / EOPF Sentinel-2 Zarr** as an experimental third source once the EOPF STAC is stable 🔍. Move harmonization, SCL mask, indices, RGB to `processing`. Keep the existing PC-vs-Earth-Search comparison notebook as a gallery example. |
| HLS L30/S30 v2.0 | CMR-STAC `LPCLOUD`, `HLSL30_2.0`/`HLSS30_2.0`, EDL via GDAL netrc + cookie file at import, per-item XML metadata fetch | ⚠️ endpoint/collection ids drift; #5, #6 | Confirm current CMR-STAC root and ids 🔍; consider PC `hls2-l30`/`hls2-s30` as a no-credential alternative source 🔍 (issue #11 links it). Replace per-item XML scraping with STAC properties. Fmask bit decoding → `processing.optical.decode_fmask`. Auth via `auth.earthdata` context manager so Dask workers inherit GDAL config (fixes #5). |
| Landsat C2 L2 | not implemented | — | Cheap to add via `providers.stac` (PC `landsat-c2-l2`, Earth Search `landsat-c2-l2`). Useful for pre-2015 snow cover. |
| MODIS surface reflectance / VIIRS | not implemented (#11) | — | Backlog; VIIRS `VNP09GA` via GEE or `earthaccess`. |

### 4.3 Snow products

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| MODIS snow cover MOD10A1 / MOD10A2 | PC `modis-10A1-061` / `modis-10A2-061` | ❌ likely stale: PC reportedly stopped archiving MOD10A2 ~2025-06 (Eric's own finding) 🔍 | Default source → **NSIDC via `earthaccess`** (HDF4, needs download + GDAL subdataset open; cache with `pooch`) with PC as historical/optional. Add **VIIRS VNP10A1 / VNP10A1F / VJ110A1** as the successor products 🔍. `get_binary_snow` → `processing.snow.binary_snow(da, product=...)` using the class table. |
| MOD10A1F (cloud-gap-filled) | `earthaccess.download` to `/tmp/local_folder`, `cloud_hosted=False` | ⚠️ works, hard-coded temp dir | Same `earthaccess` route with a proper cache dir; check cloud-hosted status 🔍. |
| SNODAS | GEE community asset `projects/climate-engine/snodas/daily` | ⚠️ works; non-authoritative mirror, GEE creds | Add **NSIDC G02158 direct** (masked/unmasked `.dat.gz` daily tarballs, no credentials) as a second source with a small reader; keep GEE as the fast path. Document the difference (authoritative vs convenient). 🔍 whether any COG/Zarr mirror exists. |
| UCLA WUS snow reanalysis | `earthaccess` cloud-hosted, `open_mfdataset` | ✅ works | Keep; check for newer version / global product 🔍; add `stats=` validation (the current dict maps `"median"` and `"25pct"` to the same index — bug). |
| Sturm & Liston 2021 snow classification | GeoTIFF on UW Azure blob | ❌ hosting expires (see §1.2) | Re-host as a COG (Zenodo record under Eric's name, or a GitHub release asset, or read from NSIDC with EDL) **[decision]**. Class table → CF flags. |
| Wrzesien 2019 mountain snow mask | `zip+https://zenodo…` GeoTIFF | ✅ works, slow (full zip download per call) | Keep source; add `pooch` caching; document the 256/265 nodata quirk as a `processing` fix. |
| Snow cover from S2/HLS (#9) | not implemented | — | Backlog: NDSI + SCL/Fmask thresholds first, let-it-snow-style algorithm later, in `snow.snow_cover`. |

### 4.4 Land cover and vegetation

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| ESA WorldCover v100/v200 | PC `esa-worldcover` | ✅ | Keep. Class table → CF flags. Consider `io-lulc-annual-v02` (PC) and Dynamic World (GEE) as alternative land-cover sources 🔍. |
| NLCD | GEE `USGS/NLCD_RELEASES/2021_REL/NLCD` | ⚠️ likely superseded by Annual NLCD 🔍 | Update asset; check for a no-GEE route (MRLC COGs) 🔍. |
| Forest cover fraction | Zenodo PROBA-V LC100 2019 GeoTIFF | ✅ works | Keep; check newer epoch / COG mirror (PC or GEE `COPERNICUS/Landcover/100m/Proba-V-C3/Global`) 🔍. |

### 4.5 Terrain

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Copernicus DEM GLO-30/90 | PC `cop-dem-glo-30/90` | ✅ | Keep default; add **AWS Open Data `copernicus-dem-30m`** (public S3, no signing) as second source. |
| CHILI | GEE `CSP/ERGo/1_0/Global/ALOS_CHILI`, min-max normalized in the AOI | ⚠️ normalization makes values AOI-dependent | Keep GEE source; **stop rescaling by default** (return native 0–255 or documented 0–1) with `normalize=` opt-in; add a local `processing.terrain.heat_load_index(dem)` implementation later. |
| 3DEP, NASADEM | `py3dep` is a declared dependency but unused | — | Add 3DEP via PC `3dep-seamless` or `py3dep`; NASADEM via PC `nasadem`. Drop the unused dependency until then. |
| Slope/aspect/hillshade | — | — | `processing.terrain` on any DEM (`xdem`/`xrspatial` or numpy). Needed by the LIA fallback. |

### 4.6 Climate and reanalysis

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| ERA5 hourly | ARCO-ERA5 Zarr on GCS (`full_37-1h-0p25deg-chunk-1.zarr-v3`) | ✅ works | Keep; verify current path and end date/latency 🔍; note that it lags the CDS. |
| ERA5 / ERA5-Land daily/monthly | GEE `ECMWF/ERA5*` collections via xee | ✅ works, GEE creds | Keep; investigate a no-GEE ERA5-Land route (ECMWF ARCO access notebook linked in #11) 🔍. |
| Köppen-Geiger (Beck 2023) | figshare zip GeoTIFF | ✅ | Keep; `pooch` cache; CF flags. |
| Daymet, PRISM, gridMET, CONUS404 | — (#11) | — | Backlog: Daymet is on PC as Zarr (`daymet-daily-na`); PRISM on GEE. |

### 4.7 Hydrography

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| HUC boundaries | GEE `USGS/WBD/2017/HUC*` via `ee.data.listFeatures` | ⚠️ works, needs GEE for a public vector dataset | Default → **USGS WBD ArcGIS REST** (`hydro.nationalmap.gov/arcgis/rest/services/wbd`) or `pynhd.WaterData("wbd08")` from HyRiver; keep GEE as a source. The module already contains a commented sketch of the REST call. |
| HydroBASINS / BasinATLAS | figshare `BasinATLAS_Data_v10.gdb.zip` read with `mask=` | ✅ works, large download | Keep; check for a GeoParquet mirror; note GEE `WWF/HydroATLAS/v1/Basins/level0X` as alternative 🔍. |
| GRDC major river basins | World Bank zip | ✅ | Keep. |
| GRDC / WMO basins | `grdc.bafg.de/downloads/wmobb_json.zip` | ❌ 404 | Find the new GRDC URL 🔍 or vendor a small GeoParquet copy (license permitting). This is the Phase 0 fix. |

### 4.8 Stations

See §9. Summary: replace the frozen `snotel_ccss_stations` CSV route with the
`global_snow_networks` clients (AWDB REST for SNOTEL/SCAN/snow courses, CDEC, BC DataBC, NVE,
Yukon) plus a fast reader for its pre-downloaded daily archive.

### 4.9 Backlog from issue #11 (not scheduled; ranked when Eric picks) **[decision]**

GOES LST (#8, via `goes-ortho`), VIIRS surface reflectance, SWOT, RADARSAT-1, PALSAR-2, Planet,
NASADEM/3DEP, PRISM, CONUS404, Sentinel-3 SYN, MetPy/metloom/hydrocloud gauges, geoBoundaries,
census, EOPF Zarr, GIBS. Each is a catalog entry + provider call once §3 exists.

---

## 5. Credentials

### 5.1 The problem, stated once

Five providers, five mechanisms, and the package currently glues them together in four places:

| Provider | What it needs | Where users store it today | What GDAL needs |
| --- | --- | --- | --- |
| NASA Earthdata (HLS, MODIS, UCLA SR, OPERA, SNODAS-free) | EDL account; `~/.netrc` **or** `EARTHDATA_TOKEN` **or** username+password env | `.netrc` via `earthaccess.login(persist=True)` | `GDAL_HTTP_NETRC=YES` + a writable cookie jar (`GDAL_HTTP_COOKIEFILE/COOKIEJAR`) for `/vsicurl` reads of protected COGs |
| Google Earth Engine (NLCD, CHILI, SNODAS-GEE, ERA5-GEE, HUC-GEE, LIA-GEE) | OAuth (`ee.Authenticate`) **and a registered Cloud project**; or a service-account key JSON; high-volume endpoint for xee | `~/.config/earthengine/credentials`; `EARTHENGINE_TOKEN` (raw/base64 JSON, our own convention) | — |
| Planetary Computer | none; optional `PC_SDK_SUBSCRIPTION_KEY` for higher rate limits | env var | signed hrefs (handled by `planetary-computer`) |
| Earth Search / AWS Open Data / GCS anonymous | none | — | `AWS_NO_SIGN_REQUEST=YES`, `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` |
| NVE HydAPI (Norway stations, from `global_snow_networks`) | free API key | `NVE_API_KEY` | — |
| Copernicus CDS / CDSE (future) | key/token | `~/.cdsapirc` / env | — |

### 5.2 Proposal: `easysnowdata.auth`

```python
import easysnowdata as esd

esd.auth.status()            # table: provider, configured?, how (netrc/env/file), needed by which products
esd.auth.login()             # interactive, only for providers not yet configured; persists per provider's own convention
esd.auth.login("earthengine", project="my-gcp-project")
```

- **One `Provider` class per credentialed service** (`earthdata`, `earthengine`, `planetary_computer`,
  `nve`, later `cdse`), each implementing `detect() -> bool`, `login(interactive=True, persist=True)`,
  `ensure()` (initialize once, idempotent, cached), `env() -> contextmanager` (yields the GDAL /
  rasterio / fsspec configuration needed for reads), and `setup_instructions: str`.
- **Detection order is fixed and documented**: explicit call arguments → environment variables
  → provider's native file (`.netrc`, `~/.config/earthengine/credentials`) → interactive prompt
  only if a TTY is present and `interactive=True`. CI sets env vars; humans use the native files.
  No new config file format of our own unless Eric wants one for the EE project id **[decision]**.
- **Earth Engine**: keep accepting `EARTHENGINE_TOKEN` (service-account JSON or OAuth JSON, raw or
  base64) because CI already depends on it, but also honour `GOOGLE_APPLICATION_CREDENTIALS` and
  require/derive a project id (`EARTHENGINE_PROJECT` → token `project` → credentials file →
  error with instructions). Initialize exactly once per process on the high-volume endpoint;
  delete the per-function `initialize_ee=` arguments.
- **Earthdata**: delegate detection and login to `earthaccess.login(strategy=...)`; the `env()`
  context manager sets `GDAL_HTTP_NETRC`, a cookie jar in the platform cache dir (not `~`), and
  registers the same config with `odc.stac.configure_rio(client=...)` when a Dask distributed
  client exists — this is the fix for #5.
- **Products declare what they need**; the loader calls `auth.ensure(*product.requires)` and
  wraps reads in `auth.env(*product.requires)`. One code path, one error message, one test.
- **`CredentialError`** stays, gains a `.provider` attribute and a docs URL.
- **The health check and the test suite use the same `auth` module** — no second token parser in
  `scripts/`.
- **Documentation**: one "Credentials" page generated from the provider registry (what, why,
  how to get it, env var names, how CI does it), linked from every product page that needs it.

---

## 6. Testing

### 6.1 Three tiers

| Tier | Marker | Network | Runs | Contents |
| --- | --- | --- | --- | --- |
| unit | (none) | no | every push/PR, all OSes | AOI parsing (tuple/geometry/gdf/GeoBox, CRS round-trips, antimeridian), water-year helpers, catalog validation, CF flag tables, `processing.*` on synthetic arrays (SCL/Fmask masks, S2 harmonization offsets, scaling, dB round-trip, indices, RGB stretch shapes/ranges, LIA on a synthetic plane and slope), `auth` detection with monkeypatched env/files, plotting smoke with the Agg backend, deprecation shims |
| recorded | `recorded` | no (cassettes/fixtures) | every push/PR | STAC searches replayed with `pytest-recording` (vcrpy) cassettes; loaders exercised against tiny local COGs/Zarr/GeoParquet fixtures generated by a script in `tests/fixtures/`; Earth Engine calls mocked at the `providers.gee` boundary |
| live | `live` + `requires_earthdata` / `requires_earthengine` | yes | nightly or weekly schedule, `workflow_dispatch`, and on release branches | one smoke test per product source: search returns items, load returns the documented dims/dtype/CRS, a small `.compute()` succeeds |

Skips are driven by `auth.status()`, so a CI runner with an Earthdata token no longer skips
Earthdata tests (today's conftest ignores `EARTHDATA_TOKEN`).

### 6.2 Infrastructure

- CI environment from **pixi** (`prefix-dev/setup-pixi`) instead of pip + third-party GDAL
  wheels; the conda-forge stack is what users get from `conda install easysnowdata` anyway.
- `pytest-xdist` for the live tier; `pytest-cov` with a modest threshold on `processing`,
  `aoi`, `catalog`, `auth` (where 100% offline coverage is realistic).
- Regression tests pinned for the closed issues: band alias drift (#6), `write_nodata(encoded=True)`
  on integer arrays (#7), submodule import (#12), kwargs precedence (#17), `stats` index mapping
  in UCLA SR, GRDC URL resolution (as a live test that also feeds the health check).
- Doc examples are executed by the gallery build (§7), which is itself a test of the public API.

---

## 7. Documentation and gallery

### 7.1 Diagnosis

The notebooks are good teaching material but are the wrong storage format: they are re-run by
hand, each covers many products, and their outputs are committed (67 MB, growing with every
re-run). Rendering them needs credentials the docs workflow does not have, which is why
`execute: false` is set and the site shows whatever was last run locally.

### 7.2 Proposal

- **Keep `mkdocs-material` + `mkdocstrings`** (they work, the theme is good, the workflow is
  green) and add **`mkdocs-gallery`** (the mkdocs port of sphinx-gallery) 🔍 maintenance status.
  Fallback if it is unmaintained: **Sphinx + `pydata-sphinx-theme` + `sphinx-gallery` +
  `myst-nb`**, the pangeo/pydata norm. **[decision]**
- **One small script per product** in `examples/<theme>/plot_<product>.py` (~30 lines: load,
  one plot, one sentence). The gallery tool executes them, captures the figure as the thumbnail,
  writes the rendered page, and offers a downloadable `.ipynb`. The gallery index is generated
  from the folder structure — this is the "automatically generate the Gallery" ask.
- **Longer how-to notebooks** (SNOTEL vs Sentinel-1 backscatter; snow cover vs SNODAS; the
  PC-vs-Earth-Search Sentinel-2 comparison; the CCSS 2023 percent-of-normal story) become
  `examples/howto/plot_*.py` or paired `.md` via jupytext — executed the same way, **no outputs
  in git**.
- **Execution in CI**: a scheduled `docs-build` workflow with the same secrets as the live tests
  executes the gallery and deploys; PR builds run with `run_stale_examples: false` so they only
  rebuild what changed and never need credentials. Rendered outputs live on the `gh-pages`
  branch, not `main`.
- **Catalog pages** generated from the registry (§3.3) with a health badge and the list of
  gallery examples using that product.
- **README gallery GIF** regenerated by a script that tiles the gallery thumbnails (`montage` or
  Pillow) as part of the docs workflow, replacing the user-attachment link.
- **API reference** per subpackage (not per 3 500-line module) with `mkdocstrings` options
  `members_order: source`, `show_signature_annotations: true`.
- Rewrite `installation.md` (pixi/uv/conda, extras), `contributing.md` (pixi tasks, test tiers,
  "how to add a product" checklist driven by the catalog), `faq.md` (credentials, Dask, CRS,
  nodata), and add a "Concepts" page (AOI, laziness, nodata policy, CRS/dims, sources).

---

## 8. Data-source health and upstream-change watch

Keep the mechanism (it caught the GRDC break) and fix what it does with the result.

- **Health probes move into the catalog** (`Source.health`), so adding a product adds its probe.
  The script becomes `easysnowdata.catalog.health.run(...)` with `--only`, `--skip-live`, JSON
  output, and the same rolling `data_status/history.json`.
- **Failures open or update a GitHub issue** (label `data-source`) with the error, the product,
  the last-good date, and a link to the catalog page, and the README/docs badge turns red.
  Two consecutive failures escalate the issue title; recovery closes it. No more silent red rows.
- **Latency probe**: for time-series products, record the latest available `datetime` per source
  each week (e.g. newest PC `sentinel-1-rtc` item, newest ARCO-ERA5 time, newest SNODAS day) and
  chart it on the status page. This is the generalization of Eric's Sentinel-1 next-overpass
  latency tool and would have exposed the MOD10A2 archiving stop immediately.
- **Upstream watch** (the "check forums, changelogs, release notes periodically" ask): a
  `WATCHLIST.toml` listing the fast-moving dependencies (xarray, zarr, odc-stac, odc-geo,
  rioxarray, rasterio/GDAL, pystac-client, planetary-computer, earthaccess, earthengine-api,
  xee, geopandas/pyogrio, dask, icechunk, virtualizarr) and the catalogs we depend on. A weekly
  workflow diffs PyPI/GitHub release feeds and STAC collection metadata (item counts, temporal
  extent) against the last run and opens a single digest issue with links to the changelogs.
  The human/agent step — reading Pangeo Discourse, CarbonPlan/Earthmover posts, earthaccess
  discussions — is written as a **Claude routine prompt** (the pattern `all_project_memory`
  already uses) that reads the digest issue and proposes changes as a PR, never pushes to main.
  **[decision]** GitHub Action only vs Action + routine.
- **Dependabot/Renovate** for the CI actions and the lockfile, so version drift is a PR, not a
  surprise.

---

## 9. Folding in `global_snow_networks`

### 9.1 What exists there

`global_snow_networks/DESIGN.md` already specifies the split this plan needs: layer 1
`clients/` (pure access, dict-record API, five networks, shared `_common.py`, ~7 000 lines with
tests), layer 2 archive pipeline (GeoJSON inventory + per-station daily CSVs + `tar.xz`), layer
3 live map. Its §2 says the clients "will eventually migrate into easysnowdata as-is" and its
§7.1 lists the migration as a future issue blocked on its own Phases 1–2 (now merged).

### 9.2 Options

| Option | Shape | Pros | Cons |
| --- | --- | --- | --- |
| A. easysnowdata imports `global_snow_networks` | pip-installable clients stay there | least code movement | makes a data/map repo a library dependency; two release cadences; the repo's tarballs and GeoJSON come along |
| B. Move `clients/` into `easysnowdata.stations`; `global_snow_networks` pins easysnowdata for its pipeline and map | data access in the library; data + map in the app | matches DESIGN.md §2 and Eric's stated end state ("global snow networks will just utilise easysnowdata"); one place for auth (NVE key), retries, units, tests | one-time move; must keep the dict-record API for the pipeline while adding xarray/GeoDataFrame returns for library users |
| C. Both repos import a third package (`snow_station_clients`) | neutral home | clean separation | a third repo to maintain for one maintainer |

**Recommendation: B.** Concretely:

1. `easysnowdata/stations/clients/` receives `clients/` **verbatim** (history preserved with
   `git subtree` or `git filter-repo`), including tests, keeping the `get_all_stations` /
   `get_data` / `get_metadata` dict-record contract from DESIGN.md §3.4 so the pipeline keeps
   working unchanged after a one-line import change.
2. A thin **adapter layer** `easysnowdata.stations` on top: `inventory(aoi=..., networks=...,
   daily_only=...) -> GeoDataFrame`; `load(station_ids | inventory, variables, time, interval)
   -> xr.Dataset` with `station` and `time` dims, metadata as non-dimension coordinates, water
   year / day-of-water-year coordinates (the design recorded in best-practices inbox item 21:
   sparse station × time, `xvec` geometry coordinate optional).
3. **Archive fast path**: `stations.archive.load(...)` reads `all_snow_stations.geojson` and
   `data/all_station_csvs.tar.xz` from `global_snow_networks` (the pattern in its
   `docs/pull_all_daily_data_into_xarray.md`) for "give me everything daily since 1980" without
   hitting five APIs.
4. `automatic_weather_stations.StationCollection` becomes a deprecation shim over the adapter
   (station codes like `679_WA_SNTL` map to AWDB triplets `679:WA:SNTL`; the six variable names
   map to the DESIGN.md type vocabulary).
5. `global_snow_networks`: replace `from clients import ...` with `from easysnowdata.stations.clients
   import ...`, delete `clients/` and `utils/` (water-year helpers already exist here), pin
   `easysnowdata>=0.2`, keep pipeline + map + data.
6. NVE's `NVE_API_KEY` becomes an `auth` provider (§5).

Sequencing: after §3 foundations exist (so the adapter has `aoi`, `auth`, logging to build on),
before the docs rewrite (so the stations gallery examples are written once).

---

## 10. Packaging, tooling, and release

- **`pyproject.toml`** as the single manifest; move pixi config into `[tool.pixi]` so there is
  one dependency list (pixi resolves conda-forge first, PyPI second). Version from git tags
  (`setuptools_scm` or `hatch-vcs`) — drop `bump-my-version`'s four-file search/replace.
- **Optional extras** to shrink the default install: `easysnowdata[earthengine]` (earthengine-api,
  xee), `[earthdata]` (earthaccess, h5netcdf), `[stations]`, `[plot]` (matplotlib, folium,
  contextily, mapclassify), `[all]`. Core: xarray, rioxarray, odc-stac, odc-geo, pystac-client,
  planetary-computer, geopandas, pyogrio, shapely, pandas, numpy, dask, zarr, gcsfs/s3fs,
  fsspec, pooch, requests, pyyaml (declared this time). **[decision]**
- **Python support** per SPEC 0: 3.11–3.13 today, add 3.14 when the stack has wheels.
- **Lint/format/type**: ruff with a broader rule set (`B`, `SIM`, `PL`, `RUF`, `D` for public
  API), `mypy --strict` on `aoi`, `auth`, `catalog`, `processing` (the pure parts), `codespell`,
  pre-commit kept.
- **Publishing**: PyPI **trusted publishing (OIDC)** in `pypi.yml` instead of a stored password;
  build sdist+wheel with `python -m build`; `git-changelog` kept but generated from conventional
  commits on release, not pushed to `main` from CI with `[skip ci]`.
- **conda-forge**: feedstock update for the new dependency split; `easysnowdata` metapackage
  pulling `[all]` equivalents.
- **Repo hygiene**: remove `MANIFEST.in` (setuptools reads pyproject), `.editorconfig` keep,
  `CITATION.cff` updated on release via the same tag workflow, `CHANGELOG.md` kept.

---

## 11. Phased roadmap

| Phase | Scope | Exit criteria | Rough size |
| --- | --- | --- | --- |
| **0 Stabilize** | Fix GRDC URL; mark live tests; split `test`/`test-live` pixi tasks; CI runs offline tests on push, live tests weekly; declare `pyyaml`/`requests`; drop unused deps; fix UCLA `stats` mapping; fix conftest to honour `EARTHDATA_TOKEN`; release 0.0.26 | CI green on `main`; README status table has no long-standing red rows without an issue | days |
| **1 Foundations** | `aoi`, `auth`, `catalog`, `providers`, `processing`, `plotting`, logging; no import side effects; deprecation shim mechanism; unit + recorded test tiers; pixi in CI | 100% offline coverage of the new modules; old public API unchanged and still passing live smoke tests | 2–3 weeks |
| **2 Products** | Migrate theme by theme in this order: terrain → land → snow (static) → hydro → climate → optical → SAR → snow (time series). Each product: catalog entry, loader on providers, source modernization from §4, one recorded test, one live smoke test, one gallery script | old modules are shims only; every product has all four artefacts (§2.11) | 4–6 weeks, parallelizable by theme |
| **3 Stations** | §9 steps 1–6 | `global_snow_networks` pipeline runs against `easysnowdata.stations.clients`; `StationCollection` shim passes its old tests | 1–2 weeks |
| **4 Docs & automation** | mkdocs-gallery (or fallback), catalog pages, credentials page, README gallery montage, health→issue, latency probe, watch digest, routine prompt | site builds in CI from examples without committed outputs; weekly digest issue appears | 2 weeks |
| **5 Release 0.1 → 1.0** | remove shims after one minor cycle; conda-forge feedstock; Zenodo version; announce | API frozen; docs and health green | after 2–3 months of use |

Phases 1 and 2 can start on a `revamp/` branch while 0 ships from `main`. Because each product
migration is independent once Phase 1 lands, this is well suited to parallel agent sessions
with the catalog entry as the shared contract.

---

## 12. Decisions needed from Eric

1. **Layout**: Option A (theme subpackages + providers) or B (minimal renames)? (§3)
2. **API style**: module-level `search`/`load` functions with `source=` (proposed) vs keeping
   classes (`Sentinel2(...)`) as the primary interface? Name for the generic entry point:
   `esd.load("product-id", ...)` in addition to `esd.<theme>.<product>.load(...)`?
3. **Dims and CRS**: always `time/y/x` (odc convention) even for EPSG:4326 products, or keep
   `latitude/longitude` for geographic grids? Always write CRS with both `.rio` and `.odc`?
4. **Nodata default**: raw sentinel + `rio.nodata` (categorical) and NaN-masked (continuous), or
   one rule for all? (§2.5)
5. **Backward compatibility**: deprecation shims for one release, or clean break at 0.1.0?
6. **Docs tooling**: stay on mkdocs + `mkdocs-gallery`, or move to Sphinx + sphinx-gallery +
   myst-nb (the pangeo norm)? Quarto is the third option if you want `.qmd`.
7. **Earth Engine as optional**: OK to make GEE an extra and to move HUC, LIA, and SNODAS
   defaults to non-GEE routes (WBD REST, OPERA static, NSIDC), keeping GEE as an alternative
   source?
8. **Sturm & Liston hosting**: Zenodo COG under your name, GitHub release asset, or NSIDC with
   Earthdata login? (Zenodo gives a DOI and outlives UW.)
9. **Stations merge**: Option B as recommended? Preserve `clients/` git history via subtree?
   Keep the `snotel_ccss_stations` CSV route alive during transition?
10. **Credential storage**: env vars + native files only (proposed), or also an
    `~/.config/easysnowdata/config.toml` for the EE project id and cache dir?
11. **Source priorities** for new routes: OPERA RTC-S1 (+ static LIA), VIIRS snow, NSIDC SNODAS,
    AWS Copernicus DEM, WBD REST, Landsat C2, 3DEP, Daymet, PC HLS — rank the first five.
12. **Watch mechanism**: GitHub Action digest issue only, or also a scheduled Claude routine
    that reads it and proposes PRs? Who is assigned the issues?
13. **Python versions and packaging**: extras split as proposed? Move pixi config into
    `pyproject.toml`? Trusted publishing (needs a one-time PyPI setting)?
14. **Scope guard**: which issue-#11 items, if any, must land in the rewrite rather than after?
15. **Plotting**: drop the `example_plot`-in-attrs pattern entirely in favour of
    `esd.plotting.categorical(da)` (proposed), or keep a convenience `da.esd.plot()` accessor?

---

## 13. Notes to file in the knowledge bases (proposed, not yet filed)

For `all_project_memory/INBOX.md`:
- easysnowdata CI has been red since ≥ June 2026 on the GRDC/WMO basins URL; the health check
  saw it weekly; plan is to make health failures open issues.
- The Sturm & Liston snow-classification GeoTIFF that easysnowdata serves lives on the `uwcryo`
  blob — same expiry problem as the P3 store; needs re-hosting before 2026-10-26 / 2026-12-31.
- `global_snow_networks` → `easysnowdata.stations` merge plan (§9) as the concrete form of the
  2026-09-15 decision recorded in `_meta/software.md`.

For `geospatial_data_and_visualization_best_practices/TO_BE_INCORPORATED.md`:
- Whatever the live verification of PC/CMR-STAC/OPERA/xee/earthaccess status turns up (§4 🔍
  rows), as dated facts for the `data-access/` pages.
- The credential-provider pattern (§5) once implemented, as a worked example for
  `data-access/earthdata-and-earthaccess.md` and `gee-and-xee.md` (GDAL cookie jar + Dask
  workers is the non-obvious part).
