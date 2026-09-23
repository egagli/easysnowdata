# easysnowdata revamp plan

_Drafted 2026-09-15 from a full read of this repository (code, tests, docs, workflows, issues,
CI history), the companion repos `global_snow_networks` and `snotel_ccss_stations`, and the two
knowledge bases (`geospatial_data_and_visualization_best_practices`, `all_project_memory`).
Items marked **[verify]** rest on memory or second-hand notes and need a live check before they
are acted on. Items marked **[decision]** were put to Eric; **his answers of 2026-09-15 are
recorded in §12 and applied throughout** — where a section below states a choice without
hedging, that is why. The companion file
[`POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md`](POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md)
holds the product list, the per-product source comparisons, and the evaluation of every
product idea collected so far._

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
  `get_grdc_wmo_basins`. The file is **not gone**: a GET of
  `https://grdc.bafg.de/downloads/wmobb_json.zip` succeeds (verified 2026-09-15, 206,
  `application/zip`, contains `wmobb_basins.json`), but the GRDC server answers **400 to HEAD**.
  GDAL's `/vsicurl` and the health check's `_head_ok()` both start with HEAD, so both report the
  file missing. Fix: fetch with GET (`pooch`/`requests`) and read locally, or set
  `CPL_VSIL_CURL_USE_HEAD=NO` for that read. The weekly health check has flagged the row for at
  least four weeks, so the monitoring works but nothing turns a red row into action (§8).
- The **Sturm & Liston snow classification is served from `uwcryo.blob.core.windows.net`**, a UW
  Azure account whose CloudBank award expires 2026-10-26 and whose custodianship ends with Eric's
  UW appointment on 2026-12-31 (per `all_project_memory/_meta/compute-and-infrastructure.md`).
  This is the only product the package hosts itself. Decided: NSIDC with Earthdata Login
  becomes the default and the hosted COG an option; **the COG stays where it is for now** and
  the catalog notes that its location is likely to change (§12 Q8).
- **MODIS snow via Planetary Computer**: Eric's own note (best-practices inbox, item 31) records
  that PC stopped archiving MOD10A2 around June 2025 after Terra's decommissioning; the package
  still defaults `MODIS_snow` to PC. Live status of `modis-10A1-061` / `modis-10A2-061` could not
  be checked on 2026-09-15 because the whole PC STAC API returned 503 during a scheduled
  maintenance window; **the NSIDC cloud copies of MOD10A1, MOD10A1F, MOD10A2, MYD10A1(F),
  VNP10A1 and VNP10A1F are all confirmed cloud-hosted** under provider `NSIDC_CPRD`, so the
  `cloud_hosted=False` in the current MOD10A1F code is stale too.
- **`earthaccess` ≥ 0.16 (Jan 2026) no longer logs in automatically**; `open()`/`download()`
  require an explicit `earthaccess.login()`. The UCLA snow reanalysis and MOD10A1F loaders never
  call it, and their tests have been skipped in CI for months (see the conftest note below), so
  they are probably broken for anyone on a current `earthaccess`. Also: 0.17 replaced
  `open_virtual_mfdataset` with `virtualize()`, 0.19 (2026-09-03) turned granule methods into
  fields, and **0.18 dropped Python 3.11**, which this package still advertises.
- `HLS` searches `https://cmr.earthdata.nasa.gov/stac/LPCLOUD`. Verified: both the `/stac` and
  `/cloudstac` roots are live and the collection ids are still `HLSL30_2.0` / `HLSS30_2.0`, so
  the search side is fine. Issue #6 was a symptom of asset metadata drifting under a hard-coded
  `stac_cfg`.
- `get_nlcd_landcover` hard-codes `USGS/NLCD_RELEASES/2021_REL/NLCD`. Verified: that is still
  the newest *official* GEE asset; the 1985–2024 Annual NLCD exists only as the community asset
  `projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER`. Not broken, but the annual product
  is what users increasingly want.
- `get_koppen_geiger_classes` reads figshare file `45057352`, which is the **v1** file; the
  article was updated in January 2026 and the current file is `61012822` (gloh2o.org). Still
  downloads, but silently stale.
- `automatic_weather_stations` reads the `snotel_ccss_stations` repo, which Eric describes as
  frozen and superseded by `global_snow_networks` (`_meta/software.md`). The underlying `ulmo`
  SOAP client that built those CSVs no longer installs (best-practices inbox, item 18).

**Design smells that make the above hard to fix**

- **Import-time side effects**: `remote_sensing` enters a global `rasterio.Env` (with a cookie
  jar path under `~`), calls `odc.stac.configure_rio`, and sets `xr.set_options(keep_attrs=True)`
  for the user's whole session (redundant since xarray 2025.11.0 made `keep_attrs=True` the
  default, and harmful if a user had set it to `False`); `today` is evaluated once at import
  and baked into default arguments. Importing the package changes the behaviour of unrelated
  user code.
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
| #11 product wish-list | PALSAR-2, VIIRS, SWOT, 3DEP, NASADEM, PRISM, CONUS404, Sentinel-3, EOPF Zarr, ... | evaluated item by item in `POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md` §C |
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
5. **Output conventions.** Dims are `time`, `y`, `x` for projected grids and `time`,
   `latitude`, `longitude` for geographic (EPSG:4326) grids — the odc-stac convention, and the
   one users expect from ERA5-style data (decided). CRS is always written with both the
   `.rio` and `.odc` accessors (they read the same metadata). Nodata: **categorical products
   keep the source sentinel with `rio.nodata` set; continuous products are NaN-masked with
   `encoded_nodata` preserved** (decided); `mask=` overrides either way. Categorical products carry
   CF-style `flag_values`, `flag_meanings`, plus `flag_colors` and `long_name` as plain strings;
   nothing in `.attrs` is a Python object. Every product carries `source`, `source_url`,
   `product_id`, `data_citation`, `license`, and `easysnowdata_version` attrs. Units are metric.
6. **Processing is pure.** Masking, scaling, baseline harmonization, dB conversion, indices,
   RGB stretches, water-year coordinates, and the LIA computation are standalone functions in
   `easysnowdata.processing` that take and return xarray objects and have no I/O. Loaders may
   *call* them via keyword options but the user can always get the raw product.
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
- Two stack facts shape `providers/stac.py`: `odc-stac` (0.5.3) reads the `proj`, `raster` and
  `eo` extensions automatically, so the hand-written `stac_cfg` YAML is only needed as a
  fallback for catalogs whose items lack `raster:bands` (Planetary Computer Sentinel-2,
  historically); and **`odc-stac` deliberately ignores `raster:bands` `scale`/`offset`**, so
  `processing.scale_offset(ds)` must apply them from the item metadata. Earth Search's
  `sentinel-2-c1-l2a` items already carry `offset: -0.1` (the post-2022 baseline offset), which
  means the harmonization step can be driven by metadata instead of a hard-coded cutoff date
  wherever the catalog provides it.
- `processing/` and `plotting/` become testable without network.
- **Why `optical` and not `multispectral`.** Every sensor in that theme (Sentinel-2, Landsat
  8/9, HLS, MODIS, VIIRS) is multispectral, so the word distinguishes nothing inside the
  package, and it would exclude products that belong with them: thermal-only (GOES LST, Landsat
  TIRS) and hyperspectral (EMIT, PRISMA). `optical` names the sensing modality — passive,
  visible through SWIR and thermal — and pairs with `sar` (active microwave) the way the
  best-practices wiki pairs `optical.md` with `sar.md` while treating `multispectral.md` as a
  property of sensors. The subpackage docstring states this scope; band-math and mask helpers
  live in `processing.optical`. Decided 2026-09-15.

### 3.2 Option B: minimal renames (not chosen)

Keep five top-level modules but rename and re-home: `stations`, `imagery` (S1, S2, HLS, MODIS,
VIIRS), `snow` (everything snow-specific), `land_terrain` (land cover, forest, DEM, CHILI),
`climate_hydro` (ERA5, Köppen, basins). Less churn, but `imagery` still mixes optical and SAR,
and there is no natural home for providers/processing. **Eric chose Option A (2026-09-15).**

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
items_gdf = esd.sar.sentinel1.search(aoi, time="2023-10/2024-06")         # GeoDataFrame
s1_ds = esd.sar.sentinel1.load(aoi, time="2023-10/2024-06",
                               source="opera-rtc-s1", bands=["VV", "VH"], units="dB")
lia_ds = esd.sar.sentinel1.local_incidence_angle(aoi)                     # OPERA static layer,
                                                                          # GEE/DEM fallback
s2_ds = esd.optical.sentinel2.load(aoi, time="2024-05", mask="scl-default", harmonize=True)
ndsi_da = esd.processing.normalized_difference(s2_ds, "green", "swir16")

swe_ds = esd.snow.snodas.load(aoi, time="2024-03", variables=["SWE"], source="nsidc")
ucla_da = esd.snow.ucla_sr.load(aoi, time="1985-10/2021-09", variable="SWE_Post",
                                virtualize="auto", access="auto")       # virtual Zarr over 36 water
                                                                        # years; direct S3 if in us-west-2
dem_da = esd.terrain.dem.load(aoi, source="copernicus-glo30")             # or "aws-cop30", "3dep"
basins_gdf = esd.hydro.basins.load(aoi, level=8, source="usgs-wbd")       # no GEE needed

# Stations (from global_snow_networks clients)
inv_gdf = esd.stations.inventory(aoi=aoi, daily_only=True)                 # GeoDataFrame
obs_ds = esd.stations.load(inv_gdf.index[:5], variables=["swe", "snwd"],
                           time="2023-10/2024-06")                         # xr.Dataset (station, time)

# Plotting helpers read CF flag attrs; nothing lives in .attrs but strings/numbers
esd.plotting.categorical(esd.snow.snow_classification.load(aoi))
```

Backward compatibility (decided): keep `easysnowdata.remote_sensing`, `hydroclimatology`,
`topography`, `automatic_weather_stations`, `utils` as thin shim modules **for one minor
release** that call the new functions and emit `DeprecationWarning`. The old classes become
factory functions returning the loaded `Dataset` plus a `.metadata` GeoDataFrame in attrs-free
form. Shims are removed in the following minor release (§11 Phase 5).

---

## 4. Data sources: audit and modernization

Legend for the "status" column: ✅ fine as is · ⚠️ works but stale/suboptimal · ❌ broken ·
🔍 **[verify]** before deciding. Endpoints, collection ids and versions below were checked live
on 2026-09-15 unless marked 🔍. One caveat applies to every Planetary Computer row: the PC STAC
API was down for scheduled maintenance during the check (HTTP 503 throughout), so PC collection
ids are taken from the `planetary-computer-tasks` dataset folders rather than a live
`/collections` call. Consequence for the design: the `stac` provider should validate collection
ids against `/collections` at first use and fail with a clear message, instead of hard-coding
assumptions (there is, for instance, still **no `sentinel-2-c1-l2a` on PC**; issue #394 there is
open, and a 2026-04 discussion reports Sentinel-2 ingestion lag).

### 4.1 SAR

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Sentinel-1 RTC | PC `sentinel-1-rtc` (10 m, 2014→, no creds, `groupby="sat:absolute_orbit"`, hand-rolled border-noise fix) | ✅ works; single route | Keep as default source. Add **OPERA RTC-S1** as a second source: ASF DAAC, 30 m, burst-based, 2016-04→present, single-band COGs per polarization + HDF5 metadata, Earthdata Login required (HTTPS via `datapool.asf.alaska.edu`, or in-region S3 with ASF temporary credentials). Search via **CMR-STAC** `https://cmr.earthdata.nasa.gov/cloudstac/ASF/collections/OPERA_L2_RTC-S1_V1_1` or `earthaccess` short name `OPERA_L2_RTC-S1_V1` — **not** `stac.asf.alaska.edu` (only two unrelated collections) and not Earth Search or PC. A third, credential-free route exists on **GEE: `OPERA/RTC/L2_V1/S1`** (VV/VH/HH/HV + mask, 30 m, daily; no incidence layers). Docs comparison: 10 m scene-based PC vs 30 m burst-based OPERA, coverage start, credentials, latency. Move border-noise / bad-scene logic to `processing.sar`. |
| Local incidence angle | Computed in GEE from `COPERNICUS/S1_GRD` `angle` band + GLO-30 DEM, per relative orbit (issue #10) | ⚠️ works, GEE-only, slow, 350 lines | Primary: **OPERA RTC-S1-STATIC** (`OPERA_L2_RTC-S1-STATIC_V1_1` on CMR-STAC ASF; `earthaccess` short name `OPERA_L2_RTC-S1-STATIC_V1`, ~370 k granules) which ships `_local_incidence_angle.tif`, `_incidence_angle.tif`, `_mask.tif` (layover/shadow), `_number_of_looks.tif` and the gamma0→beta0/sigma0 factors as COGs, one static granule per burst. Secondary: pure-xarray `processing.sar.local_incidence_angle(dem, incidence, heading)` on any DEM (no GEE), unit-testable on a synthetic slope. Keep the GEE route only as a fallback. PC's `sentinel-1-rtc` has no incidence layer. Fold in `generate_sentinel1_local_incidence_angle_maps` if it adds anything. |
| NISAR, PALSAR-2 | not implemented (#11) | — | Backlog. NISAR GCOV via ASF/`earthaccess` once operational data flows; PALSAR-2 ScanSAR via GEE `JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`. |

### 4.2 Optical imagery

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Sentinel-2 L2A | PC `sentinel-2-l2a` (default) or Earth Search `sentinel-2-l2a` / `sentinel-2-c1-l2a`; own YAML `stac_cfg`; baseline harmonization; SCL masking; RGB percentile/CLAHE | ✅ works | Keep both catalogs as sources. Earth Search v1 verified: `sentinel-2-l2a`, `sentinel-2-l1c`, `sentinel-2-c1-l2a`, `sentinel-2-pre-c1-l2a` (baseline < 05.00, same schema as c1); its items carry `raster:bands` (`nodata 0`, `uint16`, `scale 0.0001`, `offset -0.1`) and `eo:bands.common_name`, so **no `stac_cfg` is needed there** and scale/offset can be applied from metadata (⚠️ but see §13.1 C1: Earth Search `sentinel-2-l2a` has already removed the baseline offset from the pixels while still advertising `offset: -0.1`, so the item property `earthsearch:boa_offset_applied` — not `raster:bands` — decides whether to subtract it). PC still has only `sentinel-2-l2a` (no Collection-1) and could not be checked live; historically its items lack `raster:bands`, so keep the `stac_cfg` fallback for PC only (⚠️ §13.1 C2: the legacy `get_stac_cfg` keyed per-asset entries by band alias, which `odc-stac` never matches, so the overrides silently did nothing — key them by asset name). Add **Copernicus Data Space / EOPF Sentinel-2 Zarr** as an experimental third source once its STAC is stable 🔍 (CDSE needs its own credentials). Move harmonization, SCL mask, indices, RGB to `processing`. Keep the PC-vs-Earth-Search comparison as a gallery example. |
| HLS L30/S30 v2.0 | CMR-STAC `LPCLOUD`, `HLSL30_2.0`/`HLSS30_2.0`, EDL via GDAL netrc + cookie file at import, per-item XML metadata fetch | ⚠️ works; #5, #6 | Ids and both CMR-STAC roots verified current (also `HLSL30_VI_2.0` / `HLSS30_VI_2.0` vegetation-index products). PC has an `hls2` dataset folder in `planetary-computer-tasks` (no-credential alternative) 🔍 collection ids once PC is back. Replace per-item XML scraping with STAC properties. Fmask bit decoding → `processing.optical.decode_fmask`. Auth via `auth.earthdata` context manager so Dask workers inherit GDAL config (fixes #5). |
| Landsat C2 L2 | not implemented | — | Cheap to add via `providers.stac`: Earth Search `landsat-c2-l2` (verified), PC `landsat-c2-l2`, or USGS `https://landsatlook.usgs.gov/stac-server` (requester-pays S3), which also has **`landsat-c2l3-fsca`, a fractional snow cover product**. Useful for pre-2015 snow cover. |
| MODIS surface reflectance / VIIRS | not implemented (#11) | — | Backlog; VIIRS `VNP09GA` via GEE or `earthaccess`. |
| **PlanetScope (PSScene, 3 m, 4/8-band) and SkySat (50 cm)** — Planet Labs | not in the package. A rough `PlanetData` class lives in the untracked `docs/examples/sandbox.ipynb` (Data API v1 quick-search with geometry/date/cloud filters → per-item asset activation and polling → `rioxarray.open_rasterio` on the full-scene GeoTIFF → `clip_box` → `xr.concat` over time, plus a folium preview using the `tiles.planet.com` XYZ endpoint), and six `ortho_analytic_4b` scenes it downloaded (3.6 GB, untracked, 2023-07-01 over the Rainier AOI: 4-band uint16, EPSG:32610, 3 m, COG layout, nodata 0, 555–675 MB each) | 🆕 asked for on 2026-09-15 (§12 Q17); nothing to break yet | Ship in Phase 2 as `optical.planetscope` on a new `providers.planet` built on the **`planet` SDK 3.x** (3.6.0 on PyPI and conda-forge, checked 2026-09-15; sync `Planet()` client with `.data`, `.orders`, `.subscriptions`, `.features`, `.mosaics`). **Search** with the Data API (`pl.data.search(item_types=["PSScene"], search_filter=...)`; return the items as a GeoDataFrame like every other `search_*`). **Load** by default through the **Orders API with the `clip` tool** (plus `harmonize`/`composite` on request), so Planet delivers COGs cut to the AOI: the sandbox route activates and streams whole 600 MB strips to clip a 0.4°×0.3° box, which is what burns an Education & Research quota and is where its `ChunkedEncodingError`s came from. Keep the Data API asset read as `source="data-api"` for single-scene quick looks. Decode the **UDM2** usable-data mask (it has clear/snow/shadow/haze/cloud bands) in `processing.optical.decode_udm2` — the snow band is directly useful. Analysis-Ready PlanetScope (daily, harmonized) and Planetary Variables come through the Subscriptions API and stay Tier 3 until there is a use. Credentials via the `planet` auth provider (§5). Tests: recorded Data API responses for the unit tier; the live smoke test only *searches* (free) behind `requires_planet`; ordering tests are manual because they spend quota. Health probe: an authenticated `GET https://api.planet.com/data/v1/` (no quota). Licence: Planet imagery is not redistributable, so no scene crops go in git unless the E&R licence allows it `[needs Eric]`; the local scenes can seed a synthetic fixture instead. Companion §B.11 compares the access routes. ⚠️ Caveats from the implementation (§13): no Planet capability has been verified against the live API for want of a `PL_API_KEY` (§13.3 G1), and the SDK's `order_request.product()` fetches the bundle spec over the network while *building* a request, so we build the product dict locally (§13.2 S1). |

### 4.3 Snow products

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| MODIS snow cover MOD10A1 / MOD10A2 | PC `modis-10A1-061` / `modis-10A2-061` | ❌ likely stale: PC reportedly stopped archiving MOD10A2 ~2025-06 (Eric's own finding); PC unreachable during the check 🔍 | Default source → **NSIDC via `earthaccess`**: `MOD10A1`, `MOD10A2`, `MYD10A1` v61 are all cloud-hosted (`NSIDC_CPRD`, us-west-2, HDF-EOS/HDF4 → download to a cache, open GDAL subdatasets with `rioxarray`, parse the date from `.AYYYYDDD.`). Keep PC as an optional historical source. Add **VIIRS `VNP10A1` / `VNP10A1F` v2 (375 m)** as the successor products (both cloud-hosted). `get_binary_snow` → `processing.snow.binary_snow(da, product=...)` using the class table. |
| MOD10A1F (cloud-gap-filled) | `earthaccess.download` to `/tmp/local_folder`, `cloud_hosted=False`, no `login()` | ⚠️ stale flags; probably broken on `earthaccess` ≥ 0.16 | Cloud-hosted since the NSIDC migration (3 M granules, `s3://nsidc-cumulus-prod-protected/MODIS/MOD10A1F/61/...`); use `cloud_hosted=True`, explicit login via `auth.earthdata`, a platform cache dir via `pooch`. `MYD10A1F` (Aqua) is available too. |
| SNODAS | GEE community asset `projects/climate-engine/snodas/daily` (Climate Engine, not official catalog, ~1-day lag) | ⚠️ works; non-authoritative mirror, GEE creds | Add **NSIDC G02158 direct** as a second, credential-free source: `https://noaadata.apps.nsidc.org/NOAA/G02158/{masked,unmasked}/YYYY/MM_Mon/SNODAS_YYYYMMDD.tar`, each tar holding `us_ssmv1*.dat.gz` + `.txt.gz` header pairs; a small reader that streams one day's tar and builds the raster from the header. No COG/Zarr mirror exists (verified by search; GEE is the only cloud copy). Document authoritative vs convenient. ⚠️ SNODAS SWE is int16 millimetres and **saturates at 32767 mm over glaciers** (§13.2 S2); values are passed through untouched and the artefact is documented on the loader. |
| UCLA WUS snow reanalysis | `earthaccess` cloud-hosted, `open_mfdataset`, no explicit `login()` | ⚠️ probably broken on `earthaccess` ≥ 0.16 (test skipped for months) | Keep the route (`WUS_UCLA_SR` v1, 27 k NetCDF granules, WY1985–2021; no v2 and no global version exist); add explicit login; add the sibling **`HMA_SR_D` v1** (High Mountain Asia) as a second region; fix the `stats=` mapping bug (`"median"` and `"25pct"` both map to index 2). |
| Sturm & Liston 2021 snow classification | GeoTIFF on UW Azure blob | ⚠️ hosting will change (see §1.2) | Decided: **default source becomes NSIDC-0768 with Earthdata Login** (authoritative; not cloud-hosted, its HTTPS directory redirects to URS, so it goes through `auth.earthdata` and a `pooch` cache), with `source="hosted-cog"` as the credential-free option. **The hosted COG stays on the `uwcryo` blob for now** (Eric, 2026-09-15); its URL is a single catalog entry so moving it later (Zenodo record, GitHub release asset) is a one-line change plus a health-probe run. The catalog page carries a "location likely to change" note. Class table → CF flags. |
| Wrzesien 2019 mountain snow mask | `zip+https://zenodo…` GeoTIFF | ✅ works, slow (full zip download per call) | Zenodo 2626737 verified live (`MODIS_mtnsnow_classes.zip`, `MODIS_snow_classes.zip`, `MODIS_clouds.zip`). Keep source; add `pooch` caching; expose the clouds layer; document the 256/265 nodata quirk as a `processing` fix. |
| Snow cover from S2/HLS (#9) | not implemented | — | Backlog: NDSI + SCL/Fmask thresholds first, let-it-snow-style algorithm later, in `snow.snow_cover`. |

### 4.4 Land cover and vegetation

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| ESA WorldCover v100/v200 | PC `esa-worldcover` | ✅ | Keep. WorldCover ended at v200 (2021); ESA names **Copernicus LCFM** (10 m, 2020 released 2025-06, later years planned, on CDSE) as the successor. Add the public AWS bucket `esa-worldcover` (eu-central-1, `v100/`, `v200/`) as a no-signing second source. Class table → CF flags. Alternatives for *annual* land cover: GEE `GOOGLE/DYNAMICWORLD/V1` (current to today), PC `io-lulc-annual-v02` 🔍. |
| NLCD | GEE `USGS/NLCD_RELEASES/2021_REL/NLCD` | ⚠️ official asset is current but frozen at 2021 | Add **Annual NLCD 1985–2024** via the community asset `projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER` as the default `source="annual"`, keep `2021_REL` for the science products; check MRLC for a direct COG route (no GEE) 🔍. |
| Forest cover fraction | Zenodo PROBA-V LC100 2019 GeoTIFF | ✅ works | Keep (verified live; CGLS-LC100 v3.0.1 has no epoch after 2019; GEE `COPERNICUS/Landcover/100m/Proba-V-C3/Global` covers 2015–2019). Add **Hansen GFC `UMD/hansen/global_forest_change_2025_v1_13`** (GEE) as a tree-cover alternative with change/loss years. |

### 4.5 Terrain

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| Copernicus DEM GLO-30/90 | PC `cop-dem-glo-30/90` | ✅ | Keep default; add **AWS Open Data `copernicus-dem-30m` / `-90m`** (public COG, unsigned listing verified; 2021 release) and **Earth Search `cop-dem-glo-30` / `-90`** (verified) as credential-free sources. Note in docs that the newer **GLO-30 2024_1** release is on GEE (`COPERNICUS/DEM/GLO30_2024_1`) and CDSE (`cop-dem-glo-30-dged-cog`, CDSE credentials). |
| CHILI | GEE `CSP/ERGo/1_0/Global/ALOS_CHILI`, min-max normalized in the AOI | ⚠️ normalization makes values AOI-dependent | Keep GEE source; **stop rescaling by default** (return native values) with `normalize=` opt-in; add a local `processing.terrain.heat_load_index(dem)` implementation later. |
| 3DEP, NASADEM | `py3dep` is a declared dependency but unused | — | 3DEP seamless 1/3″ is public on S3 (`s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/`, verified) and on PC `3dep-seamless`; 3DEP lidar via the public EPT STAC. NASADEM via LPCLOUD `NASADEM_HGT_001` (EDL), PC `nasadem`, or GEE `NASA/NASADEM_HGT/001`. Drop the unused `py3dep` dependency until a loader exists. |
| Slope/aspect/hillshade | — | — | `processing.terrain` on any DEM (`xdem`/`xrspatial` or numpy). Needed by the LIA fallback. |

### 4.6 Climate and reanalysis

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| ERA5 hourly | ARCO-ERA5 Zarr on GCS (`full_37-1h-0p25deg-chunk-1.zarr-v3`) | ✅ works | Verified live: public, Zarr v2 with consolidated metadata, `valid_time_start` 1940-01-01, final ERA5 to 2026-05-31, **ERA5T to 2026-09-09**, `last_updated` 2026-09-15 (so ~1 week latency via ERA5T, ~3 months for final). Keep as default; expose the ERA5/ERA5T boundary in attrs. ⚠️ Open it with `chunks=None` and chunk only *after* subsetting — a `chunks=` argument at open time builds a Dask graph over all 273 variables on the full hourly grid and exhausts memory (§13.2 S3). Alternatives worth listing as sources: **Earthmover Icechunk ERA5** (`s3://earthmover-icechunk-era5/icechunkV2`, anonymous, 1940–2025, quarterly updates), **NCAR ERA5 on AWS** (`s3://nsf-ncar-era5`, NetCDF, 3–4 month lag; the old `era5-pds` is deprecated). |
| ERA5 / ERA5-Land daily/monthly | GEE `ECMWF/ERA5*` collections via xee | ✅ works, GEE creds | Keep. No-GEE ERA5-Land route: the **CDS ARCO Zarr data lake (beta since 2026-06-30)** serves ERA5 single-levels and **ERA5-Land hourly** with a CDS token; ARCO-ERA5 on GCS has no ERA5-Land. Add as `source="cds-arco"` behind a `cds` auth provider once out of beta 🔍. |
| Köppen-Geiger (Beck 2023) | figshare file `45057352` (v1) zip GeoTIFF | ⚠️ stale file id | Switch to the current file **`61012822`** (article updated 2026-01-14; periods 1901–1930 … 1991–2020 plus 2041–2070 / 2071–2099 projections — expose `period=`); fetch via `ndownloader.figshare.com` (the `figshare.com/ndownloader` host returns a bot-challenge page to non-browser clients); `pooch` cache; CF flags. |
| Daymet, PRISM, gridMET, CONUS404, NLDAS, HRRR | — (#11) | — | Backlog. Daymet V4R1 is cloud-hosted at ORNL (`Daymet_Daily_V4R1`, EDL) and on PC as Zarr (`daymet-daily-na` 🔍); gridMET on GEE `IDAHO_EPSCOR/GRIDMET` and PC; PRISM on GEE `OREGONSTATE/PRISM/AN81d`; NLDAS on GEE; HRRR as public Zarr on `s3://hrrrzarr`. |

### 4.7 Hydrography

| Product | Today | Status | Proposal |
| --- | --- | --- | --- |
| HUC boundaries | GEE `USGS/WBD/2017/HUC*` via `ee.data.listFeatures` | ⚠️ works, needs GEE for a public vector dataset | Default → **USGS WBD ArcGIS REST** (`hydro.nationalmap.gov/arcgis/rest/services/wbd`) or `pynhd.WaterData("wbd08")` from HyRiver; keep GEE (`USGS/WBD/2017/HUC02`–`HUC12`, verified, still the newest there) as a source. The module already contains a commented sketch of the REST call. |
| HydroBASINS / BasinATLAS | figshare `BasinATLAS_Data_v10.gdb.zip` (2.7 GB) read with `mask=` | ✅ works, large download | Keep, but use the `ndownloader.figshare.com` host (302 to signed S3; the other host bot-challenges non-browsers). Add **HydroSHEDS direct** (`data.hydrosheds.org/file/HydroBASINS/standard/hybas_<region>_lev01-12_v1c.zip`, per-region, much smaller) and **GEE `WWF/HydroATLAS/v1/Basins/level01`–`12`** (verified) as sources. No GeoParquet mirror found 🔍. |
| GRDC major river basins | World Bank zip | ✅ verified | Keep. |
| GRDC / WMO basins | `grdc.bafg.de/downloads/wmobb_json.zip` | ❌ fails only because the server rejects HEAD | File verified live (GET 206, zip contains `wmobb_basins.json`; siblings `wmobb_shp.zip`, `wmobb_lkp.zip`, `wmobb_tab.zip`). Fix: GET to a `pooch` cache then read, or `CPL_VSIL_CURL_USE_HEAD=NO`; make the health probe GET-first. **This is the Phase 0 fix.** |

### 4.8 Stations

See §9. Summary: replace the frozen `snotel_ccss_stations` CSV route with the
`global_snow_networks` clients (AWDB REST for SNOTEL/SCAN/snow courses, CDEC, BC DataBC, NVE,
Yukon) plus a fast reader for its pre-downloaded daily archive.

### 4.9 Virtualization: where kerchunk / VirtualiZarr / `earthaccess.virtualize()` help, and where they do not

Checked 2026-09-15 against the `earthaccess` 0.19 source, VirtualiZarr 2.7.3 and `virtual-tiff`
0.5 docs, CMR granule metadata, and live `HEAD` probes for `.dmrpp` sidecars. Principle: **we
do not build virtualization infrastructure**; we use what exists where it pays off.

**How `earthaccess.virtualize()` actually behaves (0.17+).** Signature:
`virtualize(granules, *, access="direct", load=False, group="/", concat_dim=None, preprocess=None,
parser="DMRPPParser", reference_dir=None, reference_format="json", parallel="dask", **combine_kwargs)`.
It appends `.dmrpp` to each granule URL; if the sidecar is missing it warns and falls back to
`HDFParser` (a full HDF5 metadata scan per granule). Supported parsers are DMR++, HDF(5),
NetCDF3 and kerchunk JSON/Parquet — **no HDF4 and no TIFF**. `access="direct"` uses NASA's
temporary S3 credentials, which only work **inside us-west-2**; `access="indirect"` uses HTTPS
range requests with an Earthdata bearer token and works anywhere. Multiple granules need a
single `concat_dim` (`combine="nested"`), so tile × time mosaics need `preprocess` or manual
assembly. `load=True` writes kerchunk references to `reference_dir` and opens them with
`engine="kerchunk"` — that directory is a reusable cache as long as the upstream files do not move.

**DMR++ sidecars, checked per collection** (a 303 means present, 404 absent):

| Collection | Format | `.dmrpp` |
| --- | --- | --- |
| `WUS_UCLA_SR` v1, `HMA_SR_D` v1 (NSIDC) | NetCDF-4 | **absent** → HDFParser fallback |
| `MOD10A1` / `MOD10A1F` / `MOD10A2` v61, `MCD43A3` v061 | HDF-EOS2 (HDF4) | **absent**, and HDF4 is not supported by `virtualize()` at all |
| `VNP10A1F` v2 | HDF-EOS5 (`.h5`) | **absent** → HDFParser fallback, needs `group=` |
| `Daymet_Daily_V4R1` (ORNL), `GPM_3IMERGDF` v07 (GES DISC) | NetCDF-4 | **present** (fast path works) |

**Decision matrix for easysnowdata's data types:**

| Data type | Does virtualization help? | What we do |
| --- | --- | --- |
| COGs via STAC (S1 RTC, S2, HLS, WorldCover, DEMs) | **No.** A COG is already a range-readable, tiled, overviewed chunk store; a manifest only saves the header reads odc-stac does anyway and loses GDAL warping. `virtual-tiff`'s own docs say to use stackstac/lazycogs for ad-hoc queries. | `odc-stac` / `rioxarray` |
| Static GeoTIFF over HTTPS (snow class, Köppen, forest cover) | Marginal (one header fetch). | `rioxarray.open_rasterio("/vsicurl/…", chunks=…)` + `pooch` where a download is unavoidable |
| Zipped GeoTIFF on Zenodo (Wrzesien) | **No.** DEFLATE members are not range-readable; VirtualiZarr's `ZippedZarrParser` is STORED-only and `.zarr.zip`-only. | download once, `pooch` cache |
| NetCDF-4 in Earthdata Cloud (UCLA SR, HMA SR) | **Yes, partly.** `virtualize(access="indirect", concat_dim="Day", load=True, reference_dir=cache)` gives a lazily indexed multi-water-year series; cost is one HDF5 scan per granule (no DMR++), amortized by the reference cache. | **exposed as a loader option** — `esd.snow.ucla_sr.load(..., virtualize="auto")`: `"auto"` virtualizes when more than a handful of granules are requested, `True`/`False` force it; `esd.config.cache_dir` holds the references. Also a gallery example (§D of the companion file). `earthaccess.open()` + `open_mfdataset` stays the path for a single water year |
| HDF-EOS2 MODIS snow (MOD10A1/A1F/A2) | **Not today.** `virtualize()` lacks HDF4; VirtualiZarr's `HDF4Parser` (2.7.1) is a kerchunk wrapper, root-group only, no CRS, tested on one fixture. | download + GDAL subdatasets via `rioxarray`; revisit if NSIDC publishes DMR++ or MODIS C7 moves to HDF5 |
| HDF-EOS5 VIIRS (VNP10A1F) | Possible via HDFParser fallback, worthwhile only for long single-tile series. | same as NetCDF-4 row, low priority |
| SNODAS `.dat.gz` tarballs | **No.** | download + small reader |
| Already Zarr/Icechunk (ARCO-ERA5, Earthmover ERA5, CONUS404 on OSN, NLDAS-3, WeatherBench2) | N/A — already cloud-native; just add them as sources. | `xr.open_zarr` / `icechunk` |
| Earth Engine via `xee` | **No.** | `xee` |
| Station APIs | **No.** | clients |

**Existing public virtual or Zarr stores worth wiring in as sources** (verified anonymous
access unless noted): Earthmover Icechunk ERA5 (`s3://earthmover-icechunk-era5/icechunkV2`,
1940–2025, quarterly; no ERA5-Land, no snow depth); WeatherBench2 ERA5 (1959–2023, **includes
`snow_depth`**, GCS and an Icechunk copy at `data.icechunk.cloud`); CONUS404 daily/hourly Zarr
on OSN (`usgs.osn.mghpcc.org/hytest/conus404/…`, no egress fees); NLDAS-3 beta forcing with
kerchunk Parquet and Icechunk stores on `s3://nasa-waterinsight`; NASA EODC virtual Icechunk
for MUR SST and IMERG (`s3://nasa-eodc-public/icechunk/`, but the referenced bytes still need
NASA S3 credentials in us-west-2); DestinE Earth Data Hub ERA5-Land Zarr v3 (token). Do **not**
build on `s3://hrrrzarr` — the registry says the Zarr service ends October 2026. No public Zarr
or kerchunk exists for SNODAS, MODIS/VIIRS snow, PRISM, or any DEM; Pangeo Forge is no longer
developed and kerchunk is in maintenance mode pointing at VirtualiZarr + Icechunk.

**Region-aware access.** NASA's temporary S3 credentials only work from inside AWS
`us-west-2`, so `access="direct"` is a large speed-up there and an error everywhere else. The
package detects its region cheaply and lazily (§5): environment variables and instance files at
import, the EC2 metadata service with a 200 ms timeout on first Earthdata use. Every
Earthdata-backed loader takes `access="auto"` (default): direct S3 in `us-west-2`, HTTPS
otherwise. The same flag drives `earthaccess.open()`/`download()` and `virtualize()`.

**Health check covers virtualization readiness (decided).** For every NetCDF/HDF source the
weekly probe (§8) also `HEAD`s one granule's `.dmrpp` sidecar and records whether it exists and
which parser `virtualize()` would use. When NSIDC starts publishing DMR++ for UCLA SR, HMA SR,
VNP10A1F or the MODIS products, the status page flips and a digest issue says so — that is the
trigger to switch the default to the fast path or (for HDF4) to re-evaluate virtualization.
The probe also checks that any third-party virtual store we list as a source (Earthmover ERA5,
NLDAS-3, CONUS404 on OSN) still opens and reports its latest timestamp.

**If we ever publish our own virtual store (documented, not scheduled).** The Earthmover/CNG
pattern — virtualize an archive once, persist the manifests to a small Icechunk repository,
optionally add `topozarr` multiscales — is real and needs no new infrastructure. What it would
entail for us, taking the UCLA SR archive as the natural first candidate (static, WY1985–2021,
never reprocessed):

1. Run `earthaccess.virtualize(..., load=False)` (or VirtualiZarr directly) over all 27 k
   granules once, in `us-west-2` for speed, with `preprocess` adding the water-year time axis.
2. `vds.vz.to_icechunk(repo)` into a public bucket or a GitHub release asset (Icechunk reads
   over HTTP; the GOES-16 precedent turned 115 TB of files into ~80 GB of manifests for a
   one-off ~$100 and under $2/month).
3. Ship a loader `esd.snow.ucla_sr.load(source="virtual-store")` that opens the repo and calls
   `authorize_virtual_chunk_access` with the user's Earthdata S3 credentials — so readers
   **still need an Earthdata account and must be in `us-west-2`** for `s3://` chunk containers
   (an HTTPS container with a bearer token is unverified in Icechunk today).
4. Add a health probe that compares stored etags/last-modified against CMR so a NASA
   reprocessing (MODIS Collection 7 and the Terra wind-down in late 2026 are the live examples)
   is detected rather than silently returning stale references; rebuild the manifests when it
   fires.

Why it stays out of the rewrite: the credential and region requirements mean it only helps
users already computing in `us-west-2`, and for them `virtualize(load=True, reference_dir=...)`
already gives a private version of the same thing.

### 4.10 Everything else

The full product list (what ships in the rewrite, what is a cheap follow-on, what stays on the
shelf), the per-product comparison of alternative sources, and the item-by-item evaluation of
the idea dump from issue #11 and the 2026-09-15 source audit live in
[`POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md`](POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md).
That file is meant to be maintained alongside the catalog; this plan only fixes the order of
work (§11).

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
| Planet (PlanetScope, SkySat, basemaps) | a Planet account with data access (Eric's is through the Education & Research program `[needs Eric: quota, licence terms]`); the `planet` SDK 3.x accepts an **OAuth2 user session** (`planet auth login`, browser or device-code flow), a **legacy API key** (`PL_API_KEY` for `Auth.from_env()`, `PL_AUTH_API_KEY` for the default session), or an **OAuth2 machine-to-machine client** (`PL_AUTH_CLIENT_ID` + `PL_AUTH_CLIENT_SECRET`); `PL_AUTH_PROFILE` selects a saved profile — all read from the SDK source, 2026-09-15 | `~/.planet.json` (legacy) and `~/.planet/` (OAuth2 profiles) written by the `planet` CLI; env vars in CI (no `PL_API_KEY` secret exists yet `[needs Eric]`) | none — the Orders/Data APIs hand out signed download URLs; nothing to configure in GDAL |
| Copernicus CDS / CDSE (future) | key/token | `~/.cdsapirc` / env | — |

### 5.2 Proposal: `easysnowdata.auth`

```python
import easysnowdata as esd

esd.auth.status()            # table: provider, configured?, how (netrc/env/file), needed by which products
esd.auth.login()             # interactive, only for providers not yet configured; persists per provider's own convention
esd.auth.login("earthengine", project="my-gcp-project")
```

- **One `Provider` class per credentialed service** (`earthdata`, `earthengine`, `planetary_computer`,
  `planet`, `nve`, later `cdse`), each implementing `detect() -> bool`, `login(interactive=True, persist=True)`,
  `ensure()` (initialize once, idempotent, cached), `env() -> contextmanager` (yields the GDAL /
  rasterio / fsspec configuration needed for reads), and `setup_instructions: str`.
- **Compute-region detection (decided 2026-09-15).** `esd.config.region()` answers "am I in
  AWS, and which region" in two stages so §2.9 holds: at import, environment variables
  (`AWS_REGION`, `AWS_DEFAULT_REGION`, `AWS_EXECUTION_ENV`, `ECS_CONTAINER_METADATA_URI`) and
  the DMI/hypervisor files that identify EC2 hosts (`/sys/devices/virtual/dmi/id/product_uuid`
  starting with `ec2`); on first Earthdata use only, the EC2 instance-metadata service
  (IMDSv2, 200 ms timeout, result cached). `EASYSNOWDATA_REGION` overrides both (useful for
  Coiled/Kubernetes workers where IMDS is blocked). If the region is `us-west-2`, Earthdata
  loaders default to `access="direct"` (temporary S3 credentials, no egress), `earthaccess`
  virtualization uses direct S3, and the GDAL environment gets the in-region S3 settings;
  everywhere else HTTPS is used. Users can force either with `access=`.
- **Visibility (decided 2026-09-15).** `import easysnowdata` runs the cheap `detect()` checks
  (environment variables and file existence only — no network, so §2.9 still holds) and, in an
  interactive session (IPython/Jupyter or a TTY), prints one compact line, e.g.
  `easysnowdata 0.1.0 · credentials: Earthdata ✓ (netrc) · Earth Engine ✗ · Planet ✗ · NVE ✗ · compute: local (HTTPS access) — see esd.auth.status()`
  (or `compute: AWS us-west-2 (direct S3 access)`).
  In scripts and CI the same line goes to the `easysnowdata` logger at INFO; `EASYSNOWDATA_QUIET=1`
  silences it. Calling a product whose default source needs missing credentials raises
  `CredentialError` **before any network call**, naming the provider, the setup steps, and any
  credential-free alternative source for that product (`source="hosted-cog"`, `source="gee"`,
  …) — it never falls back silently.
- **Detection order is fixed and documented**: explicit call arguments → environment variables
  → provider's native file (`.netrc`, `~/.config/earthengine/credentials`) → interactive prompt
  only if a TTY is present and `interactive=True`. CI sets env vars; humans use the native files.
  No config file of our own (decided); the EE project id comes from `EE_PROJECT_ID` /
  `EARTHENGINE_PROJECT` or the token/credentials file.
- **Earth Engine**: keep accepting `EARTHENGINE_TOKEN` (service-account JSON or OAuth JSON, raw or
  base64) because CI already depends on it and `geemap` uses the same variable name, but also
  honour Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` / Workload Identity
  Federation, Google's recommended CI route) and require/derive a project id (`EE_PROJECT_ID`
  as `geemap` spells it, with `EARTHENGINE_PROJECT` as an alias → token `project` → credentials
  file → error with instructions; `ee.Initialize` now raises `no project found` without one).
  Initialize exactly once per process on the high-volume endpoint; delete the per-function
  `initialize_ee=` arguments; pass `ee_init_if_necessary=True, ee_init_kwargs=...` to `xee` so
  Dask workers re-initialize themselves.
- **GDAL defaults** applied inside every provider's `env()` via `odc.stac.configure_rio`:
  `cloud_defaults=True`, `GDAL_HTTP_MAX_RETRY` (GDAL's default is 0 retries) with
  `GDAL_HTTP_RETRY_DELAY` and `GDAL_HTTP_RETRY_CODES`, `AWS_NO_SIGN_REQUEST` for public
  buckets, `CPL_VSIL_CURL_USE_HEAD=NO` only where a server rejects HEAD (GRDC). GDAL ≥ 3.13
  also handles 302-on-HEAD and retries range reads on 429/5xx, so the floor matters (rasterio
  1.5 wheels bundle GDAL 3.12; conda-forge has 3.13).
- **Planet**: delegate everything to the `planet` SDK's own auth stack (`planet.Auth.from_user_default_session()`,
  which already applies the SDK's precedence: env vars, then `~/.planet.json` / `~/.planet/`, then
  built-in defaults). `detect()` checks `PL_API_KEY` / `PL_AUTH_API_KEY` / `PL_AUTH_CLIENT_ID` +
  `PL_AUTH_CLIENT_SECRET` and the existence of those files — no network; `login()` runs the SDK's
  interactive OAuth2 flow (`planet auth login`) and never stores a key in a file of our own;
  `ensure()` builds one `planet.Planet` client per process; `env()` is a no-op because the APIs
  return signed URLs. `CredentialError` names the account page and the E&R program. Because every
  order spends quota, the provider also exposes the account's remaining quota when the Quota API
  lands in the SDK (listed as a future API in the SDK README), and loaders refuse to order without
  an AOI.
- **Earthdata**: delegate detection and login to `earthaccess.login(strategy=...)` and call it
  **explicitly** before any `open()`/`download()` (auto-login was removed in `earthaccess`
  0.16; `EARTHDATA_TOKEN` takes precedence over username/password since then, matching what
  this package already documents). The `env()` context manager sets `GDAL_HTTP_NETRC`, a
  cookie jar in the platform cache dir (not `~`), and registers the same config with
  `odc.stac.configure_rio(client=...)` when a Dask distributed client exists — this is the fix
  for #5. Pin `earthaccess>=0.17` and use `virtualize(access="indirect", load=True,
  reference_dir=<cache>)` for long NetCDF-4 time series (UCLA SR, HMA SR; Daymet has DMR++
  sidecars so it is fast) — see §4.9 for why it does not apply to the HDF4 MODIS products or to
  anything served as COGs.
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
| recorded | `recorded` | no (cassettes/fixtures) | every push/PR | STAC searches replayed with `pytest-recording` (vcrpy) cassettes (⚠️ Planetary Computer cassettes need `allow_playback_repeats: True` because `planetary_computer.sign` caches its SAS token per process — §13.2 S4); loaders exercised against tiny local COGs/Zarr/GeoParquet fixtures generated by a script in `tests/fixtures/`; Earth Engine calls mocked at the `providers.gee` boundary |
| live | `live` + `requires_earthdata` / `requires_earthengine` / `requires_planet` | yes | nightly or weekly schedule, `workflow_dispatch`, and on release branches | one smoke test per product source: search returns items, load returns the documented dims/dtype/CRS, a small `.compute()` succeeds |

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
- Two live-tier failures seen in Phase 2b are environment, not code: urllib-based station
  tests fail TLS verification behind a proxy CA, and the legacy live tier OOM-kills on the
  2.7 GB BasinATLAS download (§13.3 G2). Run the live tier file by file until §4.7's
  per-region HydroSHEDS route lands.

---

## 7. Documentation and gallery

### 7.1 Diagnosis

The notebooks are good teaching material but are the wrong storage format: they are re-run by
hand, each covers many products, and their outputs are committed (67 MB, growing with every
re-run). Rendering them needs credentials the docs workflow does not have, which is why
`execute: false` is set and the site shows whatever was last run locally.

### 7.2 Proposal

Two viable tool chains, checked on 2026-09-15:

| | A. Stay on mkdocs | B. Move to Sphinx |
| --- | --- | --- |
| Stack | `mkdocs-material` 9.7 + `mkdocstrings-python` 2.0 + `mkdocs-jupyter` 0.26 (`execute: true`, `execute_ignore` for credentialed notebooks — exactly what `earthaccess` does) | `sphinx` 9 + `pydata-sphinx-theme` 0.21 + **`sphinx-gallery` 0.21** + `myst-nb` 1.4 (jupyter-cache) — what xarray, geopandas, pystac-client, rioxarray and odc-stac use |
| Gallery | `mkdocs-gallery` is **dormant** (0.10.4, no release since 2024-09), so the gallery index and thumbnails would need a ~100-line script of our own (nbconvert to pull the first figure per executed notebook, write `gallery.md`) | sphinx-gallery does it all: executes `plot_*.py`, makes thumbnails, index pages, downloadable `.ipynb`, and **back-references** ("examples using `easysnowdata.snow.snodas.load`") on every API page |
| Incremental execution | `mkdocs-jupyter` has no cache (issue #161) — every build re-executes or nothing does | `run_stale_examples`/`filename_pattern` (gallery) and jupyter-cache (myst-nb) rebuild only what changed |
| Cost | keep the working workflow; write the gallery script | ~2 days to port config, nav, and the API pages |

**Recommendation: B.** The back-references alone deliver half of the catalog-page goal for
free, incremental execution is what makes a credentialed gallery buildable in CI, and it is the
convention users of this stack already know. Option A is acceptable if you prefer the mkdocs
look; the rest of this section applies to either. **Eric chose B (2026-09-15).**

- **One small script per product** in `examples/<theme>/plot_<product>.py` (~30 lines: load,
  one plot, one sentence). The gallery tool executes them, captures the figure as the thumbnail,
  writes the rendered page, and offers a downloadable `.ipynb`. The gallery index is generated
  from the folder structure — this is the "automatically generate the Gallery" ask.
- **Longer how-to notebooks** (SNOTEL vs Sentinel-1 backscatter; snow cover vs SNODAS; the
  PC-vs-Earth-Search Sentinel-2 comparison; the CCSS 2023 percent-of-normal story) become
  `examples/howto/plot_*.py` or paired `.md` via jupytext — executed the same way, **no outputs
  in git**.
- **Execution in CI**: the `docs` workflow runs the full gallery with the same secrets as the
  live tests on every push to `main` and on the weekly schedule, so the pages it deploys always
  show real output (2026-09-22: pushes used to run the credential-free subset and reuse the
  weekly cache, which left a changed credentialed example without output until Sunday); PR
  builds execute only the credential-free subset
  (`filename_pattern` / `execute_ignore`) and reuse cached outputs for the rest, so they never
  need secrets. Rendered outputs live on the `gh-pages` branch (or a build cache artifact), not
  `main`. This matches the norm the stack projects follow: pre-render heavy or credentialed
  notebooks, re-execute a small no-auth subset on every PR.
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
- **Virtualization-readiness probe** (decided): for every NetCDF/HDF source, `HEAD` one recent
  granule's `.dmrpp` sidecar and record `dmrpp: present | absent` and the parser `virtualize()`
  would fall back to; for third-party Zarr/Icechunk sources, open the store and record its
  latest time. A change from absent to present opens a digest issue titled "fast virtualization
  path now available for <product>" (§4.9).
- **Latency probe**: for time-series products, record the latest available `datetime` per source
  each week (e.g. newest PC `sentinel-1-rtc` item, newest ARCO-ERA5 time, newest SNODAS day) and
  chart it on the status page. This is the generalization of Eric's Sentinel-1 next-overpass
  latency tool and would have exposed the MOD10A2 archiving stop immediately.
- **Upstream watch** (the "check forums, changelogs, release notes periodically" ask), as a
  weekly GitHub Action (**Actions only, decided**) driven by a `WATCHLIST.toml`. Each entry has
  a `kind`, a URL or identifier, and optional keywords; the job fetches everything, diffs
  against the snapshot from the previous run (stored in `data_status/watch/`), classifies each
  new item, and opens **one digest issue** with a section per category. The digest is the
  hand-off point; reading and acting on it is a human or ad-hoc agent task.

  *What it looks for* — every new item is tagged with one or more of: **removal / retirement**
  ("decommission", "retire", "end of life", "no longer", "archived"), **deprecation /
  migration** ("deprecat", "superseded", "migrate", "moved to"), **reprocessing / new version**
  ("reprocess", "collection 7", "v2.0", "version", "baseline"), **extent or coverage change**
  ("extended", "now available for", "added years", "expanded"), **stations added or removed**
  (station-count deltas from the `stations` clients), **new dataset** (new collection id, new
  catalog entry, new short name matching our keyword list: snow, SWE, SNOTEL, Sentinel-1,
  Sentinel-2, HLS, MODIS, VIIRS, ERA5, DEM, land cover, reanalysis), **outage / access change**
  ("outage", "maintenance", "credentials", "token", "URS", "S3", "CloudFront"). Items that
  match none are listed at the end under "other", never dropped.

  *Programmatic checks* (no text parsing; these catch changes before any changelog mentions them):
  - **CMR collections** we depend on: `revision_date`, `version_id`, granule count and
    temporal extent per collection (`/search/collections.umm_json?short_name=…`) — a version or
    revision bump is how reprocessing shows up.
  - **STAC collections** (Planetary Computer, Earth Search, CMR-STAC ASF/LPCLOUD): presence in
    `/collections`, `extent.temporal`, and latest item datetime — catches a collection quietly
    disappearing (the MOD10A2 case) or stopping (latency).
  - **GEE assets**: existence and `system:version`/date range of each asset id in the catalog,
    plus the community-catalog entries we use (SNODAS, Annual NLCD).
  - **Static files** (Zenodo, figshare, GRDC, World Bank, the hosted COG): ETag / Last-Modified
    / Content-Length via GET-first.
  - **Station inventories**: counts per network from the clients, so added or retired stations
    (NRCS adds SNOTEL sites every year; Eric's "new snotel station" commit was exactly this) show
    up as a delta.
  - **Dependencies**: PyPI JSON for the watchlist packages (xarray, zarr, odc-stac, odc-geo,
    rioxarray, rasterio/GDAL, pystac-client, planetary-computer, earthaccess, earthengine-api,
    xee, geopandas/pyogrio, dask, icechunk, virtualizarr, asf_search), with links to changelogs.

  *Pages and feeds on the watchlist* (fetched, diffed, keyword-tagged):
  - Google Earth Engine data catalog release notes —
    https://developers.google.com/earth-engine/docs/data-catalog/release-notes
  - Planetary Computer changelog history — https://planetarycomputer.microsoft.com/docs/changelogs/history
    and the PC GitHub discussions/announcements feed
  - NASA Earthdata alerts and outages — https://www.earthdata.nasa.gov/data/alerts-outages
  - NSIDC DAAC data updates — https://nsidc.org/data/data-programs/nsidc-daac/data-updates
    (the canonical URL behind the UW off-campus proxy link)
  - ASF DAAC pages — https://www.earthdata.nasa.gov/centers/asf-daac and
    https://www.earthdata.nasa.gov/centers/asf-daac/data-access-tools ; `asf_search` changelog —
    https://github.com/asfadmin/Discovery-asf_search/blob/master/CHANGELOG.md
  - NRCS Snow Survey and Water Supply Forecasting program —
    https://www.nrcs.usda.gov/programs-initiatives/sswsf-snow-survey-and-water-supply-forecasting-program
    and the AWDB REST API announcements page
  - Added by this plan: LP DAAC news (HLS, NASADEM, MCD43), ORNL DAAC news (Daymet), GES DISC
    news (IMERG), NOHRSC/SNODAS notices, CDEC news, Element 84 Earth Search changelog/blog,
    Copernicus Data Space Ecosystem news (Sentinel reprocessing, EOPF Zarr rollout), ESA
    WorldCover/LCFM news, USGS EROS Landsat news, `gee-community-catalog` changelog, ECMWF
    forum ARCO thread and CDS news, USGS HyTEST catalog commits, JPL OPERA product announcements,
    plus the community forums (Pangeo Discourse categories, earthaccess discussions,
    CarbonPlan and Earthmover blogs).

  *Digest format*: one issue per week titled with the date, sections "Action needed" (removal,
  deprecation, reprocessing, outage affecting a catalog source), "Worth adding" (new datasets
  matching keywords, with a pre-filled row for the companion file's §C table), "Changed"
  (extent, version, station deltas), "Dependency releases", "Other". Each item links to its
  source and to the affected catalog entry. If nothing changed, no issue is opened. Health
  failures (above) post into the same issue so there is one place to look.
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

**Decided: B, with history preserved via `git subtree`, and no transition period for the
`snotel_ccss_stations` CSV route** (the shim in step 4 simply reads from the new clients).
Concretely:

1. `easysnowdata/stations/clients/` receives `clients/` **verbatim** (history preserved with
   `git subtree add --prefix`), including tests, keeping the `get_all_stations` /
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

### 9.3 Which repository answers which request (recorded 2026-09-17)

The rule, in one line: **`global_snow_networks` is an index and a bulk cache;
an observation always comes from the network that made it.** No measurement
reaches a user through that repo except by the explicit archive route, and
nothing in `easysnowdata.stations` treats it as the only way to get an answer.

| Call | Reads `global_snow_networks` | Reads the network APIs |
| --- | --- | --- |
| `stations.load(codes, …)` | station metadata for the coordinates | **every observation** |
| `stations.load(aoi=…)` | the station list for that AOI | every observation |
| `stations.inventory()` | ✅ default (`source="archive"`) | `source="clients"` |
| `stations.archive.load()` | ✅ inventory and daily CSVs | — |
| `automatic_weather_stations.StationCollection` | the station list | every observation |

Why the inventory defaults to the archive rather than the five APIs: it is one
HTTP request instead of five sweeps, its columns are normalized across
networks, and its `daily_or_better` flag is the pipeline's **probe-verified**
verdict rather than what a network advertises about itself (that repo's
DESIGN.md §4 is explicit that advertised capability flags are hints). The live
route exists for "what does the network say *today*" and is one keyword away.

Because a published library should not be able to be broken by a data repo,
every one of those reads degrades rather than failing:

| If `global_snow_networks` is unreachable | What happens |
| --- | --- |
| `load()` metadata coordinates | warns; the Dataset keeps `network` and the data |
| `load()` with an ambiguous code (`"QUA"`) | warns; asks CDEC and DataBC directly |
| `load(aoi=…)` | warns; falls back to `inventory(source="clients")` |
| `load()` with an unambiguous code or `networks=` | unaffected — never consulted |
| `archive.load()` | fails, correctly: that route *is* the archive |

What that repo publishes, and when: its CI rebuilds `all_snow_stations.geojson`
and the per-station CSVs daily and redeploys its Pages site from a build
artefact — since 2026-09-22 with the archive as two chunked Zarr stores in
that artefact (its `docs/STORAGE.md` and DESIGN.md §6.5). `easysnowdata`
reads the inventory over HTTPS at call time; `archive.load()` reads the
store by default, fetching only the chunks a request touches (one water year
of every station 0.7 MB, one station's whole record 0.5 MB, against the
27 MB bundle), and falls back with a warning to the bundle attached to that
repo's latest snapshot release when the store cannot be read, so a Pages
outage degrades to a possibly-stale snapshot rather than failing (Releases
are served from outside Pages). The daily-committed bundle on `main` was
retired once 0.3.2 carried this fallback; 0.2.x and 0.3.1 read that file and
must upgrade.

**What `global_snow_networks` still owes this migration** is tracked in that
repo as [`docs/EASYSNOWDATA_MIGRATION.md`](https://github.com/egagli/global_snow_networks/blob/main/docs/EASYSNOWDATA_MIGRATION.md):
switch 16 import lines across three scripts, delete `clients/*.py`, `utils/`
and the six client test files, rehome the five
`clients/*/<name>_stations.geojson` artefacts (which deleting the code
directory orphans, and which DESIGN.md §6.2 names by path), and pin
`easysnowdata>=0.2`. **All of it is blocked on releasing 0.2** — the pin
cannot resolve before then — so the order is: merge this branch, release,
then that PR.

Keeping the two copies of the clients in step: **superseded 2026-09-17.**
`global_snow_networks` deleted its `clients/` and `utils/` and now imports
these, so there is one copy and it is this one — fix a client here.
`scripts/sync_clients.sh` went with the second copy. The two client-contract
tests that repo had added and the subtree never carried (the shared type
vocabulary, and Yukon snowfall not being precipitation) were ported into
`tests/stations/test_clients_offline.py` in the same change, which is the
last thing the hand-carrying rule was there to catch.

---

Sequencing: after §3 foundations exist (so the adapter has `aoi`, `auth`, logging to build on),
before the docs rewrite (so the stations gallery examples are written once).

---

## 10. Packaging, tooling, and release

### 10.1 Dependency floors this plan assumes (checked 2026-09-15)

| Package | Current | Floor to adopt | Why |
| --- | --- | --- | --- |
| Python | — | **3.12** (support 3.12–3.14) | SPEC 0; rasterio 1.5, rioxarray 0.23, zarr 3.3, numpy 2.5, earthaccess 0.18+ all require ≥ 3.12 |
| xarray | 2026.7.0 | ≥ 2026.2 | `keep_attrs=True` default, zarr-3 minimum, `__dask_exprs__`; rioxarray 0.23 needs it |
| zarr | 3.3.0 | ≥ 3.1 | v3 default format, `FsspecStore`/`ObjectStore`, auto-detects v2 stores (ARCO-ERA5 is v2 with consolidated metadata) |
| odc-stac / odc-geo | 0.5.3 / 0.5.3 | ≥ 0.5 | raster extension v2, `fuse_func`, auxiliary bands, `GeoBox` |
| pystac-client / pystac | 0.9.0 / 1.15.2 | ≥ 0.8 | `CollectionSearch`, CQL2 handling |
| rioxarray / rasterio | 0.23.0 / 1.5.1 | ≥ 0.22 / ≥ 1.5 | GeoZarr `spatial:`/`proj:` conventions, `thread_safe=True`, GDAL ≥ 3.8 |
| geopandas / pyogrio | 1.1.4 / 0.13.0 | ≥ 1.1 / ≥ 0.12 | pyogrio default engine, `use_arrow`, GeoParquet bbox pushdown |
| earthaccess | 0.19.0 | ≥ 0.17 | explicit `login()`, `EARTHDATA_TOKEN`, `virtualize()` |
| earthengine-api / xee | 1.7.43 / 0.1.2 | ≥ 1.6.12 / ≥ 0.1 | project-required init; `crs`/`crs_transform`/`shape_2d` API the code already uses |
| dask | 2026.8.0 | ≥ 2025.1 | dataframe expressions merged; array expressions still opt-in |
| icechunk / virtualizarr | 2.2.0 / 2.7.3 | optional extra | future Zarr-store outputs and virtual NetCDF access |
| pooch | 1.9.0 | new dependency | hashed, cached downloads of the static GeoTIFF/zip products |

- **`pyproject.toml`** as the single manifest with **hatchling + hatch-vcs** (the Scientific
  Python guide default; version from git tags), pixi config in `[tool.pixi.*]` so there is one
  dependency list (pixi resolves conda-forge first, PyPI second; `pixi` 0.81 supports this
  fully). Drop `bump-my-version`'s four-file search/replace.
- **One install, no extras (decided).** All runtime dependencies stay required: xarray,
  rioxarray, odc-stac, odc-geo, pystac-client, planetary-computer, geopandas, pyogrio, shapely,
  pandas, numpy, dask, zarr, gcsfs, s3fs, fsspec, pooch, requests, pyyaml, earthaccess,
  h5netcdf, earthengine-api, xee, matplotlib, tqdm, scikit-image, beautifulsoup4/lxml. The
  notebook-only packages (`folium`, `contextily`, `mapclassify`, `py3dep`) move to the docs
  environment. "Earth Engine optional" (§12 Q7) therefore means *optional at run time* —
  credential-free default routes, `import ee`/`import xee` deferred inside `providers.gee` so
  import stays fast and a broken Google auth stack cannot break `import easysnowdata`.
- **Python support**: **3.12–3.14**. `earthaccess` dropped 3.11 in 0.18 (May 2026) and added
  3.14 in 0.19, and `global_snow_networks` already requires ≥ 3.12; keeping 3.11 would pin us to
  an `earthaccess` without the explicit-login and `virtualize()` APIs this plan relies on.
- **Lint/format/type**: ruff (weekly releases; 0.16 now) with the Scientific Python guide's
  `extend-select` set (`ARG B C4 EM EXE FURB G ICN ISC LOG NPY PD PERF PGH PL PT PTH PYI Q RET
  RSE RUF SIM SLOT T10 T20 TC TRY UP`, ignoring `PLR09`/`PLR2004`) plus `D` for the public API
  — `T20` alone catches every stray `print`; `mypy --strict` (2.x) on `aoi`, `auth`, `catalog`,
  `processing` (the pure parts), with `pyrefly`/`ty` as optional fast checkers; `codespell`;
  `pre-commit` kept (or `prek`, its Rust drop-in).
- **Publishing**: PyPI **trusted publishing (OIDC)** via `pypa/gh-action-pypi-publish` ≥ 1.11
  (attestations on by default) instead of a stored password; build sdist+wheel with
  `python -m build`; `git-changelog` kept but generated from conventional commits on release,
  not pushed to `main` from CI with `[skip ci]`.
- **conda-forge**: feedstock update for the new dependency split; `easysnowdata` metapackage
  pulling `[all]` equivalents; enable bot automerge on the feedstock.
- **Outputs**: rioxarray 0.23 writes the GeoZarr `spatial:`/`proj:` conventions (now an OGC
  SWG spec, also read by GDAL 3.13), so `to_zarr()` from any easysnowdata result should
  round-trip CRS without extra work — one more reason attrs must stay serializable (§2.5).
- **Repo hygiene**: remove `MANIFEST.in` (setuptools reads pyproject), `.editorconfig` keep,
  `CITATION.cff` updated on release via the same tag workflow, `CHANGELOG.md` kept.

---

## 11. Phased roadmap

| Phase | Scope | Exit criteria | Rough size |
| --- | --- | --- | --- |
| **0 Stabilize** | GRDC: GET-then-read and GET-first health probe; add explicit `earthaccess.login()` to UCLA SR and MOD10A1F and set `cloud_hosted=True`; switch Köppen to file `61012822`; fix UCLA `stats` mapping; fix conftest to honour `EARTHDATA_TOKEN`; mark live tests and split `test`/`test-live` pixi tasks; CI runs offline tests on push, live tests weekly; declare `pyyaml`/`requests`, drop unused deps; `requires-python>=3.12`; release 0.0.26 | CI green on `main`; README status table has no long-standing red rows without an issue | days |
| **1 Foundations** | `aoi`, `auth` (Earthdata, Earth Engine, Planet, NVE providers), `catalog`, `providers`, `processing`, `plotting`, logging; no import side effects; deprecation shim mechanism; unit + recorded test tiers; pixi in CI | 100% offline coverage of the new modules; old public API unchanged and still passing live smoke tests | 2–3 weeks |
| **2 Products** | Migrate theme by theme: terrain → land → snow (static) → hydro → climate → optical → SAR → snow (time series). Eric's three priorities land inside this phase as the first *new* routes: **VIIRS snow (VNP10A1/VNP10A1F) with the MODIS→NSIDC switch, NSIDC direct SNODAS, and the HLS modernization**; OPERA RTC-S1 + static incidence layers follow in the SAR step; **PlanetScope via the `planet` SDK (Data API search, Orders API clip) is the first key-gated commercial source and lands in the optical step** (§4.2, §12 Q17). Each product: catalog entry, loader on providers, source modernization from §4, one recorded test, one live smoke test, one gallery script | old modules are shims only; every product has all four artefacts (§2.11) | 4–6 weeks, parallelizable by theme |
| **3 Stations** | §9 steps 1–6 | `global_snow_networks` pipeline runs against `easysnowdata.stations.clients`; `StationCollection` shim passes its old tests | 1–2 weeks |
| **4 Docs & automation** | mkdocs-gallery (or fallback), catalog pages, credentials page, README gallery montage, health→issue, latency probe, watch digest, routine prompt | site builds in CI from examples without committed outputs; weekly digest issue appears | 2 weeks |
| **5 Release 0.1 → 1.0** | remove shims after one minor cycle; conda-forge feedstock; Zenodo version; announce | API frozen; docs and health green | after 2–3 months of use |

Phases 1 and 2 can start on a `revamp/` branch while 0 ships from `main`. Because each product
migration is independent once Phase 1 lands, this is well suited to parallel agent sessions
with the catalog entry as the shared contract.

---

## 12. Decisions (recorded 2026-09-15)

| # | Question | Eric's decision | Applied in |
| --- | --- | --- | --- |
| 1 | Layout | **Option A**: theme subpackages over a `providers` layer | §3 |
| 2 | API style | **Module-level `search`/`load` functions with `source=`**; classes become shims | §3.4 |
| 3 | Dims and CRS | **`latitude`/`longitude` for geographic grids, `y`/`x` for projected**; always write CRS with both `.rio` and `.odc` | §2.5 |
| 4 | Nodata | **Sentinel + `rio.nodata` for categorical, NaN-masked for continuous** | §2.5 |
| 5 | Compatibility | **Deprecation shims for one release** | §3.4, §11 |
| 6 | Docs | **Sphinx + pydata theme + sphinx-gallery + myst-nb** | §7 |
| 7 | Earth Engine | **Optional at run time**: credential-free defaults for HUC (WBD REST), LIA (OPERA static), SNODAS (NSIDC); GEE kept as an alternative source | §4, §10 |
| 8 | Sturm & Liston | **NSIDC with Earthdata Login as default, hosted COG as option.** The COG **stays on the `uwcryo` blob for now**; the catalog entry notes the location is likely to change (Zenodo DOI or GitHub release asset later) | §4.3 |
| 9 | Stations | **Option B, `git subtree`, no transition period for the old CSV route** | §9 |
| 10 | Credentials | **Env vars + native files, no config file**; **one-line credential summary on import** (interactive only, network-free) and a pre-network `CredentialError` naming alternatives | §5 |
| 11 | New-source priorities | **VIIRS snow, NSIDC SNODAS, HLS** first; a maintained product list with per-product source comparisons lives in `POTENTIAL_DATA_PRODUCTS_SOURCES_AND_EXAMPLES.md` | §11, companion file |
| 12 | Watch | **GitHub Actions only** | §8 |
| 13 | Packaging | **One install, no extras; drop Python 3.11; pixi config in `pyproject.toml`; trusted publishing** | §10 |
| 14 | Idea dump | **Evaluate each item in a table** (why it matters, access, verdict); include in the rewrite only what earns it | companion file §C |
| 15 | Plotting | **`esd.plotting.categorical(da)`**, no callables in attrs | §2.7 |
| 16 | Water-year helpers | Asked for the trade-offs — see below; recommendation: keep the function names, reimplement vectorized on top of the `global_snow_networks` `utils` implementation, add `processing.time.add_water_year_coords()`, and document the `resample(time="YS-OCT")` idiom | §2.6 |
| 17 | Planet | **Integrate Planet data access and authentication into the plan** (asked 2026-09-15 while Phase 0 ran). The rough `PlanetData` class in `docs/examples/sandbox.ipynb` is the seed; PlanetScope moves from the companion's Tier 3 shelf to Tier 2 and ships in Phase 2 behind a `planet` auth provider. No code lands in Phase 0 (no new modules, no public API). Shipped in Phase 2b; ⚠️ nothing is verified against the live API without a `PL_API_KEY` secret (§13.3 G1) | §4.2, §5, §11, §13.3; companion §A, §B.11, §C, §E.1 |

### 12.1 Water-year helpers: trade-offs (Q16)

Today `datetime_to_WY` / `datetime_to_DOWY` are scalar functions applied with
`pd.Index.map`, i.e. a Python call per timestamp, and they return `np.nan` on parse failure.
`global_snow_networks/utils/utils.py` already has vectorized `water_year()` /
`day_of_water_year()` / `add_wy_coords()` that accept scalars, arrays, Series and xarray
objects. The options:

| | Keep the current functions | Standardize on xarray idioms (`resample(time="YS-OCT")`, `water_year` coord + `UniqueGrouper`) |
| --- | --- | --- |
| Benefits | Familiar names; used in every notebook; trivial to explain | Vectorized and Dask-aware (no per-element Python); water-year *aggregation* becomes one line (`ds.resample(time="YS-OCT").max()`); composes with flox-accelerated groupby; no custom code to maintain for the aggregation case |
| Drawbacks | O(n) Python loop on long station records; silent `nan` on bad input; duplicated in two repos; not Dask-aware | `YS-OCT` is a pandas anchored offset, so the southern-hemisphere start (`YS-APR`) must be passed explicitly; **day-of-water-year has no pandas primitive** and still needs a computed coordinate; `SeasonResampler` with a single 12-month season is unverified; less discoverable for newcomers |

Recommendation: do both, cheaply. Keep `water_year()` / `day_of_water_year()` as public,
vectorized functions (adopt the `global_snow_networks` implementation as the single copy when
§9 lands), add `processing.time.add_water_year_coords(obj, hemisphere="northern")` that attaches
`water_year` and `dowy` coordinates, and teach the `resample(time="YS-OCT")` idiom in the
Concepts page for aggregation. The old names stay as aliases through the shim release.

### 12.2 Phase 2a caveats: static products (recorded 2026-09-16)

Things the plan did not settle, found while migrating terrain, land, snow (static) and
hydro. Each says what the code does today, so a later reader can change it deliberately
rather than discover it.

| # | Caveat | What the code does | Where it is recorded |
| --- | --- | --- | --- |
| A | **The NLCD default is ambiguous in this plan.** §4.4 says Annual NLCD should be the default (`source="annual"`); the companion's Tier 1 table says the official 2021 release is the default with Annual NLCD as the alternative | §4.4 wins: the default source is Annual NLCD, with the source id **`gee-annual`** (not `annual`, so it reads parallel to the other `gee` route). `source="gee"` is the 2021 release and is what the `get_nlcd_landcover` shim uses, so old calls are unchanged. **Needs Eric**: confirm the default, then fix whichever document is wrong | this row; `easysnowdata/land/nlcd.py` docstring says *why* Annual is the default but not that the two documents disagree |
| B | **Only one NSIDC-0768 file name is verified.** Every path under the NSIDC HTTPS directory redirects to Earthdata Login, including deliberately wrong ones, so file existence cannot be probed without credentials | The 10 arcsec global name is confirmed (the hosted COG carries the same name). The 2.5 arcmin, 5 arcmin and 30 arcmin grids and the `NA` regional subset follow the documented convention and are **unverified**; a wrong name surfaces as a `FileNotFoundError` naming the directory and `source="hosted-cog"`. Worth one run of `esd.snow.snow_classification.load(aoi, resolution=...)` per grid with credentials | this row; the naming scheme is in `RESOLUTIONS` and `filename()` |
| C | **The Wrzesien clouds layer has no documented class meanings.** The Zenodo record defines the two class rasters but not the cloud layer, whose values run 0-6 | It is served as a continuous level rather than a categorical variable: no CF flag attrs, and `long_name` says the levels are undefined upstream | already in the code (module docstring, `layer=` docs, the catalog `Variable`, and the `long_name` attr on the data) |
| D | **`chunks=None` is ambiguous.** In odc-stac and rioxarray `chunks=None` means "load eagerly", but a loader also needs a value meaning "the package default" | Every Phase 2a loader takes the `contract.DEFAULT` sentinel for the package default (Dask at the source's native chunking) and treats an explicit `chunks=None` as "compute now", which keeps the old `get_copernicus_dem(chunks=None)` behaviour. Any new loader should follow this | already in the code (the docstring of every loader) |

Both of the merge follow-ups this section first listed are done on
`revamp/phase2`:

- The four per-theme copies of the output-contract helpers are folded into
  `processing/contract.py` and `catalog/_access.py`. Folding them settled what the
  `source` attribute means, which the two halves disagreed about: **it is the source
  id**, so `ds.attrs["source"]` hands straight back to `load(source=...)`, with the
  display name in `source_title` and `source_id` kept as an alias. The `chunks`
  sentinel of row D lives at `contract.DEFAULT`.
- `repair_fill_values` moved to `processing.snow`.

A source note worth keeping with §4.7: the **USGS WBD ArcGIS REST service needs paging and
retries**. Its `maxRecordCount` is 2000, but any page that takes too long comes back as
HTTP 500 with an HTML body — a single HUC12 query over a 2°×2° box failed after 93 s, and
pages past the first few fail even at 100 features. The loader pages in small blocks,
rounds coordinates with `geometryPrecision`, retries each page, and follows
`exceededTransferLimit`.

---

## 13. Implementation caveats (Phase 2b, recorded 2026-09-16)

_Found while building the time-series products on `revamp/phase2b-timeseries-products`
(climate/ERA5, Köppen-Geiger; optical/Sentinel-2, HLS, PlanetScope; SAR/Sentinel-1; snow/MODIS,
VIIRS, SNODAS, UCLA SR). Three kinds of entry: **corrections** to statements made elsewhere in
this plan, **surprises** the plan did not anticipate, and **gaps** that could not be verified.
Every claim below was checked live on the date given; anything that stays true is worth folding
into the section it corrects the next time this document is revised._

### 13.1 Corrections to statements elsewhere in this plan

**C1. Earth Search `sentinel-2-l2a` has *already* removed the baseline offset from the pixels,
while still advertising it in `raster:bands`** (corrects §4.2, which says to apply the offset
from `raster:bands` where present, and the §14 note to the best-practices repo that says
scale/offset can be taken from Earth Search metadata).

Measured on one identical tile and date (10TES, 2023-08-10, band B04, 60 m, EPSG:32610, median
over the AOI):

| Route | Median DN | Offset still in the pixels? |
| --- | --- | --- |
| PC `sentinel-2-l2a` | 4162 | yes (raw, baseline ≥ 04.00 items carry +1000) |
| Earth Search `sentinel-2-l2a` | 3137 (−1025) | **no — already subtracted** |
| Earth Search `sentinel-2-c1-l2a` | 4136 (−26) | yes |

Applying `offset: -0.1` from `raster:bands` on the Earth Search `sentinel-2-l2a` route therefore
subtracts 1000 DN twice. The discriminator is the item property
**`earthsearch:boa_offset_applied`**: when it is `True` the offset is already applied and must
not be applied again. `optical/sentinel2.py::_offset_days` honours that property and falls back
to the processing-baseline date rule (items on or after 2022-01-25) only when the property is
absent. `odc-stac` ignoring `raster:bands` (§14 notes) is what makes this our problem rather
than the loader's.

**C2. The legacy `utils.get_stac_cfg` never applied its per-asset overrides.** Its per-asset
entries were keyed by the *band alias* (`scl`, `visual`) while `odc-stac` matches `cfg` keys
against the *asset name* (`SCL`, `visual`), so every override silently did nothing and SCL came
back `uint16` instead of `uint8` — the concrete form of the band-alias drift recorded as issue
#6 in §1.2. The replacement `_PC_STAC_CFG` in `optical/sentinel2.py` keys per-asset entries by
asset name, and deliberately keeps the legacy `costal` alias typo alongside the correct
`coastal` so notebooks that use the old name keep working through the shim release.

### 13.2 Upstream behaviour the plan did not anticipate

**S1. The Planet SDK fetches a spec from the network while *building* an order request.**
`planet.order_request.product()` resolves bundles through `specs._LazyBundlesLoader`, which
GETs `https://api.planet.com/compute/ops/bundles/spec` on first use. Building a request
therefore fails offline (and whenever that endpoint is down or slow), which breaks both the
recorded test tier and any request built before a network session exists.
`providers/planet.py::product()` builds the product dictionary locally from the bundle and item
type; `validate_bundle=True` opts back into the SDK's spec-backed validation for callers who
want it and have the network. Bundle names are validated against a local `BUNDLES` tuple.

**S2. SNODAS saturates its int16 range over glaciers.** SWE is stored as int16 millimetres, and
over perennial ice the model runs up to the type maximum: 32767 mm = 32.767 m. On 2024-03-15 the
masked CONUS grid has 8 saturated pixels — around 46.845–46.8625 N, −121.75 W (the Rainier ice
cap) and 48.7792 N, −121.8125 W (Mount Baker) — while the next-highest real value is 32280 mm and
the Rainier box median is 0.86 m (p90 2.10 m). This is a property of the NOHRSC model, not of
the reader, so `snow/snodas.py` passes the values through untouched and documents the artefact;
users who want it gone mask with `ds["SWE"].where(ds["SWE"] < 30)`. Worth knowing before
anyone treats a basin maximum from SNODAS as physical.

**S3. ARCO-ERA5 must be subset *before* it is chunked.** Opening
`full_37-1h-0p25deg-chunk-1.zarr-v3` with a `chunks=` argument builds a Dask graph over all 273
variables on the full hourly 1940→present grid; that graph alone exhausted memory and got the
live ERA5 test OOM-killed. `climate/era5.py::_load_arco` now opens with `chunks=None` (lazy,
no graph), selects variables/time/space, and only then calls `.chunk({"time": 24})`. The live
test went from an OOM kill to under four seconds. The same pattern applies to any wide
cloud-native store we add as a source.

**S4. Planetary Computer SAS signing makes VCR cassettes order-dependent.**
`planetary_computer.sign` caches its SAS token per process, so the token request appears once in
a cassette but is needed by every test that signs an asset, and the second test replaying the
same cassette finds the interaction already consumed. `tests/timeseries/conftest.py` overrides
`vcr_config` with `allow_playback_repeats: True`; without it, recorded PC tests pass alone and
fail as a suite.

### 13.3 What could not be verified live

**G1. Planet capabilities are unverified against the real API — there is no `PL_API_KEY` in the
environment.** Everything below is implemented and covered by recorded/synthetic tests, but has
never run against `api.planet.com`:

| Capability | State |
| --- | --- |
| Data API quick-search (`providers.planet.search`, `optical.planetscope.search`) | synthetic responses only |
| Asset activation and polling (`source="data-api"`) | unexercised |
| Orders API order creation with the `clip` tool (`optical.planetscope.order`) | unexercised; **stays manual by design** — orders spend Education & Research quota |
| Order polling and delivery download (`wait_order`, `download_order`) | unexercised |
| Authenticated health probe (`GET https://api.planet.com/data/v1/`) | unexercised |
| XYZ tile URLs (`tile_url`) | URL construction only |

To close this, add a `PL_API_KEY` repository secret and run the `requires_planet` live tests;
the search-only smoke test is free, the ordering tests should stay opt-in. The licence
constraint stands either way: **Planet imagery is not redistributable**, so no real scene crops,
thumbnails or item ids from the E&R account go in git — the recorded Data API responses in
`tests/timeseries/` are synthetic, and the PlanetScope fixture is generated, not a real scene.

**G2. Two live-tier failures are environment-only, not code.** Both reproduce on `main` and
neither is caused by Phase 2b:

- Station tests that fetch over `urllib` fail TLS verification behind the agent proxy
  ("CA cert does not include key usage extension"). `requests`-based routes are fine.
- The full legacy live tier OOM-kills on `TestGetHydroBasins`, which downloads the 2.7 GB
  `BasinATLAS_Data_v10.gdb.zip`. Run the live tier file by file, or skip that class, until the
  HydroSHEDS per-region route in §4.7 lands and makes the download small.

### 13.4 🔍 items this phase closed

- **PC HLS collection ids are `hls2-l30` and `hls2-s30`** (§4.2 marked them 🔍 pending PC coming
  back up). Confirmed live 2026-09-16; wired as the credential-free mirror source on
  `optical/hls.py`.
- **MOD10A2 is still served by NSIDC** despite PC having stopped archiving it (§4.3's ❌ row).
  Confirmed live 2026-09-16 via `earthaccess`; the NSIDC route is the default and PC stays as an
  optional historical source.

---

## 14. Notes to file in the knowledge bases (proposed 2026-09-15; filed to both inboxes 2026-09-15 and 2026-09-17)

For `all_project_memory/INBOX.md`:
- easysnowdata CI has been red since ≥ June 2026 on the GRDC/WMO basins URL; the health check
  saw it weekly; plan is to make health failures open issues.
- The Sturm & Liston snow-classification GeoTIFF that easysnowdata serves lives on the `uwcryo`
  blob — same expiry problem as the P3 store; needs re-hosting before 2026-10-26 / 2026-12-31.
- `global_snow_networks` → `easysnowdata.stations` merge plan (§9) as the concrete form of the
  2026-09-15 decision recorded in `_meta/software.md`.
- **SNODAS SWE saturates its int16 range at 32767 mm (32.767 m) over perennial ice** — 8
  pixels on 2024-03-15 over the Rainier ice cap and Mount Baker, next real value 32280 mm.
  A NOHRSC model artefact, not a reader bug; mask before taking basin maxima. Worth a line
  in the snow-product notes (and on `nsidc.md` for the G02158 route).
- Planet is now in the easysnowdata plan (§12 Q17): PlanetScope/SkySat via the `planet` SDK,
  Phase 2. Connects to the P2 note "near-daily binary snow mask (e.g., PlanetScope)" and the
  snow-ideas item "use planet imagery to confirm snowmelt (tested over Mill Creek)". The
  3.6 GB of PlanetScope scenes under `docs/examples/planet_data/` are untracked and licence-bound.

For `geospatial_data_and_visualization_best_practices/TO_BE_INCORPORATED.md` (all verified
live on 2026-09-15 unless noted):
- **OPERA RTC-S1 is searchable via CMR-STAC (`cloudstac/ASF`, ids `OPERA_L2_RTC-S1_V1_1`,
  `OPERA_L2_RTC-S1-STATIC_V1_1`) and `earthaccess`, not via `stac.asf.alaska.edu`** (which has
  only two unrelated collections); the static product ships local-incidence-angle,
  layover/shadow-mask and number-of-looks COGs; a credential-free mirror without those layers
  exists on GEE (`OPERA/RTC/L2_V1/S1`). Belongs on `alaska-satellite-facility.md`.
- **A server that rejects HEAD looks like a 404 to GDAL `/vsicurl` and to `requests.head`**
  (GRDC `grdc.bafg.de` returns 400 to HEAD, 206 to GET). Health probes should be GET-first;
  `CPL_VSIL_CURL_USE_HEAD=NO` is the GDAL escape hatch. Belongs on `gdal.md` / the
  COG page.
- **`earthaccess` 0.16–0.19 breaking changes**: explicit `login()` required, `EARTHDATA_TOKEN`
  precedence, `virtualize()` replaces `open_virtual_mfdataset`, granule methods → fields, Python
  3.11 dropped in 0.18. Belongs on `earthdata-and-earthaccess.md`.
- **NSIDC MODIS/VIIRS snow (MOD10A1, MOD10A1F, MOD10A2, MYD10A1(F), VNP10A1, VNP10A1F) are all
  cloud-hosted under `NSIDC_CPRD`**; Sturm & Liston NSIDC-0768 is not, and its HTTPS directory
  redirects to Earthdata Login. Belongs on `nsidc.md`.
- **ARCO-ERA5 status**: final ERA5 to 2026-05-31, ERA5T to 2026-09-09 as of 2026-09-15; no
  ERA5-Land; the CDS ARCO Zarr data lake (beta 2026-06-30) is the first ERA5-Land Zarr route;
  AWS `era5-pds` is deprecated in favour of `nsf-ncar-era5`; Earthmover publishes an anonymous
  Icechunk ERA5. Belongs on `other-data-portals.md` or a new reanalysis page.
- **Planetary Computer**: scheduled maintenance 2026-09-15 took the STAC API down (503) for
  hours — a reminder that PC-only pipelines need a fallback; still no `sentinel-2-c1-l2a` on PC
  (issue #394 open). Belongs on `microsoft-planetary-computer.md`.
- **figshare**: use `ndownloader.figshare.com/files/<id>` (302 to signed S3); the
  `figshare.com/ndownloader/...` host serves a bot-challenge HTML page to non-browser clients.
  Köppen-Geiger (Beck 2023) current file is `61012822`, not `45057352`.
- **xee 0.1.x** (0.1.2, 2026-07-14): `scale`/`geometry` removed in favour of `crs` /
  `crs_transform` / `shape_2d`; dims now `(time, y, x)`; helpers `extract_grid_params`,
  `fit_geometry`; `ee_init_if_necessary`/`ee_init_kwargs` for Dask workers. Belongs on
  `gee-and-xee.md` (which still says "confirm stable install path").
- **`odc-stac` ignores `raster:bands` scale/offset** (stackstac applies them); Earth Search
  Sentinel-2 items carry `scale 0.0001, offset -0.1` so `stac_cfg` is unnecessary there.
  **Corrected 2026-09-16 (§13.1 C1): Earth Search `sentinel-2-l2a` has already subtracted
  the +1000 baseline offset from the pixels while still advertising `offset: -0.1`, so
  applying that offset double-counts it; the item property `earthsearch:boa_offset_applied`
  is the discriminator, and `sentinel-2-c1-l2a` still ships the offset in the pixels.**
  Measured on tile 10TES, 2023-08-10, B04, 60 m: PC 4162 DN, Earth Search `l2a` 3137,
  Earth Search `c1-l2a` 4136. Also: `odc-stac` matches per-asset `cfg` keys against the
  **asset name**, not the band alias — keying them by alias fails silently (§13.1 C2).
  Belongs on `odc-stac-and-odc-geo.md`.
- **`planet.order_request.product()` hits the network while building a request** (it
  resolves bundles through `specs._LazyBundlesLoader`, which GETs
  `api.planet.com/compute/ops/bundles/spec`), so offline request construction and recorded
  tests fail unless you build the product dict yourself (checked 2026-09-16, SDK 3.6.0).
  Belongs on the new `planet.md` page.
- **`planetary_computer.sign` caches its SAS token per process**, which makes vcrpy
  cassettes order-dependent: the token interaction is recorded once but needed by every
  replaying test. `allow_playback_repeats: True` in the `vcr_config` fixture is the fix.
  Belongs on `microsoft-planetary-computer.md` or a testing page.
- **Wide cloud-native Zarr stores must be subset before they are chunked**: opening
  ARCO-ERA5 (`full_37-1h-0p25deg-chunk-1.zarr-v3`, 273 variables, hourly 1940→present)
  with a `chunks=` argument builds a Dask graph large enough to exhaust memory; open with
  `chunks=None`, `.sel()`, then `.chunk()` (checked 2026-09-16). Belongs on `zarr-cloud-
  visualization-ecosystem.md` or the xarray/Dask page.
- **xarray `keep_attrs=True` is the default since 2025.11.0**, zarr ≥ 3 is the minimum since
  2026.04, default netCDF engine flipped to h5netcdf and back (2025.09.1 → 2025.10.1: pass
  `engine=` explicitly). Belongs on `xarray.md`.
- **`mkdocs-gallery` is dormant (no release since 2024-09); `sphinx-gallery` is the only
  maintained thumbnail gallery; `mkdocs-jupyter` has no execution cache.** Belongs on
  `environment-and-package-management.md` or a new docs-tooling page.
- **Stack now requires Python ≥ 3.12**: rasterio 1.5 (GDAL ≥ 3.8), rioxarray 0.23, zarr 3.3,
  numpy 2.5, earthaccess 0.18. `stackstac` last released 2024-08 — treat as legacy. Belongs on
  the per-package "Version & maintenance status" notes.
- **`earthaccess.virtualize()` mechanics and limits** (appends `.dmrpp`, HDFParser fallback, no
  HDF4/TIFF, `access="direct"` only inside us-west-2, single `concat_dim`); **NSIDC publishes no
  DMR++ sidecars** for WUS_UCLA_SR, HMA_SR_D, MOD10A1(F)/A2 or VNP10A1F while ORNL Daymet and
  GES DISC IMERG do; VirtualiZarr `HDF4Parser` (2.7.1) is a kerchunk wrapper, root-group only,
  no CRS; `ZippedZarrParser` is STORED-only. Belongs on `icechunk-and-virtual-zarr.md` and
  `earthdata-and-earthaccess.md`.
- **Public Zarr/Icechunk stores found**: Earthmover ERA5 Icechunk (anon, quarterly, no
  ERA5-Land), WeatherBench2 ERA5 with `snow_depth` (GCS + Icechunk copy), CONUS404 on OSN,
  NLDAS-3 kerchunk/Icechunk on `nasa-waterinsight`, NASA EODC MUR/IMERG virtual Icechunk
  (bytes still need NASA creds in-region), DestinE EDH ERA5-Land Zarr v3 (token);
  **`s3://hrrrzarr` ends October 2026**; kerchunk is in maintenance mode and Pangeo Forge is
  no longer developed. Belongs on `zarr-cloud-visualization-ecosystem.md` / `other-data-portals.md`.
- **Planet SDK 3.x auth model** (checked 2026-09-15 from the SDK source, 3.6.0): OAuth2 is the
  default (`planet auth login`, sessions under `~/.planet/`), API keys are legacy but still read
  (`PL_API_KEY` by `Auth.from_env()`, `PL_AUTH_API_KEY` by the default session), M2M clients use
  `PL_AUTH_CLIENT_ID`/`PL_AUTH_CLIENT_SECRET`; 2.x is in maintenance since 2025-08. Access
  pattern: search with the Data API, **clip with the Orders API** rather than streaming
  full-scene GeoTIFFs and clipping locally (quota is charged by area; whole PSScene strips are
  550–675 MB). Belongs on a new `planet.md` data-access page next to the basemap-API quirks
  already in the inbox.
- The credential-provider pattern (§5) once implemented, as a worked example for
  `data-access/earthdata-and-earthaccess.md` and `gee-and-xee.md` (GDAL cookie jar + Dask
  workers is the non-obvious part).

---

## 15. Phase 5 review and the 0.3 cleanup (recorded 2026-09-21)

A review pass over the executed phases, with the stations transfer (§9) as the
focus, done on `main` after 0.2.0 shipped. What it found and what changed:

**The stations transfer held up.** Option B as decided: one copy of the clients
(`easysnowdata.stations.clients`), the adapter (`inventory`/`load`/`metadata`),
the archive fast path, `global_snow_networks` importing this package. Two
things were finished rather than redesigned: the vendored clients are now
linted and formatted with the rest of the package (the ruff exclusion is gone —
`ruff format` plus fifteen `UP` autofixes, no logic change), and the five
network products declare `provider="stations"` instead of the borrowed
`"vector_http"`. The two archive probe labels inherited from the retired
`snotel-ccss-stations` entry were renamed to say what they probe (the
five-network inventory and CSVs) and `data_status/history.json` was rewritten
with the new labels, so nothing in the history was orphaned. The NVE probe was
a genuine bug: it never sent `X-API-Key`, so the weekly check reported a 401
whenever the key *was* configured (issue #23).

**Deprecated surface removed (0.3).** `remote_sensing`, `hydroclimatology`,
`topography`, `automatic_weather_stations`, `utils`, `_deprecation`, their
tests, the pre-0.2 notebooks under `docs/examples/` and the `notebooks.md`
page — the release the shims promised. One regression this exposed: the only
`import rioxarray` in the package sat in the `topography` shim, so removing it
un-registered `.rio` for every loader on a bare import; `processing.contract`
now imports it, and `tests/test_easysnowdata.py` checks a fresh interpreter.

**Processing pruned.** `normalized_difference`, `ndsi`, `ndvi`, `ndwi`, `ndbi`,
`evi`, `binary_snow`, `rgb`, `stretch_percentile`, `stretch_clahe` and
`plotting.rgb` are gone, and `scikit-image` with them. The rule (§2.6
amended): processing helpers exist for what is *not* one line of xarray —
bit-field masks, metadata-driven scale/offset, baseline harmonization, the
SNODAS flat-file reader, the incidence-angle geometry, water years. Band
arithmetic and thresholds are written out wherever they are used, so the
gallery never hides which bands or which cut-off.

**Terrain is five DEMs, not one.** `terrain.dem.load(aoi, product=…)` serves
`copernicus-dem` (default; PC, Earth Search, and the 2024_1 release on GEE),
`nasadem` (PC, GEE), `srtm` (GEE), `3dep` (10 m / 30 m over the US; PC, GEE)
and `alos-dem` (PC v3.2, GEE v4.1), all through two code paths and one output
contract, all resampling bilinearly (elevation is continuous; nearest would
only copy the source staircase) so two routes to one DEM land on a common grid
identically; `terrain.dem.compare()` tabulates the differences (resolution, extent,
acquisition, vertical datum — EGM2008 / EGM96 / NAVD88 — surface vs terrain
model). All thirteen routes verified live over Mount Rainier on 2026-09-21.

**Plotting conventions (§2.7 amended).** `plotting.map`, `categorical`,
`points`, `timeseries`, `label`, `finish_map`: equal aspect (latitude-corrected
with a `GeographicAxesWarning` for EPSG:4326 data), colorbar matched to the
map height, a small scale bar (`matplotlib-scalebar`), a light lat/lon
graticule (pyproj, no cartopy), optional web basemap (`contextily`, Esri
shaded relief by default), legends outside the axes, units in `[ ]` never
`( )`, time axes always calendar dates (`by_water_year=True` for
October-to-September overlays). Every piece is a keyword.

**A sign error in the DEM-computed local incidence angle (0.2) is fixed.**
`processing.sar.local_incidence_angle` measured the range-facing slope
component from the look direction (sensor to ground) instead of from the
direction back toward the sensor, so a slope facing the radar was given the
*larger* angle. Found while the gallery compared it with the OPERA RTC-S1
static layer; verified on a synthetic east-facing plane with a west-looking
descending pass (35° nominal, 20° slope → 15.7°, not 55°). Every LIA computed
with 0.2's `source="dem"` or `source="gee"` route is affected; the OPERA
static layer route is not.

**The DEM-route incidence angle now uses each track's real geometry.** Eric
questioned whether a DEM plus constants could stand in for the acquisition
geometry, and it could not well: Sentinel-1's heading varies with latitude
and the ellipsoidal incidence angle runs 29°→46° across the IW swath, so a
constant 39° and a nominal heading were the crude part. `sentinel1.
scene_geometry()` now reads the heading (footprint along-track edge), look
azimuth and swath position off a representative Planetary Computer RTC scene
of the chosen `relative_orbit`, and `incidence_angle_field()` turns the swath
position into a per-pixel angle (linear model, within ~1° of OPERA's band at
Rainier). Against OPERA's per-burst layer the DEM route now agrees to a median
5.6°/5.8° (descending track 13 / ascending 137) versus 7.1°/6.6° with the old
constants; the remaining gap is DEM resolution and the absence of a
layover/shadow model. Every pass of a track is assumed to share its geometry —
the same assumption as Eric's `generate_sentinel1_local_incidence_angle_maps`
tool, which computes the exact answer from the GRD orbit file with `sarsen`;
that tool is referenced from the catalog entry rather than vendored, because
OPERA's static layer already is the exact answer where Earthdata Login is
available. `relative_orbit=` also filters the OPERA bursts (the track is in
the burst id), the GEE route filters `relativeOrbitNumber_start`, and all
three routes record `relative_orbit`, `platform_heading`, `look_azimuth` and
`incidence_angle_model` in the attrs.

**Every OPERA read from ASF had been failing.** `datapool.asf.alaska.edu` answers
GDAL's initial `/vsicurl` HEAD with a 403 from API Gateway, so both the OPERA
RTC-S1 backscatter route and the RTC-S1-STATIC incidence-angle route raised on
first read even with a valid Earthdata login; the `cmr-asf` catalog now sets
`CPL_VSIL_CURL_USE_HEAD=NO`. Found by the SAR gallery example, which now
draws Planetary Computer RTC beside OPERA RTC-S1 and the OPERA layover/shadow
mask. The same comparison showed the OPERA descending geometry at 47°N is best
reproduced with a look azimuth near 260°, not the 280° the old `S1_HEADING`
constants implied: headings are latitude-dependent and the constants were about
20° off there. **Removed 2026-09-22** (Eric: "all bogus"). No nominal heading or
constant incidence angle survives anywhere; the DEM and Earth Engine routes raise
when no scene of the pass exists over the AOI instead of guessing.

**Geometries are never fused (2026-09-22).** Eric: "We should never ever mix
geometries." `local_incidence_angle()` without `relative_orbit=` used to fuse
every OPERA burst with a max (mixing four tracks at Rainier) and, on the DEM
route, to pick the busiest track of one pass direction. It now logs an INFO line
("No relative_orbit specified: returning a local incidence angle raster for each
relative orbit within the AOI …") and returns a `relative_orbit` dimension with
one raster per track, `sat:orbit_state`/`platform_heading`/`look_azimuth` as
coordinates along it; `relative_orbit=` still gives a `(y, x)` raster.
`track_geometries()` (every track's footprint geometry, keyed by track) replaces
the busiest-track choice; `scene_geometry()` is its single-track reduction. The
OPERA backscatter route groups bursts by track before solar day and tags
`sat:relative_orbit` from the burst id, so a time step never holds two tracks.
The SAR gallery page now compares OPERA and DEM-route LIA per track (median |Δ|
5–6° at 60 m, bias −1 to −2°, RMSE ~8°) and Planetary Computer vs OPERA
backscatter per track with OPERA's mask applied to both (medians within 0.25 dB,
RMSE ~2.5 dB); the near-range tracks 13/64 lose 13 % of the AOI to layover
against 4 % for 115/137.

**Earthdata tokens expire; CI should not.** EDL user tokens last about 60
days and earthaccess trusts one from the environment without checking, so an
expired `EARTHDATA_TOKEN` secret turns the whole pipeline red. The provider
now asks URS whether the token is accepted (`token_is_valid`, through a
`requests` session with `trust_env=False`, because a netrc entry would
otherwise replace the bearer header with basic auth and validate the
password instead) and, when it is not and a username/password or netrc entry
exists, drops it and logs in with those — earthaccess then mints a fresh token,
which the GDAL bearer options pick up. The recommended CI configuration is
therefore `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`, with the token optional.
**Not yet done (2026-09-22):** `EARTHDATA_TOKEN` is the repository's only Earthdata
secret, and the first full docs build after this work failed on exactly that: six
Earthdata examples raised `CredentialError` because the token had expired and there
was nothing to fall back on. Adding the two secrets is the fix; the code is in place.

**Two more product bugs the gallery rewrite exposed.** Planetary Computer's
`modis-10A1-061` / `modis-10A2-061` collections hold Terra and Aqua granules
together and `snow.modis` did not filter on `platform`, so its daily mosaic
mixed the two overpasses (1.8 % of bytes differed from the NSIDC Terra granule;
Terra-only matches it exactly) — fixed with a default platform query, and
`MYD10A1`/`MYD10A2` are now selectable. `snow.snow_classification.RESOLUTIONS`
named NSIDC-0768 files that do not exist (`2.5km_2.5arcmin`, `10km_5.0arcmin`,
`0.5deg_30.0arcmin`); the archive's grids are `300m_10.0arcsec`,
`01km_30.0arcsec`, `05km_2.50arcmin`, `50km_0.50degree`, plus an `EA` region.
Both verified against the live archives on 2026-09-21.

**Docs organised by module.** One `KNOWN_THEMES` order (stations, snow, sar,
optical, terrain, land, hydro, climate) drives the catalog index, the status
page (health rows grouped by theme), the README table, the API navigation and
the gallery sections; a `tools` gallery section holds the cross-cutting
examples (AOI and time parsing, the catalog, water years). Example titles name
the product and its provider, never a place.
