# Modules

easysnowdata is organized by theme. Each theme is a subpackage, each product
in it is a module with a `load()` (and a `search()` where there is something to
search), and every loader follows the rules in [Concepts](concepts.md). Five
supporting modules sit underneath them.

## Data, by theme

`esd.stations`
: Point observations from five snow-station networks: the USDA NRCS
  {term}`AWDB` ({term}`SNOTEL`, {term}`SCAN` and snow courses), California's
  {term}`CDEC`, British Columbia's {term}`DataBC`, Norway's {term}`NVE` and the
  Yukon's snow survey. `esd.stations.inventory()` and `esd.stations.load()` read
  across all of them; `esd.stations.archive` reads the pre-downloaded daily
  archive of all five in one request.
  [Examples](auto_examples/stations/index.rst) · {doc}`API <api/stations>`

`esd.stations.clients` — other station datasets
: Station data that is *not* one of the five daily networks above, read only
  through its own client: GHCN-Daily snow depth worldwide, CanSWE and NorSWE
  (historical SWE, mostly snow courses), Hydro-Québec's gamma SWE sensors,
  Alberta's pillows and courses, Germany's DWD, Finland's Syke, Switzerland's
  SLF, Russia's route snow surveys and daily snow depth in the Southern Andes.
  Much of it is periodic SWE or daily snow
  depth rather than daily SWE, so it is kept out of `inventory()`, `load()` and
  the daily archive; every record says its own `interval`.
  [Examples](auto_examples/station_datasets/index.rst)

`esd.snow`
: Gridded snow products: MODIS and VIIRS snow cover, {term}`SNODAS` and the UCLA
  snow reanalysis ({term}`SWE` and depth), the Sturm & Liston snow classes, and a
  global mountain snow mask.
  [Examples](auto_examples/snow/index.rst) · {doc}`API <api/snow>`

`esd.sar`
: Sentinel-1 {term}`RTC` backscatter and its local incidence angle.
  [Examples](auto_examples/sar/index.rst) · {doc}`API <api/sar>`

`esd.optical`
: Surface reflectance from Sentinel-2 L2A, {term}`HLS` and PlanetScope.
  [Examples](auto_examples/optical/index.rst) · {doc}`API <api/optical>`

`esd.terrain`
: Six {term}`DEM`s (Copernicus, NASADEM, SRTM, 3DEP, ALOS, GEDTM30) behind
  `esd.terrain.dem.load(product=...)`, the CHILI heat-load index, and a
  hillshade basemap.
  [Examples](auto_examples/terrain/index.rst) · {doc}`API <api/terrain>`

`esd.land`
: Land cover: ESA WorldCover, the US {term}`NLCD`, and forest cover fraction.
  [Examples](auto_examples/land/index.rst) · {doc}`API <api/land>`

`esd.hydro`
: Basin outlines: {term}`HUC` watersheds, HydroBASINS, and the GRDC basins.
  [Examples](auto_examples/hydro/index.rst) · {doc}`API <api/hydro>`

`esd.boundaries`
: The vectors that frame a map: countries, states and counties, any-level
  administrative units, {term}`RGI` glacier outlines, {term}`GMBA` mountain
  ranges, and Natural Earth layers.
  [Examples](auto_examples/boundaries/index.rst) · {doc}`API <api/boundaries>`

`esd.climate`
: {term}`ERA5` reanalysis, the Köppen-Geiger climate classes, and monthly
  climate-mode indices ({term}`PDO`, {term}`ENSO`, {term}`AO`, {term}`NAO`,
  {term}`PNA`, {term}`AMO`) from NOAA's {term}`CPC`, {term}`NCEI` and
  {term}`PSL`.
  [Examples](auto_examples/climate/index.rst) · {doc}`API <api/climate>`

The [dataset catalog](catalog/index.md) has a page per product: its routes,
resolution, coverage, credentials, licence, citation and current health.

## Supporting modules

`esd.aoi` and `esd.temporal`
: Turn the `aoi` and `time` arguments every loader takes into a grid, a
  footprint and a time window. {doc}`api/aoi` · {doc}`api/temporal`

`esd.catalog`
: The registry behind the loaders: `list()`, `search()`, `describe()` and
  `get()` over every product and route. {doc}`api/catalog`

`esd.auth`
: Credential checks for the few products that need an account;
  `esd.auth.status()` says what is configured. {doc}`api/auth`

`esd.processing`
: Pure functions on loaded data: {term}`water year <water year>` coordinates,
  categorical remapping, masks, and the per-sensor helpers.
  {doc}`api/processing`

`esd.plotting`
: Maps, point maps and time series with the conventions the gallery uses
  (equal aspect, scale bar, graticule, units in brackets). {doc}`api/plotting`

`esd.providers`
: The generic access layer (STAC, Earth Engine, HTTP, Zarr) the loaders are
  built on. Most users never call it directly. {doc}`api/providers`
