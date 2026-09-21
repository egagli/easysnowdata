# Concepts

Twelve rules that every loader in the package follows. They are short on
purpose: once you know them, a product you have never used behaves the way you
expect, and adding one is a matter of filling in a catalog entry rather than
re-deciding an interface.

(concepts-aoi)=
## One spatial input: `aoi`

Every loader's first argument is an area of interest, and it accepts four
spellings of one:

```python
import geopandas as gpd
import shapely

import easysnowdata as esd

esd.terrain.dem.load((-121.94, 46.72, -121.54, 46.99))  # (W, S, E, N) in EPSG:4326
esd.terrain.dem.load(shapely.box(-121.94, 46.72, -121.54, 46.99))
esd.terrain.dem.load(gpd.read_file("basin.geojson"))  # any CRS; reprojected for you
esd.terrain.dem.load(geobox)  # an odc.geo.GeoBox — exact grid
```

Internally all four become an {py:obj}`easysnowdata.aoi.AOI`: a target grid
(`GeoBox`) plus a footprint (`GeoDataFrame`). Pass a `GeoBox` when you need
output on a grid you already have — the loader will not resample to a grid of
its own choosing.

```python
aoi = esd.parse_aoi(gpd.read_file("basin.geojson"))
aoi.bounds, aoi.utm_crs, aoi.to_geobox(resolution=30)
```

Two knobs worth knowing:

`clip`
: `True` by default — the result is cut to the AOI. `clip=False` returns the
  covering tiles or granules whole, which is what you want when the AOI is a
  point, when you are mosaicking yourself, or when you need the untouched
  source pixels.

Antimeridian
: An AOI that crosses ±180° is split before it reaches a STAC API, which is
  the form those APIs expect. You do not need to do anything.

(concepts-time)=
## One temporal input: `time`

Anything pandas or STAC understands, and partial dates expand to the period
they name:

```python
esd.snow.snodas.load(aoi, "2024-03")  # the whole month
esd.snow.snodas.load(aoi, ("2024-03-01", "2024-03-15"))  # inclusive pair
esd.snow.snodas.load(aoi, "2023-10/2024-06")  # STAC syntax
esd.snow.snodas.load(aoi, "2024-03/")  # open end = now, at call time
```

"Now" is computed when you call, never when the module is imported, so a
long-running kernel does not quietly freeze its own idea of today.

Water years are first class:

```python
from easysnowdata.processing import add_water_year_coords

obs = add_water_year_coords(obs)  # adds `water_year` and `dowy` coords
obs.resample(time="YS-OCT").max()  # water-year aggregation, the xarray way
```

`dowy` is day of water year, 1 on 1 October (northern hemisphere; pass
`hemisphere="southern"` for 1 April). It is a plain integer coordinate, so it
can be swapped in as a plotting axis or grouped on directly.

(concepts-lazy)=
## Lazy by default

Loaders return Dask-backed xarray objects and never call `.compute()`. A
`load` that returns instantly has done a metadata read, not a data read; the
bytes move when you compute, plot or write.

```python
s1 = esd.sar.sentinel1.load(aoi, "2024-03")  # seconds: STAC search only
s1.mean("time").compute()  # now the COGs are read
```

`chunks=` is passed through, and the default is the source's native chunking
where it is known. One subtlety worth naming, because the ecosystem is
inconsistent about it: **`chunks=None` means "load eagerly"**, as in odc-stac
and rioxarray. The package default is a separate sentinel, so
`load(aoi, chunks=None)` computes now and `load(aoi)` stays lazy.

:::{admonition} Subset wide stores before you chunk them
:class: warning

Opening a 273-variable hourly store like ARCO-ERA5 *with* a `chunks=` argument
builds a Dask graph over the whole archive before you have selected anything —
enough to exhaust memory. The ERA5 loader opens with `chunks=None`, selects,
and only then chunks. Do the same with any wide cloud-native store you add.
:::

(concepts-search)=
## Search and load are separate

`search` returns a GeoDataFrame of the items or granules that match, with the
STAC properties as columns. Inspect it, filter it, and hand what survives to
`load`. Nothing prints; nothing is downloaded.

```python
items = esd.optical.sentinel2.search(aoi, "2024-03", cloud_cover=30)
items[["datetime", "eo:cloud_cover", "s2:mgrs_tile"]]
best = items.sort_values("eo:cloud_cover").head(3)
data = esd.optical.sentinel2.load(aoi, items=best)
```

(concepts-output)=
## What comes back

Dims
: `time`, `y`, `x` for projected grids; `time`, `latitude`, `longitude` for
  geographic (EPSG:4326) ones. That is the odc-stac convention and the one
  ERA5-style data already uses.

CRS
: always written with **both** the `.rio` and `.odc` accessors — they read the
  same metadata, and writing both means neither library has to guess.

Nodata
: **categorical products keep the source sentinel** with `rio.nodata` set, so
  a land-cover class map still has integer classes; **continuous products are
  NaN-masked** with `encoded_nodata` preserved so the original value is
  recoverable. `mask=` overrides either way.

Attributes
: only strings and numbers, never Python objects — that is what lets a result
  round-trip through Zarr and netCDF. Every product carries `source`,
  `source_url`, `product_id`, `data_citation`, `license` and
  `easysnowdata_version`. `ds.attrs["source"]` is the **source id**, so it
  hands straight back to `load(source=...)`; the human-readable name is in
  `source_title`.

Units
: metric, always. Station values are centimetres because that is what the
  networks report; gridded SWE is metres.

Categorical products additionally carry CF flag attributes — `flag_values`,
`flag_meanings`, `flag_colors`, `long_name` — which is what lets the plotting
helpers draw a correct legend without any callable living in `.attrs`:

```python
esd.plotting.categorical(esd.land.landcover.load(aoi))
```

(concepts-sources)=
## Several routes to the same product

A product can be served by more than one provider, and that is treated as
normal rather than exceptional. The first source listed is the default;
`source=` picks another; the return contract does not change.

```python
esd.terrain.dem.load(aoi)  # Planetary Computer
esd.terrain.dem.load(aoi, source="earth-search")  # AWS Open Data, same output
```

This is the escape hatch when a provider has an outage, when you want to avoid
an account, or when you are in a cloud region where one route is direct S3 and
the other is not. Each product's catalog page compares its routes side by
side, with resolution, extent, temporal coverage, latency and what differs.

```python
esd.catalog.describe("copernicus-dem")
```

(concepts-credentials)=
## Credentials are lazy, uniform and checked early

A product declares what it needs (`requires=("earthdata",)`); the provider is
initialised on first use; a missing credential raises `CredentialError`
**before** any network request, naming the setup steps and any credential-free
alternative for that product. See [Credentials](credentials.md).

```python
esd.auth.status()  # a table: provider, configured?, how, needed by
```

(concepts-processing)=
## Processing is pure, plotting is separate

Masking, metadata-driven scaling, baseline harmonization, dB conversion,
water-year coordinates and the local-incidence-angle computation are standalone
functions in {py:obj}`easysnowdata.processing` that take and return xarray
objects and do no I/O. Loaders call them for you through keyword options, but
the raw product is always one call away.

What is *not* wrapped is band arithmetic. A normalized difference, a threshold
on a snow-cover byte or an RGB stretch is one line of xarray, and putting it
behind a function name hid which bands were used and what happened to the
sentinel values. The gallery writes these out every time:

```python
s2 = esd.optical.sentinel2.load(aoi, "2024-03", mask="scl-default")  # convenience
raw = esd.optical.sentinel2.load(aoi, "2024-03")  # nothing applied
ndsi = (raw["green"] - raw["swir16"]) / (raw["green"] + raw["swir16"])
```

{py:obj}`easysnowdata.plotting` is optional — every product plots fine with
plain xarray and geopandas — but it is where the package's map and time-series
conventions live: `map`, `categorical` and `points` give equal-aspect axes (a
latitude-corrected aspect plus a `GeographicAxesWarning` for data in degrees),
a colorbar matched to the map, a scale bar, a light latitude/longitude
graticule and an optional web basemap; `timeseries` puts calendar dates on the
x axis; and `label` writes `long name [units]`, square brackets always. Every
piece of furniture is a keyword argument.

(concepts-quiet)=
## No import-time side effects, and logging instead of printing

Importing the package configures nothing global: GDAL and rasterio settings
live in context managers around the reads that need them, and `xr.set_options`
is never called on your behalf. Progress goes to
`logging.getLogger("easysnowdata")` — INFO for progress, DEBUG for URLs,
WARNING for fallbacks — so you choose the verbosity:

```python
import logging

logging.getLogger("easysnowdata").setLevel(logging.DEBUG)
```

The single exception is the one-line credential summary printed in interactive
sessions, which makes no network request and is silenced with
`EASYSNOWDATA_QUIET=1`.

(concepts-artefacts)=
## Every product has four artefacts

A product is not finished until it has a catalog entry, an offline test, a
live smoke test, a health probe and a gallery example. The catalog entry is
what generates its docs page and its health row, so the set stays in step.
[Contributing](contributing.md) turns that into a checklist.
