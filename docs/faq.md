# FAQ

## Which product should I use for X?

Search the catalog rather than guessing:

```python
esd.catalog.search("swe")  # free-text over ids, titles, variables, sources
esd.catalog.list(theme="snow")  # everything in one theme
esd.catalog.list(credential_free=True)  # everything that needs no account
esd.catalog.describe("ucla-snow-reanalysis")  # one product, in full
```

The [catalog](catalog/index.md) is the same information as a website, and the
[gallery](auto_examples/index.rst) shows each product being loaded and drawn.

## Do I need an account?

For 22 of 28 products, no. The rest need one of five providers, and most of
those have a credential-free alternative route — `source="planetary-computer"`
for HLS and MODIS, `source="hosted-cog"` for the snow classification,
`source="dem"` for the local incidence angle. [Credentials](credentials.md)
lists which products need what, and `esd.auth.status()` says what you already
have configured.

`CredentialError` is raised before any network request and its message
contains the setup steps, so you never discover a missing account halfway
through a download.

## Why is `load` instant? Where is my data?

It is lazy — see [Concepts](#concepts-lazy). `load` performs the
metadata read (a STAC search, a Zarr `.zmetadata`, a granule query) and builds
a Dask graph. The bytes move when you `.compute()`, `.plot()`, or write.

To force it: `.compute()` for the whole thing, `.isel(time=0).compute()` for
one slice, or pass `chunks=None` to load eagerly at call time.

## The loader is using too much memory

Almost always one of three things:

**A wide store opened with `chunks=`.** Select before you chunk. See the
warning in [Concepts](#concepts-lazy).

**Chunks that do not match the access pattern.** A time-series reduction over
a small AOI wants long time chunks; a single-date map wants spatial ones.
`chunks={"time": 24}` and friends are passed straight through.

**Asking for a continental AOI at native resolution.** Pass `resolution=` to
load a coarser overview — most COG-backed products read an overview level
rather than decimating full-resolution pixels.

## Why does `esd.plotting.map` warn that my data are in a geographic CRS?

Because a degree of longitude is shorter than a degree of latitude away from
the equator, so plotting EPSG:4326 data with `aspect="equal"` stretches every
shape by `1/cos(latitude)` — 46 % at Mount Rainier. The helper applies that
correction to the axes and warns, so the figure is right and you know that
distances still vary across it. To silence it, load the product on a projected
grid (`crs="utm"` on the loaders that reproject) or pass
`warn_geographic=False` to `finish_map`.

## Which CRS and dimension names do I get?

`time`, `y`, `x` for projected grids; `time`, `latitude`, `longitude` for
geographic ones. The CRS is written with both the `.rio` and `.odc` accessors.
To force a grid, build an `odc.geo.GeoBox` and pass it as the `aoi` — the
loader will use it exactly.

## Why is my land-cover map full of NaN? / Why is my DEM's nodata still -32767?

They are the two halves of one rule: **categorical products keep their
sentinel** (so classes stay integers) and **continuous products are
NaN-masked** (so statistics are right). `mask=False` turns masking off,
`mask=True` forces it. The original fill value stays in
`encoded_nodata`.

## Can I use Dask distributed / Coiled / a cluster?

Yes; nothing in the package holds a client. Two things to know:

- Earth Engine reads need `ee.Initialize` on each worker. `xee` provides
  `ee_init_if_necessary`, and the Earth Engine provider's `env()` carries the
  init arguments.
- GDAL configuration travels through a context manager rather than global
  state, so a worker reading a signed URL needs the loader to be the one
  issuing the read (it is).

## A source is down. What do I do?

Check the [status page](status.md): it lists every route's most recent probe,
how long it has been failing and when it last worked. A route that is failing
also has an open issue labelled `data-source` against it, with the error, and
the issue closes itself when the route recovers. Then switch routes — `source=` — if the product
has another one. Provider outages are the reason multiple routes are a
first-class feature rather than a nicety.

## Why does SNODAS SWE read 32.767 m over a glacier?

Because SNODAS never melts perennial ice out, its int16 millimetre field
saturates at 32767 mm. On 2024-03-15 that is eight pixels over the Rainier ice
cap and Mount Baker. It is a NOHRSC model artefact, not a reader bug, so the
values are passed through untouched. Mask them before taking a basin maximum:

```python
seasonal = swe["SWE"].where(swe["SWE"] < 30)
```

## Is the Sentinel-2 baseline offset applied twice?

No, and the reason is worth knowing if you read Sentinel-2 yourself. Earth
Search's `sentinel-2-l2a` has **already** subtracted the +1000 baseline offset
from the pixels while still advertising `offset: -0.1` in `raster:bands`;
applying it again costs you 1000 DN. The item property
`earthsearch:boa_offset_applied` is the discriminator, and the loader honours
it. Planetary Computer and Earth Search's `sentinel-2-c1-l2a` still ship the
offset in the pixels.

## My 0.0.x code calls `easysnowdata.remote_sensing.get_*` — where did it go?

Version 0.2 kept those names as deprecation shims for one release, and 0.3
removed them. [Migrating from 0.0.x](migration.md) lists every old name next to
its replacement; the changes that are more than a rename are explained there
too. Pin `easysnowdata<0.3` if you need the old names to keep running while
you move.

## How do I cite this?

Cite the software through its Zenodo DOI **and** the data you used — every
product carries its own citation in `ds.attrs["data_citation"]` and on its
catalog page. The data citation is the one the data provider asks for and is
usually the one a reviewer wants.

## Something is missing from the catalog

Open an issue. If you want to add it yourself, [Contributing](contributing.md)
has the checklist — a product is a catalog entry, a loader, two tests, a health
probe and a gallery script, and the docs page writes itself.
