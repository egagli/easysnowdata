# Getting started

Four short pages, in the order you need them:

1. **[Installation](installation.md)**: one `pip`, `conda`, `pixi` or `uv`
   command, and a check that it worked.
2. **[Credentials](credentials.md)**: most products need no account. This page
   lists the few that do (NASA Earthdata, Earth Engine, Planet, NVE) and how to
   set each one up.
3. **[Concepts](concepts.md)**: the rules every loader follows (one `aoi`,
   one `time`, lazy results, provenance in `.attrs`, several routes to the same
   product). Once you know them, a product you have never used behaves as you
   expect.
4. **[Modules](modules.md)**: what is in each subpackage, from
   `esd.stations` to `esd.climate`, with links to its examples and API.

The [glossary](glossary.md) spells out the acronyms used across the docs
(CPC, PSL, NRCS, SNOTEL, RTC …), and
[Migrating from 0.0.x](migration.md) maps the old function names to the new
ones.

A first session looks like this:

```python
import easysnowdata as esd

esd.auth.status()                      # which accounts are configured
esd.catalog.list(credential_free=True) # what works without any

aoi = (-121.94, 46.72, -121.54, 46.99) # Mount Rainier
dem_da = esd.terrain.dem.load(aoi)
snodas_ds = esd.snow.snodas.load(aoi, "2024-03")
esd.plotting.map(dem_da, cmap="terrain")
```

From here, the [example gallery](auto_examples/index.rst) has one executed
script per product.

```{toctree}
:hidden:

installation
credentials
concepts
modules
glossary
migration
```
