---
sd_hide_title: true
---

# easysnowdata

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sd-pt-2

:::{grid-item}
:columns: 12

# easysnowdata

**Snow-relevant geospatial data, one call each, as xarray.**

Thirty-two products — station observations from five networks, SAR and optical
imagery, snow cover and SWE, five DEMs, land cover, basins and reanalysis —
behind one API that takes an area of interest and a time range, returns lazy
Dask-backed xarray objects, and never downloads more than it has to.

```{button-ref} installation
:color: primary
:expand:
Install and get going
```
:::
::::

```{code-block} python
import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)                      # Mount Rainier; any AOI form works

inv_gdf = esd.stations.inventory(aoi, daily_only=True)      # which snow stations are here
obs_ds = esd.stations.load(inv_gdf, variables=["swe", "snwd"], time="2023-10/2024-09")

dem_da = esd.terrain.dem.load(aoi)                          # Copernicus GLO-30 …
dem_da = esd.terrain.dem.load(aoi, product="3dep")          # … or NASADEM, SRTM, 3DEP, ALOS
s1_ds = esd.sar.sentinel1.load(aoi, "2024-03", units="dB")  # Sentinel-1 RTC
snodas_ds = esd.snow.snodas.load(aoi, "2024-03")             # SNODAS, no account needed

esd.plotting.map(dem_da, cmap="terrain")                    # equal aspect, scale bar, graticule
esd.plotting.timeseries(obs_ds["swe"])                      # calendar dates, units in [ ]
```

## Where to go

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`telescope` Example gallery
:link: auto_examples/index
:link-type: doc

One executed script per product, with the figure it produces.
:::

:::{grid-item-card} {octicon}`database` Data catalog
:link: catalog/index
:link-type: doc

Every product: its routes, resolution, credentials, licence and health.
:::

:::{grid-item-card} {octicon}`book` Concepts
:link: concepts
:link-type: doc

AOI, laziness, nodata, CRS and dims — the rules every loader follows.
:::

:::{grid-item-card} {octicon}`key` Credentials
:link: credentials
:link-type: doc

Which products need an account, and how to set each one up.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public function, grouped by subpackage.
:::

:::{grid-item-card} {octicon}`tools` Contributing
:link: contributing
:link-type: doc

The pixi tasks, the four test tiers, and how to add a product.
:::
::::

## Design in one paragraph

Every loader takes the same `aoi` (a bounding-box tuple, a shapely geometry, a
GeoDataFrame in any CRS, or an `odc.geo.GeoBox`) and the same `time` (anything
pandas or STAC understands). Search and load are separate calls. Results are
lazy, carry their CRS on both the `.rio` and `.odc` accessors, and carry
`source`, `license` and `data_citation` in `.attrs`. Products with more than
one route expose them through `source=`, so a Planetary Computer outage is one
keyword away from an alternative. Credentials are checked before any network
request, and the error says exactly how to fix them. Processing that is one
line of xarray (a normalized difference, a threshold) is written out rather than
wrapped, so nothing about the pipeline is hidden. See [Concepts](concepts.md)
for the whole contract.

## Citing

```{code-block} text
Gagliano, E. (2024). easysnowdata [Software]. Zenodo.
https://doi.org/10.5281/zenodo.14741502
```

Each product also carries the citation of the data it serves in
`ds.attrs["data_citation"]`, and the catalog page for that product repeats it.

```{toctree}
:hidden:
:caption: Getting started

installation
concepts
credentials
migration
faq
```

```{toctree}
:hidden:
:caption: Data

auto_examples/index
catalog/index
status
```

```{toctree}
:hidden:
:caption: Reference

api/index
contributing
releasing
changelog
```
