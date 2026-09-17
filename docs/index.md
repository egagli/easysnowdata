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

Twenty-eight products — station observations, SAR and optical imagery, snow
cover and SWE, DEMs, land cover, basins and reanalysis — behind one API that
takes an area of interest and a time range, returns lazy Dask-backed xarray
objects, and never downloads more than it has to.

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

inv = esd.stations.inventory(aoi, daily_only=True)          # which snow stations are here
obs = esd.stations.load(inv, variables=["swe", "snwd"], time="2023-10/2024-09")

dem = esd.terrain.dem.load(aoi)                             # Copernicus GLO-30
s1 = esd.sar.sentinel1.load(aoi, "2024-03", units="dB")     # Sentinel-1 RTC
swe = esd.snow.snodas.load(aoi, "2024-03")                  # SNODAS, no account needed
```

## Where to go

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`telescope` Example gallery
:link: auto_examples/index
:link-type: doc

One executed script per product, with the figure it produces.
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
request, and the error says exactly how to fix them. See
REVAMP_PLAN §2 for the whole contract.

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
faq
```

```{toctree}
:hidden:
:caption: Data

auto_examples/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
notebooks
contributing
changelog
```
