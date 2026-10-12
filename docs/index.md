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

Forty products — station observations from SNOTEL and the other AWDB networks,
CDEC, BC, NVE, and Yukon; Sentinel-1 RTC, Sentinel-2, HLS, and PlanetScope
imagery; MODIS and VIIRS snow cover; SNODAS and UCLA snow water equivalent and
depth; the Sturm & Liston snow classes and a mountain snow mask; five DEMs,
CHILI, and a hillshade; WorldCover, NLCD, and forest cover; basins; ERA5 and
Köppen-Geiger climate; and the boundaries that frame them: countries, states,
counties, mountain ranges, and glacier outlines — behind one API that takes an area of
interest and a time range, returns lazy Dask-backed xarray objects (and
GeoDataFrames for vector products), and never downloads more than it has to.

```{button-ref} getting_started
:color: primary
:expand:
Get started
```
:::
::::

```{code-block} python
import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)                      # Mount Rainier; any AOI form works

inv_gdf = esd.stations.inventory(aoi, daily_only=True)      # which snow stations are here
obs_ds = esd.stations.load(inv_gdf, variables=["swe", "snwd"], time="2023-10/2024-09")

dem_da = esd.terrain.dem.load(aoi)                          # Copernicus GLO-30 …
dem_da = esd.terrain.dem.load(aoi, product="3dep")          # … or NASADEM, SRTM, 3DEP, ALOS, GEDTM30
s1_ds = esd.sar.sentinel1.load(aoi, "2024-03", units="dB")  # Sentinel-1 RTC
snodas_ds = esd.snow.snodas.load(aoi, "2024-03")            # SNODAS, no account needed

counties_gdf = esd.boundaries.admin.counties(aoi)           # Pierce and Lewis (US Census)
ranges_gdf = esd.boundaries.mountains.load(aoi)             # GMBA mountain ranges
glaciers_gdf = esd.boundaries.glaciers.load(aoi)            # RGI 7.0 (Earthdata username + password)

ax = esd.plotting.map(dem_da, cmap="terrain")               # equal aspect, scale bar, graticule
esd.plotting.add_outline(ax, glaciers_gdf)                  # vectors in the map's CRS
esd.plotting.timeseries(obs_ds["swe"])                      # calendar dates, units in [ ]
```

## Where to go

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started
:link-type: doc

Installation, credentials, the concepts every loader shares, and a tour of the modules.
:::

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

:::{grid-item-card} {octicon}`database` Dataset catalog and status
:link: catalog/index
:link-type: doc

Every product: its routes, resolution, credentials, licence, and weekly health.
:::

:::{grid-item-card} {octicon}`question` FAQ
:link: faq
:link-type: doc

Which product to use, accounts, memory, CRS, and what to do when a source is down.
:::

:::{grid-item-card} {octicon}`tools` Development
:link: development
:link-type: doc

The pixi tasks, the four test tiers, how to add a product, and the changelog.
:::
::::

## Design in one paragraph

Every loader takes the same `aoi` (a bounding-box tuple, a shapely geometry, a
GeoDataFrame in any CRS, or an `odc.geo.GeoBox`) and the same `time` (anything
pandas or STAC understands). Search and load are separate calls. Results are
lazy, carry their CRS on both the `.rio` and `.odc` accessors, and carry
`source`, `license`, and `data_citation` in `.attrs` — GeoDataFrames too, which
come back in EPSG:4326 holding the whole features that touch the AOI. Products with more than
one route expose them through `source=`, so a Planetary Computer outage is one
keyword away from an alternative. Credentials are checked before any network
request, and the error says exactly how to fix them. Processing that is one
line of xarray (a normalized difference, a threshold) is written out rather than
wrapped, so nothing about the pipeline is hidden. See [Concepts](concepts.md)
for the whole contract.

## Citing

```{code-block} text
Gagliano, E. (2024). easysnowdata [Software]. Zenodo.
https://doi.org/10.5281/zenodo.14741501
```

Each product also carries the citation of the data it serves in
`ds.attrs["data_citation"]`, and the catalog page for that product repeats it.

```{toctree}
:hidden:

Getting started <getting_started>
Example gallery <auto_examples/index>
API reference <api/index>
Dataset catalog and status <catalog/index>
FAQ <faq>
Development <development>
```
