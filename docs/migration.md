# Migrating from 0.0.x

Version 0.2 reorganised easysnowdata around **theme modules** — `snow`,
`optical`, `sar`, `land`, `terrain`, `climate`, `hydro`, `stations` — each with
one `load()` per product, and kept every old name as a deprecation shim that
forwarded to its replacement. **Version 0.3 removed the shims**, as their
warnings said it would: `easysnowdata.remote_sensing`, `hydroclimatology`,
`topography`, `automatic_weather_stations` and `utils` no longer exist.

If you still run 0.0.x code, install `easysnowdata==0.2.*`, turn deprecation
warnings on, and let the library tell you what to change:

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning)
```

```text
easysnowdata.topography.get_chili is deprecated since easysnowdata 0.1.0 and
will be removed in 0.3.0. Use easysnowdata.terrain.chili.load instead. …
```

Each warning fires once per name per process, so a notebook gives you the whole
list on one pass. Then upgrade to 0.3 with the table below.

## Also gone in 0.3: the band-arithmetic helpers

`processing.ndsi`, `ndvi`, `ndwi`, `ndbi`, `evi`, `normalized_difference`,
`binary_snow`, `rgb`, `stretch_percentile`, `stretch_clahe` and `plotting.rgb`
were removed because they hid one-line operations behind names. Write them out:

```python
ndsi = (s2["green"] - s2["swir16"]) / (s2["green"] + s2["swir16"])
snow = (ndsi_byte >= 40).where(ndsi_byte <= 100)      # MODIS/VIIRS: >100 are sentinels
rgb = s2[["red", "green", "blue"]].to_array("band").clip(0, 0.3) / 0.3
rgb.isel(time=0).plot.imshow(rgb="band")
```

The masks (`apply_scl_mask`, `apply_fmask`), `harmonize_s2_baseline`,
`scale_offset`, `decode_udm2`, the SAR helpers and the water-year helpers stay.

## Changed values in 0.3: the DEM-computed local incidence angle

`sar.sentinel1.local_incidence_angle(..., source="dem")` (and `source="gee"`)
had the range-facing slope term with the wrong sign in 0.2, so slopes tilted
toward the radar came out with a *larger* angle than slopes tilted away. The
same route also assumed one nominal heading and a constant 39° incidence angle;
it now takes the heading, look direction and across-swath incidence field from
a representative scene of each track, and raises rather than guess when no
scene of a requested track or pass exists over the AOI. Recompute anything
derived from it. The OPERA static-layer route (`source="opera-static"`, the
default) reads a published product and was never affected by the sign error.

## Changed shape in 0.3: one raster per relative orbit, never fused

The local incidence angle belongs to one acquisition geometry, so no route
mixes tracks any more. `local_incidence_angle(aoi)` without `relative_orbit=`
logs an INFO line and returns a `relative_orbit` dimension with one raster per
track that crosses the AOI (with `sat:orbit_state`, and on the computed routes
`platform_heading` and `look_azimuth`, as coordinates along it); the OPERA
route used to fuse every burst with a max, and the DEM route used to pick the
busiest track of one pass direction. `relative_orbit=137` still returns a plain
`(y, x)` raster with the geometry in the attrs. `orbit_state=` now defaults to
`None` (both directions) and only restricts which tracks are returned.
`scene_geometry()` is kept; `track_geometries()` returns every track's geometry
keyed by relative orbit.

The OPERA backscatter route (`load(..., source="opera-rtc-s1")`) groups the
bursts within each track before grouping by solar day, so a time step never
holds two tracks, and tags each step with `sat:relative_orbit` (parsed from the
burst id; OPERA items carry no orbit properties).

DEMs are now resampled **bilinearly** whenever `crs=` or `grid_resolution=`
puts them on a new grid (`resampling=` to choose otherwise); 0.2 copied the
nearest source pixel, which folds the 1 arc-second staircase into every slope.

## The renames

| 0.0.x | 0.2 |
| --- | --- |
| `remote_sensing.get_forest_cover_fraction` | `land.forest_cover.load` |
| `remote_sensing.get_seasonal_snow_classification` | `snow.snow_classification.load` |
| `remote_sensing.get_seasonal_mountain_snow_mask` | `snow.mountain_snow_mask.load` |
| `remote_sensing.get_esa_worldcover` | `land.landcover.load` |
| `remote_sensing.get_nlcd_landcover` | `land.nlcd.load` |
| `remote_sensing.Sentinel2` | `optical.sentinel2.load` |
| `remote_sensing.Sentinel1` | `sar.sentinel1.load` |
| `remote_sensing.HLS` | `optical.hls.load` |
| `remote_sensing.MODIS_snow` | `snow.modis.load` |
| `topography.get_copernicus_dem` | `terrain.dem.load` |
| `topography.get_chili` | `terrain.chili.load` |
| `hydroclimatology.get_huc_geometries` | `hydro.basins.huc` |
| `hydroclimatology.get_hydroBASINS` | `hydro.basins.hydrobasins` |
| `hydroclimatology.get_grdc_major_river_basins_of_the_world` | `hydro.basins.grdc_major` |
| `hydroclimatology.get_grdc_wmo_basins` | `hydro.basins.grdc_wmo` |
| `hydroclimatology.get_era5` | `climate.era5.load` |
| `hydroclimatology.get_snodas` | `snow.snodas.load` |
| `hydroclimatology.get_ucla_snow_reanalysis` | `snow.ucla_sr.load` |
| `hydroclimatology.get_koppen_geiger_classes` | `climate.koppen_geiger.load` |
| `automatic_weather_stations.StationCollection` | `stations.inventory` and `stations.load` |
| `utils.get_water_year_start` | `processing.wateryear.water_year_start` |
| `utils.datetime_to_DOWY` | `processing.wateryear.day_of_water_year` |
| `utils.datetime_to_WY` | `processing.wateryear.water_year` |

## Four changes that are not just a rename

### `bbox_input=` is `aoi=`, and it takes more

```python
esd.remote_sensing.get_esa_worldcover(bbox_input=(-121.9, 46.7, -121.5, 47.0))
esd.land.landcover.load((-121.9, 46.7, -121.5, 47.0))
```

`aoi` is positional and accepts a `(west, south, east, north)` tuple in
EPSG:4326, a shapely geometry or GeoJSON-like mapping, a GeoDataFrame or
GeoSeries in any CRS, an `odc.geo.geobox.GeoBox`, or `None` for the whole
globe. See [Concepts](concepts.md).

### The classes are functions

`Sentinel2`, `Sentinel1`, `HLS` and `MODIS_snow` were classes you constructed
and then read a `.data` attribute from. They are `load()` functions returning
the `xarray.Dataset` directly:

```python
s2 = esd.remote_sensing.Sentinel2(bbox, start_date="2023-08-01").data
s2 = esd.optical.sentinel2.load(bbox, time="2023-08-01/2023-08-31")
```

Searching without loading is `optical.sentinel2.search()`.

### Class tables are CF flag attributes

`class_info`, `cmap` and `example_plot` are gone. Categorical products now
carry the standard `flag_values` / `flag_meanings` / `flag_colors` attributes,
which any CF-aware tool understands, and this package draws them:

```python
lc = esd.land.landcover.load(aoi)
esd.plotting.categorical(lc)          # also: colormap_from_flags, legend_handles
```

### Some defaults moved to a better source

Where a product has more than one route, 0.2 defaults to the archive of record
rather than whichever mirror was wired up first. The shim keeps the old route,
so this is the change most likely to surprise you:

| product | 0.0.x route | 0.2 default | keep the old one with |
| --- | --- | --- | --- |
| snow classification | hosted COG | NSIDC-0768 | `source="hosted-cog"` |
| NLCD | 2021 release | Annual NLCD, 1985–2024 | `source="gee"` |
| MODIS snow | Planetary Computer | NSIDC | `source="planetary-computer"` |
| SNODAS | GEE / Climate Engine | NSIDC G02158 | `source="gee-climate-engine"` |

Several of these need *fewer* credentials than before: `hydro.basins.huc` reads
the public USGS WBD service instead of Earth Engine, `climate.era5.load` serves
hourly ERA5 from ARCO-ERA5 without an Earth Engine account, and
`optical.hls.load` gained a credential-free Planetary Computer route. Run
`esd.auth.status()` to see what you actually still need.

## Stations

`automatic_weather_stations.StationCollection` read a frozen CSV archive. It is
now a shim over `easysnowdata.stations`, which talks to the five network APIs
(AWDB, CDEC, DataBC, NVE, Yukon AquaCache) and to the daily archive that
[`global_snow_networks`](https://github.com/egagli/global_snow_networks)
publishes:

```python
gdf = esd.stations.inventory(aoi=aoi)                  # GeoDataFrame
ds = esd.stations.load(gdf, variables=["swe", "snwd"]) # xarray Dataset
ds = esd.stations.archive.load(aoi=aoi)                # the bulk daily archive
```

One thing to know if you used the old archive for **air temperature**: it holds
uncorrected SNOTEL values for roughly 2004–2024, about 1.1 °C warm, because its
updater only ever re-fetched the last ten days and never revisited history
after NRCS bias-corrected the network. The live routes above do not have this
problem. SWE and snow depth are unaffected.

## Still stuck?

Every product's page under [Catalog](catalog/index.md) lists its sources,
variables and credentials, and the [gallery](auto_examples/index.rst) has a
runnable example for each. If a shim does something the replacement cannot,
that is a bug worth
[reporting](https://github.com/egagli/easysnowdata/issues) before 0.3 removes
it.
