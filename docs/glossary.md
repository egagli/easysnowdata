# Glossary

The acronyms and terms used across these docs, grouped by what they name. Data
providers first, because most product names are built from them.

## Agencies and data providers

```{glossary}
NOAA
  The U.S. National Oceanic and Atmospheric Administration. Three of its
  offices publish the climate-mode indices in `esd.climate.indices`:
  {term}`CPC`, {term}`NCEI` and {term}`PSL`.

CPC
  NOAA's **Climate Prediction Center**, part of the National Weather Service's
  National Centers for Environmental Prediction (NCEP). It issues the official
  ENSO advisories and maintains the operational {term}`RONI`, {term}`ONI`,
  Niño 3.4, {term}`AO`, {term}`NAO` and {term}`PNA` series.
  <https://www.cpc.ncep.noaa.gov>

NCEI
  NOAA's **National Centers for Environmental Information**, the agency's data
  archive. It maintains the {term}`ERSST` sea-surface-temperature record and
  the {term}`PDO` and {term}`AMO` indices computed from it.
  <https://www.ncei.noaa.gov>

PSL
  NOAA's **Physical Sciences Laboratory**, a research lab in Boulder,
  Colorado (formerly the Earth System Research Laboratory's Physical Sciences
  Division). It maintains {term}`MEI.v2` and a "climate indices correlation
  page" that republishes many indices in one file format; `source="psl"` in
  `esd.climate.indices` reads those copies. <https://psl.noaa.gov>

NOHRSC
  The National Weather Service's **National Operational Hydrologic Remote
  Sensing Center**, which produces {term}`SNODAS`.

NRCS
  The USDA **Natural Resources Conservation Service**. Its National Water and
  Climate Center runs {term}`SNOTEL`, {term}`SCAN` and the western snow-course
  program, and serves them through {term}`AWDB`.

AWDB
  The NRCS **Air and Water Database** and its REST API, the source for
  `esd.stations` products from {term}`SNOTEL`, {term}`SCAN` and snow courses.

SNOTEL
  **Snow Telemetry**, the NRCS network of automated snow stations in the
  western US and Alaska, most with a snow pillow ({term}`SWE`), snow depth,
  precipitation and temperature. Daily records at most sites begin around 1980.

SCAN
  The NRCS **Soil Climate Analysis Network**, automated soil-moisture and
  weather stations, some of which measure snow.

CDEC
  The **California Data Exchange Center**, run by the California Department of
  Water Resources, which serves California's snow pillows and snow courses.

DataBC
  The Government of British Columbia's open-data service, which serves the
  province's automated snow weather stations and manual snow surveys.

NVE
  The **Norwegian Water Resources and Energy Directorate**, whose HydAPI serves
  Norway's snow stations (it needs a free API key).

ECMWF
  The **European Centre for Medium-Range Weather Forecasts**, which produces
  {term}`ERA5`.
```

## Climate modes

```{glossary}
ENSO
  The **El Niño-Southern Oscillation**: the coupled warming (El Niño) and
  cooling (La Niña) of the tropical Pacific, and the atmospheric response. The
  largest single source of year-to-year variation in western-US snowpack.

Niño 3.4
  The region of the tropical Pacific (5°N-5°S, 170°W-120°W) whose
  sea-surface-temperature anomaly defines most ENSO indices.

ONI
  The **Oceanic Niño Index**: the 3-month running mean of the Niño 3.4
  anomaly. CPC's official ENSO index until February 2026.

RONI
  The **Relative Oceanic Niño Index**: like the {term}`ONI`, but with the
  tropical-mean sea-surface-temperature anomaly subtracted first, so that a
  warming ocean does not read as a permanent El Niño. CPC's official ENSO
  index since February 2026.

MEI.v2
  The **Multivariate ENSO Index**, version 2, from {term}`PSL`: ENSO measured
  from five variables (sea level pressure, sea-surface temperature, surface
  winds and outgoing longwave radiation) rather than from temperature alone.

PDO
  The **Pacific Decadal Oscillation**, the leading pattern of North Pacific
  sea-surface temperature. It changes phase on a scale of decades, unlike ENSO.

AO
  The **Arctic Oscillation**, the leading pattern of Northern Hemisphere
  sea-level pressure north of 20°N.

NAO
  The **North Atlantic Oscillation**, the pressure seesaw between Iceland and
  the Azores.

PNA
  The **Pacific/North American pattern**, a wave train of pressure anomalies
  from the central Pacific over North America; the main route by which ENSO
  reaches the western US.

AMO
  The **Atlantic Multidecadal Oscillation**, the multidecadal swing in North
  Atlantic sea-surface temperature.

ERSST
  NCEI's **Extended Reconstructed Sea Surface Temperature** record (version 5
  for the PDO and AMO; CPC's ENSO indices moved to version 6 in 2026).
```

## Snow and hydrology

```{glossary}
SWE
  **Snow water equivalent**: the depth of water the snowpack would make if it
  melted. Station SWE comes back in centimetres; every variable carries its
  units in its `units` attribute.

April 1 SWE
  SWE on April 1, the traditional index of a western-US winter's snowpack,
  close to the date of peak accumulation at many stations.

water year
  The 12 months that hold one snow season, named for the year they end in. In
  the northern hemisphere water year 2024 runs from 1 October 2023 to
  30 September 2024. See `esd.processing.wateryear`.

SNODAS
  The **Snow Data Assimilation System**, NOHRSC's daily 1 km model of SWE and
  snow depth over the conterminous US.

HUC
  A **Hydrologic Unit Code**, the USGS identifier for a watershed in the
  Watershed Boundary Dataset (WBD); more digits mean a smaller basin.
```

## Remote sensing and geospatial

```{glossary}
RTC
  **Radiometric terrain correction**: SAR backscatter corrected for the
  brightening and darkening that topography causes, so pixels are comparable
  across slopes.

HLS
  **Harmonized Landsat and Sentinel-2**, NASA's surface reflectance from both
  missions on one 30 m grid.

DEM
  A **digital elevation model**.

NLCD
  The USGS **National Land Cover Database** for the United States.

ERA5
  ECMWF's fifth-generation global atmospheric reanalysis, hourly from 1940.

RGI
  The **Randolph Glacier Inventory**, the global inventory of glacier outlines.

GMBA
  The **Global Mountain Biodiversity Assessment**, whose Mountain Inventory v2
  supplies the mountain-range outlines in `esd.boundaries.mountains`.

AOI
  **Area of interest**: the first argument of every loader. See
  {ref}`Concepts <concepts-aoi>`.

STAC
  The **SpatioTemporal Asset Catalog** specification, which Planetary Computer,
  Earth Search and NASA's CMR use to describe imagery; easysnowdata searches it
  for most image products.

COG
  A **Cloud-Optimized GeoTIFF**: a GeoTIFF laid out so a reader can fetch only
  the tiles it needs over HTTP.

Zarr
  A chunked array format for cloud storage, read lazily by xarray.
```
