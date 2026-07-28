# Clients

This folder contains API client modules for retrieving snow observation data
from external data sources.  Each client is responsible for one data source
and exposes a consistent interface for fetching stations, metadata, and
time-series data.

> The normative client contract (method signatures, record schema, units,
> intervals, error behaviour) lives in [DESIGN.md](../DESIGN.md) §3.  This
> file documents each client's API surface and source-specific quirks;
> where they disagree, `DESIGN.md` wins.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [AWDBClient — USDA NRCS AWDB REST API](#2-awdbclient)
3. [CDECClient — California Data Exchange Center](#3-cdecclient)
4. [DataBCClient — BC Data Catalogue](#4-databcclient)
5. [NVEClient — NVE HydAPI (Norway)](#5-nveclient)
6. [YukonClient — Yukon Water Data / AquaCache](#6-yukonclient)
7. [Error Handling](#7-error-handling)
8. [Adding a New Client](#8-adding-a-new-client)

---

## 1. Design Philosophy

See [DESIGN.md](../DESIGN.md) §3 for the full contract.  Summary:

- **One client per data source.**  Each client encapsulates HTTP requests,
  batching, retry logic, HTML/JSON parsing, and response normalisation.
- **`get_data()` returns flat record dicts.**  The standardized interface
  returns `list[dict]`; DataBC additionally has a pandas-DataFrame
  convenience layer for its bulk CSV files.
- **Metric-first normalisation.**  SWE and snow depth are returned in
  centimetres; every other variable is converted to metric too (°C, mm,
  km/h, …) — no imperial passthrough.
- **Handle limits internally.**  API rate limits, value-count limits, and URL
  length limits are managed by the client.
- **Fail clearly, never silently.**  All source errors raise
  `{Client}Error(Exception)` with descriptive messages; unknown variables
  and unsupported intervals also raise rather than falling back.
- **Sub-daily records carry a `datetime` key** alongside `date`, so hourly
  observations stay distinguishable.
- **Data flags are opt-in.**  Pass `include_flags=True` to `get_data()` to
  receive per-value quality flags.  Flags are NOT stored in per-station CSVs
  but are available for QC analysis.
- **Variables and flags are documented in-module.**  Each client exports
  `SENSORS`/`VARIABLES` and `DATA_FLAGS` dicts so downstream code can
  reference variable metadata without hardcoding.

---

## 2. AWDBClient

**Module:** `clients.awdb.awdb_client`
**Class:** `AWDBClient`
**API:** [AWDB REST API v1](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html)

The AWDB (Air and Water Database) REST API provides access to data from
SNOTEL, SNOLite, SCAN, COOP, Manual SNOTEL, snow courses, streamflow gauges,
reservoirs, and more.

```python
from clients.awdb import AWDBClient
client = AWDBClient()
```

### Networks supported

| Network | Code | Description |
|---|---|---|
| SNOTEL | `SNTL` | Automated snow pillow + weather stations, western U.S. |
| SNOLite | `SNTLT` | Simplified lower-cost SNOTEL variant |
| Manual SNOTEL | `MSNT` | Legacy / transitional sites (includes some BC and CCSS) |
| SCAN | `SCAN` | Soil climate network with snow sensors |
| COOP | `COOP` | NWS cooperative observer snow sites |
| Snow Course | `SNOW` | Manual snow courses (~2,700, semi-monthly/monthly WTEQ+SNWD, incl. partner courses) |
| Aerial Marker | `MPRC` | Aerial snow markers (~260, periodic depth readings) |

### Key data variables

All values are converted to metric in-client (DESIGN.md §3.5); AWDB serves
them natively in inches / °F / mph.

| Element | Description | Emitted units |
|---|---|---|
| `WTEQ` | Snow Water Equivalent | cm (from inches) |
| `SNWD` | Snow Depth | cm (from inches) |
| `TOBS` / `TMAX` / `TMIN` | Air temperature | °C (from °F) |
| `PREC` / `PRCP` / `PRCPSA` | Precipitation | mm (from inches) |
| `RHUM` | Relative humidity | % |
| `WSPDV` / `WSPDX` | Wind speed / gust | km/h (from mph) |
| `WDIRV` | Wind direction | degrees |
| `SRADV` | Solar radiation | W/m² |

`VARIABLES[code]` carries both `units` (native) and `output_units`
(emitted).  `DATA_FLAGS` is an empty dict — the AWDB REST API returns no
per-value QC flags.

### Constructor

```python
AWDBClient(
    base_url: str = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1",
    timeout: int = 180,
    max_retries: int = 3,
    backoff: int = 6,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_stations(networks, states, active_only, ...)` → `list[dict]`

Returns basic station identification fields (no element inventory).

```python
stations = client.get_stations(networks=["SNTL", "SNTLT"], states=["CO"])
```

#### `get_metadata(triplets, elements, durations, ...)` → `list[dict]`

Returns full station metadata including the `stationElements` inventory
(what variables are measured, at what resolution, and for what period).

```python
meta = client.get_metadata(
    triplets=["303:CO:SNTL", "713:CO:SNTL"],
    elements=["WTEQ", "SNWD"],
    durations=["DAILY"],
)
```

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Standardized data fetch — returns a **flat** list of observation records.
Accepts native element codes (`"WTEQ"`) or standardized types (`"swe"`).

```python
records = client.get_data(
    station_ids=["303:CO:SNTL"],
    variables=["swe", "snwd"],   # or ["WTEQ", "SNWD"]
    interval="daily",
    begin_date="2023-10-01",
    end_date="2024-09-30",
)
# records[0]
# → {"station_id": "303:CO:SNTL", "date": "2023-10-01",
#    "variable": "WTEQ", "type": "swe", "value": 5.08,
#    "units": "cm", "interval": "daily"}
```

The internal batching method `_get_data_awdb()` respects the 500k-value limit.

#### `get_normals(triplets, elements, duration, normal_period, ...)` → `list[dict]`

Fetch climatological normals (1991–2020, 1981–2010, or 1971–2000).

```python
norms = client.get_normals("303:CO:SNTL", ["WTEQ"], normal_period="1991-2020")
```

### Notes and gotchas

- **500k value limit:** `n_stations × n_elements × n_days ≤ 500,000`.  The
  client auto-splits requests.
- **Triplet format:** `{stationId}:{stateCode}:{networkCode}` — case-sensitive.
- **Canadian stations:** Accessible with state codes `BC` and `AB` (58 and 16
  snow stations respectively).  AWDB returns **no** Yukon stations with snow
  elements — for Yukon coverage use the [Yukon client](#6-yukonclient).
- **Missing values:** `None` in parsed response.
- **Units:** all elements are converted to metric in-client — WTEQ/SNWD
  inches→cm, temperatures °F→°C, precipitation inches→mm, wind mph→km/h.
- **Unknown variables / intervals raise `AWDBError`** rather than
  silently fetching everything or assuming daily.
- **Hourly records carry a `datetime` key** alongside `date`.

---

## 3. CDECClient

**Module:** `clients.cdec.cdec_client`
**Class:** `CDECClient`
**Source:** [CDEC — California Data Exchange Center](https://cdec.water.ca.gov)
**Operator:** California Department of Water Resources (CA DWR)

Provides access to California Cooperative Snow Surveys (CCSS) data — both
automated snow pillow stations (daily) and manual snow course sites
(periodic).

```python
from clients.cdec import CDECClient
client = CDECClient()
```

### Snow sensors

```python
from clients.cdec.cdec_client import SENSORS, DATA_FLAGS, DURATION_CODES
```

| Sensor | Short name | Variable | Description |
|---|---|---|---|
| 3 | SNOW WC | `swe_raw` | Raw SWE from snow pillow (inches → cm) |
| 18 | SNOW DP | `snwd` | Snow depth, ultrasonic (inches → cm) |
| 82 | SNO ADJ | `swe` | **Preferred SWE** — quality-controlled, offset-adjusted version of sensor 3 |

The registry also covers the met suite at CCSS stations (all converted to
metric in-client): precipitation 2/45/16 (in → mm), air temperature
4/30/31/32 (°F → °C), relative humidity 12, wind 9 (mph → km/h) / 10,
solar 103, and soil moisture 283/310/286/287.
`get_data(variables=None)` defaults to the snow sensors.

**SWE vs. SNO ADJ:** Sensor 82 (SNO ADJ) is the revised version of sensor 3
(raw SWE), with a calibration offset applied after manual QC.  Both represent
SWE from the same snow pillow.  Sensor 82 is always preferred and is stored
as `wteq_cm` in per-station CSVs.  If sensor 82 is unavailable, sensor 3 is
used as fallback.

### Data flags

| Flag | Meaning |
|---|---|
| ` ` (space) | Unreviewed / provisional |
| `A` | Precipitation accumulation period |
| `L` | Awaiting observer response |
| `N` | Error in data |
| `c` | Calculated (gridded precipitation) |
| `e` | Estimated |
| `o` | Calibration offset applied |
| `q` | New rating table applied |
| `r` | Revised (most sensor 82 values carry this flag) |
| `s` | New shift applied |
| `t` | Trace of precipitation |
| `v` | Out of valid range |

### Duration codes

| Code | Meaning |
|---|---|
| `D` | Daily |
| `H` | Hourly |
| `M` | Monthly (not available for sensors 3/18/82) |
| `E` | Event (sub-hourly telemetry) |

### Constructor

```python
CDECClient(
    timeout: int = 60,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_snow_courses()` → `list[dict]`

Returns the official CCSS manual snow course list (~260 stations) from the
CDEC SnowCourses report.

Fields: `station_id`, `course_number`, `name`, `elevation_ft`, `latitude`,
`longitude`, `april1_avg_swe_in`, `measuring_agency`, `is_snow_course`,
`station_url`.

```python
courses = client.get_snow_courses()
# courses[0] → {"station_id": "QUA", "name": "QUAKING ASPEN",
#               "april1_avg_swe_in": 12.3, "measuring_agency": "CA DWR", ...}
```

#### `get_snow_pillows()` → `list[dict]`

Returns the automated snow pillow station list (~137 active) from the CDEC
SnowSensors report.

Fields: `station_id`, `name`, `elevation_ft`, `latitude`, `longitude`,
`april1_avg_swe_in`, `operator`, `is_snow_pillow`, `has_daily_swe`,
`station_url`.

#### `get_stations(sensors, active_only)` → `list[dict]`

Queries the CDEC station search for each sensor number and merges results.
Also supplements with the snow course and pillow lists to set `is_snow_course`,
`is_snow_pillow`, `has_daily_swe`, `has_daily_snwd` flags on each station dict.

```python
# All stations with any snow sensor
stations = client.get_stations(sensors=(3, 18, 82))

# Filter to those with daily data — these station-dict flags are what the
# combined inventory's advertised has_daily_swe/has_daily_snwd are built from
daily = [s for s in stations if s["has_daily_swe"] or s["has_daily_snwd"]]
```

#### `get_metadata(station_id)` → `dict`

Scrapes the CDEC staMeta HTML page for a single station.

Fields: `station_id`, `name`, `elevation_ft`, `river_basin`, `county`,
`hydrologic_area`, `nearby_city`, `latitude`, `longitude`, `operator`,
`maintenance`, `sensor_inventory` (list of sensor dicts), `station_url`.

```python
meta = client.get_metadata("QUA")
for s in meta["sensor_inventory"]:
    print(s["sensor_num"], s["sensor_description"], s["data_available"])
```

Note: `get_metadata()` requires one HTTP request per station.  For bulk
metadata, call `get_stations()` first (which uses the bulk HTML reports)
and only call `get_metadata()` for stations requiring the full sensor inventory.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Standardized data fetch — returns a **flat** list of observation records.
Accepts sensor short names (`"SNO ADJ"`) or standardized types (`"swe"`).
Sensor 82 (SNO ADJ) takes priority over sensor 3 (SNOW WC) per date.
Values are converted from inches to centimetres.

```python
records = client.get_data(
    station_ids=["QUA", "BLC"],
    variables=["swe", "snwd"],   # or sensor short names
    interval="daily",
    begin_date="2023-10-01",
    end_date="2024-09-30",
    include_flags=True,
)
# records[0]
# → {"station_id": "QUA", "date": "2023-10-01",
#    "variable": "SNO ADJ", "type": "swe", "value": 5.08,
#    "units": "cm", "interval": "daily", "flag": "r"}
```

The internal fetch method `_get_data_cdec()` calls the JSONDataServlet directly.

### Data availability notes

- The JSON data service accepts multiple comma-separated station IDs.
- **Monthly duration** (`M`) returns empty results for sensors 3, 18, 82.
  Use daily (`D`) for all snow sensor data.
- **Hourly data** is available for sensors 3 and 18 at most automated
  stations; hourly/event records carry a `datetime` key and SWE priority
  resolves per timestamp.
- **Unknown variables / intervals raise `CDECError`** rather than
  silently expanding to all sensors / falling back to daily.
- Station IDs are 2–5 uppercase alphanumeric characters (e.g. `QUA`, `BLC`).

### Station URLs

```
https://cdec.water.ca.gov/dynamicapp/staMeta?station_id={ID}
```

---

## 4. DataBCClient

**Module:** `clients.databc.databc_client`
**Class:** `DataBCClient`
**Source:** [BC Data Catalogue](https://catalogue.data.gov.bc.ca) + [BC env.gov.bc.ca CSV files](https://www.env.gov.bc.ca/wsd/data_searches/snow/asws/data/)
**Operator:** BC Ministry of Environment (BC ENV)

Provides access to BC snow survey data — both Automated Snow Weather Stations
(ASWS, full meteorological suite) and Manual Snow Survey sites (MSS, periodic
surveys).  Also fetches ASWS station photos from the AQRT BCMOE portal.

```python
from clients.databc import DataBCClient
client = DataBCClient()
```

### Station types

| Type | ID suffix | Description | Data |
|---|---|---|---|
| ASWS | ends in `P` (e.g. `1A01P`) | Automated snow pillow + weather station | Daily SWE, snow depth, air temp, precip, wind, humidity, pressure |
| MSS | no `P` (e.g. `1A06A`, `1A10`) | Manual snow course | Periodic SWE (mm), depth (cm), density (%) |

### Variables

```python
from clients.databc.databc_client import VARIABLES, DATA_FLAGS
```

All ASWS variables share the same wide-format CSV structure.  The **16:00 UTC
reading** is used as the canonical daily value (~08:00 PST / 09:00 PDT).
Pass `daily_only=False` to the private ASWS DataFrame methods that accept it to
retrieve all hourly readings instead (a `datetime` column rather than
`date`).  Two methods have no `daily_only` parameter because they are
interval-specific: `_get_asws_daily_data()` (always daily, SWDaily.csv)
and `_get_asws_sw_hourly_data()` (always hourly, SW.csv).

| Variable | Units | Method | Archive? | Notes |
|---|---|---|---|---|
| `swe_mm` | mm | `_get_asws_daily_data()` | Yes | Daily pre-aggregated (SWDaily.csv) |
| `swe_mm` | mm | `_get_asws_sw_hourly_data()` | Yes | Hourly raw pillow (SW.csv) |
| `snwd_cm` | cm | `_get_asws_sd_data()` | Yes | Snow depth sensor (SD.csv) |
| `air_temp_degc` | °C | `_get_asws_ta_data()` | Yes | Air temperature (TA.csv) |
| `precip_cumul_mm` | mm | `_get_asws_pc_data()` | Yes | Cumulative precipitation (PC.csv) |
| `baro_press_hpa` | hPa | `_get_asws_pa_data()` | **No** | Barometric pressure (PA.csv) |
| `wind_dir_deg` | ° | `_get_asws_ud_data()` | **No** | Wind direction (UD.csv) |
| `wind_spd_kmh` | km/h | `_get_asws_us_data()` | **No** | Wind speed (US.csv) |
| `wind_spd_peak_kmh` | km/h | `_get_asws_up_data()` | **No** | Wind gust speed (UP.csv) |
| `wind_run_km` | km | `_get_asws_ur_data()` | **No** | Cumulative wind run (UR.csv) |
| `rh_pct` | % | `_get_asws_xr_data()` | **No** | Relative humidity (XR.csv) |
| `swe_mm` + `snwd_cm` + `air_temp_degc` + `precip_cumul_mm` | mixed | `_get_asws_combined_data(id)` | — | Per-station combined file (SnowAll/) |
| `swe_mm` | mm | `_get_mss_survey_data()` | Yes | MSS periodic survey (Water Equiv.) |
| `snwd_cm` | cm | `_get_mss_survey_data()` | Yes | MSS periodic survey (Snow Depth) |
| `density_pct` | % | `_get_mss_survey_data()` | Yes | MSS periodic only |
| `snow_line_m` | m | `_get_mss_survey_data()` | Yes | MSS periodic only |

"No archive" variables have only current-season data (the archive files do
not exist for PA, UD, US, UP, UR, XR).

### Data flags (MSS)

The `survey_code` field in MSS data acts as a quality flag.

| Flag | Meaning |
|---|---|
| `` (empty) | Normal data quality |
| `PROBLEM` | Data quality problem noted by surveyor |
| `ESTIMATE` | Estimated value |
| `EXTRAPOLATED` | Extrapolated from nearby site |

ASWS data does not include per-value quality flags.

### Constructor

```python
DataBCClient(
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 5,
    session: requests.Session | None = None,
)
```

The client maintains two internal HTTP sessions: one for data/WFS endpoints
(`env.gov.bc.ca`, `openmaps.gov.bc.ca`) and a separate session for the AQRT
BCMOE portal (`bcmoe-prod.aquaticinformatics.net`) used for station photos.

### Key methods

#### `get_asws_stations(active_only)` → `list[dict]`

Returns ASWS station locations from the BC OpenMaps WFS.

Fields: `location_id`, `name`, `elevation_m`, `latitude`, `longitude`,
`status`, `operator`, `camera_url` (or `None`), `station_type` (`"ASWS"`),
`station_url`.

```python
asws = client.get_asws_stations(active_only=True)
# asws[0] → {"location_id": "1A01P", "name": "Yellowhead Lake",
#             "elevation_m": 1860.0, "operator": "BC ENV", ...}
```

#### `get_mss_stations(active_only)` → `list[dict]`

Returns MSS site locations from the BC OpenMaps WFS.

Fields: `location_id`, `name`, `elevation_m`, `latitude`, `longitude`,
`status`, `station_type` (`"MSS"`), `station_url`.

#### `get_all_stations(active_only, bbox)` → `list[dict]`

Returns combined ASWS + MSS station list.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Standardized data fetch — returns a **flat** list of observation records.
`interval="daily"` fetches daily ASWS data (16:00 UTC canonical reading);
`interval="hourly"` / `"sub_daily"` fetch all hourly ASWS readings (records
carry a `datetime` key); `interval="periodic"` fetches MSS survey data.
Any other interval raises `DataBCError`.  `swe_mm` values are converted to
cm so all SWE is returned in cm.

```python
records = client.get_data(
    station_ids=["1A01P", "1E08P"],
    variables=["swe", "snwd"],
    interval="daily",
    begin_date="2022-10-01",
)
# records[0]
# → {"station_id": "1A01P", "date": "2022-10-01",
#    "variable": "swe_mm", "type": "swe", "value": 4.5,
#    "units": "cm", "interval": "daily"}
```

#### Private DataFrame layer (`_get_asws_*`, `_get_mss_survey_data`)

The bulk CSV parsing lives in private per-variable DataFrame methods
(`_get_asws_daily_data`, `_get_asws_sw_hourly_data`, `_get_asws_sd_data`,
`_get_asws_ta_data`, `_get_asws_pc_data`, `_get_asws_pa_data`,
`_get_asws_ud_data`, `_get_asws_us_data`, `_get_asws_up_data`,
`_get_asws_ur_data`, `_get_asws_xr_data`, `_get_asws_combined_data`,
`_get_mss_survey_data`).  `get_data()` is the public contract
(DESIGN.md §3.4); the private layer feeds it.  Notable sizes: the
TA archive is ~75 MB and SD ~37 MB — `get_data` for long hourly
periods loads them in full.

MSS surveys via the public API:

```python
records = client.get_data(
    station_ids=["1A06A", "1A10"],
    variables=["swe", "snwd", "density", "snow_line"],
    interval="periodic",
    include_flags=True,   # survey_code quality flag
)
```

#### `get_station_image_url(location_id)` → `str | None`

Fetches the station photo URL from the public AQRT BCMOE portal
(`bcmoe-prod.aquaticinformatics.net`).  The portal requires accepting a
one-time disclaimer; the client does this lazily and reuses the session.

Returns a direct `GetFileById` URL suitable for an `<img>` tag, or `None`
if the station has no photo or the portal is unreachable.

```python
url = client.get_station_image_url("1E08P")
# → "https://bcmoe-prod.aquaticinformatics.net/Data/GetFileById/12345"
```

**Note:** This performs 4 HTTP requests on first call (disclaimer acceptance)
and 2 requests per station thereafter.  For ~150 ASWS stations, expect
2–5 minutes total.  In `create_all_stations_geojson.py`, use the
`--skip-station-images` flag to skip this step.

### Data source URLs

| Data | URL |
|---|---|
| ASWS daily SWE | `…/data/SWDaily.csv` / `SW_DailyArchive.csv` |
| ASWS hourly SWE | `…/data/SW.csv` / `SW_Archive.csv` |
| ASWS snow depth | `…/data/SD.csv` / `SD_Archive.csv` |
| ASWS air temperature | `…/data/TA.csv` / `TA_Archive.csv` |
| ASWS precipitation | `…/data/PC.csv` / `PC_Archive.csv` |
| ASWS pressure | `…/data/PA.csv` |
| ASWS wind direction | `…/data/UD.csv` |
| ASWS wind speed | `…/data/US.csv` |
| ASWS wind gust | `…/data/UP.csv` |
| ASWS wind run | `…/data/UR.csv` |
| ASWS relative humidity | `…/data/XR.csv` |
| Per-station combined | `…/data/SnowAll/{ID}.csv` |
| MSS current season | `…/data/allmss_current.csv` |
| MSS archive | `…/data/allmss_archive.csv` |
| ASWS WFS locations | `https://openmaps.gov.bc.ca/geo/pub/WHSE_WATER_MANAGEMENT.SSL_SNOW_ASWS_STNS_SP/ows` |
| MSS WFS locations | `https://openmaps.gov.bc.ca/geo/pub/WHSE_WATER_MANAGEMENT.SSL_SNOW_MSS_LOCS_SP/ows` |

Base data URL: `https://www.env.gov.bc.ca/wsd/data_searches/snow/asws/data`

### Station URLs

```
# Station information (RFC AQRT portal)
https://aqrt.nrs.gov.bc.ca/Data/Location/Summary/Location/{ID}/Interval/Latest

# Station photos (BCMOE AQRT portal — used by get_station_image_url())
https://bcmoe-prod.aquaticinformatics.net/Data/Location/Summary/Location/{ID}/Interval/Latest
```

---

## 5. NVEClient

**Module:** `clients.nve.nve_client`
**Class:** `NVEClient`
**API:** [NVE HydAPI v1](https://hydapi.nve.no/)

The NVE HydAPI provides access to hydrological observations from the Norwegian
Water Resources and Energy Directorate (Norges vassdrags- og energidirektorat).

**An API key is required.**  Register for a free key at
<https://hydapi.nve.no/> and set the `NVE_API_KEY` environment variable (or
pass `api_key=` to the constructor).  Without a key the client logs a
warning and every request fails with HTTP 401.

```python
from clients.nve import NVEClient
client = NVEClient()   # reads NVE_API_KEY from the environment
```

### Key parameters

| Parameter ID | Description | Native Units | Returned Units |
|---|---|---|---|
| `2003` | Snow Water Equivalent (Snøens vannekvivalent) | **m** | cm (× 100) |
| `2002` | Snow Depth (Snødybde) | cm | cm |

> ⚠️ Parameter **2001** is *Markfuktighet* (soil water, %) — it is **not**
> a snow parameter despite its neighbouring ID.

### Constructor

```python
NVEClient(
    base_url: str = "https://hydapi.nve.no/api/v1",
    api_key: str | None = None,   # falls back to NVE_API_KEY env var
    timeout: int = 60,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_stations(parameter_ids, active_only, bbox)` → `list[dict]`

Returns stations filtered by NVE parameter ID(s).

```python
# All SWE stations
swe_stations = client.get_stations(parameter_ids=2003)

# SWE + snow depth stations, active only
stations = client.get_stations(parameter_ids=[2003, 2002], active_only=True)
```

#### `get_all_stations(active_only, bbox)` → `list[dict]`

Returns all NVE stations with snow parameters (SWE and/or snow depth).
Convenience wrapper around `get_stations(parameter_ids=[2003, 2002])`.

#### `get_metadata(station_id)` → `dict`

Returns full metadata for a single station including its available series.

```python
meta = client.get_metadata("2.11.0")
```

#### `get_series(parameter, station_id)` → `list[dict]`

Lists available time series from `/Series`.  `get_data()` depends on this
internally to discover which station+parameter pairs actually have data at
the requested resolution (requesting a non-existent series returns 404).

#### `get_observations(station_id, parameter_id, begin_date, end_date, resolution)` → `list[dict]`

Low-level endpoint — returns raw observation records from `/Observations`.

```python
obs = client.get_observations(
    station_id="2.11.0",
    parameter_id=2003,
    begin_date="2024-01-01",
    end_date="2024-04-30",
    resolution=1440,   # 1440 = daily; 60 = hourly
)
```

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Standardised flat-record output.  SWE is returned in cm (converted from
native metres × 100).  Hourly records carry a `datetime` key.  Unknown
variables and unsupported intervals raise `NVEError`.

```python
records = client.get_data(
    station_ids=["2.11.0", "12.228.0"],
    variables=["swe"],          # or "snwd", "swe_m", "snwd_cm"
    begin_date="2024-01-01",
    end_date="2024-03-31",
    interval="daily",           # or "hourly"
    include_flags=True,
)
# records[0] → {
#   "station_id": "2.11.0",
#   "date": "2024-01-01",
#   "variable": "swe_m",
#   "type": "swe",
#   "value": 45.2,       # cm
#   "units": "cm",
#   "interval": "daily",
#   "flag": "3",         # NVE quality code (only when include_flags=True)
# }
```

### Station dict schema

| Key | Type | Description |
|---|---|---|
| `station_id` | str | NVE station identifier (e.g. `"2.11.0"`) |
| `name` | str | Station name |
| `latitude` | float | WGS-84 latitude |
| `longitude` | float | WGS-84 longitude |
| `elevation_m` | float | Elevation in metres |
| `drainage_basin_key` | str | NVE drainage basin identifier |
| `status` | str | `"Active"` or `"Inactive"` |
| `station_url` | str | URL to NVE Sildre station page |
| `parameters` | list[int] | Available NVE parameter IDs at this station |
| `daily_parameters` | list[int] | Parameter IDs with a daily (1440-min) series |
| `coordinates_overridden` | bool | True when the client corrected a known-wrong HydAPI position (Nepal cooperation stations) |

### Quality flags

NVE quality codes returned when `include_flags=True`:

| Code | Meaning |
|---|---|
| `"0"` | Unknown — quality status not determined |
| `"1"` | Uncontrolled |
| `"2"` | Primary controlled |
| `"3"` | Secondary controlled (quality assured) |

Note the archive contains occasional glitches that pass NVE's own QC
(e.g. ~145 m SWE flagged "secondary controlled"); the client normalises
snow values outside 0–15 m to `None`.

### Endpoints used

| Purpose | Endpoint |
|---|---|
| Station list | `GET /Stations?ParameterId={id}` |
| Single station | `GET /Stations?StationId={id}` |
| Series catalogue | `GET /Series?Parameter={id}` / `GET /Series?StationId={id}` |
| Observations | `GET /Observations?StationId={id}&Parameter={id}&ReferenceTime=...` |

Base URL: `https://hydapi.nve.no/api/v1`.  Rate limit: 5 requests/second
per API key — the client spaces requests and honours `Retry-After` on 429.

---

## 6. YukonClient

**Module:** `clients.yukon.yukon_client`
**Class:** `YukonClient`
**API:** [AquaCache API v1](https://service.yukon.ca/water-data/api/v1/openapi.json)
**Operator:** Yukon Government Department of Environment, Water Science and
Stewardship

The public data service behind the Government of Yukon
[Water Data Explorer](https://service.yukon.ca/water-data/shiny/).  No
authentication is required — the API is fully open and self-describes at
`/openapi.json`.

```python
from clients.yukon import YukonClient
client = YukonClient()
```

### Networks and station types

| Station type | `network_code` | Count | Description |
|---|---|---|---|
| `SC` | `YSS` | 92 | Manual snow courses — 10-point surveys, records from 1964, target dates Feb 1 / Mar 1 / Apr 1 / May 1 / May 15.  **Periodic**, so `daily_or_better: false` in `all_snow_stations.geojson`. |
| `AWS` | `YSS` | 9 | Automated snow-weather stations — snow-pillow SWE at hourly to 3-hourly resolution, earliest record 1980-02-25. |
| `ECCC` | `YKEC` | 8 | Environment and Climate Change Canada climate stations mirrored into AquaCache, with daily snow depth.  Reaches the Arctic coast (Herschel Island 69.57°N; Komakuk Beach from 1963). |

17 stations (AWS + ECCC) carry a continuous series and appear in the merged
daily inventory.

### Key data variables

```python
from clients.yukon.yukon_client import VARIABLES, DATA_FLAGS, SNOW_VARIABLES
```

| Key | `type` | Native | Emitted |
|---|---|---|---|
| `swe_mm` | `swe` | mm | **cm** (÷ 10) |
| `snwd_cm` | `snwd` | cm | cm |
| `air_temp_degc` | `temp` | °C | °C |
| `air_temp_max_degc` | `temp_max` | °C | °C |
| `air_temp_min_degc` | `temp_min` | °C | °C |
| `precip_total_mm` | `precip` | mm | mm |
| `precip_rain_mm` | `precip` | mm | mm |
| `precip_snow_cm` | `precip` | cm | cm |
| `rel_humidity_pct` | `rh` | % | % |
| `wind_spd_kmh` | `wind_spd` | km/h | km/h |
| `wind_dir_deg` | `wind_dir` | degrees | degrees |
| `baro_press_kpa` | `baro` | kPa | kPa |
| `soil_moisture_pct` | `other` | % | % |

`VARIABLES[key]["units"]` is the **native** unit and `["output_units"]` is
what `get_data()` emits — the pattern `DESIGN.md` §3.2 adopts for every
client.  Only SWE needs conversion here (mm → cm); the other Yukon met
variables are already metric.

### Constructor

```python
YukonClient(
    base_url: str = "https://service.yukon.ca/water-data/api/v1",
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_locations(location_types, networks, bbox)` → `list[dict]`

All 390 AquaCache monitoring locations, optionally filtered.

```python
courses = client.get_locations(location_types="snowpack")   # 92
```

#### `get_timeseries(location_ids, variables, publicly_visible_only)` → `list[dict]`

The continuous-series catalogue.  Only series whose parameter appears in
`VARIABLES` are returned — the endpoint also lists water flow, water level,
groundwater and water-quality series that have no place in a snow archive.
This is the source of truth for per-station variable inventories, so period of
record and interval always reflect what the API currently serves.

#### `get_snow_course_stations(active_only, bbox)` → `list[dict]`

The 92 manual snow courses, built from `/locations` and enriched from
`/snow-survey/metadata` (first/last survey, per-target-date survey counts,
Snow Bulletin sub-basin).

#### `get_automated_stations(active_only, bbox, networks)` → `list[dict]`

Locations with a continuous SWE or snow-depth series, each carrying a
populated `series` list.

#### `get_all_stations(active_only, bbox)` → `list[dict]`

Courses + automated + ECCC, sorted by `station_id` (109 stations).

#### `get_metadata(station_id)` → `dict`

Full metadata for a single station, including its `series` list and a
`variables` summary.  Returns `{}` for unknown codes.

#### `get_snow_survey_data(station_ids, begin_date, end_date, include_flags)` → `list[dict]`

The manual snow course archive (~22,000 measurements).  The endpoint takes no
query parameters and returns the whole archive in one ~2 MB response, so
filtering happens client-side and the response is cached on the instance.

```python
rows = client.get_snow_survey_data(station_ids=["08AA-SC01"], include_flags=True)
apr1 = [r for r in rows if r["survey_period"] == "01-Apr"]
```

Records carry `date` (true sample date), `target_date`, `survey_period`
(`"01-Apr"`-style, matching the DataBC client), `year` and `month`.

#### `get_snow_survey_stats()` / `get_snow_survey_trends()` → `list[dict]`

Per-course summary statistics (record length, missing years, max/mean/median
SWE and depth) and Mann-Kendall trend analysis (p-values, Sen's slopes,
estimated annual percent change).  Useful as an independent cross-check on
course metadata.

#### `get_daily_measurements(timeseries_ids, begin_date, end_date, stats)` → `list[dict]`

Raw daily aggregates.  `stats=True` adds historical percentiles and 30-year
normals.

#### `get_measurements(timeseries_ids, begin_date, end_date)` → `list[dict]`

Raw instantaneous values with grade, approval and qualifier columns.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Standardized data fetch — returns a **flat** list of observation records.
`interval="daily"` reads daily aggregates, `"hourly"` / `"sub_daily"` reads
instantaneous values, and `"periodic"` reads the manual snow course archive
(as `DataBCClient` does for MSS).

```python
records = client.get_data(
    station_ids=["09AA-M1"],       # Tagish Meteorological
    variables=["swe", "snwd"],
    interval="daily",
    begin_date="2024-01-01",
)
# records[0]
# → {"station_id": "09AA-M1", "date": "2024-01-01",
#    "variable": "swe_mm", "type": "swe", "value": 8.088,
#    "units": "cm", "interval": "daily",
#    "aggregation": "instantaneous", "timeseries_id": "20"}
```

`aggregation` and `timeseries_id` are extra keys that disambiguate locations
holding several series of the same parameter — ECCC daily air temperature
exists as `minimum`, `maximum` and `(min+max)/2`.  Sub-daily records
additionally carry `datetime`.

#### `get_station_image_url(station_id)` → `None`

Always `None`.  The API exposes no station imagery and the Water Data Explorer
that would host it sits behind a Cloudflare JS challenge.  The method exists
so callers can treat every client alike.

### Station dict schema

| Key | Type | Description |
|---|---|---|
| `station_id` | str | AquaCache `location_code` (e.g. `"09AA-M1"`, `"08AA-SC01"`) |
| `location_id` | str | Numeric AquaCache ID, used by `/timeseries` |
| `name` | str | Station name |
| `latitude`, `longitude` | float | WGS-84 |
| `elevation_m` | float | Elevation in metres |
| `state` | str | `"YT"`, `"BC"` or `"AK"` — see below |
| `station_type` | str | `"SC"`, `"AWS"` or `"ECCC"` |
| `network`, `network_code` | str | AquaCache network name and short network code (`YSS` / `YKEC`) |
| `operator` | str | Operating agency |
| `status` | str | `"Active"` or `"Inactive"` |
| `variables` | list[str] | `VARIABLES` keys available at this station |
| `series` | list[dict] | Continuous series (empty for courses) |
| `station_url`, `dataset_url` | str | Water Data Explorer and Open Yukon links |

Courses additionally carry `first_survey`, `last_survey`, `survey_counts`,
`sub_basin` and `has_survey_metadata`.

### Data flags

Five vocabularies, exported separately and unioned into `DATA_FLAGS`:

| Export | Source | Applies to |
|---|---|---|
| `SNOW_SURVEY_FLAGS` | `/snow-survey/data` | `interval="periodic"` — `"Actual"` vs `"Estimated SWE"` (depth measured, SWE inferred from historical density) |
| `DAILY_FLAGS` | `/timeseries/measurementsDaily` | `interval="daily"` — `"imputed"` or `""` |
| `GRADE_FLAGS` | `GET /grades` | instantaneous, prefixed `grade:` |
| `APPROVAL_FLAGS` | `GET /approvals` | instantaneous, prefixed `approval:` |
| `QUALIFIER_FLAGS` | `GET /qualifiers` | instantaneous, prefixed `qualifier:` |

### Endpoints used

| Purpose | Endpoint |
|---|---|
| Locations | `GET /locations` |
| Series catalogue | `GET /timeseries` |
| Daily aggregates | `GET /timeseries/measurementsDaily` |
| Instantaneous values | `GET /timeseries/measurements` |
| Course metadata | `GET /snow-survey/metadata` |
| Course archive | `GET /snow-survey/data` |
| Course statistics | `GET /snow-survey/stats` |
| Course trends | `GET /snow-survey/trends` |
| Flag vocabularies | `GET /grades`, `/approvals`, `/qualifiers` |

Base URL: `https://service.yukon.ca/water-data/api/v1`

### Notes and gotchas

- **`state` is not blanket `"YT"`.**  The Yukon Snow Survey operates seven
  stations outside Yukon: `09AA-SC03` and `09AA-SC04` (named "Log Cabin
  (B.C.)" and "Atlin (B.C.)"), `09EC-SC02` ("Boundary (Alaska)"), plus
  `08AC-SC01`, `08AK-SC01`, `08AK-SC02` and `09AA-M2`, whose names do not
  declare a jurisdiction.  `_station_state()` resolves the jurisdiction
  declared in the name first, then the Yukon bounding box, then an explicit
  override table — so an upstream naming fix automatically supersedes the
  overrides.
- **`measurementsDaily.value` is a daily *mean***, computed over the local day
  given by `day_timezone` (UTC-07 for Yukon Snow Survey sites), not an
  instantaneous snapshot.  Verified by reconstructing 2024-03-01 for
  timeseries 20 from `/timeseries/measurements`.
- **Empty results arrive as a status envelope, not an empty CSV.**  A query
  matching nothing returns `status,message` with one informational row.
  `_parse_csv()` recognises and drops it so it never surfaces as data.
- **`/snow-survey/*` CSVs have a quoted comment header** whose block ends with
  a line containing exactly `""`.  Filtering on `line.strip()` alone treats
  that line as content and makes it the CSV header — see
  `_strip_api_comments()`.
- **`/snow-survey/data` leaves `units` empty for snow depth.**  It is cm,
  corroborated by the `/snow-survey/stats` field name `max_DEPTH_cm`.  The
  unit always comes from `VARIABLES`, never from the response.
- **Array columns are Postgres literals.**  `networks` and `projects` arrive as
  `{"Yukon Snow Survey Network","..."}` or `{NULL}` — see `_parse_pg_array()`.
- **Snow-survey `month` includes `5.5`** (meaning May 15), so it must be parsed
  as a float, not an int.
- **Low elevations are real, not sentinels.**  Herschel Island reports 1.2 m,
  Komakuk Beach 13.2 m and Fishing Branch River 0 m.
- **`09DC-SC01` (Mayo Airport) is a composite record** present in `/locations`
  but absent from `/snow-survey/metadata`, with no survey rows of its own — its
  constituents `09DC-SC01A`/`B` hold the data.  Courses are therefore built
  from `/locations` so composites are not silently dropped.
- **ECCC snow depth is sparse in places.**  Herschel Island reports roughly one
  value per month in the late 2000s, so short date windows can legitimately
  return nothing.
- **`/organizations` and `/csw-layer` are slow** and are never on the critical
  path.
- **Provenance overlap.**  The `YKEC` stations are a Yukon-Government mirror of
  ECCC data.  A future direct ECCC client would need lat/lon + name dedup
  against them, the same way BC and CCSS sites intentionally duplicate between
  `awdb` and their native clients.

---

## 7. Error Handling

| Client | Exception | Scenarios |
|---|---|---|
| AWDBClient | `AWDBError` | HTTP 4xx/5xx, network timeout, value limit |
| CDECClient | `CDECError` | HTTP 4xx/5xx, network timeout, HTML parse failure |
| DataBCClient | `DataBCError` | HTTP 4xx/5xx, network timeout, malformed CSV |
| NVEClient | `NVEError` | HTTP 4xx/5xx, network timeout, station not found |
| YukonClient | `YukonError` | HTTP 4xx/5xx, network timeout, malformed CSV |

All exceptions are subclasses of `Exception` with descriptive messages.

```python
from clients.awdb import AWDBClient, AWDBError
from clients.cdec import CDECClient, CDECError
from clients.databc import DataBCClient, DataBCError
from clients.nve import NVEClient, NVEError
from clients.yukon import YukonClient, YukonError

try:
    data = AWDBClient().get_data(station_ids=["303:CO:SNTL"], variables=["swe"])
except AWDBError as e:
    print(f"AWDB error: {e}")
```

HTTP 400/404 errors are not retried.  HTTP 5xx and network errors are retried
with linear backoff up to `max_retries` attempts.

---

## 8. Adding a New Client

The full normative contract and checklist live in
[DESIGN.md](../DESIGN.md) §3 and the
[new-client issue template](../.github/ISSUE_TEMPLATE/new_client.md).
In brief, to add support for a new data source:

1. Create `clients/{source}/` directory with `__init__.py` and
   `{source}_client.py`.
2. Implement a `{Source}Client` class with at minimum:
   - `get_all_stations(active_only=False, bbox=None)` → `list[dict]`
   - `get_data(..., include_flags: bool = False)` → `list[dict]`
   - `get_metadata(station_id)` → `dict`
3. Export `VARIABLES` (or `SENSORS`) and `DATA_FLAGS` module-level dicts.
4. Raise `{Source}Error(Exception)` for all errors — including unknown
   variables and unsupported intervals (no silent fallbacks).
5. Return metric units for every variable (cm for SWE and snow depth —
   see the DESIGN.md §3.5 units table).
6. Export the class from `clients/{source}/__init__.py`.
7. Add to `clients/__init__.py`.
8. Add to `scripts/create_all_stations_geojson.py`:
   - Add `run_{source}_workflow()` that returns `(all_features, daily_features)`,
     where `daily_features` is filtered on the advertised `has_daily_swe` /
     `has_daily_snwd` flags (the combined inventory keeps *every* feature;
     the daily list is for logging and counts).
   - Add a `{source}_station_to_feature()` that builds a properties dict and
     passes it through `make_feature()`, which enforces the universal schema
     (DESIGN.md §6.1) — every universal field present, `null` when the source
     has nothing. Fields the builder must actually populate:
     - `code`, `name`, `latitude`, `longitude`, `elevation_m`, `state`
     - `network_code` — the source's network label, verbatim; must match a
       `NET_LABELS` key in the live map. A client may emit more than one
       (the Yukon client uses `"YSS"` and `"YKEC"` to separate
       Yukon-operated sites from mirrored ECCC ones)
     - `operator` — only when certain; otherwise leave it `None`
       (DESIGN.md §5)
     - `client`, `data_provider` — the access path (e.g. `"databc"`) and the
       organization/portal behind it (e.g. `"BC Data Catalogue"`)
     - `station_url`, `station_image_url`, `station_camera_url` — station
       page / photo / live camera links where the source offers them.
       Fetch these even if it requires an extra HTTP request — the live map
       is the primary user interface and station images significantly
       improve UX.
     - `data_variables` — list of dicts describing every series the station
       reports: `{name, type, interval, units, description, notes,
       begin_date, end_date, n_obs}` (the last three `None` where the
       source doesn't say). `type` uses the standardized vocabulary;
       `interval` uses the shared enum from `clients/_common.py`.
     - the daily candidate flags — set via `_daily_candidate_props(data_vars)`,
       which derives `has_daily_swe` / `has_daily_snwd` (advertised) and
       initializes `daily_or_better` / `daily_verified` / `daily_provenance`
       for the probe in `get_all_stations_data.py` to verify (DESIGN.md §4).
9. Add to `scripts/get_all_stations_data.py` with a `refresh_{source}()`.
10. Add the network to `scripts/generate_live_map.py`:
    - Add an entry in `NET_LABELS` (`"CODE": "Human Name"`)
    - Add an SVG shape entry in `NET_SHAPES`
    - Add a `case "CODE"` in `buildIcon()` to assign a Leaflet marker shape
    - The legend is built dynamically from `MAP_META.available_networks`,
      so no further changes to the legend HTML are needed.
11. Document in this README.
12. Update `README.md` (root): add to the Networks section and update the
    comparison table.

**Key invariants:**

- All `get_data()` methods return a **flat** list of observation records with
  keys `station_id`, `date`, `variable`, `type`, `value`, `units`, `interval`
  (plus `datetime` on sub-daily records and optional `flag`), so pipeline
  scripts can route data uniformly.
- `station_image_url` must be a direct URL that can be used in an `<img src>`
  tag without authentication.
- `network_code` in the GeoJSON must match a `NET_LABELS` key in the live map.
