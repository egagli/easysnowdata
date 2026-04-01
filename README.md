# Clients

This folder contains API client modules for retrieving snow observation data
from external data sources.  Each client is responsible for one data source
and exposes a consistent interface for fetching stations, metadata, and
time-series data.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [AWDBClient — USDA NRCS AWDB REST API](#2-awdbclient)
3. [CDECClient — California Data Exchange Center](#3-cdecclient)
4. [DataBCClient — BC Data Catalogue](#4-databcclient)
5. [Error Handling](#5-error-handling)
6. [Adding a New Client](#6-adding-a-new-client)

---

## 1. Design Philosophy

- **One client per data source.**  Each client encapsulates HTTP requests,
  batching, retry logic, HTML/JSON parsing, and response normalisation.
- **Return plain Python objects.**  Methods return `dict` / `list[dict]` or
  pandas DataFrames so callers decide how to convert.
- **Metric-first normalisation.**  All SWE and snow depth values are returned
  in centimetres.  Imperial inputs are converted in-client.
- **Handle limits internally.**  API rate limits, value-count limits, and URL
  length limits are managed by the client.
- **Fail clearly.**  All errors raise `{Client}Error(Exception)` with
  descriptive messages.
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

### Key data variables

| Element | Description | Units |
|---|---|---|
| `WTEQ` | Snow Water Equivalent | cm (converted from inches) |
| `SNWD` | Snow Depth | cm (converted from inches) |
| `TOBS` | Air Temperature (observed) | °C |
| `PREC` | Precipitation accumulation | cm |
| `PRCP` | Precipitation increment | cm |

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

#### `get_data(triplets, elements, duration, begin_date, end_date, ...)` → `list[dict]`

Primary data retrieval.  Automatically batches to respect the 500k-value limit.

```python
data = client.get_data(
    triplets=["303:CO:SNTL"],
    elements=["WTEQ", "SNWD"],
    duration="DAILY",
    begin_date="2023-10-01",
    end_date="2024-09-30",
)
# data[0]["data"][0]["values"][0]
# → {"date": "2023-10-01", "value": 5.08}
```

#### `get_normals(triplets, elements, duration, normal_period, ...)` → `list[dict]`

Fetch climatological normals (1991–2020, 1981–2010, or 1971–2000).

```python
norms = client.get_normals("303:CO:SNTL", ["WTEQ"], normal_period="1991-2020")
```

### Notes and gotchas

- **500k value limit:** `n_stations × n_elements × n_days ≤ 500,000`.  The
  client auto-splits requests.
- **Triplet format:** `{stationId}:{stateCode}:{networkCode}` — case-sensitive.
- **BC/AB/YK stations:** Accessible with state codes `BC`, `AB`, `YK`.
- **Missing values:** `None` in parsed response.
- **WTEQ/SNWD units:** Converted from inches to cm automatically.

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

**SWE vs. SNO ADJ:** Sensor 82 (SNO ADJ) is the revised version of sensor 3
(raw SWE), with a calibration offset applied after manual QC.  Both represent
SWE from the same snow pillow.  Sensor 82 is always preferred and is stored
as `wteq_cm` in per-station CSVs.  If sensor 82 is unavailable, sensor 3 is
used as fallback.

### Data flags

| Flag | Meaning |
|---|---|
| ` ` (space) | Unreviewed / provisional |
| `r` | Revised (most sensor 82 values carry this flag) |
| `o` | Calibration offset applied |
| `e` | Estimated |
| `N` | Error in data |
| `v` | Out of valid range |
| `t` | Trace of precipitation |

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

# Filter to those with daily data (use GeoJSON dailySWE/dailySnowDepth in
# create_all_stations_geojson.py; has_daily_swe/has_daily_snwd available on
# station dicts from get_stations() for direct client use)
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

#### `get_data(station_ids, sensors, duration, begin_date, end_date, include_flags)` → `list[dict]`

Fetches time-series data from the CDEC JSONDataServlet.  Values are converted
from inches to centimetres.  Missing values (-9999) are normalised to `None`.

```python
data = client.get_data(
    station_ids=["QUA", "BLC"],
    sensors=[82, 18],
    duration="D",
    begin_date="2023-10-01",
    end_date="2024-09-30",
    include_flags=True,
)
# data[0]["data"][0]["stationElement"]
# → {"sensorNum": 82, "sensorType": "SNO ADJ", "units": "cm", ...}
# data[0]["data"][0]["values"][0]
# → {"date": "2023-10-01", "value": 5.08, "flag": "r"}
```

### Data availability notes

- The JSON data service accepts multiple comma-separated station IDs.
- **Monthly duration** (`M`) returns empty results for sensors 3, 18, 82.
  Use daily (`D`) for all snow sensor data.
- **Hourly data** is available for sensors 3 and 18 at most automated stations.
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
Pass `daily_only=False` to any ASWS method to retrieve all hourly readings
instead, which returns a `datetime` column rather than `date`.

| Variable | Units | Method | Archive? | Notes |
|---|---|---|---|---|
| `swe_mm` | mm | `get_asws_daily_data()` | Yes | Daily pre-aggregated (SWDaily.csv) |
| `swe_mm` | mm | `get_asws_sw_hourly_data()` | Yes | Hourly raw pillow (SW.csv) |
| `snwd_cm` | cm | `get_asws_sd_data()` | Yes | Snow depth sensor (SD.csv) |
| `air_temp_degc` | °C | `get_asws_ta_data()` | Yes | Air temperature (TA.csv) |
| `precip_cumul_mm` | mm | `get_asws_pc_data()` | Yes | Cumulative precipitation (PC.csv) |
| `baro_press_hpa` | hPa | `get_asws_pa_data()` | **No** | Barometric pressure (PA.csv) |
| `wind_dir_deg` | ° | `get_asws_ud_data()` | **No** | Wind direction (UD.csv) |
| `wind_spd_kmh` | km/h | `get_asws_us_data()` | **No** | Wind speed (US.csv) |
| `wind_spd_peak_kmh` | km/h | `get_asws_up_data()` | **No** | Wind gust speed (UP.csv) |
| `wind_run_km` | km | `get_asws_ur_data()` | **No** | Cumulative wind run (UR.csv) |
| `rh_pct` | % | `get_asws_xr_data()` | **No** | Relative humidity (XR.csv) |
| `swe_mm` + `snwd_cm` + `air_temp_degc` + `precip_cumul_mm` | mixed | `get_asws_combined_data(id)` | — | Per-station combined file (SnowAll/) |
| `swe_mm` | mm | `get_mss_survey_data()` | Yes | MSS periodic survey (Water Equiv.) |
| `snwd_cm` | cm | `get_mss_survey_data()` | Yes | MSS periodic survey (Snow Depth) |
| `density_pct` | % | `get_mss_survey_data()` | Yes | MSS periodic only |
| `snow_line_m` | m | `get_mss_survey_data()` | Yes | MSS periodic only |

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

#### `get_all_stations(active_only)` → `list[dict]`

Returns combined ASWS + MSS station list.

#### `get_asws_daily_data(location_ids, begin_date, end_date, archive)` → `pd.DataFrame`

Fetches daily ASWS SWE data from the pre-aggregated `SWDaily.csv` files.
Returns columns: `date`, `location_id`, `swe_mm`.

```python
df = client.get_asws_daily_data(
    location_ids=["1A01P", "1E08P"],
    begin_date="2022-10-01",
    archive=True,
)
```

Archive file (`SW_DailyArchive.csv`) is ~5 MB.

#### `get_asws_sw_hourly_data(location_ids, begin_date, end_date, archive)` → `pd.DataFrame`

Fetches raw hourly SWE from `SW.csv` / `SW_Archive.csv`.
Returns columns: `datetime` (UTC, `"YYYY-MM-DD HH:MM"`), `location_id`, `swe_mm`.

#### `get_asws_sd_data(location_ids, begin_date, end_date, archive, daily_only)` → `pd.DataFrame`

Fetches snow depth from `SD.csv` / `SD_Archive.csv`.
Returns columns: `date`/`datetime`, `location_id`, `snwd_cm`.
Archive file is ~37 MB.

#### `get_asws_ta_data(location_ids, begin_date, end_date, archive, daily_only)` → `pd.DataFrame`

Fetches air temperature from `TA.csv` / `TA_Archive.csv`.
Returns columns: `date`/`datetime`, `location_id`, `air_temp_degc`.
Archive file is ~75 MB.

#### `get_asws_pc_data(location_ids, begin_date, end_date, archive, daily_only)` → `pd.DataFrame`

Fetches cumulative precipitation from `PC.csv` / `PC_Archive.csv`.
Returns columns: `date`/`datetime`, `location_id`, `precip_cumul_mm`.
Archive file is ~63 MB.

#### `get_asws_pa_data(location_ids, begin_date, end_date, daily_only)` → `pd.DataFrame`

Fetches barometric pressure from `PA.csv` (current season only).
Returns columns: `date`/`datetime`, `location_id`, `baro_press_hpa`.

#### `get_asws_ud_data(...)` / `get_asws_us_data(...)` / `get_asws_up_data(...)` / `get_asws_ur_data(...)` / `get_asws_xr_data(...)` → `pd.DataFrame`

Fetch wind direction, wind speed, wind gust, wind run, and relative humidity
respectively from the corresponding current-season-only CSV files.
All return columns: `date`/`datetime`, `location_id`, `{variable}`.

```python
# Example: fetch all met variables for a station, hourly
df_ta = client.get_asws_ta_data(["1E08P"], daily_only=False)
df_rh = client.get_asws_xr_data(["1E08P"], daily_only=False)
df_ws = client.get_asws_us_data(["1E08P"], daily_only=False)
```

#### `get_asws_combined_data(location_id)` → `pd.DataFrame`

Fetches the per-station combined archive from `SnowAll/{id}.csv`.
Contains SWE, snow depth, air temperature, and precipitation in one file.
Uses 16:00 UTC as canonical daily value.
Returns columns: `date`, `swe_mm`, `snwd_cm`, `air_temp_degc`, `precip_cumul_mm`.

#### `get_mss_survey_data(location_ids, begin_date, end_date, archive, include_flags)` → `pd.DataFrame`

Fetches periodic manual snow survey data from `allmss_current.csv` /
`allmss_archive.csv`.

Returns: `date`, `location_id`, `name`, `swe_mm`, `snwd_cm`,
`density_pct`, `snow_line_m`, `survey_period`, optionally `survey_code`.

```python
df = client.get_mss_survey_data(archive=True, include_flags=True)
apr1 = df[df["survey_period"] == "01-Apr"]
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

## 5. Error Handling

| Client | Exception | Scenarios |
|---|---|---|
| AWDBClient | `AWDBError` | HTTP 4xx/5xx, network timeout, value limit |
| CDECClient | `CDECError` | HTTP 4xx/5xx, network timeout, HTML parse failure |
| DataBCClient | `DataBCError` | HTTP 4xx/5xx, network timeout, malformed CSV |

All exceptions are subclasses of `Exception` with descriptive messages.

```python
from clients.awdb import AWDBClient, AWDBError
from clients.cdec import CDECClient, CDECError
from clients.databc import DataBCClient, DataBCError

try:
    data = AWDBClient().get_data(["303:CO:SNTL"], ["WTEQ"])
except AWDBError as e:
    print(f"AWDB error: {e}")
```

HTTP 400/404 errors are not retried.  HTTP 5xx and network errors are retried
with linear backoff up to `max_retries` attempts.

---

## 6. Adding a New Client

To add support for a new data source (e.g. GHCND, Environment Canada):

1. Create `clients/{source}/` directory with `__init__.py` and
   `{source}_client.py`.
2. Implement a `{Source}Client` class with at minimum:
   - `get_stations(...)` → `list[dict]`
   - `get_data(..., include_flags: bool = False)` → `list[dict]`
3. Export `VARIABLES` (or `SENSORS`) and `DATA_FLAGS` module-level dicts.
4. Raise `{Source}Error(Exception)` for all errors.
5. Return metric units (cm for SWE and snow depth).
6. Export the class from `clients/{source}/__init__.py`.
7. Add to `clients/__init__.py`.
8. Add to `scripts/create_all_stations_geojson.py`:
   - Add `run_{source}_workflow()` that returns `(all_features, daily_features)`.
   - In `databc_station_to_feature()` (or its equivalent), set these required
     GeoJSON properties on every feature:
     - `code` — native station identifier
     - `name` — station name
     - `latitude`, `longitude` — WGS-84
     - `elevation_m` — elevation in metres
     - `networkCode` — short code used by the live map (e.g. `"BCSS"`)
     - `Operator` — operating agency
     - `client` — source client name (e.g. `"databc"`)
     - `station_url` — link to official station page
     - `station_image_url` — station photo URL (where available; used by
       the live map popup).  Fetch this from the source portal even if it
       requires an extra HTTP request — the live map is the primary user
       interface and station images significantly improve UX.
     - `data_variables` — list of dicts describing every variable the
       station reports: `{name, type, interval, units, description, notes}`.
       `type` must use the standardized vocabulary (`"swe"`, `"snwd"`, etc.).
     - `dailySWE` — boolean derived from `data_variables`; `True` if any
       entry has `type="swe"` and `interval` in `{"daily","sub_daily","hourly"}`.
     - `dailySnowDepth` — same for `type="snwd"`.
     - `variables_daily` — derived from `data_variables`; comma-separated
       names of variables with a daily-class interval (used by map popup).
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

- All `get_data()` methods return a list of station dicts, each with a `data`
  list of element blocks, so pipeline scripts can route data to CSVs uniformly.
- `station_image_url` must be a direct URL that can be used in an `<img src>`
  tag without authentication.
- `networkCode` in the GeoJSON must match the `NET_LABELS` key in the live map.
