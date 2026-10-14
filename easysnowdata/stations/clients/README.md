# Clients

This folder contains API client modules for retrieving snow observation data
from external data sources.  Each client is responsible for one data source
and exposes a consistent interface for fetching stations, metadata, and
time-series data.

> The normative client contract (method signatures, record schema, units,
> intervals, error behaviour) lives in [DESIGN.md](https://github.com/egagli/global_snow_networks/blob/main/DESIGN.md) §3.  This
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
7. [GHCNDClient — NOAA NCEI GHCN-Daily](#7-ghcndclient)
8. [Other SWE and snow-depth clients: overview](#8-other-swe-and-snow-depth-clients-overview)
9. [CanSWEClient — ECCC CanSWE (Zenodo)](#9-cansweclient)
10. [NorSWEClient — ECCC NorSWE (Zenodo)](#10-norsweclient)
11. [HydroQuebecClient — Hydro-Québec open data](#11-hydroquebecclient)
12. [AlbertaClient — Alberta River Basins](#12-albertaclient)
13. [DWDClient — DWD Climate Data Center (Germany)](#13-dwdclient)
14. [SYKEClient — Syke hydrology API (Finland)](#14-sykeclient)
15. [SLFClient — WSL-SLF (Switzerland)](#15-slfclient)
16. [RIHMIClient — RIHMI-WDC / NSIDC G01170 (Russia, FSU)](#16-rihmiclient)
17. [Error Handling](#17-error-handling)
18. [Adding a New Client](#18-adding-a-new-client)

---

## 1. Design Philosophy

See [DESIGN.md](https://github.com/egagli/global_snow_networks/blob/main/DESIGN.md) §3 for the full contract.  Summary:

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

**Module:** `easysnowdata.stations.clients.awdb.awdb_client`
**Class:** `AWDBClient`
**API:** [AWDB REST API v1](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html)

The AWDB (Air and Water Database) REST API provides access to data from
SNOTEL, SNOLite, SCAN, COOP, Manual SNOTEL, snow courses, streamflow gauges,
reservoirs, and more.

```python
from easysnowdata.stations.clients.awdb import AWDBClient
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

**Module:** `easysnowdata.stations.clients.cdec.cdec_client`
**Class:** `CDECClient`
**Source:** [CDEC — California Data Exchange Center](https://cdec.water.ca.gov)
**Operator:** California Department of Water Resources (CA DWR)

Provides access to California Cooperative Snow Surveys (CCSS) data — both
automated snow pillow stations (daily) and manual snow course sites
(periodic).

```python
from easysnowdata.stations.clients.cdec import CDECClient
client = CDECClient()
```

### Snow sensors

```python
from easysnowdata.stations.clients.cdec.cdec_client import SENSORS, DATA_FLAGS, DURATION_CODES
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

**Module:** `easysnowdata.stations.clients.databc.databc_client`
**Class:** `DataBCClient`
**Source:** [BC Data Catalogue](https://catalogue.data.gov.bc.ca) + [BC env.gov.bc.ca CSV files](https://www.env.gov.bc.ca/wsd/data_searches/snow/asws/data/)
**Operator:** BC Ministry of Environment (BC ENV)

Provides access to BC snow survey data — both Automated Snow Weather Stations
(ASWS, full meteorological suite) and Manual Snow Survey sites (MSS, periodic
surveys).  Also fetches ASWS station photos from the AQRT BCMOE portal.

```python
from easysnowdata.stations.clients.databc import DataBCClient
client = DataBCClient()
```

### Station types

| Type | ID suffix | Description | Data |
|---|---|---|---|
| ASWS | ends in `P` (e.g. `1A01P`) | Automated snow pillow + weather station | Daily SWE, snow depth, air temp, precip, wind, humidity, pressure |
| MSS | no `P` (e.g. `1A06A`, `1A10`) | Manual snow course | Periodic SWE (mm), depth (cm), density (%) |

### Variables

```python
from easysnowdata.stations.clients.databc.databc_client import VARIABLES, DATA_FLAGS
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

**Module:** `easysnowdata.stations.clients.nve.nve_client`
**Class:** `NVEClient`
**API:** [NVE HydAPI v1](https://hydapi.nve.no/)

The NVE HydAPI provides access to hydrological observations from the Norwegian
Water Resources and Energy Directorate (Norges vassdrags- og energidirektorat).

**An API key is required.**  Register for a free key at
<https://hydapi.nve.no/> and set the `NVE_API_KEY` environment variable (or
pass `api_key=` to the constructor).  Without a key the client logs a
warning and every request fails with HTTP 401.

```python
from easysnowdata.stations.clients.nve import NVEClient
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

**Module:** `easysnowdata.stations.clients.yukon.yukon_client`
**Class:** `YukonClient`
**API:** [AquaCache API v1](https://service.yukon.ca/water-data/api/v1/openapi.json)
**Operator:** Yukon Government Department of Environment, Water Science and
Stewardship

The public data service behind the Government of Yukon
[Water Data Explorer](https://service.yukon.ca/water-data/shiny/).  No
authentication is required — the API is fully open and self-describes at
`/openapi.json`.

```python
from easysnowdata.stations.clients.yukon import YukonClient
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
from easysnowdata.stations.clients.yukon.yukon_client import VARIABLES, DATA_FLAGS, SNOW_VARIABLES
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

## 7. GHCNDClient

**Module:** `easysnowdata.stations.clients.ghcnd`
**Source:** NOAA NCEI Global Historical Climatology Network – Daily,
<https://www.ncei.noaa.gov/pub/data/ghcn/daily/> (format:
[readme.txt](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt))
**Auth:** none
**Status:** prototype. Not yet registered in `stations.networks.NETWORKS`,
the catalog, or the `global_snow_networks` pipeline.

GHCN-D is global daily station data aggregated from national meteorological
services, U.S. COOP, CoCoRaHS, ECA&D, SNOTEL and others. For snow it adds
daily snow depth (`SNWD`) worldwide and SWE (`WESD`) at a smaller set of
stations. A live sweep in October 2026 found ~69 500 stations with `SNWD`
or `WESD`, ~27 600 with `WESD` and ~20 900 active. Active stations were
mostly in the US, Canada, Russia, Germany, Sweden, the Netherlands, Norway,
Switzerland, Kazakhstan and Japan.

### Files and access pattern

| File | Use |
|---|---|
| `ghcnd-stations.txt` (~11 MB, fixed width) | station id, lat, lon, elevation, state, name, GSN/HCN flags, WMO id |
| `ghcnd-inventory.txt` (~36 MB, fixed width) | (station, element, first year, last year): selects snow stations |
| `by_station/{ID}.csv.gz` | the full record of one station, all elements |

Each client instance downloads the two inventory files once and keeps them
in memory. `get_data()` downloads the **whole** station file and filters
rows by date, because GHCN-D has no windowed per-station endpoint. A `bbox`
query in a dense region (CoCoRaHS!) means many downloads.

### Variables

| Element | Type | Stored | Emitted |
|---|---|---|---|
| `WESD` | `swe` | tenths of mm | cm |
| `SNWD` | `snwd` | mm | cm |
| `SNOW` | `snowfall` | mm | cm |
| `WESF` | `other` | tenths of mm | mm |
| `PRCP` | `precip` | tenths of mm | mm |
| `TAVG` / `TMAX` / `TMIN` | `temp` / `temp_max` / `temp_min` | tenths of °C | °C |

`variables=None` returns `WESD` and `SNWD`. Only `interval="daily"` exists.

### Flags and QC

`include_flags=True` adds `flag = "MFLAG,QFLAG,SFLAG"`, the same layout as
the NCEI Access Data Service `*_ATTRIBUTES` fields. A value with a non-blank
`QFLAG` failed a GHCN-D QC check (`DATA_FLAGS`). Such values come back as
`None` unless the client is built with `keep_qc_failed=True`. Snow values
outside 0–15 m (snowfall: 0–5 m) are nulled.

### Station ids and networks

Ids are 11 characters: FIPS country (2) + network code (1) + number (8).
The network code is exposed as `ghcn_network` / `ghcn_network_name`
(`GHCN_NETWORKS`). `S` is the SNOTEL mirror, which duplicates AWDB. Drop it
with `get_stations(exclude_networks="S")`. COOP (`C`) stations duplicate the
AWDB `COOP` subset in a few places too.

### Key methods

```python
from easysnowdata.stations.clients import GHCNDClient

c = GHCNDClient()
stations = c.get_all_stations(bbox=(30, 50, 60, 70), active_only=True)
non_snotel = c.get_stations(exclude_networks="S")
meta = c.get_metadata("RSM00027612")   # Moscow: element first/last years
recs = c.get_data("USS0021D08S", begin_date="2024-01-01", end_date="2024-01-31")

# Bulk path: stream one by_year file (~115–170 MB) instead of N station files
for rec in c.get_year_data(2026, station_ids=["RSM00027612", "USC00457780"]):
    ...
```

### Notes and gotchas

- `status` is year-resolution: "Active" means a snow element reported this
  year or last (the inventory has years only).
- The GHCN-D record of some stations ends early even though the station
  still operates (Moscow `RSM00027612` ends in 2022). Coverage depends on
  what each national service sends NCEI.
- Many CoCoRaHS `WESD` series are core samples taken on some days only.
  They are not daily-or-better. The pipeline's regular-cadence probe
  (DESIGN.md §4) has to decide, not `has_swe`.
- `station_url` points at the NCEI Climate Data Online station page.

---

## 8. Other SWE and snow-depth clients: overview

The eight clients in §9–§16 follow the same contract (DESIGN.md §3) but
are **not wired into** `stations.networks`, the catalog,
`inventory()`/`load()` or the `global_snow_networks` archive. Use them
directly. All are exported from `easysnowdata.stations.clients`, need no
credentials, and their module docstrings document endpoints, units,
conventions and caveats in full. Counts were checked live on 2026-10-14.

"Daily SWE" means at least 30 days with SWE > 0 in the densest 90-day
window of a season.

| Client | Source | Stations | SWE method and cadence | Daily SWE | Period / updates | Licence |
|---|---|---|---|---|---|---|
| [`CanSWEClient`](#9-cansweclient) | CanSWE (Zenodo, Vionnet et al. 2021) | 2,963 (1,213 active) | manual surveys (2,678), pillows (162), GMON (121) | 253 in 2025 | 1928–2025; yearly | OGL-Canada; Hydro-Québec part CC BY-NC-SA |
| [`NorSWEClient`](#10-norsweclient) | NorSWE v3 (Zenodo) | 10,153 | surveys, pillows, GMON, 2,357 airborne-gamma lines | 1,049 in 2021 (830 SNOTEL) | 1979–2021; frozen | per source (`Distribution` attribute) |
| [`HydroQuebecClient`](#11-hydroquebecclient) | Hydro-Québec open data (Opendatasoft) | 95 SWE, 137 depth | GMON gamma, SWE updated every 6 h | live, rolling 10 days | **only the last ~10 days**; twice daily | CC BY-NC 4.0 |
| [`AlbertaClient`](#12-albertaclient) | rivers.alberta.ca (site's own endpoints) | 291 snow; 55 automated SWE | pillows (19 AEPA + BC/SNOTEL mirrors), monthly surveys | 52 in WY2026 | pillows from 1984; near real time | none stated (disclaimer only) |
| [`DWDClient`](#13-dwdclient) | DWD Climate Data Center | 1,177 SWE (527 active); 6,628 depth | manual cores when depth is about 5 cm or more | 77 in 2025–26 (7 with ≥ 60) | to ~1927 / 1854; recent files daily | CC BY 4.0 |
| [`SYKEClient`](#14-sykeclient) | SYKE Hydrologiarajapinta OData | 392 snow courses (150 active) | manual, monthly Nov–May | none | 1947–present | CC BY 4.0 |
| [`SLFClient`](#15-slfclient) | SLF on EnviDat / measurement-data.slf.ch | 11 GCOS SWE sites; 135 study plots; 179 IMIS | manual, every ~15 days | none | GCOS from 1936; yearly | CC BY 4.0 / ODbL |
| [`RIHMIClient`](#16-rihmiclient) | NSIDC G01170 mirror + AISORI-M exports | 1,552 (517 route surveys) | manual route surveys every 10 days | none | public 1966–1996; current data needs an AISORI-M login | NSIDC: open with citation; RIHMI: none stated |

Overlaps to know about: NorSWE contains CanSWE, SNOTEL, NRCS courses,
CDEC, NVE, SYKE, SLF GCOS and RIHMI route surveys; CanSWE contains the
Hydro-Québec GMON history; Alberta mirrors BC and SNOTEL pillows.

---

## 9. CanSWEClient

**Module:** `easysnowdata.stations.clients.canswe.canswe_client` (reader shared with NorSWE in `canswe._format`)
**Source:** CanSWE, Zenodo concept record 4734371
(<https://doi.org/10.5281/zenodo.4734371>); v8 = record 19075529,
`CanSWE-CanEEN_1928-2025_v8.nc`, data to 2025-07-31.
Format reference: `ReadMe_CanSWE_v8.pdf` in the same record.
**Auth:** none
**Licence:** Open Government Licence – Canada, except Hydro-Québec data
(ids `HQ-…`), which are CC BY-NC-SA 4.0.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`. Storage layer being redesigned (see below).

CanSWE is the Canadian historical SWE dataset (Vionnet et al., 2021, ESSD
13, 4603–4619), compiled by ECCC from national, provincial and territorial
agencies and hydropower companies: manual snow surveys from 1928 plus
automated pillow and passive-gamma (GMON) SWE, quality-controlled and
updated once a year.

**Storage.** The latest version is resolved through the Zenodo API
(`/api/records/4734371/versions/latest`); `record_id=` pins one. First use
downloads the 14 MB netCDF once (resumable, md5-checked), converts it to a
~10 MB Zarr store under `easysnowdata.config.cache_dir('stations', 'canswe')`
in about 15 s, and deletes the netCDF. Later calls read the store lazily;
a new Zenodo version replaces it automatically. `path=` takes a local
netCDF (converted once, left in place) or a store.

```python
from easysnowdata.stations.clients import CanSWEClient
from easysnowdata.stations.clients.canswe.canswe_client import (
    VARIABLES, DATA_FLAGS, AGENCY_FLAGS, QC_FLAGS, METHODS, ID_PREFIXES, SOURCES,
)
client = CanSWEClient()
```

### Variables

| Key | `type` | Native | Emitted |
|---|---|---|---|
| `snw` | `swe` | kg m-2 (= mm) | cm (× 0.1) |
| `snd` | `snwd` | m | cm (× 100) |
| `den` | `density` | kg m-3 | % (÷ 10); derived as `snw / snd` by the dataset authors |

`variables=None` returns `snw` and `snd`. Snow values outside 0–15 m are
nulled.

### Cadence and flags

Each station's interval comes from its WMO `type_mes` code (`METHODS`):
manual surveys (0, 1) are `"periodic"` (typically monthly Jan–Jun); snow
pillows (2) and GMON (3) are `"daily"`. v8 holds only codes 0, 2 and 3.

Two flag vocabularies, unioned into `DATA_FLAGS` as `agency:<code>` and
`qc:<code>`: `AGENCY_FLAGS` (`data_flag_snw/_snd`, e.g. `E` estimate, `T`
trace, `R` revised) and `QC_FLAGS` (`qc_flag_snw/_snd`, e.g. `W` SWE >
3000 kg m-2, `D` density outside 25–700 kg m-3). QC-failed values are
already NaN in the file. `include_flags=True` adds
`flag = "<agency flag>,<qc flag>"` (either part may be empty, e.g. `"R,"`
or `",W"`) and also returns the QC-removed days with `value=None`.

### Constructor

```python
CanSWEClient(
    path: str | Path | None = None,       # use a local file instead of Zenodo
    record_id: str | None = None,         # pin a Zenodo version, e.g. "19075529"
    cache_dir: str | Path | None = None,  # default cache_dir("stations", "canswe")
    session: requests.Session | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
)
```

### Key methods

#### `get_all_stations(active_only, bbox, sources, methods, daily_only)` → `list[dict]`

`sources` matches the provider name case-insensitively, exactly or as a
substring (`"Hydro"` matches Hydro-Quebec and Manitoba Hydro). `methods`
takes `type_mes` codes, short names (`"snow_pillow"`, `"passive_gamma"`,
`"snow_survey"`) or `"automated"` / `"manual"`. `daily_only` keeps
stations with daily SWE in at least one season.

```python
pillows = client.get_all_stations(methods="automated", active_only=True)
bc = client.get_all_stations(sources="British Columbia")
```

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables` (one entry per native variable with
data, with period of record and `n_obs`).

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

`interval="daily"` (default) returns every requested station at its own
cadence, each record labelled `"daily"` (automated) or `"periodic"`
(manual surveys), as DataBC does for its snow courses. `"periodic"` keeps
only the manual stations (the others are logged and skipped). To keep only
automated stations, pick them with `get_all_stations(methods="automated")`.
Any other interval raises `CanSWEError`. NorSWE behaves the same way.

```python
records = client.get_data("BCE-1A01P", begin_date="2024-10-01", end_date="2025-06-30")
# → {"station_id": "BCE-1A01P", "date": "2024-10-01", "variable": "snw",
#    "type": "swe", "value": 0.0, "units": "cm", "interval": "daily"}
```

### Station dict schema

`station_id`, `name`, `latitude`, `longitude`, `elevation_m`, `status`,
`source`, `operator`, `method_code`, `method`, `method_short`,
`automated`, `interval`, `has_swe`, `has_snwd`, `has_daily_swe`,
`daily_swe_seasons`, `last_daily_swe_season`, `max_swe_days_90d`,
`begin_date`, `end_date`, `n_swe`, `n_snwd`, `station_url`, plus the
CanSWE-specific `id_prefix`, `agency` (from `ID_PREFIXES`), `country`
(`"CA"`), `alt_names` and `alt_ids` (names and ids of duplicates the
authors merged into the station).

### Notes and gotchas

- Ids carry an agency prefix (`BCE-1A01P`, `ALE-05AA809P`, `HQ-…`,
  `SCD-…`); a trailing `P` on BC, Alberta and Yukon ids marks a pillow.
- `status` is `"Active"` when the station reported in the dataset's final
  snow season (2025 for v8): "still in the latest yearly update", not
  "reporting today".
- CanSWE duplicates the native BC (DataBC), Alberta, Yukon and
  Hydro-Québec records; it is where the Hydro-Québec GMON history lives
  (the live feed keeps only ~10 days, §11).

---

## 10. NorSWEClient

**Module:** `easysnowdata.stations.clients.norswe.norswe_client` (subclass of the CanSWE reader)
**Source:** NorSWE, Zenodo concept record 10287092
(<https://doi.org/10.5281/zenodo.10287092>); v3 = record 15263370,
`NorSWE-NorEEN_1979-2021_v3.zip` (one netCDF inside), published 2025-04-22.
Format reference: `Readme_NorSWE_19792021_v3.pdf`.
**Auth:** none
**Licence:** mixed, per source (netCDF `Distribution` attribute): CanSWE
data OGL-Canada except Hydro-Québec (CC BY-NC-SA 4.0); US data CC0 / US-PD;
SYKE CC BY 4.0; SLF GCOS ODbL + DbCL; NVE NLOD 2.0. NRCC and RIHMI-WDC
terms are not stated.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`. Storage layer being redesigned (see below).

NorSWE (Mortimer & Vionnet, 2025, ECCC) is a harmonised, quality-controlled
Northern Hemisphere in-situ SWE archive, 1979–2021: North America (CanSWE,
NRCS snow courses and SNOTEL, Maine, New Hampshire, NRCC), Russia (RIHMI-WDC
route surveys), Finland (SYKE), Norway (NVE), Switzerland (SLF GCOS) and
NOAA NOHRSC airborne-gamma flight lines. It is **frozen**: v3 ends
2021-07-31.

**Storage.** The latest version is resolved through the Zenodo API;
`record_id=` pins one. First use builds a local store once, under
`easysnowdata.config.cache_dir('stations', 'norswe')`:

1. the 2.4 GB zip is downloaded in 8 parallel byte ranges, resumable and
   md5-checked (about 50 minutes from Zenodo in testing; an interrupted
   run continues where it stopped);
2. the 21.6 GB netCDF is unzipped and the zip deleted;
3. the netCDF is converted to a **31 MB** Zarr store (about 8 minutes,
   1.6 GB of RAM) and deleted.

Peak disk use is about 22 GB, during steps 2–3; the client checks free
space first and says how much it needs (set `EASYSNOWDATA_CACHE_DIR` to
use a bigger disk). After that, a new client lists all 10,153 stations in
about 3 s and reads a station's record in well under a second. `path=`
takes an existing netCDF (converted once, left in place) or store.

```python
from easysnowdata.stations.clients import NorSWEClient
from easysnowdata.stations.clients.norswe.norswe_client import (
    VARIABLES, DATA_FLAGS, METHODS, ID_PREFIXES,
)
client = NorSWEClient()
```

### Variables, cadence and flags

Same as CanSWE (§9): `snw` / `snd` / `den` → cm / cm / %, the same
`AGENCY_FLAGS` + `QC_FLAGS` registry and `"<agency>,<qc>"` flag string.
`METHODS` adds code 64 (airborne gamma), which is `"periodic"` (a few
flights per season, not every year). NorSWE also uses agency codes `AI`,
`AM`, `GI`, `SE` and `MM-avgX` for airborne-gamma soil moisture, and folds
CanSWE's `P` (patchy) into `T`.

### Constructor

Same signature as `CanSWEClient` (`path`, `record_id`, `cache_dir`,
`session`, `timeout`, `max_retries`, `backoff`); the cache defaults to
`cache_dir("stations", "norswe")`.

### Key methods

`get_metadata()` and `get_data()` behave as in §9. `get_all_stations()`
takes the CanSWE filters plus three of its own:

```python
NorSWEClient.get_all_stations(
    active_only=False, bbox=None, sources=None, methods=None, daily_only=False,
    countries=None,       # "RU", "FI", ... (implied by the id prefix)
    prefixes=None,        # id prefixes, e.g. "NOHRSC" or "SNOTEL"
    mountain_only=False,  # mmask == 1
)

russia = client.get_all_stations(countries="RU")
gamma = client.get_all_stations(prefixes="NOHRSC")
```

### Station dict schema

The CanSWE fields (§9) plus `id_prefix`, `dataset` (source description
from `ID_PREFIXES`), `country`, `mmask` (1 = mountain site: GMBA v2 + 25 km
buffer or ≥ 2° slope, which switches the QC limits to 8 m / 8000 kg m-2)
and `airborne_gamma`. CanSWE-derived stations add `canswe_id` /
`canswe_prefix`; RIHMI stations add `rihmi_landscape`.

### Notes and gotchas

- Every id carries a source prefix: `CanSWE-BCE-1A01P`, `SNOTEL-<number>`
  (the readme says `SNT-`; the file uses `SNOTEL-`), `NRCS-<state>-<id>`
  (the `CA` ones carry CDEC ids, e.g. `NRCS-CA-KSR`), `MGS-`, `NRCC-`,
  `NHDES-`, `NOHRSC-`, `RIHMI-`, `SYKE-`, `NVE-`, `SLF-`.
- RIHMI ids end in `_1`, `_2`, `_3` for the field, forest and gully routes
  at one station (515 coordinates → 1163 ids).
- NOHRSC flight lines can cross the US–Canada border and are tagged
  `"US"`.
- `status` is always `"Inactive"` for v3: the archive ends in 2021. The
  time axis runs to 2021-12-31, so the 2022 season holds only Aug–Dec 2021.
- Heavy duplication with the native clients (AWDB, CDEC, NVE, SYKE, SLF,
  RIHMI) and with CanSWE itself; filter on `id_prefix` before merging.

---

## 11. HydroQuebecClient

**Module:** `easysnowdata.stations.clients.hydroquebec.hydroquebec_client`
**Source:** Hydro-Québec open data, Opendatasoft dataset
[`donnees-hydrometeorologiques`](https://donnees.hydroquebec.com/explore/dataset/donnees-hydrometeorologiques/)
(`/api/explore/v2.1`, `/exports/json`, ODSQL `where` / `group_by`)
**Auth:** none
**Licence:** CC BY-NC 4.0 — attribution, **no commercial use**; the portal
gives no quality guarantee.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Near-real-time SWE and snow depth from ~95 GMON stations in northern and
central Québec, plus the met and hydrometric variables of Hydro-Québec's
own network across its hydroelectric watersheds.

> **Rolling window, no history.** The feed holds only the last **~10
> days** of hourly data (`WINDOW_DAYS`); older values disappear and no
> archive endpoint exists. Accumulate history yourself by calling
> `get_data()` at least every few days, or use CanSWE (§9) for the GMON
> record through July 2025.

```python
from easysnowdata.stations.clients import HydroQuebecClient
from easysnowdata.stations.clients.hydroquebec.hydroquebec_client import VARIABLES, DATA_FLAGS
client = HydroQuebecClient()
```

### Variables

Keys are the native French labels (`composition_depil_type_point_donnee`).
`daily` is the aggregation `interval="daily"` applies.

| Key | `type` | Native | Emitted | Daily |
|---|---|---|---|---|
| `Équivalent en eau de la neige` | `swe` | mm | **cm** (× 0.1) | last value |
| `Épaisseur de neige` | `snwd` | cm | cm | last value |
| `Précipitation` | `precip` | mm | mm | sum |
| `Température Maximum` | `temp_max` | °C | °C | max |
| `Température Minimum` | `temp_min` | °C | °C | min |
| `Humidité relative.2 mètres` | `rh` | % | % | mean |
| `Vitesse du vent.10 mètres` | `wind_spd` | km/h | km/h | mean |
| `Direction du vent.10 mètres` | `wind_dir` | ° | degrees | circular mean |
| `Niveau` | `other` | m | m | mean |
| `Débit` | `other` | m³/s | m3/s | mean |
| `Température de l'eau` | `other` | °C | °C | mean |

`variables` accepts the labels or standardized types; `None` = SWE and
snow depth. `get_stations(variables="all")` lists every station, including
hydrometric ones.

### Flags

None: `DATA_FLAGS` is empty. `include_flags=True` puts the native
`type_mesure` (`Instantanée` / `Moyenne` / `Somme` / `Maximum` /
`Minimum`) in `flag` on hourly records and `None` on daily ones.

### Constructor

```python
HydroQuebecClient(
    base_url: str = "https://donnees.hydroquebec.com/api/explore/v2.1",
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_stations(variables, active_only, bbox)` / `get_all_stations(active_only, bbox)` → `list[dict]`

`get_all_stations()` = stations listing SWE or snow depth. `active_only`
keeps stations with at least one valid value in the window.

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables`; `begin_date` / `end_date` / `n_obs`
describe the **current window**, not a period of record.

#### `get_raw_records(station_ids, variables, begin_utc, end_utc)` → `list[dict]`

Raw portal rows (junk `date = null` rows excluded).

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

`interval="hourly"` (default, native) or `"daily"`. Hourly records carry
`datetime` in UTC (`YYYY-MM-DDTHH:MM:SSZ`) and `date` is its UTC date.
Daily values aggregate over the **America/Toronto local day**; sum, mean,
max and min need at least `MIN_HOURS_DAILY` (20) valid hours, otherwise
`None`.

```python
recs = client.get_data(["1-10613"], variables="swe", interval="daily")
# → {"station_id": "1-10613", "date": "2026-10-08", "variable":
#    "Équivalent en eau de la neige", "type": "swe", "value": 0.0,
#    "units": "cm", "interval": "daily"}
```

### Station dict schema

| Key | Description |
|---|---|
| `station_id` | Hydro-Québec `identifiant`, e.g. `"1-10613"` |
| `name`, `latitude`, `longitude`, `elevation_m` | `elevation_m` is null at ~55 % of sites (mostly hydrometric) |
| `status` | `"Active"` = a valid value in the window and not decommissioned |
| `region`, `region_code` | Québec administrative region |
| `hq_code` | 4-character code at the end of the name, e.g. `CM3V` |
| `operator`, `network`, `measurement_method` | GMON passive gamma for SWE stations |
| `commissioned`, `decommissioned` | `1800-01-01` placeholders → `None` |
| `has_swe`, `has_snwd` | |
| `variables` | `{label: {n_obs, first, last}}` in the window |
| `begin_date`, `end_date`, `latest_observation` | span of the window held |
| `license`, `station_url` | |

### Notes and gotchas

- `valeur` is text; `-999.00` means missing. Each series also carries ~4
  junk rows with `date = null` and a label in `valeur`; they are dropped.
- SWE is published hourly but the GMON value changes only every **6 h**
  (~01, 07, 13, 19 UTC); the hours between repeat the last reading.
- Snow depth hovers around zero with negative noise; negatives are nulled
  by the 0–15 m clamp. Early-season GMON SWE can be spurious (seen: one
  station alternating 0 / 75 mm every 6 h with depth ≈ 0), so cross-check
  SWE against depth.
- The upstream JSON refreshes hourly but the portal re-processes it about
  twice a day, so the newest value is typically 1–10 h old.
- Two sites carry both a mean and an instantaneous water level; the
  variable's native `measure` wins on duplicate timestamps.

---

## 12. AlbertaClient

**Module:** `easysnowdata.stations.clients.alberta.alberta_client`
**Source:** Alberta Environment and Protected Areas (AEPA),
[Alberta River Basins](https://rivers.alberta.ca/)
**Auth:** none
**Licence:** none stated; the page's download disclaimer says data are
provisional and used at your own risk. OGL–Alberta has not been verified
to cover this data.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Automated snow-pillow SWE and snow depth in the Alberta Rockies and
foothills (15-minute, near real time, back to the 1980s at most AEPA
pillows) plus manual snow-course SWE and depth (back to the 1960s–1970s).

> **No documented API.** The client calls the same endpoints the public
> page's JavaScript calls (inspected 2026-10); they may change without
> notice. Ask Alberta Environment for sanctioned access before running it
> on a schedule.

```python
from easysnowdata.stations.clients import AlbertaClient
from easysnowdata.stations.clients.alberta.alberta_client import VARIABLES, DATA_FLAGS, OPERATORS
client = AlbertaClient()
```

### Variables

Keys are the ARB `ParNumber`s.

| Key | `type` | Native | Emitted | Intervals |
|---|---|---|---|---|
| `SW` | `swe` | mm | **cm** (× 0.1) | `sub_hourly`, `hourly`, `daily` |
| `SD` | `snwd` | cm | cm | `sub_hourly`, `hourly`, `daily` |
| `SW-SS` | `swe` | mm | **cm** | `periodic` (snow survey) |
| `SD-SS` | `snwd` | cm | cm | `periodic` (snow survey) |
| `TA` | `temp` | °C | °C | `sub_hourly`, `hourly`, `daily` |
| `PC` | `precip` | mm | mm | `daily` only |

A standardized type resolves to the parameter available at the requested
interval: `swe` → `SW` for daily/hourly, `SW-SS` for periodic. Naming a
parameter at an interval it lacks raises `AlbertaError`. `variables=None`
= `SW` + `SD` (or `SW-SS` + `SD-SS` for periodic). Snow values outside
0–1500 cm (including small negative pillow readings) are nulled.

### Intervals

- `"sub_hourly"` — the native `C.Corrected-Seasonal` series (15-minute at
  AEPA pillows since the 2010s, hourly in older years and at mirrored
  SNOTEL sites).
- `"hourly"` — the native samples that fall exactly on HH:00 MST
  (instantaneous, not a mean).
- `"daily"` (default) — for SW, SD and PC the source's own WISKI daily
  aggregate (`A.DayMean`; `A.DayTotal` for PC). For TA, and wherever the
  source aggregate is missing or belongs to another sensor, an arithmetic
  mean of the native samples in the MST day, kept when they span ≥ 20
  hours. PC has no fallback (a mean of a cumulative gauge is meaningless).
- `"periodic"` — manual surveys (`P.Survey`), one value per visit,
  typically monthly Jan–Jun.

### Flags

None. `DATA_FLAGS` is empty and `include_flags=True` adds `flag: None`.

### Constructor

```python
AlbertaClient(
    base_url: str = "https://rivers.alberta.ca",
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,   # gets the required Referer header
)
```

### Key methods

#### `get_all_stations(active_only, bbox)` → `list[dict]`

Every station with automated or manual SWE / snow depth. Stations without
coordinates are dropped when `bbox` is given.

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables`, one entry per parameter × supported
interval, with `ts_id` / `ts_name` and MST begin/end dates.

#### `download_series(ts_id, begin_date, end_date)` → `(header, rows)`

Raw WISKI download for one time-series id; rows are `(naive MST datetime,
raw value)`.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Dates are inclusive and in MST. Sub-daily records carry `datetime` as ISO
8601 with `-07:00`. Unknown stations and failed downloads are logged and
skipped.

```python
recs = client.get_data("05BB803", variables="swe", interval="hourly",
                       begin_date="2026-03-01", end_date="2026-03-02")
# → {"station_id": "05BB803", "date": "2026-03-01",
#    "datetime": "2026-03-01T00:00:00-07:00", "variable": "SW",
#    "type": "swe", "value": 41.2, "units": "cm", "interval": "hourly"}
```

### Station dict schema

`station_id` (ARB station number, e.g. `"05BB803"`), `name` (operator
suffix removed), `latitude`, `longitude`, `elevation_m` (always `None`:
not published), `status`, `source_status` (the catalog's own status),
`state` (AB / BC / MT / SK), `operator`, `operator_code`,
`measurement_method`, `has_swe`, `has_snwd`, `has_pillow_swe`,
`has_survey`, `begin_date`, `end_date`, `parameters`, `station_url`.

### Endpoints used

| Purpose | Endpoint |
|---|---|
| Time-series catalog (layer 39) | `GET /proxy.ashx?<MapServer>/39/query?...` |
| Station points (layer 11) | `GET /proxy.ashx?<MapServer>/11/query?...` |
| Data | `GET /WiskiLiveDataService/Download?tsId=...&from=...&to=...&json=true` |

`<MapServer>` is
`https://geospatial.alberta.ca/europa/rest/services/arb/alberta_river_basins/MapServer`.

### Notes and gotchas

- The MapServer itself answers `499 Token Required`; the site proxy
  answers 403 without `Referer: https://rivers.alberta.ca/`, which the
  client sends.
- **Timestamps are MST (UTC−7) all year**, verified against AWDB at the
  two mirrored SNOTEL sites. The catalog's `*DateUtc` fields are not true
  UTC; the client uses the `*AsCurrentTime` fields.
- The operator is the station-name suffix (`- EPA`, `- BC`, `- Snotel`,
  `- MSC`, …), kept as `operator_code`; only unambiguous codes are
  expanded in `operator` (`OPERATORS`). Filter on `operator_code` to drop
  the 27 BC and 7 SNOTEL mirrors, which their own clients serve.
- `status` is `"Active"` when an automated snow series is flagged
  `current` or the latest survey is within 550 days.
- 18 BC stations in the time-series table are missing from the station
  layer and have no coordinates.
- An unknown `tsId` gives HTTP 500.

---

## 13. DWDClient

**Module:** `easysnowdata.stations.clients.dwd.dwd_client`
**Source:** Deutscher Wetterdienst Climate Data Center,
<https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/>
(dataset descriptions in each product directory)
**Auth:** none
**Licence:** CC BY 4.0; cite as "Deutscher Wetterdienst".
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Daily manual SWE (`WASH_6`) at ~550 DWD and partner climate stations, plus
daily snow depth (`SH_TAG`) and new snow (`NSH_TAG`) at the ~2,300
currently reporting precipitation stations — including the mountain
synoptic stations (Zugspitze, Feldberg, Fichtelberg, Brocken, Großer
Arber) that are not in the SWE product.

```python
from easysnowdata.stations.clients import DWDClient
from easysnowdata.stations.clients.dwd.dwd_client import VARIABLES, DATA_FLAGS, PRODUCTS
client = DWDClient()
```

### Variables

| Key | `type` | Native | Emitted | Product |
|---|---|---|---|---|
| `WASH_6` | `swe` | mm | **cm** (× 0.1) | `water_equiv` |
| `WAAS_6` | `other` | mm | cm | `water_equiv` (SWE of the cored sample) |
| `ASH_6` | `other` | cm | cm | `water_equiv` (height of the cored sample) |
| `SH_TAG` | `snwd` | cm | cm | `more_precip`, fallback `water_equiv` |
| `NSH_TAG` | `snowfall` | cm | cm | `more_precip` |
| `RS` | `precip` | mm | mm | `more_precip` |

`variables=None` = `WASH_6` + `SH_TAG`. `WAAS_6` is typed `other` so that
`"swe"` resolves to `WASH_6` only (in 2025–26 they are identical on every
row); requesting the type `"other"` raises. Only `interval="daily"`
exists. Days with a missing value are not emitted. Snow values outside
0–15 m (new snow: 0–5 m) are nulled.

### Flags

`QN_6` is a **row-level** quality level; `include_flags=True` emits it as
`flag`.

| `QN_6` | Meaning |
|---|---|
| `1` | Only formal control |
| `2` | Controlled with individually defined criteria |
| `3` | Automatic control and correction |
| `5` | Historic, subjective procedures |
| `7` | Second control done, before correction |
| `8` | Quality control outside ROUTINE |
| `9` | Not all parameters corrected |
| `10` | Quality control finished, all corrections finished |

Data up to 1980 reach at best QN 5, later data QN 10.

### Constructor

```python
DWDClient(
    base_url: str = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily",
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
    cache: bool = True,                  # cache historical zips on disk
    cache_path: str | Path | None = None,  # default cache_dir("stations", "dwd")
)
```

### Key methods

#### `get_stations(products, active_only, bbox)` / `get_all_stations(active_only, bbox)` → `list[dict]`

`products` is any of `"water_equiv"` (SWE stations) and `"more_precip"`;
the default (and `get_all_stations()`) merges both by id.

```python
swe_only = client.get_stations(products="water_equiv", active_only=True)
```

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables`. `begin_date` / `end_date` are the
product's period of record at the station, not the individual column's.

#### `get_station_rows(station_id, product, begin_date, end_date)` → `dict`

Merged historical + recent rows in native units:
`{YYYY-MM-DD: {column: value or None, "QN_6": str}}`. Files that cannot
hold the window are not downloaded.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Ids are accepted with or without zero padding (`"00020"`, `"20"`, `20`).

```python
recs = client.get_data(["01346"], begin_date="2026-01-01", end_date="2026-01-31",
                       include_flags=True)
# → {"station_id": "01346", "date": "2026-01-05", "variable": "WASH_6",
#    "type": "swe", "value": 4.3, "units": "cm", "interval": "daily", "flag": "3"}
```

### Station dict schema

`station_id` (5-digit), `name`, `latitude`, `longitude`, `elevation_m`,
`status` (`bis_datum` within ~400 days), `state` (Bundesland),
`operator`, `measurement_method`, `release` (`Abgabe` column), `has_swe`,
`has_snwd`, `products` (`{product: {begin_date, end_date}}`),
`begin_date`, `end_date`, `swe_begin_date`, `swe_end_date`, `station_url`
(`None`: DWD has no per-station page).

### Files used

| Purpose | Path under the base URL |
|---|---|
| Station lists (fixed-width, latin-1) | `{product}/recent/Wa_Tageswerte_Beschreibung_Stationen.txt`, `RR_…` |
| Directory listings (to find historical file names) | `{product}/historical/`, `{product}/recent/` |
| Recent data (~500 days, QC unfinished, rebuilt daily) | `{product}/recent/tageswerte_{Wa,RR}_{id5}_akt.zip` |
| Historical data (QC finished, rebuilt yearly) | `{product}/historical/tageswerte_{Wa,RR}_{id5}_{begin}_{end}_hist.zip` |

Historical wins on overlapping dates. Historical zips are immutable per
file name and cached under `easysnowdata.config.cache_dir("stations",
"dwd")`; recent zips are always re-downloaded.

### Notes and gotchas

- SWE is a **manual** core, mostly not measured every day and only above
  a minimum depth (often 5 cm), at the 06 UTC morning observation. Zero
  SWE is never reported: no measurement and no snow are both missing. So
  it is not daily at most stations.
- Mountain SWE ended at the big summits when they went automatic
  (Feldberg 2014, Arber 2016, Fichtelberg 2019, Brocken 2020). Snow depth
  (`SH_TAG`) from `more_precip` is the stronger product.
- `water_equiv` files end at the station's last SWE day, so out of season
  their newest row is months old while `more_precip` runs to yesterday.
- `NSH_TAG` was added to `more_precip` in January 2019; the automated
  mountain stations report none.
- The `RSF` precipitation-form code is not exposed (no code table in the
  dataset description).

---

## 14. SYKEClient

**Module:** `easysnowdata.stations.clients.syke.syke_client`
**Source:** Finnish Environment Institute (Syke) hydrology OData API
"Hydrologiarajapinta",
<https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1/odata/>
(OData v3, schema at `$metadata`)
**Auth:** none
**Licence:** CC BY 4.0; attribute "Finnish Environment Institute (Syke)".
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Finnish snow-course SWE: manual snow-tube surveys along fixed courses,
normally once a month from November to May, 1947 to present. The same API
serves Syke's areal (catchment-average) SWE, which is not a point
measurement and has its own methods.

```python
from easysnowdata.stations.clients import SYKEClient
from easysnowdata.stations.clients.syke.syke_client import (
    VARIABLES, AREAL_VARIABLES, DATA_FLAGS, SITE_STATES, EQUIPMENT,
)
client = SYKEClient()
```

### Variables

| Key | `type` | Native | Emitted | Interval | Served by |
|---|---|---|---|---|---|
| `S` | `swe` | mm | **cm** (× 0.1) | `periodic` | `get_data()` (table `LumiPiste`) |
| `SA` | `swe` | mm | **cm** (× 0.1) | `semi_monthly` | `get_areal_swe()` (table `LumiAlue`) |

SWE only: the API publishes no snow depth or density, and asking for
`"snwd"` raises. SWE outside 0–15 m is nulled.

### Flags

`include_flags=True` adds `flag` = the `Lippu_Id` as a string.

| Code | Meaning |
|---|---|
| `96` | Mittaus — measurement result |
| `110` | Lumimallilaskenta — calculated with the snow model |
| `117` | Lumimallilaskenta 2022 — calculated with the snow-course model introduced in autumn 2022 |

Modelled values (110, 117) are returned as `None` unless the client is
built with `keep_modelled=True`. None were present in October 2026.

### Constructor

```python
SYKEClient(
    base_url: str = "https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1/odata",
    timeout: int = 60,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
    keep_modelled: bool = False,
)
```

### Key methods

#### `get_all_stations(active_only, bbox)` → `list[dict]`

All snow courses (`Suure_Id` 8). The period of record is not in the site
table; use `get_metadata()`.

#### `get_metadata(station_id)` → `dict`

Site metadata plus `data_variables` with period of record and `n_obs`.
Works for snow courses and areal-SWE units (`kind` = `"snow_course"` /
`"areal"`).

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

Only `interval="periodic"`. Station ids are integers (`Paikka_Id`);
anything else raises `SYKEError`. Date bounds are applied server side.

```python
recs = client.get_data(["1234"], begin_date="2024-01-01", end_date="2024-05-31")
# → {"station_id": "1234", "date": "2024-03-15", "variable": "S",
#    "type": "swe", "value": 12.3, "units": "cm", "interval": "periodic"}
```

#### `get_areal_units(active_only, bbox)` / `get_areal_swe(area_ids, bbox, begin_date, end_date, active_only)`

Areal-SWE units (`Suure_Id` 7) and their values on the 1st and 16th of
each month, year round (0 in summer). With no ids and no bbox every unit
is fetched (≈116,000 rows for the full record), so pass dates.

### Station dict schema

`station_id` (`Paikka_Id` as a string), `name`, `latitude`, `longitude`,
`elevation_m` (always `None`: no elevation column), `status`,
`status_native` / `status_description` (from `SITE_STATES`),
`site_number` (official `Nro`), `municipality`, `basin`, `sub_basin`,
`operator`, `owner_id`, `measurement_method` / `equipment_id` (from
`EQUIPMENT`), `etrs_tm35fin_n` / `etrs_tm35fin_e`,
`coordinate_accuracy_code`, `station_url` (`None`).

### Endpoints used

| Purpose | Table |
|---|---|
| Sites | `GET /Paikka?$filter=Suure_Id eq 8` (courses) / `7` (areal) |
| Owners | `GET /Omistaja` |
| Snow-course values | `GET /LumiPiste?$filter=...` |
| Areal values | `GET /LumiAlue?$filter=...` |

### Notes and gotchas

- The server returns at most 500 rows per response and appends
  `odata.nextLink`; the client follows it. `$count` is not supported.
- OData v3 has no `in`, so ids go as `or` chains in chunks of 15: the
  server rejects filters over 100 parse nodes with HTTP 400.
- Coordinates are converted from ETRS-TM35FIN (EPSG:3067), falling back
  to KKJ (EPSG:2393). The `KoordLat` / `KoordLong` DDMMSS fields are on
  the KKJ datum, ~175 m off if read as WGS84.
- `status`: "Jatkuva" (continuous) and "Määräaikainen" (fixed-term) are
  Active; "Lopetettu", "Satunnainen" and "Ei poiminnassa" are Inactive.

---

## 15. SLFClient

**Module:** `easysnowdata.stations.clients.slf.slf_client`
**Source:** WSL Institute for Snow and Avalanche Research SLF — EnviDat
(CKAN `package_show`) and the file service <https://measurement-data.slf.ch/>
**Auth:** none
**Licence:** per dataset (below); terms at
<https://www.slf.ch/en/services-and-products/slf-data-service/> — cite the
DOI and credit SLF with a link to www.slf.ch.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Four open Swiss snow-station datasets behind one `dataset=` switch. Each
is a different set of stations with its own ids and cadence, not another
route to the same data.

| `dataset` | Content | Stations | Variables (intervals) | Period | Licence |
|---|---|---|---|---|---|
| `"gcos"` (default) | GCOS SWE data, DOI 10.16904/15 | 11 study plots, 1,350–2,540 m | `SWE`, `HS` (periodic) | 1936 (Weissfluhjoch) / 1947 → end of last season; yearly | ODbL 1.0 + DbCL |
| `"new_snow"` | New snow water equivalent, DOI 10.16904/envidat.590 | 41 study plots | `HS`, `HN_1D`, `HNW_1D` (daily); `SWE`, `HS_PROFILE` (periodic) | 2016-09-01 → 2022-08-31, static | CC BY 4.0 |
| `"study_plot"` | Manual measuring network, DOI 10.16904/envidat.408 | 135 (100 active) | `HS`, `HN_1D`, `HNW_1D` (daily) | 1930 → yesterday | CC BY 4.0 |
| `"imis"` | IMIS automatic stations, DOI 10.16904/envidat.406 | ~180 snow / flowcapt | `HS`, `HN_1D` (daily); `HS`, `TA_30MIN_MEAN` (sub_hourly) | 1992 → yesterday | CC BY 4.0 |

No open daily SWE exists in any of them: the only SWE is manual and
biweekly (`gcos`, `new_snow` profiles).

```python
from easysnowdata.stations.clients import SLFClient
from easysnowdata.stations.clients.slf.slf_client import VARIABLES, DATA_FLAGS, DATASETS
gcos = SLFClient()                     # dataset="gcos"
imis = SLFClient(dataset="imis")
```

### Variables

| Key | `type` | Native | Emitted | Notes |
|---|---|---|---|---|
| `SWE` | `swe` | mm | **cm** (× 0.1) | manual snow pit, ~every two weeks |
| `HS` | `snwd` | cm | cm | gcos: at the pit; new_snow: stake; study_plot: morning reading; imis: ultrasonic (daily = median over 24 h) |
| `HS_PROFILE` | `snwd` | cm | cm | new_snow only; pairs with `SWE` |
| `HN_1D` | `snowfall` | cm | cm | manual at study plots; **modelled by SNOWPACK** at IMIS |
| `HNW_1D` | `other` | mm | mm | new-snow water equivalent; excludes rain, so not `precip` |
| `TA_30MIN_MEAN` | `temp` | °C | °C | IMIS only, `sub_hourly` |

Missing values (`-9999`, empty) are skipped. SWE and depth outside 0–15 m
and new snow outside 0–5 m are nulled, as are IMIS's small negative HS
readings.

### Flags

None: `DATA_FLAGS` is empty and `include_flags=True` adds `flag: None`.

### Constructor

```python
SLFClient(
    dataset: str = "gcos",               # "gcos" | "new_snow" | "study_plot" | "imis"
    cache_dir: str | Path | None = None, # default cache_dir("stations", "slf")
    cache_ttl_hours: float = 24.0,       # static datasets never expire
    timeout: int = 120,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

An unknown `dataset` raises `SLFError`.

### Key methods

#### `get_all_stations(active_only, bbox)` → `list[dict]`

Stations of the selected dataset. IMIS wind-only stations are skipped.

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables`, with `begin_date` / `end_date` /
`n_obs` computed from the data file (not for IMIS 30-minute variables,
whose files are large), and `terms_url`.

#### `ckan_package(ckan_id=None)` → `dict`

The EnviDat CKAN record for the dataset.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags)` → `list[dict]`

`interval=None` returns each variable at its native cadence (`"periodic"`
for biweekly pits, `"daily"` otherwise). `"periodic"`, `"daily"` or (IMIS)
`"sub_hourly"` select explicitly; anything the dataset lacks raises.
Sub-hourly records carry `datetime` in UTC. Files are fetched whole; the
date bounds only filter.

```python
recs = gcos.get_data("5WJ", begin_date="2024-10-01")
# → {"station_id": "5WJ", "date": "2025-01-01", "variable": "SWE",
#    "type": "swe", "value": 21.4, "units": "cm", "interval": "periodic"}
imis.get_data("WFJ2", variables="snwd", interval="sub_hourly", begin_date="2026-01-01")
```

### Station dict schema

`station_id`, `name`, `latitude`, `longitude`, `elevation_m`, `status`,
`operator`, `dataset`, `doi`, `license`, `measurement_method`,
`period_of_record` (`"first/last"` where known), `has_swe`, `has_snwd`,
`station_url`. `study_plot` / `imis` add `network` and `station_type`;
`new_snow` adds LV03 coordinates (`easting_lv03`, `northing_lv03`),
converted to WGS84 with swisstopo's approximate formulas (~1 m); `gcos`
adds `start_year`.

### Notes and gotchas

- Station ids differ per dataset: `"5WJ"` (gcos / new_snow), `"5WJ0"`
  (study_plot), `"WFJ2"` (imis). GCOS and new-snow codes are the study-plot
  codes without the trailing `0`.
- Daily values carry the date of the morning observation that closes the
  24 h. IMIS daily values are stamped 00:00 UTC though SLF's README says
  06 UTC; the date label is used as-is.
- GCOS `status` is `"Active"` when the station reported within 200 days of
  the file's newest value and that value is under 800 days old (the file
  is extended once a year).
- The GCOS S3 folder and the new-snow zip are resolved from the CKAN
  record at runtime (GCOS falls back to a fixed URL if CKAN is down).
- The live API (measurement-api.slf.ch, last 1–7 days) is not used: the
  file service already reaches yesterday.

---

## 16. RIHMIClient

**Module:** `easysnowdata.stations.clients.rihmi.rihmi_client` (packaged catalogues: `route_survey_stations.csv`, `snow_cover_stations.csv`)
**Source:** ВНИИГМИ-МЦД (RIHMI-WDC, Obninsk), <http://meteo.ru/>, via
manual Web АИСОРИ-М exports; plus the public NSIDC copy *Former Soviet
Union Hydrological Snow Surveys, 1966–1996*, NSIDC G01170 v1
(<https://doi.org/10.7265/N58C9T60>, <https://noaadata.apps.nsidc.org/NOAA/G01170/>)
**Auth:** none for NSIDC. **RIHMI data need a (free) АИСОРИ-М login**,
which this client does not automate.
**Licence:** NSIDC — free to use with citation (Krenke 1998). RIHMI —
"open access" per the АИСОРИ-М footer, but no licence or redistribution
terms published.
**Status:** not wired into `stations.networks`, the catalog or
`global_snow_networks`.

Two RIHMI datasets: **route snow surveys** (snow courses, 517 stations,
1966 to present; every 10 days through the cold season and every 5 days
in intense melt, on separate field, forest and — on request — ravine
routes) and **daily snow cover** at 620 met stations of Russia and the
former USSR (stake depth plus cover degree and three QC codes).

| `interval` | Dataset | Public, no login | Needs АИСОРИ-М export |
|---|---|---|---|
| `"periodic"` (default) | route surveys | NSIDC G01170: 1,345 stations 1966–1990, ~300 of them 1991–1996 | current data |
| `"daily"` | snow cover | none (use `GHCNDClient`, `RSM000xxxxx`, for daily depth at many of the same stations) | all of it |

```python
from easysnowdata.stations.clients import RIHMIClient
from easysnowdata.stations.clients.rihmi.rihmi_client import (
    VARIABLES, DATA_FLAGS, SNOW_BEDDING_CODES, SNOW_STRUCTURE_CODES, parse_aisori_text,
)
client = RIHMIClient()                          # NSIDC only
client = RIHMIClient(data_dir="~/aisori")       # + your АИСОРИ-М exports
```

### Variables

Route surveys (`periodic`):

| Key | `type` | Native | Emitted | In NSIDC |
|---|---|---|---|---|
| `SWE` | `swe` | mm | **cm** (× 0.1) | yes |
| `HS_MEAN` | `snwd` | cm | cm | yes |
| `HS_MAX`, `HS_MIN` | `other` | cm | cm | yes |
| `DENSITY` | `density` | g/cm³ | % (× 100) | yes |
| `WATER_TOTAL` | `other` | mm | cm | no |
| `ICE_CRUST_THICKNESS`, `SATURATED_LAYER` | `other` | mm | mm | yes |
| `WATER_LAYER` | `other` | mm | mm | no |
| `COVER_SURROUNDINGS`, `COVER_ROUTE`, `ICE_CRUST_COVER` | `other` | tenths | % (× 10) | yes |
| `SNOW_BEDDING`, `SNOW_STRUCTURE` | `other` | code | code (`SNOW_BEDDING_CODES`, `SNOW_STRUCTURE_CODES`) | yes |

Daily snow cover (`daily`): `H` (`snwd`, stake depth, cm → cm) and `S`
(`other`, cover degree of the surroundings, tenths → %).

`variables=None` = `SWE` + `HS_MEAN` (periodic) or `H` (daily). A native
name from the other interval raises. Values outside each variable's
`valid_range` and snow outside 0–15 m are nulled.

### Flags

| Prefix | Source | Codes |
|---|---|---|
| `NSIDC:<qc>` | NSIDC column 17 | `0` not computed; `1`/`2` density fixed by NSIDC; `3`/`4` depth or SWE reset from the other two; `5` suspect, unchanged |
| `Q1:<a>,Q2:<b>,Q3:<c>` | АИСОРИ daily snow cover | Q1 depth state (`9` = rejected, value `None`); Q2 step-consistency check; Q3 consistency with air temperature |

Route-survey АИСОРИ records have no flag field (`flag` is `None`).

### Constructor

```python
RIHMIClient(
    data_dir: str | PathLike | None = None,  # folder of АИСОРИ-М exports (*.zip, or extracted *.txt)
    use_nsidc: bool = True,                  # include NSIDC G01170
    cache_dir: str | PathLike | None = None, # default cache_dir("stations", "rihmi")
    timeout: int = 300,
    max_retries: int = 3,
    backoff: int = 4,
    session: requests.Session | None = None,
)
```

### Key methods

#### `get_all_stations(active_only, bbox)` → `list[dict]`

The union of the two packaged RIHMI catalogues and (with `use_nsidc`)
NSIDC's `station.loc`: 1,552 stations.

#### `get_metadata(station_id)` → `dict`

Station dict plus `data_variables`. `begin_date` is the catalogue start
year; `end_date` is `1996-12-31` for NSIDC-only stations and `None`
otherwise (ongoing behind the login); `n_obs` is `None`.

#### `get_data(station_ids, variables, bbox, begin_date, end_date, interval, include_flags, route_types, files, source)` → `list[dict]`

Extra arguments: `route_types` (`"field"`, `"forest"`, `"ravine"`,
`"unknown"`; periodic only), `files` (export files for this call,
overriding `data_dir`) and `source` (`"auto"` merges local exports,
which win, with NSIDC; `"nsidc"` or `"files"` restrict to one). Daily data
with no export raises `RIHMIError`. Records add `route_type` and `source`
(`"aisori_export"` or `"nsidc_g01170"`), emitted field → forest → ravine
→ unknown per station and date, so a first-value pivot prefers the field
route.

```python
recs = client.get_data("27612", begin_date="1985-01-01", end_date="1985-04-30",
                       route_types="field")
# → {"station_id": "27612", "date": "1985-01-10", "variable": "SWE",
#    "type": "swe", "value": 6.1, "units": "cm", "interval": "periodic",
#    "route_type": "field", "source": "nsidc_g01170"}
```

#### `parse_aisori_text(text, kind=None)` → `list[dict]` (module function)

Parses an АИСОРИ-М export or `IIIII.dat` file into native rows; `kind` is
inferred from the field count (19 → route survey, 9 → snow cover).

### Station dict schema

`station_id` / `wmo_id` (5-digit WMO index), `name` (Cyrillic from RIHMI,
else NSIDC's Latin), `name_latin`, `latitude`, `longitude`,
`elevation_m`, `status`, `operator`, `measurement_method`, `has_swe`,
`has_snwd`, `in_route_survey_catalogue`, `in_snow_cover_catalogue`,
`in_nsidc_g01170`, `route_survey_start_year`, `snow_cover_start_year`,
`public_period`, `login_period`, `note`, `station_url`.

### Manual export (АИСОРИ-М)

1. Register (free, no confirmation step) at
   <http://aisori-m.meteo.ru/waisori/registration.xhtml> and log in through
   «Переход на страницу авторизации».
2. «Выбор данных» → «Раздел БД: Сутки» → «источник данных»
   `SNMAR - Маршрутные снегосъемки` (route surveys) or `SNOW - Снежный
   покров` (daily snow cover). Tick stations, move them right with «>», then
   «Дальше».
3. Pick a delimiter, set «Год» От/До, press «Все» (every parameter), then
   «Результат» and «Получить результат».
4. Press «Загрузить» on the result page. The `.zip` holds `wr<id>.txt`
   (data), `fld<id>.txt` (columns) and `statlist<id>.txt` (stations).
5. Point `RIHMIClient(data_dir=...)` at the folder of zips, or pass
   `get_data(files=[...])`.

### Notes and gotchas

- **АИСОРИ-М exports** are checked against real exports of both datasets
  in both delimiter variants. Point `data_dir=` at the downloaded `.zip`
  files as they are. The space-delimited variant is fixed width, not
  whitespace separated; the reader handles that, but the `;` variant is the
  safer choice for other tools. In October 2026 AISORI-M offered route
  surveys for 68 of the 517 catalogue stations, with data to 2024.
- NSIDC `.syn` files: `-2` means missing (documented); `-1` also occurs
  (often in ice-crust cover, ~7,300 times in SWE) and is treated as
  missing. Coordinates are packed degrees + minutes (`5545` = 55°45′), not
  "value / 100" as the NSIDC guide says.
- `status` is catalogue-based: `"Active"` = in the route-survey catalogue,
  or in the snow-cover catalogue without a «закрыта» / «законсервирована»
  note. Whether a station still reports cannot be checked without a login.
- The NSIDC `syn.tar` files (~50 MB in all) download once into the cache.
- Before August 1959 the daily `S` field is a 0/1 code, returned as `None`.

---

## 17. Error Handling

| Client | Exception | Scenarios |
|---|---|---|
| AWDBClient | `AWDBError` | HTTP 4xx/5xx, network timeout, value limit |
| CDECClient | `CDECError` | HTTP 4xx/5xx, network timeout, HTML parse failure |
| DataBCClient | `DataBCError` | HTTP 4xx/5xx, network timeout, malformed CSV |
| NVEClient | `NVEError` | HTTP 4xx/5xx, network timeout, station not found |
| YukonClient | `YukonError` | HTTP 4xx/5xx, network timeout, malformed CSV |
| GHCNDClient | `GHCNDError` | HTTP 4xx/5xx, network timeout, corrupt station file, unknown station, unknown variable or interval |
| CanSWEClient | `CanSWEError` | Zenodo lookup/download failure, unknown variable or interval |
| NorSWEClient | `NorSWEError` | Zenodo lookup/download failure, unknown variable or interval |
| HydroQuebecClient | `HydroQuebecError` | HTTP 4xx/5xx, network timeout, unknown variable or interval |
| AlbertaClient | `AlbertaError` | HTTP 4xx/5xx (incl. proxy 403, HTTP 500 for an unknown time series), unknown variable or interval, variable not offered at that interval |
| DWDClient | `DWDError` | HTTP 4xx/5xx, corrupt zip, unknown or non-numeric station, unknown variable, product or interval |
| SYKEClient | `SYKEError` | HTTP 4xx/5xx (incl. OData filter-size 400), unknown variable or interval, non-integer station id |
| SLFClient | `SLFError` | HTTP 4xx/5xx, EnviDat lookup failure, unknown dataset, variable or interval not in dataset |
| RIHMIClient | `RIHMIError` | NSIDC download failure, unreadable export, unknown variable, interval, route type or source, daily data without an export |

All exceptions are subclasses of `Exception` with descriptive messages.

```python
from easysnowdata.stations.clients.awdb import AWDBClient, AWDBError
from easysnowdata.stations.clients.cdec import CDECClient, CDECError
from easysnowdata.stations.clients.databc import DataBCClient, DataBCError
from easysnowdata.stations.clients.nve import NVEClient, NVEError
from easysnowdata.stations.clients.yukon import YukonClient, YukonError
from easysnowdata.stations.clients.ghcnd import GHCNDClient, GHCNDError
# ... and likewise CanSWEError, NorSWEError, HydroQuebecError, AlbertaError,
# DWDError, SYKEError, SLFError and RIHMIError, all also re-exported from
# easysnowdata.stations.clients.

try:
    data = AWDBClient().get_data(station_ids=["303:CO:SNTL"], variables=["swe"])
except AWDBError as e:
    print(f"AWDB error: {e}")
```

HTTP 400/404 errors are not retried.  HTTP 5xx and network errors are retried
with linear backoff up to `max_retries` attempts.

---

## 18. Adding a New Client

The full normative contract and checklist live in
[DESIGN.md](https://github.com/egagli/global_snow_networks/blob/main/DESIGN.md) §3 and the
[new-client issue template](https://github.com/egagli/global_snow_networks/issues/new?template=new_client.md),
both in `global_snow_networks`: a new network needs code in both
repositories. In brief:

**In this repository (easysnowdata), first:**

1. Create `easysnowdata/stations/clients/{source}/` with `__init__.py` and
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
7. Add to `easysnowdata/stations/clients/__init__.py`.
8. Document the client in this README.

**Then in [`global_snow_networks`](https://github.com/egagli/global_snow_networks), once a release carries the client:**

9. Add to `scripts/create_all_stations_geojson.py`:
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
       `interval` uses the shared enum from
       `easysnowdata.stations.clients._common`.
     - the daily candidate flags — set via `_daily_candidate_props(data_vars)`,
       which derives `has_daily_swe` / `has_daily_snwd` (advertised) and
       initializes `daily_or_better` / `daily_verified` / `daily_provenance`
       for the probe in `get_all_stations_data.py` to verify (DESIGN.md §4).
10. Add to `scripts/get_all_stations_data.py` with a `refresh_{source}()`.
11. Add the network to `scripts/generate_live_map.py`:
    - Add an entry in `NET_LABELS` (`"CODE": "Human Name"`)
    - Add an SVG shape entry in `NET_SHAPES`
    - Add a `case "CODE"` in `buildIcon()` to assign a Leaflet marker shape
    - The legend is built dynamically from `MAP_META.available_networks`,
      so no further changes to the legend HTML are needed.
12. Update that repository's `README.md`: add to the Networks section and
    update the comparison table.

**Key invariants:**

- All `get_data()` methods return a **flat** list of observation records with
  keys `station_id`, `date`, `variable`, `type`, `value`, `units`, `interval`
  (plus `datetime` on sub-daily records and optional `flag`), so pipeline
  scripts can route data uniformly.
- `station_image_url` must be a direct URL that can be used in an `<img src>`
  tag without authentication.
- `network_code` in the GeoJSON must match a `NET_LABELS` key in the live map.
