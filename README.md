# Clients

This folder contains API client modules for retrieving snow observation data from external data sources. Each client is responsible for one data source and exposes a consistent interface for fetching stations, metadata, and time-series data.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [AWDBClient — USDA NRCS AWDB REST API](#2-awdbclient)
   - [Constructor](#21-constructor)
   - [get_reference_data](#22-get_reference_data)
   - [get_stations](#23-get_stations)
   - [get_metadata](#24-get_metadata)
   - [get_data](#25-get_data)
   - [get_data_by_water_year](#26-get_data_by_water_year)
   - [get_forecasts](#27-get_forecasts)
   - [get_normals](#28-get_normals)
   - [get_reservoir_metadata](#29-get_reservoir_metadata)
3. [Error Handling](#3-error-handling)
4. [API Notes and Gotchas](#4-api-notes-and-gotchas)
5. [Adding a New Client](#5-adding-a-new-client)

---

## 1. Design Philosophy

- **One client per data source.** Each client encapsulates the full lifecycle of requests to its source: authentication (if needed), URL construction, batching, retry, and response parsing.
- **Return plain Python objects.** Methods return `dict` / `list[dict]` so callers decide how to convert to pandas or xarray. No opinionated output schema.
- **Metric-first normalization.** Known imperial AWDB snow variables are converted in-client so downstream code stays unit-consistent; specifically `WTEQ` and `SNWD` are returned in centimeters.
- **Handle limits internally.** API rate limits, value-count limits, and URL length limits are managed by the client. Callers just ask for data.
- **Fail clearly.** All errors raise `{Client}Error` subclasses of `Exception` with descriptive messages.

---

## 2. AWDBClient

**Module:** `clients.awdb_client`  
**Class:** `AWDBClient`  
**API:** [AWDB REST API v1](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html)  
**Demo:** [nrcs-nwcc/iow_awdb_rest_api_demo](https://github.com/nrcs-nwcc/iow_awdb_rest_api_demo)

The AWDB (Air and Water Database) REST API provides access to data from SNOTEL, SNOLite, SCAN, COOP, Manual SNOTEL, snow courses, streamflow gauges, reservoirs, and more. It supports all elements (snow, precipitation, temperature, soil moisture, streamflow, etc.) at all available temporal resolutions.

```python
from global_snow_point_obs.clients import AWDBClient
client = AWDBClient()
```

---

### 2.1 Constructor

```python
AWDBClient(
    base_url: str = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1",
    timeout: int = 180,
    max_retries: int = 3,
    backoff: int = 6,
    session: requests.Session | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | AWDB v1 URL | Override for staging or mirror endpoints |
| `timeout` | `int` | `180` | Per-request HTTP timeout in seconds |
| `max_retries` | `int` | `3` | Retry attempts on 5xx server errors |
| `backoff` | `int` | `6` | Base backoff delay; actual delay = `backoff × attempt` |
| `session` | `requests.Session` | `None` | Pre-configured session (for custom headers, proxies, etc.) |

---

### 2.2 `get_reference_data`

```python
get_reference_data(tables: list[str] | str = "all") -> dict[str, Any]
```

Fetch AWDB lookup/reference tables. Useful for converting network codes, element codes, and unit codes to human-readable names.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `tables` | `list[str]` or `"all"` | Table name(s): `"networks"`, `"elements"`, `"states"`, `"counties"`, `"durations"`, `"units"`, or `"all"` |

**Returns:** `dict` — keys are table names, values are `list[dict]` of `{code, name, description}`.

**Example**

```python
ref = client.get_reference_data(["networks", "elements"])

# Build a lookup dict: code → name
network_names = {n["code"]: n["name"] for n in ref["networks"]}
# → {"SNTL": "SNOTEL", "SNTLT": "SNOLite", "SCAN": "SCAN", ...}

element_names = {e["code"]: e["name"] for e in ref["elements"]}
# → {"WTEQ": "Snow Water Equivalent", "SNWD": "Snow Depth", ...}
```

---

### 2.3 `get_stations`

```python
get_stations(
    networks: list[str] | str | None = None,
    states: list[str] | str | None = None,
    huc: str | None = None,
    county_name: str | None = None,
    active_only: bool = False,
    station_triplets: list[str] | str | None = None,
) -> list[dict]
```

List stations matching the given filters. Returns basic identification fields only (no element inventory). Use `get_metadata` for the full element list.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `networks` | `list[str]` | Network code(s), e.g. `["SNTL", "SNTLT"]`. Defaults to all networks. |
| `states` | `list[str]` | Two-letter state code(s), e.g. `["CO", "UT", "WY"]` |
| `huc` | `str` | HUC prefix (2–12 digits). Returns stations whose HUC starts with this prefix. |
| `county_name` | `str` | County name (prefix match, case-insensitive) |
| `active_only` | `bool` | If `True`, exclude retired stations |
| `station_triplets` | `list[str]` | Explicit triplet list (overrides network/state filters) |

**Returns:** `list[dict]` — one dict per station with keys: `stationTriplet`, `stationId`, `networkCode`, `name`, `stateCode`, `countyName`, `huc`, `latitude`, `longitude`, `elevation`, `beginDate`, `endDate`.

**Station triplet format:** `{stationId}:{stateCode}:{networkCode}`, e.g. `303:CO:SNTL`.

**Examples**

```python
# All SNOTEL and SNOLite stations in Colorado
stations = client.get_stations(networks=["SNTL", "SNTLT"], states=["CO"])

# All stations in a HUC8 (Roaring Fork)
stations = client.get_stations(huc="14010004")

# Mix of networks
stations = client.get_stations(networks=["SNTL", "SNTLT", "MSNT", "SCAN", "COOP"])

# Specific stations by triplet
stations = client.get_stations(station_triplets=["303:CO:SNTL", "713:CO:SNTL"])
```

---

### 2.4 `get_metadata`

```python
get_metadata(
    triplets: list[str] | str,
    elements: list[str] | str = "*",
    durations: list[str] | str = "*",
    include_forecast_point: bool = True,
    include_reservoir: bool = True,
    active_only: bool = False,
) -> list[dict]
```

Retrieve full station metadata including the element inventory (what's measured, at what resolution, and for what period of record).

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `triplets` | `list[str]` | Station triplet(s) |
| `elements` | `list[str]` or `"*"` | Filter to specific element codes, e.g. `["WTEQ", "SNWD"]` |
| `durations` | `list[str]` or `"*"` | Filter to specific durations: `"DAILY"`, `"HOURLY"`, `"MONTHLY"`, `"SEMIMONTHLY"`, `"ANNUAL"`, `"WATER_YEAR"` |
| `include_forecast_point` | `bool` | Include forecast point metadata block |
| `include_reservoir` | `bool` | Include reservoir metadata block |
| `active_only` | `bool` | Exclude retired elements |

**Returns:** `list[dict]` — one dict per station (only stations with matching elements are returned). The `stationElements` key contains the element inventory:

```python
{
    "stationTriplet": "303:CO:SNTL",
    "name": "COPPER MOUNTAIN",
    "networkCode": "SNTL",
    "latitude": 39.4797,
    "longitude": -106.1597,
    "elevation": 10550,
    "huc": "14010001",
    "beginDate": "1978-10-01 00:00",
    "endDate": None,
    "stateCode": "CO",
    "countyName": "Summit",
    "stationElements": [
        {
            "elementCode": "WTEQ",
            "elementName": "Snow Water Equivalent",
            "durationName": "DAILY",
            "originalUnitCode": "in",
            "storedUnitCode": "in",
            "beginDate": "1978-10-01 00:00",
            "endDate": "2100-01-01 00:00",
            "dataPrecision": 1,
            "ordinal": 1,
        },
        ...
    ],
    "forecastPoint": { ... },   # if applicable
    "reservoir": { ... },        # if applicable
}
```

**Notes**
- Requests are automatically batched to 150 triplets per call (URL length limit).
- Only stations with at least one matching element are returned.

**Example**

```python
# All stations in CO with daily WTEQ or SNWD
stations = client.get_stations(networks=["SNTL"], states=["CO"])
triplets = [s["stationTriplet"] for s in stations]

meta = client.get_metadata(
    triplets=triplets,
    elements=["WTEQ", "SNWD"],
    durations=["DAILY"],
)

# Summarize periods of record
for s in meta:
    for el in s["stationElements"]:
        print(f"{s['name']:30s}  {el['elementCode']}  {el['beginDate'][:10]}")
```

---

### 2.5 `get_data`

```python
get_data(
    triplets: list[str] | str,
    elements: list[str] | str,
    duration: str = "DAILY",
    begin_date: str | date | None = None,
    end_date: str | date | None = None,
    period_ref: str = "START",
    central_tendency_type: str | None = None,
) -> list[dict]
```

The primary data retrieval method. Fetches time-series observations for any combination of stations, elements, and date range.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `triplets` | `list[str]` | Station triplet(s) |
| `elements` | `list[str]` | Element code(s), e.g. `["WTEQ", "SNWD", "TOBS", "PRCP"]` |
| `duration` | `str` | `"DAILY"`, `"HOURLY"`, `"MONTHLY"`, `"SEMIMONTHLY"`, `"ANNUAL"`, `"WATER_YEAR"` |
| `begin_date` | `str` or `date` | Start date (`"YYYY-MM-DD"`). Defaults to earliest available. |
| `end_date` | `str` or `date` | End date (inclusive). Defaults to today. |
| `period_ref` | `str` | Date alignment: `"START"` (date is start of period) or `"END"` |
| `central_tendency_type` | `str` | Include normals: `"MEDIAN"`, `"AVERAGE"`, or `None` |

**Returns:** `list[dict]` — one dict per station:

```python
{
    "stationTriplet": "303:CO:SNTL",
    "data": [
        {
            "stationElement": {
                "elementCode": "WTEQ",
                "durationName": "DAILY",
                "originalUnitCode": "in",
                ...
            },
            "values": [
                {"date": "2023-10-01", "value": 0.0},
                {"date": "2023-10-02", "value": 0.0},
                ...
                # If central_tendency_type provided, each record also has "median" or "average"
                {"date": "2023-10-01", "value": 0.0, "median": 1.2},
            ]
        },
        # one block per element
    ]
}
```

**API value limit and auto-batching**

The AWDB API rejects requests where `n_stations × n_elements × n_days > 500,000`. The client computes the maximum safe batch size automatically and splits the station list accordingly. For long time series with multiple elements, the effective batch size may be as small as 2–5 stations.

```
# Example: 2 elements × 47,000 days ≈ 94,000 values per station
# → max batch size = floor(450,000 / 94,000) = 4 stations per request
```

**Examples**

```python
# Daily SWE + snow depth for WY2024
data = client.get_data(
    triplets=["303:CO:SNTL", "713:CO:SNTL"],
    elements=["WTEQ", "SNWD"],
    begin_date="2023-10-01",
    end_date="2024-09-30",
)

# Hourly SWE for a single station
data = client.get_data(
    triplets="303:CO:SNTL",
    elements=["WTEQ"],
    duration="HOURLY",
    begin_date="2024-01-01",
    end_date="2024-03-31",
)

# Monthly reservoir storage with normals
data = client.get_data(
    triplets=["JAK:CO:BOR"],
    elements=["RESC"],
    duration="MONTHLY",
    begin_date="2020-10-01",
    end_date="2024-09-30",
    central_tendency_type="MEDIAN",
)

# Full period of record for a single station
data = client.get_data(
    triplets="303:CO:SNTL",
    elements=["WTEQ"],
    # begin_date / end_date omitted → all available data
)
```

---

### 2.6 `get_data_by_water_year`

```python
get_data_by_water_year(
    triplets: list[str] | str,
    elements: list[str] | str,
    water_year: int,
    duration: str = "DAILY",
    **kwargs,
) -> list[dict]
```

Convenience wrapper for `get_data` that accepts a water year integer instead of explicit begin/end dates.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `triplets` | `list[str]` | Station triplet(s) |
| `elements` | `list[str]` | Element code(s) |
| `water_year` | `int` | Water year, e.g. `2024` for Oct 1 2023 – Sep 30 2024 |
| `duration` | `str` | Temporal resolution |
| `**kwargs` | | Passed to `get_data` |

**Example**

```python
data = client.get_data_by_water_year(
    triplets=["303:CO:SNTL", "713:CO:SNTL"],
    elements=["WTEQ", "SNWD"],
    water_year=2024,
)
```

---

### 2.7 `get_forecasts`

```python
get_forecasts(
    triplets: list[str] | str,
    elements: list[str] | str = "SRVO",
    begin_publication_date: str | date | None = None,
    end_publication_date: str | date | None = None,
) -> list[dict]
```

Retrieve seasonal streamflow or SWE volume forecasts with exceedance probabilities.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `triplets` | `list[str]` | Forecast point triplet(s) (typically USGS gauge triplets) |
| `elements` | `list[str]` | Forecast element: `"SRVO"` (natural flow, kaf) is most common |
| `begin_publication_date` | `str` | Start of the range of publication dates to retrieve |
| `end_publication_date` | `str` | End of the range of publication dates |

**Returns:** `list[dict]` — one dict per station, with `data` containing a list of forecasts each with `publicationDate`, `forecastPeriod` (start, end), and `forecastValues` (dict of exceedance % → value).

**Example**

```python
# Roaring Fork River at Glenwood Springs — WY2024 forecasts
fcst = client.get_forecasts(
    "09085000:CO:USGS",
    begin_publication_date="2024-01-01",
    end_publication_date="2024-07-31",
)

for f in fcst[0]["data"]:
    period = f["forecastPeriod"]
    pub    = f["publicationDate"]
    p50    = f["forecastValues"].get("50", "N/A")
    print(f"{pub}  {period[0]}–{period[1]}  50%: {p50} kaf")
```

---

### 2.8 `get_normals`

```python
get_normals(
    triplets: list[str] | str,
    elements: list[str] | str,
    duration: str = "DAILY",
    normal_period: str = "1991-2020",
    central_tendency_type: str = "MEDIAN",
) -> list[dict]
```

Retrieve climatological normals (1991–2020 or other reference period) for stations. Returns the same structure as `get_data` but with `median` or `average` fields populated in the values list.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `triplets` | `list[str]` | Station triplet(s) |
| `elements` | `list[str]` | Element code(s) |
| `duration` | `str` | `"DAILY"`, `"MONTHLY"`, etc. |
| `normal_period` | `str` | `"1991-2020"`, `"1981-2010"`, or `"1971-2000"` |
| `central_tendency_type` | `str` | `"MEDIAN"` or `"AVERAGE"` |

**Example**

```python
# 1991–2020 daily SWE median for SNOTEL 303
norms = client.get_normals(
    "303:CO:SNTL", ["WTEQ"], normal_period="1991-2020"
)
# Each value dict: {"date": "...", "value": <observed>, "median": <normal>}
```

---

### 2.9 `get_reservoir_metadata`

```python
get_reservoir_metadata(triplets: list[str] | str) -> list[dict]
```

Convenience wrapper for `get_metadata` that filters to reservoir stations and populates the `reservoir` metadata block (capacity, active storage, dead pool, etc.).

**Example**

```python
# BOR reservoir stations in a HUC
res_stations = client.get_stations(networks=["BOR"], huc="14010004")
triplets = [s["stationTriplet"] for s in res_stations]
res_meta = client.get_reservoir_metadata(triplets)

for s in res_meta:
    r = s.get("reservoir", {})
    print(f"{s['name']:30s}  capacity={r.get('reservoirCapacity')} ac-ft")
```

---

## 3. Error Handling

All methods raise `AWDBError` on failure:

| Scenario | Behavior |
|---|---|
| HTTP 400 (bad request / value limit exceeded) | Raises `AWDBError` immediately (not retried) |
| HTTP 404 | Raises `AWDBError` immediately |
| HTTP 5xx (server error) | Retried up to `max_retries` times with exponential backoff, then raises `AWDBError` |
| Network timeout / connection error | Same retry behavior as 5xx |

```python
from global_snow_point_obs.clients.awdb_client import AWDBClient, AWDBError

client = AWDBClient()
try:
    data = client.get_data(["303:CO:SNTL"], ["WTEQ"])
except AWDBError as e:
    print(f"AWDB request failed: {e}")
```

---

## 4. API Notes and Gotchas

**Value limit per request**  
The AWDB REST API enforces a hard limit of 500,000 data values per `/data` request:
`n_stations × n_elements × n_days ≤ 500,000`. The client handles this automatically, but it means long time series (full period of record, 2+ elements) require many small batches. For 2 elements and a 130-year time axis, the maximum batch size is **4–5 stations**.

**Triplet format**  
All station references use the AWDB triplet format: `{stationId}:{stateCode}:{networkCode}`. This is case-sensitive. Example: `303:CO:SNTL`.

**Date format**  
All dates are `YYYY-MM-DD` strings. The API also returns timestamps with time components (`"1978-10-01 00:00"`); the client returns these as-is.

**Null values**  
Missing observations are returned as `null` in JSON (Python `None` in the parsed response). The `value` field in each record may be `None`. Check before converting to float.

**Water year convention**  
AWDB uses the standard water year: October 1 through September 30. WY2024 = 2023-10-01 through 2024-09-30.

**Active vs. retired stations**  
`get_stations(active_only=False)` (the default) returns all stations, including retired ones. Retired stations have a non-null `endDate`. This is important for long time series where you want historical data from stations that are no longer operational.

**BC/AB/YK stations**  
Some Canadian stations (British Columbia, Alberta, Yukon) are in AWDB under `BC`, `AB`, and `YK` state codes. The client handles these identically to U.S. stations.

---

## 5. Adding a New Client

To add support for a new data source (e.g., CDEC, GHCND, Environment Canada):

1. Create `clients/{source}_client.py` with a `{Source}Client` class.
2. Follow the same method signature patterns as `AWDBClient`.
3. Raise `{Source}Error(Exception)` for errors.
4. Export the class from `clients/__init__.py`.
5. Document the client in this README with the same section structure.

The key invariant: all `get_data()` methods should return a list of station dicts, each with a `data` field containing time-series blocks, so that the Zarr pipeline scripts can treat all clients uniformly.
