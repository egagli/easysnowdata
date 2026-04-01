# -*- coding: utf-8 -*-
"""
clients/databc/databc_client.py
================================
Python client for BC snow survey data via the BC Data Catalogue and the
BC Ministry of Environment data portal.

Data sources
------------
1. **BC OpenMaps WFS** — station locations and metadata (GeoJSON via WFS).
     - Automated Snow Weather Stations (ASWS):
       layer ``WHSE_WATER_MANAGEMENT.SSL_SNOW_ASWS_STNS_SP``
       (~146 active + inactive automated snow pillow stations).
     - Manual Snow Survey Sites (MSS):
       layer ``WHSE_WATER_MANAGEMENT.SSL_SNOW_MSS_LOCS_SP``
       (~390 manual snow course sites, active + inactive).

2. **BC env.gov.bc.ca CSV files** — time-series observations.
     All ASWS CSV files share the same wide format:
       - Rows: one per UTC timestamp (typically hourly)
       - Columns: one per station, header = ``"{ID} {Name}"``
       - The 16:00 UTC reading is used as the daily canonical value
         (~08:00 PST / 09:00 PDT).
       - Negative values and the -99999 sentinel are normalised to NaN.

     Available CSV files (``https://www.env.gov.bc.ca/wsd/data_searches/snow/asws/data/``):

     +-----------+-------------------------------------+---------+---------+
     | File      | Variable                            | Units   | Archive |
     +===========+=====================================+=========+=========+
     | SWDaily   | Snow Water Equivalent (daily avg)   | mm      | Yes     |
     | SW        | Snow Water Equivalent (hourly)      | mm      | Yes     |
     | SD        | Snow Depth                          | cm      | Yes     |
     | TA        | Air Temperature                     | °C      | Yes     |
     | PC        | Precipitation Cumulative            | mm      | Yes     |
     | PA        | Barometric Pressure                 | hPa     | No      |
     | UD        | Wind Direction                      | degrees | No      |
     | US        | Wind Speed                          | km/h    | No      |
     | UP        | Wind Speed Peak (gust)              | km/h    | No      |
     | UR        | Wind Run (cumulative)               | km      | No      |
     | XR        | Relative Humidity                   | %       | No      |
     +-----------+-------------------------------------+---------+---------+

     - MSS periodic surveys: ``allmss_current.csv`` / ``allmss_archive.csv``
       Long format: one row per station × survey date.
       Columns include Snow Depth (cm) and Water Equiv. (mm SWE).

3. **AQRT BCMOE portal** — station photos.
     ``https://bcmoe-prod.aquaticinformatics.net``
     A public (disclaimer-gated) AQUARIUS Web Portal instance.
     Station photos are accessible via the station summary page after
     accepting the disclaimer.  Use ``get_station_image_url()`` to retrieve
     a direct image URL for a given station.

Station ID conventions
----------------------
- ASWS station location IDs end in ``P`` (e.g. ``1A01P``, ``1E08P``).
- MSS location IDs do not end in ``P`` (e.g. ``1A06A``, ``1A10``).

Station URLs
------------
Each station has a page on the BC AQRT (Aquarius Report Tool) portal:
  ``https://aqrt.nrs.gov.bc.ca/Data/Location/Summary/Location/{id}/Interval/Latest``

Station image URL
-----------------
A second AQRT instance at ``bcmoe-prod.aquaticinformatics.net`` hosts station
photos.  ``DataBCClient.get_station_image_url(location_id)`` scrapes this
portal (after accepting the disclaimer) and returns a direct ``GetFileById``
URL, or ``None`` if no photo is found.

Variables and units
-------------------
+--------------------+--------+-------------------------------------------+
| Variable           | Units  | Source                                    |
+====================+========+===========================================+
| swe_mm             | mm     | ASWS daily/hourly pillow; MSS survey      |
| snwd_cm            | cm     | ASWS daily (SD.csv); MSS periodic survey  |
| air_temp_degc      | °C     | ASWS daily (TA.csv)                       |
| precip_cumul_mm    | mm     | ASWS daily (PC.csv)                       |
| baro_press_hpa     | hPa    | ASWS (PA.csv, current season only)        |
| wind_dir_deg       | °      | ASWS (UD.csv, current season only)        |
| wind_spd_kmh       | km/h   | ASWS (US.csv, current season only)        |
| wind_spd_peak_kmh  | km/h   | ASWS (UP.csv, current season only)        |
| wind_run_km        | km     | ASWS (UR.csv, current season only)        |
| rh_pct             | %      | ASWS (XR.csv, current season only)        |
| density_pct        | %      | MSS periodic survey only                  |
| snow_line_m        | m      | MSS periodic survey only                  |
+--------------------+--------+-------------------------------------------+

For the per-station CSV archive, ASWS SWE (mm) is converted to cm (÷ 10)
for ``wteq_cm``.  MSS data is periodic and not stored in the daily CSV
archive.

Design principles
-----------------
- Returns plain Python objects (dicts / lists) or pandas DataFrames.
- Metric-first: all outputs are in SI units.
- Missing / sentinel values (NaN, negative, -99999) are normalised to
  ``None``/NaN.
- Data flags in MSS survey: the ``Survey Code`` field (e.g. ``"PROBLEM"``)
  acts as a data flag.  Include ``include_flags=True`` to retain it.
- ``daily_only=True`` (default on all ASWS methods): returns one row per
  station per calendar date using the 16:00 UTC reading.
- ``daily_only=False``: returns all hourly rows with a ``datetime`` column
  (UTC string, format ``"YYYY-MM-DD HH:MM"``).
"""

from __future__ import annotations

import io
import logging
import re
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

AQRT_BASE = "https://aqrt.nrs.gov.bc.ca"
AQRT_BCMOE_BASE = "https://bcmoe-prod.aquaticinformatics.net"
WFS_BASE = "https://openmaps.gov.bc.ca/geo/pub"
DATA_BASE = "https://www.env.gov.bc.ca/wsd/data_searches/snow/asws/data"

ASWS_LAYER = "WHSE_WATER_MANAGEMENT.SSL_SNOW_ASWS_STNS_SP"
MSS_LAYER = "WHSE_WATER_MANAGEMENT.SSL_SNOW_MSS_LOCS_SP"

# Per-station combined CSV archive (SW + SD + TA + PC for each station)
SNOW_ALL_BASE = f"{DATA_BASE}/SnowAll"

_DEFAULT_TIMEOUT = 120
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 5

# The 16:00 UTC daily reading time used as the canonical daily value
_DAILY_UTC_HOUR = "16:00"


# ── Public variable / flag tables ─────────────────────────────────────────────

#: Variables available from this client.
VARIABLES: dict[str, dict] = {
    "swe_mm": {
        "name": "Snow Water Equivalent",
        "type": "swe",
        "units": "mm",
        "source": "ASWS (daily: SWDaily.csv; hourly: SW.csv) and MSS (periodic survey)",
        "description": (
            "ASWS: automated snow pillow reading. "
            "Daily file uses 16:00 UTC as canonical value. "
            "MSS: manually surveyed water equivalent."
        ),
        "notes": "",
    },
    "snwd_cm": {
        "name": "Snow Depth",
        "type": "snwd",
        "units": "cm",
        "source": "ASWS (SD.csv / SD_Archive.csv) and MSS (periodic survey)",
        "description": (
            "ASWS: automated snow depth sensor reading from SD.csv; "
            "16:00 UTC value used as daily canonical reading. "
            "MSS: manually measured snow depth from snow course surveys."
        ),
        "notes": "",
    },
    "air_temp_degc": {
        "name": "Air Temperature",
        "type": "temp",
        "units": "°C",
        "source": "ASWS (TA.csv / TA_Archive.csv)",
        "description": (
            "Hourly air temperature from ASWS stations. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "",
    },
    "precip_cumul_mm": {
        "name": "Precipitation Cumulative",
        "type": "precip",
        "units": "mm",
        "source": "ASWS (PC.csv / PC_Archive.csv)",
        "description": (
            "Cumulative precipitation accumulation from ASWS stations. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "",
    },
    "baro_press_hpa": {
        "name": "Barometric Pressure",
        "type": "baro",
        "units": "hPa",
        "source": "ASWS (PA.csv, current season only — no archive)",
        "description": (
            "Station-level barometric pressure. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "wind_dir_deg": {
        "name": "Wind Direction",
        "type": "wind_dir",
        "units": "degrees",
        "source": "ASWS (UD.csv, current season only — no archive)",
        "description": (
            "Wind direction in degrees from north (0–360). "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "wind_spd_kmh": {
        "name": "Wind Speed",
        "type": "wind_spd",
        "units": "km/h",
        "source": "ASWS (US.csv, current season only — no archive)",
        "description": (
            "Average wind speed. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "wind_spd_peak_kmh": {
        "name": "Wind Speed Peak (Gust)",
        "type": "wind_gust",
        "units": "km/h",
        "source": "ASWS (UP.csv, current season only — no archive)",
        "description": (
            "Peak (gust) wind speed. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "wind_run_km": {
        "name": "Wind Run",
        "type": "wind_run",
        "units": "km",
        "source": "ASWS (UR.csv, current season only — no archive)",
        "description": (
            "Cumulative wind run (distance travelled by wind). "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "rh_pct": {
        "name": "Relative Humidity",
        "type": "rh",
        "units": "%",
        "source": "ASWS (XR.csv, current season only — no archive)",
        "description": (
            "Relative humidity as a percentage. "
            "16:00 UTC reading used as daily canonical value."
        ),
        "notes": "No historical archive available.",
    },
    "density_pct": {
        "name": "Snow Density",
        "type": "density",
        "units": "%",
        "source": "MSS (periodic survey only)",
        "description": "Snow density calculated from depth and SWE.",
        "notes": "",
    },
    "snow_line_m": {
        "name": "Snow Line Elevation",
        "type": "snow_line",
        "units": "m",
        "source": "MSS (periodic survey only)",
        "description": "Elevation of the snow line at time of survey.",
        "notes": "",
    },
}

# Standardized type → DataBC variable key(s)
_TYPE_TO_DATABC_VARS: dict[str, list[str]] = {
    "swe":        ["swe_mm"],
    "snwd":       ["snwd_cm"],
    "temp":       ["air_temp_degc"],
    "precip":     ["precip_cumul_mm"],
    "baro":       ["baro_press_hpa"],
    "wind_dir":   ["wind_dir_deg"],
    "wind_spd":   ["wind_spd_kmh"],
    "wind_gust":  ["wind_spd_peak_kmh"],
    "wind_run":   ["wind_run_km"],
    "rh":         ["rh_pct"],
    "density":    ["density_pct"],
    "snow_line":  ["snow_line_m"],
}

#: MSS data quality flags (``Survey Code`` field values).
DATA_FLAGS: dict[str, str] = {
    "": "No flag (normal data quality)",
    "PROBLEM": "Data quality problem noted by surveyor",
    "ESTIMATE": "Estimated value",
    "EXTRAPOLATED": "Extrapolated from nearby site",
}


# ── Client ───────────────────────────────────────────────────────────────────

class DataBCClient:
    """
    Client for BC snow survey data via BC Data Catalogue.

    Provides access to both Automated Snow Weather Station (ASWS) data and
    Manual Snow Survey (MSS) data through publicly available BC government
    endpoints.  Also provides access to ASWS station photos via the AQRT
    BCMOE portal.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Retry attempts on transient server errors.
    backoff : int
        Base backoff delay in seconds.
    session : requests.Session or None
        Optional pre-configured session for data/WFS requests.
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        backoff: int = _DEFAULT_BACKOFF,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "global-snow-networks/1.0"})
        # Separate session for the AQRT BCMOE portal (disclaimer-gated)
        self._aqrt_session = requests.Session()
        self._aqrt_session.headers.update({"User-Agent": "global-snow-networks/1.0"})
        self._aqrt_disclaimer_accepted = False

    # ── Public API — station lists ────────────────────────────────────────────

    def get_asws_stations(self, active_only: bool = False) -> list[dict]:
        """
        Get Automated Snow Weather Station (ASWS) locations from WFS.

        These are automated snow pillow stations that report daily SWE.
        Location IDs end in ``P`` (e.g. ``1A01P``).

        Parameters
        ----------
        active_only : bool
            If True, filter to Active stations only.

        Returns
        -------
        list[dict]
            One dict per station with keys: ``location_id``, ``name``,
            ``elevation_m``, ``latitude``, ``longitude``, ``status``,
            ``operator``, ``camera_url``, ``station_type``,
            ``station_url``.
        """
        return self._get_wfs_stations(
            ASWS_LAYER,
            station_type="ASWS",
            active_only=active_only,
        )

    def get_mss_stations(self, active_only: bool = False) -> list[dict]:
        """
        Get Manual Snow Survey (MSS) site locations from WFS.

        These are periodic manual snow course measurement sites.
        Location IDs do NOT end in ``P`` (e.g. ``1A06A``, ``1A10``).

        Parameters
        ----------
        active_only : bool
            If True, filter to Active sites only.

        Returns
        -------
        list[dict]
            One dict per site with keys: ``location_id``, ``name``,
            ``elevation_m``, ``latitude``, ``longitude``, ``status``,
            ``station_type``, ``station_url``.
        """
        return self._get_wfs_stations(
            MSS_LAYER,
            station_type="MSS",
            active_only=active_only,
        )

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """
        Get both ASWS and MSS stations.

        Parameters
        ----------
        active_only : bool
            If True, filter to Active stations/sites only.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` bounding box filter.

        Returns
        -------
        list[dict]
            Combined list of ASWS and MSS stations.
        """
        stations = (
            self.get_asws_stations(active_only=active_only)
            + self.get_mss_stations(active_only=active_only)
        )
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            stations = [
                s for s in stations
                if s.get("latitude") is not None
                and s.get("longitude") is not None
                and min_lat <= float(s["latitude"]) <= max_lat
                and min_lon <= float(s["longitude"]) <= max_lon
            ]
        return stations

    def get_data(
        self,
        station_ids: list[str] | str | None = None,
        variables: list[str] | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        include_flags: bool = False,
    ) -> list[dict]:
        """
        Standardized data fetch — returns a flat list of observation records.

        For ASWS stations, daily SWE is sourced from ``SWDaily.csv``
        (16:00 UTC reading).  For MSS stations, all survey variables
        are returned with ``interval="periodic"``.

        Parameters
        ----------
        station_ids : list[str] or str or None
            Location ID(s), e.g. ``"1A01P"`` (ASWS) or ``"1A01"`` (MSS).
            Required unless ``bbox`` is provided.
        variables : list[str] or str or None
            DataBC variable keys (e.g. ``"swe_mm"``) **or** standardized
            types (e.g. ``"swe"``).  ``None`` returns all variables in
            :data:`VARIABLES` available for the given station type.
        bbox : tuple, optional
            ``(min_lon, min_lat, max_lon, max_lat)``.
        begin_date, end_date : str or None
            Date range (``"YYYY-MM-DD"``).
        interval : str
            ``"daily"`` fetches ASWS data; ``"periodic"`` fetches MSS
            survey data; ``"hourly"`` fetches hourly ASWS sensor data.
        include_flags : bool
            If True, MSS survey codes are included as ``"flag"`` field.

        Returns
        -------
        list[dict]
            Flat list of observation records::

                {
                    "station_id": "1A01P",
                    "date": "2024-01-15",
                    "variable": "swe_mm",
                    "type": "swe",
                    "value": 45.0,   # mm ÷ 10 → cm for swe_mm
                    "units": "cm",
                    "interval": "daily",
                    # "flag": None  (only when include_flags=True)
                }

        Notes
        -----
        ``swe_mm`` values are converted to cm (÷ 10) in this method so
        that the returned ``"units"`` is always ``"cm"`` for type ``"swe"``.

        Raises
        ------
        ValueError
            If neither ``station_ids`` nor ``bbox`` is provided.
        DataBCError
            On network / data-loading failure.
        """
        if station_ids is None and bbox is not None:
            ids: list[str] | None = [
                s["location_id"]
                for s in self.get_all_stations(bbox=bbox)
            ]
        elif station_ids is not None:
            ids = (
                [station_ids]
                if isinstance(station_ids, str)
                else list(station_ids)
            )
        else:
            raise ValueError("Provide station_ids or bbox.")

        # Resolve variables list
        var_list: list[str] | None = None
        if variables is not None:
            raw_vars = (
                [variables]
                if isinstance(variables, str)
                else list(variables)
            )
            resolved: list[str] = []
            for v in raw_vars:
                if v in VARIABLES:
                    resolved.append(v)
                elif v in _TYPE_TO_DATABC_VARS:
                    resolved.extend(_TYPE_TO_DATABC_VARS[v])
            var_list = resolved or None

        records: list[dict] = []

        if interval in ("daily", "sub_daily"):
            # ASWS stations — filter to IDs ending in 'P'
            asws_ids = (
                [i for i in ids if str(i).upper().endswith("P")]
                if ids is not None else None
            )
            if asws_ids is None or asws_ids:
                df = self.get_asws_combined_data(
                    location_ids=asws_ids or None,
                    begin_date=begin_date,
                    end_date=end_date,
                    archive=True,
                )
                # Determine which variables to emit
                emit_vars = var_list or [
                    k for k, v in VARIABLES.items()
                    if "ASWS" in v.get("source", "")
                ]
                col_map = {
                    "swe_mm": ("swe", "cm", lambda x: round(x / 10.0, 3)),
                    "snwd_cm": ("snwd", "cm", lambda x: x),
                    "air_temp_degc": ("temp", "°C", lambda x: x),
                    "precip_cumul_mm": (
                        "precip", "mm", lambda x: x
                    ),
                    "baro_press_hpa": ("baro", "hPa", lambda x: x),
                    "wind_dir_deg": (
                        "wind_dir", "degrees", lambda x: x
                    ),
                    "wind_spd_kmh": (
                        "wind_spd", "km/h", lambda x: x
                    ),
                    "wind_spd_peak_kmh": (
                        "wind_gust", "km/h", lambda x: x
                    ),
                    "wind_run_km": (
                        "wind_run", "km", lambda x: x
                    ),
                    "rh_pct": ("rh", "%", lambda x: x),
                }
                for var_key in emit_vars:
                    if var_key not in col_map:
                        continue
                    col_name = (
                        var_key if var_key in df.columns else None
                    )
                    if col_name is None:
                        continue
                    std_type, units, converter = col_map[var_key]
                    for _, row in df.iterrows():
                        raw_val = row.get(col_name)
                        import math
                        if (
                            raw_val is None
                            or (
                                isinstance(raw_val, float)
                                and math.isnan(raw_val)
                            )
                        ):
                            value = None
                        else:
                            try:
                                value = converter(float(raw_val))
                            except (TypeError, ValueError):
                                value = None
                        r: dict = {
                            "station_id": str(row.get(
                                "location_id", ""
                            )),
                            "date": str(row.get("date", ""))[:10],
                            "variable": var_key,
                            "type": std_type,
                            "value": value,
                            "units": units,
                            "interval": "daily",
                        }
                        if include_flags:
                            r["flag"] = None
                        records.append(r)

        if interval in ("periodic", "monthly") or (
            interval == "daily"
            and ids is not None
            and any(not str(i).upper().endswith("P") for i in ids)
        ):
            # MSS stations
            mss_ids = (
                [i for i in ids if not str(i).upper().endswith("P")]
                if ids is not None else None
            )
            if mss_ids is None or mss_ids:
                df_mss = self.get_mss_survey_data(
                    location_ids=mss_ids or None,
                    begin_date=begin_date,
                    end_date=end_date,
                    archive=True,
                    include_flags=include_flags,
                )
                mss_col_map = {
                    "swe_mm": ("swe", "cm",
                               lambda x: round(x / 10.0, 3)),
                    "snwd_cm": ("snwd", "cm", lambda x: x),
                    "density_pct": (
                        "density", "%", lambda x: x
                    ),
                    "snow_line_m": (
                        "snow_line", "m", lambda x: x
                    ),
                }
                emit_mss = var_list or list(mss_col_map.keys())
                for _, row in df_mss.iterrows():
                    for var_key in emit_mss:
                        if var_key not in mss_col_map:
                            continue
                        if var_key not in df_mss.columns:
                            continue
                        std_type, units, converter = (
                            mss_col_map[var_key]
                        )
                        raw_val = row.get(var_key)
                        import math
                        if (
                            raw_val is None
                            or (
                                isinstance(raw_val, float)
                                and math.isnan(raw_val)
                            )
                        ):
                            value = None
                        else:
                            try:
                                value = converter(float(raw_val))
                            except (TypeError, ValueError):
                                value = None
                        r = {
                            "station_id": str(
                                row.get("location_id", "")
                            ),
                            "date": str(row.get("date", ""))[:10],
                            "variable": var_key,
                            "type": std_type,
                            "value": value,
                            "units": units,
                            "interval": "periodic",
                        }
                        if include_flags:
                            r["flag"] = row.get(
                                "survey_code", None
                            )
                        records.append(r)

        return records

    # ── Public API — ASWS time-series data ───────────────────────────────────

    def get_asws_daily_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
        include_flags: bool = False,
    ) -> pd.DataFrame:
        """
        Get daily SWE data from Automated Snow Weather Stations.

        Data is sourced from the pre-aggregated ``SWDaily.csv`` files.  The
        16:00 UTC reading is used as the canonical daily value (~08:00 PST /
        09:00 PDT).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs (e.g. ``["1A01P", "1E08P"]``).
            If None, all stations are returned.
        begin_date : str or None
            Start date (``"YYYY-MM-DD"``).
        end_date : str or None
            End date (inclusive, ``"YYYY-MM-DD"``).
        archive : bool
            If True, also load the historical archive CSV (larger file).
            Recommended for full period-of-record retrieval.
        include_flags : bool
            Reserved for future use; ASWS CSV data does not currently
            include per-value quality flags.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date`` (str ``"YYYY-MM-DD"``),
            ``location_id`` (str),
            ``swe_mm`` (float or NaN).
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/SWDaily.csv",
            archive_url=f"{DATA_BASE}/SW_DailyArchive.csv",
            value_col="swe_mm",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=True,
        )

    def get_asws_sw_hourly_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
    ) -> pd.DataFrame:
        """
        Get hourly SWE data from Automated Snow Weather Stations.

        Data is sourced from the raw hourly ``SW.csv`` files (distinct from
        ``SWDaily.csv`` which contains pre-aggregated daily values).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start datetime string (``"YYYY-MM-DD"`` or ``"YYYY-MM-DD HH:MM"``).
        end_date : str or None
            End datetime string (inclusive).
        archive : bool
            If True, also load the historical archive CSV.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``datetime`` (str ``"YYYY-MM-DD HH:MM"`` UTC),
            ``location_id`` (str),
            ``swe_mm`` (float or NaN).
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/SW.csv",
            archive_url=f"{DATA_BASE}/SW_Archive.csv",
            value_col="swe_mm",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=False,
        )

    def get_asws_sd_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get snow depth data from Automated Snow Weather Stations.

        Data is sourced from wide-format hourly CSV files (SD.csv /
        SD_Archive.csv).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs (e.g. ``["1A01P", "1E08P"]``).
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        archive : bool
            If True, also load the historical archive CSV.
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day
            with a ``date`` column.  If False, return all hourly readings
            with a ``datetime`` column.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.  Columns depend on ``daily_only``:
            - ``daily_only=True``: ``date``, ``location_id``, ``snwd_cm``
            - ``daily_only=False``: ``datetime``, ``location_id``, ``snwd_cm``
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/SD.csv",
            archive_url=f"{DATA_BASE}/SD_Archive.csv",
            value_col="snwd_cm",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_ta_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get air temperature data from Automated Snow Weather Stations.

        Data is sourced from TA.csv / TA_Archive.csv.

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        archive : bool
            If True, also load the historical archive CSV (~75 MB).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day
            with a ``date`` column.  If False, return all hourly readings
            with a ``datetime`` column.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``air_temp_degc``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/TA.csv",
            archive_url=f"{DATA_BASE}/TA_Archive.csv",
            value_col="air_temp_degc",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_pc_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get cumulative precipitation data from Automated Snow Weather Stations.

        Data is sourced from PC.csv / PC_Archive.csv.

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        archive : bool
            If True, also load the historical archive CSV (~63 MB).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings with a ``datetime`` column.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``precip_cumul_mm``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/PC.csv",
            archive_url=f"{DATA_BASE}/PC_Archive.csv",
            value_col="precip_cumul_mm",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_pa_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get barometric pressure data from Automated Snow Weather Stations.

        Data is sourced from PA.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``baro_press_hpa``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/PA.csv",
            archive_url=None,
            value_col="baro_press_hpa",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_ud_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get wind direction data from Automated Snow Weather Stations.

        Data is sourced from UD.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``wind_dir_deg``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/UD.csv",
            archive_url=None,
            value_col="wind_dir_deg",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_us_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get wind speed data from Automated Snow Weather Stations.

        Data is sourced from US.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``wind_spd_kmh``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/US.csv",
            archive_url=None,
            value_col="wind_spd_kmh",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_up_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get peak (gust) wind speed data from Automated Snow Weather Stations.

        Data is sourced from UP.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``wind_spd_peak_kmh``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/UP.csv",
            archive_url=None,
            value_col="wind_spd_peak_kmh",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_ur_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get wind run (cumulative) data from Automated Snow Weather Stations.

        Data is sourced from UR.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``wind_run_km``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/UR.csv",
            archive_url=None,
            value_col="wind_run_km",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_xr_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get relative humidity data from Automated Snow Weather Stations.

        Data is sourced from XR.csv (current season only — no archive).

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs.
        begin_date : str or None
            Start date or datetime string.
        end_date : str or None
            End date or datetime string (inclusive).
        daily_only : bool
            If True (default), return only the 16:00 UTC reading per day.
            If False, return all hourly readings.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date``/``datetime``, ``location_id``, ``rh_pct``.
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/XR.csv",
            archive_url=None,
            value_col="rh_pct",
            archive=False,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
            daily_only=daily_only,
        )

    def get_asws_combined_data(
        self,
        location_id: str,
    ) -> pd.DataFrame:
        """
        Get the full combined time-series for a single ASWS station.

        Fetches the per-station archive from the ``SnowAll/`` directory,
        which contains SW (SWE, mm), SD (snow depth, cm), TA (air temp, °C),
        and PC (precipitation, mm) in one file.

        The 16:00 UTC reading is used as the canonical daily value, consistent
        with ``get_asws_daily_data()`` and ``get_asws_sd_data()``.

        Parameters
        ----------
        location_id : str
            ASWS station location ID (e.g. ``"1E08P"``).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date`` (str ``"YYYY-MM-DD"``),
            ``swe_mm`` (float or NaN),
            ``snwd_cm`` (float or NaN),
            ``air_temp_degc`` (float or NaN),
            ``precip_cumul_mm`` (float or NaN).
        """
        url = f"{SNOW_ALL_BASE}/{location_id}.csv"
        _cols = ["date", "swe_mm", "snwd_cm", "air_temp_degc", "precip_cumul_mm"]
        try:
            resp = self._request(url)
        except DataBCError as exc:
            logger.warning(
                "Could not load SnowAll/%s.csv: %s", location_id, exc
            )
            return pd.DataFrame(columns=_cols)

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        # Normalise column names (headers vary between files)
        col_map: dict[str, str] = {}
        for col in df.columns:
            c = col.strip().lower()
            if "date" in c or c == "datetime":
                col_map[col] = "datetime_raw"
            elif c.startswith("sw") and "unit" not in c and "grade" not in c:
                col_map[col] = "swe_mm"
            elif c.startswith("sd") and "unit" not in c and "grade" not in c:
                col_map[col] = "snwd_cm"
            elif c.startswith("ta") and "unit" not in c and "grade" not in c:
                col_map[col] = "air_temp_degc"
            elif c.startswith("pc") and "unit" not in c and "grade" not in c:
                col_map[col] = "precip_cumul_mm"
        df = df.rename(columns=col_map)

        if "datetime_raw" not in df.columns:
            return pd.DataFrame(columns=_cols)

        # Parse datetime and select the 16:00 UTC reading as daily value
        dt_series = pd.to_datetime(df["datetime_raw"], errors="coerce")
        hour_mask = dt_series.dt.strftime("%H:%M") == _DAILY_UTC_HOUR
        if hour_mask.any():
            df = df[hour_mask].copy()
            df["date"] = dt_series[hour_mask].dt.strftime("%Y-%m-%d")
        else:
            # Fallback: take last non-NaN reading per calendar day
            logger.warning(
                "No 16:00 UTC readings in SnowAll/%s.csv; "
                "falling back to last reading per day",
                location_id,
            )
            df["date"] = dt_series.dt.strftime("%Y-%m-%d")
            df = (
                df.dropna(subset=["date"])
                .groupby("date")
                .last()
                .reset_index()
            )

        df = df.drop(columns=["datetime_raw"], errors="ignore")

        for num_col in ("swe_mm", "snwd_cm", "air_temp_degc", "precip_cumul_mm"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            else:
                df[num_col] = float("nan")

        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for num_col in ("swe_mm", "snwd_cm", "air_temp_degc", "precip_cumul_mm"):
            df.loc[df[num_col] < 0, num_col] = float("nan")

        return df[_cols]

    # ── Public API — MSS time-series data ─────────────────────────────────────

    def get_mss_survey_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
        include_flags: bool = False,
    ) -> pd.DataFrame:
        """
        Get periodic manual snow survey (snow course) data.

        Data is sourced from the BC ``allmss`` CSV files.  Surveys are
        typically conducted monthly from January through May.

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs (e.g. ``["1A06A", "1A10"]``).
        begin_date : str or None
            Start date (``"YYYY-MM-DD"``).
        end_date : str or None
            End date (inclusive, ``"YYYY-MM-DD"``).
        archive : bool
            If True, also load the historical archive (~76 years).
        include_flags : bool
            If True, include the ``survey_code`` column (quality flag field).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date`` (str ``"YYYY-MM-DD"``),
            ``location_id`` (str),
            ``name`` (str),
            ``swe_mm`` (float or NaN),
            ``snwd_cm`` (float or NaN),
            ``density_pct`` (float or NaN),
            ``snow_line_m`` (float or NaN),
            ``survey_period`` (str),
            and optionally ``survey_code`` (str, if ``include_flags=True``).
        """
        dfs = []

        try:
            dfs.append(
                self._load_mss_csv(f"{DATA_BASE}/allmss_current.csv")
            )
        except DataBCError as exc:
            logger.warning("Could not load allmss_current.csv: %s", exc)

        if archive:
            try:
                dfs.append(
                    self._load_mss_csv(f"{DATA_BASE}/allmss_archive.csv")
                )
            except DataBCError as exc:
                logger.warning(
                    "Could not load allmss_archive.csv: %s", exc
                )

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["location_id", "date"])
        df = df.sort_values(["location_id", "date"]).reset_index(drop=True)

        if location_ids is not None:
            df = df[df["location_id"].isin(location_ids)]
        if begin_date:
            df = df[df["date"] >= begin_date]
        if end_date:
            df = df[df["date"] <= end_date]

        if not include_flags and "survey_code" in df.columns:
            df = df.drop(columns=["survey_code"])

        return df.reset_index(drop=True)

    # ── Public API — station photos ───────────────────────────────────────────

    def get_station_image_url(self, location_id: str) -> str | None:
        """
        Get the station photo URL for an ASWS station from the AQRT BCMOE portal.

        Scrapes ``https://bcmoe-prod.aquaticinformatics.net`` (the public
        BC Ministry of Environment AQUARIUS Web Portal).  The portal requires
        accepting a one-time disclaimer; this client does so lazily and reuses
        the session for subsequent calls.

        The returned URL is a direct ``GetFileById`` link that can be embedded
        in an ``<img>`` tag.  Returns ``None`` if the station has no photo or
        the portal is unreachable.

        Parameters
        ----------
        location_id : str
            ASWS station location ID, e.g. ``"1E08P"``.

        Returns
        -------
        str or None
            Direct image URL, e.g.
            ``"https://bcmoe-prod.aquaticinformatics.net/Data/GetFileById/12345"``,
            or ``None`` if no photo is available.
        """
        base = AQRT_BCMOE_BASE

        # Accept the disclaimer once per client session
        if not self._aqrt_disclaimer_accepted:
            try:
                d = self._aqrt_session.get(
                    f"{base}/Disclaimer", timeout=self.timeout
                )
                d.raise_for_status()
                match = re.search(
                    r'name="__RequestVerificationToken"[^>]+value="([^"]+)"',
                    d.text,
                )
                if not match:
                    logger.warning(
                        "Could not find antiforgery token on AQRT disclaimer page"
                    )
                    return None
                token = match.group(1)
                self._aqrt_session.post(
                    f"{base}/AcceptDisclaimer",
                    data={
                        "returnUrl": "/Data",
                        "__RequestVerificationToken": token,
                    },
                    timeout=self.timeout,
                ).raise_for_status()
                self._aqrt_disclaimer_accepted = True
            except Exception as exc:
                logger.warning("Could not accept AQRT BCMOE disclaimer: %s", exc)
                return None

        station_url = (
            f"{base}/Data/Location/Summary"
            f"/Location/{location_id}/Interval/Latest"
        )
        try:
            page = self._aqrt_session.get(station_url, timeout=self.timeout)
            page.raise_for_status()
            numeric_id_match = re.search(
                r"var\s+initialLocation\s*=\s*(\d+)\s*;", page.text
            )
            if not numeric_id_match:
                logger.debug(
                    "No initialLocation found for ASWS station %s", location_id
                )
                return None
            numeric_id = numeric_id_match.group(1)

            summary = self._aqrt_session.get(
                f"{base}/Data/Location_Summary/",
                params={"location": numeric_id},
                timeout=self.timeout,
            )
            summary.raise_for_status()
            m = re.search(r"/Data/GetFileById/(\d+)", summary.text)
            if m:
                return f"{base}/Data/GetFileById/{m.group(1)}"
            return None
        except Exception as exc:
            logger.warning(
                "Could not fetch image URL for %s: %s", location_id, exc
            )
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_wfs_stations(
        self,
        layer: str,
        station_type: str,
        active_only: bool,
    ) -> list[dict]:
        """Fetch station locations from BC OpenMaps WFS."""
        params: dict[str, str] = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": layer,
            "outputFormat": "application/json",
            "count": "2000",
        }
        if active_only:
            params["CQL_FILTER"] = "STATUS='Active'"

        url = f"{WFS_BASE}/{layer}/ows"
        resp = self._request(url, params=params)
        data = resp.json()

        stations = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            # WFS may return coords in properties or geometry
            geom = feature.get("geometry") or {}
            coords = geom.get("coordinates", [None, None])

            lat = props.get("LATITUDE") or (
                coords[1] if len(coords) > 1 else None
            )
            lon = props.get("LONGITUDE") or (
                coords[0] if coords else None
            )
            location_id = str(props.get("LOCATION_ID") or "").strip()
            if not location_id:
                continue

            sta: dict[str, Any] = {
                "location_id": location_id,
                "name": str(props.get("LOCATION_NAME") or "").strip(),
                "elevation_m": _to_float(props.get("ELEVATION")),
                "latitude": _to_float(lat),
                "longitude": _to_float(lon),
                "status": str(props.get("STATUS") or "").strip(),
                "station_type": station_type,
                "station_url": (
                    f"{AQRT_BASE}/Data/Location/Summary"
                    f"/Location/{location_id}/Interval/Latest"
                ),
            }

            if station_type == "ASWS":
                sta["operator"] = str(
                    props.get("OPERATOR") or ""
                ).strip()
                camera = str(props.get("CAMERA_URL") or "").strip()
                sta["camera_url"] = camera if camera else None

            stations.append(sta)

        return stations

    def _get_asws_var_data(
        self,
        current_url: str,
        archive_url: str | None,
        value_col: str,
        archive: bool,
        location_ids: list[str] | None,
        begin_date: str | None,
        end_date: str | None,
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """Generic helper for fetching a single variable from ASWS wide CSVs."""
        time_col = "date" if daily_only else "datetime"
        dfs = []
        try:
            dfs.append(
                self._load_asws_wide_csv(
                    current_url, value_col=value_col, daily_only=daily_only
                )
            )
        except DataBCError as exc:
            logger.warning("Could not load %s: %s", current_url, exc)

        if archive and archive_url:
            try:
                dfs.append(
                    self._load_asws_wide_csv(
                        archive_url, value_col=value_col, daily_only=daily_only
                    )
                )
            except DataBCError as exc:
                logger.warning("Could not load %s: %s", archive_url, exc)

        if not dfs:
            return pd.DataFrame(columns=[time_col, "location_id", value_col])

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=[time_col, "location_id"])
        df = df.sort_values(["location_id", time_col]).reset_index(drop=True)

        if location_ids is not None:
            df = df[df["location_id"].isin(location_ids)]
        if begin_date:
            df = df[df[time_col] >= begin_date]
        if end_date:
            df = df[df[time_col] <= end_date]

        return df.reset_index(drop=True)

    def _load_asws_wide_csv(
        self,
        url: str,
        value_col: str = "swe_mm",
        daily_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load a wide-format ASWS CSV and convert to long format.

        The CSV has one row per UTC timestamp and one column per station.
        Column headers are of the form ``"1A01P Yellowhead Lake"``.

        Parameters
        ----------
        url : str
            URL of the wide-format CSV file.
        value_col : str
            Name to use for the value column in the output DataFrame.
        daily_only : bool
            If True, filter to the 16:00 UTC reading and return a ``date``
            column (``"YYYY-MM-DD"``).  If False, return all rows with a
            ``datetime`` column (``"YYYY-MM-DD HH:MM"`` UTC).
        """
        resp = self._request(url)
        df = pd.read_csv(io.StringIO(resp.text))

        time_col = "date" if daily_only else "datetime"

        if df.empty or len(df.columns) < 2:
            return pd.DataFrame(
                columns=[time_col, "location_id", value_col]
            )

        date_col = df.columns[0]  # "DATE(UTC)"

        if daily_only:
            # Use only the 16:00 UTC reading as the canonical daily value
            mask = df[date_col].astype(str).str.strip().str.endswith(
                _DAILY_UTC_HOUR
            )
            df = df[mask].copy()
            if df.empty:
                return pd.DataFrame(
                    columns=[time_col, "location_id", value_col]
                )
            df[time_col] = df[date_col].astype(str).str[:10]
        else:
            # Return all hourly rows; keep "YYYY-MM-DD HH:MM" format
            df[time_col] = df[date_col].astype(str).str.strip().str[:16]

        df = df.drop(columns=[date_col])

        # Melt to long format
        df_long = df.melt(
            id_vars=[time_col],
            var_name="station_col",
            value_name=value_col,
        )

        # Extract location ID from "1A01P Yellowhead Lake"
        df_long["location_id"] = (
            df_long["station_col"].str.split().str[0].str.strip()
        )
        df_long = df_long.drop(columns=["station_col"])

        df_long[value_col] = pd.to_numeric(
            df_long[value_col], errors="coerce"
        )
        # Remove clearly invalid values (negative, -99999 sentinel)
        df_long.loc[df_long[value_col] < 0, value_col] = float("nan")

        return df_long[[time_col, "location_id", value_col]]

    def _load_mss_csv(self, url: str) -> pd.DataFrame:
        """Load a long-format MSS (manual snow survey) CSV."""
        resp = self._request(url)
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        # Normalise column names
        col_map: dict[str, str] = {}
        for col in df.columns:
            c = col.strip().lower()
            if c in ("number", "course #"):
                col_map[col] = "location_id"
            elif "name" in c or c == "snow course name":
                col_map[col] = "name"
            elif "date" in c:
                col_map[col] = "date_raw"
            elif "depth" in c:
                col_map[col] = "snwd_cm"
            elif "water equiv" in c or c.startswith("water equiv"):
                col_map[col] = "swe_mm"
            elif "survey code" in c:
                col_map[col] = "survey_code"
            elif "snow line" in c:
                col_map[col] = "snow_line_m"
            elif "density" in c:
                col_map[col] = "density_pct"
            elif "period" in c:
                col_map[col] = "survey_period"
            elif "elev" in c:
                col_map[col] = "elevation_m"

        df = df.rename(columns=col_map)

        # Parse YYYY/MM/DD date format to ISO
        if "date_raw" in df.columns:
            df["date"] = pd.to_datetime(
                df["date_raw"], format="%Y/%m/%d", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df = df.drop(columns=["date_raw"])

        # Numeric coercion
        for num_col in ("swe_mm", "snwd_cm", "density_pct", "snow_line_m"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(
                    df[num_col], errors="coerce"
                )

        # Drop rows without a usable location_id or date
        df = df[
            df.get("location_id", pd.Series(dtype=str)).notna()
            & df.get("date", pd.Series(dtype=str)).notna()
        ]

        return df

    def _request(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(
                    url, params=params, timeout=self.timeout
                )
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise DataBCError(
                        f"Request to {url} failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(self.backoff * attempt)
                continue

            if resp.ok:
                return resp

            if resp.status_code in (400, 404):
                raise DataBCError(
                    f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
                )

            if resp.status_code >= 500:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt,
                    self.max_retries,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise DataBCError(
                    f"HTTP {resp.status_code} from {url} after "
                    f"{self.max_retries} attempts"
                )

            raise DataBCError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
            )

        raise DataBCError(f"Exhausted retries for {url}")


# ── Exception ─────────────────────────────────────────────────────────────────

class DataBCError(Exception):
    """Raised when a DataBC request fails."""


# ── Utility ───────────────────────────────────────────────────────────────────

def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", "").strip())
        return None if (f != f) else f  # NaN check
    except (ValueError, TypeError):
        return None
