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
     - ASWS daily SWE:  ``SWDaily.csv`` / ``SW_DailyArchive.csv``
       Wide format: DATE(UTC) rows × location ID columns.
       Two readings per day (00:00 and 16:00 UTC); 16:00 UTC is used as the
       daily value (approximately 08:00 PST).
       Units: mm SWE.
     - MSS periodic surveys: ``allmss_current.csv`` / ``allmss_archive.csv``
       Long format: one row per station × survey date.
       Columns include Snow Depth (cm) and Water Equiv. (mm SWE).

Station ID conventions
----------------------
- ASWS station location IDs end in ``P`` (e.g. ``1A01P``, ``1E08P``).
- MSS location IDs do not end in ``P`` (e.g. ``1A06A``, ``1A10``).

Station URLs
------------
Each station has a page on the BC AQRT (Aquarius Report Tool) portal:
  ``https://aqrt.nrs.gov.bc.ca/Data/Location/Summary/Location/{id}/Interval/Latest``

Camera URLs are available for a small subset of ASWS stations
(stored in the ``CAMERA_URL`` WFS field); these are third-party webcam feeds.

Variables and units
-------------------
+-------------------+--------+-----------------------------------+
| Variable          | Units  | Source                            |
+===================+========+===================================+
| swe_mm            | mm     | ASWS daily pillow (SWDaily.csv)   |
| swe_mm            | mm     | MSS periodic survey (Water Equiv.)|
| snwd_cm           | cm     | MSS periodic survey (Snow Depth)  |
| density_pct       | %      | MSS periodic survey (Density %)   |
| snow_line_m       | m      | MSS periodic survey (Snow Line)   |
+-------------------+--------+-----------------------------------+

For the per-station CSV archive, ASWS SWE (mm) is converted to cm (÷ 10)
for ``wteq_cm``.  MSS data is periodic and not stored in the daily CSV
archive.

Design principles
-----------------
- Returns plain Python objects (dicts / lists) or pandas DataFrames.
- Metric-first: all outputs are in SI units.
- Missing / sentinel values (NaN, negative) are normalised to ``None``/NaN.
- Data flags in MSS survey: the ``Survey Code`` field (e.g. ``"PROBLEM"``)
  acts as a data flag.  Include ``include_flags=True`` to retain it.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

AQRT_BASE = "https://aqrt.nrs.gov.bc.ca"
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
VARIABLES: dict[str, dict[str, str]] = {
    "swe_mm": {
        "name": "Snow Water Equivalent",
        "units": "mm",
        "source": "ASWS (daily automated) and MSS (periodic survey)",
        "description": (
            "ASWS: automated snow pillow reading from SWDaily.csv. "
            "MSS: manually surveyed water equivalent."
        ),
    },
    "snwd_cm": {
        "name": "Snow Depth",
        "units": "cm",
        "source": "ASWS (daily automated, SD.csv) and MSS (periodic survey)",
        "description": (
            "ASWS: automated snow depth sensor reading from SD.csv, "
            "16:00 UTC value used as daily canonical reading. "
            "MSS: manually measured snow depth from snow course surveys."
        ),
    },
    "air_temp_degc": {
        "name": "Air Temperature",
        "units": "°C",
        "source": "ASWS (daily automated, TA.csv)",
        "description": "Hourly air temperature from ASWS stations; 16:00 UTC reading used.",
    },
    "precip_mm": {
        "name": "Precipitation",
        "units": "mm",
        "source": "ASWS (daily automated, PC.csv)",
        "description": "Hourly precipitation accumulation from ASWS stations.",
    },
    "density_pct": {
        "name": "Snow Density",
        "units": "%",
        "source": "MSS (periodic survey only)",
        "description": "Snow density calculated from depth and SWE.",
    },
    "snow_line_m": {
        "name": "Snow Line Elevation",
        "units": "m",
        "source": "MSS (periodic survey only)",
        "description": "Elevation of the snow line at time of survey.",
    },
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
    endpoints.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Retry attempts on transient server errors.
    backoff : int
        Base backoff delay in seconds.
    session : requests.Session or None
        Optional pre-configured session.
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

    # ── Public API ────────────────────────────────────────────────────────────

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

    def get_all_stations(self, active_only: bool = False) -> list[dict]:
        """
        Get both ASWS and MSS stations.

        Parameters
        ----------
        active_only : bool
            If True, filter to Active stations/sites only.

        Returns
        -------
        list[dict]
            Combined list of ASWS and MSS stations.
        """
        asws = self.get_asws_stations(active_only=active_only)
        mss = self.get_mss_stations(active_only=active_only)
        return asws + mss

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

        Data is sourced from wide-format CSV files.  The 16:00 UTC reading
        is used as the canonical daily value (~08:00 PST / 09:00 PDT).

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
        )

    def get_asws_sd_data(
        self,
        location_ids: list[str] | None = None,
        begin_date: str | None = None,
        end_date: str | None = None,
        archive: bool = True,
    ) -> pd.DataFrame:
        """
        Get daily snow depth data from Automated Snow Weather Stations.

        Data is sourced from wide-format hourly CSV files (SD.csv /
        SD_Archive.csv).  The 16:00 UTC reading is used as the canonical
        daily value (~08:00 PST / 09:00 PDT), matching the convention used
        for SWE in ``get_asws_daily_data()``.

        Parameters
        ----------
        location_ids : list[str] or None
            Filter to specific location IDs (e.g. ``["1A01P", "1E08P"]``).
        begin_date : str or None
            Start date (``"YYYY-MM-DD"``).
        end_date : str or None
            End date (inclusive, ``"YYYY-MM-DD"``).
        archive : bool
            If True, also load the historical archive CSV.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date`` (str ``"YYYY-MM-DD"``),
            ``location_id`` (str),
            ``snwd_cm`` (float or NaN).
        """
        return self._get_asws_var_data(
            current_url=f"{DATA_BASE}/SD.csv",
            archive_url=f"{DATA_BASE}/SD_Archive.csv",
            value_col="snwd_cm",
            archive=archive,
            location_ids=location_ids,
            begin_date=begin_date,
            end_date=end_date,
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
            ``precip_mm`` (float or NaN).
        """
        url = f"{SNOW_ALL_BASE}/{location_id}.csv"
        try:
            resp = self._request(url)
        except DataBCError as exc:
            logger.warning(
                "Could not load SnowAll/%s.csv: %s", location_id, exc
            )
            return pd.DataFrame(
                columns=["date", "swe_mm", "snwd_cm", "air_temp_degc", "precip_mm"]
            )

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
                col_map[col] = "precip_mm"
        df = df.rename(columns=col_map)

        if "datetime_raw" not in df.columns:
            return pd.DataFrame(
                columns=["date", "swe_mm", "snwd_cm", "air_temp_degc", "precip_mm"]
            )

        df["date"] = (
            pd.to_datetime(df["datetime_raw"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
        )
        df = df.drop(columns=["datetime_raw"])

        for num_col in ("swe_mm", "snwd_cm", "air_temp_degc", "precip_mm"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            else:
                df[num_col] = float("nan")

        # Take one reading per date (last non-NaN per day)
        df = df.dropna(subset=["date"])
        df = (
            df.groupby("date")
            .last()
            .reset_index()
        )
        df = df.sort_values("date").reset_index(drop=True)

        for num_col in ("swe_mm", "snwd_cm", "air_temp_degc", "precip_mm"):
            df.loc[df[num_col] < 0, num_col] = float("nan")

        return df[["date", "swe_mm", "snwd_cm", "air_temp_degc", "precip_mm"]]

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
        archive_url: str,
        value_col: str,
        archive: bool,
        location_ids: list[str] | None,
        begin_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        """Generic helper for fetching a single variable from ASWS wide CSVs."""
        dfs = []
        try:
            dfs.append(self._load_asws_wide_csv(current_url, value_col=value_col))
        except DataBCError as exc:
            logger.warning("Could not load %s: %s", current_url, exc)

        if archive:
            try:
                dfs.append(
                    self._load_asws_wide_csv(archive_url, value_col=value_col)
                )
            except DataBCError as exc:
                logger.warning("Could not load %s: %s", archive_url, exc)

        if not dfs:
            return pd.DataFrame(columns=["date", "location_id", value_col])

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "location_id"])
        df = df.sort_values(["location_id", "date"]).reset_index(drop=True)

        if location_ids is not None:
            df = df[df["location_id"].isin(location_ids)]
        if begin_date:
            df = df[df["date"] >= begin_date]
        if end_date:
            df = df[df["date"] <= end_date]

        return df.reset_index(drop=True)

    def _load_asws_wide_csv(self, url: str, value_col: str = "swe_mm") -> pd.DataFrame:
        """
        Load a wide-format ASWS CSV and convert to long format.

        The CSV has one row per timestamp and one column per station.
        Column headers are of the form ``"1A01P Yellowhead Lake"``.
        """
        resp = self._request(url)
        df = pd.read_csv(io.StringIO(resp.text))

        if df.empty or len(df.columns) < 2:
            return pd.DataFrame(
                columns=["date", "location_id", value_col]
            )

        date_col = df.columns[0]  # "DATE(UTC)"

        # Use only the 16:00 UTC reading as the daily value
        mask = df[date_col].astype(str).str.strip().str.endswith(
            _DAILY_UTC_HOUR
        )
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame(
                columns=["date", "location_id", value_col]
            )

        df["date"] = df[date_col].astype(str).str[:10]
        df = df.drop(columns=[date_col])

        # Melt to long format
        df_long = df.melt(
            id_vars=["date"],
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
        # Remove clearly invalid values (negative)
        df_long.loc[df_long[value_col] < 0, value_col] = float("nan")

        return df_long[["date", "location_id", value_col]]

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
