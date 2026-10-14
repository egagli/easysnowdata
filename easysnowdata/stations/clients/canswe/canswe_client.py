"""
clients/canswe/canswe_client.py
===============================
Python client for CanSWE — the Canadian historical Snow Water Equivalent
dataset (Vionnet et al., 2021, ESSD 13, 4603–4619), compiled by
Environment and Climate Change Canada from national, provincial and
territorial agencies and hydropower companies.

Primary use case: one consistent, quality-controlled archive of manual
snow surveys (1928 onward) and automated snow pillow / passive-gamma
(GMON) SWE across Canada, updated once a year.

Data source       : Zenodo concept record 4734371
                    (https://doi.org/10.5281/zenodo.4734371); v8 is
                    record 19075529, ``CanSWE-CanEEN_1928-2025_v8.nc``
                    (14 MB), published 2026-03-17, data to 2025-07-31.
Format reference  : ``ReadMe_CanSWE_v8.pdf`` in the same record.
Authentication    : none
Licence           : Open Government Licence – Canada; **Hydro-Québec
                    data (source "Hydro-Quebec", ids ``HQ-…``) are
                    CC BY-NC-SA 4.0** (netCDF ``redistribution`` attribute
                    and readme §1).

Files used
----------
The latest version is resolved through the Zenodo API
(``/api/records/4734371/versions/latest``); pass ``record_id=`` to pin
one.  On first use the 14 MB netCDF is downloaded once (resumable,
md5-checked) into ``easysnowdata.config.cache_dir("stations", "canswe")``,
converted to a ~10 MB Zarr store (about 15 s) and deleted; every later
call reads the store lazily (see :mod:`._format`).  A new Zenodo version
is picked up automatically and replaces the old store.  If Zenodo cannot
be reached, the newest cached store is used.  ``path=`` accepts a local
netCDF (converted once into the cache, left in place) or a store.  The CSV in the record's ``.zip`` holds the same data
(``b''``-style Python bytes reprs in its flag columns) and is not used.

Station ids and sources
-----------------------
Ids carry an agency prefix (``BCE-1A01P``, ``ALE-05AA809P``, ``HQ-…``,
``SCD-…``); see :data:`ID_PREFIXES`.  A trailing ``P`` on BC/Alberta/Yukon
ids marks a snow pillow.  ``source`` is the provider name exactly as the
file gives it (:data:`SOURCES` lists those seen in v8).  Duplicates merged
by the authors keep their alternative names / ids in ``station_name_sec``
/ ``_ter`` and ``station_id_sec`` / ``_ter``.

Cadence
-------
``type_mes`` (WMO code, :data:`METHODS`) decides each station's interval:
manual surveys (0, 1) are ``"periodic"`` (typically monthly Jan–Jun);
snow pillows (2) and passive gamma GMON (3) are ``"daily"``.  v8 holds
only codes 0, 2 and 3.

Design principles
-----------------
- Metric-first: SWE kg m-2 → cm (× 0.1), depth m → cm (× 100), bulk
  density kg m-3 → % (÷ 10).
- QC-failed values are already NaN in the file; ``include_flags=True``
  returns them as ``value=None`` with the reason in ``flag``.
- Snow values outside 0–15 m are nulled.
- ``status`` is "Active" when the station reported in the dataset's final
  snow season (2025 for v8) — the archive is annual, so "Active" means
  "still in the most recent yearly update", not "reporting today".
"""

from __future__ import annotations

from ._format import (
    AGENCY_FLAGS,
    DATA_FLAGS,
    METHODS,
    QC_FLAGS,
    CanSWEStyleClient,
    make_variables,
)

__all__ = [
    "AGENCY_FLAGS",
    "DATA_FLAGS",
    "ID_PREFIXES",
    "METHODS",
    "QC_FLAGS",
    "SOURCES",
    "VARIABLES",
    "CanSWEClient",
    "CanSWEError",
]

CONCEPT_RECID = "4734371"
DATASET_URL = "https://doi.org/10.5281/zenodo.4734371"

#: Native variables (``snw``, ``snd``, ``den``).
VARIABLES: dict[str, dict] = make_variables(
    "ECCC CanSWE (Vionnet et al. 2021) — " + DATASET_URL
)

#: Station-id prefix → contributing agency (readme v8 §1).
ID_PREFIXES: dict[str, str] = {
    "YT": "Yukon Water Resources Branch",
    "NWT": "Government of Northwest Territories",
    "SCD": "MSC 'Snow Cover Data' books, 1955–1985",
    "BCE": "British Columbia Ministry of Environment",
    "ALE": "Alberta Environment and Parks",
    "SK": "Saskatchewan Water Security Agency",
    "MB": "Manitoba Hydro",
    "OPG": "Ontario Power Generation",
    "ONR": "Ontario Ministry of Natural Resources and Forestry",
    "HQ": "Hydro-Québec",
    "ENB": "Government of New Brunswick",
    "NFL": "Government of Newfoundland and Labrador",
    "RTA": "Rio Tinto Alcan",
    "CHURCHILL": "Churchill Falls (Labrador) Corporation",
    "INA": "Crown-Indigenous Relations and Northern Affairs Canada",
    "MN": "Government of Manitoba",
    "UU": "Research sites",
}

#: ``source`` values found in v8 (station counts, v8).
SOURCES: dict[str, int] = {
    "Meteorological Service of Canada": 780,
    "British Columbia Environment": 545,
    "Ontario Ministry of Natural Resources": 359,
    "Saskatchewan Water Security Agency": 180,
    "Hydro-Quebec": 177,
    "Alberta Environment": 173,
    "Government of Manitoba": 164,
    "Environment New Brunswick": 118,
    "Yukon Territory": 92,
    "Government of Northwest Territories": 74,
    "Manitoba Hydro": 72,
    "Ontario Power Generation": 58,
    "Crown-Indigenous Relations and Northern Affairs Canada": 37,
    "Government of Newfoundland and Labrador": 36,
    "Rio Tinto Alcan": 33,
    "Canadian Rockies Hydrological Observatory": 19,
    "Churchill Falls": 17,
    "Environment and Climate Change Canada": 14,
    "Changing Cold Regions Network (CCRN)": 6,
    "Agriculture and Agri-Food Canada": 3,
    "Global Water Futures": 3,
    "University of Saskatchewan": 1,
}


class CanSWEError(Exception):
    """Raised when CanSWE cannot be fetched, opened or queried."""


class CanSWEClient(CanSWEStyleClient):
    """
    Client for the CanSWE netCDF (latest Zenodo version, cached locally).

    No credentials are needed.

    Parameters
    ----------
    path : str or Path, optional
        Use this local ``.nc`` instead of resolving / downloading.
    record_id : str, optional
        Pin a Zenodo record (e.g. ``"19075529"`` for v8).  Default: the
        latest version of concept record 4734371.
    cache_dir : str or Path, optional
        Override ``easysnowdata.config.cache_dir("stations", "canswe")``.
    session : requests.Session, optional
        Session used for the Zenodo API call.
    timeout, max_retries, backoff : int
        Shared retry-loop settings (see ``_common.request_with_retries``).
    """

    NAME = "canswe"
    CONCEPT_RECID = CONCEPT_RECID
    FILE_SUFFIX = ".nc"
    DATASET_URL = DATASET_URL
    ERROR = CanSWEError
    VARIABLES = VARIABLES

    def _extra_station_fields(self, raw: dict) -> dict:
        sid = raw["station_id"] or ""
        prefix = sid.split("-", 1)[0] if "-" in sid else None
        alt_names = [raw.get("station_name_sec"), raw.get("station_name_ter")]
        alt_ids = [raw.get("station_id_sec"), raw.get("station_id_ter")]
        return {
            "id_prefix": prefix,
            "agency": ID_PREFIXES.get(prefix or "", raw.get("source")),
            "country": "CA",
            "alt_names": [n for n in alt_names if n],
            "alt_ids": [i for i in alt_ids if i],
        }
