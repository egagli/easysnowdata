"""
clients/norswe/norswe_client.py
===============================
Python client for NorSWE — the Northern Hemisphere historical in-situ
Snow Water Equivalent dataset, 1979–2021 (Mortimer & Vionnet, 2025),
compiled by Environment and Climate Change Canada.

Primary use case: one harmonised, quality-controlled SWE archive spanning
North America (CanSWE, NRCS snow courses and SNOTEL, Maine, New
Hampshire, NRCC), Russia (RIHMI-WDC route snow surveys), Finland (SYKE
snow courses), Norway (NVE pillows / GMON / surveys), Switzerland (SLF
GCOS manual sites) and NOAA NOHRSC airborne-gamma flight lines.  It is a
frozen archive: v3 ends 2021-07-31 and is not updated in near real time.

Data source       : Zenodo concept record 10287092
                    (https://doi.org/10.5281/zenodo.10287092); v3 is
                    record 15263370, ``NorSWE-NorEEN_1979-2021_v3.zip``
                    (2.4 GB zip holding one netCDF), published 2025-04-22.
Format reference  : ``Readme_NorSWE_19792021_v3.pdf`` in the same record;
                    the netCDF follows the CanSWE conventions (see
                    :mod:`easysnowdata.stations.clients.canswe._format`)
                    plus a mountain mask ``mmask``.
Authentication    : none
Licence           : mixed, per source (netCDF ``Distribution`` attribute
                    and readme "Acknowledgements"): CanSWE data Open
                    Government Licence – Canada, except Hydro-Québec
                    (``CanSWE-HQ-…``) CC BY-NC-SA 4.0 (non-commercial);
                    US data (NOHRSC, NRCS, MGS, NH DES) CC0 / US-PD;
                    SYKE CC BY 4.0; SLF GCOS data ODbL + DbCL; NVE NLOD
                    2.0.  NRCC and RIHMI-WDC terms are not stated
                    (unverified).  The Zenodo record has no licence field.

Files used
----------
The latest version is resolved through the Zenodo API
(``/api/records/10287092/versions/latest``); pass ``record_id=`` to pin
one.  First use builds a local store, once, in
``easysnowdata.config.cache_dir("stations", "norswe")``:

1. the 2.4 GB zip is downloaded in parallel byte ranges (resumable: an
   interrupted download continues where it stopped; md5-checked);
2. the 21.6 GB netCDF is unzipped and the zip deleted;
3. the netCDF is converted to a compact Zarr store (31 MB for v3; about
   8 minutes and 1.6 GB of RAM) — data
   as ``station_id × time`` float32, flags as int8 codes — and deleted.

Peak disk use during step 2–3 is about 22 GB; the client checks free space
first and says how much it needs (set ``EASYSNOWDATA_CACHE_DIR`` to put
the cache on a bigger disk).  Every later call reads the store lazily.
The first station listing also computes a per-station summary (period
of record, counts, daily-cadence test), cached as
``<store>.summary.json``.  ``path=`` accepts an existing netCDF (converted
once into the cache, left in place) or store.

Station ids, sources and countries
----------------------------------
Every id carries a source prefix (:data:`ID_PREFIXES`): ``CanSWE-``
(then the CanSWE agency prefix, e.g. ``CanSWE-BCE-1A01P``),
``SNOTEL-<station number>`` (the readme says ``SNT-``; the file uses
``SNOTEL-``), ``NRCS-<state>-<id>`` (NRCS snow courses; the ``CA`` ones
carry CDEC 3-letter ids, e.g. ``NRCS-CA-KSR``), ``MGS-``, ``NRCC-``,
``NHDES-``, ``NOHRSC-`` (airborne gamma lines), ``RIHMI-``, ``SYKE-``,
``NVE-`` and ``SLF-``.  Each station dict carries ``id_prefix``,
``country`` (ISO-3166 alpha-2 as far as the prefix implies it; NOHRSC
lines can cross the US–Canada border and are tagged ``"US"``), and
``mmask`` (1 = mountain site: GMBA v2 inventory + 25 km buffer or ≥ 2°
slope; this switches the QC limits to 8 m / 8000 kg m-2).

RIHMI-WDC ids end in ``_1``, ``_2`` or ``_3`` for the field/open, forest
and gully routes at the same station (515 coordinates → 1163 ids; readme
Table 5); ``rihmi_landscape`` gives the decoded value.

Layout differences from CanSWE
------------------------------
Data arrays are ``time × station_id`` (CanSWE: ``station_id × time``),
``type_mes`` adds code 64 (airborne gamma), the flag arrays are
variable-length strings (``"AM"``, ``"AI-avg2"``, …) and the netCDF is
21.6 GB uncompressed on disk (2.4 GB zipped) — mostly those flag
strings.  The conversion handles both layouts; in the local store they
are int8 codes and cost almost nothing.  The first station listing scans SWE,
depth and density once (≈ 6 s from the store) and caches the result.

Cadence
-------
``type_mes`` (:data:`METHODS`) decides each station's interval: manual
surveys (0, 1) are ``"periodic"``; pillows (2) and GMON (3) ``"daily"``;
airborne gamma lines (64) are ``"periodic"`` (a few flights per season,
not every year).  The time axis runs 1979-01-01 to 2021-12-31, so the
last snow season (2022) is only August–December 2021.

Design principles
-----------------
Same as CanSWE: SWE → cm, depth → cm, density → %; QC-removed values
are NaN in the file (exposed through ``include_flags``); snow values
outside 0–15 m are nulled.  ``status`` is always "Inactive" for v3: the
archive ends in 2021, so no station is reporting into it.
"""

from __future__ import annotations

from ..canswe._format import (
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
    "VARIABLES",
    "NorSWEClient",
    "NorSWEError",
]

CONCEPT_RECID = "10287092"
DATASET_URL = "https://doi.org/10.5281/zenodo.10287092"

#: Native variables (``snw``, ``snd``, ``den``).
VARIABLES: dict[str, dict] = make_variables(
    "ECCC NorSWE (Mortimer & Vionnet 2025) — " + DATASET_URL
)

#: Station-id prefix → (source description, country code).
ID_PREFIXES: dict[str, tuple[str, str]] = {
    "CanSWE": ("CanSWE v6 (ECCC and Canadian partners)", "CA"),
    "SNOTEL": ("USDA NRCS SNOTEL", "US"),
    "NRCS": ("USDA NRCS snow courses", "US"),
    "MGS": ("Maine Geological Survey cooperative snow survey", "US"),
    "NRCC": ("Northeast Regional Climate Center", "US"),
    "NHDES": ("New Hampshire Department of Environmental Services", "US"),
    "NOHRSC": ("NOAA NOHRSC airborne gamma flight lines", "US"),
    "RIHMI": ("Russia RIHMI-WDC route snow surveys", "RU"),
    "SYKE": ("Finnish Environment Institute (SYKE) snow courses", "FI"),
    "NVE": ("Norwegian Water Resources and Energy Directorate", "NO"),
    "SLF": ("WSL-SLF GCOS SWE sites", "CH"),
}

#: Last digit of a RIHMI-WDC id → landscape of the survey route.
RIHMI_LANDSCAPE: dict[str, str] = {"1": "field", "2": "forest", "3": "gully"}


class NorSWEError(Exception):
    """Raised when NorSWE cannot be fetched, opened or queried."""


class NorSWEClient(CanSWEStyleClient):
    """
    Client for the NorSWE netCDF (latest Zenodo version, cached locally).

    No credentials are needed.  The first use downloads 2.4 GB.

    Parameters
    ----------
    path : str or Path, optional
        Use this local ``.nc`` instead of resolving / downloading.
    record_id : str, optional
        Pin a Zenodo record (e.g. ``"15263370"`` for v3).
    cache_dir : str or Path, optional
        Override ``easysnowdata.config.cache_dir("stations", "norswe")``.
    session : requests.Session, optional
        Session used for the Zenodo API call.
    timeout, max_retries, backoff : int
        Shared retry-loop settings.
    """

    NAME = "norswe"
    CONCEPT_RECID = CONCEPT_RECID
    FILE_SUFFIX = ".zip"
    DATASET_URL = DATASET_URL
    ERROR = NorSWEError
    VARIABLES = VARIABLES

    def _extra_station_fields(self, raw: dict) -> dict:
        sid = raw["station_id"] or ""
        prefix = sid.split("-", 1)[0] if "-" in sid else None
        desc, country = ID_PREFIXES.get(prefix or "", (None, None))
        extra = {
            "id_prefix": prefix,
            "dataset": desc,
            "country": country,
            "mmask": int(raw["mmask"]) if raw.get("mmask") is not None else None,
            "airborne_gamma": prefix == "NOHRSC",
        }
        if prefix == "CanSWE":
            rest = sid.split("-", 2)
            extra["canswe_id"] = sid.split("-", 1)[1]
            extra["canswe_prefix"] = rest[1] if len(rest) > 1 else None
        if prefix == "RIHMI":
            extra["rihmi_landscape"] = RIHMI_LANDSCAPE.get(sid[-1:])
        return extra

    def get_all_stations(
        self,
        active_only: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        sources: list[str] | str | None = None,
        methods: list[int | str] | int | str | None = None,
        daily_only: bool = False,
        countries: list[str] | str | None = None,
        prefixes: list[str] | str | None = None,
        mountain_only: bool = False,
    ) -> list[dict]:
        """
        Every NorSWE station; see ``CanSWEStyleClient.get_all_stations``.

        Extra filters: ``countries`` (``"RU"``, ``"FI"``, …), ``prefixes``
        (id prefixes, e.g. ``"NOHRSC"`` for airborne gamma or ``"SNOTEL"``),
        ``mountain_only`` (``mmask == 1``).
        """
        stations = super().get_all_stations(
            active_only=active_only,
            bbox=bbox,
            sources=sources,
            methods=methods,
            daily_only=daily_only,
        )
        if countries:
            want = {
                c.upper()
                for c in ([countries] if isinstance(countries, str) else countries)
            }
            stations = [s for s in stations if (s.get("country") or "") in want]
        if prefixes:
            want = {
                p.lower()
                for p in ([prefixes] if isinstance(prefixes, str) else prefixes)
            }
            stations = [
                s for s in stations if (s.get("id_prefix") or "").lower() in want
            ]
        if mountain_only:
            stations = [s for s in stations if s.get("mmask") == 1]
        return stations
