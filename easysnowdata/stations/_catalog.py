"""Catalog entries for the five snow-station networks (§3.3).

One product per network, not one product with five sources: the networks are
different geographies, not alternative routes to the same data, so
``source=`` would be the wrong knob. The same shape ``easysnowdata.hydro.basins``
uses for its four basin datasets.

Every product's loader is :func:`easysnowdata.stations.load`; ``networks=``
picks which of them a call reaches, and a station code usually says by itself.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from easysnowdata import auth, catalog
from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.stations import networks

__all__ = ["PRODUCTS", "variables_for"]

_LOADER = "easysnowdata.stations.load"


def variables_for(network: str) -> tuple[Variable, ...]:
    """The standardized types this network's client can serve, as Variables.

    Read from the client module's own ``VARIABLES`` registry (that repo's
    DESIGN.md §3.2), so a client that grows a variable grows its catalog
    entry with no edit here.
    """
    module = _variables_module(network)
    registry: dict[str, dict] = getattr(module, "VARIABLES", None) or getattr(
        module, "SENSORS", {}
    )
    types: dict[str, str] = {}
    for entry in registry.values():
        type_name = str(entry.get("type", "other"))
        types.setdefault(type_name, networks.canonical_units(type_name))
    return tuple(
        Variable(
            name=name,
            units=units or None,
            dtype="float64",
            long_name=networks.TYPES.get(name, ("", name))[1],
        )
        for name, units in sorted(types.items())
    )


def _variables_module(network: str) -> Any:
    import importlib  # noqa: PLC0415

    return importlib.import_module(
        f"easysnowdata.stations.clients.{network}.{network}_client"
    )


def _health_probe(label: str, url: str, requires: tuple[str, ...] | None = None):
    return Probe(label, partial(health.http_first_byte, url), requires=requires)


def _nve_probe() -> None:
    """``GET /Parameters`` with the ``X-API-Key`` header the endpoint requires.

    A bare first-byte GET answers 401 even when the key is configured, which
    is what the weekly check reported until this probe sent the header.
    """
    import requests  # noqa: PLC0415

    response = requests.get(
        f"{networks.NETWORKS['nve'].api_url}/Parameters",
        headers=auth.get("nve").headers(),
        timeout=health.TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Unreachable: HTTP {response.status_code}")


AWDB_PRODUCT = Product(
    id="awdb-stations",
    theme="stations",
    title="SNOTEL, SCAN and snow-course observations (USDA NRCS AWDB)",
    description=(
        "Daily and hourly SWE, snow depth, precipitation, air temperature, "
        "soil moisture and more from the USDA NRCS Air and Water Database: "
        "SNOTEL and SNOLITE telemetry sites, SCAN, and the manual snow-course "
        "network, across the western United States and parts of western "
        "Canada. Station ids are AWDB triplets (679:WA:SNTL), spelled "
        "679_WA_SNTL as a global station code."
    ),
    sources=(
        Source(
            id="awdb",
            provider="stations",
            location=networks.NETWORKS["awdb"].api_url,
            extent="western US, western Canada",
            temporal="~1978/present",
            latency="hours",
            notes=(
                "the AWDB REST API v1; no account. Values are converted to "
                "metric in the client, never passed through in inches"
            ),
            title="USDA NRCS AWDB REST API v1",
            health=_health_probe(
                "AWDB stations (NRCS REST API)",
                f"{networks.NETWORKS['awdb'].api_url}/stations?"
                "stationTriplets=679:WA:SNTL",
            ),
        ),
    ),
    variables=variables_for("awdb"),
    citation=(
        "USDA Natural Resources Conservation Service, National Water and "
        "Climate Center. Air and Water Database (AWDB) REST API."
    ),
    license="Public domain (US government data)",
    loader=_LOADER,
    examples=("stations/plot_awdb_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html",
        "https://www.nrcs.usda.gov/wps/portal/wcc/home/",
    ),
    tags=("swe", "snow depth", "stations", "snotel", "scan", "snow course"),
)

CDEC_PRODUCT = Product(
    id="cdec-stations",
    theme="stations",
    title="California snow pillows and snow courses (CDEC)",
    description=(
        "SWE, snow depth, precipitation, temperature and more from the "
        "California Data Exchange Center: the California Cooperative Snow "
        "Surveys (CCSS) snow pillows and the manual snow-course network. "
        "Station ids are CDEC's three-letter codes (QUA)."
    ),
    sources=(
        Source(
            id="cdec",
            provider="stations",
            location=networks.NETWORKS["cdec"].api_url,
            extent="California",
            temporal="~1920/present (courses), ~1980/present (pillows)",
            latency="hours",
            notes=(
                "JSON data servlet plus scraped station-metadata pages; no "
                "account. SWE prefers the adjusted sensor (82) over the raw "
                "one (3)"
            ),
            title="California Data Exchange Center",
            health=_health_probe(
                "CDEC stations (JSON data servlet)",
                "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet?"
                "Stations=QUA&SensorNums=82&dur_code=D&Start=2024-01-01&"
                "End=2024-01-02",
            ),
        ),
    ),
    variables=variables_for("cdec"),
    citation=(
        "California Department of Water Resources, California Data Exchange "
        "Center (CDEC); California Cooperative Snow Surveys."
    ),
    license="Public domain (California state government data)",
    loader=_LOADER,
    examples=("stations/plot_cdec_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://cdec.water.ca.gov/",
        "https://cdec.water.ca.gov/snow/current/snow/",
    ),
    tags=("swe", "snow depth", "stations", "ccss", "cdec", "snow course"),
)

DATABC_PRODUCT = Product(
    id="databc-stations",
    theme="stations",
    title="British Columbia automated snow weather stations and snow surveys",
    description=(
        "Hourly and daily SWE, snow depth, precipitation, temperature, wind "
        "and barometric pressure from the BC automated snow weather station "
        "(ASWS) network, plus the manual snow survey (MSS) courses. Station "
        "ids are BC location codes (1A01P)."
    ),
    sources=(
        Source(
            id="databc",
            provider="stations",
            location=networks.NETWORKS["databc"].api_url,
            extent="British Columbia",
            temporal="~1970/present",
            latency="hours to a day",
            notes=(
                "station geometry from the BC Data Catalogue WFS, "
                "observations from the ENV snow data CSVs; no account. The "
                "daily snow-depth reading is the 16:00 UTC value"
            ),
            title="BC Data Catalogue / BC ENV snow data",
            health=_health_probe(
                "BC snow stations (DataBC WFS)",
                "https://openmaps.gov.bc.ca/geo/pub/"
                "WHSE_WATER_MANAGEMENT.SSL_SNOW_ASWS_STNS_SP/ows?"
                "service=WFS&version=2.0.0&request=GetFeature&"
                "typeNames=WHSE_WATER_MANAGEMENT.SSL_SNOW_ASWS_STNS_SP&"
                "count=1&outputFormat=application/json",
            ),
        ),
    ),
    variables=variables_for("databc"),
    citation=(
        "Province of British Columbia, Ministry of Environment and Climate "
        "Change Strategy. Automated Snow Weather Station and Manual Snow "
        "Survey data, BC Data Catalogue."
    ),
    license="Open Government Licence – British Columbia",
    loader=_LOADER,
    examples=("stations/plot_databc_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://catalogue.data.gov.bc.ca/dataset/snow-weather-stations-archive",
        "https://www.env.gov.bc.ca/wsd/data_searches/snow/",
    ),
    tags=("swe", "snow depth", "stations", "british columbia", "asws"),
)

NVE_PRODUCT = Product(
    id="nve-stations",
    theme="stations",
    title="Norwegian snow pillows and snow-depth stations (NVE HydAPI)",
    description=(
        "Daily and hourly SWE and snow depth from the Norwegian Water "
        "Resources and Energy Directorate's hydrological network, through "
        "HydAPI. Station ids are NVE's dotted codes (12.142.0). This is the "
        "one network that needs a credential: a free API key in NVE_API_KEY."
    ),
    sources=(
        Source(
            id="nve",
            provider="stations",
            location=networks.NETWORKS["nve"].api_url,
            requires=("nve",),
            extent="Norway",
            temporal="~1900/present (varies widely by station)",
            latency="hours",
            notes=(
                "free API key, sent as X-API-Key; request one at "
                "https://hydapi.nve.no/. Parameter 2003 is SWE, 2002 snow "
                "depth"
            ),
            title="NVE HydAPI v1",
            # The endpoint answers 401 without the key, so the probe is only
            # meaningful where the key is configured — and it has to send it.
            health=Probe("NVE stations (HydAPI)", _nve_probe, requires=("nve",)),
        ),
    ),
    variables=variables_for("nve"),
    citation=(
        "Norges vassdrags- og energidirektorat (NVE). Hydrological API "
        "(HydAPI), https://hydapi.nve.no/."
    ),
    license="Norwegian Licence for Open Government Data (NLOD)",
    loader=_LOADER,
    examples=("stations/plot_nve_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://hydapi.nve.no/UserDocumentation/",
        "https://sildre.nve.no/",
    ),
    tags=("swe", "snow depth", "stations", "norway", "nve"),
)

YUKON_PRODUCT = Product(
    id="yukon-stations",
    theme="stations",
    title="Yukon snow survey and automated stations (AquaCache)",
    description=(
        "SWE and snow depth from the Yukon Snow Survey Network's automated "
        "snow-weather stations and manual snow courses, plus ECCC "
        "meteorological stations mirrored into the same database, served by "
        "the Yukon Water Data API. Station ids are AquaCache location codes "
        "(09AA-M1). A few courses the survey operates sit in northern BC and "
        "Alaska."
    ),
    sources=(
        Source(
            id="yukon",
            provider="stations",
            location=networks.NETWORKS["yukon"].api_url,
            extent="Yukon, with courses in northern BC and Alaska",
            temporal="~1975/present",
            latency="hours to a day",
            notes="the open AquaCache REST API; no account",
            title="Yukon Water Data (AquaCache) API v1",
            health=_health_probe(
                "Yukon stations (AquaCache API)",
                f"{networks.NETWORKS['yukon'].api_url}/locations?limit=1",
            ),
        ),
    ),
    variables=variables_for("yukon"),
    citation=(
        "Government of Yukon, Water Resources Branch. Yukon Water Data "
        "(AquaCache) API; Yukon Snow Survey Network."
    ),
    license="Open Government Licence – Yukon",
    loader=_LOADER,
    examples=("stations/plot_yukon_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://service.yukon.ca/water-data/shiny/?page=home&lang=en",
        "https://yukon.ca/en/snow-survey-bulletin",
    ),
    tags=("swe", "snow depth", "stations", "yukon", "snow course"),
)

PRODUCTS: tuple[Product, ...] = (
    AWDB_PRODUCT,
    CDEC_PRODUCT,
    DATABC_PRODUCT,
    NVE_PRODUCT,
    YUKON_PRODUCT,
)

catalog.register(*PRODUCTS, replace=True)
