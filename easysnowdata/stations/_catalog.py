"""Catalog entries for the snow-station networks (§3.3).

One product per network, not one product with one source per network: the networks are
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
    title="SNOTEL, SCAN, and snow-course observations (USDA NRCS AWDB)",
    description=(
        "Daily and hourly SWE, snow depth, precipitation, air temperature, "
        "soil moisture, and more from the USDA NRCS Air and Water Database: "
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
        "SWE, snow depth, precipitation, temperature, and more from the "
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
        "Hourly and daily SWE, snow depth, precipitation, temperature, wind, "
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
            title="Yukon Water Data (AquaCache) API v2",
            health=_health_probe(
                "Yukon stations (AquaCache API)",
                f"{networks.NETWORKS['yukon'].api_url}/locations?limit=1",
            ),
        ),
    ),
    variables=variables_for("yukon"),
    citation=(
        "Government of Yukon, Department of Environment, Water Science and "
        "Stewardship Branch. Yukon Water Data (AquaCache) API v2, "
        "https://service.yukon.ca/water-data/api/v2/; Yukon Snow Survey "
        "Network. Contains information licensed under the Open Government "
        "Licence – Yukon."
    ),
    license="Open Government Licence – Yukon",
    loader=_LOADER,
    examples=("stations/plot_yukon_stations.py", "stations/plot_all_networks.py"),
    references=(
        "https://service.yukon.ca/water-data/api/v2/__docs__/",
        "https://service.yukon.ca/water-data/shiny/?page=home&lang=en",
        "https://yukon.ca/en/snow-survey-bulletin",
        "https://yukon.ca/en/your-government/open-government/"
        "open-government-licence-yukon",
    ),
    tags=("swe", "snow depth", "stations", "yukon", "snow course"),
)

ALBERTA_PRODUCT = Product(
    id="alberta-stations",
    theme="stations",
    title="Alberta River Basins snow pillows and snow courses (AEPA)",
    description=(
        "Daily and sub-hourly SWE and snow depth from Alberta Environment and "
        "Protected Areas' snow pillows on the Rocky Mountain eastern slopes, "
        "plus monthly snow surveys, read through the Alberta River Basins "
        "site. Station codes are ARB-<station number> (ARB-05BB803); the "
        "BC Snow Survey and SNOTEL pillows the site mirrors are better read "
        "from their own networks."
    ),
    sources=(
        Source(
            id="alberta",
            provider="stations",
            location=networks.NETWORKS["alberta"].api_url,
            extent="Alberta",
            temporal="1984/present (pillows)",
            latency="hours",
            notes=(
                "the endpoints the site's own pages call; no documented API "
                "and no account. A failed download raises unless "
                "allow_partial=True"
            ),
            title="Alberta River Basins (rivers.alberta.ca)",
            health=_health_probe(
                "Alberta River Basins site", networks.NETWORKS["alberta"].api_url
            ),
        ),
    ),
    variables=variables_for("alberta"),
    citation=(
        "Alberta Environment and Protected Areas. Alberta River Basins, "
        "https://rivers.alberta.ca/."
    ),
    license="No licence stated (site disclaimer only)",
    loader=_LOADER,
    examples=("station_datasets/plot_alberta.py",),
    references=("https://rivers.alberta.ca/",),
    tags=("swe", "snow depth", "stations", "alberta", "snow pillow"),
)

HYDROQUEBEC_PRODUCT = Product(
    id="hydroquebec-stations",
    theme="stations",
    title="Hydro-Québec GMON SWE and snow depth (rolling 10-day feed)",
    description=(
        "Hourly SWE from GMON passive-gamma sensors and snow depth across "
        "Hydro-Québec's watersheds. The open feed keeps only about ten days; "
        "longer records are in global_snow_networks' archive "
        "(stations.archive.load) and, before October 2026, in CanSWE. "
        "Station codes are Hydro-Québec identifiants (1-10613). Licensed "
        "CC BY-NC 4.0: no commercial use."
    ),
    sources=(
        Source(
            id="hydroquebec",
            provider="stations",
            location=networks.NETWORKS["hydroquebec"].api_url,
            extent="Québec",
            temporal="rolling ~10 days",
            latency="hours",
            notes="Opendatasoft API, dataset donnees-hydrometeorologiques; no account",
            title="Hydro-Québec open data (Opendatasoft)",
            health=_health_probe(
                "Hydro-Québec open data (Opendatasoft)",
                f"{networks.NETWORKS['hydroquebec'].api_url}/catalog/datasets/"
                "donnees-hydrometeorologiques",
            ),
        ),
    ),
    variables=variables_for("hydroquebec"),
    citation=(
        "Hydro-Québec. Données hydrométéorologiques, "
        "https://donnees.hydroquebec.com/explore/dataset/"
        "donnees-hydrometeorologiques/ (CC BY-NC 4.0)."
    ),
    license="CC BY-NC 4.0 (attribution, no commercial use)",
    loader=_LOADER,
    examples=("station_datasets/plot_hydroquebec.py",),
    references=(
        "https://donnees.hydroquebec.com/explore/dataset/donnees-hydrometeorologiques/",
        "https://creativecommons.org/licenses/by-nc/4.0/",
    ),
    tags=("swe", "snow depth", "stations", "quebec", "gmon"),
)

SLF_PRODUCT = Product(
    id="slf-stations",
    theme="stations",
    title="SLF IMIS stations, study plots and GCOS SWE (Switzerland)",
    description=(
        "Snow depth and new snow from the WSL Institute for Snow and Avalanche "
        "Research SLF: the automatic IMIS stations and the manual study "
        "plots, plus the fortnightly GCOS SWE series. Station codes name the "
        "dataset, because SLF reuses ids across them: SLF-IMIS-WFJ2, "
        "SLF-SP-5WJ0, SLF-GCOS-5WJ."
    ),
    sources=(
        Source(
            id="slf",
            provider="stations",
            location=networks.NETWORKS["slf"].api_url,
            extent="Swiss Alps",
            temporal="study plots from the 1930s-1990s; IMIS from the late 1990s",
            latency="a day",
            notes=(
                "the SLF file service and EnviDat; no account. Cite SLF "
                "(CC BY 4.0; GCOS is ODbL)"
            ),
            title="SLF measurement data / EnviDat",
            health=_health_probe(
                "SLF measurement data", f"{networks.NETWORKS['slf'].api_url}/imis/"
            ),
        ),
    ),
    variables=variables_for("slf"),
    citation=(
        "WSL Institute for Snow and Avalanche Research SLF. IMIS and manual "
        "study-plot measurements, https://measurement-data.slf.ch/; GCOS SWE "
        "data, EnviDat doi:10.16904/15."
    ),
    license="CC BY 4.0 (IMIS, study plots); ODbL-1.0 (GCOS)",
    loader=_LOADER,
    examples=("station_datasets/plot_slf.py",),
    references=(
        "https://measurement-data.slf.ch/",
        "https://www.slf.ch/en/services-and-products/slf-data-service/",
    ),
    tags=("swe", "snow depth", "stations", "switzerland", "imis"),
)

SNIH_PRODUCT = Product(
    id="snih-stations",
    theme="stations",
    title="SNIH snow pillows and snow courses (Argentina)",
    description=(
        "Snow-pillow SWE (validated daily to 2023-24, raw telemetry since) "
        "and snow-course SWE, depth and density from Argentina's Sistema "
        "Nacional de Información Hídrica. Station codes are SNIH-<code> "
        "(SNIH-12817)."
    ),
    sources=(
        Source(
            id="snih",
            provider="stations",
            location=networks.NETWORKS["snih"].api_url,
            extent="Argentine Andes",
            temporal="1951/present (courses), 1989/present (pillows)",
            latency="hours (raw telemetry)",
            notes=(
                "the portal's own JSON page methods; no account. Cite SIPH "
                "and/or the network and the date of the last update"
            ),
            title="SNIH portal",
            health=_health_probe("SNIH portal", networks.NETWORKS["snih"].api_url),
        ),
    ),
    variables=variables_for("snih"),
    citation=(
        "Subsecretaría de Recursos Hídricos, Argentina. Sistema Nacional de "
        "Información Hídrica, https://snih.hidricosargentina.gob.ar/."
    ),
    license="SNIH terms: free use with citation of SIPH and the update date",
    loader=_LOADER,
    examples=("station_datasets/plot_snih.py",),
    references=(
        "https://snih.hidricosargentina.gob.ar/",
        "https://snih.hidricosargentina.gob.ar/Terminos.aspx",
    ),
    tags=("swe", "snow depth", "stations", "argentina", "andes"),
)

CEAZA_PRODUCT = Product(
    id="ceaza-stations",
    theme="stations",
    title="CEAZA-Met snow stations (Chile)",
    description=(
        "Snow depth and, at a few sites, SWE from the CEAZA-Met network in "
        "the arid Andes of Atacama and Coquimbo. Station codes are "
        "CEAZA-<code> (CEAZA-HDC). CEAZA asks users to register a user name "
        "by e-mail; set it in CEAZA_USER."
    ),
    sources=(
        Source(
            id="ceaza",
            provider="stations",
            location=networks.NETWORKS["ceaza"].api_url,
            extent="Atacama and Coquimbo, Chile",
            temporal="~2011/present (snow)",
            latency="hours",
            notes=(
                "the CEAZA-Met web service; a registered user name is asked "
                "for (CEAZA_USER) and the data credited to CEAZA"
            ),
            title="CEAZA-Met web service",
            health=_health_probe(
                "CEAZA-Met web service", "https://www.ceazamet.cl/ws/pop_ws_doc.php"
            ),
        ),
    ),
    variables=variables_for("ceaza"),
    citation=(
        "Datos provistos por el laboratorio de meteorología CEAZA, "
        "https://www.ceazamet.cl/."
    ),
    license="No licence stated; register a user name and credit CEAZA",
    loader=_LOADER,
    examples=("station_datasets/plot_ceaza.py",),
    references=(
        "https://www.ceazamet.cl/",
        "https://www.ceazamet.cl/ws/pop_ws_doc.php",
    ),
    tags=("swe", "snow depth", "stations", "chile", "andes"),
)

PRODUCTS: tuple[Product, ...] = (
    AWDB_PRODUCT,
    CDEC_PRODUCT,
    DATABC_PRODUCT,
    NVE_PRODUCT,
    YUKON_PRODUCT,
    ALBERTA_PRODUCT,
    HYDROQUEBEC_PRODUCT,
    SLF_PRODUCT,
    SNIH_PRODUCT,
    CEAZA_PRODUCT,
)

catalog.register(*PRODUCTS, replace=True)
