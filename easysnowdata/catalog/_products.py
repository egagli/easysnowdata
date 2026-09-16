"""Catalog entries that have not moved to a theme module yet.

Phase 2 migrated every product to its own module under
:mod:`easysnowdata.terrain`, :mod:`~easysnowdata.land`, :mod:`~easysnowdata.snow`,
:mod:`~easysnowdata.hydro`, :mod:`~easysnowdata.climate`, :mod:`~easysnowdata.optical`
and :mod:`~easysnowdata.sar`, each registering its own entry on import. The
station archive is the last holdout: it is replaced wholesale by the network
clients in Phase 3 (§9), so it keeps its Phase 1 entry until then.

Class tables live with whatever consumes them: next to a product's catalog
entry when only that entry reads it, or in :mod:`easysnowdata.processing` when
a processing function does (``processing.optical.SCL_CLASSES``,
``processing.snow.MOD10A2_CLASSES``).
"""

from __future__ import annotations

from functools import partial

from easysnowdata.catalog import health
from easysnowdata.catalog._models import Probe, Product, Source, Variable

_GITHUB_STATIONS = "https://github.com/egagli/snotel_ccss_stations"


PRODUCTS: tuple[Product, ...] = (
    Product(
        id="snotel-ccss-stations",
        theme="stations",
        title="SNOTEL and CCSS station observations",
        description=(
            "Daily SWE, snow depth, precipitation and temperature from NRCS SNOTEL and "
            "California CCSS stations, served from the frozen egagli/snotel_ccss_stations "
            "archive. Superseded by the live network clients in Phase 3 (§9)."
        ),
        sources=(
            Source(
                id="github-archive",
                provider="vector_http",
                location=_GITHUB_STATIONS,
                extent="western US",
                temporal="varies by station / 2024",
                latency="frozen archive",
                notes="GeoJSON station list plus one CSV per station",
                title="GitHub archive",
                health=(
                    Probe(
                        "SNOTEL/CCSS station list (GitHub)",
                        partial(
                            health.http_first_byte,
                            f"{_GITHUB_STATIONS}/raw/main/all_stations.geojson",
                        ),
                    ),
                    Probe(
                        "SNOTEL/CCSS station CSV (GitHub)",
                        partial(
                            health.http_first_byte,
                            "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/679_WA_SNTL.csv",
                        ),
                    ),
                ),
            ),
        ),
        variables=(
            Variable("WTEQ", units="m", long_name="snow water equivalent"),
            Variable("SNWD", units="m", long_name="snow depth"),
            Variable("PRCPSA", units="m", long_name="precipitation accumulation"),
            Variable("TAVG", units="degC", long_name="mean daily air temperature"),
        ),
        citation="USDA NRCS National Water and Climate Center (SNOTEL); California Cooperative Snow Surveys (CCSS).",
        license="Public domain (US government data)",
        loader="easysnowdata.automatic_weather_stations.StationCollection",
        tags=("swe", "snow depth", "stations", "snotel"),
    ),
)
