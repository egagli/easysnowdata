"""easysnowdata — easily retrieve data relevant to snow science."""

__author__ = "Eric Gagliano"
__email__ = "egagli@uw.edu"

try:  # written by hatch-vcs at build/install time from the git tag
    from easysnowdata._version import __version__
except ImportError:  # pragma: no cover — source checkout without an install
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("easysnowdata")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
    del PackageNotFoundError, version
__all__ = [
    "aoi",
    "auth",
    "catalog",
    "config",
    "plotting",
    "processing",
    "providers",
    "temporal",
    "AOI",
    "parse_aoi",
    "utils",
    "remote_sensing",
    "automatic_weather_stations",
    "topography",
    "hydroclimatology",
    "authenticate_all",
    "CredentialError",
    # Phase 2 product themes
    "climate",
    "hydro",
    "land",
    "optical",
    "sar",
    "snow",
    "terrain",
]

from easysnowdata import (
    aoi,
    auth,
    automatic_weather_stations,
    catalog,
    climate,
    config,
    hydro,
    hydroclimatology,
    land,
    optical,
    plotting,
    processing,
    providers,
    remote_sensing,
    sar,
    snow,
    temporal,
    terrain,
    topography,
    utils,
)
from easysnowdata.aoi import AOI, parse_aoi
from easysnowdata.remote_sensing import authenticate_all
from easysnowdata.utils import CredentialError

auth._emit_import_summary()
