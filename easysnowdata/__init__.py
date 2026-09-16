"""easysnowdata — easily retrieve data relevant to snow science."""

__author__ = "Eric Gagliano"
__email__ = "egagli@uw.edu"
__version__ = "0.0.26"
__all__ = [
    "aoi",
    "auth",
    "catalog",
    "config",
    "AOI",
    "parse_aoi",
    "utils",
    "remote_sensing",
    "automatic_weather_stations",
    "topography",
    "hydroclimatology",
    "authenticate_all",
    "CredentialError",
]

from easysnowdata import (
    aoi,
    auth,
    automatic_weather_stations,
    catalog,
    config,
    hydroclimatology,
    remote_sensing,
    topography,
    utils,
)
from easysnowdata.aoi import AOI, parse_aoi
from easysnowdata.remote_sensing import authenticate_all
from easysnowdata.utils import CredentialError

auth._emit_import_summary()
