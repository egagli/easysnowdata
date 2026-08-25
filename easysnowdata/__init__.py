"""easysnowdata — easily retrieve data relevant to snow science."""

__author__ = "Eric Gagliano"
__email__ = "egagli@uw.edu"
__version__ = "0.0.25"
__all__ = [
    "utils",
    "remote_sensing",
    "automatic_weather_stations",
    "topography",
    "hydroclimatology",
    "authenticate_all",
    "CredentialError",
]

from easysnowdata import (
    automatic_weather_stations,
    hydroclimatology,
    remote_sensing,
    topography,
    utils,
)
from easysnowdata.remote_sensing import authenticate_all
from easysnowdata.utils import CredentialError
