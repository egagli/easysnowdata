"""easysnowdata — easily retrieve data relevant to snow science."""

__author__ = "Eric Gagliano"
__email__ = "egagli@uw.edu"
__version__ = "0.0.22"
__all__ = [
    "utils",
    "remote_sensing",
    "automatic_weather_stations",
    "topography",
    "hydroclimatology",
    "authenticate_all",
    "CredentialError",
]

import easysnowdata.utils
import easysnowdata.remote_sensing
import easysnowdata.automatic_weather_stations
import easysnowdata.topography
import easysnowdata.hydroclimatology

from easysnowdata.utils import CredentialError
from easysnowdata.remote_sensing import authenticate_all
