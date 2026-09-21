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
    # Core
    "aoi",
    "auth",
    "catalog",
    "config",
    "temporal",
    "AOI",
    "parse_aoi",
    "CredentialError",
    # Product themes
    "stations",
    "snow",
    "sar",
    "optical",
    "terrain",
    "land",
    "hydro",
    "climate",
    # Processing, plotting, providers
    "processing",
    "plotting",
    "providers",
]

from easysnowdata import (
    aoi,
    auth,
    catalog,
    climate,
    config,
    hydro,
    land,
    optical,
    plotting,
    processing,
    providers,
    sar,
    snow,
    stations,
    temporal,
    terrain,
)
from easysnowdata.aoi import AOI, parse_aoi
from easysnowdata.auth import CredentialError

auth._emit_import_summary()
