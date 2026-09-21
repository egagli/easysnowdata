"""Basic package-level tests — import, version, and public API surface."""

from __future__ import annotations

import importlib

import easysnowdata

REMOVED_IN_0_3 = (
    "remote_sensing",
    "hydroclimatology",
    "topography",
    "automatic_weather_stations",
    "utils",
    "_deprecation",
)


def test_package_imports():
    assert hasattr(easysnowdata, "__version__")
    for name in ("aoi", "auth", "catalog", "processing", "plotting", "stations"):
        assert hasattr(easysnowdata, name)


def test_version_is_a_valid_version():
    """hatch-vcs derives it from the git tag: 0.3.0 on a tag, 0.3.0.devN+g… between tags."""
    from packaging.version import Version

    assert isinstance(easysnowdata.__version__, str)
    parsed = Version(easysnowdata.__version__)
    assert parsed.release[:2] == (0, 0) or parsed.release >= (0, 1)


def test_public_api_surface():
    """Every name in each subpackage's ``__all__`` resolves."""
    for name in easysnowdata.__all__:
        obj = getattr(easysnowdata, name)
        for member in getattr(obj, "__all__", ()):
            assert hasattr(obj, member), f"{name}.{member} missing"


def test_the_0_0_x_shims_are_gone():
    """The 0.2 deprecation shims were removed in 0.3, as their warnings promised."""
    for name in REMOVED_IN_0_3:
        assert not hasattr(easysnowdata, name)
        try:
            importlib.import_module(f"easysnowdata.{name}")
        except ImportError:
            continue
        raise AssertionError(f"easysnowdata.{name} still imports")


def test_import_registers_the_rio_accessor():
    """Every loader uses ``.rio``; the accessor must exist after a bare import.

    Until 0.3 the only ``import rioxarray`` in the package sat in the deleted
    ``topography`` shim, so this is checked in a fresh interpreter where no
    test module has imported rioxarray first.
    """
    import os
    import subprocess
    import sys

    code = (
        "import easysnowdata, xarray as xr; "
        "xr.DataArray([[1.0]], dims=('y', 'x')).rio.crs"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "EASYSNOWDATA_QUIET": "1"},
        timeout=120,
    )
