"""Basic package-level tests — import, version, and public API surface."""

from __future__ import annotations


def test_package_imports():
    import easysnowdata

    assert hasattr(easysnowdata, "__version__")
    assert hasattr(easysnowdata, "utils")
    assert hasattr(easysnowdata, "remote_sensing")
    assert hasattr(easysnowdata, "automatic_weather_stations")
    assert hasattr(easysnowdata, "topography")
    assert hasattr(easysnowdata, "hydroclimatology")


def test_version_is_a_valid_version():
    """hatch-vcs derives it from the git tag: 0.0.27 on a tag, 0.0.27.devN+g… between tags."""
    from packaging.version import Version

    import easysnowdata

    assert isinstance(easysnowdata.__version__, str)
    parsed = Version(easysnowdata.__version__)
    assert parsed.release[:2] == (0, 0) or parsed.release >= (0, 1)


def test_public_api_surface():
    """Verify __all__ entries are importable from each module."""
    from easysnowdata import (
        automatic_weather_stations,
        hydroclimatology,
        remote_sensing,
        topography,
        utils,
    )

    for name in utils.__all__:
        assert hasattr(utils, name), f"utils.{name} missing"

    for name in automatic_weather_stations.__all__:
        assert hasattr(automatic_weather_stations, name), f"aws.{name} missing"

    for name in hydroclimatology.__all__:
        assert hasattr(hydroclimatology, name), f"hydroclimatology.{name} missing"

    for name in remote_sensing.__all__:
        assert hasattr(remote_sensing, name), f"remote_sensing.{name} missing"

    for name in topography.__all__:
        assert hasattr(topography, name), f"topography.{name} missing"
