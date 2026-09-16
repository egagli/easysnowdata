"""The deprecation shim mechanism (one decorator, one alias helper, one module hook)."""

from __future__ import annotations

import types
import warnings

import pytest

from easysnowdata import _deprecation
from easysnowdata._deprecation import (
    EasysnowdataDeprecationWarning,
    deprecated,
    deprecated_alias,
    deprecated_module_attrs,
)


@pytest.fixture(autouse=True)
def _fresh():
    _deprecation.reset_warnings()
    yield
    _deprecation.reset_warnings()


def new_load(aoi, *, resolution=30):
    """The new function."""
    return ("loaded", aoi, resolution)


def test_decorator_warns_once_and_forwards():
    @deprecated("easysnowdata.terrain.dem.load", since="0.1.0", remove_in="0.2.0")
    def get_copernicus_dem(bbox_input=None, resolution=30):
        """Old docstring."""
        return new_load(bbox_input, resolution=resolution)

    with pytest.warns(EasysnowdataDeprecationWarning) as record:
        assert get_copernicus_dem((0, 0, 1, 1), resolution=90) == (
            "loaded",
            (0, 0, 1, 1),
            90,
        )
    assert len(record) == 1
    message = str(record[0].message)
    assert "get_copernicus_dem is deprecated since easysnowdata 0.1.0" in message
    assert (
        "removed in 0.2.0" in message
        and "Use easysnowdata.terrain.dem.load instead." in message
    )
    assert record[0].filename == __file__  # stacklevel points at the caller
    assert issubclass(EasysnowdataDeprecationWarning, DeprecationWarning)
    assert (
        get_copernicus_dem.__doc__.startswith(".. deprecated:: 0.1.0")
        and "Old docstring." in get_copernicus_dem.__doc__
    )
    assert get_copernicus_dem.__name__ == "get_copernicus_dem"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_copernicus_dem() == ("loaded", None, 30)  # second call: silent


def test_alias_and_module_getattr():
    old = deprecated_alias(
        "easysnowdata.utils.get_water_year_start",
        new_load,
        since="0.1.0",
        extra="See the Concepts page.",
    )
    with pytest.warns(
        EasysnowdataDeprecationWarning,
        match="get_water_year_start is deprecated.*Use tests.test_deprecation.new_load instead. See the Concepts page.",
    ):
        assert old("x") == ("loaded", "x", 30)

    module = types.ModuleType("easysnowdata.oldmod")
    module.__getattr__ = deprecated_module_attrs(
        "easysnowdata.oldmod",
        {"today": ("2026-09-16", "easysnowdata.temporal.today()")},
        since="0.1.0",
    )
    with pytest.warns(
        EasysnowdataDeprecationWarning,
        match="easysnowdata.oldmod.today is deprecated since easysnowdata 0.1.0. Use easysnowdata.temporal.today\\(\\) instead.",
    ):
        assert module.today == "2026-09-16"
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        module.nope


def test_message_without_versions():
    @deprecated()
    def f():
        return 1

    with pytest.warns(
        EasysnowdataDeprecationWarning,
        match=r"^tests.test_deprecation.test_message_without_versions.<locals>.f is deprecated.$",
    ):
        f()
