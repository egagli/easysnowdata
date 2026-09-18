"""The conda-forge requirements generator.

The feedstock's dependency list is maintained by hand and had drifted by
eighteen packages before anyone looked. These tests hold the two things that
made the drift consequential: the host section has to match the real build
backend, and the run section has to be the whole of pyproject's dependency
list rather than whatever was true when someone last edited the recipe.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def feedstock():
    spec = importlib.util.spec_from_file_location(
        "feedstock_requirements", REPO_ROOT / "scripts" / "feedstock_requirements.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["feedstock_requirements"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_translates_a_pypi_requirement_to_a_conda_match_spec(feedstock):
    assert feedstock.requirement("zarr>=3.1") == "zarr >=3.1"
    assert feedstock.requirement("pooch") == "pooch"


def test_renames_only_where_conda_forge_differs(feedstock):
    # The unsuffixed packages pull in a GUI toolkit and distributed.
    assert feedstock.requirement("matplotlib") == "matplotlib-base"
    assert feedstock.requirement("dask") == "dask-core"


def test_every_runtime_dependency_reaches_the_run_section(feedstock, pyproject, capsys):
    feedstock.main()
    printed = capsys.readouterr().out
    for dependency in pyproject["project"]["dependencies"]:
        assert f"- {feedstock.requirement(dependency)}\n" in printed, dependency


def test_host_section_matches_the_real_build_backend(feedstock, pyproject, capsys):
    # The recipe carried setuptools long after the backend became hatchling,
    # which fails outright under `pip install --no-build-isolation`.
    feedstock.main()
    host = capsys.readouterr().out.split("  run:")[0]
    for build_dependency in pyproject["build-system"]["requires"]:
        assert f"- {feedstock.requirement(build_dependency)}\n" in host
    assert "setuptools" not in host


def test_python_floor_follows_requires_python(feedstock, pyproject, capsys):
    feedstock.main()
    printed = capsys.readouterr().out
    floor = pyproject["project"]["requires-python"].lstrip(">=")
    assert f'{{% set python_min = "{floor}" %}}' in printed
