"""Fixtures for the Phase 2a (static products) tests.

The offline tier runs with sockets blocked, so every test here reads either a
tiny local fixture written by :mod:`tests.static.make_fixtures` or a
pytest-recording cassette under ``tests/static/cassettes``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture(scope="session")
def static_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Generate the local fixtures once per session (in a subprocess)."""
    target = tmp_path_factory.mktemp("static-fixtures")
    script = Path(__file__).parent / "make_fixtures.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}
    result = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    paths = {}
    for line in result.stdout.splitlines():
        name, _, path = line.partition(": ")
        paths[name] = Path(path)
    return paths


@pytest.fixture
def dem_items(static_fixtures) -> object:
    """An ``ItemCollection`` with one item pointing at the local DEM fixture."""
    import pystac

    item = json.loads(static_fixtures["dem_item"].read_text())
    return pystac.ItemCollection([pystac.Item.from_dict(item)])
