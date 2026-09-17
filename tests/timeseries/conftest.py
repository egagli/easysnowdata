"""Fixtures for the Phase 2b (time-series products) tests.

Everything here is offline: synthetic fixtures generated once per session by
``tests/timeseries/fixtures/make_fixtures.py`` (in a subprocess — see the
import-order note in that module) and pytest-recording cassettes under
``tests/timeseries/cassettes/``. Cassettes are recorded unsigned; refresh with::

    pixi run -e dev pytest tests/timeseries -m recorded --record-mode=rewrite
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RAINIER = (-121.94, 46.72, -121.54, 46.99)

_HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def vcr_config(vcr_config) -> dict:
    """This directory's VCR settings, on top of the shared ones.

    ``allow_playback_repeats`` matters for Planetary Computer: signing asks
    ``/api/sas/v1/token/...`` for a token and caches it per process, so the
    number of token requests depends on what ran before. Replaying the same
    recorded response for each of them keeps the cassettes independent of
    test order.
    """
    return {**vcr_config, "allow_playback_repeats": True}


@pytest.fixture(scope="module")
def vcr_cassette_dir(request) -> str:
    """Keep this directory's cassettes next to its tests."""
    return str(_HERE / "cassettes" / request.module.__name__.rsplit(".", 1)[-1])


@pytest.fixture(scope="session")
def ts_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Synthetic fixtures for every time-series product (paths keyed by name)."""
    target = tmp_path_factory.mktemp("timeseries-fixtures")
    script = _HERE / "fixtures" / "make_fixtures.py"
    env = {**os.environ, "PYTHONPATH": str(_HERE.parent.parent)}
    result = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    paths: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        name, _, path = line.partition(": ")
        if path:
            paths[name] = Path(path)
    return paths


@pytest.fixture
def no_network(monkeypatch):
    """Make any accidental provider call fail loudly."""
    from easysnowdata import providers

    def boom(*args, **kwargs):  # pragma: no cover — only runs on a bug
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(providers.stac, "open_catalog", boom)
    monkeypatch.setattr(providers.earthdata, "ensure", boom)
    monkeypatch.setattr(providers.gee, "ensure", boom)
    return boom


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """No credentials of any provider are visible."""
    from easysnowdata import auth

    for var in (
        "EARTHDATA_TOKEN",
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
        "EARTHENGINE_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "NETRC",
        "PL_API_KEY",
        "PL_AUTH_API_KEY",
        "PL_AUTH_CLIENT_ID",
        "PL_AUTH_CLIENT_SECRET",
        "PL_AUTH_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    auth.reset()
    yield
    auth.reset()


@pytest.fixture
def fake_credentials(monkeypatch):
    """Every provider *detects* as configured; nothing is initialised."""
    from easysnowdata import auth

    monkeypatch.setenv("EARTHDATA_TOKEN", "fake-token")
    monkeypatch.setenv("EARTHENGINE_TOKEN", '{"type": "service_account"}')
    monkeypatch.setenv("EE_PROJECT_ID", "fake-project")
    monkeypatch.setenv("PL_API_KEY", "PLAKfake")
    for name in auth.PROVIDERS:
        monkeypatch.setattr(auth.PROVIDERS[name], "ensure", lambda **kw: True)
    auth.reset()
    yield
    auth.reset()
