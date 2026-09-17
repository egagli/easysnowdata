"""Norwegian Water Resources and Energy Directorate HydAPI (station data).

A free API key, read from ``NVE_API_KEY`` and sent as the ``X-API-Key`` header.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from easysnowdata.auth._base import Detection, Provider

__all__ = ["NVEProvider"]


class NVEProvider(Provider):
    name = "nve"
    title = "NVE HydAPI"
    env_vars = ("NVE_API_KEY",)
    signup_url = "https://hydapi.nve.no/UserDocumentation/"
    setup_instructions = """\
NVE HydAPI setup: request a free API key at https://hydapi.nve.no/UserDocumentation/
and set NVE_API_KEY."""

    def detect(self) -> Detection:
        if os.environ.get("NVE_API_KEY", "").strip():
            return Detection(True, "env:NVE_API_KEY")
        return Detection(False)

    def ensure(self, **kwargs: Any) -> str:
        """Return the API key."""
        key = os.environ.get("NVE_API_KEY", "").strip()
        if not key:
            raise self.error()
        self._ensured = key
        return key

    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.ensure(), "Accept": "application/json"}

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        yield {"headers": self.headers()}
