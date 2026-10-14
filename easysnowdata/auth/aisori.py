"""RIHMI-WDC Web АИСОРИ-М (Russian route snow surveys and daily snow cover).

A free account at http://aisori-m.meteo.ru/waisori/registration.xhtml, read
from ``AISORI_USERNAME`` and ``AISORI_PASSWORD`` and used to log in to the
site's export form.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from easysnowdata.auth._base import Detection, Provider

__all__ = ["AisoriProvider"]


class AisoriProvider(Provider):
    name = "aisori"
    title = "RIHMI АИСОРИ-М"
    env_vars = ("AISORI_USERNAME", "AISORI_PASSWORD")
    signup_url = "http://aisori-m.meteo.ru/waisori/registration.xhtml"
    setup_instructions = """\
RIHMI АИСОРИ-М setup: register for a free account at
http://aisori-m.meteo.ru/waisori/registration.xhtml (no confirmation step),
then set AISORI_USERNAME and AISORI_PASSWORD."""

    def detect(self) -> Detection:
        user = os.environ.get("AISORI_USERNAME", "").strip()
        password = os.environ.get("AISORI_PASSWORD", "").strip()
        if user and password:
            return Detection(True, "env:AISORI_USERNAME/AISORI_PASSWORD")
        return Detection(False)

    def ensure(self, **kwargs: Any) -> tuple[str, str]:
        """Return ``(username, password)``."""
        user = os.environ.get("AISORI_USERNAME", "").strip()
        password = os.environ.get("AISORI_PASSWORD", "").strip()
        if not (user and password):
            raise self.error()
        self._ensured = (user, password)
        return user, password

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        user, password = self.ensure()
        yield {"username": user, "password": password}
