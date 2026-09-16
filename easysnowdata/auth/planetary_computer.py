"""Microsoft Planetary Computer.

No account is needed; a ``PC_SDK_SUBSCRIPTION_KEY`` only lifts rate limits.
Asset hrefs are SAS-signed by the ``planetary-computer`` package.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from easysnowdata._gdal import gdal_env
from easysnowdata.auth._base import Detection, Provider

__all__ = ["PlanetaryComputerProvider", "STAC_URL"]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class PlanetaryComputerProvider(Provider):
    name = "planetary_computer"
    title = "Planetary Computer"
    optional = True
    env_vars = ("PC_SDK_SUBSCRIPTION_KEY",)
    files = ("~/.planetarycomputer/settings.env",)
    signup_url = "https://planetarycomputer.microsoft.com"
    setup_instructions = """\
The Planetary Computer needs no account. To raise the anonymous rate limits,
set PC_SDK_SUBSCRIPTION_KEY (or write it to ~/.planetarycomputer/settings.env
with `planetarycomputer configure`)."""

    def detect(self) -> Detection:
        if os.environ.get("PC_SDK_SUBSCRIPTION_KEY", "").strip():
            return Detection(True, "env:PC_SDK_SUBSCRIPTION_KEY")
        if Path("~/.planetarycomputer/settings.env").expanduser().is_file():
            return Detection(True, "file:~/.planetarycomputer/settings.env")
        return Detection(True, "anonymous")

    def ensure(self, **kwargs: Any) -> Any:
        """Return the ``planetary_computer`` module (nothing to initialise)."""
        if self._ensured is None:
            import planetary_computer  # noqa: PLC0415

            self._ensured = planetary_computer
        return self._ensured

    @property
    def sign(self) -> Any:
        """The ``pystac_client`` modifier that signs asset hrefs."""
        return self.ensure().sign_inplace

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        with gdal_env() as options:
            yield options
