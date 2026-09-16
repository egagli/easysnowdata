"""Planet Labs (PlanetScope, SkySat, basemaps) via the ``planet`` SDK 3.x.

Everything is delegated to the SDK's own auth stack: an OAuth2 user session
(``planet auth login``), a legacy API key (``PL_API_KEY`` for
``Auth.from_env()``, ``PL_AUTH_API_KEY`` for the default session), or an
OAuth2 machine-to-machine client (``PL_AUTH_CLIENT_ID`` +
``PL_AUTH_CLIENT_SECRET``); ``PL_AUTH_PROFILE`` selects a saved profile under
``~/.planet/``. :meth:`detect` only looks at those variables and files. The
SDK is not a dependency of Phase 1; it is imported lazily on :meth:`login` /
:meth:`ensure`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from easysnowdata.auth._base import Detection, Provider

__all__ = ["PlanetProvider"]


class PlanetProvider(Provider):
    name = "planet"
    title = "Planet"
    env_vars = (
        "PL_API_KEY",
        "PL_AUTH_API_KEY",
        "PL_AUTH_CLIENT_ID",
        "PL_AUTH_CLIENT_SECRET",
        "PL_AUTH_PROFILE",
    )
    files = ("~/.planet.json", "~/.planet/")
    signup_url = "https://www.planet.com/account/"
    setup_instructions = """\
Planet setup (needs a Planet account with data access, e.g. through the
Education & Research program):

    pip install planet          # or conda install -c conda-forge planet
    planet auth login           # OAuth2 in the browser; saves a session under ~/.planet/

or, in scripts and CI, set PL_API_KEY (legacy API key from
https://www.planet.com/account/) or PL_AUTH_CLIENT_ID + PL_AUTH_CLIENT_SECRET
(OAuth2 machine-to-machine client). Every order spends quota."""

    def detect(self) -> Detection:
        for var in ("PL_API_KEY", "PL_AUTH_API_KEY"):
            if os.environ.get(var, "").strip():
                return Detection(True, f"env:{var}")
        if (
            os.environ.get("PL_AUTH_CLIENT_ID", "").strip()
            and os.environ.get("PL_AUTH_CLIENT_SECRET", "").strip()
        ):
            return Detection(True, "env:PL_AUTH_CLIENT_ID")
        legacy = Path("~/.planet.json").expanduser()
        if legacy.is_file():
            return Detection(True, "file:~/.planet.json")
        sessions = Path("~/.planet").expanduser()
        if sessions.is_dir() and any(sessions.iterdir()):
            profile = os.environ.get("PL_AUTH_PROFILE", "").strip() or "default"
            return Detection(True, "file:~/.planet/", f"profile {profile}")
        return Detection(False)

    @staticmethod
    def _sdk() -> Any:
        try:
            import planet  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The `planet` SDK is not installed; run `pip install planet` "
                "or `conda install -c conda-forge planet`."
            ) from exc
        return planet

    def login(
        self, *, interactive: bool = True, persist: bool = True, **kwargs: Any
    ) -> None:
        """Run the SDK's OAuth2 user login (the same flow as ``planet auth login``)."""
        if self.detect():
            self.ensure()
            return
        if not interactive:
            raise self.error()
        try:
            planet = self._sdk()
        except ImportError as exc:
            raise self.error(str(exc)) from exc
        auth = planet.Auth.from_user_default_session()
        user_login = getattr(auth, "user_login", None)
        if user_login is None:  # pragma: no cover — older SDK
            raise self.error(
                "This `planet` SDK has no user_login(); run `planet auth login`."
            )
        user_login(allow_open_browser=True, allow_tty_prompt=True, **kwargs)
        self.reset()
        self.ensure()

    def ensure(self, **kwargs: Any) -> Any:
        """Return one ``planet.Planet`` client per process."""
        if self._ensured is not None:
            return self._ensured
        if not self.detect():
            raise self.error()
        try:
            planet = self._sdk()
        except ImportError as exc:
            raise self.error(str(exc)) from exc
        self._ensured = planet.Planet(**kwargs)
        return self._ensured
