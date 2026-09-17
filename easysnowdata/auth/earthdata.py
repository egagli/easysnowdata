"""NASA Earthdata Login (EDL) via ``earthaccess``.

Detection order (fixed, §5.2): ``EARTHDATA_TOKEN`` → ``EARTHDATA_USERNAME`` +
``EARTHDATA_PASSWORD`` → a ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``.
``earthaccess`` ≥ 0.16 no longer logs in implicitly, so :meth:`ensure` calls
``earthaccess.login(strategy=...)`` explicitly before any ``open()`` or
``download()``.
"""

from __future__ import annotations

import contextlib
import logging
import netrc
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from easysnowdata import config
from easysnowdata._gdal import gdal_env
from easysnowdata.auth._base import Detection, Provider

__all__ = ["EarthdataProvider", "NETRC_HOST"]

_logger = logging.getLogger(__name__)

NETRC_HOST = "urs.earthdata.nasa.gov"


def netrc_path() -> Path | None:
    """Return the netrc file earthaccess would read, if one exists."""
    override = os.environ.get("NETRC")
    candidates = [Path(override).expanduser()] if override else []
    candidates += [Path("~/.netrc").expanduser(), Path("~/_netrc").expanduser()]
    for path in candidates:
        if path.is_file():
            return path
    return None


def netrc_has_edl() -> bool:
    """``True`` when a readable netrc file has a ``urs.earthdata.nasa.gov`` entry."""
    path = netrc_path()
    if path is None:
        return False
    try:
        return netrc.netrc(str(path)).authenticators(NETRC_HOST) is not None
    except (netrc.NetrcParseError, OSError):
        return False


class EarthdataProvider(Provider):
    name = "earthdata"
    title = "NASA Earthdata"
    env_vars = ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    files = ("~/.netrc",)
    signup_url = "https://urs.earthdata.nasa.gov"
    setup_instructions = """\
NASA Earthdata Login setup (once):

    import earthaccess
    earthaccess.login(persist=True)   # prompts; saves to ~/.netrc

or, in scripts and CI, set one of:
  - EARTHDATA_TOKEN                        (recommended; generate at urs.earthdata.nasa.gov,
                                            user tokens expire after ~60 days)
  - EARTHDATA_USERNAME + EARTHDATA_PASSWORD

Register for a free account at https://urs.earthdata.nasa.gov"""

    # -- detection -------------------------------------------------------------

    def detect(self) -> Detection:
        if os.environ.get("EARTHDATA_TOKEN", "").strip():
            return Detection(True, "env:EARTHDATA_TOKEN")
        if os.environ.get("EARTHDATA_USERNAME") and os.environ.get(
            "EARTHDATA_PASSWORD"
        ):
            return Detection(True, "env:EARTHDATA_USERNAME")
        if netrc_has_edl():
            return Detection(True, "netrc", str(netrc_path()))
        return Detection(False)

    def strategy(self) -> str | None:
        """The ``earthaccess.login(strategy=...)`` matching :meth:`detect`."""
        how = self.detect().how
        if how is None:
            return None
        return "environment" if how.startswith("env:") else "netrc"

    # -- login / ensure --------------------------------------------------------

    def login(
        self, *, interactive: bool = True, persist: bool = True, **kwargs: Any
    ) -> Any:
        """Log in, prompting for username/password when nothing is configured."""
        if self.detect():
            return self.ensure()
        if not interactive:
            raise self.error()
        import earthaccess  # noqa: PLC0415

        auth = earthaccess.login(strategy="interactive", persist=persist)
        if not getattr(auth, "authenticated", False):
            raise self.error("Earthdata Login rejected the interactive credentials.")
        self._ensured = auth
        return auth

    def ensure(self, **kwargs: Any) -> Any:
        """Return an authenticated ``earthaccess.Auth``, logging in once.

        With ``EARTHDATA_TOKEN`` earthaccess marks the ``Auth`` authenticated
        before it creates the ``Store``; a failed Store leaves ``__store__``
        ``None`` and the next ``open()`` fails with a bare ``AttributeError``,
        so both are checked. One retry covers transient connection errors.
        """
        import earthaccess  # noqa: PLC0415

        auth = getattr(earthaccess, "__auth__", None)
        store = getattr(earthaccess, "__store__", None)
        if (
            auth is not None
            and getattr(auth, "authenticated", False)
            and store is not None
        ):
            self._ensured = auth
            return auth

        strategy = self.strategy()
        if strategy is None:
            raise self.error()

        _logger.debug("Logging in to NASA Earthdata with strategy %r.", strategy)
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                auth = earthaccess.login(strategy=strategy)
            except Exception as exc:
                last_exc = exc
                if auth is not None:
                    auth.authenticated = False  # do not leave half-initialised state
                if attempt == 1:
                    _logger.warning(
                        "Earthdata login attempt failed (%s); retrying.", exc
                    )
                    time.sleep(2)
                continue
            if getattr(auth, "authenticated", False) and (
                getattr(earthaccess, "__store__", None) is not None
            ):
                self._ensured = auth
                return auth
            last_exc = None
            break
        raise self.error(
            f"NASA Earthdata login failed (strategy {strategy!r}).",
            cause=str(last_exc) if last_exc is not None else None,
        ) from last_exc

    # -- GDAL environment ------------------------------------------------------

    def gdal_options(self) -> dict[str, Any]:
        """GDAL options for ``/vsicurl`` reads of EDL-protected COGs.

        A bearer token is sent directly; otherwise GDAL follows the EDL
        redirect dance with the netrc entry and a cookie jar kept in the
        easysnowdata cache directory (not the home directory).
        """
        cookies = config.cache_dir("gdal") / "earthdata-cookies.txt"
        options: dict[str, Any] = {
            "GDAL_HTTP_COOKIEFILE": str(cookies),
            "GDAL_HTTP_COOKIEJAR": str(cookies),
            "GDAL_HTTP_UNSAFESSL": None,
        }
        token = os.environ.get("EARTHDATA_TOKEN", "").strip()
        if token:
            options["GDAL_HTTP_AUTH"] = "BEARER"
            options["GDAL_HTTP_BEARER"] = token
        else:
            options["GDAL_HTTP_NETRC"] = "YES"
            path = netrc_path()
            if path is not None and os.environ.get("NETRC"):
                options["GDAL_HTTP_NETRC_FILE"] = str(path)
        return {k: v for k, v in options.items() if v is not None}

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        with gdal_env(**self.gdal_options()) as options:
            yield options
