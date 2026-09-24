"""NASA Earthdata Login (EDL) via ``earthaccess``.

Detection order (fixed, §5.2): ``EARTHDATA_TOKEN`` → ``EARTHDATA_USERNAME`` +
``EARTHDATA_PASSWORD`` → a ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``.
``earthaccess`` ≥ 0.16 no longer logs in implicitly, so :meth:`ensure` calls
``earthaccess.login(strategy=...)`` explicitly before any ``open()`` or
``download()``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import netrc
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from easysnowdata import config
from easysnowdata._gdal import gdal_env
from easysnowdata.auth._base import Detection, Provider

__all__ = ["EarthdataProvider", "NETRC_HOST", "token_expiry", "token_is_valid"]

_logger = logging.getLogger(__name__)

NETRC_HOST = "urs.earthdata.nasa.gov"
#: Treat a token this close to its expiry as expired: a long read started a
#: few seconds before ``exp`` would fail halfway through.
EXPIRY_SKEW_S = 60


def token_expiry(token: str) -> datetime | None:
    """When an Earthdata Login user token expires, read from the token itself.

    EDL user tokens are signed JWTs whose payload carries ``exp`` (seconds
    since the epoch). The payload is only base64url-decoded, not verified:
    this answers "has it expired?", and a token that is forged or revoked
    still fails at the first authenticated request. ``None`` when *token* is
    not a JWT with an ``exp`` claim.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return datetime.fromtimestamp(float(claims["exp"]), UTC)
    except (ValueError, KeyError, TypeError):
        return None


def token_is_valid(token: str, *, now: datetime | None = None) -> bool | None:
    """Whether *token* is still inside its lifetime, without a network request.

    ``True``/``False`` from the token's ``exp`` claim (with a
    :data:`EXPIRY_SKEW_S` margin); ``None`` when that cannot be read, in which
    case the token is given the benefit of the doubt rather than blocking a
    read.

    The check used to ask URS (``GET /api/users/tokens`` with the token as a
    bearer). That endpoint manages tokens and only accepts a username and
    password, so it answered 401 to every token, valid or not, and a
    token-only setup was always reported as expired.
    """
    expiry = token_expiry(token)
    if expiry is None:
        return None
    current = now or datetime.now(UTC)
    return (expiry - current).total_seconds() > EXPIRY_SKEW_S


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


def _module_auth(earthaccess: Any) -> Any:
    """earthaccess's process-wide ``Auth`` (``_auth`` since 0.18, ``__auth__`` before)."""
    return getattr(earthaccess, "_auth", None) or getattr(earthaccess, "__auth__", None)


def _module_store(earthaccess: Any) -> Any:
    return getattr(earthaccess, "_store", None) or getattr(
        earthaccess, "__store__", None
    )


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
  - EARTHDATA_USERNAME + EARTHDATA_PASSWORD  (recommended for CI: earthaccess mints and
                                              renews its own token from these, so nothing
                                              expires)
  - EARTHDATA_TOKEN                          (generate at urs.earthdata.nasa.gov; user tokens
                                              expire after ~60 days, and an expired one is
                                              ignored in favour of the credentials above
                                              when they are also set)

A token alone does not reach NSIDC's on-premises archive (the RGI glacier
outlines): it answers with the login page. Use the username and password
(or the netrc) for those.

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

        auth = _module_auth(earthaccess)
        store = _module_store(earthaccess)
        if (
            auth is not None
            and getattr(auth, "authenticated", False)
            and store is not None
        ):
            self._ensured = auth
            return auth

        self._drop_expired_token()
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
                _module_store(earthaccess) is not None
            ):
                self._ensured = auth
                return auth
            last_exc = None
            break
        raise self.error(
            f"NASA Earthdata login failed (strategy {strategy!r}).",
            cause=str(last_exc) if last_exc is not None else None,
        ) from last_exc

    def _drop_expired_token(self) -> None:
        """Stop using an expired ``EARTHDATA_TOKEN`` when there is a fallback.

        earthaccess trusts a token from the environment without checking it,
        so an expired one (they last ~60 days) only surfaces as a 401 halfway
        through a read. With a username and password also configured — the
        CI secrets, or a netrc entry — the token is removed from this
        process's environment and the login proceeds with those, which makes
        earthaccess mint a fresh token itself. Without a fallback the token is
        left in place and the error names the cause.
        """
        token = os.environ.get("EARTHDATA_TOKEN", "").strip()
        if not token:
            return
        valid = token_is_valid(token)
        if valid is not False:
            return
        expiry = token_expiry(token)
        when = f" on {expiry:%Y-%m-%d %H:%M} UTC" if expiry else ""
        fallback = (
            os.environ.get("EARTHDATA_USERNAME")
            and os.environ.get("EARTHDATA_PASSWORD")
        ) or netrc_has_edl()
        if not fallback:
            raise self.error(
                f"EARTHDATA_TOKEN expired{when} (user tokens last about 60 days; "
                "generate a new one at urs.earthdata.nasa.gov) and no "
                "EARTHDATA_USERNAME/EARTHDATA_PASSWORD or netrc entry is configured "
                "to fall back on."
            )
        _logger.warning(
            "EARTHDATA_TOKEN expired%s; logging in with the username and password "
            "instead, which mints a new token.",
            when,
        )
        os.environ.pop("EARTHDATA_TOKEN", None)
        self._rejected_token = token

    # -- GDAL environment ------------------------------------------------------

    def gdal_options(self) -> dict[str, Any]:
        """GDAL options for ``/vsicurl`` reads of EDL-protected COGs.

        With a username and password — a netrc entry, or the
        ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` variables — GDAL follows
        the EDL redirect dance: the DAAC bounces the request to URS, libcurl
        answers with the netrc credentials, and the session cookie lands in a
        jar kept in the easysnowdata cache directory (not the home directory).
        That is the one method every DAAC accepts. A bearer token is used only
        when it is the sole credential: ASF's datapool hands a request across
        three hosts and libcurl does not carry an ``Authorization`` header to
        another host, so a token alone gets a 401 there (the CI runners, which
        have no netrc, hit exactly this).
        """
        cookies = config.cache_dir("gdal") / "earthdata-cookies.txt"
        options: dict[str, Any] = {
            "GDAL_HTTP_COOKIEFILE": str(cookies),
            "GDAL_HTTP_COOKIEJAR": str(cookies),
            "GDAL_HTTP_UNSAFESSL": None,
        }
        netrc_file = self._netrc_for_gdal()
        if netrc_file is not None:
            options["GDAL_HTTP_NETRC"] = "YES"
            options["GDAL_HTTP_NETRC_FILE"] = str(netrc_file)
            return {k: v for k, v in options.items() if v is not None}
        token = os.environ.get("EARTHDATA_TOKEN", "").strip() or self._session_token()
        if token:
            options["GDAL_HTTP_AUTH"] = "BEARER"
            options["GDAL_HTTP_BEARER"] = token
        else:
            options["GDAL_HTTP_NETRC"] = "YES"
        return {k: v for k, v in options.items() if v is not None}

    @staticmethod
    def _netrc_for_gdal() -> Path | None:
        """The netrc file GDAL should read, or ``None`` when there is no password.

        A netrc file with a URS entry is used as it is. When the username and
        password come from the environment instead (the CI secrets), a
        one-entry netrc is written to the easysnowdata cache directory with
        owner-only permissions and kept current, because libcurl can only take
        credentials for the URS redirect from a file.
        """
        path = netrc_path()
        if path is not None and netrc_has_edl():
            return path
        user = os.environ.get("EARTHDATA_USERNAME", "").strip()
        password = os.environ.get("EARTHDATA_PASSWORD", "")
        if not (user and password):
            return None
        target = config.cache_dir("gdal") / "earthdata-netrc"
        text = f"machine {NETRC_HOST} login {user} password {password}\n"
        try:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                target.touch(mode=0o600)
                target.write_text(text, encoding="utf-8")
            target.chmod(0o600)
        except OSError as exc:  # pragma: no cover — a read-only cache dir
            _logger.warning("Could not write %s for GDAL: %s", target, exc)
            return None
        return target

    @staticmethod
    def _session_token() -> str:
        """The bearer token earthaccess obtained from a username/password login, if any."""
        try:
            import earthaccess  # noqa: PLC0415

            auth = _module_auth(earthaccess)
            if auth is not None and getattr(auth, "authenticated", False):
                token = getattr(auth, "token", None) or {}
                return str(token.get("access_token", "")).strip()
        except Exception:  # noqa: BLE001 — GDAL falls back to the netrc dance
            pass
        return ""

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        with gdal_env(**self.gdal_options()) as options:
            yield options
