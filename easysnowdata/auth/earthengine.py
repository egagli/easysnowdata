"""Google Earth Engine.

Accepted credentials, in detection order: ``EARTHENGINE_TOKEN`` (a
service-account key JSON or the ``~/.config/earthengine/credentials`` OAuth
JSON, raw or base64 — the convention CI and ``geemap`` share), Application
Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``), or the credentials
file written by ``ee.Authenticate()``. A Cloud project id is required by
``ee.Initialize`` and is taken from ``EE_PROJECT_ID`` / ``EARTHENGINE_PROJECT``,
the token, or the credentials file. Initialisation happens once per process
on the high-volume endpoint. ``import ee`` is deferred so a broken Google auth
stack cannot break ``import easysnowdata``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from easysnowdata.auth._base import Detection, Provider

__all__ = [
    "EarthEngineProvider",
    "HIGH_VOLUME_URL",
    "decode_token",
    "credentials_from_token",
    "credentials_path",
]

_logger = logging.getLogger(__name__)

HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
_TOKEN_URI = "https://oauth2.googleapis.com/token"


def credentials_path() -> Path:
    """The file ``ee.Authenticate()`` writes (mirrors ``ee.oauth.get_credentials_path``)."""
    return Path("~/.config/earthengine/credentials").expanduser()


def decode_token(token: str | None) -> dict[str, Any] | None:
    """Decode ``EARTHENGINE_TOKEN`` (raw or base64-encoded JSON) into a dict.

    The original text is kept under ``"_raw"`` for ``ee.ServiceAccountCredentials``.
    """
    if token is None or not token.strip():
        return None
    raw = token.strip()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", raw), validate=True).decode()
            info = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                "EARTHENGINE_TOKEN is neither JSON nor base64-encoded JSON."
            ) from exc
    if not isinstance(info, dict):
        raise ValueError("EARTHENGINE_TOKEN must decode to a JSON object.")
    info["_raw"] = raw
    return info


def credentials_from_token(token: str | None = None) -> Any:
    """Build Earth Engine credentials from ``EARTHENGINE_TOKEN``.

    Returns ``None`` when the token is unset or empty. Raises ``ValueError``
    when it is neither a service-account key nor an Earth Engine OAuth token.
    """
    info = decode_token(os.environ.get("EARTHENGINE_TOKEN") if token is None else token)
    if info is None:
        return None
    import ee  # noqa: PLC0415
    import google.oauth2.credentials  # noqa: PLC0415

    if info.get("type") == "service_account":
        return ee.ServiceAccountCredentials(info["client_email"], key_data=info["_raw"])
    if "refresh_token" in info:
        # Newer credentials files omit client id/secret and rely on Earth
        # Engine's default OAuth client, as ee.oauth does.
        return google.oauth2.credentials.Credentials(
            None,
            token_uri=info.get("token_uri", _TOKEN_URI),
            client_id=info.get("client_id", ee.oauth.CLIENT_ID),
            client_secret=info.get("client_secret", ee.oauth.CLIENT_SECRET),
            refresh_token=info["refresh_token"],
            scopes=info.get("scopes"),
            quota_project_id=info.get("project"),
        )
    raise ValueError(
        "EARTHENGINE_TOKEN is neither a service-account key nor an Earth Engine "
        "OAuth token (expected 'type': 'service_account' or a 'refresh_token')."
    )


class EarthEngineProvider(Provider):
    name = "earthengine"
    title = "Earth Engine"
    env_vars = (
        "EARTHENGINE_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "EE_PROJECT_ID",
        "EARTHENGINE_PROJECT",
    )
    files = ("~/.config/earthengine/credentials",)
    signup_url = "https://earthengine.google.com"
    setup_instructions = """\
Google Earth Engine setup (once, in a terminal or notebook):

    import ee
    ee.Authenticate()                    # opens a browser; writes ~/.config/earthengine/credentials
    earthengine set_project <project>    # or set EE_PROJECT_ID; Earth Engine requires a Cloud project

In scripts and CI set EARTHENGINE_TOKEN to a service-account key JSON (raw or
base64), or point GOOGLE_APPLICATION_CREDENTIALS at one, and set EE_PROJECT_ID
when the key does not carry a project.

Sign up at https://earthengine.google.com"""

    def __init__(self) -> None:
        super().__init__()
        self._init_kwargs: dict[str, Any] | None = None

    # -- detection -------------------------------------------------------------

    def detect(self) -> Detection:
        if os.environ.get("EARTHENGINE_TOKEN", "").strip():
            return Detection(True, "env:EARTHENGINE_TOKEN")
        adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if adc and Path(adc).expanduser().is_file():
            return Detection(True, "adc", adc)
        path = credentials_path()
        if path.is_file():
            return Detection(True, "file", str(path))
        return Detection(False)

    def project_id(self, token_info: dict[str, Any] | None = None) -> str | None:
        """Resolve the Cloud project: env → token → credentials file → ``None``."""
        for var in ("EE_PROJECT_ID", "EARTHENGINE_PROJECT"):
            value = os.environ.get(var, "").strip()
            if value:
                return value
        if token_info is None:
            try:
                token_info = decode_token(os.environ.get("EARTHENGINE_TOKEN"))
            except ValueError:
                token_info = None
        if token_info:
            value = token_info.get("project") or token_info.get("project_id")
            if value:
                return str(value)
        path = credentials_path()
        if path.is_file():
            try:
                stored = json.loads(path.read_text())
                if isinstance(stored, dict) and stored.get("project"):
                    return str(stored["project"])
            except (OSError, ValueError):
                pass
        return None

    # -- login / ensure --------------------------------------------------------

    def login(
        self, *, interactive: bool = True, persist: bool = True, **kwargs: Any
    ) -> None:
        """Run ``ee.Authenticate()`` (browser or device flow) and initialise."""
        if not self.detect():
            if not interactive:
                raise self.error()
            import ee  # noqa: PLC0415

            ee.Authenticate()
        self.ensure(**kwargs)

    def ensure(self, **kwargs: Any) -> None:
        """Initialise Earth Engine once, on the high-volume endpoint.

        Keyword arguments are forwarded to ``ee.Initialize`` (``project``,
        ``opt_url``/``url``…); passing any forces a re-initialisation.
        """
        import ee  # noqa: PLC0415

        if not kwargs and self._ensured and self._is_initialized(ee):
            return
        detection = self.detect()
        if not detection:
            raise self.error()

        init: dict[str, Any] = dict(kwargs)
        if "url" not in init:
            init.setdefault("opt_url", HIGH_VOLUME_URL)
        token_info = None
        if detection.how == "env:EARTHENGINE_TOKEN":
            try:
                token_info = decode_token(os.environ["EARTHENGINE_TOKEN"])
            except ValueError as exc:
                raise self.error(
                    "EARTHENGINE_TOKEN could not be decoded.", cause=str(exc)
                ) from exc
            init.setdefault("credentials", credentials_from_token(token_info["_raw"]))  # type: ignore[index]
        if init.get("project") is None:
            project = self.project_id(token_info)
            if project is None and detection.how != "adc":
                raise self.error(
                    "Earth Engine needs a Cloud project id and none was found "
                    "(set EE_PROJECT_ID, run `earthengine set_project`, or use a "
                    "service-account key)."
                )
            init["project"] = project
        _logger.debug(
            "Initialising Earth Engine (how=%s, project=%s).",
            detection.how,
            init.get("project"),
        )
        ee.Initialize(**init)
        self._ensured = True
        self._init_kwargs = {k: v for k, v in init.items() if k != "credentials"}

    @staticmethod
    def _is_initialized(ee: Any) -> bool:
        try:
            return bool(ee.data.is_initialized())
        except Exception:  # pragma: no cover — very old ee
            return True

    def xee_init_kwargs(self) -> dict[str, Any]:
        """``ee_init_kwargs`` for ``xarray.open_dataset(engine="ee")`` on Dask workers."""
        kwargs = dict(self._init_kwargs or {})
        kwargs.setdefault("opt_url", HIGH_VOLUME_URL)
        if kwargs.get("project") is None:
            kwargs["project"] = self.project_id()
        return kwargs

    def reset(self) -> None:
        super().reset()
        self._init_kwargs = None
