"""Runtime configuration: quiet mode, interactivity, cache directory, compute region.

Everything here is cheap and network-free unless explicitly asked
(:func:`region` with ``probe=True``), so it is safe to call at import time
(design contract §2.9).

Environment variables
---------------------
``EASYSNOWDATA_QUIET``
    ``1``/``true`` silences the one-line credential summary printed on import.
``EASYSNOWDATA_CACHE_DIR``
    Root of the download cache (default: the platform user cache directory).
``EASYSNOWDATA_REGION``
    Overrides compute-region detection, e.g. ``us-west-2`` on a Coiled or
    Kubernetes worker where the EC2 metadata service is blocked, or ``local``
    to force HTTPS access.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "quiet",
    "is_interactive",
    "cache_dir",
    "Region",
    "region",
    "access_mode",
    "DIRECT_S3_REGION",
]

_logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
DIRECT_S3_REGION = "us-west-2"  # where NASA Earthdata's cloud archive lives
IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_REGION_URL = "http://169.254.169.254/latest/meta-data/placement/region"
IMDS_TIMEOUT = 0.2  # seconds; the plan's 200 ms budget
_DMI_DIR = Path("/sys/devices/virtual/dmi/id")
_AWS_EXECUTION_HINTS = (
    "AWS_EXECUTION_ENV",
    "ECS_CONTAINER_METADATA_URI",
    "ECS_CONTAINER_METADATA_URI_V4",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_BATCH_JOB_ID",
)


def quiet() -> bool:
    """Return ``True`` when ``EASYSNOWDATA_QUIET`` asks for no import-time output."""
    return os.environ.get("EASYSNOWDATA_QUIET", "").strip().lower() in _TRUTHY


def is_interactive() -> bool:
    """Return ``True`` in IPython/Jupyter, the Python REPL, or a terminal session."""
    ipython = sys.modules.get("IPython")
    if ipython is not None:
        try:
            if ipython.get_ipython() is not None:  # type: ignore[attr-defined]
                return True
        except Exception:  # pragma: no cover — defensive
            pass
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return True
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):  # closed or replaced stdout
        return False


def cache_dir(*subdirs: str) -> Path:
    """Return (and create) the easysnowdata cache directory.

    Defaults to the platform user cache dir (``~/.cache/easysnowdata`` on
    Linux, ``~/Library/Caches/easysnowdata`` on macOS,
    ``%LOCALAPPDATA%\\easysnowdata\\cache`` on Windows). Set
    ``EASYSNOWDATA_CACHE_DIR`` to override the root.
    """
    root = os.environ.get("EASYSNOWDATA_CACHE_DIR")
    if not root:
        import pooch  # noqa: PLC0415 — keep import light

        root = pooch.os_cache("easysnowdata")
    path = Path(root).expanduser().joinpath(*subdirs)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── compute region ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Region:
    """Where this process runs, as far as data access is concerned.

    Attributes
    ----------
    cloud
        ``"aws"`` when running on AWS, otherwise ``None``.
    region
        The AWS region (``"us-west-2"``) when known, otherwise ``None``.
    source
        How it was determined (``"EASYSNOWDATA_REGION"``, ``"env"``,
        ``"dmi"``, ``"imds"``, ``"default"``).
    """

    cloud: str | None = None
    region: str | None = None
    source: str = "default"

    @property
    def on_aws(self) -> bool:
        return self.cloud == "aws"

    @property
    def direct_s3(self) -> bool:
        """``True`` when Earthdata loaders should default to in-region S3 access."""
        return self.on_aws and self.region == DIRECT_S3_REGION

    def describe(self) -> str:
        access = "direct S3 access" if self.direct_s3 else "HTTPS access"
        if not self.on_aws:
            return f"local ({access})"
        where = f"AWS {self.region}" if self.region else "AWS (region unknown)"
        return f"{where} ({access})"

    def __str__(self) -> str:
        return self.describe()


_REGION: Region | None = None
_PROBED = False


def _dmi_says_ec2() -> bool:
    """Cheap EC2 host check from the DMI/hypervisor files (Linux only)."""
    checks = (
        ("product_uuid", lambda s: s.lower().startswith("ec2")),
        ("sys_vendor", lambda s: s.strip() == "Amazon EC2"),
        ("board_asset_tag", lambda s: s.startswith("i-")),
    )
    for name, test in checks:
        try:
            if test(Path(_DMI_DIR, name).read_text(errors="ignore").strip()):
                return True
        except OSError:
            continue
    return False


def _region_from_environment() -> Region:
    override = os.environ.get("EASYSNOWDATA_REGION", "").strip()
    if override:
        if override.lower() in {"local", "none", "off"}:
            return Region(None, None, "EASYSNOWDATA_REGION")
        return Region("aws", override, "EASYSNOWDATA_REGION")
    env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    on_aws = any(os.environ.get(v) for v in _AWS_EXECUTION_HINTS)
    if on_aws:
        return Region("aws", env_region or None, "env")
    if _dmi_says_ec2():
        return Region("aws", env_region or None, "dmi")
    return Region()


def _region_from_imds() -> str | None:
    """Ask the EC2 instance-metadata service (IMDSv2) for the region."""
    import requests  # noqa: PLC0415

    try:
        token = requests.put(
            IMDS_TOKEN_URL,
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=IMDS_TIMEOUT,
        )
        token.raise_for_status()
        resp = requests.get(
            IMDS_REGION_URL,
            headers={"X-aws-ec2-metadata-token": token.text},
            timeout=IMDS_TIMEOUT,
        )
        resp.raise_for_status()
        value = resp.text.strip()
        return value or None
    except Exception as exc:  # any failure means "not on EC2 / blocked"
        _logger.debug("IMDS region probe failed: %s", exc)
        return None


def region(*, probe: bool = False, refresh: bool = False) -> Region:
    """Return the compute :class:`Region`, detecting it once per process.

    Parameters
    ----------
    probe
        Also query the EC2 instance-metadata service (200 ms timeout) when the
        environment and DMI files say "AWS" but not which region. Loaders pass
        ``probe=True`` on first Earthdata use; import-time callers do not.
    refresh
        Discard the cached answer and detect again.
    """
    global _REGION, _PROBED
    if refresh:
        _REGION, _PROBED = None, False
    if _REGION is None:
        _REGION = _region_from_environment()
    if probe and not _PROBED and _REGION.source != "EASYSNOWDATA_REGION":
        _PROBED = True
        if _REGION.region is None and (_REGION.on_aws or _REGION.source == "default"):
            found = _region_from_imds()
            if found:
                _REGION = Region("aws", found, "imds")
    return _REGION


def access_mode(reg: Region | None = None) -> str:
    """``"direct"`` for in-region S3 reads of NASA data, else ``"https"``."""
    reg = region() if reg is None else reg
    return "direct" if reg.direct_s3 else "https"
