"""Credentials: one ``Provider`` per credentialed service, used uniformly.

::

    import easysnowdata as esd
    esd.auth.status()                     # table: provider, configured?, how, needed by
    esd.auth.login()                      # interactive, only for what is missing
    esd.auth.login("earthengine", project="my-gcp-project")

Loaders call ``auth.ensure(*product.requires)`` before any network request and
wrap reads in ``auth.env(*product.requires)``. Missing credentials raise
:class:`CredentialError` (with ``.provider``) before any data is requested.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from easysnowdata import config
from easysnowdata.auth._base import CredentialError, Detection, Provider
from easysnowdata.auth.earthdata import EarthdataProvider
from easysnowdata.auth.earthengine import EarthEngineProvider
from easysnowdata.auth.nve import NVEProvider
from easysnowdata.auth.planet import PlanetProvider
from easysnowdata.auth.planetary_computer import PlanetaryComputerProvider

__all__ = [
    "CredentialError",
    "Detection",
    "Provider",
    "PROVIDERS",
    "get",
    "detect",
    "status",
    "login",
    "ensure",
    "env",
    "reset",
    "summary_line",
]

_logger = logging.getLogger("easysnowdata")

PROVIDERS: dict[str, Provider] = {
    p.name: p
    for p in (
        EarthdataProvider(),
        EarthEngineProvider(),
        PlanetaryComputerProvider(),
        PlanetProvider(),
        NVEProvider(),
    )
}
# Providers shown in the import-time line (the optional, anonymous ones are not)
_SUMMARY_PROVIDERS = ("earthdata", "earthengine", "planet", "nve")


def get(name: str) -> Provider:
    """Return the provider called *name* (``"earthdata"``, ``"earthengine"``, …)."""
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown auth provider {name!r}. Known providers: {', '.join(PROVIDERS)}."
        ) from None


def detect(name: str) -> Detection:
    """Network-free credential check for one provider."""
    return get(name).detect()


def _needed_by() -> dict[str, list[str]]:
    """Product ids per provider, from the catalog (empty until it is populated)."""
    try:
        from easysnowdata import catalog  # noqa: PLC0415
    except ImportError:  # pragma: no cover — catalog lands in a later commit
        return {}
    needed: dict[str, list[str]] = {name: [] for name in PROVIDERS}
    for product in catalog.products().values():
        for source in product.sources:
            for req in source.requires:
                if req in needed and product.id not in needed[req]:
                    needed[req].append(product.id)
    return needed


def status() -> Any:
    """Return a table of providers: configured?, how, needed by which products."""
    import pandas as pd  # noqa: PLC0415

    needed = _needed_by()
    rows = []
    for name, provider in PROVIDERS.items():
        detection = provider.detect()
        rows.append(
            {
                "provider": name,
                "title": provider.title,
                "configured": detection.configured,
                "how": detection.how,
                "optional": provider.optional,
                "env_vars": ", ".join(provider.env_vars),
                "needed_by": ", ".join(needed.get(name, [])),
            }
        )
    return pd.DataFrame(rows).set_index("provider")


def login(
    name: str | None = None,
    *,
    interactive: bool = True,
    persist: bool = True,
    **kwargs: Any,
) -> None:
    """Interactively set up credentials.

    With no *name*, every non-optional provider that is not yet configured is
    asked in turn (failures are logged, not raised). With a *name*, that
    provider's login runs and errors propagate; extra keyword arguments go to
    it (``project=`` for Earth Engine).
    """
    if name is not None:
        get(name).login(interactive=interactive, persist=persist, **kwargs)
        return
    for provider in PROVIDERS.values():
        if provider.optional or provider.detect():
            continue
        try:
            provider.login(interactive=interactive, persist=persist)
        except Exception as exc:  # keep going; the table shows what is missing
            _logger.warning("%s: %s", provider.title, str(exc).splitlines()[0])


def ensure(*names: str, **kwargs: Any) -> dict[str, Any]:
    """Initialise the named providers once; raise ``CredentialError`` if any is missing."""
    return {name: get(name).ensure(**kwargs) for name in names}


@contextlib.contextmanager
def env(*names: str) -> Iterator[dict[str, Any]]:
    """Enter the named providers' read environments (GDAL options, headers…)."""
    merged: dict[str, Any] = {}
    with contextlib.ExitStack() as stack:
        for name in names:
            merged.update(stack.enter_context(get(name).env()) or {})
        yield merged


def reset() -> None:
    """Forget every provider's cached initialisation."""
    for provider in PROVIDERS.values():
        provider.reset()


def summary_line() -> str:
    """The one-line credential summary shown on import (network-free)."""
    from easysnowdata import __version__  # noqa: PLC0415

    parts = []
    for name in _SUMMARY_PROVIDERS:
        provider = PROVIDERS[name]
        detection = provider.detect()
        mark = "✓" if detection else "✗"
        how = detection.how or ""
        how = how.split(":", 1)[-1] if how.startswith(("env:", "file:")) else how
        parts.append(
            f"{provider.title} {mark}" + (f" ({how})" if detection and how else "")
        )
    return (
        f"easysnowdata {__version__} · credentials: "
        + " · ".join(parts)
        + f" · compute: {config.region().describe()} — see esd.auth.status()"
    )


def _emit_import_summary() -> None:
    """Print the summary in interactive sessions, log it otherwise; honour EASYSNOWDATA_QUIET."""
    if config.quiet():
        return
    try:
        line = summary_line()
    except Exception as exc:  # pragma: no cover — never break import
        _logger.debug("Credential summary skipped: %s", exc)
        return
    if config.is_interactive():
        print(line)  # noqa: T201 — the one sanctioned print (§5.2 visibility)
    else:
        _logger.info(line)
