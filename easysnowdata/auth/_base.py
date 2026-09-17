"""Provider base class, detection result, and CredentialError."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

__all__ = ["CredentialError", "Detection", "Provider"]

DOCS_URL = "https://egagli.github.io/easysnowdata/credentials/"


class CredentialError(Exception):
    """Raised when a data provider's credentials are missing or rejected.

    Attributes
    ----------
    provider
        Name of the auth provider (``"earthdata"``, ``"earthengine"``, …) or
        ``None`` when raised outside the provider machinery.
    docs_url
        Where the setup steps are documented.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.docs_url = docs_url or DOCS_URL


@dataclass(frozen=True)
class Detection:
    """Result of a network-free credential check.

    ``how`` says where the credentials were found (``"env:EARTHDATA_TOKEN"``,
    ``"netrc"``, ``"file:~/.planet.json"``, ``"anonymous"``…).
    """

    configured: bool
    how: str | None = None
    details: str = ""

    def __bool__(self) -> bool:
        return self.configured


class Provider:
    """One credentialed service. Subclasses implement the four verbs.

    * :meth:`detect` — environment variables and file existence only; no
      network, no heavy imports (it runs at ``import easysnowdata``).
    * :meth:`login` — interactive setup that persists per the service's own
      convention (never a config file of our own).
    * :meth:`ensure` — initialise once, idempotently; raise
      :class:`CredentialError` with the setup text before any data request.
    * :meth:`env` — context manager yielding the GDAL / HTTP configuration
      that reads from this service need.
    """

    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    optional: ClassVar[bool] = False  # works anonymously; a key only lifts limits
    env_vars: ClassVar[tuple[str, ...]] = ()
    files: ClassVar[tuple[str, ...]] = ()
    signup_url: ClassVar[str] = ""
    setup_instructions: ClassVar[str] = ""

    def __init__(self) -> None:
        self._ensured: Any = None

    # -- verbs -----------------------------------------------------------------

    def detect(self) -> Detection:
        raise NotImplementedError

    def login(
        self, *, interactive: bool = True, persist: bool = True, **kwargs: Any
    ) -> None:
        raise self.error(
            f"{self.title} has no interactive login; follow the setup steps below."
        )

    def ensure(self, **kwargs: Any) -> Any:
        if self._ensured is not None:
            return self._ensured
        if not self.detect():
            raise self.error()
        self._ensured = True
        return self._ensured

    @contextlib.contextmanager
    def env(self) -> Iterator[dict[str, Any]]:
        yield {}

    def reset(self) -> None:
        """Forget cached initialisation (tests, or after credentials change)."""
        self._ensured = None

    # -- helpers ---------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.detect())

    def error(
        self,
        message: str | None = None,
        *,
        alternatives: tuple[str, ...] = (),
        cause: str | None = None,
    ) -> CredentialError:
        """Build the standard :class:`CredentialError` for this provider."""
        lines = [
            message or f"{self.title} credentials are required but were not found."
        ]
        if cause:
            lines.append(f"Cause: {cause}")
        if self.setup_instructions:
            lines.extend(["", self.setup_instructions.rstrip()])
        if alternatives:
            lines.extend(
                [
                    "",
                    "Credential-free alternatives for this product: "
                    + ", ".join(alternatives),
                ]
            )
        lines.extend(["", f"Documentation: {DOCS_URL}"])
        return CredentialError("\n".join(lines), provider=self.name)

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return f"<{type(self).__name__} {self.name!r} configured={self.configured}>"
