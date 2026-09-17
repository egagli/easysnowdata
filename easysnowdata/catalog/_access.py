"""Resolve a product's access route and its credentials before any network call.

Every Phase 2 loader starts with::

    source = resolve_source(PRODUCT, source)      # Source dataclass, default first
    ensure_source(PRODUCT, source)                # CredentialError names alternatives

``ensure_source`` calls ``auth.ensure(*source.requires)``; when a provider is
missing it re-raises the :class:`~easysnowdata.auth.CredentialError` with the
product's credential-free sources appended (§5.2: "never falls back silently").
"""

from __future__ import annotations

from typing import Any

from easysnowdata import auth
from easysnowdata.auth import CredentialError
from easysnowdata.catalog._models import Product, Source

__all__ = ["resolve_source", "ensure_source", "requires_for"]


def resolve_source(product: Product, source: str | Source | None = None) -> Source:
    """Return the :class:`Source` for *source* (an id, a Source, or ``None`` for the default)."""
    if isinstance(source, Source):
        return source
    return product.source(source)


def requires_for(source: Source, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The auth providers a read from *source* needs (deduplicated, in order)."""
    return tuple(dict.fromkeys([*source.requires, *extra]))


def ensure_source(
    product: Product,
    source: Source,
    *,
    extra_requires: tuple[str, ...] = (),
    **kwargs: Any,
) -> dict[str, Any]:
    """Initialise the providers *source* needs; raise a helpful ``CredentialError``."""
    names = requires_for(source, extra_requires)
    if not names:
        return {}
    try:
        return auth.ensure(*names, **kwargs)
    except CredentialError as exc:
        alternatives = tuple(
            f'source="{s.id}"'
            for s in product.credential_free_sources
            if s is not source
        )
        provider = auth.get(exc.provider) if exc.provider in auth.PROVIDERS else None
        if provider is None or not alternatives:
            raise
        first_line = str(exc).splitlines()[0]
        raise provider.error(
            f"{product.title} ({product.id}) via source={source.id!r}: {first_line}",
            alternatives=alternatives,
        ) from exc
