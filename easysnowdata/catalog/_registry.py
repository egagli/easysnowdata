"""The product registry and its query functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from easysnowdata.catalog import _models
from easysnowdata.catalog._models import Product, validate

__all__ = [
    "register",
    "products",
    "get",
    "list_products",
    "search",
    "describe",
    "themes",
    "validate_all",
]

_PRODUCTS: dict[str, Product] = {}


def register(*items: Product, replace: bool = False) -> None:
    """Add products to the registry (duplicate ids raise unless *replace*)."""
    for product in items:
        if product.id in _PRODUCTS and not replace:
            raise ValueError(f"Product {product.id!r} is already registered.")
        _PRODUCTS[product.id] = product


def products() -> Mapping[str, Product]:
    """Read-only view of every registered product, keyed by id."""
    return MappingProxyType(_PRODUCTS)


def get(product_id: str) -> Product:
    """Return one product by id."""
    try:
        return _PRODUCTS[product_id]
    except KeyError:
        raise KeyError(
            f"Unknown product {product_id!r}. Try easysnowdata.catalog.search(...) "
            f"or .list(); known ids: {', '.join(sorted(_PRODUCTS))}."
        ) from None


def themes() -> list[str]:
    """The themes with at least one product, in :data:`KNOWN_THEMES` order."""
    return sorted({p.theme for p in _PRODUCTS.values()}, key=_models.theme_order)


def _rows(items: Iterable[Product]) -> list[dict[str, Any]]:
    rows = []
    for p in items:
        default = p.default_source
        rows.append(
            {
                "id": p.id,
                "theme": p.theme,
                "title": p.title,
                "default_source": default.id,
                "provider": default.provider,
                "requires": ", ".join(default.requires) or "none",
                "sources": len(p.sources),
                "credential_free": bool(p.credential_free_sources),
                "resolution_m": default.resolution_m,
                "extent": default.extent,
                "temporal": default.temporal,
                "loader": p.loader,
            }
        )
    return rows


def list_products(
    theme: str | None = None,
    *,
    provider: str | None = None,
    requires: str | None = None,
    credential_free: bool | None = None,
) -> Any:
    """Return a DataFrame of products, optionally filtered.

    Parameters
    ----------
    theme
        ``"snow"``, ``"terrain"``, …
    provider
        Keep products with at least one source on this provider (``"stac"``…).
    requires
        Keep products whose default source needs this auth provider.
    credential_free
        ``True`` keeps products with at least one credential-free source,
        ``False`` those without any.
    """
    import pandas as pd  # noqa: PLC0415

    items = list(_PRODUCTS.values())
    if theme is not None:
        items = [p for p in items if p.theme == theme]
    if provider is not None:
        items = [p for p in items if any(s.provider == provider for s in p.sources)]
    if requires is not None:
        items = [p for p in items if requires in p.requires]
    if credential_free is not None:
        items = [p for p in items if bool(p.credential_free_sources) is credential_free]
    products_df = pd.DataFrame(_rows(items), columns=list(_rows([]) or _COLUMNS))
    return products_df.set_index("id") if len(products_df) else products_df


_COLUMNS = [
    "id", "theme", "title", "default_source", "provider", "requires", "sources",
    "credential_free", "resolution_m", "extent", "temporal", "loader",
]  # fmt: skip


def search(text: str) -> Any:
    """Case-insensitive search over id, title, description, tags, variables and sources."""
    needle = text.lower()

    def hit(p: Product) -> bool:
        hay = " ".join(
            [p.id, p.title, p.description, p.theme, *p.tags]
            + [v.name for v in p.variables]
            + [f"{s.id} {s.title} {s.location} {s.notes}" for s in p.sources]
        ).lower()
        return needle in hay

    import pandas as pd  # noqa: PLC0415

    matches_df = pd.DataFrame(
        _rows(p for p in _PRODUCTS.values() if hit(p)), columns=_COLUMNS
    )
    return matches_df.set_index("id") if len(matches_df) else matches_df


def describe(product_id: str) -> str:
    """Return a Markdown description of a product: sources, variables, citation.

    The same tables the generated docs page uses (:mod:`easysnowdata.catalog.pages`),
    so the two cannot disagree.
    """
    from easysnowdata.catalog import pages  # noqa: PLC0415 — avoids an import cycle

    p = get(product_id)
    lines = [f"# {p.title} (`{p.id}`)", "", p.description.strip(), ""]
    lines += pages.sources_table(p)
    variables = pages.variables_table(p)
    if variables:
        lines += [""] + variables
    lines += ["", f"**Loader:** `{p.loader}`", f"**License:** {p.license}"]
    if p.doi:
        lines.append(f"**DOI:** {p.doi}")
    lines += ["", "**Citation:** " + p.citation.strip()]
    if p.references:
        lines += ["", "**References:**"] + [f"- {r}" for r in p.references]
    return "\n".join(lines) + "\n"


def validate_all(*, known_auth: tuple[str, ...] = ()) -> list[str]:
    """Validate every registered product; returns all problems found."""
    problems: list[str] = []
    for product in _PRODUCTS.values():
        problems.extend(validate(product, known_auth=known_auth))
    return problems
