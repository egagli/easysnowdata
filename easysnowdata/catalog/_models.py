"""Catalog dataclasses: Variable, Source, Probe, Product (§3.3)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Variable", "Probe", "Source", "Product", "validate"]

KNOWN_PROVIDERS = (
    "stac",
    "earthdata",
    "gee",
    "raster_http",
    "zarr_cloud",
    "vector_http",
    "planet",
)
KNOWN_THEMES = (
    "stations",
    "sar",
    "optical",
    "snow",
    "land",
    "terrain",
    "climate",
    "hydro",
)


@dataclass(frozen=True)
class Variable:
    """One data variable of a product.

    Categorical variables carry CF-style ``flag_values`` / ``flag_meanings``
    plus ``flag_colors`` (hex strings); :meth:`cf_attrs` renders them as the
    plain-string/number attrs the design contract requires (§2.5).
    """

    name: str
    units: str | None = None
    dtype: str | None = None
    nodata: float | int | None = None
    long_name: str = ""
    flag_values: tuple[int, ...] = ()
    flag_meanings: tuple[str, ...] = ()
    flag_colors: tuple[str, ...] = ()

    @property
    def categorical(self) -> bool:
        return bool(self.flag_values)

    def cf_attrs(self) -> dict[str, Any]:
        """CF flag attributes (lists of ints and space-separated strings)."""
        attrs: dict[str, Any] = {}
        if self.long_name:
            attrs["long_name"] = self.long_name
        if self.units is not None:
            attrs["units"] = self.units
        if self.categorical:
            attrs["flag_values"] = list(self.flag_values)
            attrs["flag_meanings"] = " ".join(
                m.strip().replace(" ", "_") for m in self.flag_meanings
            )
            if self.flag_colors:
                attrs["flag_colors"] = " ".join(self.flag_colors)
        return attrs


@dataclass(frozen=True)
class Probe:
    """A minimal live check of one access route (§8).

    ``requires`` overrides the source's credential requirement for the probe
    itself (a STAC search of an EDL-protected catalog needs no login).
    """

    label: str
    fn: Callable[[], None]
    requires: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Source:
    """One access route to a product."""

    id: str
    provider: str
    location: str
    requires: tuple[str, ...] = ()
    resolution_m: float | None = None
    extent: str = "global"
    temporal: str | None = None
    latency: str | None = None
    notes: str = ""
    title: str = ""
    health: tuple[Probe, ...] = ()

    def __post_init__(self) -> None:
        # Accept a bare callable or a single Probe for `health`
        health = self.health
        if callable(health):
            health = (Probe(self.title or self.id, health),)
        elif isinstance(health, Probe):
            health = (health,)
        object.__setattr__(self, "health", tuple(health))
        if not self.title:
            object.__setattr__(self, "title", self.id)

    @property
    def credential_free(self) -> bool:
        return not self.requires


@dataclass(frozen=True)
class Product:
    """One dataset, with one or more sources (the first is the default)."""

    id: str
    theme: str
    title: str
    description: str
    sources: tuple[Source, ...]
    citation: str
    license: str
    loader: str
    variables: tuple[Variable, ...] = ()
    doi: str | None = None
    references: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Gallery scripts that load this product, as paths under ``docs/gallery``
    #: (``"snow/plot_snodas.py"``). §2.11 asks every product for one; the docs
    #: page links them, and the offline catalog test checks they exist.
    examples: tuple[str, ...] = ()

    @property
    def default_source(self) -> Source:
        return self.sources[0]

    def source(self, source_id: str | None = None) -> Source:
        """Return the source called *source_id* (default: the first)."""
        if source_id is None:
            return self.default_source
        for src in self.sources:
            if src.id == source_id:
                return src
        raise ValueError(
            f"Product {self.id!r} has no source {source_id!r}; "
            f"available: {', '.join(s.id for s in self.sources)}."
        )

    def variable(self, name: str) -> Variable:
        """Return the variable called *name*."""
        for var in self.variables:
            if var.name == name:
                return var
        raise KeyError(
            f"Product {self.id!r} has no variable {name!r}; "
            f"available: {', '.join(v.name for v in self.variables)}."
        )

    @property
    def requires(self) -> tuple[str, ...]:
        """Credentials the default source needs."""
        return self.default_source.requires

    @property
    def credential_free_sources(self) -> tuple[Source, ...]:
        return tuple(s for s in self.sources if s.credential_free)

    def resolve_loader(self) -> Any:
        """Import and return the object at :attr:`loader` (``module:attr`` or dotted)."""
        module_name, _, attr = self.loader.replace(":", ".").rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, attr)


def validate(product: Product, *, known_auth: tuple[str, ...] = ()) -> list[str]:
    """Return the list of problems with *product* (empty when valid)."""
    problems: list[str] = []
    pid = product.id
    if not pid or pid != pid.lower() or " " in pid or "_" in pid:
        problems.append(f"{pid!r}: id must be lower-case kebab-case")
    if product.theme not in KNOWN_THEMES:
        problems.append(f"{pid}: unknown theme {product.theme!r}")
    for attr in ("title", "description", "citation", "license", "loader"):
        if not getattr(product, attr):
            problems.append(f"{pid}: {attr} is empty")
    if not product.sources:
        problems.append(f"{pid}: no sources")
    if not any(src.health for src in product.sources):
        problems.append(f"{pid}: no health probe on any source")
    ids = [s.id for s in product.sources]
    if len(set(ids)) != len(ids):
        problems.append(f"{pid}: duplicate source ids {ids}")
    for src in product.sources:
        if src.provider not in KNOWN_PROVIDERS:
            problems.append(f"{pid}/{src.id}: unknown provider {src.provider!r}")
        if not src.location:
            problems.append(f"{pid}/{src.id}: location is empty")
        for req in src.requires:
            if known_auth and req not in known_auth:
                problems.append(f"{pid}/{src.id}: unknown auth provider {req!r}")
        for probe in src.health:
            if not callable(probe.fn) or not probe.label:
                problems.append(f"{pid}/{src.id}: malformed health probe")
    for example in product.examples:
        if not example.endswith(".py") or example.startswith("/"):
            problems.append(
                f"{pid}: example {example!r} must be a path under docs/gallery, "
                "like 'snow/plot_snodas.py'"
            )
    for var in product.variables:
        if var.categorical:
            n = len(var.flag_values)
            if len(var.flag_meanings) != n:
                problems.append(
                    f"{pid}/{var.name}: flag_meanings length != flag_values"
                )
            if var.flag_colors and len(var.flag_colors) != n:
                problems.append(f"{pid}/{var.name}: flag_colors length != flag_values")
            if len(set(var.flag_values)) != n:
                problems.append(f"{pid}/{var.name}: duplicate flag_values")
    if product.loader:
        try:
            product.resolve_loader()
        except Exception as exc:  # noqa: BLE001 — report every import problem
            problems.append(f"{pid}: loader {product.loader!r} does not import ({exc})")
    return problems
