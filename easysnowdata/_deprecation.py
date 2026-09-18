"""Deprecation shims (§3.4, §11): old names wrap new functions for one minor release.

::

    from easysnowdata._deprecation import deprecated, deprecated_alias

    @deprecated("easysnowdata.terrain.dem.load", since="0.1.0", remove_in="0.3.0")
    def get_copernicus_dem(bbox_input=None, resolution=30, **kwargs):
        return dem.load(bbox_input, source=f"copernicus-glo{resolution}", **kwargs)

    get_water_year_start = deprecated_alias(
        "get_water_year_start", wateryear.water_year_start, since="0.1.0"
    )

Warnings are :class:`EasysnowdataDeprecationWarning` (a ``DeprecationWarning``),
raised with ``stacklevel=2`` so notebooks and scripts see them at the call
site. Each old name warns once per process. Phase 2 migrates the products
onto this; Phase 5 removes the shims.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "EasysnowdataDeprecationWarning",
    "deprecated",
    "deprecated_alias",
    "deprecated_module_attrs",
    "warn_once",
    "reset_warnings",
]

F = TypeVar("F", bound=Callable[..., Any])


class EasysnowdataDeprecationWarning(DeprecationWarning):
    """Raised when an old easysnowdata name is used."""


_WARNED: set[str] = set()


def _message(
    old: str,
    replacement: str | None,
    since: str | None,
    remove_in: str | None,
    extra: str,
) -> str:
    text = f"{old} is deprecated"
    if since:
        text += f" since easysnowdata {since}"
    if remove_in:
        text += f" and will be removed in {remove_in}"
    text += "."
    if replacement:
        text += f" Use {replacement} instead."
    if extra:
        text += f" {extra}"
    return text


def warn_once(key: str, message: str, *, stacklevel: int = 3) -> None:
    """Emit *message* as an :class:`EasysnowdataDeprecationWarning` once per *key*."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, EasysnowdataDeprecationWarning, stacklevel=stacklevel)


def reset_warnings() -> None:
    """Forget which names have warned (tests)."""
    _WARNED.clear()


def deprecated(
    replacement: str | None = None,
    *,
    since: str | None = None,
    remove_in: str | None = None,
    extra: str = "",
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator: the wrapped callable warns once, then runs.

    The docstring gains a ``.. deprecated::`` note so it shows in the docs.
    """

    def decorate(func: F) -> F:
        old = name or f"{func.__module__}.{func.__qualname__}"
        message = _message(old, replacement, since, remove_in, extra)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_once(old, message)
            return func(*args, **kwargs)

        note = f".. deprecated:: {since or ''}\n    {message}\n\n"
        wrapper.__doc__ = note + (func.__doc__ or "")
        wrapper.__deprecated__ = message  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def deprecated_alias(
    old_name: str,
    func: Callable[..., Any],
    *,
    since: str | None = None,
    remove_in: str | None = None,
    extra: str = "",
) -> Callable[..., Any]:
    """Return *func* wrapped so calling it as *old_name* warns once.

    The new function's own name is used as the replacement in the message.
    """
    replacement = f"{func.__module__}.{func.__qualname__}"
    return deprecated(
        replacement, since=since, remove_in=remove_in, extra=extra, name=old_name
    )(func)


def deprecated_module_attrs(
    module_name: str,
    mapping: dict[str, tuple[Callable[..., Any] | Any, str]],
    *,
    since: str | None = None,
    remove_in: str | None = None,
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that serves old attribute names with a warning.

    *mapping* is ``{old_name: (new_object, replacement_text)}``. Attribute
    access warns once and returns ``new_object``; unknown names raise
    ``AttributeError`` as usual.
    """

    def __getattr__(name: str) -> Any:
        if name in mapping:
            new_object, replacement = mapping[name]
            warn_once(
                f"{module_name}.{name}",
                _message(f"{module_name}.{name}", replacement, since, remove_in, ""),
            )
            return new_object
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
