"""CF flag attributes for categorical products.

The design contract (§2.5) forbids Python objects in ``.attrs``: a categorical
variable carries ``flag_values`` (list of ints), ``flag_meanings`` (one
space-separated string, blanks inside a meaning become underscores) and
``flag_colors`` (space-separated hex strings). These helpers write and read
that form; :mod:`easysnowdata.plotting` turns it into colormaps and legends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["set_flags", "flags", "from_class_info", "flag_mask", "meaning_key"]


def meaning_key(meaning: str) -> str:
    """Normalise a class name to a CF ``flag_meanings`` token (blanks → underscores)."""
    return "_".join(str(meaning).strip().split())


def set_flags(
    obj: xr.DataArray,
    values: Iterable[int],
    meanings: Iterable[str],
    colors: Iterable[str] | None = None,
    *,
    long_name: str | None = None,
    inplace: bool = False,
) -> xr.DataArray:
    """Attach CF flag attributes to a categorical DataArray.

    Lengths must match. Returns the (copied unless *inplace*) array.
    """
    values = [int(v) for v in values]
    meanings = [meaning_key(m) for m in meanings]
    if len(values) != len(meanings):
        raise ValueError("flag_values and flag_meanings must have the same length.")
    attrs: dict[str, Any] = {"flag_values": values, "flag_meanings": " ".join(meanings)}
    if colors is not None:
        colors = [str(c) for c in colors]
        if len(colors) != len(values):
            raise ValueError("flag_colors must have the same length as flag_values.")
        attrs["flag_colors"] = " ".join(colors)
    if long_name is not None:
        attrs["long_name"] = long_name
    flagged_da = obj if inplace else obj.copy(deep=False)
    flagged_da.attrs.update(attrs)
    return flagged_da


def flags(obj: xr.DataArray | Mapping[str, Any]) -> pd.DataFrame:
    """Read CF flag attributes into a DataFrame with ``value``, ``meaning``, ``color``.

    Accepts a DataArray or its ``attrs``. Raises ``KeyError`` without
    ``flag_values``; ``color`` is ``None`` when ``flag_colors`` is absent.
    """
    attrs = obj.attrs if isinstance(obj, xr.DataArray) else obj
    if "flag_values" not in attrs:
        raise KeyError("No flag_values in attrs; not a categorical variable.")
    values = [int(v) for v in np.atleast_1d(attrs["flag_values"])]
    meanings = _split(attrs.get("flag_meanings", ""))
    colors = (
        _split(attrs.get("flag_colors", ""))
        if attrs.get("flag_colors")
        else [None] * len(values)
    )
    if len(meanings) != len(values) or len(colors) != len(values):
        raise ValueError("flag_values, flag_meanings and flag_colors lengths differ.")
    return pd.DataFrame({"value": values, "meaning": meanings, "color": colors})


def _split(text: Any) -> list[str]:
    if isinstance(text, str):
        return text.split()
    return [str(t) for t in np.atleast_1d(text)]


def from_class_info(
    class_info: Mapping[int, Mapping[str, Any]],
) -> tuple[list[int], list[str], list[str]]:
    """Convert the legacy ``{value: {"name": ..., "color": ...}}`` dict to flag lists.

    Colors given as RGB triplets (0–255) are converted to hex.
    """
    values, meanings, colors = [], [], []
    for value, info in class_info.items():
        values.append(int(value))
        meanings.append(meaning_key(info["name"]))
        color = info.get("color", "#000000")
        if not isinstance(color, str):
            r, g, b = (int(c) for c in color[:3])
            color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append(color)
    return values, meanings, colors


def flag_mask(obj: xr.DataArray, *meanings: str) -> xr.DataArray:
    """Boolean mask that is ``True`` where the array holds any of *meanings*.

    Meanings are matched case-insensitively after :func:`meaning_key` normalisation.
    """
    flags_df = flags(obj)
    known = flags_df["meaning"].str.lower()
    wanted = {meaning_key(m).lower() for m in meanings}
    unknown = wanted - set(known)
    if unknown:
        raise ValueError(
            f"Unknown flag meanings {sorted(unknown)}; known: {list(flags_df['meaning'])}."
        )
    values = flags_df.loc[known.isin(wanted), "value"].tolist()
    return obj.isin(values)
