"""Plotting helpers that read CF flag attributes (design contract §2.7).

Nothing here is required to use the data: the helpers build a
``ListedColormap`` + ``BoundaryNorm`` and a legend from ``flag_values`` /
``flag_meanings`` / ``flag_colors`` written by
:mod:`easysnowdata.processing.categorical`, register named colormaps, and draw
RGB quicklooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from easysnowdata.processing.categorical import flags

__all__ = [
    "colormap_from_flags",
    "legend_handles",
    "categorical",
    "register_colormap",
    "rgb",
]


def colormap_from_flags(obj: xr.DataArray | dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return ``(cmap, norm, table)`` for a categorical array's CF flags.

    The norm maps each ``flag_value`` to its own color regardless of gaps
    between values (5, 37, 200 …).
    """
    import matplotlib.colors as mcolors  # noqa: PLC0415

    table = flags(obj).sort_values("value").reset_index(drop=True)
    colors = [c if c is not None else "#808080" for c in table["color"]]
    cmap = mcolors.ListedColormap(colors, name="easysnowdata_flags")
    values = table["value"].to_numpy(dtype="float64")
    edges = np.concatenate(
        [[values[0] - 0.5], (values[:-1] + values[1:]) / 2, [values[-1] + 0.5]]
    )
    norm = mcolors.BoundaryNorm(edges, cmap.N)
    return cmap, norm, table


def legend_handles(obj: xr.DataArray | dict[str, Any]) -> tuple[list[Any], list[str]]:
    """``(handles, labels)`` for ``ax.legend`` from the CF flags (underscores → spaces)."""
    from matplotlib.patches import Patch  # noqa: PLC0415

    table = flags(obj).sort_values("value")
    handles = [
        Patch(facecolor=c or "#808080", edgecolor="black") for c in table["color"]
    ]
    labels = [m.replace("_", " ") for m in table["meaning"]]
    return handles, labels


def categorical(
    da: xr.DataArray,
    ax: Any = None,
    *,
    legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    **imshow_kwargs: Any,
) -> Any:
    """Plot a categorical DataArray with one color per class and a legend.

    Returns the matplotlib ``Axes``. ``da`` must carry CF ``flag_values`` /
    ``flag_meanings`` (and ideally ``flag_colors``); a ``time`` dimension of
    length one is squeezed.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        _, ax = plt.subplots(figsize=imshow_kwargs.pop("figsize", (8, 8)))
    data = da.squeeze(drop=True) if da.ndim > 2 else da
    cmap, norm, _ = colormap_from_flags(da)
    imshow_kwargs.setdefault("add_colorbar", False)
    data.plot.imshow(ax=ax, cmap=cmap, norm=norm, **imshow_kwargs)
    if legend:
        handles, labels = legend_handles(da)
        kwargs = {"loc": "center left", "bbox_to_anchor": (1.01, 0.5), "frameon": False}
        kwargs.update(legend_kwargs or {})
        ax.legend(handles, labels, **kwargs)
    ax.set_title(
        title if title is not None else str(da.attrs.get("long_name", da.name or ""))
    )
    ax.set_aspect("equal")
    return ax


def register_colormap(name: str, colors: list[str], *, overwrite: bool = False) -> Any:
    """Register a ``ListedColormap`` under *name* with matplotlib and return it."""
    import matplotlib  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415

    cmap = mcolors.ListedColormap(list(colors), name=name)
    if name in matplotlib.colormaps:
        if not overwrite:
            return matplotlib.colormaps[name]
        matplotlib.colormaps.unregister(name)
    matplotlib.colormaps.register(cmap)
    return cmap


def rgb(
    composite: xr.DataArray,
    ax: Any = None,
    *,
    dim: str = "band",
    title: str | None = None,
    **imshow_kwargs: Any,
) -> Any:
    """Show a 0–1 RGB composite from :func:`easysnowdata.processing.rgb`. Returns the ``Axes``."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        _, ax = plt.subplots(figsize=imshow_kwargs.pop("figsize", (8, 8)))
    data = composite.squeeze(drop=True) if composite.ndim > 3 else composite
    data.plot.imshow(ax=ax, rgb=dim, **imshow_kwargs)
    ax.set_title(title or "")
    ax.set_aspect("equal")
    return ax
