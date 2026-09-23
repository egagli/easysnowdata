"""Plotting helpers: maps that are true to scale, time series on calendar dates.

Nothing here is required to use the data: every product plots with plain
xarray and geopandas. What these helpers add is the map furniture and the
labelling conventions the gallery follows, so an example is one call rather
than fifteen lines of matplotlib:

* :func:`map` — one raster on equal-aspect axes with a matched colorbar, a
  scale bar, a light latitude/longitude graticule and an optional web
  basemap. Geographic (EPSG:4326) data get a latitude-corrected aspect and a
  warning, because a degree of longitude is not a degree of latitude.
* :func:`categorical` — the same for a classified raster, drawing its colours
  and legend from the CF ``flag_values`` / ``flag_meanings`` / ``flag_colors``
  attributes :mod:`easysnowdata.processing.categorical` writes.
* :func:`points` — a GeoDataFrame of stations (or any vector layer) over a
  basemap, coloured by a column with a legend or colorbar placed outside.
* :func:`timeseries` — one line per station (or other dimension) against
  calendar dates, with ``long name [units]`` axis labels. ``by_water_year``
  overlays water years on one October-to-September axis.
* :func:`label` — ``"long name [units]"`` from an object's attrs, the one
  convention every axis label here follows (units in square brackets).

Every piece of furniture is a keyword argument (``scalebar=False``,
``graticule=False``, ``basemap="CartoDB.Positron"`` …), and every function
returns the matplotlib ``Axes`` so anything can be added afterwards.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import xarray as xr

from easysnowdata.processing.categorical import flags

__all__ = [
    "map",
    "categorical",
    "points",
    "timeseries",
    "label",
    "finish_map",
    "add_scalebar",
    "add_graticule",
    "add_basemap",
    "colormap_from_flags",
    "legend_handles",
    "register_colormap",
    "DEFAULT_BASEMAP",
]

#: Web basemap drawn when ``basemap=True``: Esri's shaded relief, which needs
#: no API key and shows the terrain a snow map sits in without competing
#: labels. Any ``xyzservices`` provider or its dotted name works instead.
DEFAULT_BASEMAP = "Esri.WorldShadedRelief"

_GRATICULE_STEPS = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30,
)  # fmt: skip
_GRID_STYLE = {"color": "0.45", "linewidth": 0.5, "alpha": 0.55, "linestyle": ":"}
_LEGEND_OUTSIDE = {
    "loc": "center left",
    "bbox_to_anchor": (1.01, 0.5),
    "frameon": False,
}
_M_PER_DEG = 111_320.0


class GeographicAxesWarning(UserWarning):
    """Raised when a map is drawn in a geographic CRS (degrees, not metres)."""


# ── labels ────────────────────────────────────────────────────────────────────


def label(obj: Any, *, name: str | None = None, units: str | None = None) -> str:
    """``"long name [units]"`` from *obj* (a DataArray, Dataset or attrs mapping).

    Units go in square brackets, never parentheses. Falls back to the
    variable's name, then to an empty string.
    """
    attrs = obj if isinstance(obj, dict) else dict(getattr(obj, "attrs", {}) or {})
    text = name or str(attrs.get("long_name") or getattr(obj, "name", None) or "")
    unit = units if units is not None else attrs.get("units")
    if unit not in (None, "", "1"):
        return f"{text} [{unit}]"
    return text


def _time_suffix(da: xr.DataArray) -> str:
    """``", 2024-03-10"`` when *da* carries one date, else ``""``."""
    if "time" not in da.coords or da["time"].ndim:
        return ""
    value = da["time"].values
    try:
        stamp = np.datetime_as_string(np.datetime64(value), unit="D")
    except (ValueError, TypeError):
        return ""
    return f", {stamp}"


def _default_title(da: xr.DataArray) -> str:
    attrs = da.attrs or {}
    head = str(attrs.get("title") or attrs.get("long_name") or da.name or "")
    return head + _time_suffix(da)


# ── CRS helpers ───────────────────────────────────────────────────────────────


def _crs_of(obj: Any) -> Any:
    """The pyproj CRS of a DataArray/Dataset/GeoDataFrame, or ``None``."""
    from pyproj import CRS  # noqa: PLC0415

    crs = getattr(obj, "crs", None)
    if crs is None and hasattr(obj, "rio"):
        try:
            crs = obj.rio.crs
        except Exception:  # noqa: BLE001 — no spatial dims
            crs = None
    if crs is None and hasattr(obj, "odc"):
        try:
            crs = obj.odc.crs
        except Exception:  # noqa: BLE001
            crs = None
    return None if crs is None else CRS.from_user_input(str(crs))


def _axis_names(crs: Any) -> tuple[str, str]:
    """``("longitude [°]", "latitude [°]")`` or ``("easting [m]", "northing [m]")``."""
    if crs is None:
        return "x", "y"
    if crs.is_geographic:
        return "longitude [°]", "latitude [°]"
    names = []
    for axis in list(crs.axis_info)[:2]:
        unit = axis.unit_name
        unit = {"metre": "m", "meter": "m", "degree": "°"}.get(unit, unit)
        names.append(f"{axis.name.lower()} [{unit}]")
    if len(names) == 2:
        # pyproj lists northing first for some CRSs; keep x, y order.
        if names[0].startswith("northing"):
            names.reverse()
        return names[0], names[1]
    return "x", "y"


def _mid_latitude(ax: Any, crs: Any) -> float:
    """Latitude at the centre of the axes, in degrees."""
    from pyproj import Transformer  # noqa: PLC0415

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if crs is None or crs.is_geographic:
        return (y0 + y1) / 2
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    _, lat = to_lonlat.transform((x0 + x1) / 2, (y0 + y1) / 2)
    return float(lat)


def _format_lon(value: float) -> str:
    hemi = "W" if value < 0 else "E"
    return f"{abs(value):g}°{hemi}" if value else "0°"


def _format_lat(value: float) -> str:
    hemi = "S" if value < 0 else "N"
    return f"{abs(value):g}°{hemi}" if value else "0°"


def _nice_step(span: float, max_lines: int = 6) -> float:
    """The finest round step that draws at most *max_lines* lines across *span*."""
    for step in _GRATICULE_STEPS:
        if span / step <= max_lines:
            return step
    return _GRATICULE_STEPS[-1]


# ── furniture ─────────────────────────────────────────────────────────────────


def add_scalebar(ax: Any, crs: Any = None, **kwargs: Any) -> Any:
    """Add a small scale bar (``matplotlib_scalebar``) to *ax*.

    For a projected CRS one data unit is one metre. For a geographic CRS one
    data unit is a degree of longitude, converted at the axes' central
    latitude, so the bar is right along the x axis where it is drawn.
    """
    from matplotlib_scalebar.scalebar import ScaleBar  # noqa: PLC0415

    dx = 1.0
    if crs is not None and crs.is_geographic:
        dx = _M_PER_DEG * math.cos(math.radians(_mid_latitude(ax, crs)))
    options = {
        # The bar measures along x; a latitude-corrected aspect is deliberate.
        "rotation": "horizontal-only",
        "units": "m",
        "location": "lower left",
        "box_alpha": 0.75,
        "color": "0.15",
        "font_properties": {"size": 8},
        "length_fraction": 0.2,
        "scale_loc": "top",
        "pad": 0.4,
        "border_pad": 0.6,
    }
    options.update(kwargs)
    bar = ScaleBar(dx, **options)
    ax.add_artist(bar)
    return bar


def add_graticule(
    ax: Any,
    crs: Any = None,
    *,
    step: float | tuple[float, float] | None = None,
    labels: bool = True,
    **line_kwargs: Any,
) -> None:
    """Draw light latitude/longitude lines on *ax* and label them on the frame.

    On geographic axes the graticule is the tick grid itself, with ticks at
    round degrees and labels such as ``121.8°W``. On projected axes the
    meridians and parallels are drawn as (curved) lines in the map CRS, and
    the x and y tick labels are replaced by the longitude and latitude where
    each line meets the bottom and left edges, the way cartopy labels a map.
    """
    from pyproj import Transformer  # noqa: PLC0415

    style = dict(_GRID_STYLE)
    style.update(line_kwargs)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    geographic = crs is None or crs.is_geographic
    if geographic:
        lon0, lon1, lat0, lat1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    else:
        to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        edge = np.linspace(0, 1, 60)
        xs = np.concatenate(
            [
                x0 + (x1 - x0) * edge,
                np.full(60, x1),
                x1 - (x1 - x0) * edge,
                np.full(60, x0),
            ]
        )
        ys = np.concatenate(
            [
                np.full(60, y0),
                y0 + (y1 - y0) * edge,
                np.full(60, y1),
                y1 - (y1 - y0) * edge,
            ]
        )
        lons, lats = to_lonlat.transform(xs, ys)
        lon0, lon1, lat0, lat1 = np.min(lons), np.max(lons), np.min(lats), np.max(lats)
    if step is None:
        step_lon = _nice_step(lon1 - lon0)
        step_lat = _nice_step(lat1 - lat0)
    else:
        step_lon, step_lat = (step, step) if np.isscalar(step) else step
    meridians = np.arange(math.ceil(lon0 / step_lon) * step_lon, lon1 + 1e-9, step_lon)
    parallels = np.arange(math.ceil(lat0 / step_lat) * step_lat, lat1 + 1e-9, step_lat)
    meridians = np.round(meridians, 6)
    parallels = np.round(parallels, 6)

    if geographic:
        ax.set_xticks(meridians)
        ax.set_yticks(parallels)
        if labels:
            ax.set_xticklabels([_format_lon(v) for v in meridians])
            ax.set_yticklabels([_format_lat(v) for v in parallels])
        ax.grid(True, **style)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        return

    from_lonlat = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    dense_lat = np.linspace(lat0 - step_lat, lat1 + step_lat, 200)
    dense_lon = np.linspace(lon0 - step_lon, lon1 + step_lon, 200)
    x_ticks, x_labels, y_ticks, y_labels = [], [], [], []
    for lon in meridians:
        px, py = from_lonlat.transform(np.full_like(dense_lat, lon), dense_lat)
        ax.plot(px, py, zorder=2.5, **style)
        crossing = _crossing(px, py, y0, axis="y")
        if crossing is not None and x0 <= crossing <= x1:
            x_ticks.append(crossing)
            x_labels.append(_format_lon(lon))
    for lat in parallels:
        px, py = from_lonlat.transform(dense_lon, np.full_like(dense_lon, lat))
        ax.plot(px, py, zorder=2.5, **style)
        crossing = _crossing(px, py, x0, axis="x")
        if crossing is not None and y0 <= crossing <= y1:
            y_ticks.append(crossing)
            y_labels.append(_format_lat(lat))
    if labels:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("longitude [°]")
        ax.set_ylabel("latitude [°]")
        _corner_note(ax, f"grid {_crs_name(crs)}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _crossing(
    px: np.ndarray, py: np.ndarray, level: float, *, axis: str
) -> float | None:
    """Where the polyline ``(px, py)`` crosses ``y == level`` (axis="y") or ``x == level``."""
    along, across = (py, px) if axis == "y" else (px, py)
    diff = along - level
    sign = np.sign(diff)
    idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if diff[i + 1] == diff[i]:
        return float(across[i])
    t = -diff[i] / (diff[i + 1] - diff[i])
    return float(across[i] + t * (across[i + 1] - across[i]))


def _corner_note(ax: Any, text: str) -> None:
    """Small grey text in the lower-right corner; several calls stack into one line."""
    existing = getattr(ax, "_esd_corner_note", None)
    if existing is not None:
        text = f"{existing.get_text()} · {text}"
        existing.remove()
    ax._esd_corner_note = ax.text(
        0.995,
        0.006,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.3",
        zorder=6,
        bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.2},
    )


def _crs_name(crs: Any) -> str:
    """``EPSG:4326``, the CRS name, or its projection (``Sinusoidal``) when unnamed."""
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    name = str(crs.name)
    if name.lower() in ("", "unnamed", "unknown") and crs.coordinate_operation:
        return str(crs.coordinate_operation.method_name)
    return name


def _provider(basemap: Any) -> Any:
    import xyzservices.providers as xyz  # noqa: PLC0415

    if basemap is True or basemap is None:
        basemap = DEFAULT_BASEMAP
    if isinstance(basemap, str):
        provider: Any = xyz
        for part in basemap.split("."):
            provider = getattr(provider, part)
        return provider
    return basemap


def add_basemap(ax: Any, crs: Any, basemap: Any = True, **kwargs: Any) -> None:
    """Draw web tiles under the data on *ax* (``contextily``), warped to *crs*.

    *basemap* is ``True`` for :data:`DEFAULT_BASEMAP`, a dotted
    ``xyzservices`` name such as ``"CartoDB.Positron"`` or
    ``"Esri.WorldImagery"``, or a provider object. The tiles are fetched at
    call time, so this needs the network, and it is off by default for
    rasters (which cover the axes anyway) and on for :func:`points`.
    """
    import contextily as cx  # noqa: PLC0415

    provider = _provider(basemap)
    options: dict[str, Any] = {
        "source": provider,
        "crs": str(crs) if crs is not None else "EPSG:4326",
        # Drawn by _corner_note below, out of the scale bar's way.
        "attribution": False,
        "zorder": 0,
    }
    options.update(kwargs)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    try:
        cx.add_basemap(ax, **options)
    except Exception as exc:  # noqa: BLE001 — a tile server is not worth a failed figure
        warnings.warn(
            f"Could not draw the basemap ({exc}); continuing without it.", stacklevel=2
        )
    else:
        credit = str(getattr(provider, "attribution", "") or "").strip()
        if credit:
            _corner_note(ax, f"basemap: {credit}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def finish_map(
    ax: Any,
    crs: Any = None,
    *,
    scalebar: bool | dict[str, Any] = True,
    graticule: bool | dict[str, Any] = True,
    basemap: Any = False,
    axis_labels: bool = True,
    warn_geographic: bool = True,
) -> Any:
    """Apply the map conventions to *ax*: aspect, labels, scale bar, graticule, basemap.

    Parameters
    ----------
    ax
        Axes something has already been drawn on, so the extent is known.
    crs
        The CRS of the data on the axes (a pyproj ``CRS`` or anything it
        accepts). ``None`` is treated as geographic.
    scalebar, graticule
        ``True``/``False``, or a dict of keyword arguments for
        :func:`add_scalebar` / :func:`add_graticule`.
    basemap
        ``False`` (default), ``True`` for :data:`DEFAULT_BASEMAP`, or a
        provider name / object; see :func:`add_basemap`.
    axis_labels
        Label the axes ``longitude [°]`` / ``latitude [°]`` or
        ``easting [m]`` / ``northing [m]`` from the CRS.
    warn_geographic
        Warn when the data are in degrees: the axes then get a
        latitude-corrected aspect (``1 / cos(latitude)``) so shapes are not
        stretched, but distances still vary across the map.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from pyproj import CRS  # noqa: PLC0415

    crs_obj = None if crs is None else CRS.from_user_input(str(crs))
    geographic = crs_obj is None or crs_obj.is_geographic
    if geographic:
        lat = _mid_latitude(ax, crs_obj)
        correction = 1.0 / max(math.cos(math.radians(lat)), 1e-6)
        ax.set_aspect(correction)
        if warn_geographic:
            warnings.warn(
                f"The data are in a geographic CRS ({_crs_name(crs_obj) if crs_obj else 'degrees'}), "
                f"so one axis unit is a degree, not a metre. The axes use a latitude-corrected "
                f"aspect (1/cos({lat:.1f}°) = {correction:.2f}) so shapes are not stretched; for a "
                "true equal-area grid load the product with crs='utm'.",
                GeographicAxesWarning,
                stacklevel=3,
            )
    else:
        ax.set_aspect("equal")
    if axis_labels:
        xlab, ylab = _axis_names(crs_obj)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    if basemap:
        add_basemap(ax, crs_obj, basemap if basemap is not True else True)
    if graticule:
        add_graticule(ax, crs_obj, **(graticule if isinstance(graticule, dict) else {}))
    if scalebar:
        add_scalebar(ax, crs_obj, **(scalebar if isinstance(scalebar, dict) else {}))
    return ax


# ── figure sizing ─────────────────────────────────────────────────────────────


def _figsize_for(
    extent: tuple[float, float, float, float], correction: float, width: float = 7.0
) -> tuple[float, float]:
    """A figure size whose axes match the data's aspect, so equal aspect fills it."""
    x0, x1, y0, y1 = extent
    w = abs(x1 - x0)
    h = abs(y1 - y0) * correction
    if w <= 0 or h <= 0 or not np.isfinite(w) or not np.isfinite(h):
        return (width, width * 0.75)
    ratio = h / w
    ratio = min(max(ratio, 0.35), 1.6)
    return (width, width * ratio + 0.6)


def _extent(da: xr.DataArray) -> tuple[float, float, float, float]:
    x_dim, y_dim = _spatial_dims(da)
    x = da[x_dim].values
    y = da[y_dim].values
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    for x, y in (("x", "y"), ("longitude", "latitude"), ("lon", "lat")):
        if x in da.dims and y in da.dims:
            return x, y
    dims = [d for d in da.dims]
    if len(dims) >= 2:
        return dims[-1], dims[-2]
    raise ValueError("A map needs two spatial dimensions.")


def _new_axes(da: xr.DataArray, crs: Any, figsize: Any) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if figsize is None:
        correction = 1.0
        if crs is None or crs.is_geographic:
            x0, x1, y0, y1 = _extent(da)
            correction = 1.0 / max(math.cos(math.radians((y0 + y1) / 2)), 1e-6)
        figsize = _figsize_for(_extent(da), correction)
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _squeeze(da: xr.DataArray) -> xr.DataArray:
    """Drop length-one dims (a single time step) so a 2-D image remains."""
    extra = [d for d in da.dims if da.sizes[d] == 1 and d not in _spatial_dims(da)]
    return da.squeeze(extra, drop=False) if extra else da


def _colorbar_axes(ax: Any) -> Any:
    """An axes for a colorbar that hugs the *drawn* map, whatever its aspect.

    ``make_axes_locatable`` positions against the axes' allotted box, so once
    a fixed aspect shrinks the map inside that box the bar drifts away from
    it. An inset in axes coordinates is re-laid out at draw time and stays
    put.
    """
    return ax.inset_axes([1.025, 0.0, 0.035, 1.0])


def _matched_colorbar(ax: Any, mappable: Any, text: str, **kwargs: Any) -> Any:
    """A colorbar the height of the map, however the aspect came out."""
    cbar = ax.figure.colorbar(mappable, cax=_colorbar_axes(ax), **kwargs)
    cbar.set_label(text)
    return cbar


# ── maps ──────────────────────────────────────────────────────────────────────


def map(  # noqa: A001 — the natural name for the natural thing
    da: xr.DataArray,
    ax: Any = None,
    *,
    title: str | None = None,
    cbar_label: str | None = None,
    colorbar: bool = True,
    scalebar: bool | dict[str, Any] = True,
    graticule: bool | dict[str, Any] = True,
    basemap: Any = False,
    figsize: tuple[float, float] | None = None,
    **imshow_kwargs: Any,
) -> Any:
    """Draw one raster as a map with the package's conventions.

    Parameters
    ----------
    da
        A 2-D DataArray (a length-one ``time`` is squeezed) with a CRS on
        ``.rio`` / ``.odc``.
    ax
        Axes to draw on; a new figure sized to the data's aspect otherwise.
    title
        Defaults to the array's ``long_name`` plus its date when it has one.
    cbar_label
        Defaults to :func:`label`, i.e. ``long name [units]``.
    colorbar
        Draw a colorbar matched to the map's height.
    scalebar, graticule, basemap
        See :func:`finish_map`.
    figsize
        Figure size when *ax* is ``None``; derived from the extent otherwise.
    **imshow_kwargs
        Passed to :meth:`xarray.DataArray.plot.imshow` (``cmap``, ``vmin``,
        ``vmax``, ``norm`` …).

    Returns
    -------
    matplotlib.axes.Axes
    """
    squeezed_da = _squeeze(da)
    crs = _crs_of(da)
    if ax is None:
        ax = _new_axes(squeezed_da, crs, figsize)
    imshow_kwargs.setdefault("add_colorbar", False)
    imshow_kwargs.setdefault("add_labels", False)
    image = squeezed_da.plot.imshow(ax=ax, **imshow_kwargs)
    if colorbar:
        _matched_colorbar(
            ax, image, cbar_label if cbar_label is not None else label(da)
        )
    ax.set_title(title if title is not None else _default_title(da))
    finish_map(ax, crs, scalebar=scalebar, graticule=graticule, basemap=basemap)
    return ax


def categorical(
    da: xr.DataArray,
    ax: Any = None,
    *,
    legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
    title: str | None = None,
    scalebar: bool | dict[str, Any] = True,
    graticule: bool | dict[str, Any] = True,
    basemap: Any = False,
    figsize: tuple[float, float] | None = None,
    **imshow_kwargs: Any,
) -> Any:
    """Plot a categorical DataArray with one colour per class and a legend.

    *da* must carry CF ``flag_values`` / ``flag_meanings`` (and ideally
    ``flag_colors``); a ``time`` dimension of length one is squeezed. The
    legend sits outside the axes on the right, listing only the classes
    present unless ``legend_kwargs={"all_classes": True}``.

    Returns the matplotlib ``Axes``.
    """
    squeezed_da = _squeeze(da)
    crs = _crs_of(da)
    if ax is None:
        ax = _new_axes(squeezed_da, crs, figsize)
    cmap, norm, flags_df = colormap_from_flags(da)
    imshow_kwargs.setdefault("add_colorbar", False)
    imshow_kwargs.setdefault("add_labels", False)
    imshow_kwargs.setdefault("interpolation", "nearest")
    squeezed_da.plot.imshow(ax=ax, cmap=cmap, norm=norm, **imshow_kwargs)
    if legend:
        kwargs = dict(_LEGEND_OUTSIDE)
        kwargs.update({"fontsize": 8, "handlelength": 1.2, "borderaxespad": 0.0})
        kwargs.update(legend_kwargs or {})
        all_classes = kwargs.pop("all_classes", False)
        present = None
        if not all_classes:
            try:
                present = set(
                    np.unique(
                        np.asarray(squeezed_da.values)[
                            np.isfinite(np.asarray(squeezed_da.values, dtype="float64"))
                        ]
                    ).tolist()
                )
            except (TypeError, ValueError):
                present = None
        handles, labels = legend_handles(da, present=present)
        ax.legend(handles, labels, **kwargs)
    ax.set_title(title if title is not None else _default_title(da))
    finish_map(ax, crs, scalebar=scalebar, graticule=graticule, basemap=basemap)
    return ax


def points(
    gdf: Any,
    ax: Any = None,
    *,
    column: str | None = None,
    title: str | None = None,
    legend: bool = True,
    legend_label: str | None = None,
    basemap: Any = True,
    scalebar: bool | dict[str, Any] = True,
    graticule: bool | dict[str, Any] = True,
    figsize: tuple[float, float] | None = None,
    pad: float = 0.05,
    **plot_kwargs: Any,
) -> Any:
    """Draw a GeoDataFrame (stations, basins) over a basemap with the map conventions.

    Parameters
    ----------
    gdf
        Any GeoDataFrame; plotted in its own CRS.
    column
        Colour by this column: a colorbar for numbers, a legend for categories.
    legend_label
        Colorbar label; defaults to the column name.
    basemap
        On by default (:data:`DEFAULT_BASEMAP`); see :func:`add_basemap`.
    pad
        Fraction of the extent to add around the points.
    **plot_kwargs
        Passed to :meth:`geopandas.GeoDataFrame.plot` (``markersize``,
        ``cmap``, ``edgecolor`` …).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    crs = _crs_of(gdf)
    x0, y0, x1, y1 = gdf.total_bounds
    if x1 == x0:
        x0, x1 = x0 - 0.05, x1 + 0.05
    if y1 == y0:
        y0, y1 = y0 - 0.05, y1 + 0.05
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    extent = (x0 - dx, x1 + dx, y0 - dy, y1 + dy)
    if ax is None:
        correction = 1.0
        if crs is None or crs.is_geographic:
            correction = 1.0 / max(math.cos(math.radians((y0 + y1) / 2)), 1e-6)
        _, ax = plt.subplots(figsize=figsize or _figsize_for(extent, correction))
    plot_kwargs.setdefault("markersize", 28)
    plot_kwargs.setdefault("edgecolor", "black")
    plot_kwargs.setdefault("linewidth", 0.4)
    plot_kwargs.setdefault("zorder", 4)
    if column is not None and legend:
        if pd.api.types.is_numeric_dtype(gdf[column]):
            gdf.plot(
                ax=ax,
                column=column,
                legend=True,
                cax=_colorbar_axes(ax),
                legend_kwds={"label": legend_label or column},
                **plot_kwargs,
            )
        else:
            kwds = dict(_LEGEND_OUTSIDE)
            kwds.update(
                {"fontsize": 8, "title": legend_label or column, "title_fontsize": 8}
            )
            # geopandas spreads n classes across the whole colormap, which for
            # two classes gives tab10's blue and cyan; take the first n instead.
            if "cmap" not in plot_kwargs:
                from matplotlib.colors import ListedColormap  # noqa: PLC0415

                n_classes = int(gdf[column].nunique(dropna=True))
                base = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
                plot_kwargs["cmap"] = ListedColormap(
                    [base(i) for i in range(max(n_classes, 1))]
                )
            gdf.plot(
                ax=ax,
                column=column,
                categorical=True,
                legend=True,
                legend_kwds=kwds,
                **plot_kwargs,
            )
    else:
        gdf.plot(ax=ax, column=column, **plot_kwargs)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if title is not None:
        ax.set_title(title)
    finish_map(ax, crs, scalebar=scalebar, graticule=graticule, basemap=basemap)
    return ax


# ── time series ───────────────────────────────────────────────────────────────


def _series_label(da: xr.DataArray, key: Any, hue: str) -> str:
    """``"Paradise (1650 m)"`` from the station coordinates when they exist."""
    parts = []
    if "name" in da.coords and hue in da["name"].dims:
        name = str(da["name"].sel({hue: key}).values)
        if name:
            parts.append(name)
    if not parts:
        parts.append(str(key))
    if "elevation_m" in da.coords and hue in da["elevation_m"].dims:
        elev = da["elevation_m"].sel({hue: key}).values
        try:
            if np.isfinite(float(elev)):
                parts.append(f"({float(elev):.0f} m)")
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def _date_axis(ax: Any) -> None:
    import matplotlib.dates as mdates  # noqa: PLC0415

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _water_year_axis(ax: Any) -> None:
    """Month names, October to September, on an axis that holds a synthetic year."""
    import matplotlib.dates as mdates  # noqa: PLC0415

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def timeseries(
    obj: xr.DataArray | xr.Dataset,
    ax: Any = None,
    *,
    variable: str | None = None,
    hue: str | None = None,
    labels: dict[Any, str] | None = None,
    title: str | None = None,
    by_water_year: bool = False,
    hemisphere: str = "northern",
    legend: bool = True,
    figsize: tuple[float, float] = (9, 4),
    **plot_kwargs: Any,
) -> Any:
    """Plot a time series, one line per *hue* value, against calendar dates.

    Parameters
    ----------
    obj
        A DataArray with a ``time`` dim, or a Dataset with *variable* named.
    hue
        Dimension to draw one line per value of (default: the non-time dim
        when there is exactly one, e.g. ``station``).
    labels
        ``{hue value: legend text}``; defaults to the ``name`` and
        ``elevation_m`` coordinates when present, else the value itself.
    by_water_year
        Overlay each water year on one October-to-September axis (month
        names) instead of a continuous calendar. The lines are labelled by
        water year, and *hue* must then be absent or of length one.
    hemisphere
        For *by_water_year*: ``"northern"`` (1 October) or ``"southern"``.
    **plot_kwargs
        Passed to ``Axes.plot`` (``linewidth``, ``alpha``, ``color`` …).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from easysnowdata.processing import wateryear  # noqa: PLC0415

    if isinstance(obj, xr.Dataset):
        if variable is None:
            if len(obj.data_vars) != 1:
                raise ValueError("Pass variable= to choose a Dataset variable.")
            variable = str(next(iter(obj.data_vars)))
        da = obj[variable]
    else:
        da = obj
    if "time" not in da.dims:
        raise ValueError("timeseries() needs a `time` dimension.")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    others = [d for d in da.dims if d != "time"]
    if hue is None and len(others) == 1:
        hue = others[0]
    if others and hue not in others:
        raise ValueError(f"hue must be one of {others}, got {hue!r}.")

    times = pd.DatetimeIndex(da["time"].values)
    plot_kwargs.setdefault("linewidth", 1.4)
    if by_water_year:
        series_da = da.squeeze() if hue and da.sizes[hue] == 1 else da
        if series_da.ndim != 1:
            raise ValueError(
                "by_water_year overlays one series; select one station first."
            )
        years = wateryear.water_year(times, hemisphere)
        dowy = wateryear.day_of_water_year(times, hemisphere)
        # A synthetic, non-leap water year to hang the month axis on.
        start = pd.Timestamp("2001-10-01" if hemisphere == "northern" else "2001-04-01")
        for year in np.unique(years):
            rows = years == year
            x = start + pd.to_timedelta(np.asarray(dowy)[rows] - 1, unit="D")
            ax.plot(x, series_da.values[rows], label=f"WY{int(year)}", **plot_kwargs)
        _water_year_axis(ax)
        ax.set_xlim(start, start + pd.Timedelta(days=365))
        ax.set_xlabel(
            "water year (1 October to 30 September)"
            if hemisphere == "northern"
            else "water year (1 April to 31 March)"
        )
    elif hue is None:
        ax.plot(times, da.values, **plot_kwargs)
        _date_axis(ax)
    else:
        for key in da[hue].values:
            text = (labels or {}).get(key, _series_label(da, key, hue))
            ax.plot(times, da.sel({hue: key}).values, label=text, **plot_kwargs)
        _date_axis(ax)
    ax.set_ylabel(label(da))
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title is not None:
        ax.set_title(title)
    elif da.attrs.get("long_name"):
        ax.set_title(str(da.attrs["long_name"]))
    if legend and ax.get_legend_handles_labels()[0]:
        n_lines = len(ax.get_legend_handles_labels()[0])
        if n_lines > 4:
            ax.legend(fontsize=8, **_LEGEND_OUTSIDE)
        else:
            ax.legend(fontsize=8, frameon=False, loc="best")
    return ax


# ── flags → colours ───────────────────────────────────────────────────────────


def colormap_from_flags(obj: xr.DataArray | dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return ``(cmap, norm, flags_df)`` for a categorical array's CF flags.

    The norm maps each ``flag_value`` to its own colour regardless of gaps
    between values (5, 37, 200 …).
    """
    import matplotlib.colors as mcolors  # noqa: PLC0415

    flags_df = flags(obj).sort_values("value").reset_index(drop=True)
    colors = [c if c is not None else "#808080" for c in flags_df["color"]]
    cmap = mcolors.ListedColormap(colors, name="easysnowdata_flags")
    # BoundaryNorm sends values outside the flag range to the colormap's
    # under/over colours, which default to the first and last class. A nodata
    # sentinel such as 0 (ocean) would then be painted as class one.
    cmap.set_under("none")
    cmap.set_over("none")
    values = flags_df["value"].to_numpy(dtype="float64")
    edges = np.concatenate(
        [[values[0] - 0.5], (values[:-1] + values[1:]) / 2, [values[-1] + 0.5]]
    )
    norm = mcolors.BoundaryNorm(edges, cmap.N)
    return cmap, norm, flags_df


def legend_handles(
    obj: xr.DataArray | dict[str, Any], *, present: set[Any] | None = None
) -> tuple[list[Any], list[str]]:
    """``(handles, labels)`` for ``ax.legend`` from the CF flags (underscores → spaces).

    *present* restricts the legend to those flag values (the classes that
    actually occur in the plotted array).
    """
    from matplotlib.patches import Patch  # noqa: PLC0415

    flags_df = flags(obj).sort_values("value")
    if present is not None:
        keep = flags_df["value"].isin(list(present))
        if keep.any():
            flags_df = flags_df[keep]
    handles = [
        Patch(facecolor=c or "#808080", edgecolor="black", linewidth=0.4)
        for c in flags_df["color"]
    ]
    labels = [str(m).replace("_", " ") for m in flags_df["meaning"]]
    return handles, labels


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
