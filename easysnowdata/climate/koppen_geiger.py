"""Köppen-Geiger climate classification (Beck et al. 2023).

One source: the figshare archive of the 2023 paper, file **61012822** (the
January 2026 release; file 45057352 was the superseded v1). The archive holds
five historical periods and the 2041-2070 / 2071-2099 CMIP6 projections at
four resolutions, so this loader exposes ``period=`` and ``scenario=``
alongside ``resolution=``::

    import easysnowdata as esd
    kg = esd.climate.koppen_geiger.load(aoi)                      # 1991-2020, 0.1°
    kg = esd.climate.koppen_geiger.load(aoi, period="2071_2099",
                                        scenario="ssp245", resolution="1 km")
    esd.climate.koppen_geiger.search()                            # what is in the archive
    esd.plotting.categorical(kg)                                  # CF flags → legend

The 30 classes are returned as the source ``uint8`` values with CF
``flag_values`` / ``flag_meanings`` / ``flag_colors``; 0 is ocean and is kept
as the nodata sentinel (§2.5).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT",
    "ZIP_URL",
    "CLASSES",
    "PERIODS",
    "SCENARIOS",
    "RESOLUTIONS",
    "search",
    "load",
]

_logger = logging.getLogger(__name__)

#: figshare file 61012822 — the current release of the Beck et al. (2023) archive.
ZIP_URL = "https://ndownloader.figshare.com/files/61012822/koppen_geiger_tif.zip"
ARTICLE_URL = "https://doi.org/10.1038/s41597-023-02549-6"

#: Historical periods (no scenario) and future periods (one per SSP).
HISTORICAL_PERIODS = ("1901_1930", "1931_1960", "1961_1990", "1991_2020")
FUTURE_PERIODS = ("2041_2070", "2071_2099")
PERIODS = HISTORICAL_PERIODS + FUTURE_PERIODS
SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp585")

#: Human names → the resolution token in the archive's file names.
RESOLUTIONS: dict[str, str] = {
    "1 degree": "1p0",
    "0.5 degree": "0p5",
    "0.1 degree": "0p1",
    "1 km": "0p00833333",
}
_RESOLUTION_M = {"1p0": 111000.0, "0p5": 55500.0, "0p1": 11100.0, "0p00833333": 1000.0}

#: class value → (symbol, description, hex colour), from the paper's legend.
CLASSES: dict[int, tuple[str, str, str]] = {
    1: ("Af", "Tropical, rainforest", "#0000ff"),
    2: ("Am", "Tropical, monsoon", "#0078ff"),
    3: ("Aw", "Tropical, savannah", "#46aafa"),
    4: ("BWh", "Arid, desert, hot", "#ff0000"),
    5: ("BWk", "Arid, desert, cold", "#ff9696"),
    6: ("BSh", "Arid, steppe, hot", "#f5a500"),
    7: ("BSk", "Arid, steppe, cold", "#ffdc64"),
    8: ("Csa", "Temperate, dry summer, hot summer", "#ffff00"),
    9: ("Csb", "Temperate, dry summer, warm summer", "#c8c800"),
    10: ("Csc", "Temperate, dry summer, cold summer", "#969600"),
    11: ("Cwa", "Temperate, dry winter, hot summer", "#96ff96"),
    12: ("Cwb", "Temperate, dry winter, warm summer", "#64c864"),
    13: ("Cwc", "Temperate, dry winter, cold summer", "#329632"),
    14: ("Cfa", "Temperate, no dry season, hot summer", "#c8ff50"),
    15: ("Cfb", "Temperate, no dry season, warm summer", "#64ff50"),
    16: ("Cfc", "Temperate, no dry season, cold summer", "#32c800"),
    17: ("Dsa", "Cold, dry summer, hot summer", "#ff00ff"),
    18: ("Dsb", "Cold, dry summer, warm summer", "#c800c8"),
    19: ("Dsc", "Cold, dry summer, cold summer", "#963296"),
    20: ("Dsd", "Cold, dry summer, very cold winter", "#966496"),
    21: ("Dwa", "Cold, dry winter, hot summer", "#aaafff"),
    22: ("Dwb", "Cold, dry winter, warm summer", "#5a78dc"),
    23: ("Dwc", "Cold, dry winter, cold summer", "#4b50b4"),
    24: ("Dwd", "Cold, dry winter, very cold winter", "#320087"),
    25: ("Dfa", "Cold, no dry season, hot summer", "#00ffff"),
    26: ("Dfb", "Cold, no dry season, warm summer", "#37c8ff"),
    27: ("Dfc", "Cold, no dry season, cold summer", "#007d7d"),
    28: ("Dfd", "Cold, no dry season, very cold winter", "#00465f"),
    29: ("ET", "Polar, tundra", "#b2b2b2"),
    30: ("EF", "Polar, frost", "#666666"),
}

_CLASS_VARIABLE = Variable(
    "koppen_geiger_class",
    dtype="uint8",
    nodata=0,
    long_name="Köppen-Geiger climate class",
    flag_values=tuple(CLASSES),
    flag_meanings=tuple(symbol for symbol, _, _ in CLASSES.values()),
    flag_colors=tuple(color for _, _, color in CLASSES.values()),
)

PRODUCT = Product(
    id="koppen-geiger",
    theme="climate",
    title="Köppen-Geiger climate classification (Beck et al. 2023)",
    description=(
        "Global Köppen-Geiger climate classes at 1 km to 1°, for five historical "
        "30-year periods (1901-1930 … 1991-2020) and the 2041-2070 / 2071-2099 "
        "CMIP6 projections under seven SSPs."
    ),
    sources=(
        Source(
            id="figshare",
            provider="raster_http",
            location=ZIP_URL,
            resolution_m=11100,
            temporal="1901-1930 … 1991-2020; 2041-2070 and 2071-2099 projections",
            latency="static (2026-01 release)",
            notes=(
                "file 61012822 is the current release (45057352 was v1); read through "
                "ndownloader.figshare.com, which answers with a plain 302 (the "
                "figshare.com/ndownloader host bot-challenges non-browser clients)"
            ),
            title="figshare zip",
            health=Probe(
                "Köppen-Geiger classification (figshare)",
                partial(health.http_first_byte, ZIP_URL),
            ),
        ),
    ),
    variables=(_CLASS_VARIABLE,),
    citation=(
        "Beck, H. E., McVicar, T. R., Vergopolan, N., et al. (2023). High-resolution (1 km) "
        "Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. "
        "Scientific Data 10, 724."
    ),
    license="CC BY 4.0",
    doi="10.1038/s41597-023-02549-6",
    references=(ARTICLE_URL,),
    loader="easysnowdata.climate.koppen_geiger.load",
    tags=("climate classification", "koppen"),
)
catalog.register(PRODUCT, replace=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _resolution_token(resolution: str) -> str:
    if resolution in RESOLUTIONS:
        return RESOLUTIONS[resolution]
    if resolution in RESOLUTIONS.values():
        return resolution
    raise ValueError(
        f"resolution must be one of {list(RESOLUTIONS)}, got {resolution!r}."
    )


def _member(period: str, scenario: str | None, token: str) -> str:
    """The path of one GeoTIFF inside the archive."""
    period = str(period).replace("-", "_")
    if period in HISTORICAL_PERIODS:
        if scenario is not None:
            raise ValueError(
                f"period={period!r} is historical and takes no scenario; "
                f"scenarios apply to {FUTURE_PERIODS}."
            )
        return f"{period}/koppen_geiger_{token}.tif"
    if period in FUTURE_PERIODS:
        scenario = "ssp245" if scenario is None else str(scenario).lower()
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}.")
        return f"{period}/{scenario}/koppen_geiger_{token}.tif"
    raise ValueError(f"period must be one of {PERIODS}, got {period!r}.")


def class_table() -> pd.DataFrame:
    """The 30 classes as a DataFrame (``value``, ``symbol``, ``description``, ``color``)."""
    return pd.DataFrame(
        [
            {
                "value": value,
                "symbol": symbol,
                "description": description,
                "color": color,
            }
            for value, (symbol, description, color) in CLASSES.items()
        ]
    ).set_index("value")


# ── public API ────────────────────────────────────────────────────────────────


def search(
    aoi: Any = None, time: Any = None, *, source: str | None = None
) -> pd.DataFrame:
    """What the archive holds: one row per (period, scenario, resolution) raster.

    Nothing is downloaded — the layout is fixed by the release and is listed
    here so ``load`` arguments can be discovered without opening the 1.3 GB zip.
    """
    src = resolve_source(PRODUCT, source)
    if aoi is not None:
        parse_aoi(aoi)
    rows = []
    for period in PERIODS:
        scenarios = (None,) if period in HISTORICAL_PERIODS else SCENARIOS
        for scenario in scenarios:
            for name, token in RESOLUTIONS.items():
                rows.append(
                    {
                        "period": period,
                        "scenario": scenario,
                        "resolution": name,
                        "resolution_m": _RESOLUTION_M[token],
                        "member": _member(period, scenario, token),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.attrs = {"source": src.id, "url": ZIP_URL}
    return frame


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    period: str = "1991_2020",
    scenario: str | None = None,
    resolution: str = "0.1 degree",
    source: str | None = None,
    chunks: Any = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Load the Köppen-Geiger classes as a categorical ``xarray.DataArray``.

    Parameters
    ----------
    aoi
        Any form :func:`~easysnowdata.aoi.parse_aoi` accepts; ``None`` is global.
    time
        Ignored: the product is a set of fixed 30-year periods. Use *period*.
    period
        One of :data:`PERIODS` (``"1991_2020"`` by default).
    scenario
        SSP for a future *period* (default ``"ssp245"``); must be ``None`` for
        a historical one.
    resolution
        ``"1 degree"``, ``"0.5 degree"``, ``"0.1 degree"`` or ``"1 km"``.
    chunks
        Passed to ``rioxarray.open_rasterio``; ``True`` (default) gives a
        Dask-backed array.
    **kwargs
        Forwarded to ``rioxarray.open_rasterio``.

    Returns
    -------
    xarray.DataArray
        ``uint8`` classes with CF flag attrs; 0 (ocean) is kept as the nodata
        sentinel, so :func:`easysnowdata.plotting.categorical` can draw it.
    """
    if time is not None:
        _logger.debug("koppen_geiger.load ignores time=%r; use period=.", time)
    src = resolve_source(PRODUCT, source)
    token = _resolution_token(resolution)
    member = _member(period, scenario, token)
    parsed = parse_aoi(aoi) if aoi is not None else None
    ensure_source(PRODUCT, src)

    da = providers.raster_http.open(
        providers.raster_http.zip_url(ZIP_URL, member),
        None,  # clipped below: a small AOI can be narrower than one 1° cell
        chunks=chunks,
        requires=src.requires,
        **kwargs,
    )
    if parsed is not None and parsed.clip and not parsed.is_global:
        da = da.rio.clip_box(
            *parsed.footprint.total_bounds,
            crs=parsed.footprint.crs,
            allow_one_dimensional_raster=True,
        )
    da = da.rename(_CLASS_VARIABLE.name)
    da = contract.write_crs(da, da.rio.crs or "EPSG:4326")
    da = contract.set_categorical_nodata(da, _CLASS_VARIABLE.nodata)
    da.attrs.update(_CLASS_VARIABLE.cf_attrs())
    period_label = str(period).replace("-", "_")
    da = contract.finalize(
        da,
        PRODUCT,
        src,
        source_url=ARTICLE_URL,
        attrs={
            "period": period_label,
            "scenario": None
            if period_label in HISTORICAL_PERIODS
            else (scenario or "ssp245"),
            "resolution": resolution,
            "archive_member": member,
        },
    )
    return da
