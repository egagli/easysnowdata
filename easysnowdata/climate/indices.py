"""Climate-mode indices: PDO, ENSO (RONI, ONI, Niño 3.4, MEI.v2), AO, NAO, PNA, AMO.

Small monthly time series, one number per month, read straight from the
providers' text files. The same name covers several different series, so
the default source, ``"noaa"``, reads each index from the provider that
maintains it rather than from a mirror::

    import easysnowdata as esd

    idx_ds = esd.climate.indices.load(indices=["pdo", "roni"], time="1950/..")
    esd.climate.indices.search()                     # what is offered, and from where
    winter_df = esd.climate.indices.seasonal_mean(idx_ds)        # Nov-Mar, by water year
    phase_da = esd.climate.indices.enso_phase(idx_ds["roni"])    # CPC's event rule

Which file each index comes from, and why (checked 2026-10-09):

* **RONI** is CPC's official ENSO index since 2026-02-01 (NWS PNS 26-05); the
  legacy **ONI** and the monthly **Niño 3.4** are still published for
  continuity. All three are built on ERSSTv6. RONI and ONI now disagree by
  up to half a degree, because RONI subtracts the tropical-mean warming.
* **PDO** and **AMO** come from NCEI's ERSSTv5 files. NOAA names no official
  PDO; NCEI's is the maintained, documented, 1854-onward default. PSL's
  ``pdo.data`` stopped at 2025-08 and its Kaplan AMO at 2023-01.
* **MEI.v2** has one home, PSL. **AO**, **NAO** and **PNA** come from CPC.

``source="psl"`` reads PSL's correlation-page copies instead (``pdo.data``,
``oni.data``, ``nina34.anom.data`` …), for reproducing work that used them.
They are not the same series: PSL's PDO correlates at about 0.92 with NCEI's,
and two of them have stopped updating, which ``load`` logs as a warning.

Timestamps are the first of the month. A 3-month-season index (RONI, ONI) is
stamped at its centre month (DJF → January), and MEI.v2's 2-month seasons at
their later month (DJ → January), which is how both providers lay out their
own year-by-month tables. JPL's "Vital Signs" PDO is a different quantity, a
sea-surface-*height* index, and is not offered here; see the best-practices
page linked in the catalog entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from easysnowdata import catalog, providers
from easysnowdata.catalog import Probe, Product, Source, Variable, health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.processing import contract, wateryear
from easysnowdata.temporal import now, parse_time

__all__ = [
    "PRODUCT",
    "INDICES",
    "SOURCES",
    "search",
    "load",
    "parse",
    "seasonal_mean",
    "enso_phase",
]

_logger = logging.getLogger(__name__)

CPC = "https://www.cpc.ncep.noaa.gov"
NCEI = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index"
PSL = "https://psl.noaa.gov/data/correlation"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
GUIDE_URL = (
    "https://github.com/egagli/geospatial_data_and_visualization_best_practices/"
    "blob/main/data-access/climate-indices.md"
)

#: A file whose newest value is older than this many days is reported as stale.
STALE_AFTER_DAYS = 120
#: Values at or beyond this magnitude are missing-value sentinels (99.99,
#: -99.9, -999 …); no index here comes within an order of magnitude of it.
#: PSL's year-by-month files also declare their own sentinel on the line after
#: the table, and some are small: ``pdo.data`` uses -9.90, which :func:`parse`
#: reads and drops too.
SENTINEL_ABS = 90.0

#: Season label → the month it is stamped at (3-month: centre; 2-month: later).
SEASON_MONTH: dict[str, int] = {
    label: i + 1
    for i, label in enumerate(
        ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND")
        + ("NDJ",)
    )
}


@dataclass(frozen=True)
class _File:
    """One index file: where it is, how it is laid out, and what it holds."""

    url: str
    layout: str  # "columns" | "seasons" | "wide"
    provider: str
    notes: str = ""


@dataclass(frozen=True)
class _Index:
    name: str
    long_name: str
    units: str
    cadence: str
    files: dict[str, _File]


INDICES: dict[str, _Index] = {
    index.name: index
    for index in (
        _Index(
            "pdo",
            "Pacific Decadal Oscillation index",
            "1",
            "monthly",
            {
                "noaa": _File(
                    f"{NCEI}/ersst.v5.pdo.dat",
                    "wide",
                    "NOAA NCEI",
                    "ERSSTv5 anomalies projected onto the Mantua PDO pattern; 1854-",
                ),
                "psl": _File(
                    f"{PSL}/pdo.data",
                    "wide",
                    "NOAA PSL",
                    "last value 2025-08; r≈0.92 with NCEI's series",
                ),
            },
        ),
        _Index(
            "roni",
            "Relative Oceanic Niño Index (CPC's official ENSO index since 2026-02)",
            "degC",
            "3-month season, stamped at the centre month",
            {
                "noaa": _File(
                    f"{CPC}/data/indices/RONI.ascii.txt",
                    "seasons",
                    "NOAA CPC",
                    "ERSSTv6 Niño 3.4 minus the 20°N-20°S mean, rescaled; 1950-",
                ),
            },
        ),
        _Index(
            "oni",
            "Oceanic Niño Index (legacy ENSO index)",
            "degC",
            "3-month season, stamped at the centre month",
            {
                "noaa": _File(
                    f"{CPC}/data/indices/oni.ascii.txt",
                    "seasons",
                    "NOAA CPC",
                    "ERSSTv6 Niño 3.4, centred 30-year base periods; 1950-",
                ),
                "psl": _File(
                    f"{PSL}/oni.data",
                    "wide",
                    "NOAA PSL",
                    "splices ERSSTv5 history with v6 recent months; differs from CPC "
                    "by up to 0.44 °C",
                ),
            },
        ),
        _Index(
            "nino34",
            "Niño 3.4 SST anomaly (monthly)",
            "degC",
            "monthly",
            {
                "noaa": _File(
                    f"{CPC}/data/indices/detrend.nino34.ascii.txt",
                    "columns",
                    "NOAA CPC",
                    "ERSSTv6, the monthly input to the ONI; 1950-",
                ),
                "psl": _File(
                    f"{PSL}/nina34.anom.data",
                    "wide",
                    "NOAA PSL",
                    "PSL's own Niño 3.4 anomaly series; 1948-",
                ),
            },
        ),
        _Index(
            "rnino34",
            "Relative Niño 3.4 SST anomaly (monthly)",
            "degC",
            "monthly",
            {
                "noaa": _File(
                    f"{CPC}/data/indices/Rnino34.ascii.txt",
                    "columns",
                    "NOAA CPC",
                    "ERSSTv6, the monthly input to RONI; 1950-",
                ),
            },
        ),
        _Index(
            "mei",
            "Multivariate ENSO Index version 2",
            "1",
            "2-month season, stamped at the later month",
            {
                "noaa": _File(
                    MEI_URL,
                    "wide",
                    "NOAA PSL",
                    "leading combined EOF of SLP, SST, winds and OLR; 1979-",
                ),
                "psl": _File(MEI_URL, "wide", "NOAA PSL", "the same file"),
            },
        ),
        _Index(
            "ao",
            "Arctic Oscillation index",
            "1",
            "monthly",
            {
                "noaa": _File(
                    f"{CPC}/products/precip/CWlink/daily_ao_index/"
                    "monthly.ao.index.b50.current.ascii",
                    "columns",
                    "NOAA CPC",
                    "1000 hPa height EOF, normalized by 1979-2000; 1950-",
                ),
                "psl": _File(f"{PSL}/ao.data", "wide", "NOAA PSL", "copy of CPC's"),
            },
        ),
        _Index(
            "nao",
            "North Atlantic Oscillation index",
            "1",
            "monthly",
            {
                "noaa": _File(
                    f"{CPC}/products/precip/CWlink/pna/"
                    "norm.nao.monthly.b5001.current.ascii",
                    "columns",
                    "NOAA CPC",
                    "rotated PCA of 500 hPa heights; 1950-",
                ),
                "psl": _File(
                    f"{PSL}/nao.data",
                    "wide",
                    "NOAA PSL",
                    "copy of CPC's; updates later than CPC",
                ),
            },
        ),
        _Index(
            "pna",
            "Pacific/North American pattern index",
            "1",
            "monthly",
            {
                "noaa": _File(
                    f"{CPC}/products/precip/CWlink/pna/"
                    "norm.pna.monthly.b5001.current.ascii",
                    "columns",
                    "NOAA CPC",
                    "rotated PCA of 500 hPa heights; 1950-",
                ),
                "psl": _File(
                    f"{PSL}/pna.data",
                    "wide",
                    "NOAA PSL",
                    "copy of CPC's; updates later than CPC",
                ),
            },
        ),
        _Index(
            "amo",
            "Atlantic Multidecadal Oscillation (North Atlantic 0-60°N SST anomaly)",
            "degC",
            "monthly",
            {
                "noaa": _File(
                    f"{NCEI}/ersst.v5.amo.dat",
                    "columns",
                    "NOAA NCEI",
                    "ERSSTv5, 1971-2000 climatology, not detrended; 1854-",
                ),
                "psl": _File(
                    f"{PSL}/amon.us.data",
                    "wide",
                    "NOAA PSL",
                    "Kaplan SST, detrended; stopped at 2023-01",
                ),
            },
        ),
    )
}

#: Source id → the indices it offers.
SOURCES: dict[str, tuple[str, ...]] = {
    "noaa": tuple(n for n, i in INDICES.items() if "noaa" in i.files),
    "psl": tuple(n for n, i in INDICES.items() if "psl" in i.files),
}

DEFAULT_INDICES = ("pdo", "roni", "oni", "nino34", "mei", "ao", "nao", "pna", "amo")


def _latest_pdo() -> str:
    """The newest month in NCEI's PDO file (the status page's freshness row)."""
    series = _read("pdo", "noaa", max_age=0)
    return health._iso(series.index.max())


PRODUCT = Product(
    id="climate-indices",
    theme="climate",
    title="Climate-mode indices: PDO, ENSO, AO, NAO, PNA, AMO (NOAA)",
    description=(
        "Monthly climate-mode indices from the NOAA offices that maintain them: the "
        "PDO and AMO from NCEI (ERSSTv5), RONI (CPC's official ENSO index since "
        "February 2026), the legacy ONI, Niño 3.4 and relative Niño 3.4, AO, NAO and "
        "PNA from CPC, and MEI.v2 from PSL. Interannual context for snowpack and "
        "melt timing."
    ),
    sources=(
        Source(
            id="noaa",
            provider="table_http",
            location=f"{CPC}/data/indices/",
            extent="global (index time series)",
            temporal="1854- (PDO, AMO); 1950- (CPC); 1979- (MEI.v2)",
            latency="monthly, by about the 10th",
            notes=(
                "each index from the office that maintains it (NCEI ERSSTv5 PDO/AMO; "
                "CPC RONI, ONI, Niño 3.4, AO, NAO, PNA; PSL MEI.v2); the newest CPC "
                "ENSO values can be revised for two months"
            ),
            title="NOAA CPC / NCEI / PSL text files",
            health=(
                Probe(
                    "Climate indices (CPC RONI)",
                    partial(
                        health.http_first_byte, f"{CPC}/data/indices/RONI.ascii.txt"
                    ),
                ),
                Probe(
                    "Climate indices (NCEI PDO)",
                    partial(health.http_first_byte, f"{NCEI}/ersst.v5.pdo.dat"),
                ),
                Probe(
                    "Climate indices (PSL MEI.v2)",
                    partial(health.http_first_byte, MEI_URL),
                ),
                Probe("Climate indices (NCEI PDO latest)", _latest_pdo, kind="latency"),
            ),
        ),
        Source(
            id="psl",
            provider="table_http",
            location=f"{PSL}/",
            extent="global (index time series)",
            temporal="1948- (most); PDO to 2025-08; AMO to 2023-01",
            latency="monthly; some files no longer updated",
            notes=(
                "PSL's correlation-page copies, for reproducing work that used them; "
                "its PDO and AMO are different series and have stopped, and its ONI "
                "splices ERSSTv5 and v6 — not a default"
            ),
            title="NOAA PSL correlation-page files",
            health=Probe(
                "Climate indices (PSL pdo.data)",
                partial(health.http_first_byte, f"{PSL}/pdo.data"),
            ),
        ),
    ),
    variables=tuple(
        Variable(
            index.name, units=index.units, dtype="float64", long_name=index.long_name
        )
        for index in INDICES.values()
    ),
    citation=(
        "NOAA Climate Prediction Center (RONI, ONI, Niño 3.4, AO, NAO, PNA); NOAA NCEI "
        "ERSSTv5 PDO and AMO indices (Huang et al. 2017, doi:10.1175/JCLI-D-16-0836.1); "
        "Wolter & Timlin (2011) and Zhang et al. (2019) for MEI.v2; Mantua et al. (1997) "
        "and Newman et al. (2016) for the PDO."
    ),
    license="public domain (U.S. Government work)",
    references=(
        f"{CPC}/products/analysis_monitoring/enso/roni/",
        "https://www.weather.gov/media/notification/pdf_2026/pns26-05_Relative_ONI.pdf",
        "https://www.ncei.noaa.gov/access/monitoring/pdo/",
        "https://psl.noaa.gov/enso/mei/",
        "https://psl.noaa.gov/data/climateindices/list/",
        GUIDE_URL,
    ),
    loader="easysnowdata.climate.indices.load",
    examples=("climate/plot_climate_indices.py",),
    tags=("pdo", "enso", "roni", "oni", "nino34", "mei", "ao", "nao", "pna", "amo"),
)
catalog.register(PRODUCT, replace=True)


# ── parsing (pure) ────────────────────────────────────────────────────────────


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit()


def _float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def parse(text: str, layout: str, *, name: str | None = None) -> pd.Series:
    """Parse one index file into a monthly ``pandas.Series``.

    Parameters
    ----------
    text
        The file's contents.
    layout
        ``"columns"`` — one row per month, ``year month … value`` with the
        value in the last column (CPC's monthly files, NCEI's AMO; header
        lines are skipped). ``"seasons"`` — ``SEAS YEAR … value`` with a
        3-letter season label (CPC's RONI and ONI). ``"wide"`` — one row per
        year with twelve monthly values (NCEI's PDO and every PSL file; the
        PSL first line of two years, the sentinel line and the notes are
        skipped because they do not have thirteen numbers). A line holding a
        single number after the table is PSL's missing-value sentinel, and
        values equal to it are dropped.
    name
        Name for the returned series.

    Returns
    -------
    pandas.Series
        Float values on a month-start ``DatetimeIndex``, missing-value
        sentinels dropped, sorted.
    """
    records: list[tuple[int, int, float]] = []
    sentinel: float | None = None
    for line in text.splitlines():
        tokens = line.split()
        if layout == "wide" and len(tokens) == 1 and records and sentinel is None:
            sentinel = _float(tokens[0])
        if len(tokens) < 3:
            continue
        if layout == "seasons":
            month = SEASON_MONTH.get(tokens[0].upper())
            value = _float(tokens[-1])
            if month is not None and _is_year(tokens[1]) and value is not None:
                records.append((int(tokens[1]), month, value))
        elif layout == "columns":
            value = _float(tokens[-1])
            if _is_year(tokens[0]) and tokens[1].isdigit() and value is not None:
                month = int(tokens[1])
                if 1 <= month <= 12:
                    records.append((int(tokens[0]), month, value))
        elif layout == "wide":
            values = [_float(t) for t in tokens[1:]]
            if _is_year(tokens[0]) and len(values) == 12 and None not in values:
                year = int(tokens[0])
                records.extend(
                    (year, month, value) for month, value in enumerate(values, start=1)
                )
        else:
            raise ValueError(
                f"layout must be 'columns', 'seasons' or 'wide', got {layout!r}."
            )
    if not records:
        raise ValueError(f"No {layout!r} rows found in the {name or 'index'} file.")
    frame = pd.DataFrame(records, columns=["year", "month", "value"])
    index = pd.to_datetime(frame[["year", "month"]].assign(day=1))
    series = pd.Series(frame["value"].to_numpy(dtype=float), index=index, name=name)
    series = series[series.abs() < SENTINEL_ABS]
    if sentinel is not None:
        series = series[~np.isclose(series, sentinel)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.index.name = "time"
    return series


# ── access ────────────────────────────────────────────────────────────────────


def _file(name: str, source_id: str) -> _File:
    if name not in INDICES:
        raise ValueError(f"Unknown index {name!r}; choose from {', '.join(INDICES)}.")
    files = INDICES[name].files
    if source_id not in files:
        offered = [s for s in SOURCES if name in SOURCES[s]]
        raise ValueError(
            f"Index {name!r} is not offered by source {source_id!r}; "
            f"it is offered by {', '.join(offered)}."
        )
    return files[source_id]


def _read(name: str, source_id: str, *, max_age: float) -> pd.Series:
    spec = _file(name, source_id)
    path: Path = providers.table_http.fetch(
        spec.url,
        f"{source_id}_{Path(spec.url).name}",
        subdir="climate_indices",
        max_age=max_age,
    )
    return parse(path.read_text(errors="replace"), spec.layout, name=name)


def search(
    aoi: Any = None, time: Any = None, *, source: str | None = None
) -> pd.DataFrame:
    """The indices a source offers, one row each, with the file each comes from.

    Nothing is downloaded. *aoi* and *time* are accepted for a uniform
    signature and ignored: an index is one global time series.
    """
    src = resolve_source(PRODUCT, source)
    rows = []
    for name in SOURCES[src.id]:
        index, spec = INDICES[name], INDICES[name].files[src.id]
        rows.append(
            {
                "index": name,
                "long_name": index.long_name,
                "units": index.units,
                "cadence": index.cadence,
                "provider": spec.provider,
                "url": spec.url,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows).set_index("index")


def load(
    aoi: Any = None,
    time: Any = None,
    *,
    indices: str | list[str] | tuple[str, ...] | None = None,
    source: str | None = None,
    max_age: float = 86400,
) -> xr.Dataset:
    """Load climate-mode indices as a monthly ``xarray.Dataset``.

    Parameters
    ----------
    aoi
        Ignored (an index is one global series); accepted for a uniform
        signature.
    time
        Any form :func:`~easysnowdata.temporal.parse_time` accepts, e.g.
        ``"1950/.."`` or ``("1981-10", "2024-09")``; ``None`` is the whole
        record.
    indices
        One name or a list from :data:`INDICES`; ``None`` loads every index
        the source offers.
    source
        ``"noaa"`` (default: each index from the office that maintains it) or
        ``"psl"`` (PSL's correlation-page copies; see the module docstring).
    max_age
        Seconds before a cached file is downloaded again (default one day;
        the providers update monthly and revise recent values).

    Returns
    -------
    xarray.Dataset
        One ``float64`` variable per index on a month-start ``time``
        dimension, NaN where an index has no value. Each variable's attrs give
        its ``units``, ``long_name``, ``cadence``, ``provider``, ``source_url``
        and the ``latest`` month in the file; the dataset carries the
        package's provenance attrs. Small and read eagerly.
    """
    src = resolve_source(PRODUCT, source)
    if aoi is not None:
        _logger.debug("climate.indices.load ignores aoi=%r.", aoi)
    if indices is None:
        names = [n for n in DEFAULT_INDICES if n in SOURCES[src.id]]
    elif isinstance(indices, str):
        names = [indices]
    else:
        names = list(indices)
    for name in names:
        _file(name, src.id)  # fail on a bad name before downloading anything

    start, end = parse_time(time)
    stale_before = now() - pd.Timedelta(days=STALE_AFTER_DAYS)
    data_vars = {}
    for name in names:
        index, spec = INDICES[name], INDICES[name].files[src.id]
        series = _read(name, src.id, max_age=max_age)
        latest = series.index.max()
        if latest < stale_before:
            _logger.warning(
                "%s from source %r ends %s and is no longer updated%s.",
                name,
                src.id,
                latest.strftime("%Y-%m"),
                "; source='noaa' has a current series"
                if "noaa" in index.files and src.id != "noaa"
                else "",
            )
        series = series.loc[start:end] if start is not None else series.loc[:end]
        da = xr.DataArray(series, dims="time", name=name)
        da.attrs = {
            "long_name": index.long_name,
            "units": index.units,
            "cadence": index.cadence,
            "provider": spec.provider,
            "source_url": spec.url,
            "latest": latest.strftime("%Y-%m"),
        }
        data_vars[name] = da
    ds = xr.Dataset(data_vars)
    if "time" not in ds.dims:
        ds = ds.expand_dims(time=pd.DatetimeIndex([], name="time"))
    return contract.finalize(
        ds,
        PRODUCT,
        src,
        mask=False,
        source_url=GUIDE_URL,
        attrs={"indices": " ".join(names)},
    )


# ── analysis helpers (pure) ───────────────────────────────────────────────────


def seasonal_mean(
    obj: xr.Dataset | xr.DataArray | pd.Series | pd.DataFrame,
    months: tuple[int, ...] = (11, 12, 1, 2, 3),
    *,
    hemisphere: str = "northern",
    min_months: int | None = None,
) -> pd.DataFrame:
    """Average each index over *months*, one row per water year.

    The default, November-March, is the accumulation season most snowpack
    teleconnection studies use. Months are assigned to the water year they
    fall in (:func:`easysnowdata.processing.wateryear.water_year`), so a
    December-February window is one row, not two.

    Parameters
    ----------
    obj
        The output of :func:`load`, one of its variables, or a pandas
        Series/DataFrame on a ``DatetimeIndex``.
    months
        Calendar months to average.
    hemisphere
        Water-year convention (``"northern"`` starts in October).
    min_months
        Years with fewer valid months than this are NaN (default: all of
        *months*, so a season still in progress is not reported).

    Returns
    -------
    pandas.DataFrame
        Index ``water_year``; one column per index.
    """
    if isinstance(obj, xr.Dataset):
        frame = obj.to_dataframe()
    elif isinstance(obj, xr.DataArray):
        frame = obj.to_series().to_frame()
    elif isinstance(obj, pd.Series):
        frame = obj.to_frame()
    else:
        frame = obj
    frame = frame[frame.index.month.isin(months)]
    years = wateryear.water_year(frame.index, hemisphere=hemisphere)
    grouped = frame.groupby(np.asarray(years))
    need = len(months) if min_months is None else min_months
    means = grouped.mean().where(grouped.count() >= need)
    means.index.name = "water_year"
    return means


def enso_phase(
    oni: xr.DataArray | pd.Series, threshold: float = 0.5, min_seasons: int = 5
) -> pd.Series:
    """Label each month El Niño, La Niña or neutral with CPC's event rule.

    An event is at least *min_seasons* consecutive overlapping 3-month seasons
    at or beyond ±*threshold* °C (CPC's definition, applied to the ONI and
    to RONI alike).

    Parameters
    ----------
    oni
        A RONI or ONI series (``load(indices="roni")["roni"]``).
    threshold, min_seasons
        The event rule; the defaults are CPC's.

    Returns
    -------
    pandas.Series
        ``"El Niño"``, ``"La Niña"`` or ``"neutral"`` per month (missing
        months stay missing).
    """
    series = oni.to_series() if isinstance(oni, xr.DataArray) else oni
    series = series.dropna()
    phase = pd.Series("neutral", index=series.index, dtype=object)
    for label, mask in (
        ("El Niño", series >= threshold),
        ("La Niña", series <= -threshold),
    ):
        run_id = (mask != mask.shift()).cumsum()
        run_len = mask.groupby(run_id).transform("size")
        phase[mask & (run_len >= min_seasons)] = label
    phase.name = f"{series.name or 'enso'}_phase"
    return phase
