"""easysnowdata.climate.indices — offline tier on trimmed copies of the real
files' layouts, plus live smoke tests against NOAA."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.climate import indices as ci

# Trimmed from the real files on 2026-10-09: same headers, spacing, sentinels.
NCEI_PDO = """ ERSST PDO Index:
Year  Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
2025 -1.29 -1.40 -1.12 -1.15 -1.66 -2.64 -4.21 -3.23 -2.31 -2.37 -1.51 -0.96
2026 -1.24 -1.00 -1.42 -1.62 -1.68 -1.72 -2.16 -1.89 -1.55 99.99 99.99 99.99
"""
NCEI_AMO = """ ERSST AMO (North Atlantic 0-60N SSTA) Index:
Year  month      SSTA
2026      8      1.19
2026      9      1.25
"""
CPC_RONI = """SEAS   YR  ANOM
NDJ  2025 -0.71
DJF  2026 -0.83
JFM  2026 -0.62
"""
CPC_ONI = """ SEAS  YR   TOTAL   ANOM
  NDJ 2025  25.89  -0.55
  DJF 2026  26.10  -0.40
  JFM 2026  26.62  -0.18
"""
CPC_RNINO34 = """YR   MTH   ANOM
2026   8   1.67
2026   9   2.08
"""
CPC_NINO34 = """ YR   MON  TOTAL ClimAdjust ANOM
2026   8   29.05   26.87    2.17
2026   9   29.27   26.71    2.56
"""
CPC_AO = """ 2026    8   -0.0560
 2026    9    1.1556
"""
PSL_PDO = """         1948 2025
2025    -0.78  -0.82  -0.41  -0.57  -1.14  -2.32  -3.51  -2.71  -9.90  -9.90  -9.90  -9.90
  -9.90
  PDO from PSL
"""
PSL_MEI = """1979     2026
2026    -0.90    -0.85    -0.55    -0.20     0.37     1.52     2.41     2.54     2.37  -999.00  -999.00  -999.00
  -999.00
Multivariate ENSO Index Version 2 (MEI.v2)
Row values are 2 month seasons (YEAR DJ JF FM MA AM MJ JJ JA AS SO ON ND)
"""

FILES = {
    "ersst.v5.pdo.dat": NCEI_PDO,
    "ersst.v5.amo.dat": NCEI_AMO,
    "RONI.ascii.txt": CPC_RONI,
    "oni.ascii.txt": CPC_ONI,
    "Rnino34.ascii.txt": CPC_RNINO34,
    "detrend.nino34.ascii.txt": CPC_NINO34,
    "monthly.ao.index.b50.current.ascii": CPC_AO,
    "norm.nao.monthly.b5001.current.ascii": CPC_AO,
    "norm.pna.monthly.b5001.current.ascii": CPC_AO,
    "meiv2.data": PSL_MEI,
    "pdo.data": PSL_PDO,
}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Serve the trimmed files instead of downloading; record what was asked."""
    calls = []

    def fake_fetch(url, fname=None, *, subdir=None, max_age=None):
        calls.append((url, fname, subdir, max_age))
        path = tmp_path / (fname or Path(url).name)
        path.write_text(FILES[Path(url).name])
        return path

    monkeypatch.setattr(esd.providers.table_http, "fetch", fake_fetch)
    return calls


# ── catalog entry ─────────────────────────────────────────────────────────────


def test_catalog_entry_is_registered_from_this_module():
    product = catalog.get("climate-indices")
    assert product is ci.PRODUCT
    assert product.resolve_loader() is ci.load
    assert [s.id for s in product.sources] == ["noaa", "psl"]
    assert all(s.requires == () for s in product.sources)
    assert {s.provider for s in product.sources} == {"table_http"}
    assert {v.name for v in product.variables} == set(ci.INDICES)
    kinds = [p.kind for p in product.default_source.health]
    assert kinds.count("health") == 3 and kinds.count("latency") == 1
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_every_index_has_a_noaa_file_and_psl_lacks_the_relative_ones():
    assert set(ci.SOURCES["noaa"]) == set(ci.INDICES)
    assert "roni" not in ci.SOURCES["psl"] and "rnino34" not in ci.SOURCES["psl"]
    assert ci.INDICES["pdo"].files["noaa"].provider == "NOAA NCEI"
    assert ci.INDICES["roni"].files["noaa"].url.endswith("RONI.ascii.txt")


def test_search_lists_without_downloading(offline):
    frame = ci.search()
    assert list(frame.index) == list(ci.SOURCES["noaa"])
    assert frame.loc["pdo", "provider"] == "NOAA NCEI"
    assert {"url", "units", "cadence", "notes"} <= set(frame.columns)
    assert len(ci.search(source="psl")) == len(ci.SOURCES["psl"])
    assert offline == []


# ── parsing ───────────────────────────────────────────────────────────────────


def test_parse_wide_ncei_drops_the_trailing_sentinel():
    series = ci.parse(NCEI_PDO, "wide", name="pdo")
    assert series.index[0] == pd.Timestamp("2025-01-01")
    assert series.index[-1] == pd.Timestamp("2026-09-01")
    assert len(series) == 21 and series.name == "pdo"
    assert series["2025-07-01"] == pytest.approx(-4.21)
    assert series.index.name == "time"


def test_parse_wide_psl_reads_its_own_small_sentinel():
    # pdo.data marks missing months -9.90, inside the range a PDO can take
    series = ci.parse(PSL_PDO, "wide")
    assert series.index[-1] == pd.Timestamp("2025-08-01")
    assert len(series) == 8 and series.min() > -4


def test_parse_seasons_stamps_the_centre_month():
    series = ci.parse(CPC_RONI, "seasons")
    assert list(series.index) == list(
        pd.to_datetime(["2025-12-01", "2026-01-01", "2026-02-01"])
    )
    assert series["2026-01-01"] == pytest.approx(-0.83)
    # the ONI file has an extra TOTAL column; the anomaly is the last one
    assert ci.parse(CPC_ONI, "seasons")["2026-01-01"] == pytest.approx(-0.40)


def test_parse_columns_takes_the_last_column_and_skips_headers():
    assert ci.parse(CPC_NINO34, "columns")["2026-09-01"] == pytest.approx(2.56)
    assert ci.parse(NCEI_AMO, "columns").tolist() == [1.19, 1.25]
    assert len(ci.parse(CPC_AO, "columns")) == 2


def test_parse_errors():
    with pytest.raises(ValueError, match="layout must be"):
        ci.parse(CPC_AO, "csv")
    with pytest.raises(ValueError, match="No 'seasons' rows"):
        ci.parse(CPC_AO, "seasons", name="ao")


# ── load ──────────────────────────────────────────────────────────────────────


def test_load_returns_the_contract(offline, monkeypatch):
    monkeypatch.setattr(ci, "now", lambda: pd.Timestamp("2026-10-09"))
    ds = ci.load(indices=["pdo", "roni", "mei"])
    assert isinstance(ds, xr.Dataset) and list(ds.data_vars) == ["pdo", "roni", "mei"]
    assert ds.sizes["time"] == 21 and ds["pdo"].dtype == np.float64
    assert np.isnan(ds["roni"].sel(time="2025-01-01"))  # outer join, NaN-padded
    assert ds["mei"].sel(time="2026-08-01") == pytest.approx(2.54)
    assert ds["pdo"].attrs["provider"] == "NOAA NCEI"
    assert ds["pdo"].attrs["latest"] == "2026-09"
    assert ds["roni"].attrs["units"] == "degC"
    assert ds.attrs["product_id"] == "climate-indices"
    assert ds.attrs["source"] == "noaa" and ds.attrs["indices"] == "pdo roni mei"
    # files are cached under one subdir, named by source, one day old at most
    assert {c[2] for c in offline} == {"climate_indices"}
    assert offline[0][1] == "noaa_ersst.v5.pdo.dat" and offline[0][3] == 86400


def test_load_time_and_single_name(offline, monkeypatch):
    monkeypatch.setattr(ci, "now", lambda: pd.Timestamp("2026-10-09"))
    ds = ci.load(None, "2026-01/2026-03", indices="pdo")
    assert list(ds.data_vars) == ["pdo"]
    assert ds["pdo"].values.tolist() == [-1.24, -1.00, -1.42]
    # an aoi is accepted and ignored
    assert ci.load((-122, 46, -121, 47), "2026", indices="ao").sizes["time"] == 2


def test_load_default_set_per_source(offline, monkeypatch):
    monkeypatch.setattr(ci, "now", lambda: pd.Timestamp("2026-10-09"))
    monkeypatch.setattr(ci, "DEFAULT_INDICES", ("pdo", "roni", "mei"))
    assert list(ci.load().data_vars) == ["pdo", "roni", "mei"]
    # psl has no RONI, so the default set skips it rather than failing
    assert list(ci.load(source="psl").data_vars) == ["pdo", "mei"]


def test_load_warns_on_a_stale_file(offline, monkeypatch, caplog):
    monkeypatch.setattr(ci, "now", lambda: pd.Timestamp("2026-10-09"))
    with caplog.at_level(logging.WARNING, logger="easysnowdata.climate.indices"):
        ds = ci.load(indices="pdo", source="psl")
    assert ds["pdo"].attrs["latest"] == "2025-08"
    assert "no longer updated" in caplog.text and "source='noaa'" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="easysnowdata.climate.indices"):
        ci.load(indices="pdo")
    assert caplog.text == ""


def test_load_rejects_bad_names_before_downloading(offline):
    with pytest.raises(ValueError, match="Unknown index 'soi'"):
        ci.load(indices=["pdo", "soi"])
    with pytest.raises(ValueError, match="offered by noaa"):
        ci.load(indices="roni", source="psl")
    with pytest.raises(ValueError, match="no source"):
        ci.load(indices="pdo", source="jpl")
    assert offline == []


def test_load_with_no_overlap_keeps_a_time_dimension(offline, monkeypatch):
    monkeypatch.setattr(ci, "now", lambda: pd.Timestamp("2026-10-09"))
    ds = ci.load(time="1990/1991", indices="ao")
    assert ds.sizes["time"] == 0 and "ao" in ds


def test_load_without_any_index(offline):
    ds = ci.load(indices=[])
    assert ds.sizes["time"] == 0 and list(ds.data_vars) == []


def test_table_http_fetch_wraps_the_cache(monkeypatch, tmp_path):
    seen = {}

    def fake(url, fname=None, *, subdir=None, progressbar=True, max_age=None, **_):
        seen.update(url=url, fname=fname, subdir=subdir, bar=progressbar, age=max_age)
        return tmp_path / "x"

    monkeypatch.setattr(esd.providers.raster_http, "fetch", fake)
    from easysnowdata.providers import table_http

    assert table_http.fetch("https://h/a.txt", "b.txt", subdir="s") == tmp_path / "x"
    assert seen == {
        "url": "https://h/a.txt",
        "fname": "b.txt",
        "subdir": "s",
        "bar": False,
        "age": 86400,
    }


def test_latency_probe_reads_the_newest_month(offline):
    assert ci._latest_pdo() == "2026-09-01T00:00:00Z"
    assert offline[-1][3] == 0  # the probe always downloads


# ── helpers ───────────────────────────────────────────────────────────────────


def _monthly(values, start="2019-10-01"):
    index = pd.date_range(start, periods=len(values), freq="MS", name="time")
    return pd.Series(values, index=index, dtype=float)


def test_seasonal_mean_groups_by_water_year():
    # WY2020 (Oct 2019 - Sep 2020) = 1.0 everywhere, WY2021 = 2.0
    series = _monthly([1.0] * 12 + [2.0] * 12, start="2019-10-01").rename("pdo")
    out = ci.seasonal_mean(series)
    assert out.index.name == "water_year"
    assert out["pdo"].to_dict() == {2020: 1.0, 2021: 2.0}
    # a Dataset, a DataArray and a DataFrame give the same answer
    ds = xr.Dataset({"pdo": xr.DataArray(series, dims="time")})
    pd.testing.assert_frame_equal(ci.seasonal_mean(ds), out)
    pd.testing.assert_frame_equal(ci.seasonal_mean(ds["pdo"]), out)
    pd.testing.assert_frame_equal(ci.seasonal_mean(series.to_frame()), out)


def test_seasonal_mean_needs_every_month_by_default():
    series = _monthly([1.0, 1.0, 1.0, 5.0], start="2020-11-01").rename("x")  # Nov-Feb
    assert np.isnan(ci.seasonal_mean(series).loc[2021, "x"])
    assert ci.seasonal_mean(series, min_months=4).loc[2021, "x"] == 2.0
    djf = ci.seasonal_mean(series, months=(12, 1, 2))
    assert djf.loc[2021, "x"] == pytest.approx(7 / 3)


def test_enso_phase_applies_the_five_season_rule():
    values = [0.6] * 5 + [0.0] + [0.7] * 4 + [-0.5] * 6
    phase = ci.enso_phase(_monthly(values).rename("roni"))
    assert phase.name == "roni_phase"
    assert phase.iloc[:5].eq("El Niño").all()
    assert phase.iloc[5] == "neutral"
    assert phase.iloc[6:10].eq("neutral").all()  # four seasons is not an event
    assert phase.iloc[10:].eq("La Niña").all()
    da = xr.DataArray(_monthly(values), dims="time")
    assert ci.enso_phase(da).tolist() == phase.tolist()


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_noaa_indices_are_current():
    ds = ci.load(time="1950/..", max_age=0)
    assert set(ds.data_vars) == set(ci.DEFAULT_INDICES)
    for name in ds.data_vars:
        latest = pd.Timestamp(ds[name].attrs["latest"])
        assert latest > pd.Timestamp.now() - pd.Timedelta(days=ci.STALE_AFTER_DAYS)
        assert float(np.abs(ds[name]).max()) < 6
    assert ds["pdo"].sel(time="1950-01-01").notnull()


@pytest.mark.live
def test_live_psl_copies_parse():
    ds = ci.load(indices=["pdo", "oni"], source="psl", max_age=0)
    assert ds["pdo"].attrs["source_url"].startswith("https://psl.noaa.gov/")
    assert ds["oni"].notnull().sum() > 800
