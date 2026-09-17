"""Unit tests for easysnowdata.temporal."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from easysnowdata import temporal


class TestParseTime:
    def test_month_expands(self):
        start, end = temporal.parse_time("2023-10")
        assert start == pd.Timestamp("2023-10-01")
        assert end == pd.Timestamp("2023-10-31 23:59:59")

    def test_year_and_day(self):
        assert temporal.parse_time("2023")[1] == pd.Timestamp("2023-12-31 23:59:59")
        s, e = temporal.parse_time("2023-10-01")
        assert (s, e) == (
            pd.Timestamp("2023-10-01"),
            pd.Timestamp("2023-10-01 23:59:59"),
        )

    def test_stac_interval_and_open_ends(self):
        s, e = temporal.parse_time("2023-10/2024-06")
        assert s == pd.Timestamp("2023-10-01") and e == pd.Timestamp(
            "2024-06-30 23:59:59"
        )
        s, e = temporal.parse_time("../2024-06")
        assert s is None and e == pd.Timestamp("2024-06-30 23:59:59")
        s, e = temporal.parse_time("2024-06/..")
        assert s == pd.Timestamp("2024-06-01") and e >= pd.Timestamp("2026-01-01")

    def test_tuple_slice_and_none(self):
        s, e = temporal.parse_time(("2023-10-01", "2024-06-30"))
        assert s == pd.Timestamp("2023-10-01") and e == pd.Timestamp(
            "2024-06-30 23:59:59"
        )
        s, e = temporal.parse_time(slice("2023-10-01", None))
        assert s == pd.Timestamp("2023-10-01") and e > pd.Timestamp("2026-01-01")
        s, e = temporal.parse_time(None)
        assert s is None and abs((e - temporal.now()).total_seconds()) < 5
        s, e = temporal.parse_time(None, default_start="2000")
        assert s == pd.Timestamp("2000-01-01")

    def test_timestamps_periods_and_timezones(self):
        ts = pd.Timestamp("2023-10-01 12:00")
        assert temporal.parse_time(ts) == (ts, ts)
        assert temporal.parse_time(dt.date(2023, 10, 1))[0] == pd.Timestamp(
            "2023-10-01"
        )
        assert temporal.parse_time(pd.Period("2023-10"))[1] == pd.Timestamp(
            "2023-10-31 23:59:59"
        )
        s, _ = temporal.parse_time("2023-10-01T00:00:00Z/2023-10-02T00:00:00Z")
        assert s == pd.Timestamp("2023-10-01")
        s, _ = temporal.parse_time(pd.Timestamp("2023-10-01 00:00", tz="US/Pacific"))
        assert s == pd.Timestamp("2023-10-01 07:00")

    def test_errors(self):
        with pytest.raises(ValueError, match="after end"):
            temporal.parse_time("2024-06/2023-10")
        with pytest.raises(ValueError, match="start, end"):
            temporal.parse_time(("2023", "2024", "2025"))
        with pytest.raises(ValueError):
            temporal._period_bounds(None, end=False)

    def test_to_stac_datetime_and_today(self):
        assert (
            temporal.to_stac_datetime("2023-10")
            == "2023-10-01T00:00:00Z/2023-10-31T23:59:59Z"
        )
        assert temporal.to_stac_datetime("../2023-10").startswith("../")
        today = temporal.today()
        assert len(today) == 10 and today[:4] >= "2026"
