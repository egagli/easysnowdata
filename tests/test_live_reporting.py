"""The live-test-to-issue planner (scripts/report_live.py).

Offline: the report is a JUnit XML string and GitHub is a fake, so the
grouping by data source, the open/update/close paths and the "all skipped
proves nothing" rule are tested without a network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def live():
    """scripts/report_live.py, imported by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "report_live", REPO_ROOT / "scripts" / "report_live.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_live"] = module
    spec.loader.exec_module(module)
    return module


# Shaped like the 2026-10-20 run: Yukon broken, CMR down, AWDB fine, Planet
# skipped for want of a key.
REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.stations.test_yukon_client" name="test_get_timeseries">
    <failure message="YukonError: HTTP 500 from https://service.yukon.ca/water-data/api/v1/timeseries after 3 attempts">trace</failure>
  </testcase>
  <testcase classname="tests.stations.test_yukon_client" name="test_get_data_bbox">
    <failure message="YukonError: HTTP 500 from https://service.yukon.ca/water-data/api/v1/timeseries after 3 attempts">trace</failure>
  </testcase>
  <testcase classname="tests.stations.test_yukon_client" name="test_get_data_snwd_type">
    <error message="assert 0 &gt; 100">trace</error>
  </testcase>
  <testcase classname="tests.stations.test_yukon_client" name="test_variables_required_fields"/>
  <testcase classname="tests.timeseries.test_hls" name="test_live_hls_search">
    <failure message="APIError: {&quot;errors&quot;:[&quot;Oops! Something has gone wrong.&quot;]}">trace</failure>
  </testcase>
  <testcase classname="tests.stations.test_awdb_client" name="test_get_data"/>
  <testcase classname="tests.timeseries.test_planetscope" name="test_live_search">
    <skipped message="PL_API_KEY not set"/>
  </testcase>
</testsuite></testsuites>
"""


def test_tests_are_grouped_by_their_file(live):
    outcomes = live.parse(REPORT)
    assert set(outcomes) == {"yukon", "hls", "awdb", "planetscope"}
    yukon = outcomes["yukon"]
    assert yukon.path == "tests/stations/test_yukon_client.py"
    assert len(yukon.failed) == 3  # a failure and an error both count
    assert yukon.passed == ["test_variables_required_fields"]
    assert outcomes["planetscope"].skipped == ["test_live_search"]


def test_a_junit_file_attribute_wins_over_the_classname(live):
    assert live.source_of("x.y", "tests/stations/test_nve_client.py") == (
        "nve",
        "tests/stations/test_nve_client.py",
    )


def test_failing_sources_open_passing_close_skipped_are_left_alone(live):
    actions = live.plan(live.parse(REPORT), checked="2026-10-20")
    by_key = {a.key: a for a in actions}
    assert by_key["yukon"].kind == "open"
    assert by_key["yukon"].title == "Live tests failing: yukon (3 tests)"
    assert by_key["hls"].title == "Live tests failing: hls (1 test)"
    assert by_key["awdb"].kind == "close"
    assert "planetscope" not in by_key  # all skipped proves nothing


def test_the_body_groups_identical_errors(live):
    body = live.plan(live.parse(REPORT))[2].body  # sorted: awdb, hls, yukon
    assert body.startswith("<!-- esd-live: yukon -->")
    assert "| 2 | `YukonError: HTTP 500" in body
    assert "| 1 | `assert 0 > 100` |" in body
    assert "pytest tests/stations/test_yukon_client.py -m live" in body


class FakeGitHub:
    def __init__(self, issues):
        self.issues = issues
        self.calls = []

    def ensure_label(self):
        pass

    def open_issues(self):
        return self.issues

    def create(self, action):
        self.calls.append(("create", action.key))
        return {"number": 99}

    def update(self, number, **fields):
        self.calls.append(("update", number, fields.get("state", "edit")))
        return {}

    def comment(self, number, body):
        self.calls.append(("comment", number))
        return {}


def test_apply_opens_updates_and_closes_by_marker(live):
    issues = [
        {"number": 5, "body": "<!-- esd-live: yukon -->\nold"},
        {"number": 6, "body": "<!-- esd-live: awdb -->\nold"},
        # The health check's issue for the same source is a different issue.
        {"number": 7, "body": "<!-- esd-health: hls -->\nold"},
    ]
    github = FakeGitHub(issues)
    live.apply(live.plan(live.parse(REPORT)), github, log=lambda _: None)
    assert ("create", "hls") in github.calls  # not merged into #7
    assert ("update", 5, "edit") in github.calls
    assert ("comment", 5) in github.calls  # a repeat failure surfaces again
    assert ("update", 6, "closed") in github.calls


def test_main_without_a_token_only_prints(live, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    report = tmp_path / "live.xml"
    report.write_text(REPORT)
    summary = tmp_path / "summary.md"
    assert live.main([str(report), "--summary", str(summary)]) == 0
    out = capsys.readouterr().out
    assert "[open]  Live tests failing: yukon (3 tests)" in out
    assert "| 🔴 hls | 1 | 0 | 0 |" in summary.read_text()
