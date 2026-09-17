"""Probe kinds, the status page, and the health-to-issue planner.

Offline: the planner is a pure function of the history, so the escalation,
the recovery and the "nothing changed" paths are all testable without GitHub
and without a network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from easysnowdata import catalog
from easysnowdata.catalog import health, pages
from easysnowdata.catalog._models import PROBE_KINDS, Probe

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reporter():
    """scripts/report_health.py, imported by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "report_health", REPO_ROOT / "scripts" / "report_health.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_health"] = module
    spec.loader.exec_module(module)
    return module


def _run(*rows, checked_at="2026-09-14T08:00:00Z"):
    """One history run from (label, status, ...) tuples."""
    out = []
    for row in rows:
        label, status = row[0], row[1]
        entry = {
            "source": label,
            "status": status,
            "error": None if status == "pass" else "boom",
            "checked_at": checked_at,
            "product": row[2] if len(row) > 2 else "a-product",
            "route": "a-route",
            "kind": row[3] if len(row) > 3 else "health",
        }
        if len(row) > 4:
            entry["value"] = row[4]
        out.append(entry)
    return out


class TestProbeKinds:
    def test_default_kind_is_health(self):
        assert Probe("x", lambda: None).kind == "health"

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            Probe("x", lambda: None, kind="vibes")

    def test_every_kind_is_represented_in_the_catalog(self):
        for kind in PROBE_KINDS:
            assert health.probes(kinds=[kind]), f"no {kind} probe anywhere"

    def test_latency_probes_sit_on_time_series_products(self):
        products = {p.id for p, _s, _probe in health.probes(kinds=["latency"])}
        assert {"snodas", "era5", "sentinel-1-rtc", "modis-snow"} <= products

    def test_virtualization_probes_sit_on_the_netcdf_hdf_products(self):
        products = {p.id for p, _s, _probe in health.probes(kinds=["virtualization"])}
        assert products == {"ucla-snow-reanalysis", "modis-snow", "viirs-snow"}

    def test_cmr_probes_declare_no_credentials(self):
        # CMR's metadata search needs no Earthdata Login, so these must run in
        # a credential-free weekly check rather than reporting "skip".
        for _p, _s, probe in health.probes(kinds=["latency", "virtualization"]):
            if "(CMR)" in probe.label:
                assert probe.requires == ()

    def test_a_product_still_needs_a_real_health_probe(self):
        from easysnowdata.catalog._models import Product, Source, validate

        only_latency = Product(
            id="x",
            theme="snow",
            title="t",
            description="d",
            citation="c",
            license="l",
            loader="easysnowdata.snow.snodas.load",
            sources=(
                Source(
                    id="s",
                    provider="stac",
                    location="loc",
                    health=(Probe("l", lambda: "2026-01-01", kind="latency"),),
                ),
            ),
        )
        assert "no health probe on any source" in " ".join(validate(only_latency))

    def test_runner_records_the_value_and_the_kind(self, monkeypatch):
        product = catalog.get("snodas")
        source = product.sources[0]
        probe = Probe("fake latency", lambda: "2026-09-16T00:00:00Z", kind="latency")
        result = health._check(product, source, probe)
        assert result["status"] == "pass"
        assert result["kind"] == "latency"
        assert result["value"] == "2026-09-16T00:00:00Z"

    def test_a_health_probe_records_no_value(self):
        product = catalog.get("snodas")
        result = health._check(product, product.sources[0], Probe("ok", lambda: None))
        assert "value" not in result


class TestPlanner:
    def test_a_failure_opens_an_issue(self, reporter):
        actions = reporter.plan([_run(("probe A", "fail", "snodas"))])
        opens = [a for a in actions if a.kind == "open"]
        assert len(opens) == 1
        assert opens[0].key == "snodas"
        assert "Data source unreachable: snodas" in opens[0].title
        assert "<!-- esd-health: snodas -->" in opens[0].body
        assert "🔴" not in opens[0].title  # first failure is not escalated

    def test_a_second_consecutive_failure_escalates(self, reporter):
        history = [_run(("probe A", "fail", "snodas"))] * 3
        opens = [a for a in reporter.plan(history) if a.kind == "open"]
        assert opens[0].title.startswith("🔴")
        assert "(3 runs)" in opens[0].title

    def test_the_body_names_the_last_good_run(self, reporter):
        history = [
            _run(("probe A", "fail", "snodas"), checked_at="2026-09-14T00:00:00Z"),
            _run(("probe A", "pass", "snodas"), checked_at="2026-09-07T00:00:00Z"),
        ]
        body = reporter.plan(history)[0].body
        assert "2026-09-07" in body

    def test_recovery_closes(self, reporter):
        history = [
            _run(("probe A", "pass", "snodas")),
            _run(("probe A", "fail", "snodas")),
        ]
        closes = [a for a in reporter.plan(history) if a.kind == "close"]
        assert closes and closes[0].key == "snodas"
        assert "Recovered" in closes[0].comment

    def test_a_run_of_only_skips_claims_nothing(self, reporter):
        actions = reporter.plan([_run(("probe A", "skip", "snodas"))])
        assert actions == []

    def test_one_issue_per_product_not_per_route(self, reporter):
        actions = reporter.plan(
            [
                _run(
                    ("probe A", "fail", "snodas"),
                    ("probe B", "fail", "snodas"),
                    ("probe C", "fail", "era5"),
                )
            ]
        )
        assert sorted(a.key for a in actions) == ["era5", "snodas"]
        body = next(a for a in actions if a.key == "snodas").body
        assert "probe A" in body and "probe B" in body

    def test_a_partly_working_product_does_not_open_an_issue(self, reporter):
        # One route down, another up: the product is still reachable, and the
        # catalog page already shows the broken route.
        actions = reporter.plan(
            [_run(("probe A", "fail", "snodas"), ("probe B", "pass", "snodas"))]
        )
        assert [a.kind for a in actions] == ["open"]

    def test_a_new_dmrpp_sidecar_opens_its_own_issue(self, reporter):
        history = [
            _run(
                (
                    "UCLA DMR++",
                    "pass",
                    "ucla-snow-reanalysis",
                    "virtualization",
                    "present",
                )
            ),
            _run(
                (
                    "UCLA DMR++",
                    "pass",
                    "ucla-snow-reanalysis",
                    "virtualization",
                    "absent; fallback: VirtualiZarr HDFParser",
                )
            ),
        ]
        opens = [a for a in reporter.plan(history) if a.kind == "open"]
        assert any("Fast virtualization path" in a.title for a in opens)
        issue = next(a for a in opens if "virtualization" in a.key)
        assert issue.key == "virtualization/ucla-snow-reanalysis"
        assert "absent" in issue.body and "present" in issue.body

    def test_a_sidecar_that_is_still_absent_opens_nothing(self, reporter):
        history = [_run(("UCLA DMR++", "pass", "u", "virtualization", "absent"))] * 2
        assert not [
            a for a in reporter.plan(history) if a.key.startswith("virtualization/")
        ]

    def test_a_sidecar_that_was_already_present_opens_nothing(self, reporter):
        history = [_run(("UCLA DMR++", "pass", "u", "virtualization", "present"))] * 2
        assert not [
            a for a in reporter.plan(history) if a.key.startswith("virtualization/")
        ]

    def test_empty_history(self, reporter):
        assert reporter.plan([]) == []
        assert reporter.plan([[]]) == []


class TestApply:
    class FakeGitHub:
        def __init__(self, issues=()):
            self.issues = list(issues)
            self.calls = []

        def ensure_label(self):
            self.calls.append(("label",))

        def open_issues(self):
            return self.issues

        def create(self, action):
            self.calls.append(("create", action.key, action.title))
            return {"number": 42}

        def update(self, number, **fields):
            self.calls.append(("update", number, tuple(sorted(fields))))
            return {}

        def comment(self, number, body):
            self.calls.append(("comment", number))
            return {}

    def test_creates_when_no_issue_exists(self, reporter):
        github = self.FakeGitHub()
        actions = reporter.plan([_run(("probe A", "fail", "snodas"))])
        reporter.apply(actions, github, log=lambda _line: None)
        assert ("create", "snodas", actions[0].title) in github.calls

    def test_updates_the_issue_it_already_opened(self, reporter):
        github = self.FakeGitHub(
            [{"number": 7, "body": "<!-- esd-health: snodas -->\nold"}]
        )
        reporter.apply(
            reporter.plan([_run(("probe A", "fail", "snodas"))]),
            github,
            log=lambda _line: None,
        )
        assert ("update", 7, ("body", "title")) in github.calls
        assert not any(call[0] == "create" for call in github.calls)

    def test_closes_on_recovery(self, reporter):
        github = self.FakeGitHub(
            [{"number": 7, "body": "<!-- esd-health: snodas -->\nold"}]
        )
        history = [
            _run(("probe A", "pass", "snodas")),
            _run(("probe A", "fail", "snodas")),
        ]
        reporter.apply(reporter.plan(history), github, log=lambda _line: None)
        assert ("comment", 7) in github.calls
        assert ("update", 7, ("state", "state_reason")) in github.calls

    def test_closing_what_was_never_opened_is_a_no_op(self, reporter):
        github = self.FakeGitHub()
        history = [
            _run(("probe A", "pass", "snodas")),
            _run(("probe A", "fail", "snodas")),
        ]
        reporter.apply(reporter.plan(history), github, log=lambda _line: None)
        assert [c for c in github.calls if c[0] != "label"] == []


class TestStatusPage:
    HISTORY = [
        _run(
            ("Route A", "pass", "snodas"),
            ("Route B", "fail", "era5"),
            ("SNODAS latency", "pass", "snodas", "latency", "2026-09-13T00:00:00Z"),
            (
                "UCLA DMR++",
                "pass",
                "ucla-snow-reanalysis",
                "virtualization",
                "absent; fallback: VirtualiZarr HDFParser",
            ),
            checked_at="2026-09-14T08:00:00Z",
        ),
        _run(
            ("Route A", "pass", "snodas"),
            ("Route B", "pass", "era5"),
            ("SNODAS latency", "pass", "snodas", "latency", "2026-09-05T00:00:00Z"),
            checked_at="2026-09-07T08:00:00Z",
        ),
    ]

    def test_sections_and_counts(self):
        page = pages.status_page(self.HISTORY, now="2026-09-14T08:00:00Z")
        assert page.startswith("# Source status")
        assert "2 runs on record" in page
        assert "## Health" in page
        assert "## Latency" in page
        assert "## Virtualization readiness" in page
        assert "Route B" in page and "failing" in page

    def test_latency_shows_the_lag_in_days(self):
        page = pages.status_page(self.HISTORY, now="2026-09-14T00:00:00Z")
        assert "2026-09-13T00:00:00Z" in page
        assert "| 1 |" in page  # one day behind

    def test_virtualization_reports_the_fallback_parser(self):
        page = pages.status_page(self.HISTORY, now="2026-09-14T08:00:00Z")
        assert "VirtualiZarr HDFParser" in page
        assert "{bdg-secondary}`absent`" in page

    def test_empty_history_still_renders(self):
        page = pages.status_page([])
        assert "never" in page
        assert "No latency probe has run yet" in page

    def test_products_with_no_recorded_run_are_listed(self):
        page = pages.status_page(self.HISTORY, now="2026-09-14T08:00:00Z")
        assert "## Not yet in the history" in page
        assert "[`copernicus-dem`](catalog/copernicus-dem.md)" in page


class TestSparkline:
    def test_needs_two_points(self):
        assert pages.sparkline([]) == ""
        assert pages.sparkline([1.0]) == ""
        assert pages.sparkline([None, None]) == ""

    def test_draws_a_polyline(self):
        svg = pages.sparkline([1.0, 5.0, 2.0])
        assert svg.startswith("<svg") and "polyline" in svg
        assert "currentColor" in svg  # legible in both themes

    def test_a_gap_breaks_the_line(self):
        assert pages.sparkline([1.0, 2.0, None, 4.0, 5.0]).count("<polyline") == 2

    def test_a_flat_series_does_not_divide_by_zero(self):
        assert "polyline" in pages.sparkline([3.0, 3.0, 3.0])
