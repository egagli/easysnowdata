"""The upstream watch: the watchlist, the diff, the tagging and the digest.

Offline. The fetching is one function per kind and is exercised against fake
responses; the diff and the digest are pure and get the real attention,
because those are what decide whether a week's change is seen or lost.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

from easysnowdata import catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCHLIST = REPO_ROOT / "WATCHLIST.toml"


@pytest.fixture(scope="module")
def watch():
    spec = importlib.util.spec_from_file_location(
        "watch", REPO_ROOT / "scripts" / "watch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["watch"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def watchlist():
    return tomllib.loads(WATCHLIST.read_text())


class TestWatchlist:
    def test_it_parses_and_every_kind_is_known(self, watch, watchlist):
        assert watchlist
        assert set(watchlist) <= set(watch.CHECKS)

    def test_every_entry_has_a_unique_id(self, watchlist):
        ids = [entry["id"] for entries in watchlist.values() for entry in entries]
        assert len(ids) == len(set(ids)), "snapshot file names must be unique"

    def test_every_named_product_is_in_the_catalog(self, watchlist):
        known = set(catalog.products())
        for entries in watchlist.values():
            for entry in entries:
                unknown = set(entry.get("products", [])) - known
                assert not unknown, f"{entry['id']} names {unknown}"

    def test_every_entry_says_what_to_fetch(self, watchlist):
        keys = {
            "cmr": "short_name",
            "stac": "url",
            "gee": "assets",
            "stations": "networks",
            "pypi": "packages",
            "page": "url",
            "feed": "url",
        }
        for kind, entries in watchlist.items():
            for entry in entries:
                if kind == "file":
                    assert entry.get("url") or entry.get("product"), entry["id"]
                else:
                    assert entry.get(keys[kind]), entry["id"]

    def test_a_file_entry_can_take_its_url_from_the_catalog(self, watch, watchlist):
        entry = next(
            e for e in watchlist["file"] if e["id"] == "file-snow-classification-cog"
        )
        url = watch._entry_url(entry)
        assert url == catalog.get("snow-classification").source("hosted-cog").location

    def test_every_product_with_a_gated_route_is_watched_somewhere(self, watchlist):
        # Not every product needs an entry, but the ones whose upstream is
        # known to move (NSIDC, PC, GEE) should be covered by something.
        watched = {
            product
            for entries in watchlist.values()
            for entry in entries
            for product in entry.get("products", [])
        }
        assert {"modis-snow", "viirs-snow", "hls", "sentinel-1-rtc", "era5"} <= watched


class TestClassify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("This collection will be decommissioned in March", "removal"),
            ("MOD10A2 is deprecated; migrate to MOD10A1F", "deprecation"),
            ("Collection 7 reprocessing has begun", "reprocessing"),
            ("Coverage extended to 2026", "extent"),
            ("Six new SNOTEL stations were added", "stations"),
            ("Scheduled maintenance on the S3 endpoint", "outage"),
        ],
    )
    def test_the_section_8_vocabulary(self, watch, text, expected):
        assert expected in watch.classify(text)

    def test_an_unmatched_line_gets_no_tag_and_is_still_reported(self, watch):
        change = watch.Change("e", "s", "the quick brown fox", categories=[])
        assert watch.classify("the quick brown fox") == []
        assert change.section == "Other"

    def test_entry_keywords_promote_a_line(self, watch):
        assert "action" in watch.classify("the blob expiry", ["expiry"])
        assert "action" not in watch.classify("the blob expiry", ["cheese"])


class TestDiffMapping:
    ENTRY = {"id": "e", "products": ["snodas"]}

    def test_the_first_run_records_but_reports_nothing(self, watch):
        after = {"a": {"version": "1"}}
        assert watch.diff_mapping(self.ENTRY, {}, after) == []

    def test_a_version_bump_is_reprocessing(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY, {"a": {"version": "1"}}, {"a": {"version": "2"}}
        )
        assert len(changes) == 1
        assert "reprocessing" in changes[0].categories
        assert changes[0].section == "Action needed"
        assert "`1` → `2`" in changes[0].detail
        assert changes[0].products == ["snodas"]

    def test_a_collection_disappearing_is_a_removal(self, watch):
        changes = watch.diff_mapping(self.ENTRY, {"a": {"v": "1"}}, {})
        assert changes[0].categories == ["removal"]
        assert changes[0].detail == "disappeared"

    def test_a_collection_appearing_is_a_new_dataset(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY, {"a": {"v": "1"}}, {"a": {"v": "1"}, "b": {"v": "1"}}
        )
        assert [c.subject for c in changes] == ["b"]
        assert changes[0].section == "Worth adding"

    def test_a_temporal_end_moving_is_an_extent_change(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY,
            {"a": {"temporal_end": "2025-01-01"}},
            {"a": {"temporal_end": "2026-01-01"}},
        )
        assert "extent" in changes[0].categories
        assert changes[0].section == "Changed"

    def test_a_station_count_moving_is_reported(self, watch):
        changes = watch.diff_mapping(
            {"id": "stations-counts", "products": ["awdb-stations"]},
            {"awdb": {"count": 7146}},
            {"awdb": {"count": 7151}},
        )
        assert "extent" in changes[0].categories
        assert "`7146` → `7151`" in changes[0].detail

    def test_the_url_field_is_metadata_not_a_change(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY, {"a": {"v": "1", "url": "x"}}, {"a": {"v": "1", "url": "y"}}
        )
        assert changes == []

    def test_nothing_moving_reports_nothing(self, watch):
        same = {"a": {"v": "1"}, "b": {"v": "2"}}
        assert watch.diff_mapping(self.ENTRY, same, dict(same)) == []


class TestDiffLines:
    ENTRY = {"id": "p", "url": "https://example.test", "products": []}

    def test_first_run_reports_nothing(self, watch):
        assert watch.diff_lines(self.ENTRY, {}, {"lines": ["a", "b"]}) == []

    def test_only_new_lines_are_reported(self, watch):
        changes = watch.diff_lines(
            self.ENTRY, {"lines": ["a", "b"]}, {"lines": ["b", "c is deprecated"]}
        )
        assert [c.detail for c in changes] == ["c is deprecated"]
        assert "deprecation" in changes[0].categories

    def test_a_flood_of_new_lines_is_truncated(self, watch):
        changes = watch.diff_lines(
            self.ENTRY,
            {"lines": ["old"]},
            {"lines": [f"line {i}" for i in range(50)]},
            limit=5,
        )
        assert len(changes) == 6
        assert "45 more new lines" in changes[-1].detail


class TestDigest:
    def test_sections_in_order_and_items_placed(self, watch):
        changes = [
            watch.Change("e", "A", "gone", categories=["removal"]),
            watch.Change("e", "B", "appeared", categories=["new-dataset"]),
            watch.Change("e", "C", "extended", categories=["extent"]),
            watch.Change("e", "D", "v2", categories=["dependency"]),
            watch.Change("e", "E", "who knows", categories=[]),
        ]
        body = watch.digest(changes, [], when="2026-09-17")
        assert "<!-- esd-watch: 2026-09-17 -->" in body
        positions = [
            body.index(name)
            for name in (
                "## Action needed",
                "## Worth adding",
                "## Changed",
                "## Dependency releases",
                "## Other",
            )
        ]
        assert positions == sorted(positions)
        assert "who knows" in body  # §8: an unmatched item is listed, never dropped

    def test_a_long_section_is_capped_but_counted(self, watch):
        changes = [
            watch.Change("noisy-page", f"S{i}", "x", categories=["removal"])
            for i in range(40)
        ]
        body = watch.digest(changes, [], when="2026-09-17")
        assert "## Action needed (40)" in body
        assert "...and 15 more, from noisy-page" in body

    def test_unreachable_entries_are_their_own_section(self, watch):
        body = watch.digest([], ["page-x: HTTPError: 404"], when="2026-09-17")
        assert "## Could not be checked (1)" in body
        assert "page-x" in body

    def test_products_are_named_so_the_stake_is_obvious(self, watch):
        body = watch.digest(
            [
                watch.Change(
                    "e", "A", "gone", products=["snodas"], categories=["removal"]
                )
            ],
            [],
        )
        assert "affects `snodas`" in body


class TestRun:
    class FakeResponse:
        def __init__(self, payload=None, text="", status=200, headers=None):
            self._payload = payload
            self.text = text
            self.status_code = status
            self.headers = headers or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def close(self):
            pass

    def test_a_dead_entry_is_an_error_not_a_crash(self, watch, tmp_path):
        class Boom:
            def get(self, *_args, **_kwargs):
                raise RuntimeError("kaboom")

        changes, errors = watch.run(
            {"pypi": [{"id": "p", "packages": ["xarray"]}]},
            directory=tmp_path,
            session=Boom(),
            log=lambda _line: None,
        )
        assert changes == []
        assert errors and "kaboom" in errors[0]

    def test_snapshots_are_written_then_diffed(self, watch, tmp_path):
        payloads = iter(
            [
                {"info": {"version": "1.0", "project_url": ""}},
                {"info": {"version": "2.0", "project_url": ""}},
            ]
        )

        class Session:
            def get(self, *_args, **_kwargs):
                return TestRun.FakeResponse(next(payloads))

        entry = {"pypi": [{"id": "p", "packages": ["xarray"]}]}
        changes, _ = watch.run(
            entry, directory=tmp_path, session=Session(), log=lambda _l: None
        )
        assert changes == []  # first run records
        saved = json.loads((tmp_path / "p.json").read_text())
        assert saved["xarray"]["latest_version"] == "1.0"

        changes, _ = watch.run(
            entry, directory=tmp_path, session=Session(), log=lambda _l: None
        )
        assert len(changes) == 1
        assert changes[0].categories == ["dependency"]  # not "reprocessing"
        assert changes[0].section == "Dependency releases"

    def test_dry_run_writes_no_snapshot(self, watch, tmp_path):
        class Session:
            def get(self, *_args, **_kwargs):
                return TestRun.FakeResponse({"info": {"version": "1.0"}})

        watch.run(
            {"pypi": [{"id": "p", "packages": ["xarray"]}]},
            directory=tmp_path,
            session=Session(),
            write=False,
            log=lambda _l: None,
        )
        assert not (tmp_path / "p.json").exists()

    def test_only_filters_by_kind(self, watch, tmp_path):
        called = []

        class Session:
            def get(self, url, *_args, **_kwargs):
                called.append(url)
                return TestRun.FakeResponse({"info": {"version": "1.0"}})

        watch.run(
            {
                "pypi": [{"id": "p", "packages": ["xarray"]}],
                "page": [{"id": "q", "url": "https://example.test"}],
            },
            directory=tmp_path,
            only=["pypi"],
            session=Session(),
            log=lambda _l: None,
        )
        assert all("pypi.org" in url for url in called)


class TestVisibleText:
    def test_scripts_and_styles_are_dropped(self, watch):
        html = (
            "<html><script>var x = 'a long line of javascript here';</script>"
            "<style>.a { color: red; padding: 1px 2px 3px; }</style>"
            "<p>A sentence that is long enough to keep around.</p></html>"
        )
        lines = watch.visible_text(html)
        assert lines == ["A sentence that is long enough to keep around."]

    def test_short_lines_and_duplicates_are_dropped(self, watch):
        html = "<p>ok</p><p>A repeated sentence long enough to keep.</p>" * 3
        assert watch.visible_text(html) == ["A repeated sentence long enough to keep."]

    def test_the_line_count_is_bounded(self, watch):
        html = "".join(
            f"<p>Sentence number {i} is long enough to keep.</p>" for i in range(999)
        )
        assert len(watch.visible_text(html, limit=50)) == 50
