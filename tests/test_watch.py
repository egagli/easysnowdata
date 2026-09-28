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
                named = set(entry.get("products", []))
                for products in (entry.get("match_products") or {}).values():
                    named |= set(products)
                unknown = named - known
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
            for product in [
                *entry.get("products", []),
                *(
                    p
                    for products in (entry.get("match_products") or {}).values()
                    for p in products
                ),
            ]
        }
        assert {"modis-snow", "viirs-snow", "hls", "sentinel-1-rtc", "era5"} <= watched
        # Every boundaries product and the hillshade: static files that are
        # still republished (Census yearly, RGI versions, a mirror for RGI 6.0).
        boundaries = set(catalog.list(theme="boundaries").index)
        assert boundaries | {"hillshade"} <= watched


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

    @pytest.mark.parametrize(
        "text",
        [
            "Free courses for your team",  # "urs" inside a word
            "An academic partnership",  # "dem"
            "The answer is here",  # "swe"
        ],
    )
    def test_short_keywords_match_whole_words_only(self, watch, text):
        assert watch.classify(text) == []

    def test_a_sensor_name_alone_is_not_worth_adding(self, watch):
        # #51: every line naming VIIRS, MODIS, HLS or ERA5 used to land here.
        assert "new-dataset" not in watch.classify(
            "NASA/VIIRS/002/AERDB_D3_VIIRS_NOAA20"
        )
        assert "new-dataset" not in watch.classify("ERA5 end moved")
        assert "new-dataset" in watch.classify("New VIIRS snow cover product released")
        assert "new-dataset" in watch.classify(
            "Snowmelt timing maps now in the catalog"
        )


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

    def test_a_temporal_end_moving_forward_is_routine(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY,
            {"a": {"temporal_end": "2025-01-01"}},
            {"a": {"temporal_end": "2026-01-01"}},
        )
        assert changes[0].categories == ["routine"]
        assert changes[0].section == "Routine"

    @pytest.mark.parametrize(
        ("old", "new", "section"),
        [
            ("ongoing", "2026-01-01", "Action needed"),  # it gained an end
            ("2026-01-01", "2025-06-01", "Action needed"),  # data withdrawn
            ("2026-01-01", "ongoing", "Changed"),  # live again
        ],
    )
    def test_an_end_that_stops_or_moves_back_is_not_routine(
        self, watch, old, new, section
    ):
        changes = watch.diff_mapping(
            self.ENTRY, {"a": {"temporal_end": old}}, {"a": {"temporal_end": new}}
        )
        assert changes[0].section == section

    def test_the_51_era5_case(self, watch):
        """#51: ERA5's end date moving a week was "Worth adding" and blamed on six products."""
        entry = {
            "id": "gee-assets",
            "products": [],
            "match_products": {
                "ECMWF/ERA5_LAND": ["era5"],
                "OPERA/RTC": ["sentinel-1-rtc"],
            },
        }
        before = {
            "ECMWF/ERA5_LAND/HOURLY": {"end": "2026-09-15", "updated": "2026-09-21"},
            "OPERA/RTC/L2_V1/S1": {"end": "2026-09-20", "window_days": 30},
        }
        after = {
            "ECMWF/ERA5_LAND/HOURLY": {"end": "2026-09-22", "updated": "2026-09-28"},
            "OPERA/RTC/L2_V1/S1": {"end": "2026-09-20", "window_days": 365},
        }
        changes = watch.diff_mapping(entry, before, after)
        by_subject = {(c.subject, c.detail.split(":")[0]): c for c in changes}
        era5 = by_subject[("ECMWF/ERA5_LAND/HOURLY", "end")]
        assert era5.section == "Routine" and era5.products == ["era5"]
        stalled = by_subject[("OPERA/RTC/L2_V1/S1", "window_days")]
        assert stalled.section == "Action needed"
        assert stalled.categories == ["stalled"]
        assert stalled.products == ["sentinel-1-rtc"]

    def test_a_file_republished_and_a_license_changing(self, watch):
        changes = watch.diff_mapping(
            self.ENTRY,
            {"f": {"etag": "a", "status": 200}, "c": {"license": "CC-BY-4.0"}},
            {"f": {"etag": "b", "status": 404}, "c": {"license": "proprietary"}},
        )
        sections = {(c.subject, c.detail.split(":")[0]): c.section for c in changes}
        assert sections == {
            ("c", "license"): "Action needed",
            ("f", "etag"): "Changed",
            ("f", "status"): "Action needed",
        }

    def test_next_years_file_appearing_is_worth_adding(self, watch):
        (change,) = watch.diff_mapping(
            self.ENTRY, {"f": {"status": 404}}, {"f": {"status": 206}}
        )
        assert change.section == "Worth adding"

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

    SCOPED = {
        "id": "alerts",
        "url": "https://example.test",
        "products": [],
        "keywords": ["nsidc"],
        "match_products": {"modis": ["modis-snow"]},
    }

    @pytest.mark.parametrize(
        ("line", "section", "products"),
        [
            # News about a dataset we do not use is never an action for us.
            ("GEDI Level 4A Version 3 released", "Other", []),
            ("ASDC planned monthly maintenance", "Other", []),
            # Naming one of ours, or a keyword (a DAAC we read from), it stands.
            ("MODIS maintenance on Tuesday", "Action needed", ["modis-snow"]),
            ("NSIDC maintenance on Tuesday", "Action needed", []),
            # A snow term is worth a look wherever it comes from.
            ("A new glacier inventory is released", "Worth adding", []),
        ],
    )
    def test_a_scoped_page_reports_only_what_touches_us(
        self, watch, line, section, products
    ):
        (change,) = watch.diff_lines(self.SCOPED, {"lines": ["x"]}, {"lines": [line]})
        assert change.section == section and change.products == products

    def test_lines_that_change_on_every_visit_are_ignored(self, watch):
        entry = {**self.ENTRY, "ignore": [r"^Feral Swine"]}
        changes = watch.diff_lines(
            entry,
            {"lines": ["x"]},
            {
                "lines": [
                    "]. Date Accessed 09-28-2026.",
                    "Feral Swine Eradication and Control Pilot Program",
                    "MOD10A2 is deprecated",
                ]
            },
        )
        assert [c.detail for c in changes] == ["MOD10A2 is deprecated"]

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
                "<summary><b>Other",
            )
        ]
        assert positions == sorted(positions)
        assert "who knows" in body  # §8: an unmatched item is listed, never dropped

    def test_a_long_section_is_capped_but_counted(self, watch):
        changes = [
            watch.Change("noisy-page", f"S{i}", "x", categories=["extent"])
            for i in range(40)
        ]
        body = watch.digest(changes, [], when="2026-09-17")
        assert "## Changed (40)" in body
        assert "...and 15 more, from noisy-page" in body

    def test_to_dos_are_checkboxes_and_never_capped(self, watch):
        changes = [
            watch.Change("e", f"S{i}", "x", categories=["removal"]) for i in range(40)
        ]
        body = watch.digest(changes, [], when="2026-09-17")
        assert body.count("- [ ] **S") == 40 and "more, from" not in body

    def test_routine_and_other_are_folded(self, watch):
        body = watch.digest(
            [
                watch.Change("e", "A", "end moved", categories=["routine"]),
                watch.Change("e", "B", "who knows", categories=[]),
            ],
            [],
        )
        assert "<details><summary><b>Routine (1)</b></summary>" in body
        assert "<details><summary><b>Other (1)</b></summary>" in body
        assert "## Routine" not in body

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


LEGACY_51 = """<!-- esd-watch: 2026-09-28 -->

## Worth adding (1)

- **[ERA5](u)** end: `a` → `b` — affects `era5` `new-dataset`

## Changed (1)

- **awdb** count: `1` → `2`

## Other (1)

- **x** y
"""


class TestSupersede:
    def test_unticked_to_dos_are_carried_and_ticked_ones_are_not(self, watch):
        body = watch.digest(
            [
                watch.Change("e", "A", "gone", categories=["removal"]),
                watch.Change("e", "B", "new", categories=["new-dataset"]),
                watch.Change("e", "C", "moved", categories=["extent"]),
            ],
            [],
            when="2026-10-05",
        )
        body = body.replace("- [ ] **A**", "- [x] **A**")
        assert watch.unaddressed(body, 60) == ["**B** new `new-dataset` _(from #60)_"]

    def test_a_digest_from_before_the_checkboxes_loses_nothing(self, watch):
        assert watch.unaddressed(LEGACY_51, 51) == [
            "**[ERA5](u)** end: `a` → `b` — affects `era5` `new-dataset` _(from #51)_"
        ]

    def test_carried_items_keep_their_origin_and_are_not_repeated(self, watch):
        carried = watch.unaddressed(LEGACY_51, 51)
        body = watch.digest(
            [watch.Change("e", "A", "gone", categories=["removal"])],
            [],
            carried=[*carried, "**A** gone `removal` _(from #51)_"],
            discussions=[51],
        )
        assert "## Carried over (1)" in body
        assert "- [ ] **[ERA5](u)**" in body and "_(from #51)_" in body
        assert body.count("**A** gone") == 1  # already in this week's list
        assert "Earlier discussion is in #51." in body
        # carried again next week, still from #51
        assert watch.unaddressed(body, 70)[-1].endswith("_(from #51)_")

    def test_the_old_digest_closes_only_after_the_new_one_opens(
        self, watch, tmp_path, monkeypatch
    ):
        calls: list[tuple[str, str]] = []

        def github(method, path, token, **kwargs):
            calls.append((method, path))
            if method == "GET":
                return [
                    {"number": 51, "body": LEGACY_51, "comments": 0},
                    {"number": 52, "pull_request": {}, "body": ""},
                ]
            if path.endswith("/issues"):
                assert "_(from #51)_" in kwargs["json"]["body"]
                return {"number": 60, "html_url": "https://x/60"}
            return {}

        monkeypatch.setattr(watch, "_github", github)
        monkeypatch.setattr(
            watch,
            "run",
            lambda *a, **k: (
                [watch.Change("e", "A", "gone", categories=["removal"])],
                [],
            ),
        )
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        (tmp_path / "w.toml").write_text("")
        assert (
            watch.main(
                ["--watchlist", str(tmp_path / "w.toml"), "--issue", "--repo", "o/r"]
            )
            == 0
        )
        assert calls == [
            ("GET", "/repos/o/r/issues"),
            ("POST", "/repos/o/r/issues"),
            ("POST", "/repos/o/r/issues/51/comments"),
            ("PATCH", "/repos/o/r/issues/51"),
        ]

    def test_a_failed_open_closes_nothing(self, watch, tmp_path, monkeypatch):
        calls: list[str] = []

        def github(method, path, token, **kwargs):
            calls.append(method)
            if method == "GET":
                return [{"number": 51, "body": LEGACY_51, "comments": 0}]
            raise RuntimeError("HTTP 502")

        monkeypatch.setattr(watch, "_github", github)
        monkeypatch.setattr(
            watch,
            "run",
            lambda *a, **k: (
                [watch.Change("e", "A", "gone", categories=["removal"])],
                [],
            ),
        )
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        (tmp_path / "w.toml").write_text("")
        with pytest.raises(RuntimeError):
            watch.main(
                ["--watchlist", str(tmp_path / "w.toml"), "--issue", "--repo", "o/r"]
            )
        assert calls == ["GET", "POST"]


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
