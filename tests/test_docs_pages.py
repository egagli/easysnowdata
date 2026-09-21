"""The generated documentation: catalog pages, credentials page, gallery links.

Offline. These are the tests that keep §2.11's "every product has all four
artefacts" true — a product added without a gallery example, or a gallery
script no product claims, fails here rather than quietly producing a thinner
page.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from easysnowdata import auth, catalog
from easysnowdata.catalog import pages

REPO_ROOT = Path(__file__).resolve().parent.parent
GALLERY = REPO_ROOT / "docs" / "gallery"

#: Products with no gallery example, and why. Anything else must have one.
#: Empty since 0.3: every product is drawn somewhere in the gallery.
NO_EXAMPLE: dict[str, str] = {}


@pytest.fixture(scope="module")
def esd_docs():
    """The docs extension, importable without Sphinx installed."""
    sys.path.insert(0, str(REPO_ROOT / "docs" / "_ext"))
    try:
        import esd_docs as module
    finally:
        sys.path.pop(0)
    return module


class TestGalleryLinks:
    def test_every_product_has_an_example(self):
        missing = {
            pid
            for pid, product in catalog.products().items()
            if not product.examples and pid not in NO_EXAMPLE
        }
        assert not missing, (
            f"products without a gallery example: {sorted(missing)}. Add one under "
            "docs/gallery/<theme>/plot_<product>.py and list it in the catalog "
            "entry's examples=, or record the reason in NO_EXAMPLE."
        )

    def test_recorded_exceptions_are_still_exceptions(self):
        # If one of these grows an example, delete its excuse.
        for pid, reason in NO_EXAMPLE.items():
            assert not catalog.get(pid).examples, f"{pid} now has an example: {reason}"

    def test_declared_examples_exist(self):
        for product in catalog.products().values():
            for example in product.examples:
                assert (GALLERY / example).is_file(), f"{product.id} -> {example}"

    def test_no_orphan_gallery_scripts(self):
        claimed = {e for p in catalog.products().values() for e in p.examples}
        on_disk = {
            path.relative_to(GALLERY).as_posix()
            for path in GALLERY.rglob("plot_*.py")
            # tools/ holds the cross-cutting examples (AOI, time, water years)
            # that belong to no single product.
            if path.relative_to(GALLERY).parts[0] != "tools"
        }
        assert on_disk - claimed == set(), (
            "gallery scripts no catalog entry claims: "
            f"{sorted(on_disk - claimed)}. Add them to a product's examples=."
        )
        assert claimed <= on_disk


class TestCredentialMarkers:
    def test_markers_name_known_providers(self, esd_docs):
        for name, needs in esd_docs.example_requirements().items():
            for provider in needs:
                assert provider in auth.PROVIDERS, (
                    f"{name}: unknown provider {provider}"
                )

    def test_free_examples_need_nothing(self, esd_docs):
        free = set(esd_docs.credential_free_examples())
        requirements = esd_docs.example_requirements()
        assert free == {n for n, needs in requirements.items() if not needs}
        # The credentialed examples run in the scheduled build, which has the
        # secrets; the pull-request build still needs something to execute.
        assert len(free) >= 5
        sections = {Path(name).parent.name for name in requirements}
        assert sections <= {*catalog.KNOWN_THEMES, "tools"}

    def test_pattern_selects_exactly_the_free_examples(self, esd_docs):
        pattern = re.compile(esd_docs.credential_free_pattern())
        for name, needs in esd_docs.example_requirements().items():
            hit = bool(pattern.search(str(GALLERY / name)))
            assert hit is (not needs), name

    def test_a_marked_example_declares_what_its_product_needs(self, esd_docs):
        """A marker covers the default route, and names only routes that exist.

        An example may go beyond its product's default route to compare a
        credentialed alternative (SNODAS against its Earth Engine mirror), so
        extra markers are allowed when some route of some product the example
        draws requires them. The tools examples belong to no product and are
        checked only for known providers.
        """
        requirements = esd_docs.example_requirements()
        claimed: dict[str, set[str]] = {}
        for product in catalog.products().values():
            for example in product.examples:
                claimed.setdefault(example, set()).add(product.id)
        for name, needs in requirements.items():
            if name not in claimed:
                assert Path(name).parts[0] == "tools", f"{name} is unclaimed"
                continue
            products = [catalog.get(pid) for pid in claimed[name]]
            # the default route of at least one claiming product must be covered
            defaults = [set(p.requires) for p in products]
            assert any(d <= set(needs) for d in defaults), (
                f"{name}: marker {needs} does not cover a default route ({defaults})"
            )
            offered = {r for p in products for s in p.sources for r in s.requires}
            assert set(needs) <= offered, (
                f"{name}: marker names {set(needs) - offered}, which no route "
                "of its products requires"
            )
        # the two anchors that must stay marked
        assert requirements["snow/plot_ucla_sr.py"] == ("earthdata",)
        assert requirements["stations/plot_nve_stations.py"] == ("nve",)


class TestProductPage:
    def test_contains_the_catalog_entry(self):
        product = catalog.get("snodas")
        page = pages.product_page(product)
        assert page.startswith(f"# {product.title}")
        assert "`snodas` · theme `snow`" in page
        for source in product.sources:
            assert f"`{source.id}`" in page
            assert source.location in page
        assert product.citation.strip() in page
        assert product.license in page
        assert f"https://doi.org/{product.doi}" in page
        assert "../auto_examples/snow/plot_snodas" in page
        assert "esd.snow.snodas.load(aoi" in page

    def test_example_cards_use_the_title_and_thumbnail_when_known(self):
        page = pages.product_page(
            catalog.get("snodas"),
            examples={
                "snow/plot_snodas.py": {
                    "title": "SNODAS against the mirror",
                    "thumbnail": "../auto_examples/snow/images/thumb/x.png",
                }
            },
        )
        assert ":::{grid-item-card} SNODAS against the mirror" in page
        assert ":img-top: ../auto_examples/snow/images/thumb/x.png" in page

    def test_an_unexecuted_example_gets_a_card_without_an_image(self):
        # The thumbnail is left out rather than pointing at the placeholder.
        page = pages.product_page(
            catalog.get("snodas"), examples={"snow/plot_snodas.py": {"title": "T"}}
        )
        assert ":::{grid-item-card} T" in page
        assert ":img-top:" not in page

    def test_categorical_variables_get_a_class_table(self):
        page = pages.product_page(catalog.get("esa-worldcover"))
        assert "classes (11)" in page
        assert "| 70 | Snow and ice |" in page

    def test_credential_notes_follow_the_default_source(self):
        # Default open, alternative credentialed.
        assert "No account needed for the default route" in pages.product_page(
            catalog.get("snodas")
        )
        # Default credentialed, open alternative.
        hls = pages.product_page(catalog.get("hls"))
        assert "The default route needs `earthdata`" in hls
        assert 'An open alternative exists: `source="planetary-computer"`' in hls
        # Nothing anywhere needs an account.
        assert "No account needed" in pages.product_page(catalog.get("koppen-geiger"))

    def test_station_products_get_the_two_call_quickstart(self):
        page = pages.product_page(catalog.get("awdb-stations"))
        assert 'esd.stations.inventory(aoi, networks="awdb")' in page
        assert "esd.stations.load(inv" in page

    def test_every_product_renders(self):
        for product in catalog.products().values():
            page = pages.product_page(product)
            assert page.startswith("# ")
            assert "## Sources" in page


class TestHealthBadges:
    HISTORY = [
        [
            {
                "source": "A",
                "status": "fail",
                "error": "boom",
                "checked_at": "2026-09-14T00:00:00Z",
            },
            {
                "source": "B",
                "status": "pass",
                "error": None,
                "checked_at": "2026-09-14T00:00:00Z",
            },
        ],
        [
            {
                "source": "A",
                "status": "pass",
                "error": None,
                "checked_at": "2026-09-07T00:00:00Z",
            },
            {
                "source": "C",
                "status": "skip",
                "error": "no creds",
                "checked_at": "2026-09-07T00:00:00Z",
            },
        ],
    ]

    def test_latest_wins_and_older_labels_survive(self):
        latest = pages.latest_status(self.HISTORY)
        assert latest["A"]["status"] == "fail"  # newest run first
        assert latest["B"]["status"] == "pass"
        assert latest["C"]["status"] == "skip"  # only in the older run

    def test_missing_history_is_not_an_error(self, tmp_path):
        assert pages.read_history(tmp_path / "nope.json") == []

    def test_product_badge_prefers_a_working_route(self):
        product = catalog.get("snodas")
        labels = [
            p.label for s in product.sources for p in s.health if p.kind == "health"
        ]
        history = [
            [
                {
                    "source": labels[0],
                    "status": "pass",
                    "error": None,
                    "checked_at": "x",
                },
                {
                    "source": labels[1],
                    "status": "fail",
                    "error": "x",
                    "checked_at": "x",
                },
            ]
        ]
        page = pages.product_page(product, latest=pages.latest_status(history))
        assert pages.BADGES["pass"] in page
        assert pages.BADGES["fail"] in page  # the broken route still says so

    def test_all_routes_failing_reads_as_failing(self):
        product = catalog.get("snodas")
        history = [
            [
                {"source": p.label, "status": "fail", "error": "x", "checked_at": "x"}
                for s in product.sources
                for p in s.health
                if p.kind == "health"
            ]
        ]
        latest = pages.latest_status(history)
        assert pages._product_status(product, latest) == "fail"


class TestIndexAndCredentialsPages:
    def test_index_lists_every_product(self):
        page = pages.index_page()
        for pid, product in catalog.products().items():
            assert f"]({pid}.md)" in page
            assert product.theme in page
        assert f"{len(catalog.products())} products" in page

    def test_credentials_page_covers_every_provider(self):
        page = pages.credentials_page()
        for name, provider in auth.PROVIDERS.items():
            assert provider.title in page
            assert f"({name.replace('_', '-')})=" in page
            for var in provider.env_vars:
                assert f"`{var}`" in page

    def test_credentials_page_lists_the_products_per_provider(self):
        page = pages.credentials_page()
        assert "(catalog/hls.md)" in page  # earthdata
        assert "(catalog/nlcd.md)" in page  # earthengine
        assert "(catalog/nve-stations.md)" in page  # nve

    def test_write_all_is_idempotent(self, tmp_path):
        first = pages.write_all(tmp_path, credentials=tmp_path / "credentials.md")
        assert len(first) == len(catalog.products()) + 2
        stamps = {p: p.stat().st_mtime_ns for p in first}
        pages.write_all(tmp_path, credentials=tmp_path / "credentials.md")
        assert {p: p.stat().st_mtime_ns for p in first} == stamps


class TestDescribeUsesTheSameTables:
    def test_describe_and_page_share_the_source_table(self):
        product = catalog.get("copernicus-dem")
        rows = pages.sources_table(product)
        text = catalog.describe(product.id)
        page = pages.product_page(product)
        for row in rows:
            assert row in text
            assert row in page
