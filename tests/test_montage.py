"""The README gallery montage.

Offline, against a synthetic gallery tree. The point of these tests is the one
thing that is easy to get wrong and invisible until someone looks at the
README: an example that did not run must not be tiled in as a blank square.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def montage_module():
    spec = importlib.util.spec_from_file_location(
        "make_montage", REPO_ROOT / "scripts" / "make_montage.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_montage"] = module
    spec.loader.exec_module(module)
    return module


def _example(gallery: Path, theme: str, name: str, *, executed: bool) -> Path:
    """Write a fake sphinx-gallery output for one example."""
    from PIL import Image

    images = gallery / theme / "images"
    (images / "thumb").mkdir(parents=True, exist_ok=True)
    thumb = images / "thumb" / f"sphx_glr_{name}_thumb.png"
    Image.new("RGB", (400, 280), (10, 20, 30)).save(thumb)
    if executed:
        Image.new("RGB", (640, 480), (200, 100, 50)).save(
            images / f"sphx_glr_{name}_001.png"
        )
    return thumb


class TestThumbnailSelection:
    def test_unexecuted_examples_are_left_out(self, montage_module, tmp_path):
        gallery = tmp_path / "auto_examples"
        _example(gallery, "snow", "plot_snodas", executed=True)
        _example(gallery, "snow", "plot_ucla_sr", executed=False)
        found = montage_module.thumbnails(gallery)
        assert [p.name for p in found] == ["sphx_glr_plot_snodas_thumb.png"]

    def test_themes_come_out_in_gallery_order(self, montage_module, tmp_path):
        gallery = tmp_path / "auto_examples"
        for theme in ("climate", "stations", "snow"):
            _example(gallery, theme, f"plot_{theme}", executed=True)
        found = montage_module.thumbnails(gallery)
        assert [p.parent.parent.parent.name for p in found] == [
            "stations",
            "snow",
            "climate",
        ]

    def test_an_unknown_theme_still_appears(self, montage_module, tmp_path):
        gallery = tmp_path / "auto_examples"
        _example(gallery, "snow", "plot_a", executed=True)
        _example(gallery, "zzz-new-theme", "plot_b", executed=True)
        assert len(montage_module.thumbnails(gallery)) == 2

    def test_a_multi_figure_example_is_tiled_once(self, montage_module, tmp_path):
        from PIL import Image

        gallery = tmp_path / "auto_examples"
        _example(gallery, "snow", "plot_snodas", executed=True)
        Image.new("RGB", (640, 480)).save(
            gallery / "snow" / "images" / "sphx_glr_plot_snodas_002.png"
        )
        assert len(montage_module.thumbnails(gallery)) == 1


class TestMontage:
    def test_grid_shape_and_cropping(self, montage_module, tmp_path):
        from PIL import Image

        gallery = tmp_path / "auto_examples"
        paths = [
            _example(gallery, "snow", f"plot_{i}", executed=True) for i in range(5)
        ]
        out = montage_module.montage(
            paths, tmp_path / "g.png", columns=3, width=300, gap=0
        )
        with Image.open(out) as image:
            # 3 columns of 100 px at 4:3, two rows.
            assert image.size == (300, 150)

    def test_webp_is_written_for_a_webp_suffix(self, montage_module, tmp_path):
        from PIL import Image

        gallery = tmp_path / "auto_examples"
        paths = [_example(gallery, "snow", "plot_a", executed=True)]
        out = montage_module.montage(paths, tmp_path / "g.webp", columns=2, width=200)
        with Image.open(out) as image:
            assert image.format == "WEBP"

    def test_an_empty_gallery_says_so(self, montage_module, tmp_path):
        with pytest.raises(RuntimeError, match="build the gallery first"):
            montage_module.montage([], tmp_path / "g.png")

    def test_main_reports_a_missing_gallery(self, montage_module, tmp_path, capsys):
        code = montage_module.main(["--gallery", str(tmp_path / "nope")])
        assert code == 1
        assert "Run `pixi run -e docs docs-build-free`" in capsys.readouterr().err


class TestReadmeBlocks:
    """The README's generated blocks, from scripts/update_readme_status.py."""

    @pytest.fixture(scope="class")
    def status(self):
        spec = importlib.util.spec_from_file_location(
            "update_readme_status", REPO_ROOT / "scripts" / "update_readme_status.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["update_readme_status"] = module
        spec.loader.exec_module(module)
        return module

    def test_the_status_table_is_health_probes_only(self, status):
        run = [
            {
                "source": "A",
                "status": "pass",
                "kind": "health",
                "checked_at": "2026-09-14T00:00:00Z",
            },
            {
                "source": "A latency",
                "status": "pass",
                "kind": "latency",
                "value": "x",
                "checked_at": "2026-09-14T00:00:00Z",
            },
        ]
        table = status.build_table([run])
        assert "| A |" in table
        assert "A latency" not in table

    def test_the_catalog_block_lists_every_product(self, status):
        from easysnowdata import catalog

        block = status.build_catalog_table()
        assert f"{len(catalog.products())} products" in block
        for pid in catalog.products():
            assert f"`{pid}`" in block

    def test_both_blocks_are_replaced_in_place(self, status, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            "top\n"
            f"{status.SENTINEL_START}\nold\n{status.SENTINEL_END}\n"
            "middle\n"
            f"{status.CATALOG_START}\nold\n{status.CATALOG_END}\n"
            "bottom\n"
        )
        status.update_readme("STATUS\n", readme, catalog_table="CATALOG\n")
        text = readme.read_text()
        assert "STATUS" in text and "CATALOG" in text and "old" not in text
        assert text.startswith("top") and text.rstrip().endswith("bottom")

    def test_a_missing_marker_is_an_error_not_a_silent_no_op(self, status, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("nothing to replace\n")
        with pytest.raises(ValueError, match="Add the markers first"):
            status.update_readme("x", readme)

    def test_an_empty_history_leaves_a_note_rather_than_a_table(self, status):
        assert "No data yet" in status.build_table([])
