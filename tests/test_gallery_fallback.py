"""scripts/gallery_fallback.py: a failed example keeps its last good output."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GOOD = (
    '<html><body><nav>old nav</nav><article class="bd-article"><h1>Glaciers</h1>'
    '<img src="../../_images/sphx_glr_plot_glaciers_001.png"/></article>'
    "<!-- esd-gallery-ran: 2026-09-20 --></body></html>"
)
FAILED = (
    '<html><body><nav>new nav</nav><article class="bd-article"><h1>Glaciers</h1>'
    '<div class="sphx-glr-script-out highlight-pytb notranslate"><pre>'
    "Traceback (most recent call last):\n  File &quot;x&quot;\n"
    "CredentialError: login failed\n</pre></div></article></body></html>"
)


@pytest.fixture(scope="module")
def fallback():
    spec = importlib.util.spec_from_file_location(
        "gallery_fallback", REPO_ROOT / "scripts" / "gallery_fallback.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gallery_fallback"] = module
    spec.loader.exec_module(module)
    return module


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    """A repo whose `deployed` branch is the last site; cwd is that repo."""
    repo = tmp_path / "repo"
    (repo / "auto_examples" / "boundaries").mkdir(parents=True)
    (repo / "_images").mkdir()
    (repo / "auto_examples/boundaries/plot_glaciers.html").write_text(GOOD)
    (repo / "auto_examples/boundaries/plot_mountains.html").write_text(FAILED)
    (repo / "_images/sphx_glr_plot_glaciers_001.png").write_bytes(b"figure")
    (repo / "_images/sphx_glr_plot_glaciers_thumb.png").write_bytes(b"thumb")
    (repo / "_images/sphx_glr_plot_glaciers_other_001.png").write_bytes(b"no")
    _git(repo, "init", "-q", "-b", "deployed")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "site")
    monkeypatch.chdir(repo)
    site = tmp_path / "site"
    (site / "auto_examples" / "boundaries").mkdir(parents=True)
    (site / "_images").mkdir()
    return site


def _run(fallback, site, tmp_path):
    return fallback.main(
        [
            "--site",
            str(site),
            "--source",
            str(tmp_path / "src"),
            "--ref",
            "deployed",
            "--today",
            "2026-09-29",
        ]
    )


def test_a_failed_example_gets_its_last_good_output(fallback, deployed, tmp_path):
    page = deployed / "auto_examples/boundaries/plot_glaciers.html"
    page.write_text(FAILED)
    assert _run(fallback, deployed, tmp_path) == 0
    text = page.read_text()
    assert "highlight-pytb" not in text and "sphx_glr_plot_glaciers_001.png" in text
    assert "new nav" in text  # only the article is swapped
    assert "Output from an earlier build" in text and "on 2026-09-20" in text
    assert "CredentialError: login failed" in text
    assert "esd-gallery-ran: 2026-09-20" in text  # the stale date is carried
    assert (
        deployed / "_images/sphx_glr_plot_glaciers_001.png"
    ).read_bytes() == b"figure"
    assert (
        tmp_path / "src/boundaries/images/thumb/sphx_glr_plot_glaciers_thumb.png"
    ).exists()
    assert not (deployed / "_images/sphx_glr_plot_glaciers_other_001.png").exists()


def test_a_page_that_ran_is_stamped_and_left_alone(fallback, deployed, tmp_path):
    page = deployed / "auto_examples/boundaries/plot_glaciers.html"
    page.write_text(GOOD.replace("<!-- esd-gallery-ran: 2026-09-20 -->", ""))
    _run(fallback, deployed, tmp_path)
    text = page.read_text()
    assert "esd-gallery-ran: 2026-09-29" in text and "earlier build" not in text


def test_nothing_good_to_fall_back_to_keeps_the_traceback(fallback, deployed, tmp_path):
    for name in ("plot_mountains", "plot_new"):
        (deployed / f"auto_examples/boundaries/{name}.html").write_text(FAILED)
    _run(fallback, deployed, tmp_path)
    for name in ("plot_mountains", "plot_new"):
        assert (
            "highlight-pytb"
            in (deployed / f"auto_examples/boundaries/{name}.html").read_text()
        )


def test_a_restored_page_that_fails_again_keeps_one_note(fallback, deployed, tmp_path):
    page = deployed / "auto_examples/boundaries/plot_glaciers.html"
    page.write_text(FAILED)
    _run(fallback, deployed, tmp_path)
    restored = page.read_text()
    # Deploy the restored page, then fail again.
    repo = Path.cwd()
    (repo / "auto_examples/boundaries/plot_glaciers.html").write_text(restored)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "again")
    page.write_text(FAILED)
    _run(fallback, deployed, tmp_path)
    text = page.read_text()
    assert text.count("Output from an earlier build") == 1 and "on 2026-09-20" in text
