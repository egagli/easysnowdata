"""Local Sphinx extension: everything on the site that is generated.

Three things are derived rather than written by hand, so that adding a
product cannot leave the docs behind (REVAMP_PLAN §2.11, §3.3, §7.2):

``docs/api/*.rst``
    One page per subpackage, walking ``__all__`` from the package down.
    Each function gets an autosummary stub whose template ends in a
    ``minigallery``, which is how "examples using ``esd.snow.snodas.load``"
    appears under the API entry.
``docs/catalog/*.md``
    One page per catalog product (added in the catalog-pages commit).
``docs/credentials.md``, ``docs/status.md``
    Generated from the auth provider registry and from
    ``data_status/history.json`` (added in the prose and health commits).

The other job of this module is telling the gallery which examples can run
without credentials. An example declares what it needs with a marker comment
**above its docstring**::

    # esd-requires: earthdata, earthengine
    \"\"\"
    Title
    =====
    \"\"\"

sphinx-gallery drops everything before the docstring, so the marker never
reaches the rendered page. Examples with no marker run on a pull request;
the rest run only in the scheduled build that has the secrets.
"""

from __future__ import annotations

import importlib
import inspect
import re
import types
from pathlib import Path
from typing import TYPE_CHECKING

from easysnowdata.catalog import pages

if TYPE_CHECKING:  # pragma: no cover — Sphinx is only needed to run the build
    from sphinx.application import Sphinx

try:  # the offline test tier imports this module without Sphinx installed
    from sphinx.util import logging
except ImportError:  # pragma: no cover
    import logging  # type: ignore[no-redef]

__all__ = [
    "GALLERY_DIR",
    "HISTORY_PATH",
    "REPO_ROOT",
    "example_requirements",
    "credential_free_examples",
    "credential_free_pattern",
    "setup",
]

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_DIR.parent
GALLERY_DIR = DOCS_DIR / "gallery"
HISTORY_PATH = REPO_ROOT / "data_status" / "history.json"

_REQUIRES_RE = re.compile(r"^#\s*esd-requires:\s*(.+)$", re.MULTILINE)


# ── the gallery's credential map ──────────────────────────────────────────────


def example_requirements() -> dict[str, tuple[str, ...]]:
    """Map every gallery script to the auth providers it needs.

    Keys are paths relative to ``docs/gallery`` (``"snow/plot_ucla_sr.py"``);
    an empty tuple means the example runs without credentials.
    """
    found: dict[str, tuple[str, ...]] = {}
    for path in sorted(GALLERY_DIR.rglob("plot_*.py")):
        # Only the header matters, and reading it is cheap.
        head = path.read_text(encoding="utf-8")[:4000]
        match = _REQUIRES_RE.search(head)
        needs = (
            tuple(part.strip() for part in match.group(1).split(",") if part.strip())
            if match
            else ()
        )
        found[path.relative_to(GALLERY_DIR).as_posix()] = needs
    return found


def credential_free_examples() -> list[str]:
    """The gallery scripts that need no credentials, as relative paths."""
    return [name for name, needs in example_requirements().items() if not needs]


def credential_free_pattern() -> str:
    """A ``filename_pattern`` that executes only the credential-free examples.

    sphinx-gallery matches this against the full path of each script, so the
    pattern anchors on the file name. Scripts that do not match are still
    rendered — with their source and a note — they are simply not run.
    """
    names = [Path(name).name for name in credential_free_examples()]
    if not names:  # pragma: no cover — every theme has at least one free example
        return r"(?!x)x"  # matches nothing
    return r"[\\/](" + "|".join(re.escape(name) for name in names) + ")$"


# ── generated API reference ───────────────────────────────────────────────────

#: Subpackages, in the order they appear in the API nav, with the heading to
#: give each group. Modules are discovered from each subpackage's ``__all__``.
API_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Core",
        ("aoi", "auth", "catalog", "config", "temporal"),
    ),
    (
        "Products",
        ("stations", "snow", "sar", "optical", "terrain", "land", "hydro", "climate"),
    ),
    (
        "Processing and plotting",
        ("processing", "plotting"),
    ),
    (
        "Providers",
        ("providers",),
    ),
    (
        "Deprecated modules",
        (
            "remote_sensing",
            "hydroclimatology",
            "topography",
            "automatic_weather_stations",
            "utils",
        ),
    ),
)

TITLES = {
    "aoi": "Areas of interest",
    "auth": "Credentials",
    "catalog": "Catalog",
    "config": "Configuration",
    "temporal": "Time inputs",
    "stations": "Stations",
    "snow": "Snow",
    "sar": "SAR",
    "optical": "Optical",
    "terrain": "Terrain",
    "land": "Land cover",
    "hydro": "Hydrography",
    "climate": "Climate",
    "processing": "Processing",
    "plotting": "Plotting",
    "providers": "Providers",
    "remote_sensing": "remote_sensing (deprecated)",
    "hydroclimatology": "hydroclimatology (deprecated)",
    "topography": "topography (deprecated)",
    "automatic_weather_stations": "automatic_weather_stations (deprecated)",
    "utils": "utils (deprecated)",
}


def _public_members(module: types.ModuleType) -> tuple[list[str], list[str]]:
    """Return ``(callables, submodules)`` from a module's ``__all__``."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in vars(module) if not n.startswith("_")]
    callables, submodules = [], []
    for name in names:
        try:
            obj = getattr(module, name)
        except AttributeError:  # pragma: no cover — a lazy shim attribute
            continue
        if inspect.ismodule(obj):
            if obj.__name__.startswith(module.__name__ + "."):
                submodules.append(name)
        elif inspect.isclass(obj) or inspect.isroutine(obj):
            callables.append(name)
    return callables, submodules


def _autosummary(fullnames: list[str], *, indent: str = "   ") -> list[str]:
    if not fullnames:
        return []
    lines = [
        ".. autosummary::",
        f"{indent}:toctree: ../generated/",
        f"{indent}:template: autosummary/base.rst",
        f"{indent}:nosignatures:",
        "",
    ]
    lines += [f"{indent}{name}" for name in fullnames]
    lines.append("")
    return lines


def _module_section(dotted: str, underline: str) -> list[str]:
    module = importlib.import_module(dotted)
    callables, submodules = _public_members(module)
    lines = [dotted, underline * len(dotted), ""]
    lines += [f".. automodule:: {dotted}", "   :no-members:", "   :no-index:", ""]
    # Bare names under the module autodoc has just entered: autosummary still
    # names the stub files by the full dotted path, so nothing collides.
    lines += _autosummary(callables)
    for sub in submodules:
        lines += _module_section(f"{dotted}.{sub}", "^" if underline == "-" else '"')
    return lines


def _api_page(name: str) -> str:
    title = TITLES.get(name, name)
    lines = [f".. _api-{name}:", "", title, "=" * len(title), ""]
    lines += _module_section(f"easysnowdata.{name}", "-")
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    """Write *text* only when it changed, so Sphinx's cache stays warm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _write_api_index(names: list[str]) -> str:
    lines = [
        "# API reference",
        "",
        "Every public function, grouped by subpackage. The `esd.<theme>.<product>`",
        "modules are the ones to reach for; the deprecated modules at the bottom are",
        "shims kept for one release (see [](../contributing.md)).",
        "",
        "```{toctree}",
        ":maxdepth: 2",
        "",
    ]
    lines += names
    lines += ["```", ""]
    return "\n".join(lines)


def generate_api(app: Sphinx) -> None:
    """Write ``docs/api/*.rst`` from the package's own ``__all__`` lists."""
    out = DOCS_DIR / "api"
    written: list[str] = []
    for _group, names in API_GROUPS:
        for name in names:
            try:
                _write(out / f"{name}.rst", _api_page(name))
            except Exception as exc:  # pragma: no cover — surfaced as a warning
                logger.warning("api page for %s failed: %s", name, exc)
                continue
            written.append(name)
    _write(out / "index.md", _write_api_index(written))
    logger.info("[esd] wrote %d API pages", len(written))


# ── "this example was not executed" notes ─────────────────────────────────────

_UNEXECUTED_NOTE = """.. admonition:: Not executed in this build
   :class: warning

   This example needs {providers} credentials, which the build that produced
   this page did not have, so the code below is shown without its output. The
   scheduled build refreshes it; :doc:`/credentials` has the setup to run it
   yourself.
"""

_CACHED_NOTE = """.. admonition:: Output from the last scheduled build
   :class: note

   This example needs {providers} credentials. The figures and printed output
   below come from the most recent scheduled build, which has them; this build
   reused that output instead of re-running the example.
"""

#: Both notes, so a page that already carries one is not annotated twice.
_NOTE_MARKERS = (
    "Not executed in this build",
    "Output from the last scheduled build",
)


def annotate_unexecuted(app: Sphinx) -> None:
    """Mark the generated pages of examples this build did not run.

    sphinx-gallery renders an example that ``filename_pattern`` skips exactly
    like one that ran and produced nothing, which is misleading. This runs
    after the gallery is generated and inserts a note under the title of every
    example that was skipped for want of credentials — one note when the page
    has no output at all, a different one when the output was restored from
    the cache of the last scheduled build.
    """
    conf = app.config.sphinx_gallery_conf
    if str(conf.get("plot_gallery", "True")).lower() in ("false", "0"):
        return  # nothing ran; the gallery index already says so
    pattern = re.compile(conf["filename_pattern"])
    out = DOCS_DIR / str(conf["gallery_dirs"])
    for name, needs in example_requirements().items():
        if not needs or pattern.search(str(GALLERY_DIR / name)):
            continue
        page = out / Path(name).with_suffix(".rst")
        if not page.exists():  # pragma: no cover — gallery layout changed
            continue
        text = page.read_text(encoding="utf-8")
        if any(marker in text for marker in _NOTE_MARKERS):
            continue  # already annotated: a restored cache carries its note
        # A page restored from the cache of a full build still has its figures
        # and printed output. Say where they came from rather than claiming
        # the example did not run.
        cached = "sphx-glr-script-out" in text or "image-sg::" in text
        template = _CACHED_NOTE if cached else _UNEXECUTED_NOTE
        note = template.format(providers=" and ".join(f"``{n}``" for n in needs))
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if set(line.strip()) == {"="} and i and lines[i - 1].strip():
                lines.insert(i + 1, "\n" + note)
                page.write_text("".join(lines), encoding="utf-8")
                break


# ── generated catalog pages ───────────────────────────────────────────────────


def generate_catalog(app: Sphinx) -> None:
    """Write ``docs/catalog/*.md``: one page per product, plus the index.

    The renderer lives in the package (:mod:`easysnowdata.catalog.pages`) so
    that the offline test tier can check it without Sphinx.
    """
    written = pages.write_all(
        DOCS_DIR / "catalog",
        history=pages.read_history(HISTORY_PATH),
        gallery_dir=str(GALLERY_DIR.relative_to(DOCS_DIR)),
        credentials=DOCS_DIR / "credentials.md",
    )
    logger.info("[esd] wrote %d catalog pages", len(written))


def setup(app: Sphinx) -> dict[str, object]:
    app.connect("builder-inited", generate_api, priority=100)
    app.connect("builder-inited", generate_catalog, priority=100)
    # After sphinx-gallery's own generate_gallery_rst (priority 500).
    app.connect("builder-inited", annotate_unexecuted, priority=600)
    return {
        "version": "1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
