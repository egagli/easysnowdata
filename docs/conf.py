"""Sphinx configuration (REVAMP_PLAN §7.2, decision Q6).

The site is built from three sources and nothing is hand-maintained twice:

* **prose** — the Markdown pages in this directory, parsed by MyST;
* **the gallery** — ``docs/gallery/<theme>/plot_<product>.py``, executed by
  sphinx-gallery into ``docs/auto_examples`` with thumbnails, an index,
  downloadable notebooks, and the back-references that put "examples using
  ``esd.snow.snodas.load``" on the API and catalog pages;
* **the catalog** — ``easysnowdata.catalog``, which the local ``esd_docs``
  extension renders into one page per product, the credentials page, the
  status page and the API reference.

Nothing generated is committed: ``docs/auto_examples``, ``docs/catalog``,
``docs/api``, ``docs/gen_modules``, ``docs/credentials.md``, ``docs/status.md``
and ``docs/_build`` are all in ``.gitignore``.

Gallery execution is chosen with ``ESD_DOCS_GALLERY``:

``full``
    Execute every example. Needs the provider credentials; examples whose
    credentials are missing fail and are reported, so the scheduled build is
    also a credential check.
``free`` (the default)
    Execute only the examples that declare no ``# esd-requires:`` marker.
    This is what a pull request builds — no secrets, no live credentialed
    requests. The rest are rendered as source with a note.
``none``
    Execute nothing; the fastest way to check prose, API and catalog pages.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE / "_ext"))

# Before importing easysnowdata: no credential banner in the build log, and
# no "today" surprises in the examples.
os.environ.setdefault("EASYSNOWDATA_QUIET", "1")

from esd_docs import GALLERY_DIR, credential_free_pattern  # noqa: E402

import easysnowdata  # noqa: E402

# ── Project ───────────────────────────────────────────────────────────────────
project = "easysnowdata"
author = "Eric Gagliano"
copyright = f"{datetime.now(tz=UTC):%Y}, Eric Gagliano"  # noqa: A001
release = easysnowdata.__version__
version = release.split("+")[0]

# ── General ───────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_nb",
    "esd_docs",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_ext",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    # sphinx-gallery reads these; Sphinx must not also try to parse the
    # per-theme README files as standalone documents.
    "gallery",
    # Multi-gigabyte local downloads, never part of the site.
    "examples/planet_data",
    "examples/data",
    # A local scratch notebook (untracked) that is not part of the docs.
    "examples/sandbox.ipynb",
    # sphinx-gallery writes a .py, .ipynb and .zip beside every generated
    # .rst; myst-nb claims .ipynb as a source suffix, so without this Sphinx
    # sees several candidate sources for one document.
    "auto_examples/**/*.ipynb",
    "README.md",
]

suppress_warnings = ["mystnb.unknown_mime_type", "config.cache"]
default_role = "py:obj"
nitpicky = False

# ── MyST and notebooks ────────────────────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# The notebooks under docs/examples/ are the legacy long-form user guides.
# They carry their own outputs, were run by hand against credentialed
# sources, and are not re-executed here (REVAMP_PLAN §7.1). The gallery is
# what CI runs.
nb_execution_mode = "off"
nb_merge_streams = True

# ── autodoc / autosummary ─────────────────────────────────────────────────────
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable", None),
    "dask": ("https://docs.dask.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "pystac_client": ("https://pystac-client.readthedocs.io/en/stable", None),
    "odc-geo": ("https://odc-geo.readthedocs.io/en/latest", None),
}
# Network failures against a peer's objects.inv must not fail a docs build.
intersphinx_timeout = 10

extlinks = {
    "issue": ("https://github.com/egagli/easysnowdata/issues/%s", "issue #%s"),
    "plan": (
        "https://github.com/egagli/easysnowdata/blob/main/REVAMP_PLAN.md#%s",
        "§%s",
    ),
}

# ── sphinx-gallery ────────────────────────────────────────────────────────────
GALLERY_MODE = os.environ.get("ESD_DOCS_GALLERY", "free").lower()
if GALLERY_MODE not in {"full", "free", "none"}:
    raise ValueError(
        f"ESD_DOCS_GALLERY={GALLERY_MODE!r}; expected 'full', 'free' or 'none'."
    )

sphinx_gallery_conf = {
    "examples_dirs": str(GALLERY_DIR.relative_to(HERE)),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"[\\/]plot_"
    if GALLERY_MODE == "full"
    else credential_free_pattern(),
    "plot_gallery": "False" if GALLERY_MODE == "none" else "True",
    "subsection_order": [
        "gallery/stations",
        "gallery/snow",
        "gallery/sar",
        "gallery/optical",
        "gallery/terrain",
        "gallery/land",
        "gallery/hydro",
        "gallery/climate",
    ],
    "within_subsection_order": "FileNameSortKey",
    "doc_module": ("easysnowdata",),
    "prefer_full_module": {"easysnowdata"},
    "backreferences_dir": "gen_modules/backreferences",
    "reference_url": {"easysnowdata": None},
    "remove_config_comments": True,
    "download_all_examples": False,
    "thumbnail_size": (500, 350),
    "min_reported_time": 5,
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": False,
    "show_signature": False,
    # A provider hiccup during a scheduled build should leave a visible,
    # reported failure in the log and on the page, not kill the deploy.
    "only_warn_on_example_error": True,
    "abort_on_example_error": False,
    "line_numbers": False,
    "nested_sections": False,
}

# ── HTML ──────────────────────────────────────────────────────────────────────
html_theme = "pydata_sphinx_theme"
html_title = "easysnowdata"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False
html_sidebars = {
    "index": [],
    "auto_examples/index": [],
}
html_theme_options = {
    "github_url": "https://github.com/egagli/easysnowdata",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/easysnowdata/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "show_prev_next": True,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}
html_context = {
    "github_user": "egagli",
    "github_repo": "easysnowdata",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
