"""Render the catalog as documentation pages (§3.3, §7.2).

One page per product — description, the comparison of its access routes, its
variables, what credentials it needs, its licence and citation, its health
badge and the gallery examples that use it — plus an index. Everything comes
from the registry, so a product that is added, re-sourced or retired changes
its page in the same commit.

The output is MyST Markdown for Sphinx (``{bdg-*}`` badges from sphinx-design,
``{minigallery}`` from sphinx-gallery), but nothing here imports Sphinx: the
renderer is plain string building, which is what lets the offline test tier
check it.

::

    from easysnowdata.catalog import pages
    pages.write_all(Path("docs/catalog"), history=pages.read_history(path))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from easysnowdata.catalog import _registry
from easysnowdata.catalog._models import Product, Source

__all__ = [
    "read_history",
    "latest_status",
    "sources_table",
    "variables_table",
    "product_page",
    "quickstart",
    "index_page",
    "credentials_page",
    "status_page",
    "sparkline",
    "write_all",
]

#: How a probe status is drawn, as a sphinx-design badge role.
BADGES = {
    "pass": "{bdg-success}`live`",
    "fail": "{bdg-danger}`failing`",
    "skip": "{bdg-warning}`not checked`",
    None: "{bdg-secondary}`no data`",
}


# ── health history ────────────────────────────────────────────────────────────


def read_history(path: str | Path) -> list[list[dict[str, Any]]]:
    """Read ``data_status/history.json``; an empty list when it is absent."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):  # pragma: no cover — corrupt file
        return []


def latest_status(history: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """The most recent result per probe label.

    Keyed by the probe label alone, not by product and route: the label is
    what stays stable when a product is re-sourced, which is what keeps the
    history continuous (§8 — two SNOTEL/CCSS labels are deliberately
    inherited from a retired entry for exactly this reason), and it is the
    only key the runs recorded before the catalog existed carry.
    """
    latest: dict[str, dict[str, Any]] = {}
    for run in history:  # newest first
        for result in run:
            latest.setdefault(result["source"], result)
    return latest


def _source_status(
    product: Product, source: Source, latest: dict[str, dict[str, Any]]
) -> str | None:
    """The worst status among a source's health probes: fail beats skip beats pass.

    Only ``health`` probes count. A latency probe that times out says the
    measurement failed, not that the route is unreachable.
    """
    seen = [
        latest.get(probe.label, {}).get("status")
        for probe in source.health
        if probe.kind == "health"
    ]
    seen = [s for s in seen if s]
    for status in ("fail", "skip", "pass"):
        if status in seen:
            return status
    return None


def _product_status(product: Product, latest: dict[str, dict[str, Any]]) -> str | None:
    """A product is live if any of its routes is; failing only if all fail."""
    seen = [_source_status(product, src, latest) for src in product.sources]
    seen = [s for s in seen if s]
    if not seen:
        return None
    if "pass" in seen:
        return "pass"
    return "fail" if "fail" in seen else "skip"


# ── shared tables (also used by catalog.describe) ─────────────────────────────


def sources_table(product: Product) -> list[str]:
    """The Markdown source-comparison table, one row per access route."""
    lines = [
        "| source | provider | credentials | resolution | extent | temporal | latency | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in enumerate(product.sources):
        default = " (default)" if i == 0 else ""
        res = f"{s.resolution_m:g} m" if s.resolution_m else "—"
        lines.append(
            f"| `{s.id}`{default} | {s.provider} | {', '.join(s.requires) or 'none'} | {res} | "
            f"{s.extent} | {s.temporal or '—'} | {s.latency or '—'} | {s.notes} |"
        )
    return lines


def variables_table(product: Product) -> list[str]:
    """The Markdown variable table, or an empty list when there are none."""
    if not product.variables:
        return []
    lines = [
        "| variable | units | dtype | nodata | categorical |",
        "| --- | --- | --- | --- | --- |",
    ]
    for v in product.variables:
        categorical = f"yes ({len(v.flag_values)} classes)" if v.categorical else "no"
        lines.append(
            f"| `{v.name}` | {v.units or '—'} | {v.dtype or '—'} | "
            f"{v.nodata if v.nodata is not None else '—'} | {categorical} |"
        )
    return lines


def _flag_table(product: Product) -> list[str]:
    """Class tables for the categorical variables, if any."""
    lines: list[str] = []
    for v in product.variables:
        if not v.categorical:
            continue
        lines += [
            "",
            f":::{{dropdown}} `{v.name}` classes ({len(v.flag_values)})",
            "",
            "| value | meaning |",
            "| --- | --- |",
        ]
        lines += [
            f"| {value} | {meaning} |"
            for value, meaning in zip(v.flag_values, v.flag_meanings, strict=False)
        ]
        lines += ["", ":::"]
    return lines


# ── pages ─────────────────────────────────────────────────────────────────────


#: AOIs the quickstart snippets use, for products that are not over Rainier.
_AOIS = {
    "nve-stations": ("(6.0, 60.0, 11.0, 63.0)", "southern Norway"),
}
_DEFAULT_AOI = ("(-121.94, 46.72, -121.54, 46.99)", "Mount Rainier")


def quickstart(product: Product) -> list[str]:
    """The smallest call that returns this product, read off its loader."""
    import inspect  # noqa: PLC0415 — only needed when rendering pages

    dotted = product.loader.replace("easysnowdata.", "esd.", 1)
    box, place = _AOIS.get(product.id, _DEFAULT_AOI)
    lines = ["import easysnowdata as esd", "", f"aoi = {box}   # {place}"]

    if product.id == "snow-station-archive":
        return lines + [
            "",
            "inv = esd.stations.archive.inventory(aoi)      # one HTTP request",
            "obs = esd.stations.archive.load(inv)           # SWE and snow depth",
        ]
    try:
        params = list(inspect.signature(product.resolve_loader()).parameters)
    except (ImportError, TypeError, ValueError):  # pragma: no cover
        params = ["aoi"]

    if params[:1] == ["stations"]:
        network = product.id.removesuffix("-stations")
        return lines + [
            "",
            f'inv = esd.stations.inventory(aoi, networks="{network}")',
            'obs = esd.stations.load(inv, variables=["swe", "snwd"],',
            '                        time="2023-10/2024-09")',
        ]
    second = params[1] if len(params) > 1 else ""
    extra = {"time": ', time="2024-03"', "level": ", level=8"}.get(second, "")
    return lines + [f"data = {dotted}(aoi{extra})"]


def _credentials_note(product: Product) -> list[str]:
    needed = sorted({req for s in product.sources for req in s.requires})
    if not needed:
        return [
            "",
            ":::{admonition} No account needed",
            ":class: tip",
            "",
            "Every route to this product is open. Nothing to set up.",
            ":::",
        ]
    free = product.credential_free_sources
    credentialed = [
        f'`source="{s.id}"` needs {", ".join(f"`{r}`" for r in s.requires)}'
        for s in product.sources
        if s.requires
    ]
    if product.default_source.credential_free:
        lines = [
            "",
            ":::{admonition} No account needed for the default route",
            ":class: tip",
            "",
            f'`source="{product.default_source.id}"` is open. '
            + "; ".join(credentialed)
            + " — see [the credentials page](../credentials.md).",
            ":::",
        ]
        return lines
    lines = [
        "",
        ":::{admonition} Credentials",
        ":class: note",
        "",
        "The default route needs "
        + ", ".join(f"`{name}`" for name in product.default_source.requires)
        + "; see [the credentials page](../credentials.md) for the setup.",
    ]
    if free:
        lines += [
            "",
            "An open alternative exists: "
            + ", ".join(f'`source="{s.id}"`' for s in free)
            + ".",
        ]
    lines.append(":::")
    return lines


def product_page(
    product: Product,
    *,
    latest: dict[str, dict[str, Any]] | None = None,
    gallery_dir: str = "gallery",
) -> str:
    """Render one product's Markdown page."""
    latest = latest or {}
    status = _product_status(product, latest)
    lines = [
        f"# {product.title}",
        "",
        f"`{product.id}` · theme `{product.theme}` · " + BADGES[status],
        "",
        product.description.strip(),
        "",
        "```python",
        *quickstart(product),
        "```",
    ]
    lines += _credentials_note(product)

    lines += ["", "## Sources", ""]
    if len(product.sources) > 1:
        lines += [
            f"{len(product.sources)} routes serve this product; the first is the "
            "default and `source=` picks another. The return contract is the "
            "same either way.",
            "",
        ]
    lines += sources_table(product)
    lines += ["", "### Where each route points", ""]
    for s in product.sources:
        badge = BADGES[_source_status(product, s, latest)]
        lines += [f"`{s.id}` — {s.title} {badge}", "", f": `{s.location}`", ""]

    variables = variables_table(product)
    if variables:
        lines += ["## Variables", ""] + variables
        lines += _flag_table(product)

    lines += ["", "## Provenance", ""]
    lines += [f"**Licence** · {product.license}", ""]
    if product.doi:
        lines += [f"**DOI** · [{product.doi}](https://doi.org/{product.doi})", ""]
    lines += ["**Cite the data as**", "", f"> {product.citation.strip()}", ""]
    if product.references:
        lines += ["**Documentation**", ""]
        lines += [f"- <{ref}>" for ref in product.references]
        lines += [""]
    lines += [f"**Loader** · {{py:obj}}`{product.loader}`", ""]

    if product.examples:
        # eval-rst, not a MyST directive: minigallery calls insert_input on the
        # state machine, which MyST's mock does not implement.
        lines += [
            "## Gallery examples",
            "",
            "```{eval-rst}",
            ".. minigallery::",
            "",
            *(f"    {gallery_dir}/{example}" for example in product.examples),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def _index_row(product: Product, latest: dict[str, dict[str, Any]]) -> str:
    default = product.default_source
    return (
        f"| [{product.title}]({product.id}.md) | `{product.id}` | "
        f"{len(product.sources)} | {', '.join(product.requires) or 'none'} | "
        f"{BADGES[_product_status(product, latest)]} |"
    )


def index_page(
    products: dict[str, Product] | None = None,
    *,
    latest: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the catalog index: every product, grouped by theme."""
    products = dict(products if products is not None else _registry.products())
    latest = latest or {}
    themes = sorted({p.theme for p in products.values()})
    lines = [
        "# Data catalog",
        "",
        f"{len(products)} products across {len(themes)} themes. Every page below is "
        "generated from the product's catalog entry, which is the same entry that "
        "drives `esd.catalog.describe()`, the weekly health probes and the README "
        "status table — so a page cannot drift from the code that serves the data.",
        "",
        "The badge is the most recent weekly probe of that product's best route: "
        f"{BADGES['pass']} at least one route answered, {BADGES['fail']} every route "
        f"failed, {BADGES['skip']} only credentialed routes exist and the probe run "
        "had no credentials.",
        "",
        "```python",
        "import easysnowdata as esd",
        "",
        'esd.catalog.list(theme="snow")        # the same table, as a DataFrame',
        'esd.catalog.search("swe")             # free-text search',
        'esd.catalog.describe("snodas")        # one product, as Markdown',
        "```",
        "",
    ]
    for theme in themes:
        in_theme = sorted(
            (p for p in products.values() if p.theme == theme), key=lambda p: p.id
        )
        lines += [
            f"## {theme}",
            "",
            "| product | id | routes | credentials | health |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [_index_row(p, latest) for p in in_theme]
        lines += [""]

    lines += [
        "```{toctree}",
        ":hidden:",
        ":glob:",
        "",
        "*",
        "```",
        "",
    ]
    return "\n".join(lines)


# ── the credentials page ──────────────────────────────────────────────────────


def _needed_by(products: dict[str, Product]) -> dict[str, list[Product]]:
    """Auth provider name -> the products with a source that needs it."""
    needed: dict[str, list[Product]] = {}
    for product in products.values():
        for source in product.sources:
            for req in source.requires:
                bucket = needed.setdefault(req, [])
                if product not in bucket:
                    bucket.append(product)
    return needed


def credentials_page(products: dict[str, Product] | None = None) -> str:
    """Render the credentials page from the auth provider registry (§5.2).

    Each provider states its own env vars, files, setup text and sign-up URL,
    and the catalog says which products need it — so this page is exactly the
    text ``CredentialError`` prints, with the product list attached.
    """
    from easysnowdata import auth  # noqa: PLC0415 — keeps the import graph acyclic

    products = dict(products if products is not None else _registry.products())
    needed = _needed_by(products)

    lines = [
        "# Credentials",
        "",
        "Most of the catalog is open: "
        f"{sum(1 for p in products.values() if p.credential_free_sources)} of "
        f"{len(products)} products have at least one route that needs no account. "
        "The rest need one of the providers below.",
        "",
        "Credentials are checked **before** the first network request, so a missing "
        "account raises `CredentialError` naming the provider, the setup steps and "
        "any credential-free alternative for that product — it never surfaces as a "
        "403 halfway through a download.",
        "",
        "```python",
        "import easysnowdata as esd",
        "",
        "esd.auth.status()          # which providers are configured, and how",
        "esd.auth.login()           # interactive setup for what is missing",
        'esd.auth.login("earthengine", project="my-gcp-project")',
        "```",
        "",
        "| provider | required? | environment variables | files it reads | products |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, provider in auth.PROVIDERS.items():
        count = len(needed.get(name, []))
        lines.append(
            f"| [{provider.title}](#{name.replace('_', '-')}) | "
            f"{'optional' if provider.optional else 'yes, for those products'} | "
            + (", ".join(f"`{v}`" for v in provider.env_vars) or "—")
            + " | "
            + (", ".join(f"`{f}`" for f in provider.files) or "—")
            + f" | {count} |"
        )
    lines += [
        "",
        "Nothing here is written to a configuration file of ours: every provider is "
        "set up the way that service documents, and read from the environment or "
        "from the file that service already uses (decision Q10).",
        "",
    ]

    for name, provider in auth.PROVIDERS.items():
        anchor = name.replace("_", "-")
        # A MyST target, so the summary table above can link to each section.
        lines += [f"({anchor})=", ""]
        lines += [f"## {provider.title}", ""]
        if provider.optional:
            lines += [
                ":::{admonition} Optional",
                ":class: tip",
                "",
                "Everything works anonymously; credentials only raise the rate limits.",
                ":::",
                "",
            ]
        if provider.setup_instructions:
            lines += ["```text", provider.setup_instructions.strip(), "```", ""]
        if provider.signup_url:
            lines += [f"Sign up: <{provider.signup_url}>", ""]

        users = needed.get(name, [])
        if users:
            lines += [
                "Products with a route that needs it:",
                "",
                "| product | route | credential-free alternative |",
                "| --- | --- | --- |",
            ]
            for product in sorted(users, key=lambda p: p.id):
                routes = ", ".join(
                    f"`{s.id}`" for s in product.sources if name in s.requires
                )
                free = (
                    ", ".join(f"`{s.id}`" for s in product.credential_free_sources)
                    or "— none, this provider is the only way in"
                )
                lines.append(
                    f"| [{product.title}](catalog/{product.id}.md) | {routes} | {free} |"
                )
            lines += [""]
        else:
            lines += ["No catalog product needs it today.", ""]

    lines += [
        "## In CI",
        "",
        "The repository's scheduled workflows read the same environment variables "
        "from GitHub secrets, which is why a credentialed gallery example or live "
        "test simply skips — rather than failing — when a secret is absent:",
        "",
        "| secret | used by |",
        "| --- | --- |",
        "| `EARTHDATA_TOKEN` (or `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`) | live tests, health probes, scheduled docs build |",
        "| `EARTHENGINE_TOKEN` | live tests, health probes, scheduled docs build |",
        "| `PL_API_KEY` | Planet live tests and the PlanetScope gallery example |",
        "| `NVE_API_KEY` | the NVE station live test and gallery example |",
        "",
        "Pull-request builds get none of them, by design: they run the offline test "
        "tiers and the credential-free subset of the gallery.",
        "",
    ]
    return "\n".join(lines)


# ── the status page ───────────────────────────────────────────────────────────


def sparkline(values: list[float | None], *, width: int = 110, height: int = 20) -> str:
    """An inline SVG line of *values*, oldest first, scaled to its own range.

    Inline rather than a rendered image: it uses ``currentColor``, so it is
    legible in both themes, it costs no build step, and it survives being
    regenerated on every docs build. Gaps (``None``) break the line.
    """
    known = [v for v in values if v is not None]
    if len(known) < 2:
        return ""
    low, high = min(known), max(known)
    span = (high - low) or 1.0
    step = width / max(len(values) - 1, 1)
    pad = 2
    segments: list[list[str]] = [[]]
    for i, value in enumerate(values):
        if value is None:
            segments.append([])
            continue
        x = i * step
        y = pad + (height - 2 * pad) * (1 - (value - low) / span)
        segments[-1].append(f"{x:.1f},{y:.1f}")
    paths = "".join(
        f'<polyline points="{" ".join(seg)}" fill="none" stroke="currentColor" '
        f'stroke-width="1.5" stroke-linejoin="round" />'
        for seg in segments
        if len(seg) > 1
    )
    if not paths:
        return ""
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="recent values, oldest first" '
        f'style="vertical-align:middle;opacity:0.8">{paths}</svg>'
    )


def _lag_days(value: str, now: Any = None) -> float | None:
    """Days between an ISO timestamp and *now*."""
    import pandas as pd  # noqa: PLC0415

    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):  # pragma: no cover — a non-time value
        return None
    reference = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    return round((reference - stamp).total_seconds() / 86400, 1)


def _series(history: list[list[dict[str, Any]]], label: str, key: str) -> list[Any]:
    """One probe's values across the history, oldest run first."""
    out = []
    for run in reversed(history):
        match = next((r for r in run if r["source"] == label), None)
        out.append(None if match is None else match.get(key))
    return out


def status_page(
    history: list[list[dict[str, Any]]],
    *,
    products: dict[str, Product] | None = None,
    now: Any = None,
) -> str:
    """Render the health, latency and virtualization status page (§8)."""
    products = dict(products if products is not None else _registry.products())
    latest = latest_status(history)
    run = history[0] if history else []
    checked = run[0].get("checked_at", "never") if run else "never"

    health = [r for r in run if r.get("kind", "health") == "health"]
    counts = {
        status: sum(1 for r in health if r["status"] == status)
        for status in ("pass", "fail", "skip")
    }

    lines = [
        "# Source status",
        "",
        f"Last run **{checked}**, {len(history)} runs on record. "
        f"{counts['pass']} routes answered, {counts['fail']} failed, and "
        f"{counts['skip']} "
        f"{'was' if counts['skip'] == 1 else 'were'} skipped for want of "
        "credentials.",
        "",
        "Every route of every product is probed once a week by the "
        "`Data Source Health Check` workflow. A failure opens an issue "
        "labelled [`data-source`](https://github.com/egagli/easysnowdata/issues?q=label%3Adata-source) "
        "and a recovery closes it, so this page is a summary and the tracker "
        "is the record.",
        "",
        "```bash",
        "pixi run -e dev python scripts/check_data_sources.py   # run them yourself",
        "```",
        "",
        "## Health",
        "",
        "| route | product | status | last good | recent | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in health:
        label = result["source"]
        product = result.get("product", "—")
        good = next(
            (
                str(r.get("checked_at", ""))[:10]
                for older in history
                for r in older
                if r["source"] == label and r["status"] == "pass"
            ),
            "—",
        )
        strip = "".join(
            {"pass": "▪", "fail": "▴", "skip": "▫"}.get(status or "", " ")
            for status in _series(history, label, "status")[-12:]
        )
        note = str(result.get("error") or "").replace("|", "\\|")[:90]
        lines.append(
            f"| {label} | `{product}` | {BADGES[result['status']]} | {good} | "
            f"`{strip}` | {note} |"
        )
    lines += [
        "",
        "`▪` answered · `▴` failed · `▫` skipped, oldest run on the left.",
        "",
    ]

    latency = [r for r in run if r.get("kind") == "latency"]
    lines += ["## Latency", ""]
    if latency:
        lines += [
            "How far behind real time each time-series route is, measured every "
            "week. A step change here is how a product that quietly stops being "
            "archived shows up — the MOD10A2 case (§8).",
            "",
            "| route | product | newest data | days behind | trend |",
            "| --- | --- | --- | --- | --- |",
        ]
        for result in sorted(latency, key=lambda r: r["source"]):
            value = str(result.get("value", "")) or "—"
            lag = _lag_days(value, now) if result.get("value") else None
            series = [
                _lag_days(v, now) if v else None
                for v in _series(history, result["source"], "value")[-12:]
            ]
            chart = sparkline(series)
            lines.append(
                f"| {result['source']} | `{result.get('product', '—')}` | {value} | "
                f"{'—' if lag is None else f'{lag:g}'} | "
                f"{chart or '_not enough runs yet_'} |"
            )
        lines += [
            "",
            "The trend line rises when a route falls further behind.",
            "",
        ]
    else:
        lines += [
            "_No latency probe has run yet. The next weekly check records one "
            "per time-series route._",
            "",
        ]

    virtual = [r for r in run if r.get("kind") == "virtualization"]
    lines += ["## Virtualization readiness", ""]
    if virtual:
        lines += [
            "Whether NASA publishes a DMR++ sidecar for each NetCDF/HDF product. "
            "With one, `earthaccess.virtualize()` reads a sidecar; without one it "
            "scans every granule's metadata, which is why the loaders cache their "
            "references. A sidecar appearing opens an issue (§4.9).",
            "",
            "| route | product | sidecar | fallback parser |",
            "| --- | --- | --- | --- |",
        ]
        for result in sorted(virtual, key=lambda r: r["source"]):
            value = str(result.get("value", "—"))
            state, _, fallback = value.partition("; fallback: ")
            lines.append(
                f"| {result['source']} | `{result.get('product', '—')}` | "
                f"{'{bdg-success}`present`' if state.startswith('present') else '{bdg-secondary}`absent`'} | "
                f"{fallback or '—'} |"
            )
        lines += [""]
    else:
        lines += [
            "_No virtualization probe has run yet._",
            "",
        ]

    lines += [
        "## Upstream watch",
        "",
        "A second weekly job reads `WATCHLIST.toml` — CMR collections, STAC "
        "collections, Earth Engine assets, static files, station counts, PyPI "
        "releases, and a dozen provider changelogs — diffs each against the "
        "snapshot in `data_status/watch/`, and opens one digest issue labelled "
        "[`upstream-watch`](https://github.com/egagli/easysnowdata/issues?q=label%3Aupstream-watch) "
        "with a section per category. If nothing changed, no issue is opened.",
        "",
        "```bash",
        "pixi run -e dev watch   # fetch and diff, writing nothing",
        "```",
        "",
    ]

    unprobed = sorted(
        pid
        for pid, product in products.items()
        if not any(
            latest.get(probe.label)
            for source in product.sources
            for probe in source.health
        )
    )
    if unprobed:
        lines += [
            "## Not yet in the history",
            "",
            "These products have probes that no recorded run has executed yet — "
            "they were added to the catalog after the most recent run:",
            "",
            *(f"- [`{pid}`](catalog/{pid}.md)" for pid in unprobed),
            "",
        ]
    return "\n".join(lines)


def write_all(
    out_dir: str | Path,
    *,
    products: dict[str, Product] | None = None,
    history: list[list[dict[str, Any]]] | None = None,
    gallery_dir: str = "gallery",
    credentials: str | Path | None = None,
    status: str | Path | None = None,
) -> list[Path]:
    """Write ``index.md`` and one page per product into *out_dir*.

    *credentials*, when given, is where the credentials page is written; it
    sits beside the catalog directory rather than inside it because it is not
    a product.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    products = dict(products if products is not None else _registry.products())
    latest = latest_status(history or [])
    written = []
    for product in products.values():
        path = out_dir / f"{product.id}.md"
        _write_if_changed(
            path, product_page(product, latest=latest, gallery_dir=gallery_dir)
        )
        written.append(path)
    index = out_dir / "index.md"
    _write_if_changed(index, index_page(products, latest=latest))
    written.append(index)
    if credentials is not None:
        _write_if_changed(Path(credentials), credentials_page(products))
        written.append(Path(credentials))
    if status is not None:
        _write_if_changed(Path(status), status_page(history or [], products=products))
        written.append(Path(status))
    return written


def _write_if_changed(path: Path, text: str) -> None:
    """Leave the file alone when the content matches, so Sphinx can cache it."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")
