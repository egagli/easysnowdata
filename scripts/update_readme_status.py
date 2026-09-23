"""Update the data-source status table in README.md from history.json.

Usage
-----
    python scripts/update_readme_status.py \\
        [--history data_status/history.json] \\
        [--readme README.md]

The script reads the last 4 weekly snapshots from *history.json* and
replaces the content between the sentinel comments in *readme*:

    <!-- DATA_STATUS_START -->
    ...table...
    <!-- DATA_STATUS_END -->
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

SENTINEL_START = "<!-- DATA_STATUS_START -->"
SENTINEL_END = "<!-- DATA_STATUS_END -->"
CATALOG_START = "<!-- CATALOG_START -->"
CATALOG_END = "<!-- CATALOG_END -->"
CATALOG_URL = "https://egagli.github.io/easysnowdata/catalog"

STATUS_EMOJI = {"pass": "✅", "fail": "❌", "skip": "⚠️"}


def _week_label(iso_ts: str) -> str:
    """Convert an ISO timestamp to a short 'Mon DD' label."""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%b %-d")
    except Exception:
        return iso_ts[:10]


def _health_rows(run: list[dict]) -> list[dict]:
    """Only the pass/fail probes. Latency and virtualization measure, not judge."""
    return [r for r in run if r.get("kind", "health") == "health"]


def _module_groups(run: list[dict]) -> list[tuple[str | None, list[str]]]:
    """Probe sources of *run* grouped under a module heading, in catalog order.

    A row names its product, and the catalog knows the product's module; a
    row without one (or for a product that has since left the catalog) goes
    under a final *Retired* heading. When no row names a product at all there
    is nothing to group by and a single unheaded table is returned.
    """
    from easysnowdata import catalog
    from easysnowdata.catalog._models import THEME_TITLES, theme_order

    products = catalog.products()
    order = {pid: i for i, pid in enumerate(products)}
    if not any(r.get("product") in products for r in run):
        return [(None, [r["source"] for r in run])]

    grouped: dict[str, list[dict]] = {}
    for row in run:
        product = products.get(str(row.get("product", "")))
        grouped.setdefault(product.theme if product else "retired", []).append(row)

    groups: list[tuple[str | None, list[str]]] = []
    for theme in sorted(grouped, key=theme_order):
        rows = sorted(
            grouped[theme],
            key=lambda r: (order.get(str(r.get("product", "")), 10**6), run.index(r)),
        )
        if theme in THEME_TITLES:
            heading = f"{THEME_TITLES[theme]} (`esd.{theme}`)"
        else:
            heading = theme.capitalize()
        groups.append((heading, [r["source"] for r in rows]))
    return groups


def build_table(history: list[list[dict]]) -> str:
    """Build the Markdown status tables from up to 4 weekly snapshots.

    One table per module, under a ``###`` heading with the module's name, so
    the README reads the way the package is organised.
    """
    weeks = [_health_rows(run) for run in history[:4]]
    weeks = [week for week in weeks if week]
    if not weeks:
        return "_No data yet — run `python scripts/check_data_sources.py` first._\n"

    # Column headers: Latest + up to 3 prior weeks
    col_headers = []
    for i, week in enumerate(weeks):
        ts = week[0]["checked_at"] if week else ""
        label = _week_label(ts)
        col_headers.append(f"Latest ({label})" if i == 0 else label)

    header = "| Data Source | " + " | ".join(col_headers) + " |"
    separator = "| :---------- | " + " | ".join([":------:"] * len(col_headers)) + " |"

    def row_for(source: str) -> str:
        cells = []
        for week in weeks:
            match = next((r for r in week if r["source"] == source), None)
            if match is None:
                cells.append("—")
            else:
                emoji = STATUS_EMOJI.get(match["status"], "?")
                if match["status"] == "fail" and match.get("error"):
                    # Truncate long errors in the tooltip, and escape them: an
                    # API error is often JSON, whose double quotes would end
                    # the title attribute early and break the cell.
                    tip = html.escape(match["error"][:80]).replace("|", "\\|")
                    cells.append(f'<abbr title="{tip}">{emoji}</abbr>')
                else:
                    cells.append(emoji)
        return f"| {source} | " + " | ".join(cells) + " |"

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"_Last updated: {now}_  ",
        "_⚠️ = skipped (credentials not available in this run). "
        "Latency and virtualization probes are on the "
        "[status page](https://egagli.github.io/easysnowdata/status.html)._",
    ]
    for heading, sources in _module_groups(weeks[0]):
        lines += [""]
        if heading is not None:
            lines += [f"### {heading}", ""]
        lines += [header, separator, *(row_for(s) for s in sources)]
    return "\n".join(lines) + "\n"


def build_catalog_table() -> str:
    """A theme-by-theme summary of the catalog, for the README.

    Generated rather than written, so a product added to the registry appears
    here without anyone remembering to edit the README.
    """
    from easysnowdata import catalog
    from easysnowdata.catalog._models import theme_order

    products = catalog.products()
    by_theme: dict[str, list] = {}
    for product in products.values():
        by_theme.setdefault(product.theme, []).append(product)

    lines = [
        f"{len(products)} products across {len(by_theme)} themes, each with one or "
        "more access routes:",
        "",
        "| theme | products | open without an account |",
        "| --- | --- | --- |",
    ]
    for theme in sorted(by_theme, key=theme_order):
        items = sorted(by_theme[theme], key=lambda p: p.id)
        free = sum(1 for p in items if p.credential_free_sources)
        names = ", ".join(f"[`{p.id}`]({CATALOG_URL}/{p.id}.html)" for p in items)
        lines.append(f"| **{theme}** | {names} | {free} of {len(items)} |")
    return "\n".join(lines) + "\n"


def _replace_block(text: str, start: str, end: str, body: str, path: Path) -> str:
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if not re.search(pattern, text):
        raise ValueError(
            f"Could not find {start} / {end} in {path}. Add the markers first."
        )
    return re.sub(pattern, lambda _m: f"{start}\n{body}{end}", text)


def update_readme(
    table: str, readme_path: Path, *, catalog_table: str | None = None
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    text = _replace_block(text, SENTINEL_START, SENTINEL_END, table, readme_path)
    if catalog_table is not None:
        text = _replace_block(
            text, CATALOG_START, CATALOG_END, catalog_table, readme_path
        )
    readme_path.write_text(text, encoding="utf-8")
    print(f"README updated: {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update README data-source status table."
    )
    parser.add_argument("--history", default="data_status/history.json")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="leave the catalog summary block alone",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    readme_path = Path(args.readme)

    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        print(f"History file not found: {history_path}; the status table is unchanged.")

    update_readme(
        build_table(history),
        readme_path,
        catalog_table=None if args.no_catalog else build_catalog_table(),
    )


if __name__ == "__main__":
    main()
