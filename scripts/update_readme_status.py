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


def build_table(history: list[list[dict]]) -> str:
    """Build a Markdown status table from up to 4 weekly snapshots."""
    weeks = [_health_rows(run) for run in history[:4]]
    weeks = [week for week in weeks if week]
    if not weeks:
        return "_No data yet — run `python scripts/check_data_sources.py` first._\n"

    # Collect all source names in order from the most recent run
    source_names = [r["source"] for r in weeks[0]]

    # Column headers: Latest + up to 3 prior weeks
    col_headers = []
    for i, week in enumerate(weeks):
        ts = week[0]["checked_at"] if week else ""
        label = _week_label(ts)
        col_headers.append(f"Latest ({label})" if i == 0 else label)

    header = "| Data Source | " + " | ".join(col_headers) + " |"
    separator = "| :---------- | " + " | ".join([":------:"] * len(col_headers)) + " |"

    rows = []
    for source in source_names:
        cells = []
        for week in weeks:
            match = next((r for r in week if r["source"] == source), None)
            if match is None:
                cells.append("—")
            else:
                emoji = STATUS_EMOJI.get(match["status"], "?")
                if match["status"] == "fail" and match.get("error"):
                    # Truncate long errors in the tooltip
                    tip = match["error"][:80].replace("|", "\\|")
                    cells.append(f'<abbr title="{tip}">{emoji}</abbr>')
                else:
                    cells.append(emoji)
        rows.append(f"| {source} | " + " | ".join(cells) + " |")

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    note = (
        f"_Last updated: {now}_  \n"
        "_⚠️ = skipped (credentials not available in this run). "
        "Latency and virtualization probes are on the "
        "[status page](https://egagli.github.io/easysnowdata/status.html)._\n"
    )

    return "\n".join([note, header, separator] + rows) + "\n"


def build_catalog_table() -> str:
    """A theme-by-theme summary of the catalog, for the README.

    Generated rather than written, so a product added to the registry appears
    here without anyone remembering to edit the README.
    """
    from easysnowdata import catalog  # noqa: PLC0415

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
    for theme in sorted(by_theme):
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
