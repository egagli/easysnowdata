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
from datetime import datetime, timezone
from pathlib import Path

SENTINEL_START = "<!-- DATA_STATUS_START -->"
SENTINEL_END = "<!-- DATA_STATUS_END -->"

STATUS_EMOJI = {"pass": "✅", "fail": "❌", "skip": "⚠️"}


def _week_label(iso_ts: str) -> str:
    """Convert an ISO timestamp to a short 'Mon DD' label."""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%b %-d")
    except Exception:
        return iso_ts[:10]


def build_table(history: list[list[dict]]) -> str:
    """Build a Markdown status table from up to 4 weekly snapshots."""
    weeks = history[:4]
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

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = f"_Last updated: {now}_  \n_⚠️ = skipped (credentials not available in this run)_\n"

    return "\n".join([note, header, separator] + rows) + "\n"


def update_readme(table: str, readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")

    pattern = re.compile(
        rf"{re.escape(SENTINEL_START)}.*?{re.escape(SENTINEL_END)}",
        re.DOTALL,
    )
    replacement = f"{SENTINEL_START}\n{table}{SENTINEL_END}"

    if not re.search(pattern, text):
        raise ValueError(
            f"Could not find sentinel comments in {readme_path}. "
            f"Add '{SENTINEL_START}' and '{SENTINEL_END}' markers to the README."
        )

    new_text = re.sub(pattern, replacement, text)
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"README updated: {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update README data-source status table.")
    parser.add_argument("--history", default="data_status/history.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    history_path = Path(args.history)
    readme_path = Path(args.readme)

    if not history_path.exists():
        print(f"History file not found: {history_path}. Nothing to do.")
        return

    with open(history_path) as f:
        history = json.load(f)

    table = build_table(history)
    update_readme(table, readme_path)


if __name__ == "__main__":
    main()
