"""Turn the weekly health run into GitHub issues (§8).

A red row in the README that nobody opens an issue about is a row nobody
fixes: the GRDC/WMO basins URL answered 404 every week from at least June 2026
and the only record was a table cell. This script closes that loop.

What it does with one run of ``data_status/history.json``:

* a **failing health probe** opens an issue labelled ``data-source``, one per
  product, listing every failing route, the error, and when the route last
  worked. A second consecutive failure escalates the title;
* a **recovery** closes that issue with a comment saying what came back — a
  close for a product that never had an issue is a no-op;
* a **virtualization probe that flips from absent to present** opens its own
  issue — a DMR++ sidecar appearing is the signal to change how a loader
  reads that product (§4.9), and nothing else would notice.

The planning half is pure: :func:`plan` takes the history and returns the
actions, so the offline tests can check the escalation, the recovery and the
"do nothing" cases without touching the network. Only :func:`apply` talks to
GitHub.

Usage
-----
    python scripts/report_health.py [--history data_status/history.json]
                                    [--repo owner/name] [--dry-run]

``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY`` come from the Actions environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LABEL = "data-source"
LABEL_COLOR = "d93f0b"
LABEL_DESCRIPTION = "A data provider changed, broke, or came back"

#: Hidden in the issue body so a later run finds the issue it opened.
MARKER = "<!-- esd-health: {key} -->"

CATALOG_URL = "https://egagli.github.io/easysnowdata/catalog/{product}.html"


@dataclass
class Action:
    """One thing to do on GitHub."""

    kind: str  # "open" | "update" | "close"
    key: str  # the marker key: "<product>" or "virtualization/<product>"
    title: str = ""
    body: str = ""
    comment: str = ""
    labels: list[str] = field(default_factory=lambda: [LABEL])

    @property
    def marker(self) -> str:
        return MARKER.format(key=self.key)


# ── reading the history ───────────────────────────────────────────────────────


def _kind(result: dict[str, Any]) -> str:
    return result.get("kind", "health")


def consecutive_failures(history: list[list[dict]], label: str) -> int:
    """How many runs in a row, counting back from the newest, this probe failed."""
    count = 0
    for run in history:
        match = next((r for r in run if r["source"] == label), None)
        if match is None or match["status"] != "fail":
            break
        count += 1
    return count


def last_good(history: list[list[dict]], label: str) -> str | None:
    """The timestamp of the most recent run in which this probe passed."""
    for run in history:
        match = next((r for r in run if r["source"] == label), None)
        if match is not None and match["status"] == "pass":
            return str(match.get("checked_at", ""))[:10]
    return None


def previous_value(history: list[list[dict]], label: str) -> str | None:
    """The value this probe reported the run before last."""
    for run in history[1:]:
        match = next((r for r in run if r["source"] == label), None)
        if match is not None and "value" in match:
            return str(match["value"])
    return None


# ── planning ──────────────────────────────────────────────────────────────────


def _health_body(product: str, failures: list[dict], history: list[list[dict]]) -> str:
    lines = [
        MARKER.format(key=product),
        "",
        f"The weekly data-source health check could not reach "
        f"{'every route' if len(failures) > 1 else 'a route'} of "
        f"[`{product}`]({CATALOG_URL.format(product=product)}).",
        "",
        "| route | probe | failing for | last good | error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in failures:
        label = result["source"]
        weeks = consecutive_failures(history, label)
        good = last_good(history, label) or "not in the recorded history"
        error = str(result.get("error", "")).replace("|", "\\|")[:200]
        lines.append(
            f"| `{result.get('route', '?')}` | {label} | "
            f"{weeks} run{'s' if weeks != 1 else ''} | {good} | `{error}` |"
        )
    lines += [
        "",
        f"Checked at {failures[0].get('checked_at', 'unknown')}. This issue is "
        "updated by each weekly run and closed automatically when the route "
        "answers again.",
        "",
        "Reproduce locally:",
        "",
        "```bash",
        f"pixi run -e dev python scripts/check_data_sources.py --product {product} --no-history",
        "```",
    ]
    return "\n".join(lines)


def _virtualization_body(product: str, result: dict, was: str | None) -> str:
    return "\n".join(
        [
            MARKER.format(key=f"virtualization/{product}"),
            "",
            f"A DMR++ sidecar now exists for "
            f"[`{product}`]({CATALOG_URL.format(product=product)}).",
            "",
            f"- probe: {result['source']}",
            f"- was: `{was}`",
            f"- now: `{result.get('value')}`",
            "",
            "That is the fast path for `earthaccess.virtualize()`: reading a "
            "sidecar instead of scanning every granule's HDF5 metadata "
            "(REVAMP_PLAN §4.9). Worth revisiting the loader's `virtualize=` "
            "default and the threshold at which it switches on.",
        ]
    )


def plan(history: list[list[dict]]) -> list[Action]:
    """Work out what the newest run in *history* should do to the tracker."""
    if not history or not history[0]:
        return []
    latest = history[0]
    actions: list[Action] = []

    by_product: dict[str, list[dict]] = {}
    for result in latest:
        if _kind(result) != "health":
            continue
        by_product.setdefault(result.get("product") or result["source"], []).append(
            result
        )

    for product, results in by_product.items():
        failures = [r for r in results if r["status"] == "fail"]
        if failures:
            weeks = max(consecutive_failures(history, r["source"]) for r in failures)
            prefix = "🔴 " if weeks >= 2 else ""
            suffix = f" ({weeks} runs)" if weeks >= 2 else ""
            actions.append(
                Action(
                    kind="open",
                    key=product,
                    title=f"{prefix}Data source unreachable: {product}{suffix}",
                    body=_health_body(product, failures, history),
                )
            )
        elif any(r["status"] == "pass" for r in results):
            # Only claim a recovery when something actually answered; a run
            # where every probe skipped for want of credentials proves nothing.
            good = [r for r in results if r["status"] == "pass"]
            actions.append(
                Action(
                    kind="close",
                    key=product,
                    comment=(
                        f"Recovered: {', '.join(sorted(r['source'] for r in good))} "
                        f"answered at {good[0].get('checked_at', 'unknown')}. "
                        "Closing; the weekly check will reopen this if it breaks again."
                    ),
                )
            )

    for result in latest:
        if _kind(result) != "virtualization" or result["status"] != "pass":
            continue
        value = str(result.get("value", ""))
        was = previous_value(history, result["source"])
        if value.startswith("present") and (
            was is None or not was.startswith("present")
        ):
            product = result.get("product") or result["source"]
            actions.append(
                Action(
                    kind="open",
                    key=f"virtualization/{product}",
                    title=f"Fast virtualization path now available for {product}",
                    body=_virtualization_body(product, result, was),
                )
            )
    return actions


# ── applying ──────────────────────────────────────────────────────────────────


class GitHub:
    """The four REST calls this script needs."""

    def __init__(self, repo: str, token: str, session: Any = None) -> None:
        import requests  # noqa: PLC0415

        self.repo = repo
        self.api = f"https://api.github.com/repos/{repo}"
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method, f"{self.api}{path}", timeout=30, **kwargs
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def ensure_label(self) -> None:
        try:
            self._json(
                "POST",
                "/labels",
                json={
                    "name": LABEL,
                    "color": LABEL_COLOR,
                    "description": LABEL_DESCRIPTION,
                },
            )
        except Exception:  # noqa: BLE001 — 422 means it already exists
            pass

    def open_issues(self) -> list[dict]:
        return self._json(
            "GET", "/issues", params={"state": "open", "labels": LABEL, "per_page": 100}
        )

    def create(self, action: Action) -> dict:
        return self._json(
            "POST",
            "/issues",
            json={"title": action.title, "body": action.body, "labels": action.labels},
        )

    def update(self, number: int, **fields: Any) -> dict:
        return self._json("PATCH", f"/issues/{number}", json=fields)

    def comment(self, number: int, body: str) -> dict:
        return self._json("POST", f"/issues/{number}/comments", json={"body": body})


def apply(actions: list[Action], github: GitHub, *, log: Any = print) -> list[str]:
    """Carry out *actions*, returning one line of description per change."""
    github.ensure_label()
    existing = {}
    for issue in github.open_issues():
        body = issue.get("body") or ""
        if "<!-- esd-health:" in body:
            key = body.split("<!-- esd-health:", 1)[1].split("-->", 1)[0]
            existing[key.strip()] = issue

    done = []
    for action in actions:
        issue = existing.get(action.key)
        if action.kind == "open" and issue is None:
            created = github.create(action)
            done.append(f"opened #{created['number']}: {action.title}")
        elif action.kind == "open":
            github.update(issue["number"], title=action.title, body=action.body)
            done.append(f"updated #{issue['number']}: {action.title}")
        elif action.kind == "close" and issue is not None:
            github.comment(issue["number"], action.comment)
            github.update(issue["number"], state="closed", state_reason="completed")
            done.append(f"closed #{issue['number']}: {action.key}")
    for line in done:
        log(line)
    if not done:
        log("No issue changes.")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history", default="data_status/history.json")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--dry-run", action="store_true", help="print the actions, change nothing"
    )
    args = parser.parse_args(argv)

    path = Path(args.history)
    if not path.exists():
        print(f"No history at {path}; nothing to report.")
        return 0
    actions = plan(json.loads(path.read_text()))

    if args.dry_run or not os.environ.get("GITHUB_TOKEN"):
        opens = [a for a in actions if a.kind == "open"]
        closes = [a for a in actions if a.kind == "close"]
        for action in opens:
            print(f"[open]  {action.key}: {action.title}")
        # A close is a no-op unless an issue with that marker is open, so the
        # healthy majority is a count rather than a wall of text.
        print(f"[close] {len(closes)} healthy products (closes any open issue)")
        if not opens:
            print("Nothing failing.")
        if not args.dry_run:
            print("(GITHUB_TOKEN is not set, so nothing was sent.)")
        return 0
    if not args.repo:
        print("--repo or GITHUB_REPOSITORY is required.", file=sys.stderr)
        return 2
    apply(actions, GitHub(args.repo, os.environ["GITHUB_TOKEN"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
