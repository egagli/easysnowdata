"""Turn the weekly live-test run into GitHub issues, one per data source (§8).

The scheduled live tier failed four weeks running in September and October
2026 and nobody looked: a NASA CMR outage made the run red every week, so the
Yukon API breaking underneath it (29 failed tests on 2026-10-20) was one more
line in a log already known to be red. A single red run hides which provider
broke; one issue per provider does not.

What it does with one pytest JUnit XML report:

* every **data source with a failing or erroring live test** gets an issue
  labelled ``data-source`` (the label the health check uses), listing the
  failing tests grouped by error message. A source still failing a week later
  has its issue updated and a comment added, so it surfaces again;
* a source whose tests **pass again** has its issue closed. A source whose
  tests all skipped (no credentials) proves nothing and is left alone.

A test's data source is its file: ``tests/stations/test_yukon_client.py`` is
``yukon``, ``tests/timeseries/test_hls.py`` is ``hls``.

The planning half is pure: :func:`plan` takes the parsed report and returns
the actions, so the offline tests check it without GitHub. Applying reuses
``report_health``'s GitHub calls under its own marker.

Usage
-----
    python scripts/report_live.py live-results.xml [--repo owner/name]
                                  [--run-url URL] [--dry-run]

``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY`` come from the Actions environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "report_health", _HERE / "report_health.py"
)
report_health = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("report_health", report_health)
_spec.loader.exec_module(report_health)

Action = report_health.Action

#: Marker namespace, distinct from the health check's ``esd-health``.
MARKER_PREFIX = "esd-live"
MARKER = "<!-- " + MARKER_PREFIX + ": {key} -->"

REPRODUCE = "pixi run -e test-py313 pytest {path} -m live -v --tb=short"


@dataclass
class Outcome:
    """The live tests of one data source in one run."""

    source: str
    path: str
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (test, message)
    skipped: list[str] = field(default_factory=list)


def source_of(classname: str, file: str | None = None) -> tuple[str, str]:
    """Map a JUnit test case to ``(source, test file path)``."""
    if file:
        path = file
    else:
        parts = classname.split(".")
        module = next(
            (i for i, part in enumerate(parts) if part.startswith("test_")),
            len(parts) - 1,
        )
        path = "/".join(parts[: module + 1]) + ".py"
    stem = Path(path).stem.removeprefix("test_").removesuffix("_client")
    return stem, path


def _message(node: ET.Element) -> str:
    """The first line of a failure, compact enough for a table cell."""
    text = node.get("message") or (node.text or "").strip()
    first = text.splitlines()[0] if text else node.tag
    return first.strip()[:240]


def parse(xml_text: str) -> dict[str, Outcome]:
    """Read a pytest JUnit XML report into one :class:`Outcome` per source."""
    outcomes: dict[str, Outcome] = {}
    for case in ET.fromstring(xml_text).iter("testcase"):
        source, path = source_of(case.get("classname", ""), case.get("file"))
        outcome = outcomes.setdefault(source, Outcome(source, path))
        name = case.get("name", "?")
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is not None:
            outcome.failed.append((name, _message(bad)))
        elif case.find("skipped") is not None:
            outcome.skipped.append(name)
        else:
            outcome.passed.append(name)
    return outcomes


def _body(outcome: Outcome, run_url: str | None, checked: str) -> str:
    total = len(outcome.failed) + len(outcome.passed)
    lines = [
        MARKER.format(key=outcome.source),
        "",
        f"The weekly live tests failed for **`{outcome.source}`**: "
        f"{len(outcome.failed)} of {total} run tests in `{outcome.path}`"
        + (f" ({len(outcome.skipped)} skipped)." if outcome.skipped else "."),
        "",
        "| tests | error |",
        "| --- | --- |",
    ]
    for message, count in Counter(m for _, m in outcome.failed).most_common():
        cell = message.replace("|", "\\|")
        lines.append(f"| {count} | `{cell}` |")
    lines += ["", "<details><summary>Failing tests</summary>", ""]
    lines += [f"- `{name}`" for name, _ in outcome.failed]
    lines += [
        "",
        "</details>",
        "",
        f"Run of {checked}"
        + (f": {run_url}" if run_url else "")
        + ". Each weekly run updates this issue and closes it when the tests "
        "pass again. An error naming the provider's own server (HTTP 5xx, an "
        "outage page) is usually theirs; an assertion or a parse error is "
        "usually a change in what the provider returns.",
        "",
        "Reproduce locally:",
        "",
        "```bash",
        REPRODUCE.format(path=outcome.path),
        "```",
    ]
    return "\n".join(lines)


def plan(
    outcomes: dict[str, Outcome], *, run_url: str | None = None, checked: str = ""
) -> list[Action]:
    """Work out what this run should do to the tracker."""
    actions: list[Action] = []
    for source, outcome in sorted(outcomes.items()):
        if outcome.failed:
            actions.append(
                Action(
                    kind="open",
                    key=source,
                    title=(
                        f"Live tests failing: {source} "
                        f"({len(outcome.failed)} test"
                        f"{'s' if len(outcome.failed) != 1 else ''})"
                    ),
                    body=_body(outcome, run_url, checked),
                    comment=f"Still failing in the run of {checked}"
                    + (f": {run_url}" if run_url else "."),
                )
            )
        elif outcome.passed:
            actions.append(
                Action(
                    kind="close",
                    key=source,
                    comment=(
                        f"Recovered: {len(outcome.passed)} live test"
                        f"{'s' if len(outcome.passed) != 1 else ''} passed in "
                        f"the run of {checked}. Closing; the weekly run reopens "
                        "this if it breaks again."
                    ),
                )
            )
    return actions


def summary(outcomes: dict[str, Outcome]) -> str:
    """A Markdown table for the job summary: failing sources first."""
    rows = sorted(outcomes.values(), key=lambda o: (not o.failed, o.source))
    lines = [
        "### Live tests by data source",
        "",
        "| source | failed | passed | skipped | most common error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in rows:
        top = Counter(m for _, m in o.failed).most_common(1)
        error = f"`{top[0][0][:120]}`".replace("|", "\\|") if top else ""
        mark = "🔴 " if o.failed else ""
        lines.append(
            f"| {mark}{o.source} | {len(o.failed)} | {len(o.passed)} | "
            f"{len(o.skipped)} | {error} |"
        )
    return "\n".join(lines)


def apply(actions: list[Action], github, *, log=print) -> list[str]:
    """Carry out *actions*; unlike the health check, a repeat open also comments."""
    github.ensure_label()
    existing = {}
    for issue in github.open_issues():
        body = issue.get("body") or ""
        tag = f"<!-- {MARKER_PREFIX}:"
        if tag in body:
            existing[body.split(tag, 1)[1].split("-->", 1)[0].strip()] = issue

    done = []
    for action in actions:
        issue = existing.get(action.key)
        if action.kind == "open" and issue is None:
            created = github.create(action)
            done.append(f"opened #{created['number']}: {action.title}")
        elif action.kind == "open":
            github.update(issue["number"], title=action.title, body=action.body)
            github.comment(issue["number"], action.comment)
            done.append(f"updated #{issue['number']}: {action.title}")
        elif action.kind == "close" and issue is not None:
            github.comment(issue["number"], action.comment)
            github.update(issue["number"], state="closed", state_reason="completed")
            done.append(f"closed #{issue['number']}: live tests for {action.key}")
    for line in done:
        log(line)
    if not done:
        log("No issue changes.")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", help="pytest --junitxml output")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-url", default=None)
    parser.add_argument("--checked", default="", help="date of the run, for the text")
    parser.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="append the per-source table here (the Actions job summary)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the actions, change nothing"
    )
    args = parser.parse_args(argv)

    path = Path(args.report)
    if not path.exists():
        print(f"No report at {path}; nothing to report.")
        return 0
    outcomes = parse(path.read_text(encoding="utf-8"))
    table = summary(outcomes)
    print(table)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(table + "\n")

    actions = plan(outcomes, run_url=args.run_url, checked=args.checked)
    if args.dry_run or not os.environ.get("GITHUB_TOKEN"):
        for action in actions:
            if action.kind == "open":
                print(f"[open]  {action.title}")
        closes = sum(a.kind == "close" for a in actions)
        print(f"[close] {closes} passing sources (closes any open issue)")
        if not args.dry_run:
            print("(GITHUB_TOKEN is not set, so nothing was sent.)")
        return 0
    if not args.repo:
        print("--repo or GITHUB_REPOSITORY is required.", file=sys.stderr)
        return 2
    apply(actions, report_health.GitHub(args.repo, os.environ["GITHUB_TOKEN"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
