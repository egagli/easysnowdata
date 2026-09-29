"""Keep the last good output of a gallery example that failed to run.

A full docs build executes every gallery script against live providers. When a
provider is briefly down (Earthdata Login was unreachable for half an hour on
2026-09-28), the scripts that need it fail, and without this step the site
publishes their tracebacks in place of the figures that were there the day
before.

After the build, and before the site is deployed, this script:

1. finds every example page in the built site that shows a traceback (the
   ``sphx-glr-script-out highlight-pytb`` block sphinx-gallery writes for a
   failed script);
2. looks the same page up in the last deployed site (the ``gh-pages`` branch)
   and, if that copy ran, puts its article (code, printed output, and figures)
   back in place of the failed one, together with its figures and thumbnail;
3. says so at the top of the page: when the example last ran, when it failed,
   and the error, so a stale page is never mistaken for a fresh one;
4. stamps every page that did run with today's date, which is what step 3
   quotes the next time.

An example that has never run successfully keeps its traceback: there is
nothing to fall back to. An example that keeps failing keeps its last good
output and its note, whose "last ran" date shows how stale it is.

Usage (from the repository root, after ``sphinx-build``)::

    git fetch --depth 1 origin gh-pages:refs/remotes/origin/gh-pages
    python scripts/gallery_fallback.py --site docs/_build/html --summary "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

FAILED_MARKER = "sphx-glr-script-out highlight-pytb"
STAMP_RE = re.compile(r"<!-- esd-gallery-ran: (\d{4}-\d{2}-\d{2}) -->")
ARTICLE_RE = re.compile(r'<article class="bd-article"[^>]*>.*?</article>', re.S)
NOTE_CLASS = "esd-gallery-fallback"


@dataclass
class Outcome:
    page: str
    status: str  # "restored", "no-previous", "previous-failed"
    error: str
    last_ran: str = ""


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def previous_file(ref: str, path: str) -> bytes | None:
    """*path* as the last deploy has it, or ``None``."""
    try:
        return _git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def previous_listing(ref: str, prefix: str) -> list[str]:
    try:
        out = _git("ls-tree", "-r", "--name-only", ref, prefix).decode()
    except subprocess.CalledProcessError:
        return []
    return [line for line in out.splitlines() if line]


def ref_date(ref: str) -> str:
    try:
        return _git("log", "-1", "--format=%cs", ref).decode().strip()
    except subprocess.CalledProcessError:
        return ""


def error_line(page_html: str) -> str:
    """The last line of the page's traceback: the exception and its message."""
    start = page_html.find(FAILED_MARKER)
    block = page_html[start : page_html.find("</pre>", start)]
    text = html.unescape(re.sub(r"<[^>]+>", "", block))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else "unknown error"
    return last if len(last) <= 300 else last[:297] + "…"


def note(last_ran: str, failed_on: str, error: str) -> str:
    ran = f"on {last_ran}" if last_ran else "in an earlier build"
    return (
        f'<div class="admonition warning {NOTE_CLASS}">'
        '<p class="admonition-title">Output from an earlier build</p>'
        f"<p>This example could not run in the build of {failed_on}, so the code, "
        f"printed output, and figures below are from the last build in which it "
        f"ran, {ran}; the download buttons give the current script. The error was: "
        f"<code>{html.escape(error)}</code></p></div>"
    )


def _insert_after_article_open(article: str, snippet: str) -> str:
    end = article.index(">") + 1
    return article[:end] + snippet + article[end:]


def figure_stem(page: str) -> str:
    """``auto_examples/snow/plot_modis_snow.html`` → ``sphx_glr_plot_modis_snow_``."""
    return f"sphx_glr_{Path(page).stem}_"


def restore(site: Path, source: Path, ref: str, page: str, today: str) -> Outcome:
    current = (site / page).read_text(encoding="utf-8")
    error = error_line(current)
    old = previous_file(ref, page)
    if old is None:
        return Outcome(page, "no-previous", error)
    old_html = old.decode("utf-8")
    if FAILED_MARKER in old_html:
        return Outcome(page, "previous-failed", error)
    old_article = ARTICLE_RE.search(old_html)
    new_article = ARTICLE_RE.search(current)
    if old_article is None or new_article is None:
        return Outcome(page, "no-previous", error)

    stamp = STAMP_RE.search(old_html)
    last_ran = stamp.group(1) if stamp else ref_date(ref)
    # A page restored before already carries a note; replace it, not stack it.
    article = re.sub(
        rf'<div class="admonition warning {NOTE_CLASS}">.*?</p></div>',
        "",
        old_article.group(0),
        count=1,
        flags=re.S,
    )
    article = _insert_after_article_open(article, note(last_ran, today, error))
    page_html = current[: new_article.start()] + article + current[new_article.end() :]
    page_html = STAMP_RE.sub("", page_html)
    page_html = page_html.replace(
        "</body>", f"<!-- esd-gallery-ran: {last_ran} --></body>", 1
    )
    (site / page).write_text(page_html, encoding="utf-8")

    # The figures and the thumbnail, into the site and into the gallery source
    # tree (the README montage reads its thumbnails from there).
    theme = Path(page).parent.name
    images = source / theme / "images"
    figure_re = re.compile(
        rf"^{re.escape(figure_stem(page))}(\d{{3}}|thumb)\.(png|jpg|jpeg|svg|gif|webp)$"
    )
    for name in previous_listing(ref, "_images"):
        filename = Path(name).name
        if not figure_re.match(filename):
            continue
        data = previous_file(ref, name)
        if data is None:
            continue
        (site / "_images" / filename).write_bytes(data)
        target = (
            images / ("thumb" if filename.endswith("_thumb.png") else "") / filename
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return Outcome(page, "restored", error, last_ran)


def stamp_good(site: Path, pages: list[str], today: str) -> None:
    for page in pages:
        path = site / page
        text = path.read_text(encoding="utf-8")
        if FAILED_MARKER in text or NOTE_CLASS in text:
            continue
        text = STAMP_RE.sub("", text)
        path.write_text(
            text.replace("</body>", f"<!-- esd-gallery-ran: {today} --></body>", 1),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default="docs/_build/html", type=Path)
    parser.add_argument("--source", default="docs/auto_examples", type=Path)
    parser.add_argument("--ref", default="origin/gh-pages")
    parser.add_argument("--summary", help="append a Markdown report to this file")
    parser.add_argument("--today", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    args = parser.parse_args(argv)

    pages = sorted(
        str(path.relative_to(args.site))
        for path in (args.site / "auto_examples").glob("*/plot_*.html")
    )
    failed = [
        page
        for page in pages
        if FAILED_MARKER in (args.site / page).read_text(encoding="utf-8")
    ]
    outcomes = [
        restore(args.site, args.source, args.ref, p, args.today) for p in failed
    ]
    stamp_good(args.site, pages, args.today)

    lines = [
        f"### Gallery fallback ({len(failed)} of {len(pages)} examples failed)",
        "",
    ]
    for o in outcomes:
        what = {
            "restored": f"showing the output from {o.last_ran or 'an earlier build'}",
            "no-previous": "no earlier output to fall back to; the traceback stays",
            "previous-failed": "the last deploy failed too; the traceback stays",
        }[o.status]
        lines.append(f"- `{o.page}`: {what}. Error: `{o.error}`")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
