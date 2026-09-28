"""The upstream-change watch: fetch, diff, classify, digest (REVAMP_PLAN §8).

The ask this answers is "check forums, changelogs and release notes
periodically". Once a week a GitHub Action runs this script; it fetches every
entry in ``WATCHLIST.toml``, compares it with the snapshot the previous run
left in ``data_status/watch/``, tags what changed, and opens **one** digest
issue. Reading the digest and acting on it is a human (or ad-hoc agent) job:
this script never edits code. The one thing it closes is the previous digest,
once the new one is open, carrying its unticked to-dos forward.

Two halves, deliberately:

*Programmatic checks* have no text in them. A CMR collection's ``version_id``
or ``revision_date``, a STAC collection's presence and newest item, an Earth
Engine asset's date range, a static file's ETag, a network's station count, a
package's version on PyPI. These catch a change *before* any changelog
mentions it, and they never produce noise: either the value moved or it did
not.

*Pages and feeds* are fetched, reduced to visible text, and diffed line by
line. Every new line is tagged against the §8 categories, and anything that
matches none is listed under "other" rather than dropped.

Usage
-----
    python scripts/watch.py                    # fetch, diff, write snapshots, print
    python scripts/watch.py --dry-run          # fetch and diff, write nothing
    python scripts/watch.py --only cmr stac    # just those kinds
    python scripts/watch.py --issue            # also open the digest issue

``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY`` come from the Actions environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TIMEOUT = 45
SNAPSHOT_DIR = Path("data_status/watch")
WATCHLIST = Path("WATCHLIST.toml")
LABEL = "upstream-watch"
USER_AGENT = "easysnowdata-watch (+https://github.com/egagli/easysnowdata)"

#: The §8 tag list. Order matters: the first match names the digest section.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "removal": (
        "decommission",
        "retire",
        "end of life",
        "no longer",
        "archived",
        "sunset",
        "withdraw",
    ),
    "deprecation": ("deprecat", "superseded", "migrate", "moved to", "replaced by"),
    "reprocessing": (
        "reprocess",
        "collection 7",
        "new version",
        "version ",
        "baseline",
        "v2.0",
    ),
    "extent": (
        "extended",
        "now available for",
        "added years",
        "expanded",
        "new region",
        "coverage",
    ),
    "stations": ("station", "snotel", "gauge", "site added", "site removed"),
    # "Worth adding" means worth adding *here*: a line about snow, ice or
    # glaciers. Sensor and product names (MODIS, HLS, ERA5, Sentinel) used to
    # be on this list, so every ERA5 end date that moved and every new MODIS
    # aerosol product landed under "Worth adding" (#51).
    "new-dataset": (
        "snow",
        "swe",
        "glacier",
        "cryosphere",
        "albedo",
        "permafrost",
        "frozen ground",
        "freeze",
        "sea ice",
        "ice sheet",
        "avalanche",
    ),
    "outage": (
        "outage",
        "maintenance",
        "credential",
        "token",
        "urs",
        "s3",
        "cloudfront",
        "downtime",
        "unavailable",
        "unreachable",
    ),
    # Set from what moved, never from words (see FIELD_CATEGORY).
    "stalled": (),
    "license": (),
    "republished": (),
    "routine": (),
}

#: Which digest section each category lands in.
SECTIONS: dict[str, tuple[str, ...]] = {
    "Action needed": (
        "removal",
        "deprecation",
        "reprocessing",
        "outage",
        "stalled",
        "license",
    ),
    "Worth adding": ("new-dataset",),
    "Changed": ("extent", "stations", "republished"),
    "Dependency releases": ("dependency",),
    "Routine": ("routine",),
}

#: Sections whose items are to-dos: rendered as checkboxes, never capped, and
#: carried into the next digest until someone ticks them off.
TODO_SECTIONS = ("Action needed", "Worth adding")

#: Sections folded into a <details> block: listed, never dropped, but out of
#: the way of the ones above.
FOLDED_SECTIONS = ("Routine", "Other")

#: Fields that advance by themselves on a live, rolling dataset. Moving
#: forward is the dataset working; the signal is when one moves backward or
#: gains an end, and a *stall* shows up as ``window_days`` growing (the GEE
#: check had to look further back to find the newest image).
ROLLING_FIELDS = ("temporal_end", "end", "updated")

#: What a field moving means, whatever its value says.
FIELD_CATEGORY = {
    "version": "reprocessing",
    "version_id": "reprocessing",
    "revision_date": "reprocessing",
    "latest_version": "reprocessing",
    "temporal_start": "extent",
    "count": "extent",
    "granules": "extent",
    "images": "extent",
    "features": "extent",
    "bands": "extent",
    "active": "stations",
    "daily": "stations",
    "daily_or_better": "stations",
    "etag": "republished",
    "last_modified": "republished",
    "length": "republished",
    "license": "license",
}

#: Lines no page diff should report: text that changes on every visit.
DEFAULT_IGNORE = (
    r"date accessed",  # NSIDC's citation block stamps today's date
    r"\baccessed (on )?\d",
    r"^(©|copyright)",
    r"last (updated|modified|reviewed)",
)

#: Kinds whose changes always belong to one section, whatever the words say.
#: A new xarray release is not a "new dataset" and a yanked package is not a
#: data removal, so the keyword tagging is overridden for these.
FORCED_CATEGORY = {"pypi": "dependency"}

#: At most this many items per section, so one chatty page cannot bury the rest.
SECTION_LIMIT = 25

CHANGELOGS = {
    "xarray": "https://docs.xarray.dev/en/stable/whats-new.html",
    "zarr": "https://zarr.readthedocs.io/en/stable/release-notes.html",
    "rasterio": "https://rasterio.readthedocs.io/en/stable/",
    "earthaccess": "https://github.com/nsidc/earthaccess/releases",
    "asf-search": "https://github.com/asfadmin/Discovery-asf_search/blob/master/CHANGELOG.md",
    "icechunk": "https://github.com/earth-mover/icechunk/releases",
    "virtualizarr": "https://virtualizarr.readthedocs.io/en/latest/releases.html",
}


# ── the change record ─────────────────────────────────────────────────────────


@dataclass
class Change:
    """One thing that moved since the previous run."""

    entry: str  # the watchlist entry id
    subject: str  # what moved: a collection id, a package, a page
    detail: str  # old -> new, or the new line of text
    url: str = ""
    products: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        for section, names in SECTIONS.items():
            if any(name in self.categories for name in names):
                return section
        return "Other"


def _keyword_re(word: str) -> re.Pattern[str]:
    """Match *word* at a word start, and also at a word end when it is 1-3 letters.

    Plain substring matching tagged "courses" as an outage (``urs``) and
    "academic" as a DEM. Longer keywords stay open-ended on purpose, so
    ``deprecat`` still matches "deprecated" and "deprecation".
    """
    tail = r"\b" if len(word) <= 3 else ""
    return re.compile(r"(?<![a-z0-9])" + re.escape(word) + tail, re.I)


_CATEGORY_RES = {
    name: [_keyword_re(word) for word in words] for name, words in CATEGORIES.items()
}


def classify(text: str, extra_keywords: list[str] | None = None) -> list[str]:
    """Tag a line of text against the §8 categories.

    ``"action"`` marks a line that hit one of the entry's own keywords: it is
    about something this package uses, so its other tags stand (see
    :func:`diff_lines`).
    """
    tags = [
        name
        for name, patterns in _CATEGORY_RES.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
    if extra_keywords and any(_keyword_re(w).search(text) for w in extra_keywords):
        tags.append("action")
    return tags


def products_for(entry: dict[str, Any], text: str) -> list[str]:
    """The catalog products a change touches.

    ``match_products`` maps a key (an asset id, a collection id, a product
    name as upstream spells it) to products; a subject or a line that
    contains one of the keys, matched as :func:`classify` matches keywords,
    touches those. Without a match, the entry's own
    ``products`` apply. An entry that covers several unrelated datasets (the
    Earth Engine assets, a DAAC's news page) lists them in ``match_products``
    only, so a change names the product it is about, not all of them.
    """
    found: list[str] = []
    for key, products in (entry.get("match_products") or {}).items():
        if _keyword_re(key).search(text):
            found += [p for p in products if p not in found]
    return found or list(entry.get("products", []))


def _ignored(entry: dict[str, Any], line: str) -> bool:
    patterns = (*DEFAULT_IGNORE, *entry.get("ignore", []))
    return any(re.search(pattern, line, re.I) for pattern in patterns)


def _field_categories(key: str, old: Any, new: Any) -> list[str]:
    """What one field moving from *old* to *new* means."""
    if key in ROLLING_FIELDS:
        was, now = str(old or ""), str(new or "")
        if was == "ongoing" and now not in ("", "ongoing"):
            return ["removal"]  # an open-ended collection gained an end date
        if now == "ongoing":
            return ["extent"]  # it was closed and is live again
        if was and now and now < was:
            return ["removal"]  # the newest data moved back: withdrawn
        return ["routine"]
    if key == "window_days":
        # The GEE check widens its window until it finds an image; needing a
        # wider one than last week means nothing new arrived.
        try:
            return ["stalled"] if int(new) > int(old) else ["routine"]
        except (TypeError, ValueError):
            return ["stalled"]
    if key == "status":
        ok = ("200", "206")
        if str(new) not in ok:
            return ["removal"]
        # A watched next-release URL (next year's Census file) going from 404
        # to 206 is the signal that the new release is out.
        return ["routine"] if str(old) in ok else ["new-dataset"]
    if key in FIELD_CATEGORY:
        return [FIELD_CATEGORY[key]]
    # Anything else (an Earth Engine asset's `state`, `type`): only its new
    # value can say what happened. The subject name is not classified: it is
    # always a product name, which used to make every change look new.
    return classify(str(new))


# ── snapshots ─────────────────────────────────────────────────────────────────


def read_snapshot(directory: Path, entry_id: str) -> dict[str, Any]:
    path = directory / f"{entry_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:  # pragma: no cover — a truncated write
        return {}


def write_snapshot(directory: Path, entry_id: str, data: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{entry_id}.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def diff_mapping(
    entry: dict[str, Any], before: dict[str, Any], after: dict[str, Any]
) -> list[Change]:
    """Compare two ``{subject: {field: value}}`` snapshots.

    A subject that appears is a new dataset; one that disappears is a removal;
    a field that moves is reported with its old and new value.
    """
    changes: list[Change] = []
    entry_id = entry["id"]

    for subject in sorted(set(after) - set(before)):
        if not before:
            continue  # the first run records, it does not report
        changes.append(
            Change(
                entry_id,
                subject,
                "appeared",
                url=str(after[subject].get("url", "")),
                products=products_for(entry, subject),
                categories=["new-dataset"],
            )
        )
    for subject in sorted(set(before) - set(after)):
        changes.append(
            Change(
                entry_id,
                subject,
                "disappeared",
                products=products_for(entry, subject),
                categories=["removal"],
            )
        )
    for subject in sorted(set(before) & set(after)):
        old, new = before[subject], after[subject]
        for key in sorted(set(old) | set(new)):
            if key == "url" or old.get(key) == new.get(key):
                continue
            detail = f"{key}: `{old.get(key)}` → `{new.get(key)}`"
            categories = _field_categories(key, old.get(key), new.get(key))
            changes.append(
                Change(
                    entry_id,
                    subject,
                    detail,
                    url=str(new.get("url", "")),
                    products=products_for(entry, subject),
                    categories=sorted(set(categories)),
                )
            )
    return changes


def diff_lines(
    entry: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    limit: int = 12,
) -> list[Change]:
    """Compare two page snapshots line by line, reporting what is new."""
    if not before.get("lines"):
        return []  # first run: record only
    old = set(before["lines"])
    new_lines = [
        line for line in after["lines"] if line not in old and not _ignored(entry, line)
    ]
    keywords = list(entry.get("keywords", []))
    scoped = bool(entry.get("match_products"))
    changes = []
    for line in new_lines[:limit]:
        categories = classify(line, keywords)
        products = products_for(entry, line)
        if scoped and not products and "action" not in categories:
            # A news line about a dataset this package does not use: whatever
            # it says ("Version 3 released", "maintenance"), it is not an
            # action for us. A snow term still makes it worth a look.
            categories = [c for c in categories if c == "new-dataset"]
        changes.append(
            Change(
                entry["id"],
                entry.get("url", entry["id"]),
                line,
                url=str(entry.get("url", "")),
                products=products,
                categories=categories,
            )
        )
    if len(new_lines) > limit:
        changes.append(
            Change(
                entry["id"],
                entry.get("url", entry["id"]),
                f"...and {len(new_lines) - limit} more new lines",
                url=str(entry.get("url", "")),
                products=list(entry.get("products", [])),
            )
        )
    return changes


# ── the checks ────────────────────────────────────────────────────────────────


def _session() -> Any:
    import requests  # noqa: PLC0415

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def check_cmr(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    """Version, revision date, granule count and temporal extent of a collection."""
    response = session.get(
        "https://cmr.earthdata.nasa.gov/search/collections.umm_json",
        params={"short_name": entry["short_name"], "page_size": 5},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    out: dict[str, Any] = {}
    for item in response.json().get("items", []):
        meta, umm = item.get("meta", {}), item.get("umm", {})
        concept = meta.get("concept-id", "?")
        temporal = (umm.get("TemporalExtents") or [{}])[0]
        ranges = (temporal.get("RangeDateTimes") or [{}])[0]
        out[f"{entry['short_name']} ({concept})"] = {
            "version": umm.get("Version"),
            "revision_date": str(meta.get("revision-date", ""))[:10],
            "temporal_start": str(ranges.get("BeginningDateTime", ""))[:10],
            "temporal_end": str(ranges.get("EndingDateTime", "") or "ongoing")[:10],
            "url": f"https://cmr.earthdata.nasa.gov/search/concepts/{concept}.html",
        }
    if not out:
        raise RuntimeError(f"CMR knows no collection {entry['short_name']!r}.")
    return out


def check_stac(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    """Presence, temporal extent and newest item of each watched collection."""
    root = entry["url"].rstrip("/")
    response = session.get(f"{root}/collections", timeout=TIMEOUT)
    response.raise_for_status()
    present = {c["id"]: c for c in response.json().get("collections", [])}
    out: dict[str, Any] = {}
    for name in entry.get("collections", []):
        collection = present.get(name)
        if collection is None:
            continue  # absence is reported by the diff, not recorded here
        interval = collection.get("extent", {}).get("temporal", {}).get("interval") or [
            []
        ]
        bounds = interval[0] if interval and interval[0] else [None, None]
        out[f"{entry['id']}/{name}"] = {
            "temporal_start": str(bounds[0] or "")[:10],
            "temporal_end": str(bounds[1] or "ongoing")[:10],
            "license": collection.get("license"),
            "url": f"{root}/collections/{name}",
        }
    return out


def _ee_time(millis: Any) -> str:
    """An Earth Engine ``system:time_start`` as a date, or an empty string."""
    if not millis:
        return ""
    return datetime.fromtimestamp(int(millis) / 1000, tz=UTC).strftime("%Y-%m-%d")


def check_gee(entry: dict[str, Any], _session: Any) -> dict[str, Any]:
    """Existence, image count and date range of each Earth Engine asset.

    ``ee.data.getAsset`` gives the type and an update time but, for the public
    catalog assets this package uses, no date range — so the newest
    acquisition comes from the collection itself. An asset that stops being
    updated then shows up as an ``end`` that stops moving.

    This is the slowest check in the watch (a few seconds per asset, and the
    only one that needs a credential), which is why it asks for the sorted
    head rather than a count.
    """
    from easysnowdata import auth  # noqa: PLC0415

    auth.get("earthengine").ensure()
    import ee  # noqa: PLC0415

    out: dict[str, Any] = {}
    for asset in entry.get("assets", []):
        record: dict[str, Any] = {
            "url": "https://developers.google.com/earth-engine/datasets/catalog/"
            + asset.replace("/", "_")
        }
        try:
            info = ee.data.getAsset(asset)
            record["type"] = info.get("type", "")
            record["updated"] = str(info.get("updateTime", ""))[:10]
        except Exception as exc:  # noqa: BLE001 — a community asset may not resolve
            record["type"] = "unknown"
            record["getAsset"] = f"{type(exc).__name__}"
        try:
            if record["type"] == "IMAGE":
                record["bands"] = len(ee.Image(asset).bandNames().getInfo())
            elif record["type"] == "TABLE":
                record["features"] = ee.FeatureCollection(asset).size().getInfo()
            else:
                collection = ee.ImageCollection(asset)
                # Filter to a recent window *before* sorting. Sorting the
                # whole of OPERA/RTC/L2_V1/S1 by system:time_start never
                # returns — it is global Sentinel-1, millions of images —
                # while the same sort over the last 30 days answers in five
                # seconds. Widen the window until something is found, so a
                # collection that stopped a year ago still reports its end.
                newest = None
                for days in (30, 365, 3650, 20000):
                    since = (datetime.now(UTC) - timedelta(days=days)).strftime(
                        "%Y-%m-%d"
                    )
                    # aggregate_array, not .first().get(): an empty window
                    # gives [] here and raises "Parameter 'object' may not be
                    # null" there, which would end the loop on the first miss.
                    times = (
                        collection.filterDate(since, "2100-01-01")
                        .limit(1, "system:time_start", False)
                        .aggregate_array("system:time_start")
                        .getInfo()
                    )
                    if times:
                        newest = times[0]
                        record["window_days"] = days
                        break
                record["end"] = _ee_time(newest)
                # size() is a server-side scan of the whole collection, so it
                # is opt-in per entry rather than the default.
                if entry.get("counts"):
                    record["images"] = collection.size().getInfo()
        except Exception as exc:  # noqa: BLE001 — record the failure as a value
            record["state"] = f"unreachable: {type(exc).__name__}: {exc}"[:150]
        out[asset] = record
    return out


def _entry_url(entry: dict[str, Any]) -> str:
    """The URL to fetch: given outright, or taken from the catalog source.

    An entry that names ``product`` and ``source`` instead of ``url`` cannot
    drift from the location the loader actually reads.
    """
    if entry.get("url"):
        return str(entry["url"])
    from easysnowdata import catalog  # noqa: PLC0415

    return catalog.get(entry["product"]).source(entry.get("source")).location


def check_file(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    """ETag, Last-Modified and Content-Length of a static file.

    GET with a one-byte range, not HEAD: GRDC answers HEAD with 400 and that
    is indistinguishable from a 404 (§8).
    """
    url = _entry_url(entry)
    response = session.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"Range": "bytes=0-0"},
    )
    headers = response.headers
    status = response.status_code
    response.close()
    return {
        url: {
            "status": status,
            "etag": headers.get("ETag", ""),
            "last_modified": headers.get("Last-Modified", ""),
            "length": headers.get("Content-Range", "").rsplit("/", 1)[-1]
            or headers.get("Content-Length", ""),
            "url": url,
        }
    }


def check_stations(entry: dict[str, Any], _session: Any) -> dict[str, Any]:
    """How many stations each network reports today."""
    from easysnowdata import stations  # noqa: PLC0415

    out: dict[str, Any] = {}
    for network in entry.get("networks", []):
        try:
            inventory = stations.inventory(networks=network, source="clients")
        except Exception as exc:  # noqa: BLE001 — a network being down is a value
            out[network] = {"count": f"unreachable: {type(exc).__name__}"}
            continue
        record: dict[str, Any] = {"count": len(inventory)}
        # The clients advertise a `daily` flag; the archive's probe-verified
        # `daily_or_better` is a different, stronger claim (§9.3). Record
        # whichever the route offers, and say which.
        for column in ("daily_or_better", "daily"):
            if column not in inventory:
                continue
            values = inventory[column]
            # The AWDB client leaves `daily` unset on all ~7 000 of its
            # stations while DataBC and Yukon fill it in. Recording 0 there
            # would read as "none are daily", which is false; it is unknown.
            record[column] = (
                "not advertised"
                if values.isna().all()
                else int(values.fillna(False).astype(bool).sum())
            )
            break
        if "is_active" in inventory:
            record["active"] = int(
                inventory["is_active"].fillna(False).astype(bool).sum()
            )
        out[network] = record
    return out


def check_pypi(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    """The newest release of each watched package."""
    out: dict[str, Any] = {}
    for package in entry.get("packages", []):
        response = session.get(f"https://pypi.org/pypi/{package}/json", timeout=TIMEOUT)
        if response.status_code != 200:
            out[package] = {"latest_version": f"HTTP {response.status_code}"}
            continue
        info = response.json()["info"]
        out[package] = {
            "latest_version": info["version"],
            "url": CHANGELOGS.get(package, info.get("project_url") or ""),
        }
    return out


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_MARKUP_RE = re.compile(r"<[^>]+>")


def visible_text(html: str, *, min_length: int = 25, limit: int = 600) -> list[str]:
    """The visible lines of a page, deduplicated and trimmed.

    Deliberately crude: a real parse would need the page's own structure, and
    what the digest needs is only "which sentences are new since last week".
    """
    text = _TAG_RE.sub(" ", html)
    text = _MARKUP_RE.sub("\n", text)
    seen, lines = set(), []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if len(line) < min_length or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def check_page(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    response = session.get(entry["url"], timeout=TIMEOUT)
    response.raise_for_status()
    return {"lines": visible_text(response.text)}


def check_feed(entry: dict[str, Any], session: Any) -> dict[str, Any]:
    """A JSON feed or a plain-text changelog, reduced to comparable lines."""
    response = session.get(entry["url"], timeout=TIMEOUT)
    response.raise_for_status()
    body = response.text
    if entry["url"].endswith(".json"):
        payload = response.json()
        topics = (
            payload.get("topic_list", {}).get("topics", [])
            if isinstance(payload, dict)
            else []
        )
        if topics:
            return {
                "lines": [str(t.get("title", "")) for t in topics if t.get("title")]
            }
        return {"lines": visible_text(json.dumps(payload))}
    return {"lines": [line.strip() for line in body.splitlines() if line.strip()][:600]}


CHECKS = {
    "cmr": (check_cmr, diff_mapping),
    "stac": (check_stac, diff_mapping),
    "gee": (check_gee, diff_mapping),
    "file": (check_file, diff_mapping),
    "stations": (check_stations, diff_mapping),
    "pypi": (check_pypi, diff_mapping),
    "page": (check_page, diff_lines),
    "feed": (check_feed, diff_lines),
}


# ── the run ───────────────────────────────────────────────────────────────────


def run(
    watchlist: dict[str, list[dict]],
    *,
    directory: Path = SNAPSHOT_DIR,
    only: list[str] | None = None,
    write: bool = True,
    session: Any = None,
    log: Any = print,
) -> tuple[list[Change], list[str]]:
    """Fetch and diff every entry. Returns ``(changes, errors)``."""
    session = session or _session()
    changes: list[Change] = []
    errors: list[str] = []
    for kind, entries in watchlist.items():
        if kind not in CHECKS or (only and kind not in only):
            continue
        fetch, diff = CHECKS[kind]
        for entry in entries:
            entry_id = entry["id"]
            try:
                after = fetch(entry, session)
            except Exception as exc:  # noqa: BLE001 — one dead page is not fatal
                errors.append(f"{entry_id}: {type(exc).__name__}: {exc}")
                log(f"  ⚠️  {entry_id}: {type(exc).__name__}: {exc}")
                continue
            before = read_snapshot(directory, entry_id)
            found = diff(entry, before, after)
            forced = FORCED_CATEGORY.get(kind)
            if forced:
                for change in found:
                    change.categories = [forced]
            changes.extend(found)
            log(f"  {'•' if found else '·'} {entry_id}: {len(found)} change(s)")
            if write:
                write_snapshot(directory, entry_id, after)
    return changes, errors


def _item(change: Change) -> str:
    """One digest line, without the list marker."""
    products = (
        " — affects " + ", ".join(f"`{p}`" for p in change.products)
        if change.products
        else ""
    )
    subject = f"[{change.subject}]({change.url})" if change.url else change.subject
    tags = " ".join(f"`{t}`" for t in change.categories)
    return f"**{subject}** {change.detail}{products} {tags}".rstrip()


def digest(
    changes: list[Change],
    errors: list[str],
    *,
    when: str | None = None,
    carried: list[str] | None = None,
    discussions: list[int] | None = None,
) -> str:
    """The Markdown body of one weekly digest issue.

    *carried* are the unticked to-dos of the digests this one supersedes
    (:func:`unaddressed`), already tagged with the issue they came from;
    *discussions* the numbers of those digests that have comments, so a
    thread is not lost when its issue closes.
    """
    when = when or datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        f"<!-- esd-watch: {when} -->",
        "",
        f"What moved upstream in the week to {when}, from `WATCHLIST.toml`. "
        "Nothing here is acted on automatically; this is the hand-off point. "
        "Tick an item off once it is handled or judged not to matter; an "
        "unticked one is carried into next week's digest when this one closes.",
        "",
    ]
    order = [*SECTIONS, "Other"]
    by_section: dict[str, list[Change]] = {name: [] for name in order}
    for change in changes:
        by_section[change.section].append(change)

    fresh = {_item(c) for name in TODO_SECTIONS for c in by_section[name]}
    carried = [c for c in (carried or []) if _strip_origin(c) not in fresh]

    for section in order:
        found = by_section[section]
        if section == "Changed" and carried:
            # Right after this week's to-dos, so all of them are in one place.
            _carried_section(lines, carried, discussions or [])
        if not found:
            continue
        todo = section in TODO_SECTIONS
        shown, hidden = (
            (found, []) if todo else (found[:SECTION_LIMIT], found[SECTION_LIMIT:])
        )
        if section in FOLDED_SECTIONS:
            lines += [
                f"<details><summary><b>{section} ({len(found)})</b></summary>",
                "",
            ]
        else:
            lines += [f"## {section} ({len(found)})", ""]
        marker = "- [ ] " if todo else "- "
        lines += [marker + _item(change) for change in shown]
        if hidden:
            entries = ", ".join(sorted({c.entry for c in hidden}))
            lines.append(f"- _...and {len(hidden)} more, from {entries}_")
        lines.append("")
        if section in FOLDED_SECTIONS:
            lines += ["</details>", ""]

    if errors:
        lines += [
            f"## Could not be checked ({len(errors)})",
            "",
            "A watchlist entry that stops answering is itself worth knowing about.",
            "",
            *(f"- `{error}`" for error in errors),
            "",
        ]
    lines += [
        "---",
        "",
        "Snapshots are in `data_status/watch/`; the diff that produced this is "
        "against the previous run. Reproduce with `python scripts/watch.py "
        "--dry-run`.",
    ]
    return "\n".join(lines)


# ── superseding the previous digest ──────────────────────────────────────────

CARRIED_HEADING = "## Carried over"
_ORIGIN_RE = re.compile(r"\s*_\(from #\d+\)_$")


def _strip_origin(item: str) -> str:
    return _ORIGIN_RE.sub("", item)


def _carried_section(
    lines: list[str], carried: list[str], discussions: list[int]
) -> None:
    lines += [
        f"{CARRIED_HEADING} ({len(carried)})",
        "",
        "Unticked to-dos from the digests this one replaces.",
        "",
        *(f"- [ ] {item}" for item in carried),
        "",
    ]
    if discussions:
        refs = ", ".join(f"#{n}" for n in discussions)
        lines += [f"Earlier discussion is in {refs}.", ""]


def unaddressed(body: str, number: int) -> list[str]:
    """The to-dos of an old digest that nobody ticked off.

    Items under "Action needed", "Worth adding" and "Carried over", as
    unticked checkboxes (``- [ ]``); in a digest from before the checkboxes,
    every bullet in those sections, including a "...and N more" line, since
    there is no way to tell what was handled. Each is tagged with the issue
    it first appeared in, which it keeps as it is carried on.
    """
    items: list[str] = []
    section = ""
    for line in body.splitlines():
        if line.startswith("## "):
            section = line[3:]
            continue
        if not any(
            section.startswith(name) for name in (*TODO_SECTIONS, "Carried over")
        ):
            continue
        if re.match(r"- \[[xX]\] ", line):
            continue
        if line.startswith("- [ ] "):
            text = line[6:]
        elif line.startswith("- "):
            text = line[2:]
        else:
            continue
        if not _ORIGIN_RE.search(text):
            text = f"{text} _(from #{number})_"
        items.append(text)
    return items


def _github(method: str, path: str, token: str, **kwargs: Any) -> Any:
    import requests  # noqa: PLC0415

    response = requests.request(
        method,
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def open_digests(repo: str, token: str) -> list[dict[str, Any]]:
    """The open digest issues, oldest first (pull requests excluded)."""
    issues = _github(
        "GET",
        f"/repos/{repo}/issues",
        token,
        params={"labels": LABEL, "state": "open", "per_page": 100},
    )
    return sorted(
        (i for i in issues if "pull_request" not in i), key=lambda i: i["number"]
    )


def open_issue(body: str, title: str, repo: str, token: str) -> dict[str, Any]:
    """Open the digest issue, returning GitHub's record of it."""
    return _github(
        "POST",
        f"/repos/{repo}/issues",
        token,
        json={"title": title, "body": body, "labels": [LABEL]},
    )


def supersede(
    old: dict[str, Any], new_number: int, carried: int, repo: str, token: str
) -> None:
    """Point an old digest at its replacement, then close it."""
    note = f"Superseded by #{new_number}. " + (
        f"Its {carried} unticked to-do(s) are carried over there."
        if carried
        else "It had no unticked to-dos left."
    )
    _github(
        "POST",
        f"/repos/{repo}/issues/{old['number']}/comments",
        token,
        json={"body": note},
    )
    _github(
        "PATCH",
        f"/repos/{repo}/issues/{old['number']}",
        token,
        json={"state": "closed", "state_reason": "completed"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watchlist", default=str(WATCHLIST))
    parser.add_argument("--snapshots", default=str(SNAPSHOT_DIR))
    parser.add_argument(
        "--only", nargs="*", choices=sorted(CHECKS), help="only these kinds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="do not write the snapshots"
    )
    parser.add_argument(
        "--issue", action="store_true", help="open the digest issue on GitHub"
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--output", help="also write the digest to this file")
    args = parser.parse_args(argv)

    path = Path(args.watchlist)
    if not path.exists():
        print(f"No watchlist at {path}.", file=sys.stderr)
        return 2
    watchlist = tomllib.loads(path.read_text())

    print("Watching upstream…\n")
    changes, errors = run(
        watchlist,
        directory=Path(args.snapshots),
        only=args.only,
        write=not args.dry_run,
    )
    print(f"\n{len(changes)} change(s), {len(errors)} entry/entries unreachable.")

    if not changes and not errors:
        print("Nothing to report; no issue opened.")
        return 0

    when = datetime.now(UTC).strftime("%Y-%m-%d")
    issuing = bool(args.issue and os.environ.get("GITHUB_TOKEN") and args.repo)
    previous: list[dict[str, Any]] = []
    carried: dict[int, list[str]] = {}
    if issuing:
        previous = open_digests(args.repo, os.environ["GITHUB_TOKEN"])
        carried = {
            old["number"]: unaddressed(old.get("body") or "", old["number"])
            for old in previous
        }
    body = digest(
        changes,
        errors,
        when=when,
        carried=[item for items in carried.values() for item in items],
        discussions=[old["number"] for old in previous if old.get("comments")],
    )
    if args.output:
        Path(args.output).write_text(body)
        print(f"Digest written to {args.output}")
    if issuing:
        token = os.environ["GITHUB_TOKEN"]
        issue = open_issue(body, f"Upstream watch digest — {when}", args.repo, token)
        print(f"Digest issue: {issue['html_url']}")
        # Only after the new one exists, so a failed open closes nothing.
        for old in previous:
            supersede(
                old, issue["number"], len(carried[old["number"]]), args.repo, token
            )
            print(f"Closed #{old['number']} (superseded).")
    else:
        print("\n" + body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
