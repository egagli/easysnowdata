"""Live health probes for every catalog source (§8).

Each :class:`~easysnowdata.catalog.Probe` performs the minimal request that
confirms an access route is alive: a GET of the first byte for static file
hosts (GET first — GRDC answers HEAD with 400; only a final 200/206 counts,
figshare's bot-challenge page answers 202), a STAC search, a Zarr metadata
read, an Earth Engine ``getInfo()`` or an ``earthaccess`` search. Credentials
come from :mod:`easysnowdata.auth`, so a probe whose source requires a
provider that is not configured is reported as ``skip`` rather than ``fail``.

``python -m easysnowdata.catalog.health`` (or the thin
``scripts/check_data_sources.py`` wrapper) runs everything and appends the
results to ``data_status/history.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from easysnowdata import auth
from easysnowdata.catalog import _registry
from easysnowdata.catalog._models import Probe, Product, Source

__all__ = [
    "TIMEOUT",
    "http_first_byte",
    "stac_search",
    "zarr_metadata",
    "gee_asset",
    "earthdata_search",
    "run",
    "update_history",
    "summarize",
    "main",
]

_logger = logging.getLogger(__name__)

TIMEOUT = 20  # seconds for HTTP requests
TEST_BBOX = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier


# ── generic probes ────────────────────────────────────────────────────────────


def http_first_byte(url: str) -> None:
    """Raise unless a GET of the first byte of *url* returns 200 or 206.

    GET-first, ``Range: bytes=0-0``, streamed and closed immediately, redirects
    followed. HEAD is only a fallback when the GET itself fails.
    """
    import requests  # noqa: PLC0415

    response = requests.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"Range": "bytes=0-0"},
    )
    status = response.status_code
    response.close()
    if status in (200, 206):
        return
    head = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
    if head.status_code in (200, 206):
        return
    raise RuntimeError(f"Unreachable: HTTP {status}")


def stac_search(
    api_url: str,
    collection: str,
    *,
    bbox: tuple[float, float, float, float] = TEST_BBOX,
    datetime_range: str | None = None,
    query: dict[str, Any] | None = None,
    sign: bool = False,
) -> None:
    """Raise unless a STAC search of *collection* returns at least one item."""
    import pystac_client  # noqa: PLC0415

    modifier = auth.get("planetary_computer").sign if sign else None
    catalog = pystac_client.Client.open(api_url, modifier=modifier)
    kwargs: dict[str, Any] = {"collections": [collection], "max_items": 1}
    if bbox is not None:
        kwargs["bbox"] = bbox
    if datetime_range:
        kwargs["datetime"] = datetime_range
    if query:
        kwargs["query"] = query
    items = list(catalog.search(**kwargs).items())
    if not items:
        raise RuntimeError(f"No {collection} items found in {api_url}.")


def zarr_metadata(url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Raise unless the Zarr store at *url* opens with at least one variable."""
    import xarray as xr  # noqa: PLC0415

    ds = xr.open_zarr(
        url,
        chunks=None,
        storage_options=storage_options or {"token": "anon"},
        consolidated=True,
    )
    if not ds.data_vars:
        raise RuntimeError(f"{url} opened but has no data variables.")


def gee_asset(asset_id: str, kind: str = "image_collection", **filters: Any) -> None:
    """Raise unless an Earth Engine asset answers ``getInfo()``.

    *kind* is ``"image"``, ``"image_collection"`` or ``"feature_collection"``;
    ``start``/``end`` filter a collection by date first.
    """
    auth.get("earthengine").ensure()
    import ee  # noqa: PLC0415

    if kind == "image":
        if not ee.Image(asset_id).getInfo():
            raise RuntimeError(f"{asset_id} returned no info.")
        return
    if kind == "feature_collection":
        info = ee.FeatureCollection(asset_id).limit(1).getInfo()
        if not info.get("features"):
            raise RuntimeError(f"{asset_id} returned no features.")
        return
    col = ee.ImageCollection(asset_id)
    if filters.get("start") and filters.get("end"):
        col = col.filterDate(filters["start"], filters["end"])
    if col.size().getInfo() == 0:
        raise RuntimeError(f"{asset_id} returned 0 images.")


def earthdata_search(
    short_name: str,
    *,
    bbox: tuple[float, float, float, float] = TEST_BBOX,
    temporal: tuple[str, str] | None = ("2020-01-01", "2020-01-07"),
    cloud_hosted: bool = True,
) -> None:
    """Raise unless ``earthaccess.search_data`` finds a granule."""
    auth.get("earthdata").ensure()
    import earthaccess  # noqa: PLC0415

    kwargs: dict[str, Any] = {"short_name": short_name, "count": 1}
    if cloud_hosted:
        kwargs["cloud_hosted"] = True
    if bbox is not None:
        kwargs["bounding_box"] = bbox
    if temporal is not None:
        kwargs["temporal"] = temporal
    if not earthaccess.search_data(**kwargs):
        raise RuntimeError(f"{short_name}: no granules found.")


# ── the runner ────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _missing_credentials(requires: Iterable[str]) -> str | None:
    missing = [name for name in requires if not auth.detect(name)]
    if not missing:
        return None
    wanted = "; ".join(
        f"{auth.get(name).title}: {' or '.join(auth.get(name).env_vars) or auth.get(name).files}"
        for name in missing
    )
    return f"Missing credentials — {wanted}"


def _check(product: Product, source: Source, probe: Probe) -> dict[str, Any]:
    requires = source.requires if probe.requires is None else probe.requires
    result: dict[str, Any] = {
        "source": probe.label,
        "product": product.id,
        "route": source.id,
        "requires": list(requires),
        "checked_at": _now(),
    }
    reason = _missing_credentials(requires)
    if reason:
        result.update(status="skip", error=reason)
        return result
    try:
        probe.fn()
    except Exception as exc:  # noqa: BLE001 — any failure is a health failure
        result.update(status="fail", error=f"{type(exc).__name__}: {exc}")
    else:
        result.update(status="pass", error=None)
    return result


def probes(
    product_ids: Iterable[str] | None = None,
) -> list[tuple[Product, Source, Probe]]:
    """Every (product, source, probe) triple, in catalog order."""
    items = (
        [_registry.get(pid) for pid in product_ids]
        if product_ids is not None
        else list(_registry.products().values())
    )
    return [(p, s, probe) for p in items for s in p.sources for probe in s.health]


def run(
    product_ids: Iterable[str] | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every health probe (or those of *product_ids*) and return the results.

    Each result has ``source`` (the row label), ``product``, ``route``,
    ``requires``, ``status`` (``pass``/``fail``/``skip``), ``error`` and
    ``checked_at``. *progress* receives one line per probe.
    """
    results = []
    for product, source, probe in probes(product_ids):
        result = _check(product, source, probe)
        results.append(result)
        if progress is not None:
            symbol = {"pass": "✅", "fail": "❌", "skip": "⚠️"}[result["status"]]
            line = f"  {symbol} {probe.label}"
            if result["status"] != "pass":
                line += f"\n     → {result['error']}"
            progress(line)
        _logger.info("%s: %s", probe.label, result["status"])
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for r in results if r["status"] == status)
        for status in ("pass", "fail", "skip")
    }


def update_history(
    results: list[dict[str, Any]], history_path: Path, keep: int = 52
) -> None:
    """Prepend *results* to the rolling history JSON (at most *keep* runs)."""
    history: list[list[dict[str, Any]]] = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
    history.insert(0, results)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history[:keep], indent=2, ensure_ascii=False) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: run the probes, print a summary, append to the history file.

    Returns 0 even when probes fail — the README table and the (planned)
    health→issue workflow are the reporting channel, not the exit code; pass
    ``--strict`` to exit 1 on any failure.
    """
    parser = argparse.ArgumentParser(description="Check easysnowdata data sources.")
    parser.add_argument(
        "--output", default="data_status/history.json", help="rolling history JSON"
    )
    parser.add_argument(
        "--product", action="append", help="only these product ids (repeatable)"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="do not write the history file"
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if any probe fails"
    )
    args = parser.parse_args(argv)

    out = sys.stdout
    out.write("Running data source health checks…\n\n")
    results = run(args.product, progress=lambda line: out.write(line + "\n"))
    counts = summarize(results)
    out.write(
        f"\nSummary: {counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped\n"
    )
    if not args.no_history:
        update_history(results, Path(args.output))
        out.write(f"History written to {args.output}\n")
    return 1 if args.strict and counts["fail"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
