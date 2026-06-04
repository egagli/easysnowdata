"""Probe each easysnowdata data source and record pass/fail/skip status.

Usage
-----
    python scripts/check_data_sources.py [--output data_status/history.json]

Each check performs the minimal request necessary to confirm that an
endpoint is reachable and returns data:

* HTTP HEAD/GET for static file hosts (figshare, Zenodo, World Bank, GRDC,
  GitHub, Azure Blob).
* Zarr metadata read for ARCO-ERA5 on GCS (anonymous).
* STAC catalog search for Planetary Computer endpoints.
* GEE ImageCollection.first() for Earth Engine-backed sources.
* earthaccess.search_data() for NASA NSIDC sources.

Credentials are read from environment variables:
    EARTHENGINE_TOKEN     — Google Earth Engine (JSON string)
    EARTHDATA_USERNAME    — NASA EarthData username
    EARTHDATA_PASSWORD    — NASA EarthData password
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT = 20  # seconds for HTTP requests


def _head_ok(url: str) -> tuple[bool, str]:
    """Return (True, '') if the URL responds 200–399, else (False, reason)."""
    try:
        r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code < 400:
            return True, ""
        # Some servers reject HEAD; fall back to GET with stream
        r2 = requests.get(url, timeout=TIMEOUT, stream=True)
        r2.close()
        if r2.status_code < 400:
            return True, ""
        return False, f"HTTP {r2.status_code}"
    except Exception as exc:
        return False, str(exc)


def _check(
    name: str,
    fn: Callable[[], None],
    *,
    requires_env: list[str] | None = None,
) -> dict:
    """Run *fn* and return a result dict."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if requires_env:
        missing = [v for v in requires_env if not os.getenv(v)]
        if missing:
            return {
                "source": name,
                "status": "skip",
                "error": f"Missing env vars: {', '.join(missing)}",
                "checked_at": now,
            }
    try:
        fn()
        return {"source": name, "status": "pass", "error": None, "checked_at": now}
    except Exception as exc:
        return {
            "source": name,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": now,
        }


# ---------------------------------------------------------------------------
# Individual source checks
# ---------------------------------------------------------------------------

def check_snotel_station_list() -> None:
    url = "https://github.com/egagli/snotel_ccss_stations/raw/main/all_stations.geojson"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_snotel_single_csv() -> None:
    url = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data/679_WA_SNTL.csv"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_hydroatlas_figshare() -> None:
    url = "https://figshare.com/ndownloader/files/20082137/BasinATLAS_Data_v10.gdb.zip"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_grdc_major_river_basins() -> None:
    url = "https://datacatalogfiles.worldbank.org/ddh-published/0041426/DR0051689/major_basins_of_the_world_0_0_0.zip"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_grdc_wmo_basins() -> None:
    url = "https://grdc.bafg.de/downloads/wmobb_json.zip/wmobb_basins.json"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_koppen_geiger_figshare() -> None:
    url = "https://figshare.com/ndownloader/files/45057352/koppen_geiger_tif.zip"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_snow_classification_azure() -> None:
    url = "https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_forest_cover_zenodo() -> None:
    url = "https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_mountain_snow_mask_zenodo() -> None:
    url = "https://zenodo.org/records/2626737/files/MODIS_mtnsnow_classes.zip"
    ok, reason = _head_ok(url)
    if not ok:
        raise RuntimeError(f"Unreachable: {reason}")


def check_arco_era5_gcs() -> None:
    import xarray as xr

    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options={"token": "anon"},
        consolidated=True,
    )
    assert len(ds.data_vars) > 0


def check_copernicus_dem_planetary_computer() -> None:
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    results = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=(-121.94, 46.72, -121.54, 46.99),
        max_items=1,
    )
    items = list(results.items())
    if not items:
        raise RuntimeError("No Copernicus DEM items found.")


def check_esa_worldcover_planetary_computer() -> None:
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    results = catalog.search(
        collections=["esa-worldcover"],
        bbox=(-121.94, 46.72, -121.54, 46.99),
        max_items=1,
    )
    items = list(results.items())
    if not items:
        raise RuntimeError("No ESA WorldCover items found.")


def _init_ee_from_token() -> None:
    import google.oauth2.credentials
    import ee

    stored = json.loads(os.environ["EARTHENGINE_TOKEN"])
    credentials = google.oauth2.credentials.Credentials(
        None,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=stored["client_id"],
        client_secret=stored["client_secret"],
        refresh_token=stored["refresh_token"],
        quota_project_id=stored["project"],
    )
    ee.Initialize(
        credentials=credentials,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )


def check_huc_geometries_gee() -> None:
    _init_ee_from_token()
    import ee

    fc = ee.FeatureCollection("USGS/WBD/2017/HUC08").limit(1)
    info = fc.getInfo()
    if not info.get("features"):
        raise RuntimeError("No HUC features returned from GEE.")


def check_snodas_gee() -> None:
    _init_ee_from_token()
    import ee

    col = ee.ImageCollection(
        "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"
    ).filterDate("2020-01-01", "2020-01-02")
    if col.size().getInfo() == 0:
        raise RuntimeError("SNODAS collection returned 0 images.")


def check_era5_gee() -> None:
    _init_ee_from_token()
    import ee

    col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(
        "2020-01-01", "2020-01-02"
    )
    if col.size().getInfo() == 0:
        raise RuntimeError("ERA5 GEE collection returned 0 images.")


def check_chili_gee() -> None:
    _init_ee_from_token()
    import ee

    img = ee.Image("CSP/ERGo/1_0/Global/ALOS_CHILI")
    info = img.getInfo()
    if not info:
        raise RuntimeError("CHILI image returned no info from GEE.")


def check_nlcd_gee() -> None:
    _init_ee_from_token()
    import ee

    col = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
    if col.size().getInfo() == 0:
        raise RuntimeError("NLCD collection returned 0 images.")


def check_ucla_snow_reanalysis_earthaccess() -> None:
    import earthaccess

    earthaccess.login(
        strategy="environment",
        username=os.environ["EARTHDATA_USERNAME"],
        password=os.environ["EARTHDATA_PASSWORD"],
    )
    results = earthaccess.search_data(
        short_name="WUS_UCLA_SR",
        cloud_hosted=True,
        bounding_box=(-121.94, 46.72, -121.54, 46.99),
        temporal=("2020-01-01", "2020-01-07"),
        count=1,
    )
    if not results:
        raise RuntimeError("UCLA Snow Reanalysis: no granules found.")


# ---------------------------------------------------------------------------
# Full check suite
# ---------------------------------------------------------------------------

CHECKS: list[dict] = [
    # No-credential checks
    {"name": "SNOTEL/CCSS station list (GitHub)", "fn": check_snotel_station_list},
    {"name": "SNOTEL/CCSS station CSV (GitHub)", "fn": check_snotel_single_csv},
    {"name": "HydroATLAS basins (figshare)", "fn": check_hydroatlas_figshare},
    {"name": "GRDC major river basins (World Bank)", "fn": check_grdc_major_river_basins},
    {"name": "GRDC WMO basins", "fn": check_grdc_wmo_basins},
    {"name": "Köppen-Geiger classification (figshare)", "fn": check_koppen_geiger_figshare},
    {"name": "Sturm & Liston snow classification (Azure)", "fn": check_snow_classification_azure},
    {"name": "Forest cover fraction (Zenodo)", "fn": check_forest_cover_zenodo},
    {"name": "Mountain snow mask (Zenodo)", "fn": check_mountain_snow_mask_zenodo},
    {"name": "ARCO-ERA5 (GCS anonymous)", "fn": check_arco_era5_gcs},
    {"name": "Copernicus DEM (Planetary Computer)", "fn": check_copernicus_dem_planetary_computer},
    {"name": "ESA WorldCover (Planetary Computer)", "fn": check_esa_worldcover_planetary_computer},
    # GEE-credential checks
    {
        "name": "HUC geometries (GEE/USGS WBD)",
        "fn": check_huc_geometries_gee,
        "requires_env": ["EARTHENGINE_TOKEN"],
    },
    {
        "name": "SNODAS (GEE/Climate Engine)",
        "fn": check_snodas_gee,
        "requires_env": ["EARTHENGINE_TOKEN"],
    },
    {
        "name": "ERA5 (Google Earth Engine)",
        "fn": check_era5_gee,
        "requires_env": ["EARTHENGINE_TOKEN"],
    },
    {
        "name": "CHILI (GEE/CSP ERGo)",
        "fn": check_chili_gee,
        "requires_env": ["EARTHENGINE_TOKEN"],
    },
    {
        "name": "NLCD (GEE/USGS)",
        "fn": check_nlcd_gee,
        "requires_env": ["EARTHENGINE_TOKEN"],
    },
    # earthaccess checks
    {
        "name": "UCLA Snow Reanalysis (NASA NSIDC)",
        "fn": check_ucla_snow_reanalysis_earthaccess,
        "requires_env": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
    },
]


def run_all_checks() -> list[dict]:
    results = []
    for spec in CHECKS:
        print(f"  Checking: {spec['name']} …", end=" ", flush=True)
        result = _check(
            spec["name"],
            spec["fn"],
            requires_env=spec.get("requires_env"),
        )
        symbol = {"pass": "✅", "fail": "❌", "skip": "⚠️"}[result["status"]]
        print(symbol)
        if result["status"] == "fail":
            print(f"    → {result['error']}")
        results.append(result)
    return results


def update_history(results: list[dict], history_path: Path) -> None:
    """Prepend this week's results to the rolling history JSON file."""
    history: list[list[dict]] = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    history.insert(0, results)
    history = history[:52]  # keep at most one year

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory written to {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check easysnowdata data sources.")
    parser.add_argument(
        "--output",
        default="data_status/history.json",
        help="Path to the rolling history JSON file.",
    )
    args = parser.parse_args()

    print("Running data source health checks…\n")
    results = run_all_checks()

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")

    update_history(results, Path(args.output))


if __name__ == "__main__":
    main()
