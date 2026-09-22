"""Write the tiny station fixtures the offline tier reads.

Run as a subprocess by ``tests/stations/conftest.py``, printing ``name: path``
lines. It is a subprocess for the same reason the other tiers' generators are:
writing a vector file through geopandas in a process that has already imported
rasterio crashes GDAL in this environment, so rasterio is imported first and
the whole thing is kept out of the test process.

Three fixtures, all scaled down from the real artefacts:

``inventory``
    A four-station GeoJSON with the properties ``all_snow_stations.geojson``
    carries (that repo's DESIGN.md §6.1), one station per network shape that
    matters — an AWDB triplet code, a CDEC code, an NVE dotted code and a
    Yukon hyphenated one.
``archive``
    A ``.tar.xz`` of ``date,wteq_cm,snwd_cm`` CSVs under ``stations/``, laid
    out exactly like ``data/all_station_csvs.tar.xz``.
``zarr_by_time`` / ``zarr_by_station`` / ``manifest``
    The same series as Zarr stores in the two chunk layouts that repo's
    ``scripts/build_zarr_archive.py`` publishes (its DESIGN.md §6.5): format
    3, consolidated metadata, float32 centimetres, ``swe`` and ``snow_depth``
    on ``(station, time)``, a complete daily ``time`` axis, station metadata
    as coordinates, and an ``archive.json`` manifest.
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import rasterio  # noqa: F401 — import first; see the module docstring

import geopandas as gpd  # isort: skip
import numpy as np  # isort: skip
import pandas as pd  # isort: skip
import shapely  # isort: skip
import xarray as xr  # isort: skip

STATIONS = [
    {
        "code": "679_WA_SNTL",
        "name": "Paradise",
        "client": "awdb",
        "network": None,
        "network_code": "SNTL",
        "latitude": 46.7825,
        "longitude": -121.7458,
        "elevation_m": 1569.7,
        "state": "WA",
        "operator": "USDA NRCS",
        "data_provider": "USDA NRCS AWDB",
        "status": "Active",
        "is_active": True,
        "daily_or_better": True,
        "daily_verified": True,
        "daily_provenance": "native_daily",
        "has_daily_swe": True,
        "has_daily_snwd": True,
    },
    {
        "code": "QUA",
        "name": "QUARTZ BASIN",
        "client": "cdec",
        "network": None,
        "network_code": "CCSS",
        "latitude": 38.4,
        "longitude": -119.8,
        "elevation_m": 2400.0,
        "state": "CA",
        "operator": "CA DWR",
        "data_provider": "CDEC (CA DWR)",
        "status": "Active",
        "is_active": True,
        "daily_or_better": True,
        "daily_verified": True,
        "daily_provenance": "native_daily",
        "has_daily_swe": True,
        "has_daily_snwd": False,
    },
    {
        "code": "12.142.0",
        "name": "Bakko",
        "client": "nve",
        "network": None,
        "network_code": None,
        "latitude": 59.8,
        "longitude": 7.5,
        "elevation_m": 1100.0,
        "state": None,
        "operator": "NVE",
        "data_provider": "NVE HydAPI",
        "status": "Active",
        "is_active": True,
        "daily_or_better": True,
        "daily_verified": True,
        "daily_provenance": "native_daily",
        "has_daily_swe": True,
        "has_daily_snwd": True,
    },
    {
        # A periodic CCSS snow course: in the inventory, no archive CSV.
        "code": "HNT",
        "name": "HUNTINGTON LAKE",
        "client": "cdec",
        "network": None,
        "network_code": "CCSS",
        "latitude": 37.2,
        "longitude": -119.2,
        "elevation_m": 2134.0,
        "state": "CA",
        "operator": "CA DWR",
        "data_provider": "CDEC (CA DWR)",
        "status": "Active",
        "is_active": True,
        "daily_or_better": False,
        "daily_verified": True,
        "daily_provenance": "none",
        "has_daily_swe": False,
        "has_daily_snwd": False,
    },
    {
        # A periodic snow course: in the inventory, but with no archive CSV.
        "code": "08AA-SC01",
        "name": "Canyon Lake",
        "client": "yukon",
        "network": "Yukon Snow Survey Network",
        "network_code": None,
        "latitude": 60.6,
        "longitude": -135.0,
        "elevation_m": 900.0,
        "state": "YT",
        "operator": "Government of Yukon",
        "data_provider": "Yukon Water Data (AquaCache)",
        "status": "Active",
        "is_active": True,
        "daily_or_better": False,
        "daily_verified": True,
        "daily_provenance": "none",
        "has_daily_swe": False,
        "has_daily_snwd": False,
    },
]

#: Stations that have a CSV in the archive, with a plausible winter series.
WITH_CSVS = ("679_WA_SNTL", "QUA", "12.142.0")


def _series(code: str) -> pd.DataFrame:
    """A short daily series spanning a water-year boundary."""
    dates = pd.date_range("2023-09-29", "2023-10-05")
    rng = np.random.default_rng(abs(hash(code)) % (2**32))
    swe = np.round(
        np.linspace(0.0, 12.0, len(dates)) + rng.normal(0, 0.2, len(dates)), 3
    )
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wteq_cm": swe})
    # QUA has no snow-depth sensor, so its whole column is missing. `to_csv`
    # writes NaN as an empty field, which is what the real archive does:
    # missing values are empty, never the string "nan" (DESIGN.md §6.3).
    frame["snwd_cm"] = np.nan if code == "QUA" else np.round(swe * 3.2, 3)
    frame.loc[1, "wteq_cm"] = np.nan  # a genuine gap
    return frame


def main(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)

    features = [
        {
            "type": "Feature",
            "properties": station,
            "geometry": json.loads(
                shapely.to_geojson(
                    shapely.Point(station["longitude"], station["latitude"])
                )
            ),
        }
        for station in STATIONS
    ]
    inventory = target / "all_snow_stations.geojson"
    gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").to_file(
        inventory, driver="GeoJSON"
    )
    print(f"inventory: {inventory}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "stations"
        staging.mkdir()
        for code in WITH_CSVS:
            _series(code).to_csv(staging / f"{code}.csv", index=False)
        archive = target / "all_station_csvs.tar.xz"
        with tarfile.open(archive, "w:xz") as tar:
            tar.add(staging, arcname="stations")
    print(f"archive: {archive}")

    for code in WITH_CSVS:
        path = target / f"{code}.csv"
        _series(code).to_csv(path, index=False)
        print(f"csv_{code}: {path}")

    _write_zarr_stores(target)


def _write_zarr_stores(target: Path) -> None:
    """The CSV series again, as the two chunked stores Pages serves."""
    import warnings

    frames = {code: _series(code).set_index("date") for code in WITH_CSVS}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    tmin = min(f.index.min() for f in frames.values())
    tmax = max(f.index.max() for f in frames.values())
    time = pd.date_range(tmin, tmax, freq="D")
    codes = np.array(list(frames), dtype=str)
    swe = np.full((len(codes), len(time)), np.nan, dtype="float32")
    snd = np.full_like(swe, np.nan)
    for i, frame in enumerate(frames.values()):
        pos = (frame.index - tmin).days.to_numpy()
        swe[i, pos] = frame["wteq_cm"].to_numpy(dtype="float32")
        snd[i, pos] = frame["snwd_cm"].to_numpy(dtype="float32")
    meta = {s["code"]: s for s in STATIONS}
    ds = xr.Dataset(
        {
            "swe": (("station", "time"), swe, {"units": "cm"}),
            "snow_depth": (("station", "time"), snd, {"units": "cm"}),
        },
        coords={
            "station": codes,
            "time": time,
            "name": ("station", np.array([meta[c]["name"] for c in codes], dtype=str)),
            "client": (
                "station",
                np.array([meta[c]["client"] for c in codes], dtype=str),
            ),
            "latitude": ("station", np.array([meta[c]["latitude"] for c in codes])),
            "longitude": ("station", np.array([meta[c]["longitude"] for c in codes])),
        },
        attrs={"title": "global_snow_networks daily archive", "units": "cm"},
    )
    layouts = {
        "by_time": (len(codes), min(366, len(time))),
        "by_station": (min(64, len(codes)), len(time)),
    }
    manifest: dict = {"built_from_commit": "fixture", "stores": {}}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        for layout, chunks in layouts.items():
            path = target / f"{layout}.zarr"
            ds.assign_attrs(layout=layout).to_zarr(
                path,
                mode="w",
                zarr_format=3,
                consolidated=True,
                encoding={name: {"chunks": chunks} for name in ds.data_vars},
            )
            manifest["stores"][path.name] = {"layout": layout}
            print(f"zarr_{layout}: {path}")
    manifest_path = target / "archive.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
