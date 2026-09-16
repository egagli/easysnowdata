"""Write the tiny station fixtures the offline tier reads.

Run as a subprocess by ``tests/stations/conftest.py``, printing ``name: path``
lines. It is a subprocess for the same reason the other tiers' generators are:
writing a vector file through geopandas in a process that has already imported
rasterio crashes GDAL in this environment, so rasterio is imported first and
the whole thing is kept out of the test process.

Two fixtures, both scaled down from the real artefacts:

``inventory``
    A four-station GeoJSON with the properties ``all_snow_stations.geojson``
    carries (that repo's DESIGN.md §6.1), one station per network shape that
    matters — an AWDB triplet code, a CDEC code, an NVE dotted code and a
    Yukon hyphenated one.
``archive``
    A ``.tar.xz`` of ``date,wteq_cm,snwd_cm`` CSVs under ``stations/``, laid
    out exactly like ``data/all_station_csvs.tar.xz``.
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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
