"""Generate the tiny local fixtures the offline (recorded) tier reads.

Run as a script to write them somewhere (``python tests/fixtures/make_fixtures.py
<dir>``), or call :func:`make_all` from a fixture. Everything is synthetic and a
few kilobytes: a Cloud-Optimized GeoTIFF with a categorical band, a Zarr store
shaped like a global reanalysis (time, latitude, longitude 0–360), and a
GeoParquet of a few polygons with a GeoParquet 1.1 bbox column.

The AOI every fixture covers is the Mount Rainier test box
``(-121.94, 46.72, -121.54, 46.99)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio  # noqa: F401 — must be imported before geopandas/pyproj: writing a

# GeoTIFF after a geopandas reprojection double-frees in the conda-forge GDAL/PROJ
# stack when geopandas was imported first (observed with geopandas 1.1.3,
# rasterio 1.5.0, WSL2). The test session generates fixtures in a subprocess.

RAINIER = (-121.94, 46.72, -121.54, 46.99)


def make_cog(path: Path, *, crs: str = "EPSG:32610", size: int = 64) -> Path:
    """A 64×64 uint8 categorical COG (values 1–4, nodata 0) covering Rainier in UTM 10N."""
    import geopandas as gpd
    import shapely
    from rasterio.transform import from_bounds

    west, south, east, north = (
        gpd.GeoSeries([shapely.box(*RAINIER)], crs="EPSG:4326").to_crs(crs).total_bounds
    )
    transform = from_bounds(west, south, east, north, size, size)
    rng = np.random.default_rng(0)
    data = rng.integers(1, 5, size=(size, size), dtype="uint8")
    data[:4, :4] = 0  # a nodata corner
    profile = {
        "driver": "COG",
        "dtype": "uint8",
        "width": size,
        "height": size,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "DEFLATE",
        "blocksize": 32,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            1, flag_values="1 2 3 4", flag_meanings="tundra maritime prairie ice"
        )
    return path


def make_zarr(path: Path) -> Path:
    """A reanalysis-like Zarr store: 3 hourly steps × 0.25° global grid, longitude 0–360."""
    import pandas as pd
    import xarray as xr

    lat = np.arange(90, -90.01, -0.25)
    lon = np.arange(0, 360, 0.25)
    time = pd.date_range("2020-01-01", periods=3, freq="h")
    rng = np.random.default_rng(1)
    t2m = 250 + 30 * rng.random((time.size, lat.size, lon.size), dtype="float32")
    ds = xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), t2m, {"units": "K"})},
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={"title": "synthetic ARCO-ERA5 look-alike"},
    )
    ds = ds.chunk({"time": 1, "latitude": 721, "longitude": 1440})
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def make_geoparquet(path: Path) -> Path:
    """Six basin-like polygons, two of which intersect the Rainier box, with a bbox column."""
    import geopandas as gpd
    import shapely

    polys = [
        shapely.box(-122.0, 46.6, -121.7, 46.9),  # intersects
        shapely.box(-121.7, 46.8, -121.4, 47.1),  # intersects
        shapely.box(-123.0, 46.0, -122.5, 46.5),
        shapely.box(-120.0, 47.5, -119.5, 48.0),
        shapely.box(-121.0, 45.0, -120.5, 45.5),
        shapely.box(170.0, -20.0, 175.0, -15.0),
    ]
    gdf = gpd.GeoDataFrame(
        {"name": [f"basin-{i}" for i in range(len(polys))], "level": [5] * len(polys)},
        geometry=polys,
        crs="EPSG:4326",
    )
    gdf.to_parquet(path, write_covering_bbox=True, schema_version="1.1.0")
    return path


def make_geojson(path: Path) -> Path:
    """The same polygons as GeoJSON, for the pyogrio route."""
    import geopandas as gpd

    gdf = gpd.read_parquet(path.with_suffix(".parquet"))
    gdf.to_file(path, driver="GeoJSON")
    return path


def make_all(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {
        "cog": make_cog(directory / "classes.tif"),
        "zarr": make_zarr(directory / "reanalysis.zarr"),
        "geoparquet": make_geoparquet(directory / "basins.parquet"),
    }
    out["geojson"] = make_geojson(directory / "basins.geojson")
    return out


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    for name, path in make_all(target).items():
        print(f"{name}: {path}")
