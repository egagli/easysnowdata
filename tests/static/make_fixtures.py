"""Tiny local fixtures for the Phase 2a (static products) offline tier.

Run as a script (``python tests/static/make_fixtures.py <dir>``) or call
:func:`make_all`; :mod:`tests.static.conftest` does the latter in a subprocess.
Everything is synthetic and a few kilobytes, and every raster covers the Mount
Rainier test box ``(-121.94, 46.72, -121.54, 46.99)``.

``rasterio`` is imported before ``geopandas``/``pyproj`` on purpose: writing a
GeoTIFF after a geopandas reprojection double-frees in the conda-forge
GDAL/PROJ stack when geopandas was imported first (see
``tests/fixtures/make_fixtures.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # noqa: F401 — see the module docstring
from rasterio.transform import from_bounds

RAINIER = (-121.94, 46.72, -121.54, 46.99)


def _write_cog(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float | int | None,
    bounds: tuple[float, float, float, float] = RAINIER,
    crs: str = "EPSG:4326",
    tags: dict[str, str] | None = None,
) -> Path:
    height, width = data.shape
    profile = {
        "driver": "COG",
        "dtype": data.dtype.name,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": from_bounds(*bounds, width, height),
        "nodata": nodata,
        "compress": "DEFLATE",
        "blocksize": 32,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if tags:
            dst.update_tags(**tags)
    return path


def make_dem_cog(path: Path, *, size: int = 48) -> Path:
    """A float32 elevation COG (metres, nodata -32767) with a sea corner."""
    y, x = np.mgrid[0:size, 0:size]
    data = (500 + 40 * (x + y)).astype("float32")  # 500 m → ~4.3 km, Rainier-like
    data[:4, :4] = -32767.0
    return _write_cog(path, data, nodata=-32767.0)


def make_stac_item(
    path: Path,
    *,
    item_id: str = "fixture-item",
    collection: str = "cop-dem-glo-30",
    datetime: str = "2021-04-22T00:00:00Z",
    asset: str = "data",
    nodata: float | int | None = None,
) -> dict[str, Any]:
    """A STAC item dict whose single asset is the local COG at *path*.

    Carries the ``proj`` and ``raster`` extension fields odc-stac reads, so
    ``odc.stac.load`` builds the grid without opening the file first.
    """
    with rasterio.open(path) as src:
        bounds = tuple(src.bounds)
        shape = [src.height, src.width]
        transform = list(src.transform)[:6] + [0.0, 0.0, 1.0]
        dtype = src.dtypes[0]
        epsg = src.crs.to_epsg()
        fill = src.nodata if nodata is None else nodata
    west, south, east, north = bounds
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        ],
        "id": item_id,
        "collection": collection,
        "bbox": [west, south, east, north],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ],
        },
        "properties": {
            "datetime": datetime,
            "proj:epsg": epsg,
            "proj:shape": shape,
            "proj:transform": transform,
        },
        "links": [],
        "assets": {
            asset: {
                "href": str(path),
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
                "raster:bands": [
                    {"data_type": dtype, "nodata": fill, "spatial_resolution": 30}
                ],
            }
        },
    }


def make_categorical_cog(
    path: Path,
    values: list[int],
    *,
    nodata: int = 0,
    dtype: str = "uint8",
    bounds: tuple[float, float, float, float] = RAINIER,
    size: int = 48,
) -> Path:
    """A COG whose pixels cycle through *values*, with a nodata corner."""
    data = np.resize(np.array(values, dtype=dtype), (size, size))
    data[:4, :4] = nodata
    return _write_cog(path, data, nodata=nodata, bounds=bounds)


def make_continuous_cog(
    path: Path,
    *,
    low: int = 0,
    high: int = 100,
    nodata: int = 255,
    dtype: str = "uint8",
    size: int = 48,
) -> Path:
    """A uint8 COG of a 0-100 fraction with a nodata corner."""
    y, x = np.mgrid[0:size, 0:size]
    data = (low + (high - low) * (x + y) / (2 * (size - 1))).astype(dtype)
    data[:4, :4] = nodata
    return _write_cog(path, data, nodata=nodata)


def make_worldcover_tiles(directory: Path) -> tuple[Path, Path, Path]:
    """Two side-by-side WorldCover-like tiles plus a grid GeoJSON naming them."""
    import geopandas as gpd
    import shapely

    west, south, east, north = RAINIER
    middle = (west + east) / 2
    classes = [10, 30, 60, 80, 70]
    left = make_categorical_cog(
        directory / "worldcover_left.tif", classes, bounds=(west, south, middle, north)
    )
    right = make_categorical_cog(
        directory / "worldcover_right.tif", classes, bounds=(middle, south, east, north)
    )
    grid = gpd.GeoDataFrame(
        {"ll_tile": ["N45W123", "N45W120"]},
        geometry=[
            shapely.box(west, south, middle, north),
            shapely.box(middle, south, east, north),
        ],
        crs="EPSG:4326",
    )
    path = directory / "worldcover_grid.geojson"
    grid.to_file(path, driver="GeoJSON")
    return left, right, path


def make_all(directory: Path) -> dict[str, Path]:
    """Write every fixture into *directory* and return the paths by name."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {"dem_cog": make_dem_cog(directory / "dem.tif")}
    item = make_stac_item(out["dem_cog"], nodata=-32767.0)
    out["dem_item"] = directory / "dem_item.json"
    out["dem_item"].write_text(json.dumps(item))

    out["forest_cog"] = make_continuous_cog(
        directory / "forest_cover.tif", low=0, high=100, nodata=255
    )

    left, right, grid = make_worldcover_tiles(directory)
    out["worldcover_left"] = left
    out["worldcover_right"] = right
    out["worldcover_grid"] = grid
    worldcover_item = make_stac_item(
        left,
        item_id="ESA_WorldCover_10m_2021_v200_N45W123",
        collection="esa-worldcover",
        datetime="2021-01-01T00:00:00Z",
        asset="map",
        nodata=0,
    )
    worldcover_item["properties"]["esa_worldcover:product_version"] = "2.0.0"
    worldcover_item["properties"]["start_datetime"] = "2021-01-01T00:00:00Z"
    out["worldcover_item"] = directory / "worldcover_item.json"
    out["worldcover_item"].write_text(json.dumps(worldcover_item))
    return out


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    for name, path in make_all(target).items():
        print(f"{name}: {path}")
