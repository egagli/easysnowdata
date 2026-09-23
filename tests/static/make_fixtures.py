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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # noqa: F401 — see the module docstring
from rasterio.transform import from_bounds

RAINIER = (-121.94, 46.72, -121.54, 46.99)


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float | int | None,
    bounds: tuple[float, float, float, float] = RAINIER,
    crs: str = "EPSG:4326",
    tags: dict[str, str] | None = None,
) -> Path:
    """A tiled, DEFLATE-compressed GeoTIFF.

    A tiled GTiff rather than the COG driver, for a reason that cost an
    afternoon: **on rasterio 1.5.1 / GDAL 3.13.3 the COG driver truncates every
    written value to eight bits** while still reporting the declared dtype. A
    uint32 1000 reads back as 232, a float32 -9999 as 241, and the nodata 256
    these class rasters use as 0 — so the fixture silently stopped containing
    the fill the tests are about. rasterio 1.5.0 / GDAL 3.12.3 is unaffected,
    which is why this only appeared once CI ran the `test-py3XX` environments
    (they solve separately from `dev` and pick up the newer stack).

    Nothing is lost by the change: no test here reads an overview, the package
    never writes a raster, and the upstream archives these imitate — the
    Wrzesien Zenodo zips — hold plain GeoTIFFs anyway.
    """
    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": from_bounds(*bounds, width, height),
        "nodata": nodata,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 32,
        "blockysize": 32,
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
    return _write_tif(path, data, nodata=-32767.0)


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
    return _write_tif(path, data, nodata=nodata, bounds=bounds)


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
    return _write_tif(path, data, nodata=nodata)


def make_zipped_class_tif(
    directory: Path,
    archive: str,
    member: str,
    values: list[int],
    *,
    nodata: int = 256,
    dtype: str = "uint32",
    size: int = 32,
) -> Path:
    """A zipped class raster shaped like the Wrzesien archives (nodata 256)."""
    data = np.resize(np.array(values, dtype=dtype), (size, size))
    data[:4, :4] = nodata
    tif = directory / member
    _write_tif(tif, data, nodata=nodata)
    path = directory / archive
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, member)
    tif.unlink()
    return path


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


def _basin_frame():
    import geopandas as gpd
    import shapely

    west, south, east, north = RAINIER
    polys = [
        shapely.box(west - 0.2, south - 0.2, east + 0.2, north + 0.2),  # intersects
        shapely.box(west, south, (west + east) / 2, north),  # intersects
        shapely.box(west - 5, south - 5, west - 4, south - 4),  # far away
    ]
    return gpd.GeoDataFrame(
        {
            "HYBAS_ID": [7050000010, 7050000020, 7050000030],
            "WMOBB": [4130, 4131, 4132],
            "WMOBB_NAME": ["Nisqually", "Puyallup", "Elsewhere"],
            "SUB_AREA": [1994.5, 2455.0, 12.0],
        },
        geometry=polys,
        crs="EPSG:4326",
    )


def make_basin_zips(directory: Path) -> tuple[Path, Path]:
    """A GRDC-like zipped GeoJSON and a HydroSHEDS-like zipped shapefile."""
    frame = _basin_frame()

    geojson = directory / "wmobb_basins.json"
    frame.to_file(geojson, driver="GeoJSON")
    wmo_zip = directory / "wmobb_json.zip"
    with zipfile.ZipFile(wmo_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(geojson, "wmobb_basins.json")
    geojson.unlink()

    shape_dir = directory / "hybas"
    shape_dir.mkdir(exist_ok=True)
    stem = "hybas_na_lev05_v1c"
    frame.to_file(shape_dir / f"{stem}.shp", driver="ESRI Shapefile")
    hybas_zip = directory / "hybas_na_lev01-12_v1c.zip"
    with zipfile.ZipFile(hybas_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for part in sorted(shape_dir.glob(f"{stem}.*")):
            zf.write(part, part.name)
    for part in sorted(shape_dir.glob("*")):
        part.unlink()
    shape_dir.rmdir()
    return wmo_zip, hybas_zip


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

    out["snow_class_cog"] = make_categorical_cog(
        directory / "snow_class.tif", [1, 3, 4, 7], nodata=9
    )

    out["mountain_snow_zip"] = make_zipped_class_tif(
        directory,
        "MODIS_mtnsnow_classes.zip",
        "MODIS_mtnsnow_classes.tif",
        [0, 1, 2, 3],
    )
    out["snow_zip"] = make_zipped_class_tif(
        directory, "MODIS_snow_classes.zip", "MODIS_snow_classes.tif", [0, 1, 2, 3]
    )
    out["clouds_zip"] = make_zipped_class_tif(
        directory,
        "MODIS_clouds.zip",
        "MODISclouds.tif",
        [0, 1, 2, 3, 4, 5, 6],
        nodata=255,
        dtype="uint8",
    )

    # Natural Earth shaded relief: uint8 brightness, no nodata, one GeoTIFF per zip.
    hillshade_tif = _write_tif(
        directory / "GRAY_HR_SR_OB_DR.tif",
        np.resize(np.arange(40, 240, 5, dtype="uint8"), (32, 32)),
        nodata=None,
        tags={"TIFFTAG_SOFTWARE": "Adobe Photoshop CS5 Macintosh"},
    )
    out["hillshade_zip"] = directory / "GRAY_HR_SR_OB_DR.zip"
    with zipfile.ZipFile(out["hillshade_zip"], "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(hillshade_tif, hillshade_tif.name)
    hillshade_tif.unlink()

    out["wmo_zip"], out["hybas_zip"] = make_basin_zips(directory)

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
