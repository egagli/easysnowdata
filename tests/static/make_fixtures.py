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


def _zip_shapefile(
    frame, directory: Path, stem: str, archive: str, *, extra=()
) -> Path:
    """Write *frame* as ``<stem>.shp`` into the zip *archive* (plus *extra* members)."""
    shape_dir = directory / f"_{stem}"
    shape_dir.mkdir(exist_ok=True)
    frame.to_file(shape_dir / f"{stem}.shp", driver="ESRI Shapefile")
    path = directory / archive
    mode = "a" if path.exists() else "w"
    with zipfile.ZipFile(path, mode, zipfile.ZIP_DEFLATED) as zf:
        for part in sorted(shape_dir.glob(f"{stem}.*")):
            zf.write(part, part.name)
        for name, text in extra:
            zf.writestr(name, text)
    for part in sorted(shape_dir.glob("*")):
        part.unlink()
    shape_dir.rmdir()
    return path


def make_boundary_archives(directory: Path) -> dict[str, Path]:
    """Archives shaped like Natural Earth, Census, geoBoundaries, GMBA and RGI.

    Rainier sits in the "USA" / "Washington" / "Pierce" units, the first GMBA
    units and RGI region 2; everything else is placed so a Rainier AOI misses it.
    """
    import geopandas as gpd
    import shapely

    west, south, east, north = RAINIER
    around = shapely.box(west - 1, south - 1, east + 1, north + 1)
    north_of = shapely.box(west - 1, 49.0, east + 1, 52.0)
    far = shapely.box(5.0, 58.0, 10.0, 62.0)
    out: dict[str, Path] = {}

    countries = gpd.GeoDataFrame(
        {
            "NAME": ["United States of America", "Canada", "Norway"],
            "ADM0_A3": ["USA", "CAN", "NOR"],
            "ISO_A3": ["USA", "CAN", "-99"],
            "CONTINENT": ["North America", "North America", "Europe"],
        },
        geometry=[around, north_of, far],
        crs="EPSG:4326",
    )
    out["ne_countries_zip"] = _zip_shapefile(
        countries, directory, "ne_countries", "ne_admin_0_countries.zip"
    )
    admin1 = gpd.GeoDataFrame(
        {
            "name": ["Washington", "British Columbia", "Innlandet"],
            "adm0_a3": ["USA", "CAN", "NOR"],
        },
        geometry=[around, north_of, far],
        crs="EPSG:4326",
    )
    out["ne_states_zip"] = _zip_shapefile(
        admin1, directory, "ne_states", "ne_admin_1_states_provinces.zip"
    )
    lakes = gpd.GeoDataFrame(
        {"name": ["Mowich Lake", "Far Lake"], "scalerank": [9, 9]},
        geometry=[shapely.box(west, south, west + 0.05, south + 0.05), far],
        crs="EPSG:4326",
    )
    out["ne_lakes_zip"] = _zip_shapefile(lakes, directory, "ne_lakes", "ne_lakes.zip")

    census_states = gpd.GeoDataFrame(
        {
            "NAME": ["Washington", "Oregon"],
            "STUSPS": ["WA", "OR"],
            "STATEFP": ["53", "41"],
        },
        geometry=[around, shapely.box(west - 1, 42.0, east + 1, south - 1.5)],
        crs="EPSG:4269",
    )
    out["census_state_zip"] = _zip_shapefile(
        census_states, directory, "cb_state", "cb_state.zip"
    )
    middle = (west + east) / 2
    census_counties = gpd.GeoDataFrame(
        {
            "NAME": ["Pierce", "Lewis", "King", "Multnomah"],
            "STUSPS": ["WA", "WA", "WA", "OR"],
            "STATE_NAME": ["Washington", "Washington", "Washington", "Oregon"],
            "GEOID": ["53053", "53041", "53033", "41051"],
        },
        geometry=[
            shapely.box(west - 0.5, (south + north) / 2, middle, north + 0.5),
            shapely.box(west - 0.5, south - 0.5, east + 0.5, (south + north) / 2),
            shapely.box(west - 0.5, north + 0.6, east, north + 1.0),
            shapely.box(-123.0, 45.4, -122.2, 45.7),
        ],
        crs="EPSG:4269",
    )
    out["census_county_zip"] = _zip_shapefile(
        census_counties, directory, "cb_county", "cb_county.zip"
    )

    units = gpd.GeoDataFrame(
        {
            "shapeName": ["Washington", "Oregon"],
            "shapeISO": ["US-WA", "US-OR"],
            "shapeID": ["USA-ADM1-1", "USA-ADM1-2"],
            "shapeGroup": ["USA", "USA"],
            "shapeType": ["ADM1", "ADM1"],
        },
        geometry=[around, shapely.box(west - 1, 42.0, east + 1, south - 1.5)],
        crs="EPSG:4326",
    )
    out["geoboundaries_geojson"] = directory / "geoBoundaries-USA-ADM1.geojson"
    units.to_file(out["geoboundaries_geojson"], driver="GeoJSON")

    path_prefix = "North America > American Cordillera > Pacific Coast Ranges"
    ranges = gpd.GeoDataFrame(
        {
            "GMBA_V2_ID": [17024, 11202, 12159],
            "MapName": ["Mount Rainier Massif", "Cascade Range", "Far Range"],
            "Hier_Lvl": ["7", "4", "3"],
            "Path": [
                f"{path_prefix} > Cascade Range > Mount Rainier Massif",
                f"{path_prefix} > Cascade Range",
                "Europe > Far Range",
            ],
        },
        geometry=[shapely.box(west, south, east, north), around, far],
        crs="EPSG:4326",
    )
    for subset, suffix in (("basic", "_basic"), ("300", "_300"), ("all", "")):
        stem = f"GMBA_Inventory_v2.0_standard{suffix}"
        frame = ranges if subset == "all" else ranges.iloc[[0, 2]]
        out[f"gmba_{subset}_zip"] = _zip_shapefile(
            frame, directory, stem, f"{stem}.zip"
        )

    region_polys = [shapely.box(-170, 30, -100, 70), far]
    out["rgi7_regions_zip"] = _zip_shapefile(
        gpd.GeoDataFrame(
            {
                "o1region": ["02", "08"],
                "full_name": ["Western Canada and USA", "Scandinavia"],
            },
            geometry=region_polys,
            crs="EPSG:4326",
        ),
        directory,
        "RGI2000-v7.0-o1regions",
        "RGI2000-v7.0-regions.zip",
    )
    out["rgi6_regions_zip"] = _zip_shapefile(
        gpd.GeoDataFrame(
            {
                "RGI_CODE": [2, 8],
                "FULL_NAME": ["Western Canada and USA", "Scandinavia"],
            },
            geometry=region_polys,
            crs="EPSG:4326",
        ),
        directory,
        "00_rgi60_O1Regions",
        "00_rgi60_regions.zip",
    )
    glacier_a = shapely.box(-121.78, 46.84, -121.74, 46.88)
    glacier_b = shapely.box(-121.74, 46.84, -121.70, 46.88)
    glacier_far = shapely.box(-123.5, 48.5, -123.4, 48.6)
    out["rgi7_g02_zip"] = _zip_shapefile(
        gpd.GeoDataFrame(
            {
                "rgi_id": [
                    "RGI2000-v7.0-G-02-1",
                    "RGI2000-v7.0-G-02-2",
                    "RGI2000-v7.0-G-02-3",
                ],
                "glac_name": ["Emmons Glacier", "Winthrop Glacier", "Far Glacier"],
                "area_km2": [10.59, 9.19, 0.5],
                "o1region": ["02", "02", "02"],
                "is_rgi6": [1, 1, 1],
                "src_date": ["1970-09-01", "1970-09-01", "1970-09-01"],
            },
            geometry=[
                shapely.force_3d(g, 1500.0) for g in (glacier_a, glacier_b, glacier_far)
            ],
            crs="EPSG:4326",
        ),
        directory,
        "RGI2000-v7.0-G-02_western_canada_usa",
        "RGI2000-v7.0-G-02_western_canada_usa.zip",
    )
    out["rgi7_c02_zip"] = _zip_shapefile(
        gpd.GeoDataFrame(
            {
                "rgi_id": ["RGI2000-v7.0-C-02-1"],
                "area_km2": [19.78],
                "o1region": ["02"],
            },
            geometry=[shapely.force_3d(glacier_a.union(glacier_b), 1500.0)],
            crs="EPSG:4326",
        ),
        directory,
        "RGI2000-v7.0-C-02_western_canada_usa",
        "RGI2000-v7.0-C-02_western_canada_usa.zip",
    )
    out["rgi6_02_zip"] = _zip_shapefile(
        gpd.GeoDataFrame(
            {
                "RGIId": ["RGI60-02.1", "RGI60-02.2", "RGI60-02.3"],
                "Name": ["Emmons Glacier WA", "Winthrop Glacier WA", "WA"],
                "Area": [10.594, 9.195, 0.01],
                "O1Region": ["2", "2", "2"],
            },
            geometry=[
                glacier_a,
                glacier_b,
                shapely.box(-121.70, 46.84, -121.69, 46.85),
            ],
            crs="EPSG:4326",
        ),
        directory,
        "02_rgi60_WesternCanadaUS",
        "02_rgi60_WesternCanadaUS.zip",
    )
    return out


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
    # Vector archives last: geopandas is imported by now, and nothing below
    # writes a raster (see the module docstring).
    out.update(make_boundary_archives(directory))
    return out


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    for name, path in make_all(target).items():
        print(f"{name}: {path}")
