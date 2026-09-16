"""Synthetic fixtures for the time-series product tests (offline tier).

Run as a script (``python tests/timeseries/fixtures/make_fixtures.py <dir>``)
or through the ``ts_fixtures`` fixture, which runs it in a subprocess. Every
file is a few kilobytes and covers the Mount Rainier test box
``(-121.94, 46.72, -121.54, 46.99)``:

* ``reanalysis.zarr`` — an ARCO-ERA5 look-alike (0–360° longitudes, ERA5/ERA5T
  validity attrs)
* ``koppen.zip`` — the figshare archive layout with one historical and one
  projected period at 1°
* ``SNODAS_20240315.tar`` — one masked-grid day with SWE and snow-depth pairs
  in the NOHRSC ``.dat.gz`` + ``.txt.gz`` format
* ``MOD10A1.A2023060.h09v04…tif`` / ``…h10v04…tif`` — two adjacent MODIS-like
  sinusoidal tiles (GeoTIFF stand-ins for the HDF-EOS2 granules)
* ``VNP10A1F.A2023060.h09v04….h5`` — an HDF-EOS5 stand-in with the VIIRS
  group layout and ``StructMetadata.0``
* ``WUS_UCLA_SR_…_SWE_SCA_POST.nc`` (two water years) — the UCLA reanalysis
  layout (``Day``, ``Stats``, ``Latitude``, ``Longitude``)
* ``planet_scene_clip.tif`` + ``planet_udm2_clip.tif`` — a synthetic 4-band
  PlanetScope order delivery with its UDM2 mask (no real imagery)
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio  # noqa: F401 — import before geopandas/pyproj (see tests/fixtures)
from rasterio.transform import from_bounds, from_origin

RAINIER = (-121.94, 46.72, -121.54, 46.99)
SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"


def _write_tif(path: Path, data: np.ndarray, transform, crs, nodata, **tags) -> Path:
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": data.shape[-1],
        "height": data.shape[-2],
        "count": data.shape[0] if data.ndim == 3 else 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data if data.ndim == 3 else data[None])
        if tags:
            dst.update_tags(**tags)
    return path


def make_reanalysis_zarr(path: Path) -> Path:
    import pandas as pd
    import xarray as xr

    lat = np.arange(90, -90.01, -0.25)
    lon = np.arange(0, 360, 0.25)
    time = pd.date_range("2020-01-01", periods=6, freq="h")
    rng = np.random.default_rng(1)
    shape = (time.size, lat.size, lon.size)
    ds = xr.Dataset(
        {
            "2m_temperature": (
                ("time", "latitude", "longitude"),
                (250 + 30 * rng.random(shape)).astype("float32"),
                {"units": "K", "long_name": "2 metre temperature", "short_name": "t2m"},
            ),
            "snow_depth": (
                ("time", "latitude", "longitude"),
                rng.random(shape).astype("float32"),
                {"units": "m of water equivalent", "long_name": "Snow depth"},
            ),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={
            "valid_time_start": "2020-01-01",
            "valid_time_stop": "2020-01-01",
            "valid_time_stop_era5t": "2020-01-01",
            "last_updated": "2020-01-02 00:00:00+00:00",
        },
    )
    ds = ds.chunk({"time": 1, "latitude": 721, "longitude": 1440})
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def make_koppen_zip(path: Path) -> Path:
    rng = np.random.default_rng(2)
    transform = from_bounds(-180, -90, 180, 90, 360, 180)
    buffers: dict[str, bytes] = {}
    for member, seed in (
        ("1991_2020/koppen_geiger_1p0.tif", 2),
        ("2071_2099/ssp245/koppen_geiger_1p0.tif", 3),
    ):
        data = np.random.default_rng(seed).integers(1, 31, (180, 360), dtype="uint8")
        data[:, :5] = 0  # ocean / nodata strip
        tmp = path.parent / f"_kg_{seed}.tif"
        _write_tif(tmp, data, transform, "EPSG:4326", 0)
        buffers[member] = tmp.read_bytes()
        tmp.unlink()
    del rng
    with zipfile.ZipFile(path, "w") as z:
        for member, payload in buffers.items():
            z.writestr(member, payload)
        z.writestr("legend.txt", "1: Af Tropical, rainforest [0 0 255]\n")
    return path


_SNODAS_HEADER = """Format version: NOHRSC GIS/RS raster file v1.1
Data source: RUC2, NESDIS, etc.
Created by module: sm_products
Description: {description}
Data units: {units}
Product code: {code}
Data file pathname: {datafile}
Data type: integer
Data bytes per pixel: 2
Data intercept: 0.00000000000000
Data slope: 1.00000000000000
Minimum data value: 0.00000000000000
Maximum data value: 32767.0000000000
No data value: -9999.00000000000
Number of columns: {ncols}
Number of rows: {nrows}
Geographically corrected: yes
Projected: no
Horizontal datum: WGS84
Benchmark column: 0
Benchmark row: 0
Benchmark x-axis coordinate: {bx:.12f}
Benchmark y-axis coordinate: {by:.12f}
X-axis resolution: {res:.17f}
Y-axis resolution: {res:.17f}
X-axis offset: {half:.17f}
Y-axis offset: {half:.17f}
Minimum x-axis coordinate: {minx:.12f}
Maximum x-axis coordinate: {maxx:.12f}
Minimum y-axis coordinate: {miny:.12f}
Maximum y-axis coordinate: {maxy:.12f}
Start year: 2024
Start month: 3
Start day: 15
Start hour: 6
Start minute: 0
Start second: 0
Stop year: 2024
Stop month: 3
Stop day: 15
Stop hour: 6
Stop minute: 0
Stop second: 0
Compressed: no
"""


def make_snodas_tar(path: Path) -> Path:
    """One SNODAS day on a 60×40 masked-grid subset that covers Rainier."""
    res = 1 / 120
    ncols, nrows = 60, 40
    minx, maxy = -122.10, 47.10
    maxx, miny = minx + ncols * res, maxy - nrows * res
    rng = np.random.default_rng(4)
    members = {
        "us_ssmv11034tS__T0001TTNATS2024031505HP001": (
            "Modeled snow water equivalent, total of snow layers",
            "Meters / 1000.000000",
            "559505818",
            rng.integers(0, 1500, (nrows, ncols)),
        ),
        "us_ssmv11036tS__T0001TTNATS2024031505HP001": (
            "Modeled snow layer thickness, total of snow layers",
            "Meters / 1000.000000",
            "559505819",
            rng.integers(0, 4000, (nrows, ncols)),
        ),
    }
    with tarfile.open(path, "w") as tar:
        for stem, (description, units, code, values) in members.items():
            values = values.astype(">i2")
            values[:3, :3] = -9999  # a nodata corner
            header = _SNODAS_HEADER.format(
                description=description,
                units=units,
                code=code,
                datafile=f"{stem}.dat",
                ncols=ncols,
                nrows=nrows,
                bx=minx + res / 2,
                by=maxy - res / 2,
                res=res,
                half=res / 2,
                minx=minx,
                maxx=maxx,
                miny=miny,
                maxy=maxy,
            )
            for suffix, payload in (
                (".txt.gz", gzip.compress(header.encode())),
                (".dat.gz", gzip.compress(values.tobytes())),
            ):
                info = tarfile.TarInfo(f"{stem}{suffix}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return path


def _sinusoidal_bounds():
    import geopandas as gpd
    import shapely

    box = gpd.GeoSeries([shapely.box(*RAINIER)], crs="EPSG:4326").to_crs(SINUSOIDAL)
    return box.total_bounds


def make_modis_tiles(directory: Path) -> list[Path]:
    """Two adjacent 500 m sinusoidal tiles split down the middle of the AOI."""
    west, south, east, north = _sinusoidal_bounds()
    res = 463.31271653
    mid = west + ((east - west) // (2 * res)) * res
    paths = []
    rng = np.random.default_rng(5)
    for tile, (x0, x1) in (
        ("h09v04", (west - 3 * res, mid)),
        ("h10v04", (mid, east + 3 * res)),
    ):
        ncols = int(round((x1 - x0) / res))
        nrows = int(round((north - south) / res)) + 6
        transform = from_origin(x0, north + 3 * res, res, res)
        for product, var_values in (
            ("MOD10A1", rng.integers(0, 101, (nrows, ncols), dtype="uint8")),
            ("MOD10A2", rng.choice([25, 50, 200], (nrows, ncols)).astype("uint8")),
        ):
            data = var_values.copy()
            data[:2, :2] = 255 if product == "MOD10A1" else 255
            if product == "MOD10A1":
                data[2:4, 2:4] = 250  # cloud
            name = f"{product}.A2023060.{tile}.061.2023062000000.tif"
            paths.append(_write_tif(directory / name, data, transform, SINUSOIDAL, 255))
    return paths


def make_viirs_h5(path: Path) -> Path:
    import h5py

    west, south, east, north = _sinusoidal_bounds()
    res = 375.0
    ncols, nrows = int((east - west) / res) + 6, int((north - south) / res) + 6
    ulx, uly = west - 3 * res, north + 3 * res
    lrx, lry = ulx + ncols * res, uly - nrows * res
    rng = np.random.default_rng(6)
    data = rng.integers(0, 101, (nrows, ncols), dtype="uint8")
    data[:2, :2] = 255
    data[2:4, 2:4] = 250
    struct = (
        'GROUP=GridStructure\n\tGROUP=GRID_1\n\t\tGridName="VIIRS_Grid_IMG_2D"\n'
        f"\t\tXDim={ncols}\n\t\tYDim={nrows}\n"
        f"\t\tUpperLeftPointMtrs=({ulx:.6f},{uly:.6f})\n"
        f"\t\tLowerRightMtrs=({lrx:.6f},{lry:.6f})\n"
        "\t\tProjection=GCTP_SNSOID\n"
        "\t\tProjParams=(6371007.181000,0,0,0,0,0,0,0,0,0,0,0,0)\n"
        "\tEND_GROUP=GRID_1\nEND_GROUP=GridStructure\nEND\n"
    )
    with h5py.File(path, "w") as f:
        group = f.create_group("HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields")
        for name in ("CGF_NDSI_Snow_Cover", "Cloud_Persistence", "Basic_QA"):
            ds = group.create_dataset(name, data=data, chunks=True, compression="gzip")
            ds.attrs["_FillValue"] = np.uint8(255)
            ds.attrs["long_name"] = np.bytes_(name.replace("_", " "))
        f.create_group("HDFEOS INFORMATION").create_dataset(
            "StructMetadata.0", data=np.bytes_(struct.encode())
        )
    return path


def make_ucla_nc(directory: Path) -> list[Path]:
    import pandas as pd
    import xarray as xr

    lat = np.arange(47 - 1 / 240, 46, -1 / 120)[:120]
    lon = np.arange(-122 + 1 / 240, -121, 1 / 120)[:120]
    paths = []
    for wy in (2019, 2020):
        days = pd.date_range(f"{wy}-10-01", f"{wy + 1}-09-30")
        rng = np.random.default_rng(wy)
        base = rng.random((days.size, 5, lat.size, lon.size), dtype="float32")
        base[:, 3] = base[:, 2] * 0.8  # 25th percentile below the median
        base[:, 4] = base[:, 2] * 1.2  # 75th percentile above it
        ds = xr.Dataset(
            {
                "SWE_Post": (("Day", "Stats", "Latitude", "Longitude"), base),
                "SCA_Post": (("Day", "Stats", "Latitude", "Longitude"), base / 2),
            },
            coords={"Latitude": lat, "Longitude": lon},
        )
        ds["SWE_Post"].attrs = {"units": "m", "long_name": "posterior SWE"}
        ds["SCA_Post"].attrs = {"units": "1", "long_name": "posterior SCA"}
        name = f"WUS_UCLA_SR_v01_N46_0W122_0_agg_16_WY{wy}_{str(wy + 1)[-2:]}_SWE_SCA_POST.nc"
        ds.isel(Day=slice(0, 12)).to_netcdf(
            directory / name,
            engine="h5netcdf",
            encoding={
                "SWE_Post": {"_FillValue": -999.0},
                "SCA_Post": {"_FillValue": -999.0},
            },
        )
        paths.append(directory / name)
    return paths


def make_planet_scene(directory: Path) -> list[Path]:
    import geopandas as gpd
    import shapely

    west, south, east, north = (
        gpd.GeoSeries([shapely.box(*RAINIER)], crs="EPSG:4326")
        .to_crs("EPSG:32610")
        .total_bounds
    )
    ncols, nrows = 48, 40
    transform = from_bounds(
        west, north - nrows * 3.0, west + ncols * 3.0, north, ncols, nrows
    )
    rng = np.random.default_rng(7)
    scene = rng.integers(1, 6000, (4, nrows, ncols), dtype="uint16")
    scene[:, :3, :3] = 0
    udm2 = np.zeros((8, nrows, ncols), dtype="uint8")
    udm2[0] = 1  # clear
    udm2[1, 10:14, 10:14] = 1  # snow
    udm2[0, 10:14, 10:14] = 0
    udm2[5, 20:24, 20:24] = 1  # cloud
    udm2[0, 20:24, 20:24] = 0
    udm2[6] = 90  # confidence
    udm2[7, :3, :3] = 1  # unusable bit 0 (blackfill)
    return [
        _write_tif(
            directory / "20230701_183012_12_2465_3B_AnalyticMS_clip.tif",
            scene,
            transform,
            "EPSG:32610",
            0,
        ),
        _write_tif(
            directory / "20230701_183012_12_2465_3B_udm2_clip.tif",
            udm2,
            transform,
            "EPSG:32610",
            None,
        ),
    ]


def make_all(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {
        "reanalysis": make_reanalysis_zarr(directory / "reanalysis.zarr"),
        "koppen_zip": make_koppen_zip(directory / "koppen.zip"),
        "snodas_tar": make_snodas_tar(directory / "SNODAS_20240315.tar"),
        "viirs_h5": make_viirs_h5(
            directory / "VNP10A1F.A2023060.h09v04.002.2023062000000.h5"
        ),
    }
    modis_dir = directory / "modis"
    modis_dir.mkdir(exist_ok=True)
    for i, p in enumerate(make_modis_tiles(modis_dir)):
        out[f"modis_{i}"] = p
    out["modis_dir"] = modis_dir
    ucla_dir = directory / "ucla"
    ucla_dir.mkdir(exist_ok=True)
    for i, p in enumerate(make_ucla_nc(ucla_dir)):
        out[f"ucla_{i}"] = p
    out["ucla_dir"] = ucla_dir
    planet_dir = directory / "planet"
    planet_dir.mkdir(exist_ok=True)
    scene, udm2 = make_planet_scene(planet_dir)
    out["planet_scene"] = scene
    out["planet_udm2"] = udm2
    out["planet_dir"] = planet_dir
    return out


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    for name, path in make_all(target).items():
        print(f"{name}: {path}")
