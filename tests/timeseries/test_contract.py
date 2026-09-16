"""Unit tests for easysnowdata.processing.contract (offline)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_bounds

import easysnowdata as esd
from easysnowdata.catalog import Product, Source, Variable
from easysnowdata.processing import contract


def _product(**variables):
    return Product(
        id="test-product",
        theme="snow",
        title="Test product",
        description="d",
        sources=(
            Source(
                "src",
                "raster_http",
                "https://example.com/x.tif",
                title="Example host",
                health=lambda: None,
            ),
        ),
        citation="Someone (2024)",
        license="CC0",
        doi="10.1/x",
        loader="easysnowdata.aoi.parse_aoi",
        variables=tuple(variables.values()),
    )


def _grid(crs="EPSG:32610"):
    ny, nx = 4, 5
    if crs == "EPSG:4326":
        x = np.linspace(-121.9, -121.6, nx)
        y = np.linspace(46.95, 46.75, ny)
        dims = ("y", "x")
    else:
        x = np.linspace(580000, 580400, nx)
        y = np.linspace(5180400, 5180000, ny)
        dims = ("y", "x")
    return dims, {"y": y, "x": x}


def test_write_crs_renames_geographic_dims_and_sets_both_accessors():
    dims, coords = _grid("EPSG:4326")
    da = xr.DataArray(np.zeros((4, 5)), dims=dims, coords=coords)
    out = contract.write_crs(da, "EPSG:4326")
    assert set(out.dims) == {"latitude", "longitude"}
    assert out.rio.crs.to_epsg() == 4326 and out.odc.crs.epsg == 4326
    dims, coords = _grid()
    da = xr.DataArray(
        np.zeros((4, 5)),
        dims=("latitude", "longitude"),
        coords={"latitude": coords["y"], "longitude": coords["x"]},
    )
    out = contract.write_crs(da, "EPSG:32610")
    assert set(out.dims) == {"y", "x"} and out.rio.crs.to_epsg() == 32610
    assert out.odc.geobox.shape == (4, 5)
    assert contract.geographic_dims(xr.DataArray([1.0], dims=["z"])).dims == ("z",)


def test_mask_continuous_and_categorical_nodata():
    dims, coords = _grid()
    data = np.arange(20, dtype="int16").reshape(4, 5)
    da = xr.DataArray(
        data, dims=dims, coords=coords, attrs={"nodata": 0, "units": "mm"}
    )
    da = da.rio.write_crs("EPSG:32610")
    masked = contract.mask_continuous(da)
    assert masked.dtype == np.float32 and np.isnan(masked.values[0, 0])
    assert masked.rio.encoded_nodata == 0 and np.isnan(masked.rio.nodata)
    assert "nodata" not in masked.attrs and masked.attrs["units"] == "mm"
    # explicit nodata and float input
    f = contract.mask_continuous(da.astype("float64"), nodata=5)
    assert np.isnan(f.values[1, 0]) and f.dtype == np.float64
    # nothing known → unchanged values
    plain = xr.DataArray(data, dims=dims, coords=coords)
    assert not np.isnan(contract.mask_continuous(plain).values).any()
    cat = contract.set_categorical_nodata(da, 255)
    assert (
        cat.rio.nodata == 255 and cat.attrs["nodata"] == 255 and cat.dtype == np.int16
    )


def test_apply_variables_and_finalize():
    dims, coords = _grid()
    rng = np.random.default_rng(0)
    swe = xr.DataArray(
        rng.integers(0, 100, (2, 4, 5)).astype("int16"),
        dims=("time", *dims),
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            **coords,
        },
    )
    swe.values[0, 0, 0] = -9999
    cls = xr.DataArray(
        rng.integers(0, 3, (2, 4, 5)).astype("uint8"),
        dims=("time", *dims),
        coords=swe.coords,
    )
    cls.values[0, 0, 0] = 255
    ds = xr.Dataset({"SWE": swe, "snow_class": cls}).transpose("y", "x", "time")
    product = _product(
        swe=Variable(
            "SWE",
            units="m",
            dtype="int16",
            nodata=-9999,
            long_name="snow water equivalent",
        ),
        cls=Variable(
            "snow_class",
            dtype="uint8",
            nodata=255,
            long_name="class",
            flag_values=(0, 1, 2, 255),
            flag_meanings=("no snow", "snow", "cloud", "fill"),
            flag_colors=("#000", "#fff", "#ccc", "#f00"),
        ),
        missing=Variable("not_there", units="1"),
    )
    out = contract.finalize(
        ds,
        product,
        product.default_source,
        crs="EPSG:32610",
        source_url="https://example.com",
    )
    assert out["SWE"].dims == ("time", "y", "x") and out["snow_class"].dims == (
        "time",
        "y",
        "x",
    )
    assert (
        np.isnan(out["SWE"].values[0, 0, 0]) and out["SWE"].rio.encoded_nodata == -9999
    )
    assert (
        out["SWE"].attrs["units"] == "m"
        and out["SWE"].attrs["long_name"] == "snow water equivalent"
    )
    assert (
        out["snow_class"].dtype == np.uint8 and out["snow_class"].values[0, 0, 0] == 255
    )
    assert out["snow_class"].rio.nodata == 255
    assert out["snow_class"].attrs["flag_meanings"] == "no_snow snow cloud fill"
    assert out["snow_class"].attrs["flag_values"] == [0, 1, 2, 255]
    for key in contract.PROVENANCE_KEYS:
        assert key in out.attrs, key
    assert out.attrs["source"] == "Example host" and out.attrs["source_id"] == "src"
    assert (
        out.attrs["source_url"] == "https://example.com"
        and out.attrs["doi"] == "10.1/x"
    )
    assert out.attrs["easysnowdata_version"] == esd.__version__
    assert out.rio.crs.to_epsg() == 32610 and out.odc.crs.epsg == 32610
    assert all(not callable(v) for v in out.attrs.values())

    # mask=False keeps the raw integers, mask=True masks the categorical too
    raw = contract.finalize(
        ds, product, product.default_source, crs="EPSG:32610", mask=False
    )
    assert raw["SWE"].dtype == np.int16 and raw["SWE"].rio.nodata == -9999
    both = contract.finalize(
        ds, product, product.default_source, crs="EPSG:32610", mask=True
    )
    assert np.isnan(both["snow_class"].values[0, 0, 0])

    # default source_url falls back to a URL-like location; extra attrs win
    extra = contract.finalize(
        ds,
        product,
        product.default_source,
        crs="EPSG:32610",
        attrs={"note": "x", "skip": None},
    )
    assert (
        extra.attrs["source_url"] == "https://example.com/x.tif"
        and extra.attrs["note"] == "x"
    )
    assert "skip" not in extra.attrs
    # a DataArray without a CRS or spatial dims passes through untouched
    da = contract.finalize(
        xr.DataArray([1.0], dims=["z"]), product, product.default_source
    )
    assert da.attrs["product_id"] == "test-product"


def test_provenance_from_plain_source_string():
    product = _product()
    attrs = contract.provenance(product, "nowhere", extra="y", nothing=None)
    assert (
        attrs["source"] == "nowhere"
        and attrs["source_url"] == ""
        and attrs["extra"] == "y"
    )
    assert "nothing" not in attrs


def test_fixture_generator_writes_every_fixture(ts_fixtures):
    import rioxarray  # noqa: F401

    assert {
        "reanalysis",
        "koppen_zip",
        "snodas_tar",
        "viirs_h5",
        "modis_0",
        "ucla_0",
        "planet_scene",
    } <= set(ts_fixtures)
    tif = rioxarray.open_rasterio(ts_fixtures["modis_0"])
    assert tif.rio.crs is not None and tif.dtype == np.uint8
    with pytest.raises(KeyError):
        ts_fixtures["nope"]
    assert from_bounds(0, 0, 1, 1, 1, 1) is not None
