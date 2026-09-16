"""The per-theme output-contract helpers (§2.5) and the theme exports.

Phase 2a runs in parallel with Phase 2b, so each theme package carries its own
copy of these helpers; this module checks every copy behaves the same, which
is what lets the merge session fold them into one.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.hydro import _common as hydro_common
from easysnowdata.land import _common as land_common
from easysnowdata.snow import _common as snow_common
from easysnowdata.terrain import _common as terrain_common

COMMONS = (terrain_common, land_common, snow_common, hydro_common)


def _array(values, dims=("y", "x"), crs="EPSG:32610", dtype="uint8"):
    ny, nx = np.asarray(values).shape
    da = xr.DataArray(
        np.asarray(values, dtype=dtype),
        dims=dims,
        coords={
            dims[0]: np.linspace(5_200_000, 5_199_000, ny),
            dims[1]: np.linspace(600_000, 601_000, nx),
        },
    )
    return da.rio.write_crs(crs)


@pytest.mark.parametrize("common", COMMONS, ids=lambda m: m.__name__.split(".")[1])
class TestCommon:
    def test_resolve(self, common):
        product, source = common.resolve("copernicus-dem")
        assert product.id == "copernicus-dem" and source.id == "planetary-computer"
        assert common.resolve("copernicus-dem", "earth-search")[1].id == "earth-search"
        with pytest.raises(ValueError, match="no source"):
            common.resolve("copernicus-dem", "nope")

    def test_attrs_for(self, common):
        product, source = common.resolve("copernicus-dem")
        attrs = common.attrs_for(product, source, units="m", skipped=None)
        assert attrs["product_id"] == "copernicus-dem"
        assert attrs["source"] == "planetary-computer"
        assert attrs["source_url"] and attrs["data_citation"] and attrs["license"]
        assert attrs["easysnowdata_version"] == esd.__version__
        assert attrs["doi"] == "10.5069/G9028PQB" and attrs["units"] == "m"
        assert "skipped" not in attrs
        assert all(isinstance(v, (str, int, float)) for v in attrs.values())

    def test_variable_lookup(self, common):
        product = catalog.get("mountain-snow-mask")
        assert common.variable(product, "clouds").units == "1"
        with pytest.raises(KeyError, match="no variable"):
            common.variable(product, "nope")

    def test_standardize_names_projected_dims(self, common):
        da = _array([[1, 2], [3, 4]], dims=("latitude", "longitude"))
        out = common.standardize(da, crs="EPSG:32610")
        assert out.dims == ("y", "x")
        assert out.rio.crs.to_epsg() == 32610 and out.odc.crs.epsg == 32610

    def test_standardize_names_geographic_dims(self, common):
        da = _array([[1, 2], [3, 4]], crs="EPSG:4326")
        out = common.standardize(da)
        assert out.dims == ("latitude", "longitude")
        assert out.rio.crs.to_epsg() == 4326 and out.odc.crs.epsg == 4326

    def test_standardize_without_a_crs_raises(self, common):
        da = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        with pytest.raises(ValueError, match="no CRS"):
            common.standardize(da)

    def test_apply_nodata_categorical_and_continuous(self, common):
        da = _array([[1, 2], [3, 255]])
        kept = common.apply_nodata(da, 255, mask=False)
        assert kept.dtype == "uint8" and kept.rio.nodata == 255
        masked = common.apply_nodata(da, 255, mask=True)
        assert masked.dtype == "float32" and masked.rio.encoded_nodata == 255
        assert bool(masked.isnull().any())
        floats = _array([[1.0, -9999.0]], dtype="float32")
        assert common.apply_nodata(floats, -9999, mask=True).dtype == "float32"
        assert common.apply_nodata(da, None, mask=True) is da

    def test_set_variable_flags(self, common):
        product = catalog.get("mountain-snow-mask")
        flagged = common.set_variable_flags(_array([[0, 3]]), product, "snow")
        assert flagged.attrs["flag_values"] == [0, 1, 2, 3, 255]
        assert flagged.attrs["long_name"] == "snow class"
        plain = common.set_variable_flags(_array([[0, 3]]), product, "clouds")
        assert "flag_values" not in plain.attrs

    def test_default_sentinel(self, common):
        assert common.DEFAULT is not None and common.DEFAULT is not True


def test_theme_packages_are_exported():
    for theme, modules in (
        (esd.terrain, ("dem", "chili")),
        (esd.land, ("landcover", "nlcd", "forest_cover")),
        (esd.snow, ("snow_classification", "mountain_snow_mask")),
        (esd.hydro, ("basins",)),
    ):
        assert set(modules) <= set(theme.__all__)
        for name in modules:
            assert hasattr(theme, name)
        assert theme.__doc__ and "Products" in theme.__doc__
    assert {"terrain", "land", "snow", "hydro"} <= set(esd.__all__)


def test_every_migrated_product_points_at_its_theme_module():
    migrated = {
        "copernicus-dem": "easysnowdata.terrain.dem.load",
        "chili": "easysnowdata.terrain.chili.load",
        "esa-worldcover": "easysnowdata.land.landcover.load",
        "nlcd": "easysnowdata.land.nlcd.load",
        "forest-cover-fraction": "easysnowdata.land.forest_cover.load",
        "snow-classification": "easysnowdata.snow.snow_classification.load",
        "mountain-snow-mask": "easysnowdata.snow.mountain_snow_mask.load",
        "huc": "easysnowdata.hydro.basins.huc",
        "hydrobasins": "easysnowdata.hydro.basins.hydrobasins",
        "grdc-major-river-basins": "easysnowdata.hydro.basins.grdc_major",
        "grdc-wmo-basins": "easysnowdata.hydro.basins.grdc_wmo",
    }
    for product_id, loader in migrated.items():
        product = catalog.get(product_id)
        assert product.loader == loader
        assert callable(product.resolve_loader())
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_old_names_still_work_and_warn():
    from easysnowdata import _deprecation

    old_names = {
        esd.topography.get_copernicus_dem: "easysnowdata.terrain.dem.load",
        esd.topography.get_chili: "easysnowdata.terrain.chili.load",
        esd.remote_sensing.get_esa_worldcover: "easysnowdata.land.landcover.load",
        esd.remote_sensing.get_nlcd_landcover: "easysnowdata.land.nlcd.load",
        esd.remote_sensing.get_forest_cover_fraction: (
            "easysnowdata.land.forest_cover.load"
        ),
        esd.remote_sensing.get_seasonal_snow_classification: (
            "easysnowdata.snow.snow_classification.load"
        ),
        esd.remote_sensing.get_seasonal_mountain_snow_mask: (
            "easysnowdata.snow.mountain_snow_mask.load"
        ),
        esd.hydroclimatology.get_huc_geometries: "easysnowdata.hydro.basins.huc",
        esd.hydroclimatology.get_hydroBASINS: "easysnowdata.hydro.basins.hydrobasins",
        esd.hydroclimatology.get_grdc_major_river_basins_of_the_world: (
            "easysnowdata.hydro.basins.grdc_major"
        ),
        esd.hydroclimatology.get_grdc_wmo_basins: "easysnowdata.hydro.basins.grdc_wmo",
    }
    _deprecation.reset_warnings()
    for function, replacement in old_names.items():
        assert replacement in function.__deprecated__
        assert "deprecated" in (function.__doc__ or "")
