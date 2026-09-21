"""How the static product modules meet the output contract (§2.5).

The per-theme helper copies Phase 2a carried while it ran in parallel with
Phase 2b are gone: every loader now uses :mod:`easysnowdata.processing.contract`
and :mod:`easysnowdata.catalog._access`, which ``tests/timeseries/test_contract.py``
covers directly. What is left here is what those modules promise as a set.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.processing import contract

STATIC_MODULES = (
    "easysnowdata.terrain.dem",
    "easysnowdata.terrain.chili",
    "easysnowdata.land.landcover",
    "easysnowdata.land.nlcd",
    "easysnowdata.land.forest_cover",
    "easysnowdata.snow.snow_classification",
    "easysnowdata.snow.mountain_snow_mask",
    "easysnowdata.hydro.basins",
)


def test_no_theme_ships_its_own_contract_helper():
    """The Phase 2a duplicates are folded into processing.contract."""
    for theme in ("terrain", "land", "snow", "hydro"):
        with pytest.raises(ImportError):
            importlib.import_module(f"easysnowdata.{theme}._common")


def test_loaders_use_the_shared_helpers():
    for name in STATIC_MODULES:
        source = importlib.import_module(name).__file__
        text = pathlib.Path(source).read_text()
        assert "_common" not in text, name
        assert "contract." in text or "resolve_source" in text, name


def test_source_attr_is_the_id_so_it_round_trips():
    """`source` is what load(source=...) takes; the title is separate."""
    product = catalog.get("esa-worldcover")
    for src in product.sources:
        attrs = contract.provenance(product, src)
        assert attrs["source"] == src.id
        assert attrs["source_title"] == src.title
        assert product.source(attrs["source"]) is src


def test_default_sentinel_is_shared():
    assert contract.DEFAULT is not None and contract.DEFAULT is not True


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
        # The theme docstring names every product module it exports.
        assert theme.__doc__
        for name in theme.__all__:
            assert name in theme.__doc__
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
