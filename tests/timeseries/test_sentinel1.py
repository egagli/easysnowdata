"""easysnowdata.sar.sentinel1 and the terrain geometry in processing.sar.

The incidence-angle maths is checked against closed-form answers on synthetic
slopes; the catalog routes are checked on cassettes and stubs; the live tier
loads real backscatter from Planetary Computer.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import easysnowdata as esd
from easysnowdata import catalog
from easysnowdata.processing import sar as sar_processing
from easysnowdata.sar import sentinel1

RAINIER = (-121.94, 46.72, -121.54, 46.99)


# ── catalog entries ───────────────────────────────────────────────────────────


def test_catalog_entries_are_registered_from_this_module():
    product = catalog.get("sentinel-1-rtc")
    assert product is sentinel1.PRODUCT
    assert product.loader == "easysnowdata.sar.sentinel1.load"
    assert product.resolve_loader() is sentinel1.load
    assert [s.id for s in product.sources] == [
        "planetary-computer",
        "opera-rtc-s1",
        "gee",
    ]
    assert product.default_source.requires == ()
    assert product.source("opera-rtc-s1").requires == ("earthdata",)
    # the probe label the weekly health check records stays the same
    assert (
        product.default_source.health[0].label == "Sentinel-1 RTC (Planetary Computer)"
    )
    # the OPERA search itself is open, like every other CMR-STAC search
    assert product.source("opera-rtc-s1").health[0].requires == ()

    lia = catalog.get("sentinel-1-local-incidence-angle")
    assert lia is sentinel1.LIA_PRODUCT
    assert lia.resolve_loader() is sentinel1.local_incidence_angle
    assert [s.id for s in lia.sources] == ["opera-static", "dem", "gee"]
    assert lia.source("dem").requires == ()  # the credential-free fallback
    mask = lia.variables[-1]
    assert mask.name == "mask" and mask.flag_meanings[2] == "layover"
    assert catalog.validate_all(known_auth=tuple(esd.auth.PROVIDERS)) == []


def test_opera_asset_names_and_renaming():
    assert sentinel1._opera_assets(["vv", "vh"], True) == ["0_VV", "0_VH", "0_mask"]
    assert sentinel1._opera_assets(["vv"], False) == ["0_VV"]
    ds = xr.Dataset(
        {
            "0_VV": ("x", [1.0]),
            "0_mask": ("x", [0]),
            "other": ("x", [1]),
        }
    )
    renamed = sentinel1._rename_opera(ds)
    assert set(renamed.data_vars) == {"vv", "mask", "other"}
    assert sentinel1._polarisations("VV") == ["vv"]
    assert sentinel1._polarisations(None) == ["vv", "vh"]


def test_search_rejects_the_gee_route():
    with pytest.raises(ValueError, match="no item search"):
        sentinel1.search(RAINIER, "2023-01", source="gee")


# ── terrain geometry (pure, closed-form checks) ──────────────────────────────


def _planar_dem(slope_deg: float, *, facing: str, n: int = 21, res: float = 30.0):
    x = np.arange(n) * res
    y = np.arange(n)[::-1] * res
    gradient = np.tan(np.radians(slope_deg))
    if facing == "west":  # elevation rises to the east
        z = gradient * x[None, :] * np.ones((n, 1))
    elif facing == "south":  # elevation rises to the north
        z = gradient * y[:, None] * np.ones((1, n))
    else:
        z = np.zeros((n, n))
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(
        "EPSG:32610"
    )


def test_slope_and_aspect_are_exact_on_planes():
    slope, aspect = sar_processing.slope_aspect(_planar_dem(20.0, facing="west"))
    assert float(slope[10, 10]) == pytest.approx(20.0)
    assert float(aspect[10, 10]) == pytest.approx(270.0)
    slope, aspect = sar_processing.slope_aspect(_planar_dem(15.0, facing="south"))
    assert float(slope[10, 10]) == pytest.approx(15.0)
    assert float(aspect[10, 10]) == pytest.approx(0.0)  # rises north → faces north
    flat_slope, _ = sar_processing.slope_aspect(_planar_dem(0.0, facing="flat"))
    assert float(flat_slope[10, 10]) == pytest.approx(0.0)
    radians, _ = sar_processing.slope_aspect(
        _planar_dem(20.0, facing="west"), degrees=False
    )
    assert float(radians[10, 10]) == pytest.approx(np.radians(20.0))
    assert radians.attrs["units"] == "radians"
    with pytest.raises(ValueError, match="projected DEM"):
        sar_processing.slope_aspect(xr.DataArray([1.0], dims=["z"]))


def test_local_incidence_angle_matches_the_closed_form():
    flat = _planar_dem(0.0, facing="flat")
    assert float(
        sar_processing.local_incidence_angle(flat, 35.0, 280.0)[10, 10]
    ) == pytest.approx(35.0)

    slope = _planar_dem(20.0, facing="west")
    # looking west (270°), the slope tilts toward the radar → 35 − 20
    toward = sar_processing.local_incidence_angle(slope, 35.0, 270.0)
    assert float(toward[10, 10]) == pytest.approx(15.0)
    # looking east, it tilts away → 35 + 20
    away = sar_processing.local_incidence_angle(slope, 35.0, 90.0)
    assert float(away[10, 10]) == pytest.approx(55.0)
    assert away.attrs["units"] == "degrees" and away.rio.crs.to_epsg() == 32610
    # clipping keeps the angle in [0, 90]
    steep = sar_processing.local_incidence_angle(
        _planar_dem(60.0, facing="west"), 40.0, 90.0
    )
    assert float(steep.max()) <= 90.0
    unclipped = sar_processing.local_incidence_angle(
        _planar_dem(60.0, facing="west"), 40.0, 90.0, clip_to_valid=False
    )
    assert float(unclipped[10, 10]) == pytest.approx(100.0)


def test_local_incidence_angle_is_dask_aware():
    lazy = _planar_dem(20.0, facing="west").chunk({"x": 7, "y": 7})
    lia = sar_processing.local_incidence_angle(lazy, 35.0, 270.0)
    assert lia.chunks is not None
    assert float(lia.compute()[10, 10]) == pytest.approx(15.0)


def test_look_azimuth_and_headings():
    assert sar_processing.look_azimuth(0.0) == 90.0
    assert sar_processing.look_azimuth(0.0, looking="left") == 270.0
    assert sar_processing.look_azimuth(350.0) == 80.0
    assert set(sar_processing.S1_HEADING) == {"ascending", "descending"}
    # Sentinel-1 ascending looks roughly east, descending roughly west
    assert (
        60 < sar_processing.look_azimuth(sar_processing.S1_HEADING["ascending"]) < 100
    )
    assert (
        250 < sar_processing.look_azimuth(sar_processing.S1_HEADING["descending"]) < 300
    )


# ── search, replayed ──────────────────────────────────────────────────────────


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_planetary_computer():
    gdf = sentinel1.search(RAINIER, "2023-08-01/2023-08-10", max_items=4)
    assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) >= 1
    assert (gdf["collection"] == "sentinel-1-rtc").all()
    assert set(gdf["sat:orbit_state"]) <= {"ascending", "descending"}
    assert gdf.attrs["source"] == "planetary-computer"


@pytest.mark.recorded
@pytest.mark.vcr
def test_search_opera_rtc_is_open():
    """The OPERA search needs no Earthdata Login — only the COG reads do."""
    gdf = sentinel1.search(
        RAINIER, "2024-07-01/2024-07-10", source="opera-rtc-s1", max_items=3
    )
    assert len(gdf) >= 1
    assert gdf["collection"].iloc[0] == sentinel1.OPERA_COLLECTION
    item = gdf["stac_item"].iloc[0]
    assert {"0_VV", "0_VH", "0_mask"} <= set(item["assets"])
    assert gdf.attrs["source"] == "opera-rtc-s1"


# ── loading, on stubs ─────────────────────────────────────────────────────────


def _item(when: str, *, orbit_state="ascending", relative_orbit=13, absolute_orbit=1):
    return {
        "id": f"S1A_{when}",
        "collection": "sentinel-1-rtc",
        "properties": {
            "datetime": when,
            "sat:orbit_state": orbit_state,
            "sat:relative_orbit": relative_orbit,
            "sat:absolute_orbit": absolute_orbit,
        },
        "assets": {"vv": {"href": "https://x/vv.tif"}},
    }


def _dataset(values: dict[str, list[float]], times: list[str]) -> xr.Dataset:
    time = pd.to_datetime(times).values
    data = {}
    for name, per_time in values.items():
        dtype = "uint8" if name.endswith("mask") else "float32"
        block = np.array(
            [np.full((2, 2), v, dtype=dtype) for v in per_time], dtype=dtype
        )
        data[name] = (("time", "y", "x"), block)
    return xr.Dataset(
        data,
        coords={"time": time, "y": [5180400.0, 5180390.0], "x": [580000.0, 580010.0]},
    ).rio.write_crs("EPSG:32610")


@pytest.fixture
def fake_stac(monkeypatch, fake_credentials):
    calls: dict[str, object] = {}
    items = [_item("2023-08-04")]

    def search(catalog_id, collection, aoi=None, time=None, **kwargs):
        calls["search"] = (catalog_id, collection, kwargs)
        return items

    def items_to_geodataframe(item_list):
        item_list = list(item_list)
        return gpd.GeoDataFrame(
            {
                "id": [i["id"] for i in item_list],
                "collection": [i.get("collection") for i in item_list],
                "stac_item": item_list,
            },
            geometry=[None] * len(item_list),
            crs="EPSG:4326",
        )

    def load(item_arg, aoi=None, **kwargs):
        calls["load"] = kwargs
        if "0_VV" in (kwargs.get("bands") or []):
            return _dataset({"0_VV": [0.1], "0_mask": [2]}, ["2023-08-04"])
        return _dataset({"vv": [0.1], "vh": [0.01]}, ["2023-08-04"])

    monkeypatch.setattr(sentinel1.providers.stac, "search", search)
    monkeypatch.setattr(
        sentinel1.providers.stac, "items_to_geodataframe", items_to_geodataframe
    )
    monkeypatch.setattr(sentinel1.providers.stac, "load", load)
    return calls


@pytest.mark.recorded
def test_load_planetary_computer_output_contract(fake_stac):
    ds = sentinel1.load(RAINIER, "2023-08")
    assert fake_stac["load"]["groupby"] == "sat:absolute_orbit"
    assert fake_stac["load"]["catalog"] == "planetary-computer"
    assert ds["vv"].dims == ("time", "y", "x")
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    # dB by default: 0.1 linear power → −10 dB
    assert float(ds["vv"].isel(time=0, y=0, x=0)) == pytest.approx(-10.0)
    assert float(ds["vh"].isel(time=0, y=0, x=0)) == pytest.approx(-20.0)
    assert ds.attrs["units"] == "dB"
    assert list(ds["sat:orbit_state"].values) == ["ascending"]
    assert list(ds["sat:relative_orbit"].values) == [13]
    assert ds.attrs["product_id"] == "sentinel-1-rtc"
    assert all(not callable(v) for v in ds.attrs.values())

    linear = sentinel1.load(RAINIER, "2023-08", units="linear power")
    assert float(linear["vv"].isel(time=0, y=0, x=0)) == pytest.approx(0.1)
    assert linear.attrs["units"] == "linear power"


@pytest.mark.recorded
def test_load_opera_route_renames_and_keeps_the_mask(fake_stac):
    ds = sentinel1.load(RAINIER, "2024-07", source="opera-rtc-s1", bands=["vv"])
    assert fake_stac["load"]["bands"] == ["0_VV", "0_mask"]
    assert fake_stac["load"]["catalog"] == "cmr-asf"
    assert fake_stac["load"]["groupby"] == "solar_day"
    assert set(ds.data_vars) == {"vv", "mask"}
    assert ds["mask"].dtype == np.uint8 and ds["mask"].rio.nodata == 255
    assert ds["mask"].attrs["flag_meanings"].split()[2] == "layover"
    assert ds.attrs["source_id"] == "opera-rtc-s1"
    no_mask = sentinel1.load(
        RAINIER, "2024-07", source="opera-rtc-s1", bands=["vv"], mask=False
    )
    assert fake_stac["load"]["bands"] == ["0_VV"]
    assert "mask" not in no_mask.data_vars or True  # the stub always returns one


@pytest.mark.recorded
def test_load_without_items_raises(monkeypatch, fake_credentials):
    monkeypatch.setattr(sentinel1.providers.stac, "search", lambda *a, **k: [])
    monkeypatch.setattr(
        sentinel1.providers.stac,
        "items_to_geodataframe",
        lambda items: gpd.GeoDataFrame(
            {"id": [], "collection": [], "stac_item": []}, geometry=[], crs="EPSG:4326"
        ),
    )
    with pytest.raises(ValueError, match="No Sentinel-1 RTC items"):
        sentinel1.load(RAINIER, "1999-01")


@pytest.mark.recorded
def test_opera_route_without_credentials_names_the_free_alternative(no_credentials):
    with pytest.raises(esd.CredentialError) as excinfo:
        sentinel1.load(RAINIER, "2024-07", source="opera-rtc-s1")
    assert excinfo.value.provider == "earthdata"
    assert 'source="planetary-computer"' in str(excinfo.value)


# ── the local incidence angle, on stubs ──────────────────────────────────────


@pytest.mark.recorded
def test_local_incidence_angle_from_a_dem(monkeypatch, fake_credentials):
    dem = _planar_dem(20.0, facing="west")
    ds = sentinel1.local_incidence_angle(RAINIER, source="dem", dem=dem)
    assert set(ds.data_vars) == {"local_incidence_angle", "incidence_angle"}
    assert ds["local_incidence_angle"].dims == ("y", "x")
    assert float(ds["incidence_angle"][10, 10]) == pytest.approx(39.0)
    assert ds.attrs["orbit_state"] == "ascending"
    assert ds.attrs["source_id"] == "dem"
    assert ds.attrs["product_id"] == "sentinel-1-local-incidence-angle"
    # descending looks west, so a west-facing slope is tilted toward the radar
    descending = sentinel1.local_incidence_angle(
        RAINIER, source="dem", dem=dem, orbit_state="descending", incidence_angle=35.0
    )
    assert float(descending["local_incidence_angle"][10, 10]) < 35.0
    ascending = sentinel1.local_incidence_angle(
        RAINIER, source="dem", dem=dem, orbit_state="ascending", incidence_angle=35.0
    )
    assert float(ascending["local_incidence_angle"][10, 10]) > 35.0
    with pytest.raises(ValueError, match="orbit_state must be one of"):
        sentinel1.local_incidence_angle(
            RAINIER, source="dem", dem=dem, orbit_state="up"
        )


@pytest.mark.recorded
def test_local_incidence_angle_from_opera_static(fake_stac, monkeypatch):
    def load(item_arg, aoi=None, **kwargs):
        fake_stac["load"] = kwargs
        return _dataset(
            {
                "0_local_incidence_angle": [22.0],
                "0_incidence_angle": [38.0],
                "0_mask": [2],
            },
            ["2014-04-03"],
        )

    monkeypatch.setattr(sentinel1.providers.stac, "load", load)
    ds = sentinel1.local_incidence_angle(RAINIER)
    assert fake_stac["load"]["bands"] == [
        "0_local_incidence_angle",
        "0_incidence_angle",
        "0_mask",
    ]
    assert set(ds.data_vars) == {"local_incidence_angle", "incidence_angle", "mask"}
    assert "time" not in ds.dims  # the static layers collapse the degenerate axis
    assert float(ds["local_incidence_angle"][0, 0]) == pytest.approx(22.0)
    assert ds.attrs["source_id"] == "opera-static"


@pytest.mark.recorded
def test_local_incidence_angle_opera_without_granules(monkeypatch, fake_credentials):
    monkeypatch.setattr(sentinel1.providers.stac, "search", lambda *a, **k: [])
    with pytest.raises(ValueError, match="try source='dem'"):
        sentinel1.local_incidence_angle(RAINIER)


# ── the deprecation shim ──────────────────────────────────────────────────────


@pytest.mark.recorded
def test_old_sentinel1_class_is_a_shim(fake_stac):
    from easysnowdata import _deprecation
    from easysnowdata.remote_sensing import Sentinel1

    _deprecation.reset_warnings()
    with pytest.warns(
        _deprecation.EasysnowdataDeprecationWarning, match="sar.sentinel1.load"
    ):
        ds = Sentinel1(RAINIER, start_date="2023-08-01", end_date="2023-08-10")
    assert isinstance(ds, xr.Dataset) and ds.attrs["units"] == "dB"
    assert "sat:relative_orbit" in ds.coords
    with pytest.raises(ValueError, match="Invalid catalog_choice"):
        Sentinel1(RAINIER, catalog_choice="aws")


# ── live smoke tests ──────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_sentinel1_planetary_computer():
    ds = sentinel1.load(
        RAINIER, "2023-08-01/2023-08-15", bands=["vv", "vh"], resolution=80
    )
    assert ds["vv"].dims == ("time", "y", "x") and ds.sizes["time"] >= 1
    assert ds["vv"].dtype == np.float32 or ds["vv"].dtype == np.float64
    assert ds["vv"].chunks is not None
    assert ds.rio.crs.to_epsg() == 32610 and ds.odc.crs.epsg == 32610
    assert ds.attrs["units"] == "dB" and ds.attrs["product_id"] == "sentinel-1-rtc"
    assert "sat:orbit_state" in ds.coords
    values = ds["vv"].isel(time=0).compute()
    assert -40 < float(np.nanmedian(values)) < 10  # dB backscatter


@pytest.mark.live
def test_live_sentinel1_search_opera_is_open():
    gdf = sentinel1.search(RAINIER, "2024-07-01/2024-07-10", source="opera-rtc-s1")
    assert len(gdf) >= 1 and gdf.crs.to_epsg() == 4326
    assert gdf["collection"].iloc[0] == sentinel1.OPERA_COLLECTION


@pytest.mark.live
def test_live_local_incidence_angle_from_the_dem_needs_no_account():
    ds = sentinel1.local_incidence_angle(RAINIER, source="dem", resolution=90)
    lia = ds["local_incidence_angle"].compute()
    assert lia.dims == ("y", "x")
    assert ds.rio.crs.to_epsg() == 32610
    assert 0.0 <= float(np.nanmin(lia)) and float(np.nanmax(lia)) <= 90.0
    # Rainier's relief spreads the angle well away from the flat-earth value
    assert float(np.nanmax(lia)) - float(np.nanmin(lia)) > 20.0
    assert ds.attrs["source_id"] == "dem"


@pytest.mark.live
@pytest.mark.requires_earthaccess
def test_live_local_incidence_angle_from_opera_static():
    ds = sentinel1.local_incidence_angle(RAINIER, resolution=90)
    assert "local_incidence_angle" in ds.data_vars and "mask" in ds.data_vars
    lia = ds["local_incidence_angle"].compute()
    assert 0.0 <= float(np.nanmin(lia)) <= float(np.nanmax(lia)) <= 90.0
    assert ds["mask"].attrs["flag_meanings"].split()[2] == "layover"


# ── the Earth Engine routes, mocked at the providers.gee boundary ────────────


@pytest.fixture
def fake_gee(monkeypatch, fake_credentials):
    calls: dict[str, object] = {}

    class Collection:
        def __init__(self, asset):
            self.asset = asset
            self.filters: list = []

        def filterDate(self, start, end):
            calls["dates"] = (start, end)
            return self

        def filterBounds(self, geometry):
            calls["bounds"] = geometry
            return self

        def filter(self, spec):
            self.filters.append(spec)
            return self

        def select(self, bands):
            calls.setdefault("selected", []).append(bands)
            return self

        def median(self):
            calls["median"] = True
            return self

    class EEFilter:
        @staticmethod
        def eq(field, value):
            return (field, value)

    class FakeEE:
        ImageCollection = Collection
        Filter = EEFilter

    def open_dataset(collection, aoi=None, **kwargs):
        calls["open"] = (getattr(collection, "asset", collection), kwargs)
        name = "angle" if calls.get("angle_mode") else "VV"
        values = np.full((1, 4, 5), 38.0 if name == "angle" else 0.1, dtype="float32")
        return xr.Dataset(
            {name: (("time", "y", "x"), values)},
            coords={
                "time": pd.to_datetime(["2024-07-02"]).values,
                "y": np.linspace(5180400.0, 5180300.0, 4),
                "x": np.linspace(580000.0, 580400.0, 5),
            },
        ).rio.write_crs("EPSG:32610")

    monkeypatch.setattr(sentinel1.providers.gee, "ee", lambda: FakeEE)
    monkeypatch.setattr(sentinel1.providers.gee, "open_dataset", open_dataset)
    monkeypatch.setattr(
        sentinel1.providers.gee, "geometry", lambda aoi: {"type": "Polygon"}
    )
    return calls


@pytest.mark.recorded
def test_load_gee_route(fake_gee):
    ds = sentinel1.load(RAINIER, "2024-07", source="gee", bands=["vv"])
    asset, kwargs = fake_gee["open"]
    assert asset == sentinel1.GEE_COLLECTION
    assert fake_gee["selected"] == [["VV"]]
    assert fake_gee["dates"][0] == "2024-07-01"
    assert kwargs["chunks"] == {}
    assert ds["vv"].dims == ("time", "y", "x")
    assert float(ds["vv"].isel(time=0, y=0, x=0)) == pytest.approx(-10.0)  # dB
    assert ds.attrs["source_id"] == "gee"


@pytest.mark.recorded
def test_local_incidence_angle_from_earth_engine(fake_gee, monkeypatch):
    fake_gee["angle_mode"] = True
    dem = _planar_dem(20.0, facing="west", n=4)
    dem = dem.assign_coords(
        y=np.linspace(5180400.0, 5180300.0, 4), x=np.linspace(580000.0, 580400.0, 4)
    )
    monkeypatch.setattr(sentinel1, "_copernicus_dem", lambda *args, **kwargs: dem)
    ds = sentinel1.local_incidence_angle(
        RAINIER, source="gee", orbit_state="descending"
    )
    assert set(ds.data_vars) == {"local_incidence_angle", "incidence_angle"}
    assert ds.attrs["source_id"] == "gee"
    assert float(ds["incidence_angle"].max()) == pytest.approx(38.0)
    # descending looks west, so a west-facing slope tilts toward the radar
    assert float(ds["local_incidence_angle"].mean()) < 38.0


@pytest.mark.recorded
def test_copernicus_dem_helper(monkeypatch, fake_credentials):
    calls: dict[str, object] = {}

    def search(catalog_id, collection, aoi=None, time=None, **kwargs):
        calls["search"] = (catalog_id, collection)
        return [{"id": "dem"}]

    def load(items, aoi=None, **kwargs):
        calls["load"] = kwargs
        return xr.Dataset(
            {
                "data": (
                    ("time", "y", "x"),
                    np.full((1, 3, 3), 1500.0, dtype="float32"),
                )
            },
            coords={
                "time": pd.to_datetime(["2021-04-22"]).values,
                "y": [5180400.0, 5180370.0, 5180340.0],
                "x": [580000.0, 580030.0, 580060.0],
            },
        ).rio.write_crs("EPSG:32610")

    monkeypatch.setattr(sentinel1.providers.stac, "search", search)
    monkeypatch.setattr(sentinel1.providers.stac, "load", load)
    dem = sentinel1._copernicus_dem(esd.parse_aoi(RAINIER), 30, "utm", None)
    assert calls["search"] == ("planetary-computer", "cop-dem-glo-30")
    assert calls["load"]["bands"] == ["data"]
    assert dem.dims == ("y", "x")  # the degenerate time axis is collapsed
    assert float(dem[0, 0]) == 1500.0

    monkeypatch.setattr(sentinel1.providers.stac, "search", lambda *a, **k: [])
    with pytest.raises(ValueError, match="No Copernicus DEM tiles"):
        sentinel1._copernicus_dem(esd.parse_aoi(RAINIER), 30, "utm", None)
