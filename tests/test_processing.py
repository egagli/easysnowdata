"""Unit tests for easysnowdata.processing on synthetic arrays — no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easysnowdata import processing as proc
from easysnowdata.processing import categorical, masks, optical, sar, wateryear


def _cube(values: np.ndarray, times=None, name=None, **attrs) -> xr.DataArray:
    values = np.asarray(values)
    if values.ndim == 2:
        return xr.DataArray(values, dims=("y", "x"), name=name, attrs=attrs)
    times = (
        pd.to_datetime(times)
        if times is not None
        else pd.date_range("2022-01-01", periods=values.shape[0], freq="10D")
    )
    return xr.DataArray(
        values, dims=("time", "y", "x"), coords={"time": times}, name=name, attrs=attrs
    )


# ── categorical ───────────────────────────────────────────────────────────────


class TestCategorical:
    def test_set_and_read_flags(self):
        da = _cube(np.array([[1, 2], [3, 9]], dtype="uint8"))
        out = categorical.set_flags(
            da,
            [1, 2, 3, 9],
            ["Tundra", "Boreal Forest", "Maritime", "Fill"],
            ["#a", "#b", "#c", "#fff"],
            long_name="snow class",
        )
        assert out.attrs["flag_values"] == [1, 2, 3, 9]
        assert out.attrs["flag_meanings"] == "Tundra Boreal_Forest Maritime Fill"
        assert (
            out.attrs["flag_colors"] == "#a #b #c #fff"
            and out.attrs["long_name"] == "snow class"
        )
        assert "flag_values" not in da.attrs  # copy by default
        categorical.set_flags(da, [1], ["a"], inplace=True)
        assert da.attrs["flag_values"] == [1]
        table = categorical.flags(out)
        assert list(table.columns) == ["value", "meaning", "color"] and list(
            table["value"]
        ) == [1, 2, 3, 9]
        assert all(isinstance(v, (str, list)) for v in out.attrs.values())

    def test_flags_without_colors_and_errors(self):
        da = _cube(np.zeros((2, 2)))
        with pytest.raises(KeyError, match="flag_values"):
            categorical.flags(da)
        with pytest.raises(ValueError, match="same length"):
            categorical.set_flags(da, [1, 2], ["a"])
        with pytest.raises(ValueError, match="flag_colors"):
            categorical.set_flags(da, [1, 2], ["a", "b"], ["#a"])
        out = categorical.set_flags(da, np.array([1, 2]), ["a", "b"])
        assert categorical.flags(out)["color"].tolist() == [None, None]
        out.attrs["flag_meanings"] = "only_one"
        with pytest.raises(ValueError, match="lengths differ"):
            categorical.flags(out)
        assert categorical.flags(
            {"flag_values": np.array([5]), "flag_meanings": ["x"]}
        )["meaning"].tolist() == ["x"]

    def test_from_class_info_and_flag_mask(self):
        info = {
            1: {"name": "Af", "color": [0, 0, 255]},
            2: {"name": "Snow or ice", "color": "#ff96ff"},
        }
        values, meanings, colors = categorical.from_class_info(info)
        assert (
            values == [1, 2]
            and meanings == ["Af", "Snow_or_ice"]
            and colors == ["#0000ff", "#ff96ff"]
        )
        da = categorical.set_flags(
            _cube(np.array([[1, 2], [2, 1]])), values, meanings, colors
        )
        mask = categorical.flag_mask(da, "snow or ice")
        assert mask.values.tolist() == [[False, True], [True, False]]
        with pytest.raises(ValueError, match="Unknown flag meanings"):
            categorical.flag_mask(da, "water")
        assert (
            categorical.meaning_key("  Cloud   high  probability ")
            == "Cloud_high_probability"
        )


# ── masks ─────────────────────────────────────────────────────────────────────


class TestMasks:
    scl = _cube(np.arange(12, dtype="uint8").reshape(3, 4))

    def test_scl_default_matches_legacy(self):
        keep = masks.scl_mask(self.scl)
        kept_values = sorted(int(v) for v in self.scl.values[keep.values])
        assert kept_values == [4, 5, 6, 7, 11]  # everything mask_data() kept by default
        assert masks.scl_mask(self.scl, remove=["Snow or ice", 6]).values.sum() == 10
        with pytest.raises(ValueError, match="Unknown SCL class"):
            masks.scl_mask(self.scl, remove=["fog"])

    def test_apply_scl_mask(self):
        ds = xr.Dataset({"red": self.scl.astype("uint16") * 100, "scl": self.scl})
        out = masks.apply_scl_mask(ds)
        assert np.isnan(out["red"].values[0, 0]) and out["red"].values[1, 0] == 400
        assert not np.isnan(out["scl"].values[1, 0])
        out2 = masks.apply_scl_mask(ds["red"], scl=ds["scl"], remove=("vegetation",))
        assert np.isnan(out2.values[1, 0]) and out2.values[0, 0] == 0
        with pytest.raises(ValueError, match="scl="):
            masks.apply_scl_mask(ds["red"])

    def test_fmask_bits(self):
        # bit 1 (cloud) + bit 3 (shadow) + aerosol high (3 << 6) and fill
        fmask = _cube(np.array([[0, 2], [8, 0b11000000], [255, 1]], dtype="uint8"))
        assert masks.fmask_bit(fmask, 1).values.tolist() == [[0, 1], [0, 0], [1, 0]]
        assert masks.fmask_aerosol_level(fmask).values.tolist() == [
            [0, 0],
            [0, 3],
            [3, 0],
        ]
        keep = masks.fmask_mask(fmask)
        assert keep.values.tolist() == [[True, False], [False, True], [False, False]]
        keep_aero = masks.fmask_mask(fmask, remove=(), aerosol_remove=("high",))
        assert keep_aero.values.tolist() == [[True, True], [True, False], [False, True]]
        with pytest.raises(ValueError, match="Unknown Fmask flag"):
            masks.fmask_mask(fmask, remove=("haze",))
        with pytest.raises(ValueError, match="Unknown aerosol level"):
            masks.fmask_mask(fmask, aerosol_remove=("extreme",))
        ds = xr.Dataset({"red": fmask.astype("int16"), "Fmask": fmask})
        out = masks.apply_fmask(ds)
        assert np.isnan(out["red"].values[0, 1]) and out["red"].values[0, 0] == 0
        assert (
            masks.apply_fmask(ds["red"], fmask=fmask, remove=("cirrus",)).isnull().sum()
            == 2
        )
        with pytest.raises(ValueError, match="fmask="):
            masks.apply_fmask(ds["red"])

    def test_mask_nodata(self):
        da = _cube(np.array([[0, 5], [7, 0]], dtype="uint8"), nodata=0, units="1")
        out = masks.mask_nodata(da)
        assert (
            np.isnan(out.values[0, 0])
            and out.values[0, 1] == 5
            and "nodata" not in out.attrs
            and out.attrs["units"] == "1"
        )
        assert (
            masks.mask_nodata(_cube(np.ones((2, 2)))).dtype == float
        )  # nothing known → unchanged
        rio_da = _cube(np.array([[9, 1]], dtype="uint8")).rio.write_nodata(9)
        assert np.isnan(masks.mask_nodata(rio_da).values[0, 0])
        ds = xr.Dataset({"a": da, "b": _cube(np.ones((2, 2)))})
        out = masks.mask_nodata(ds, nodata=5)
        assert np.isnan(out["a"].values[0, 1]) and out["b"].values.sum() == 4
        nan_da = _cube(np.ones((2, 2)), nodata=float("nan"))
        assert masks.mask_nodata(nan_da) is nan_da


# ── optical ───────────────────────────────────────────────────────────────────


class TestOptical:
    def test_harmonize_baseline(self):
        values = np.full((3, 2, 2), 1500, dtype="uint16")
        values[0] = 500  # 2021 scene: raw
        values[2, 0, 0] = 800  # below the offset: clips to 0
        times = ["2021-06-01", "2022-01-25", "2023-06-01"]
        ds = xr.Dataset({"red": _cube(values, times), "scl": _cube(values, times)})
        out = optical.harmonize_s2_baseline(ds)
        assert out["red"].values[0].tolist() == [[500, 500], [500, 500]]
        assert out["red"].values[1, 0, 0] == 500 and out["red"].values[2, 0, 0] == 0
        assert (out["scl"] == ds["scl"]).all()
        assert out["red"].dims == ("time", "y", "x")
        da = optical.harmonize_s2_baseline(ds["red"], offset=100, cutoff="2023-01-01")
        assert da.values[1, 0, 0] == 1500 and da.values[2, 0, 1] == 1400
        with pytest.raises(ValueError, match="dimension"):
            optical.harmonize_s2_baseline(ds["red"].isel(time=0))

    def test_scale_offset(self):
        da = _cube(
            np.array([[1000, 0], [2000, 65535]], dtype="uint16"),
            scale="0.0001",
            offset=-0.1,
            nodata=0,
        )
        out = optical.scale_offset(da)
        assert out.dtype == np.float32
        assert out.values[0, 0] == pytest.approx(0.0, abs=1e-6) and out.values[
            1, 0
        ] == pytest.approx(0.1)
        assert (
            np.isnan(out.values[0, 1])
            and "scale" not in out.attrs
            and "nodata" not in out.attrs
        )
        raw = optical.scale_offset(da, scale=2, offset=0, nodata_to_nan=False)
        assert raw.values[0, 1] == 0 and raw.values[0, 0] == 2000
        untouched = optical.scale_offset(
            _cube(np.ones((2, 2), dtype="int16")), nodata_to_nan=False
        )
        assert untouched.dtype == np.int16
        ds = xr.Dataset(
            {"red": da, "scl": _cube(np.array([[4, 0], [4, 4]], dtype="uint8"))}
        )
        out = optical.scale_offset(ds, scale={"red": 0.001}, offset={"red": 0.0})
        assert out["red"].values[1, 0] == pytest.approx(2.0)
        assert out["scl"].dtype == np.float32  # no attrs → just float, nothing to mask
        assert optical._coerce("x") is None


# ── sar ───────────────────────────────────────────────────────────────────────


class TestSar:
    def test_db_round_trip(self):
        da = _cube(np.array([[0.01, 1.0], [0.0, 10.0]]), units="linear power")
        db = sar.linear_to_db(da)
        assert db.values[0].tolist() == pytest.approx([-20.0, 0.0]) and np.isnan(
            db.values[1, 0]
        )
        assert db.attrs["units"] == "dB"
        back = sar.db_to_linear(db)
        assert (
            back.values[0, 0] == pytest.approx(0.01)
            and back.attrs["units"] == "linear power"
        )
        ds = sar.linear_to_db(xr.Dataset({"vv": da}))
        assert ds["vv"].attrs["units"] == "dB" and ds.attrs["units"] == "dB"

    def test_remove_border_noise(self):
        values = np.array([[[0.0005, 0.5]], [[0.0005, 0.0]]])
        da = _cube(values, ["2017-01-01", "2019-01-01"])
        out = sar.remove_border_noise(da)
        assert np.isnan(out.values[0, 0, 0]) and out.values[0, 0, 1] == 0.5
        assert out.values[1, 0, 0] == 0.0005 and np.isnan(out.values[1, 0, 1])
        with pytest.raises(ValueError, match="dimension"):
            sar.remove_border_noise(da.isel(time=0))


# ── water year ────────────────────────────────────────────────────────────────


class TestWaterYear:
    def test_scalars(self):
        assert wateryear.water_year("2020-10-01") == 2021
        assert wateryear.water_year(pd.Timestamp("2020-09-30")) == 2020
        assert wateryear.water_year("2021-05-15", "southern") == 2021
        assert wateryear.water_year("2021-03-15", "southern") == 2020
        assert wateryear.day_of_water_year("2020-10-01") == 1
        assert wateryear.day_of_water_year("2020-11-01") == 32
        assert wateryear.day_of_water_year("2021-09-30") == 365
        assert (
            wateryear.day_of_water_year("2024-09-30") == 366
        )  # leap year inside WY 2024
        assert wateryear.water_year_start("2021-03-15") == pd.Timestamp("2020-10-01")
        assert wateryear.water_year_start("2021-05-15", "southern") == pd.Timestamp(
            "2021-04-01"
        )
        with pytest.raises(ValueError, match="hemisphere"):
            wateryear.water_year("2021-01-01", "equatorial")

    def test_vectorized_containers(self):
        idx = pd.date_range("2020-09-29", "2020-10-02")
        wy = wateryear.water_year(idx)
        assert isinstance(wy, pd.Index) and wy.tolist() == [2020, 2020, 2021, 2021]
        series = wateryear.day_of_water_year(pd.Series(idx, index=list("abcd")))
        assert isinstance(series, pd.Series) and series.tolist() == [
            365,
            366,
            1,
            2,
        ]  # WY2020 has 366 days
        arr = wateryear.water_year(np.array(idx))
        assert isinstance(arr, np.ndarray) and arr.tolist() == [2020, 2020, 2021, 2021]
        starts = wateryear.water_year_start(idx)
        assert isinstance(starts, pd.Index) and str(starts[0])[:10] == "2019-10-01"
        da = xr.DataArray(idx, dims="time")
        out = wateryear.water_year(da)
        assert (
            isinstance(out, xr.DataArray)
            and out.dims == ("time",)
            and out.name == "water_year"
        )

    def test_add_coords(self):
        times = pd.date_range("2020-09-30", periods=3)
        ds = xr.Dataset({"swe": ("time", [1.0, 2.0, 3.0])}, coords={"time": times})
        out = wateryear.add_water_year_coords(ds)
        assert out["water_year"].values.tolist() == [2020, 2021, 2021]
        assert out["dowy"].values.tolist() == [366, 1, 2]
        assert out.groupby("water_year").max()["swe"].values.tolist() == [1.0, 3.0]
        south = wateryear.add_water_year_coords(
            ds["swe"], "southern", names=("wy", "d")
        )
        assert south["wy"].values.tolist() == [2020, 2020, 2020]
        with pytest.raises(ValueError, match="dimension"):
            wateryear.add_water_year_coords(ds.rename({"time": "t"}))


def test_public_surface():
    for name in proc.__all__:
        assert hasattr(proc, name), name
    assert proc.SCL_CLASSES[11][0] == "Snow or ice"
