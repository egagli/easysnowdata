"""Smoke tests for easysnowdata.plotting with the Agg backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from easysnowdata import plotting  # noqa: E402
from easysnowdata.processing import categorical, optical  # noqa: E402


@pytest.fixture
def classes():
    da = xr.DataArray(
        np.array([[1, 5], [37, 200]], dtype="uint8"),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="snow",
    )
    return categorical.set_flags(
        da,
        [1, 5, 37, 200],
        ["no decision", "lake", "ocean", "snow"],
        ["#111111", "#0064C8", "#B4B4B4", "#0096A0"],
        long_name="MOD10A2 classes",
    )


class TestCategorical:
    def test_colormap_from_flags_handles_gaps(self, classes):
        cmap, norm, table = plotting.colormap_from_flags(classes)
        assert cmap.N == 4 and list(table["value"]) == [1, 5, 37, 200]
        assert norm(1) == 0 and norm(5) == 1 and norm(37) == 2 and norm(200) == 3
        assert cmap(norm(200)) == matplotlib.colors.to_rgba("#0096A0")

    def test_legend_handles(self, classes):
        handles, labels = plotting.legend_handles(classes)
        assert labels == ["no decision", "lake", "ocean", "snow"] and len(handles) == 4

    def test_categorical_plot(self, classes):
        ax = plotting.categorical(classes)
        assert ax.get_title() == "MOD10A2 classes" and ax.get_legend() is not None
        assert len(ax.get_images()) == 1
        plt.close("all")
        fig, ax = plt.subplots()
        out = plotting.categorical(
            classes.expand_dims(time=[np.datetime64("2020-01-01")]),
            ax=ax,
            legend=False,
            title="t",
        )
        assert out is ax and ax.get_legend() is None and ax.get_title() == "t"
        plt.close("all")

    def test_missing_colors_default_grey(self):
        da = categorical.set_flags(
            xr.DataArray(np.array([[1, 2]]), dims=("y", "x")), [1, 2], ["a", "b"]
        )
        cmap, _, _ = plotting.colormap_from_flags(da)
        assert cmap(0) == matplotlib.colors.to_rgba("#808080")
        handles, _ = plotting.legend_handles(da)
        assert handles[0].get_facecolor() == matplotlib.colors.to_rgba("#808080")


class TestOther:
    def test_register_colormap(self):
        cmap = plotting.register_colormap("esd_test_cmap", ["#000000", "#ffffff"])
        assert matplotlib.colormaps["esd_test_cmap"] is not None and cmap.N == 2
        same = plotting.register_colormap("esd_test_cmap", ["#ff0000"])
        assert same.N == 2  # kept without overwrite
        new = plotting.register_colormap("esd_test_cmap", ["#ff0000"], overwrite=True)
        assert new.N == 1 and matplotlib.colormaps["esd_test_cmap"].N == 1

    def test_rgb(self):
        rng = np.random.default_rng(0)
        ds = xr.Dataset(
            {
                b: xr.DataArray(rng.random((4, 4)), dims=("y", "x"))
                for b in ("red", "green", "blue")
            }
        )
        comp = optical.rgb(ds, vmin=0, vmax=1)
        ax = plotting.rgb(comp, title="quicklook")
        assert ax.get_title() == "quicklook" and len(ax.get_images()) == 1
        plt.close("all")
        ax = plotting.rgb(comp.expand_dims(time=[0]))
        assert len(ax.get_images()) == 1
        plt.close("all")
