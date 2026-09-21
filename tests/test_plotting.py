"""easysnowdata.plotting with the Agg backend: labels, furniture, legends, dates."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import warnings  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from easysnowdata import plotting  # noqa: E402
from easysnowdata.processing import categorical, contract  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def classes():
    da = xr.DataArray(
        np.array([[1, 5], [37, 200]], dtype="uint8"),
        dims=("y", "x"),
        coords={"y": [5_200_100.0, 5_200_000.0], "x": [600_000.0, 600_100.0]},
        name="snow",
    )
    da = categorical.set_flags(
        da,
        [1, 5, 37, 200, 254],
        ["no decision", "lake", "ocean", "snow", "saturated"],
        ["#111111", "#0064C8", "#B4B4B4", "#0096A0", "#ff0000"],
        long_name="MOD10A2 classes",
    )
    return contract.write_crs(da, "EPSG:32610")


@pytest.fixture
def geographic():
    lon = np.linspace(-121.9, -121.5, 20)
    lat = np.linspace(47.0, 46.7, 15)
    da = xr.DataArray(
        np.random.default_rng(0).random((15, 20)) * 3000,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="elevation",
        attrs={"long_name": "elevation", "units": "m"},
    )
    return contract.write_crs(da, "EPSG:4326")


@pytest.fixture
def projected():
    x = np.arange(600_000.0, 610_000.0, 500.0)
    y = np.arange(5_205_000.0, 5_195_000.0, -500.0)
    da = xr.DataArray(
        np.random.default_rng(1).random((len(y), len(x))),
        dims=("y", "x"),
        coords={"y": y, "x": x, "time": np.datetime64("2024-03-10")},
        name="swe",
        attrs={"long_name": "snow water equivalent", "units": "m"},
    )
    return contract.write_crs(da, "EPSG:32610")


class TestLabel:
    def test_units_in_square_brackets(self, projected):
        assert plotting.label(projected) == "snow water equivalent [m]"
        assert plotting.label({"long_name": "fraction", "units": "1"}) == "fraction"
        assert plotting.label({"units": "cm"}, name="SWE") == "SWE [cm]"
        assert plotting.label(xr.DataArray([1.0], name="x")) == "x"


class TestMap:
    def test_projected_map_has_equal_aspect_labels_and_furniture(self, projected):
        ax = plotting.map(projected, cmap="Blues")
        assert ax.get_aspect() == 1.0
        assert ax.get_title() == "snow water equivalent, 2024-03-10"
        # graticule labels replace the metric ticks on projected axes
        assert ax.get_xlabel() == "longitude [°]" and ax.get_ylabel() == "latitude [°]"
        assert any("°W" in t.get_text() for t in ax.get_xticklabels())
        assert any("°N" in t.get_text() for t in ax.get_yticklabels())
        # a colorbar that hugs the drawn map (an inset of it), labelled with the units
        labels = [a.get_ylabel() for a in ax.child_axes]
        assert "snow water equivalent [m]" in labels
        # the scale bar is an artist on the axes
        assert any(type(a).__name__ == "ScaleBar" for a in ax.artists)

    def test_geographic_map_warns_and_corrects_the_aspect(self, geographic):
        with pytest.warns(plotting.GeographicAxesWarning, match="1/cos"):
            ax = plotting.map(geographic, cmap="terrain")
        expected = 1.0 / np.cos(np.radians(46.85))
        assert ax.get_aspect() == pytest.approx(expected, rel=1e-2)
        assert ax.get_xlabel() == "longitude [°]"
        assert any("°W" in t.get_text() for t in ax.get_xticklabels())

    def test_everything_is_a_keyword(self, projected):
        fig, ax = plt.subplots()
        out = plotting.map(
            projected,
            ax=ax,
            scalebar=False,
            graticule=False,
            colorbar=False,
            title="t",
            cmap="Blues",
        )
        assert out is ax and ax.get_title() == "t"
        assert not any(type(a).__name__ == "ScaleBar" for a in ax.artists)
        assert ax.get_xlabel() == "easting [m]" and ax.get_ylabel() == "northing [m]"
        assert len(fig.axes) == 1 and not ax.child_axes  # no colorbar axes

    def test_length_one_time_is_squeezed(self, projected):
        cube = projected.drop_vars("time").expand_dims(
            time=[np.datetime64("2024-03-10")]
        )
        ax = plotting.map(cube, graticule=False, scalebar=False)
        assert len(ax.get_images()) == 1


class TestCategorical:
    def test_colormap_from_flags_handles_gaps(self, classes):
        cmap, norm, table = plotting.colormap_from_flags(classes)
        assert cmap.N == 5 and list(table["value"]) == [1, 5, 37, 200, 254]
        assert norm(1) == 0 and norm(5) == 1 and norm(37) == 2 and norm(200) == 3
        assert cmap(norm(200)) == matplotlib.colors.to_rgba("#0096A0")

    def test_legend_handles(self, classes):
        handles, labels = plotting.legend_handles(classes)
        assert labels == ["no decision", "lake", "ocean", "snow", "saturated"]
        assert len(handles) == 5
        _, present = plotting.legend_handles(classes, present={1, 200})
        assert present == ["no decision", "snow"]

    def test_categorical_plot_lists_only_the_classes_present(self, classes):
        ax = plotting.categorical(classes)
        assert ax.get_title() == "MOD10A2 classes"
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == [
            "no decision",
            "lake",
            "ocean",
            "snow",
        ]
        # outside the axes, on the right, without a frame
        assert legend.get_frame_on() is False
        assert legend.get_bbox_to_anchor().x0 >= ax.bbox.x1
        assert len(ax.get_images()) == 1 and ax.get_aspect() == 1.0

    def test_all_classes_and_no_legend(self, classes):
        ax = plotting.categorical(classes, legend_kwargs={"all_classes": True})
        assert len(ax.get_legend().get_texts()) == 5
        fig, ax = plt.subplots()
        out = plotting.categorical(
            classes.expand_dims(time=[np.datetime64("2020-01-01")]),
            ax=ax,
            legend=False,
            title="t",
            scalebar=False,
            graticule=False,
        )
        assert out is ax and ax.get_legend() is None and ax.get_title() == "t"

    def test_missing_colors_default_grey(self):
        da = categorical.set_flags(
            xr.DataArray(np.array([[1, 2]]), dims=("y", "x")), [1, 2], ["a", "b"]
        )
        cmap, _, _ = plotting.colormap_from_flags(da)
        assert cmap(0) == matplotlib.colors.to_rgba("#808080")
        handles, _ = plotting.legend_handles(da)
        assert handles[0].get_facecolor() == matplotlib.colors.to_rgba("#808080")


class TestPoints:
    @pytest.fixture
    def stations(self):
        return gpd.GeoDataFrame(
            {
                "name": ["a", "b", "c"],
                "elevation_m": [1000.0, 1500.0, 2000.0],
                "network": ["awdb", "awdb", "cdec"],
            },
            geometry=gpd.points_from_xy([-121.9, -121.7, -121.5], [46.8, 46.9, 46.7]),
            crs="EPSG:4326",
        )

    def test_numeric_column_gets_a_colorbar(self, stations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", plotting.GeographicAxesWarning)
            ax = plotting.points(
                stations,
                column="elevation_m",
                basemap=False,
                legend_label="elevation [m]",
            )
        labels = [a.get_ylabel() for a in ax.child_axes]
        assert "elevation [m]" in labels
        assert ax.get_xlabel() == "longitude [°]"

    def test_categorical_column_gets_an_outside_legend(self, stations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", plotting.GeographicAxesWarning)
            ax = plotting.points(stations, column="network", basemap=False, title="t")
        legend = ax.get_legend()
        assert legend is not None and legend.get_title().get_text() == "network"
        assert sorted(t.get_text() for t in legend.get_texts()) == ["awdb", "cdec"]
        assert ax.get_title() == "t"

    def test_basemap_failure_is_a_warning_not_an_error(self, stations, monkeypatch):
        import contextily

        def boom(*args, **kwargs):
            raise RuntimeError("no tiles")

        monkeypatch.setattr(contextily, "add_basemap", boom)
        with pytest.warns(UserWarning, match="Could not draw the basemap"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", plotting.GeographicAxesWarning)
                plotting.points(stations, basemap=True)


class TestTimeseries:
    @pytest.fixture
    def obs(self):
        time = pd.date_range("2022-10-01", "2024-09-30", freq="D")
        stations = ["a", "b"]
        values = np.random.default_rng(0).random((2, len(time))) * 100
        return xr.Dataset(
            {
                "swe": (
                    ("station", "time"),
                    values,
                    {"long_name": "snow water equivalent", "units": "cm"},
                )
            },
            coords={
                "station": stations,
                "time": time,
                "name": ("station", ["Paradise", "Morse Lake"]),
                "elevation_m": ("station", [1650.0, np.nan]),
            },
        )

    def test_calendar_dates_and_station_labels(self, obs):
        ax = plotting.timeseries(obs["swe"])
        assert isinstance(
            ax.xaxis.get_major_formatter(), matplotlib.dates.ConciseDateFormatter
        )
        assert ax.get_ylabel() == "snow water equivalent [cm]"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "Paradise (1650 m)",
            "Morse Lake",
        ]
        assert ax.get_title() == "snow water equivalent"

    def test_dataset_needs_a_variable_when_ambiguous(self, obs):
        two = obs.assign(snwd=obs["swe"])
        with pytest.raises(ValueError, match="variable="):
            plotting.timeseries(two)
        ax = plotting.timeseries(two, variable="snwd", hue="station", legend=False)
        assert ax.get_legend() is None

    def test_by_water_year_overlays_on_a_month_axis(self, obs):
        ax = plotting.timeseries(obs["swe"].sel(station="a"), by_water_year=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["WY2023", "WY2024"]
        lo, hi = ax.get_xlim()
        assert 364 <= hi - lo <= 366  # one synthetic water year of days
        assert "1 October" in ax.get_xlabel()
        with pytest.raises(ValueError, match="one series"):
            plotting.timeseries(obs["swe"], by_water_year=True)

    def test_many_lines_move_the_legend_outside(self, obs):
        wide = obs["swe"].isel(station=[0] * 6).assign_coords(station=list("abcdef"))
        wide = wide.drop_vars(["name", "elevation_m"])
        ax = plotting.timeseries(wide)
        assert ax.get_legend().get_bbox_to_anchor().x0 >= ax.bbox.x1


class TestOther:
    def test_register_colormap(self):
        cmap = plotting.register_colormap("esd_test_cmap", ["#000000", "#ffffff"])
        assert matplotlib.colormaps["esd_test_cmap"] is not None and cmap.N == 2
        same = plotting.register_colormap("esd_test_cmap", ["#ff0000"])
        assert same.N == 2  # kept without overwrite
        new = plotting.register_colormap("esd_test_cmap", ["#ff0000"], overwrite=True)
        assert new.N == 1 and matplotlib.colormaps["esd_test_cmap"].N == 1

    def test_graticule_step_is_nice(self):
        assert plotting._nice_step(0.4) == 0.1
        assert plotting._nice_step(12) == 2
        assert plotting._nice_step(0.27) == 0.05
        assert plotting._format_lon(-121.8) == "121.8°W"
        assert (
            plotting._format_lat(46.9) == "46.9°N" and plotting._format_lat(0) == "0°"
        )
