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


class TestInternals:
    """The branches the maps take on unusual inputs."""

    def test_crs_of_falls_back_and_gives_up(self):
        plain = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        assert plotting._crs_of(plain) is None
        assert plotting._crs_of(object()) is None
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0], [0]), crs="EPSG:32610")
        assert plotting._crs_of(gdf).to_epsg() == 32610

    def test_axis_names_follow_the_crs_units(self):
        from pyproj import CRS

        assert plotting._axis_names(None) == ("x", "y")
        assert plotting._axis_names(CRS("EPSG:32610")) == (
            "easting [m]",
            "northing [m]",
        )
        assert plotting._axis_names(CRS("EPSG:4326")) == (
            "longitude [°]",
            "latitude [°]",
        )
        feet = plotting._axis_names(CRS("EPSG:2285"))  # Washington North, US feet
        assert feet[0].startswith("easting [") and feet[1].startswith("northing [")

    def test_mid_latitude_on_projected_axes(self, projected):
        fig, ax = plt.subplots()
        projected.plot.imshow(ax=ax, add_colorbar=False)
        lat = plotting._mid_latitude(ax, projected.rio.crs)
        assert 46.5 < lat < 47.5

    def test_nice_step_and_crossing_edge_cases(self):
        assert plotting._nice_step(1e6) == plotting._GRATICULE_STEPS[-1]
        assert (
            plotting._crossing(
                np.array([0.0, 1.0]), np.array([5.0, 6.0]), 10.0, axis="y"
            )
            is None
        )
        # a segment lying exactly on the level
        assert (
            plotting._crossing(
                np.array([0.0, 1.0]), np.array([5.0, 5.0]), 5.0, axis="y"
            )
            == 0.0
        )

    def test_graticule_accepts_a_tuple_step(self, projected):
        fig, ax = plt.subplots()
        projected.plot.imshow(ax=ax, add_colorbar=False)
        plotting.add_graticule(ax, projected.rio.crs, step=(0.05, 0.02))
        assert any("°" in t.get_text() for t in ax.get_xticklabels())

    def test_graticule_on_a_global_projected_map(self):
        # Robinson's corners lie outside the projection and transform to inf;
        # the graticule used to take its range from the edges alone and crash.
        from pyproj import CRS

        fig, ax = plt.subplots()
        ax.set_xlim(-17_005_833, 17_014_167)
        ax.set_ylim(-8_634_845, 8_625_155)
        plotting.add_graticule(ax, CRS.from_user_input("ESRI:54030"))
        assert len(ax.lines) > 10  # meridians and parallels across the globe

    def test_corner_notes_stack_into_one_line(self, projected):
        fig, ax = plt.subplots()
        plotting._corner_note(ax, "one")
        plotting._corner_note(ax, "two")
        assert len(ax.texts) == 1 and ax.texts[0].get_text() == "one · two"

    def test_crs_name_for_a_sinusoidal_grid(self):
        from pyproj import CRS

        sinusoidal = CRS(
            "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
        )
        assert plotting._crs_name(sinusoidal) == "Sinusoidal"
        assert plotting._crs_name(CRS("EPSG:32610")) == "EPSG:32610"

    def test_provider_lookup(self):
        import xyzservices.providers as xyz

        assert plotting._provider(True) is xyz.Esri.WorldShadedRelief
        assert plotting._provider("CartoDB.Positron") is xyz.CartoDB.Positron
        assert plotting._provider(xyz.CartoDB.Positron) is xyz.CartoDB.Positron

    def test_basemap_success_credits_the_tiles(self, projected, monkeypatch):
        import contextily

        monkeypatch.setattr(contextily, "add_basemap", lambda ax, **kw: None)
        fig, ax = plt.subplots()
        projected.plot.imshow(ax=ax, add_colorbar=False)
        plotting.add_basemap(ax, projected.rio.crs, "CartoDB.Positron")
        assert ax.texts and ax.texts[0].get_text().startswith("basemap: ")

    def test_figsize_and_spatial_dims_edge_cases(self):
        assert plotting._figsize_for((0.0, 0.0, 0.0, 0.0), 1.0) == (7.0, 5.25)
        odd = xr.DataArray(np.zeros((3, 4)), dims=("row", "col"))
        assert plotting._spatial_dims(odd) == ("col", "row")
        with pytest.raises(ValueError, match="two spatial"):
            plotting._spatial_dims(xr.DataArray(np.zeros(3), dims=("t",)))

    def test_time_suffix_ignores_an_unparseable_time(self, projected):
        odd = projected.assign_coords(time="not a date")
        assert plotting._time_suffix(odd) == ""
        assert plotting._default_title(odd) == "snow water equivalent"

    def test_single_point_gets_an_extent(self):
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([-121.9], [46.8]), crs="EPSG:4326"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", plotting.GeographicAxesWarning)
            ax = plotting.points(gdf, basemap=False)
        x0, x1 = ax.get_xlim()
        assert x1 > x0

    def test_series_label_tolerates_a_bad_elevation(self):
        da = xr.DataArray(
            np.zeros((1, 3)),
            dims=("station", "time"),
            coords={
                "station": ["a"],
                "time": pd.date_range("2024-01-01", periods=3),
                "elevation_m": ("station", ["n/a"]),
            },
        )
        assert plotting._series_label(da, "a", "station") == "a"

    def test_timeseries_single_series_dataset_and_errors(self):
        time = pd.date_range("2024-01-01", periods=5)
        da = xr.DataArray(
            np.arange(5.0), dims=("time",), coords={"time": time}, name="swe"
        )
        ax = plotting.timeseries(xr.Dataset({"swe": da}), title="one")
        assert ax.get_title() == "one" and len(ax.lines) == 1
        with pytest.raises(ValueError, match="time"):
            plotting.timeseries(xr.DataArray([1.0, 2.0], dims=("x",)))
        two = xr.DataArray(
            np.zeros((2, 5)),
            dims=("station", "time"),
            coords={"time": time, "station": ["a", "b"]},
        )
        with pytest.raises(ValueError, match="hue must be"):
            plotting.timeseries(two, hue="nope")
