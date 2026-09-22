"""SAR backscatter helpers: dB conversion and Sentinel-1 border-noise removal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "linear_to_db",
    "db_to_linear",
    "remove_border_noise",
    "slope_aspect",
    "look_azimuth",
    "local_incidence_angle",
    "S1_BORDER_NOISE_CUTOFF",
]

#: ESA's IPF 2.90 (2018-03-13) fixed border noise; earlier scenes need a threshold.
S1_BORDER_NOISE_CUTOFF = "2018-03-14"


def linear_to_db(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """``10 log10(x)``; non-positive values become NaN. Sets ``units="dB"``."""
    positive = obj.where(obj > 0)
    out = 10 * np.log10(positive)
    _set_units(out, "dB")
    return out


def db_to_linear(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """``10 ** (x / 10)``. Sets ``units="linear power"``."""
    out = 10 ** (obj / 10)
    _set_units(out, "linear power")
    return out


def _set_units(obj: xr.DataArray | xr.Dataset, units: str) -> None:
    obj.attrs["units"] = units
    if isinstance(obj, xr.Dataset):
        for var in obj.data_vars.values():
            var.attrs["units"] = units


def remove_border_noise(
    obj: xr.DataArray | xr.Dataset,
    threshold: float = 0.001,
    *,
    cutoff: str | pd.Timestamp = S1_BORDER_NOISE_CUTOFF,
    dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Mask falsely low backscatter (border noise) in linear-power data.

    Scenes before *cutoff* keep values above *threshold*; later scenes keep
    values above 0. CRS metadata is preserved.
    """
    if dim not in obj.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input.")
    is_old = obj[dim] < np.datetime64(pd.Timestamp(cutoff))
    limit = xr.where(is_old, threshold, 0.0)
    return obj.where(obj > limit)


# ── terrain geometry for the local incidence angle ───────────────────────────


def slope_aspect(
    dem: xr.DataArray, *, degrees: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """Slope and aspect of a projected DEM, by central differences.

    Parameters
    ----------
    dem
        Elevation in metres on a projected grid whose ``x``/``y`` coordinates
        are metres too (a UTM DEM). The spacing is read from the coordinates.
    degrees
        Return degrees (default) rather than radians. Aspect is measured
        clockwise from north, in ``[0, 360)``.

    Returns
    -------
    (slope, aspect)
        Two DataArrays shaped like *dem*.
    """
    for dim in ("x", "y"):
        if dim not in dem.dims:
            raise ValueError(
                f"slope_aspect needs a projected DEM with x/y dims; got {dem.dims}."
            )
    dx = float(np.abs(np.diff(dem["x"].values[:2])[0])) if dem.sizes["x"] > 1 else 1.0
    dy = float(np.abs(np.diff(dem["y"].values[:2])[0])) if dem.sizes["y"] > 1 else 1.0
    values = dem.astype("float64")
    # np.gradient over (y, x); y decreases downward on a north-up grid, so the
    # sign of the y derivative is flipped to point north.
    gradients = xr.apply_ufunc(
        lambda block: np.stack(np.gradient(block, dy, dx), axis=-1),
        values,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x", "gradient"]],
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs={
            "output_sizes": {"gradient": 2},
            "allow_rechunk": True,
        },
    )
    dz_dy = -gradients.isel(gradient=0, drop=True)
    dz_dx = gradients.isel(gradient=1, drop=True)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dx, dz_dy) % (2 * np.pi)
    if degrees:
        slope = np.degrees(slope)
        aspect = np.degrees(aspect)
    slope = slope.rename("slope").assign_attrs(
        long_name="terrain slope", units="degrees" if degrees else "radians"
    )
    aspect = aspect.rename("aspect").assign_attrs(
        long_name="terrain aspect clockwise from north",
        units="degrees" if degrees else "radians",
    )
    return slope, aspect


def look_azimuth(heading: float, *, looking: str = "right") -> float:
    """Radar look azimuth in degrees from the platform heading.

    Sentinel-1 is right-looking, so the look direction is the heading plus
    90°. Pass ``looking="left"`` for a left-looking sensor.
    """
    offset = 90.0 if looking == "right" else -90.0
    return float((heading + offset) % 360.0)


def local_incidence_angle(
    dem: xr.DataArray,
    incidence_angle: xr.DataArray | float,
    look_azimuth_deg: float | xr.DataArray,
    *,
    clip_to_valid: bool = True,
) -> xr.DataArray:
    """Local incidence angle from a DEM and the ellipsoidal incidence angle.

    The credential-free counterpart of the OPERA RTC-S1-STATIC layer, and the
    replacement for the 350-line Earth Engine routine (issue #10). Pure
    xarray, so it works on any DEM and is unit-testable on a synthetic slope.

    Parameters
    ----------
    dem
        Elevation in metres on a projected (metric) grid.
    incidence_angle
        Ellipsoidal incidence angle in degrees: a scalar for a scene, or a
        raster aligned with *dem*.
    look_azimuth_deg
        Radar look azimuth in degrees clockwise from north (see
        :func:`look_azimuth`). The heading it comes from varies with latitude,
        so take it from a scene footprint, as
        :func:`easysnowdata.sar.sentinel1.scene_geometry` does, not from a
        constant.
    clip_to_valid
        Clip the result to ``[0, 90]`` degrees; ``False`` keeps the raw
        arccosine, which exceeds 90° where the slope faces away from the
        radar (layover and shadow candidates).

    Returns
    -------
    xarray.DataArray
        Local incidence angle in degrees, shaped like *dem*.

    Notes
    -----
    Following the standard radar-geometry formulation (Small 2011; the
    Earth Engine recipe at https://gis.stackexchange.com/a/352658):

    ``cos(θ_lia) = cos(α_az) · cos(θ_i − α_r)``

    where ``α_r`` and ``α_az`` are the terrain slope components in the range
    and azimuth directions.
    """
    slope, aspect = slope_aspect(dem, degrees=False)
    look_rad = np.radians(look_azimuth_deg)
    theta_i = np.radians(incidence_angle)
    # The look azimuth points from the sensor to the ground; a slope faces the
    # radar when its aspect points back along it, so the range component is
    # measured from the direction *toward* the sensor (look + 180°). Checked
    # against the OPERA RTC-S1-STATIC layer for a descending track (2026-09).
    phi_r = look_rad + np.pi - aspect
    alpha_r = np.arctan(np.tan(slope) * np.cos(phi_r))
    alpha_az = np.arctan(np.tan(slope) * np.sin(phi_r))
    cos_theta = (np.cos(alpha_az) * np.cos(theta_i - alpha_r)).clip(-1.0, 1.0)
    lia = np.degrees(np.arccos(cos_theta))
    if clip_to_valid:
        lia = lia.clip(0.0, 90.0)
    lia = lia.rename("local_incidence_angle")
    lia.attrs = {
        "long_name": "local incidence angle",
        "units": "degrees",
        "description": (
            "angle between the radar look vector and the terrain normal, "
            "computed from a DEM"
        ),
    }
    if hasattr(dem, "rio") and dem.rio.crs is not None:
        lia = lia.rio.write_crs(dem.rio.crs)
    return lia
