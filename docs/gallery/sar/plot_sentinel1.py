# esd-requires: earthdata
"""
Sentinel-1 RTC backscatter and local incidence angle
====================================================

Sentinel-1 is C-band radar: it images through cloud, by day and by night, and
its backscatter responds to the snowpack's liquid water content and structure.
That gives it a wide variety of applications, among them snowmelt phase
delineation, wet snow detection and snow depth estimation. Radiometrically
terrain-corrected (RTC) gamma0 removes the brightening and darkening the
terrain itself imposes, so that slopes facing the radar can be compared with
slopes facing away.

Three routes serve the backscatter. ``source="planetary-computer"`` (the
default) is 10 m scene-based RTC from 2014 onward, with no account.
``source="opera-rtc-s1"`` is NASA/JPL's 30 m burst-based OPERA RTC-S1 on ASF,
read through CMR-STAC with an Earthdata Login, and the only one that ships a
layover and shadow mask. ``source="gee"`` is the same OPERA product re-served
by Earth Engine (Earth Engine credentials).

The local incidence angle, the angle between the radar beam and the terrain
normal, has three routes of its own: the OPERA static layers (Earthdata),
a computation from any DEM (no account at all) and the legacy Earth Engine
routine. The figures show one pass in both polarisations, a summer of passes
split by relative orbit, the computed angle for both orbit directions, and the
Planetary Computer and OPERA products side by side with OPERA's mask.
"""

import matplotlib.pyplot as plt
import numpy as np

import easysnowdata as esd

aoi = (-121.94, 46.72, -121.54, 46.99)  # Mount Rainier
nisqually = (-121.80, 46.82, -121.72, 46.88)  # its Nisqually glacier side

esd.parse_aoi(aoi)  # the AOI as every loader below will see it

# %%
# Where the two products come from, and what each route asks for.
for product in ("sentinel-1-rtc", "sentinel-1-local-incidence-angle"):
    for src in esd.catalog.get(product).sources:
        print(
            f"{product:34} {src.id:20} {src.title:40} "
            f"{', '.join(src.requires) or 'no account'}"
        )

# %%
# Two weeks of VV and VH in dB, one raster per pass. The orbit direction and
# relative orbit ride along as coordinates on ``time``: four tracks cross this
# mountain every twelve days, two ascending (evening) and two descending
# (morning).
s1 = esd.sar.sentinel1.load(
    aoi, "2023-08-01/2023-08-15", bands=["vv", "vh"], resolution=40
).compute()
for when, state, orbit in zip(
    s1["time"].values, s1["sat:orbit_state"].values, s1["sat:relative_orbit"].values
):
    print(f"{str(when)[:19]}  {state:10}  relative orbit {orbit}")

# %%
# One pass in both polarisations. VH is 6–8 dB weaker than VV over the same
# ground; the fixed −25 to 0 dB range keeps the two panels comparable.
first = s1.isel(time=0)
stamp = (
    f"{str(first['time'].values)[:10]}, {str(first['sat:orbit_state'].values)} "
    f"orbit {int(first['sat:relative_orbit'])}"
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
esd.plotting.map(
    first["vv"], ax=axes[0], cmap="Greys_r", vmin=-25, vmax=0,
    title=f"VV, {stamp}", cbar_label="gamma0 VV [dB]",
)  # fmt: skip
esd.plotting.map(
    first["vh"], ax=axes[1], cmap="Greys_r", vmin=-25, vmax=0,
    title=f"VH, {stamp}", cbar_label="gamma0 VH [dB]",
)  # fmt: skip
fig.tight_layout()

# %%
# A summer of passes over the glacier as a time series. Each track looks at
# the mountain from its own direction, and the mean backscatter of the same
# box differs by about 2 dB between tracks for that reason alone. Mixed into
# one series that becomes a twelve-day sawtooth with nothing to do with the
# snow, which is why backscatter time series are split by relative orbit.
summer = esd.sar.sentinel1.load(
    nisqually, "2023-06-01/2023-09-30", bands=["vv"], resolution=100
)
mean_vv = summer["vv"].mean(dim=["x", "y"]).compute()
mean_vv.attrs = {"long_name": "mean gamma0 VV over the box", "units": "dB"}

fig, ax = plt.subplots(figsize=(9, 4.2))
for orbit in np.unique(mean_vv["sat:relative_orbit"].values):
    passes = mean_vv.where(mean_vv["sat:relative_orbit"] == orbit, drop=True)
    state = str(passes["sat:orbit_state"].values[0])
    esd.plotting.timeseries(
        passes,
        ax=ax,
        marker="o",
        markersize=4,
        label=f"relative orbit {orbit} ({state})",
    )
ax.set_title("Sentinel-1 VV over the Nisqually glacier, summer 2023")
fig.tight_layout()

# %%
# The local incidence angle is a property of one acquisition geometry: which
# track (relative orbit) the scene came from, whether the pass was ascending
# or descending, and where in the swath a pixel sits. The published answer is
# OPERA's per-burst static layer (the default ``source="opera-static"``,
# Earthdata Login). The credential-free ``source="dem"`` approximates it: the
# heading and look direction come from a representative Planetary Computer
# scene of the chosen track, the ellipsoidal incidence angle grows linearly
# across that scene's swath, and the terrain part is computed from the
# Copernicus DEM. Ascending passes look east from the west (Sentinel-1 is
# right-looking), descending passes look west from the east, so a slope that
# faces one track at a grazing angle faces the other almost head-on.
ascending = esd.sar.sentinel1.local_incidence_angle(
    aoi, source="dem", orbit_state="ascending", resolution=60
).compute()
descending = esd.sar.sentinel1.local_incidence_angle(
    aoi, source="dem", orbit_state="descending", resolution=60
).compute()
for name, ds in (("ascending", ascending), ("descending", descending)):
    print(
        f"{name}: track {ds.attrs['relative_orbit']}, heading "
        f"{ds.attrs['platform_heading']:.1f}°, look azimuth "
        f"{ds.attrs['look_azimuth']:.1f}°"
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
esd.plotting.map(
    ascending["local_incidence_angle"], ax=axes[0], cmap="magma", vmin=0, vmax=90,
    title="Local incidence angle, ascending",
)  # fmt: skip
esd.plotting.map(
    descending["local_incidence_angle"], ax=axes[1], cmap="magma", vmin=0, vmax=90,
    title="Local incidence angle, descending",
)  # fmt: skip
fig.tight_layout()

difference = ascending["local_incidence_angle"] - descending["local_incidence_angle"]
print(f"max |ascending − descending|: {float(np.nanmax(np.abs(difference))):.1f}°")

# %%
# How good is the approximation? Against OPERA's published layer for the
# same descending track (this cell needs an Earthdata Login) the DEM route
# agrees to a median of about 5° on open slopes; it knows nothing about
# layover and shadow, which OPERA masks. Pass ``relative_orbit=`` to keep one
# track's bursts, or the fused raster mixes geometries.
track = int(descending.attrs["relative_orbit"])
opera_lia = esd.sar.sentinel1.local_incidence_angle(
    aoi, source="opera-static", relative_orbit=track, resolution=60
).compute()
approx = descending["local_incidence_angle"]
published = opera_lia["local_incidence_angle"].reindex_like(approx, method="nearest")
gap = (approx - published).compute()
print(
    f"track {track}: median |DEM route − OPERA| = "
    f"{float(np.nanmedian(np.abs(gap))):.1f}°"
)

# %%
# The same descending pass from both catalogs (this cell needs an Earthdata
# Login). Planetary Computer's scene-based RTC and OPERA's burst-based RTC are
# processed by different pipelines from the same GRD, both against the
# Copernicus DEM, and read alike to within a dB. They part ways where the
# geometry defeats terrain correction: OPERA leaves radar shadow and layover
# as no data (the white patches) and says which is which in its mask, while
# Planetary Computer fills those pixels with values no RTC can make physical.
pc = esd.sar.sentinel1.load(nisqually, "2023-08-15", bands=["vv"], resolution=30)
opera = esd.sar.sentinel1.load(
    nisqually, "2023-08-15", bands=["vv"], source="opera-rtc-s1", resolution=30
)
pc, opera = pc.compute(), opera.compute()
print(
    f"median VV: Planetary Computer {float(pc['vv'].median()):.1f} dB, "
    f"OPERA {float(opera['vv'].median()):.1f} dB"
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
esd.plotting.map(
    pc["vv"], ax=axes[0], cmap="Greys_r", vmin=-25, vmax=0,
    title="VV, Planetary Computer RTC, 2023-08-15", cbar_label="gamma0 VV [dB]",
)  # fmt: skip
esd.plotting.map(
    opera["vv"], ax=axes[1], cmap="Greys_r", vmin=-25, vmax=0,
    title="VV, OPERA RTC-S1, 2023-08-15", cbar_label="gamma0 VV [dB]",
)  # fmt: skip
esd.plotting.categorical(opera["mask"], ax=axes[2], title="OPERA layover and shadow")
for ax in axes[1:]:
    ax.set_ylabel("")  # the panels share their latitude axis
fig.tight_layout()
