"""Snow-station observations from five public networks (§9).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)                 # Mount Rainier
    inv = esd.stations.inventory(aoi, daily_only=True)      # GeoDataFrame
    obs = esd.stations.load(inv, variables=["swe", "snwd"],
                            time="2023-10/2024-06")         # (station, time)

Two layers, as ``global_snow_networks``' DESIGN.md §2 lays them out:

:mod:`~easysnowdata.stations.clients`
    Pure access, moved here verbatim with its history: one client per network
    (AWDB, CDEC, DataBC, NVE, Yukon), each answering ``get_all_stations`` /
    ``get_data`` / ``get_metadata`` with plain dict records in metric units.
    Use it directly when you want flags, a native variable name or an
    interval the adapter does not surface.
this module
    :func:`inventory` and :func:`load`, which put those records into the
    package's own return types and take ``aoi`` and ``time`` like every other
    loader.

The ``station`` × ``time`` grid comes back dense with NaN where a station has
no observation. That is what xarray, Dask and ``groupby`` want, and a station
record is only sparse in the everyday sense — nothing here needs a sparse
array backend. Station positions are ``latitude`` and ``longitude``
coordinates rather than an ``xvec`` geometry coordinate: the plan lists
``xvec`` as optional, it would be a new dependency, and float coordinates
round-trip through Zarr and netCDF where a geometry object does not.

Station identity has two spellings, both accepted everywhere: the globally
unique ``code`` (``"679_WA_SNTL"``) and the network's own station id
(``"679:WA:SNTL"``). See :mod:`easysnowdata.stations.networks`.
"""

from __future__ import annotations

from easysnowdata.stations import _catalog, _frames, clients, networks
from easysnowdata.stations._adapter import PRODUCT_IDS, inventory, load, metadata

__all__ = [
    "PRODUCT_IDS",
    "_frames",
    "clients",
    "inventory",
    "load",
    "metadata",
    "networks",
]

# Registering the catalog entries is this module's import side effect, the
# same way every Phase 2 product module registers its own (§3.3).
_ = _catalog
