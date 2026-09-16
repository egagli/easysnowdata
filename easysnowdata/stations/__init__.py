"""Snow-station observations from five public networks (§9).

The access layer is :mod:`easysnowdata.stations.clients`, moved here verbatim
from `global_snow_networks <https://github.com/egagli/global_snow_networks>`_
with its history: one client per network, each answering
``get_all_stations`` / ``get_data`` / ``get_metadata`` with plain dict records
(that repo's ``DESIGN.md`` §3.4), in metric units, with one shared retry
policy.

::

    from easysnowdata.stations.clients import AWDBClient

    AWDBClient().get_data(["679:WA:SNTL"], variables=["swe"],
                          begin_date="2023-10-01", end_date="2024-06-30")

The xarray/GeoDataFrame adapter on top of them arrives in §9 step 2.
"""

from __future__ import annotations

from easysnowdata.stations import clients

__all__ = ["clients"]
