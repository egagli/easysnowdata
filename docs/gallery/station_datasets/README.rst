Other station datasets
======================

Station data that is **not** part of the five daily networks above. These
datasets are reached only through their clients in
``easysnowdata.stations.clients``: they are not in ``esd.stations.inventory()``,
``esd.stations.load()``, the catalog, or the pre-downloaded daily archive that
``global_snow_networks`` publishes, and they are not mixed into the "all
networks" example.

They differ from the daily networks in what they measure and how often.

- **Periodic SWE.** Manual snow courses and route surveys measured every one to
  four weeks: CanSWE and NorSWE (mostly), Finland's SYKE, Switzerland's SLF and
  Russia's RIHMI-WDC.
- **Snow depth, daily.** GHCN-Daily worldwide, DWD in Germany: many stations,
  but daily SWE at only a few.
- **Daily SWE that is new.** Hydro-Québec's gamma sensors (a rolling 10-day
  feed), the automated stations in CanSWE, and Alberta's pillows (which AWDB
  also mirrors).

Every record carries an ``interval`` (``"daily"``, ``"periodic"``, ``"hourly"``,
…), so check it before treating a series as daily. The clients return plain
lists of dicts; the examples turn them into pandas and geopandas objects in a
few lines.
