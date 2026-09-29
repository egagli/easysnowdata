Example gallery
===============

One short script per product: load it over one area of interest, say which
routes serve it, and draw one or two figures. Every script is executed when the
docs are built, so a figure on this page is a figure the package produced
against live data on that day.

The sections follow the package: ``esd.stations``, ``esd.snow``, ``esd.sar``,
``esd.optical``, ``esd.terrain``, ``esd.land``, ``esd.hydro``, ``esd.boundaries``, and
``esd.climate``, then the cross-cutting tools. Pushes to ``main`` and the weekly
scheduled build run every script with the provider credentials
(:doc:`/credentials`); a pull-request build runs only the scripts that need no
account and reuses the last output of the rest.

Each page offers the script and a Jupyter notebook to download.
