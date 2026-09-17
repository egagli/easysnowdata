# Long-form notebooks

These are the original per-module tutorial notebooks. They cover more ground
per page than the [gallery](auto_examples/index.rst) — several products, side
by side, with commentary — and they are the best introduction if you are
reading rather than running.

Two caveats. They were executed by hand against live data on the dates shown
in their outputs, and they are **not** re-executed when these docs are built,
so an output can lag the current release. And they still call the pre-0.1
module API (`easysnowdata.remote_sensing.get_*`), which works today through
deprecation shims but is removed in the release after next. The
[gallery](auto_examples/index.rst) is the executed, current-API counterpart;
each of its pages ends with a link back to the catalog entry for the product.

```{toctree}
:maxdepth: 1

examples/how_easy
examples/automatic_weather_station_examples
examples/hydroclimatology_examples
examples/remote_sensing_examples
examples/topography_examples
examples/sentinel2_planetarycomputer_vs_earthaccess
```
