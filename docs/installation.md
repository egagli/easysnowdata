# Installation

::::{tab-set}

:::{tab-item} pip
```bash
pip install easysnowdata
```
:::

:::{tab-item} conda / mamba
```bash
conda install -c conda-forge easysnowdata
# or
mamba install -c conda-forge easysnowdata
```
:::

:::{tab-item} pixi
```bash
pixi add easysnowdata
```
:::

:::{tab-item} uv
```bash
uv pip install easysnowdata
```
:::

::::

**One install, no extras** (decision Q13). Everything in the catalog is
reachable from a plain `pip install easysnowdata` — there is no
`easysnowdata[sar]` to remember, and no import that fails because an optional
dependency is missing. The cost is a larger install; the benefit is that a
snippet from these docs runs in anyone's environment.

Python **3.12 or newer** is required. The floor comes from the stack, not from
us: rasterio 1.5 (GDAL ≥ 3.8), rioxarray 0.23, zarr 3.3, numpy 2.5 and
earthaccess 0.18 have all dropped 3.11.

## conda-forge or PyPI?

Both work. Prefer conda-forge when you can, for two reasons specific to this
package:

- **GDAL drivers.** Reading MODIS `MOD10A1F` granules needs GDAL's HDF4
  driver, which conda-forge splits into its own `libgdal-hdf4` package and
  which the PyPI `rasterio` wheels do not carry. Without it that one product
  raises a driver error; everything else is unaffected.
- **PROJ data.** The conda-forge build ships a complete PROJ grid set, which
  matters for accurate vertical and datum transforms.

## Development install

The repository is a [pixi](https://pixi.sh) workspace, and `pixi.lock` pins
every environment down to the GDAL build, so CI and your laptop resolve to the
same stack.

```bash
git clone https://github.com/egagli/easysnowdata.git
cd easysnowdata
pixi install                 # the default environment
pixi run -e dev test-unit    # offline tests: no network, no credentials
```

The environments and what they are for:

| environment | contents | used by |
| --- | --- | --- |
| `default` | the package and its runtime dependencies | `pixi run python` |
| `dev` | plus pytest, pytest-socket, pytest-recording, ruff, jupyter | the test tiers |
| `docs` | plus Sphinx, sphinx-gallery, myst-nb, pydata-sphinx-theme | building this site |
| `lint` | ruff only | the fast CI lint job |
| `test-py312`, `test-py313`, `test-py314` | `dev` on one interpreter | the CI test matrix |

[Contributing](contributing.md) lists the tasks each environment offers.

## Checking the install

```python
import easysnowdata as esd

print(esd.__version__)
esd.auth.status()  # which providers are configured; network-free
esd.catalog.list()  # the 28 products
```

On an interactive session `import easysnowdata` prints a single line naming
the version, which credentials it found and whether it is running somewhere
with direct S3 access. It makes no network request to do that. Set
`EASYSNOWDATA_QUIET=1` to silence it.

## Credentials

Most of the catalog is open. The products that are not say so on their
[catalog page](catalog/index.md), and [Credentials](credentials.md) has the
setup for each provider. Nothing needs to be configured before installing.
