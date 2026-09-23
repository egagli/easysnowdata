# Contributing

Bug reports, new data products and documentation fixes are all welcome. The
issue tracker is at <https://github.com/egagli/easysnowdata/issues>.

## Setting up

```bash
git clone https://github.com/egagli/easysnowdata.git
cd easysnowdata
pixi install                         # resolves from pixi.lock; conda-forge
pixi run -e test-py313 test-unit     # should be green before you change anything
```

**Test in `test-py313`, not `dev`.** `pixi.lock` pins every environment, but the
per-interpreter environments have no `solve-group`, so they resolve
independently of `dev` and land on a different stack — at one point `dev` had
rasterio 1.5.0 / GDAL 3.12.3 while `test-py313` had 1.5.1 / 3.13.3. That is
deliberate (testing three interpreters against their own best resolutions is
the point of the matrix) but it means **`dev` is for interactive work — a
notebook, a REPL, a docs build — and `test-py3XX` is what CI runs**. CI now has
a `dev` leg too, so the gap cannot silently reopen (#20).

Never hand-edit the lock; `pixi install` regenerates it when `pyproject.toml`
changes, and CI runs with `locked: true` so a stale lock fails the build rather
than silently resolving something else. One caveat worth knowing: `pixi lock`
exits 0 even when it drops a dependency it cannot satisfy, so check that a new
dependency is actually in `pixi.lock` rather than trusting the exit code.

## The tasks

| task | what it does |
| --- | --- |
| `pixi run -e test-py313 test-unit` | the whole offline tier, sockets blocked |
| `pixi run -e test-py313 test-recorded` | just the cassette-replayed tests |
| `pixi run -e test-py313 test-cov` | offline tier with coverage (fails under 95%) |
| `pixi run -e test-py313 test-live` | live tier: one smoke test per source |
| `pixi run -e test-py313 record-cassettes` | re-record the cassettes (needs network) |
| `pixi run -e lint lint` / `format` | ruff check / ruff format |
| `pixi run -e docs docs-fast` | build the site without running the gallery (~17 s) |
| `pixi run -e docs docs-build-free` | build and run the credential-free gallery (~4 min) |
| `pixi run -e docs docs-build` | build and run the whole gallery (needs every credential) |
| `pixi run -e docs docs-serve` | serve `docs/_build/html` on port 8000 |
| `pixi run -e dev check-sources` | run the health probes and update the history |
| `pixi run -e dev report-health` | print the issues the last health run would open |
| `pixi run -e dev watch` | run the upstream watch and print the digest, writing nothing |

## Credentials on your machine

CI holds every provider secret, and the scheduled docs build and the weekly
health check use them; GitHub never hands a secret back, so the same values
have to reach your shell separately. Two ways that need no code:

- **A `.env` file at the repo root** (already in `.gitignore`), one
  `NAME=value` per line, loaded for a shell with `set -a; . ./.env; set +a`
  before `pixi run …`. Good for `NVE_API_KEY`, `PL_API_KEY`,
  `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` and `EARTHENGINE_TOKEN`.
- **The provider's own store**: `planet auth login` saves a Planet session
  under `~/.planet/`, `earthaccess.login(persist=True)` writes `~/.netrc`, and
  `earthengine authenticate` writes `~/.config/earthengine/credentials`. The
  auth providers read all of these without any variable set.

`esd.auth.status()` shows what was found and how. The `requires_*` test
markers and the gallery's `# esd-requires:` lines skip cleanly when a
credential is absent, so a partial set is fine for local work.

## The four test tiers

Each tier answers a different question, and they are deliberately separate so
that a provider outage cannot turn a pull request red.

**unit** — pure functions, no I/O. Sockets are blocked with `pytest-socket`
(`--allow-unix-socket`, because zarr 3's asyncio loop needs a socketpair), so a
test that quietly reaches the network fails instead of passing by accident.

**recorded** — searches and reads replayed from `tests/cassettes/` with
`pytest-recording`, plus the tiny generated rasters in `tests/fixtures/`. Also
offline, also socket-blocked, `--record-mode=none` so a missing cassette is a
failure rather than a live request. Re-record with `record-cassettes` when a
search changes, and check the diff: cassettes are scrubbed of cookies and
tokens but you are still committing someone's API response.

**live** — one smoke test per source against the real provider, marked `live`
and deselected everywhere else. Runs weekly in CI and on demand. Tests marked
`requires_earthdata` / `requires_earthengine` / `requires_planet` /
`requires_nve` **skip** when the credential is absent, so a contributor without
an Earth Engine account still gets a meaningful run.

**docs** — the gallery is a test tier in disguise: every credential-free
example is executed on every pull request, and every push to `main` (and the
weekly scheduled build) executes the whole gallery against live data with the
secrets, so the published pages show every example's real output.

Run the offline tiers before every commit and the live tier before claiming a
data route works.

## Adding a data product

The catalog entry is the contract: it generates the docs page, the health row
and the credential table, so the artefacts cannot drift apart. In order:

```{card} 1. Write the loader
Put it in `easysnowdata/<theme>/<product>.py` with module-level `search` and
`load` functions. Take `aoi` first and `time` second; accept `source=`; return
lazy xarray. Build the return value with `processing.contract.finalize()` so
the dims, CRS, nodata policy and provenance attributes come out right without
you re-deciding them. A vector product returns a GeoDataFrame through
`processing.contract.finalize_frame()` instead (EPSG:4326, 2-D, provenance in
`.attrs`); fetch its archive once with `providers.raster_http.fetch` and read
it with `providers.vector_http.read`, which pushes the AOI down. Read [Concepts](concepts.md) once before you start.
```

```{card} 2. Add the catalog entry
A `Product` with one `Source` per access route — the first is the default.
Every source needs a `location`, an `extent`, a `notes` line saying what
differs from the other routes, and a `health=Probe(...)`. Register it at the
bottom of the module with `catalog.register(PRODUCT, replace=True)`, and
import the module from the theme's `__init__.py` so registration happens.

Give every variable its units, dtype and nodata. Categorical variables need
`flag_values`, `flag_meanings` and `flag_colors` of equal length — the offline
test checks that, and the plotting helpers read them.
```

```{card} 3. Two tests
One offline: a recorded search or a tiny fixture raster, asserting the shape
of the contract (dims, CRS, dtype, attrs) rather than the values. One live:
marked `@pytest.mark.live`, plus `requires_*` if it needs credentials, doing
the smallest real request that proves the route works.
```

```{card} 4. A health probe
The minimal request that shows the route is alive: a first-byte GET for a
static file, a one-item STAC search, a `getInfo()` for Earth Engine, an
`earthaccess` search for NSIDC. Reuse the helpers in
`easysnowdata.catalog.health`. **GET first, not HEAD** — some servers (GRDC)
answer HEAD with a 400 that looks exactly like a 404. For a file behind
Earthdata Login that is not in CMR (NSIDC's `daacdata` tree), use
`earthdata_https_first_byte`: an anonymous GET there lands on the login page
with a 200, which a plain first-byte probe would count as a pass.

Give the probe a stable `label`. It is the key the weekly history is stored
under, so renaming one starts its history over.
```

```{card} 5. A gallery example
`docs/gallery/<theme>/plot_<product>.py`: load it for one AOI, draw one or two
figures, say one thing that is true about the product and not obvious. Use the
sphinx-gallery format — a docstring title with an `===` underline of exactly
the right length, then `# %%` cells. List it in the catalog entry's
`examples=` tuple.

House style: the title names the product and its provider (`SNODAS snow water
equivalent and depth (NOHRSC)`), never a place. A cell near the top prints the
product's routes (`esd.catalog.get(pid).sources`). Maps go through
`esd.plotting.map` / `categorical` / `points`, vectors over them through
`esd.plotting.add_outline`, time series through
`esd.plotting.timeseries`, so every figure has equal aspect, a scale bar, a
graticule, units in `[ ]` and calendar dates without repeating the matplotlib.
Band arithmetic and thresholds are written out, not wrapped. Where a product has
two routes, show them side by side.

If it needs credentials, put `# esd-requires: earthdata` (comma-separated, one
or more providers) on the **first line, above the docstring**. That keeps it
out of the credential-free pull-request build; the marker itself never appears
on the rendered page.
```

```{card} 6. An upstream watch entry
Add the product to `WATCHLIST.toml`, where `scripts/watch.py` looks every week
for the change a health probe cannot see: a CMR revision, a republished file
(`[[file]]`: ETag, length, status), a changelog or release feed, a
provider news page. A health probe says the route still answers; the watch
says what it answers with has changed. For a file that will be superseded
(yearly releases), watch next year's URL too: its 404 turning into a 206 is
the signal to move on. Seed the snapshot once with
`pixi run -e dev python scripts/watch.py --watchlist <just-the-new-entries.toml>`
so the first weekly digest compares rather than records.
```

```{card} 7. Check the loop closed
`pixi run -e test-py313 test-unit` — `tests/test_docs_pages.py` fails if the product
has no example, if the example file is missing, or if a gallery script no
product claims has appeared. `pixi run -e docs docs-fast` then shows you the
generated page.
```

## Documentation

Nothing under `docs/catalog/`, `docs/api/`, `docs/auto_examples/`,
`docs/generated/`, `docs/gen_modules/`, `docs/credentials.md` or
`docs/status.md` is written by hand — they are generated at build time from
the registry and the gallery, and they are all in `.gitignore`. Editing a
catalog page means editing the catalog entry.

What *is* hand-written: `index.md`, `installation.md`, `concepts.md`,
`faq.md`, this page, and the gallery scripts.

The README's header image is generated too: `scripts/make_montage.py` tiles
the thumbnails of the executed gallery into `_static/gallery.webp`, which the
docs build on `main` publishes and the README links to. It skips examples that
did not run, so a build without credentials does not turn seven products into
blank squares. Its status table and catalog summary come from
`scripts/update_readme_status.py`, between the sentinel comments — edit the
catalog entry, not the README.

## Style

- `ruff format` and `ruff check` must pass: `pixi run -e lint lint`. Line
  length 88, double quotes, isort with `easysnowdata` as first-party.
- NumPy-style docstrings. The API pages are generated from `__all__`, so a
  function without a docstring shows up as a blank page.
- `logging`, never `print` — the one sanctioned print is the import-time
  credential summary.
- No import-time side effects, and no global `xr.set_options` or GDAL
  configuration.
- Metric units everywhere.

The network clients under `easysnowdata/stations/clients/` arrived here as a
`git subtree` of
[`global_snow_networks`](https://github.com/egagli/global_snow_networks). That
repo deleted its copy in 2026-09 and imports these instead, so **this is now
the only copy: fix a client here.** They are linted and formatted like the
rest of the package. Their contract is still global_snow_networks'
`DESIGN.md` §3, and their tests are in `tests/stations/`.

## Pull requests

- Branch off `main`, one topic per pull request.
- The PR build runs ruff, the offline tiers on three interpreters and three
  operating systems, coverage, and a docs build with the credential-free
  gallery. None of it needs secrets.
- Include tests. Coverage of the new modules must stay above 95%.
- If you changed a public function's behaviour, say so in the PR body; the
  changelog is generated from commit messages with `git-changelog`.
