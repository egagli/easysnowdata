"""Sturm & Liston global seasonal snow classification (§4.3, companion §B.6).

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    snow_class = esd.snow.snow_classification.load(aoi)                     # NSIDC-0768
    snow_class = esd.snow.snow_classification.load(aoi, source="hosted-cog")  # no login
    esd.plotting.categorical(snow_class)

Six seasonal snow classes (tundra, boreal forest, maritime, ephemeral,
prairie, montane forest) plus ice and ocean, from air-temperature,
precipitation and wind-speed climatologies.

**Sources.** NSIDC-0768 is authoritative and is the default (§12 Q8); its
HTTPS directory redirects to Earthdata Login, so the file is fetched once into
the easysnowdata cache with an authenticated session. ``source="hosted-cog"``
needs no credentials and reads the 10 arcsec global map as a COG with range
requests — **that location is likely to change** (the account hosting it is
funded to 2026-10-26), which is why it is one entry here.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from easysnowdata import auth, catalog, config, providers
from easysnowdata.catalog import health
from easysnowdata.catalog._access import resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT_ID",
    "RESOLUTIONS",
    "NSIDC_DIRECTORY",
    "HOSTED_COG_URL",
    "filename",
    "load",
]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "snow-classification"
NSIDC_DIRECTORY = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/"
    "nsidc0768_global_seasonal_snow_classification_v01"
)
HOSTED_COG_URL = (
    "https://uwcryo.blob.core.windows.net/snowmelt/eric/snow_classification/"
    "SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
)
#: Resolution name → (nominal grid label, the file's arc-unit label).
RESOLUTIONS: dict[str, tuple[str, str]] = {
    "10arcsec": ("300m", "10.0arcsec"),
    "2.5arcmin": ("2.5km", "2.5arcmin"),
    "5arcmin": ("10km", "5.0arcmin"),
    "30arcmin": ("0.5deg", "30.0arcmin"),
}
#: The nine seasonal snow classes: value → (name, colour).
SNOW_CLASSIFICATION_CLASSES: dict[int, tuple[str, str]] = {
    1: ("Tundra", "#a100c8"),
    2: ("Boreal Forest", "#00a0fe"),
    3: ("Maritime", "#fe0000"),
    4: ("Ephemeral (includes no snow)", "#e7dc32"),
    5: ("Prairie", "#f08328"),
    6: ("Montane Forest", "#00dc00"),
    7: ("Ice (glaciers and ice sheets)", "#aaaaaa"),
    8: ("Ocean", "#0000ff"),
    9: ("Fill", "#ffffff"),
}
#: The class table's "Fill" value.
NODATA = 9
#: The single epoch the product covers.
EPOCH = "2021"


def filename(resolution: str = "10arcsec", region: str = "GL") -> str:
    """The NSIDC-0768 file name for a resolution and region (``GL`` or ``NA``)."""
    try:
        grid, arc = RESOLUTIONS[resolution]
    except KeyError:
        raise ValueError(
            f"Unknown resolution {resolution!r}; available: {', '.join(RESOLUTIONS)}."
        ) from None
    if region not in ("GL", "NA"):
        raise ValueError(f"region must be 'GL' (global) or 'NA', got {region!r}.")
    return f"SnowClass_{region}_{grid}_{arc}_{EPOCH}_v01.0.tif"


def _fetch_nsidc(name: str) -> Path:
    """Download one NSIDC-0768 file into the cache (once) through Earthdata Login."""
    if not auth.detect("earthdata"):
        raise auth.get("earthdata").error(
            "NSIDC-0768 is served behind Earthdata Login.",
            alternatives=('source="hosted-cog" (the 10 arcsec global map, no login)',),
        )
    target = config.cache_dir("snow_classification") / name
    if target.exists():
        return target
    auth.get("earthdata").ensure()
    import earthaccess  # noqa: PLC0415

    url = f"{NSIDC_DIRECTORY}/{name}"
    _logger.info("Downloading %s into %s (once).", name, target.parent)
    session = earthaccess.get_requests_https_session()
    with session.get(url, stream=True, allow_redirects=True, timeout=60) as response:
        if response.status_code == 404:
            raise FileNotFoundError(
                f"{url} does not exist. Check the resolution/region, browse "
                f"{NSIDC_DIRECTORY}, or use source='hosted-cog'."
            )
        response.raise_for_status()
        partial_path = target.with_suffix(target.suffix + ".part")
        with partial_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                handle.write(chunk)
    partial_path.replace(target)
    return target


def load(
    aoi: Any = None,
    *,
    source: str | None = None,
    resolution: str = "10arcsec",
    region: str = "GL",
    chunks: Any = contract.DEFAULT,
    mask: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Load the seasonal snow classification for *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts.
    source
        ``"nsidc"`` (default, needs Earthdata Login) or ``"hosted-cog"``
        (credential-free, 10 arcsec global only, location likely to change).
    resolution, region
        NSIDC route only: one of :data:`RESOLUTIONS` and ``"GL"`` or ``"NA"``.
    chunks
        Dask chunks; ``None`` loads eagerly.
    mask
        ``False`` (default, §2.5 for categorical products) keeps the 9 (Fill)
        sentinel with ``rio.nodata`` set; ``True`` masks it to NaN.
    **kwargs
        Passed to ``rioxarray.open_rasterio``.

    Returns
    -------
    xarray.DataArray
        ``snow_class``, uint8 class values with CF flag attributes for
        :func:`easysnowdata.plotting.categorical`.

    Raises
    ------
    easysnowdata.auth.CredentialError
        When the NSIDC route is asked for without Earthdata credentials; the
        message names ``source="hosted-cog"`` as the credential-free route.
    """
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    if src.id == "nsidc":
        name = filename(resolution, region)
        url: str | Path = _fetch_nsidc(name)
        source_url = f"{NSIDC_DIRECTORY}/{name}"
    else:
        url = source_url = HOSTED_COG_URL
        name = HOSTED_COG_URL.rsplit("/", 1)[-1]
    da = providers.raster_http.open(
        url,
        aoi,
        chunks=True if chunks in (None, contract.DEFAULT) else chunks,
        **kwargs,
    )
    da = contract.write_crs(da, da.rio.crs)
    da = (
        contract.mask_continuous(da, NODATA)
        if mask
        else contract.set_categorical_nodata(da, NODATA)
    )
    da = da.rename("snow_class")
    da = da.assign_coords(time=pd.Timestamp(f"{EPOCH}-01-01"))
    da.attrs.update(
        contract.provenance(product, src, source_url=source_url, file=name, epoch=EPOCH)
    )
    da = da.assign_attrs(product.variable("snow_class").cf_attrs())
    if chunks is None:
        da = da.compute()
    return da


def _nsidc_directory_probe() -> None:
    """The NSIDC directory is alive when it answers or redirects to Earthdata Login."""
    import requests  # noqa: PLC0415

    response = requests.get(
        f"{NSIDC_DIRECTORY}/",
        timeout=health.TIMEOUT,
        stream=True,
        allow_redirects=False,
        headers={"Range": "bytes=0-0"},
    )
    location = response.headers.get("location", "")
    response.close()
    if response.status_code in (200, 206):
        return
    if (
        response.status_code in (301, 302, 303, 307)
        and "earthdata.nasa.gov" in location
    ):
        return
    raise RuntimeError(f"Unreachable: HTTP {response.status_code} {location}")


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="snow",
    title="Sturm & Liston seasonal snow classification",
    description=(
        "Global seasonal snow classes (tundra, boreal forest, maritime, "
        "ephemeral, prairie, montane forest, ice) from climatologies of air "
        "temperature, precipitation and wind speed, at 10 arcsec (~300 m) and "
        "three coarser grids."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="raster_http",
            location=NSIDC_DIRECTORY,
            requires=("earthdata",),
            resolution_m=300,
            temporal="static (2021)",
            notes=(
                "authoritative; not cloud-hosted, the HTTPS directory redirects "
                "to Earthdata Login, so files are cached locally on first use. "
                "Also serves the 2.5 arcmin, 5 arcmin and 30 arcmin grids and "
                "the North America subset"
            ),
            title="NSIDC-0768",
            health=Probe(
                "Sturm & Liston snow classification (NSIDC-0768)",
                _nsidc_directory_probe,
                requires=(),  # the redirect to URS is proof enough that it is up
            ),
        ),
        Source(
            id="hosted-cog",
            provider="raster_http",
            location=HOSTED_COG_URL,
            resolution_m=300,
            temporal="static (2021)",
            notes=(
                "COG on the uwcryo Azure blob, anonymous range reads; 10 arcsec "
                "global only. Location likely to change (a Zenodo record or a "
                "GitHub release asset): the hosting account is funded to "
                "2026-10-26 (§12 Q8)"
            ),
            title="hosted COG (Azure)",
            health=Probe(
                "Sturm & Liston snow classification (Azure)",
                partial(health.http_first_byte, HOSTED_COG_URL),
            ),
        ),
    ),
    variables=(
        Variable(
            "snow_class",
            dtype="uint8",
            nodata=9,
            long_name="seasonal snow class",
            flag_values=tuple(SNOW_CLASSIFICATION_CLASSES),
            flag_meanings=tuple(n for n, _ in SNOW_CLASSIFICATION_CLASSES.values()),
            flag_colors=tuple(c for _, c in SNOW_CLASSIFICATION_CLASSES.values()),
        ),
    ),
    citation=(
        "Liston, G. E. and Sturm, M. (2021). Global Seasonal-Snow Classification, "
        "Version 1. Boulder, Colorado USA. NSIDC."
    ),
    license="NASA Earthdata (free registration)",
    doi="10.5067/99FTCYYYLAQ0",
    loader="easysnowdata.snow.snow_classification.load",
    examples=("snow/plot_snow_classification.py",),
    references=(
        "https://nsidc.org/data/nsidc-0768/versions/1",
        "https://doi.org/10.1175/2010JCLI3544.1",
    ),
    tags=("snow class", "climatology"),
)

catalog.register(PRODUCT, replace=True)
