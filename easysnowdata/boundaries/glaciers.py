"""Randolph Glacier Inventory (RGI) glacier outlines, version 7.0 or 6.0.

::

    import easysnowdata as esd

    aoi = (-121.94, 46.72, -121.54, 46.99)
    glaciers_gdf = esd.boundaries.glaciers.load(aoi)                   # RGI 7.0, NSIDC
    complexes_gdf = esd.boundaries.glaciers.load(aoi, product="complexes")
    rgi6_gdf = esd.boundaries.glaciers.load(aoi, version="6.0")        # NSIDC
    rgi6_gdf = esd.boundaries.glaciers.load(aoi, version="6.0", source="oggm-mirror")
    regions_gdf = esd.boundaries.glaciers.regions()                    # the 19 first-order regions

The RGI is the global inventory of glacier outlines outside the ice sheets.
**Version 7.0** (RGI 7.0 Consortium 2023) is a new inventory with outlines
targeted at the year 2000, a glacier product and a glacier-*complex* product
(contiguous ice as one polygon). **Version 6.0** (RGI Consortium 2017) is the
inventory most published work up to 2023 used, including the OGGM and many
mass-balance datasets; keep it for comparison with that literature.

Both are distributed per first-order region, one zipped shapefile each. The
loader reads the region outlines, picks the regions the AOI touches, fetches
each regional archive once into the easysnowdata cache (3-190 MB), and reads
the AOI from it. ``region=`` bypasses the lookup; a whole-world read needs
``region=`` (every region is roughly a gigabyte).

**Sources.** ``"nsidc"`` (default) is the archive of record for both
versions and needs an Earthdata Login — as a username and password
(``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` or a ``~/.netrc`` entry): NSIDC's
on-premises archive does not accept a bearer ``EARTHDATA_TOKEN`` on its own.
``"oggm-mirror"`` is OGGM's credential-free mirror of the original GLIMS
release files for **6.0 only**.

The first columns are the same for both versions: ``rgi_id``, ``name``,
``area_km2`` and ``o1region``, followed by the version's own attributes (for
7.0 ``glims_id``, ``src_date``, ``zmin_m`` … ``term_type``; for 6.0 ``RGIId``,
``GLIMSId``, ``BgnDate``, ``Zmin`` … ``TermType``). RGI 7.0 stores a constant
Z coordinate, which is dropped.
"""

from __future__ import annotations

import logging
import zipfile
from functools import partial
from typing import Any

import geopandas as gpd
import pandas as pd

from easysnowdata import catalog, providers
from easysnowdata.aoi import parse_aoi
from easysnowdata.catalog import health
from easysnowdata.catalog._access import ensure_source, resolve_source
from easysnowdata.catalog._models import Probe, Product, Source, Variable
from easysnowdata.processing import contract

__all__ = [
    "PRODUCT_ID",
    "VERSIONS",
    "PRODUCTS",
    "RGI7_REGIONS",
    "RGI6_REGIONS",
    "load",
    "regions",
    "url",
]

_logger = logging.getLogger(__name__)

PRODUCT_ID = "rgi-glaciers"
NSIDC_RGI7 = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7"
NSIDC_RGI6 = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v6"
OGGM_RGI6 = (
    "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/rgi60_files"
)
VERSIONS = ("7.0", "6.0")
#: RGI 7.0 products → the letter in their file names. RGI 6.0 has glaciers only.
PRODUCTS: dict[str, str] = {"glaciers": "G", "complexes": "C"}
#: First-order region → the RGI 7.0 file-name stem.
RGI7_REGIONS: dict[int, str] = {
    1: "01_alaska",
    2: "02_western_canada_usa",
    3: "03_arctic_canada_north",
    4: "04_arctic_canada_south",
    5: "05_greenland_periphery",
    6: "06_iceland",
    7: "07_svalbard_jan_mayen",
    8: "08_scandinavia",
    9: "09_russian_arctic",
    10: "10_north_asia",
    11: "11_central_europe",
    12: "12_caucasus_middle_east",
    13: "13_central_asia",
    14: "14_south_asia_west",
    15: "15_south_asia_east",
    16: "16_low_latitudes",
    17: "17_southern_andes",
    18: "18_new_zealand",
    19: "19_subantarctic_antarctic_islands",
}
#: First-order region → the RGI 6.0 file-name stem.
RGI6_REGIONS: dict[int, str] = {
    1: "01_rgi60_Alaska",
    2: "02_rgi60_WesternCanadaUS",
    3: "03_rgi60_ArcticCanadaNorth",
    4: "04_rgi60_ArcticCanadaSouth",
    5: "05_rgi60_GreenlandPeriphery",
    6: "06_rgi60_Iceland",
    7: "07_rgi60_Svalbard",
    8: "08_rgi60_Scandinavia",
    9: "09_rgi60_RussianArctic",
    10: "10_rgi60_NorthAsia",
    11: "11_rgi60_CentralEurope",
    12: "12_rgi60_CaucasusMiddleEast",
    13: "13_rgi60_CentralAsia",
    14: "14_rgi60_SouthAsiaWest",
    15: "15_rgi60_SouthAsiaEast",
    16: "16_rgi60_LowLatitudes",
    17: "17_rgi60_SouthernAndes",
    18: "18_rgi60_NewZealand",
    19: "19_rgi60_AntarcticSubantarctic",
}
#: Per version: the region-outline archive, its member and its code column.
_REGION_FILES: dict[tuple[str, str], tuple[str, str]] = {
    ("7.0", "nsidc"): (
        f"{NSIDC_RGI7}/RGI2000-v7.0-regions.zip",
        "RGI2000-v7.0-o1regions.shp",
    ),
    ("6.0", "nsidc"): (
        f"{NSIDC_RGI6}/nsidc0770_00.rgi60.regions.zip",
        "00_rgi60_O1Regions.shp",
    ),
    ("6.0", "oggm-mirror"): (
        f"{OGGM_RGI6}/00_rgi60_regions.zip",
        "00_rgi60_O1Regions.shp",
    ),
}
_REGION_CODE = {"7.0": "o1region", "6.0": "RGI_CODE"}
#: Per version: the source column behind each shared column.
_LEAD_COLUMNS = {
    "7.0": {
        "rgi_id": "rgi_id",
        "name": "glac_name",
        "area_km2": "area_km2",
        "o1region": "o1region",
    },
    "6.0": {
        "rgi_id": "RGIId",
        "name": "Name",
        "area_km2": "Area",
        "o1region": "O1Region",
    },
}


def _version(version: Any) -> str:
    text = str(version).lower().lstrip("v")
    text = {"7": "7.0", "6": "6.0"}.get(text, text)
    if text not in VERSIONS:
        raise ValueError(
            f"Invalid RGI version {version!r}. Choose from {', '.join(repr(v) for v in VERSIONS)}."
        )
    return text


def _check(version: str, product: str, source_id: str) -> None:
    if product not in PRODUCTS:
        raise ValueError(
            f"Invalid product {product!r}. Choose from {', '.join(repr(p) for p in PRODUCTS)}."
        )
    if version == "6.0" and product != "glaciers":
        raise ValueError("RGI 6.0 has glacier outlines only; complexes are new in 7.0.")
    if version == "7.0" and source_id == "oggm-mirror":
        raise ValueError(
            'The OGGM mirror holds RGI 6.0 only; use source="nsidc" for 7.0 '
            '(Earthdata Login) or version="6.0".'
        )


def url(
    region: int,
    *,
    version: str = "7.0",
    product: str = "glaciers",
    source: str = "nsidc",
) -> tuple[str, str]:
    """``(archive URL, shapefile member)`` for one first-order *region*."""
    version = _version(version)
    _check(version, product, source)
    table = RGI7_REGIONS if version == "7.0" else RGI6_REGIONS
    try:
        stem = table[int(region)]
    except (KeyError, ValueError):
        raise ValueError(
            f"Invalid RGI region {region!r}; the regional files are 1-19."
        ) from None
    if version == "7.0":
        letter = PRODUCTS[product]
        name = f"RGI2000-v7.0-{letter}-{stem}"
        return (
            f"{NSIDC_RGI7}/regional_files/RGI2000-v7.0-{letter}/{name}.zip",
            f"{name}.shp",
        )
    if source == "oggm-mirror":
        return f"{OGGM_RGI6}/{stem}.zip", f"{stem}.shp"
    code, rest = stem.split("_rgi60_")
    return f"{NSIDC_RGI6}/nsidc0770_{code}.rgi60.{rest}.zip", f"{stem}.shp"


def _fetch(archive_url: str, src: Source, version: str) -> Any:
    """Download one archive into the cache: through Earthdata Login, or plainly."""
    subdir = f"boundaries/rgi/{version}"
    if not src.requires:
        return providers.raster_http.fetch(archive_url, subdir=subdir)
    path = providers.earthdata.download([archive_url], subdir)[0]
    if not zipfile.is_zipfile(path):
        # NSIDC's on-premises tree ignores a bearer token and answers with the
        # Earthdata login page (HTTP 200), which earthaccess saves as the file.
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"NSIDC returned the Earthdata login page instead of "
            f"{archive_url.rsplit('/', 1)[-1]}. Its on-premises archive (daacdata) "
            "accepts a username and password (EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD, or a ~/.netrc entry) but not a bearer "
            'EARTHDATA_TOKEN on its own. For RGI 6.0, source="oggm-mirror" needs '
            "no account."
        )
    return path


def regions(
    aoi: Any = None, *, version: str = "7.0", source: str | None = None
) -> gpd.GeoDataFrame:
    """The RGI first-order region outlines (intersecting *aoi*, if given)."""
    version = _version(version)
    product = catalog.get(PRODUCT_ID)
    src = resolve_source(product, source)
    _check(version, "glaciers", src.id)
    ensure_source(product, src)
    archive_url, member = _REGION_FILES[(version, src.id)]
    path = _fetch(archive_url, src, version)
    raw_gdf = providers.vector_http.read(f"zip://{path}!{member}", aoi)
    codes = pd.to_numeric(raw_gdf[_REGION_CODE[version]], errors="coerce").astype(
        "Int64"
    )
    names = raw_gdf["full_name" if version == "7.0" else "FULL_NAME"]
    regions_gdf = gpd.GeoDataFrame(
        {"o1region": codes, "name": names, "geometry": raw_gdf.geometry},
        crs=raw_gdf.crs,
    )
    return contract.finalize_frame(
        regions_gdf, product, src, source_url=archive_url, version=version
    )


def _regions_for(aoi: Any, version: str, source: str | None) -> list[int]:
    touched = regions(aoi, version=version, source=source)["o1region"].dropna()
    # Region 20 (Antarctic mainland) has outlines in the region file but no
    # glacier file; a region split at the antimeridian appears twice.
    codes = sorted({int(c) for c in touched if int(c) in RGI7_REGIONS})
    if not codes:
        _logger.warning("The AOI touches no RGI region with glacier files.")
    return codes


def load(
    aoi: Any = None,
    *,
    version: str = "7.0",
    product: str = "glaciers",
    region: int | list[int] | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """RGI glacier outlines intersecting *aoi*.

    Parameters
    ----------
    aoi
        Any form :func:`easysnowdata.aoi.parse_aoi` accepts. ``None`` needs
        *region* and returns whole regions.
    version
        ``"7.0"`` (default) or ``"6.0"``; ``7``, ``"v6"`` and similar work too.
    product
        RGI 7.0 only: ``"glaciers"`` (default) or ``"complexes"``.
    region
        First-order region(s) 1-19 to read, instead of looking them up from
        the AOI (see :func:`regions`).
    source
        ``"nsidc"`` (default, Earthdata Login) or ``"oggm-mirror"``
        (credential-free, 6.0 only).
    **kwargs
        Passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame
        ``rgi_id``, ``name``, ``area_km2``, ``o1region``, then the version's
        own attributes, in EPSG:4326.
    """
    version = _version(version)
    catalog_product = catalog.get(PRODUCT_ID)
    src = resolve_source(catalog_product, source)
    _check(version, product, src.id)
    if region is None:
        if aoi is None or parse_aoi(aoi).is_global:
            raise ValueError(
                "A whole-world RGI read downloads every region (about a gigabyte): "
                "pass an AOI, or region= (1-19) to read whole regions."
            )
        codes = _regions_for(aoi, version, src.id)
    else:
        codes = [
            int(r) for r in ([region] if isinstance(region, (int, str)) else region)
        ]
    ensure_source(catalog_product, src)
    frames, urls = [], []
    for code in codes:
        archive_url, member = url(code, version=version, product=product, source=src.id)
        path = _fetch(archive_url, src, version)
        frames.append(
            providers.vector_http.read(f"zip://{path}!{member}", aoi, **kwargs)
        )
        urls.append(archive_url)
    if frames:
        raw_gdf = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs=frames[0].crs
        )
    else:
        raw_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    mapping = _LEAD_COLUMNS[version]
    lead = pd.DataFrame(
        {
            shared: raw_gdf[column] if column in raw_gdf.columns else None
            for shared, column in mapping.items()
        },
        index=raw_gdf.index,
    )
    lead["o1region"] = pd.to_numeric(lead["o1region"], errors="coerce").astype("Int64")
    rest = raw_gdf.drop(columns=[c for c in lead.columns if c in raw_gdf.columns])
    glaciers_gdf = gpd.GeoDataFrame(pd.concat([lead, rest], axis=1), crs=raw_gdf.crs)
    return contract.finalize_frame(
        glaciers_gdf,
        catalog_product,
        src,
        source_url=" ".join(urls),
        version=version,
        rgi_product=product,
        regions=",".join(str(c) for c in codes),
    )


PRODUCT = Product(
    id=PRODUCT_ID,
    theme="boundaries",
    title="Randolph Glacier Inventory (RGI 7.0 and 6.0)",
    description=(
        "Global glacier outlines outside the ice sheets, per first-order region: "
        "RGI 7.0 glaciers and glacier complexes (2023), or RGI 6.0 (2017) for "
        "comparison with the literature built on it."
    ),
    sources=(
        Source(
            id="nsidc",
            provider="earthdata",
            location=f"{NSIDC_RGI7}/regional_files/ ; {NSIDC_RGI6}/",
            requires=("earthdata",),
            temporal="static (7.0: 2023, outlines targeted at 2000; 6.0: 2017)",
            notes=(
                "NSIDC-0770, the archive of record for both versions; one zipped "
                "shapefile per region (3-190 MB), cached on first use; needs the "
                "Earthdata username and password (a bearer token alone is not "
                "accepted by NSIDC's on-premises archive)"
            ),
            title="NSIDC (Earthdata Login)",
            health=(
                Probe(
                    "RGI 7.0 glacier outlines (NSIDC)",
                    partial(
                        health.earthdata_https_first_byte,
                        f"{NSIDC_RGI7}/RGI2000-v7.0-regions.zip",
                    ),
                ),
                Probe(
                    "RGI 6.0 glacier outlines (NSIDC)",
                    partial(
                        health.earthdata_https_first_byte,
                        f"{NSIDC_RGI6}/nsidc0770_00.rgi60.regions.zip",
                    ),
                ),
            ),
        ),
        Source(
            id="oggm-mirror",
            provider="vector_http",
            location=f"{OGGM_RGI6}/<nn>_rgi60_<Region>.zip",
            temporal="static (6.0, 2017)",
            notes=(
                "OGGM's mirror of the original GLIMS release files; RGI 6.0 only, "
                "no account"
            ),
            title="OGGM mirror of GLIMS (6.0)",
            health=Probe(
                "RGI 6.0 glacier outlines (OGGM mirror)",
                partial(health.http_first_byte, f"{OGGM_RGI6}/00_rgi60_regions.zip"),
            ),
        ),
    ),
    variables=(
        Variable("rgi_id", long_name="RGI glacier identifier"),
        Variable("name", long_name="glacier name"),
        Variable("area_km2", units="km2", long_name="glacier area"),
        Variable("o1region", long_name="RGI first-order region"),
    ),
    citation=(
        "RGI 7.0 Consortium (2023). Randolph Glacier Inventory - A Dataset of "
        "Global Glacier Outlines, Version 7.0. NSIDC, doi:10.5067/f6jmovy5navz. "
        "RGI Consortium (2017). Randolph Glacier Inventory 6.0, "
        "doi:10.7265/4m1f-gd79."
    ),
    license="CC BY 4.0",
    doi="10.5067/f6jmovy5navz",
    loader="easysnowdata.boundaries.glaciers.load",
    examples=("boundaries/plot_glaciers.py",),
    references=(
        "https://www.glims.org/rgi_user_guide/",
        "https://nsidc.org/data/nsidc-0770/versions/7",
        "https://nsidc.org/data/nsidc-0770/versions/6",
    ),
    tags=("glaciers", "rgi", "glacier outlines", "ice"),
)

catalog.register(PRODUCT, replace=True)
