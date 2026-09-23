"""Boundaries: catalog entries, every loader offline on fixtures, live smoke."""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely

import easysnowdata as esd
from easysnowdata import catalog, providers
from easysnowdata.auth import CredentialError
from easysnowdata.boundaries import admin, glaciers, mountains, natural_earth
from easysnowdata.processing import contract

RAINIER = (-121.94, 46.72, -121.54, 46.99)
PRODUCTS = {
    "countries": admin.countries,
    "states-provinces": admin.states,
    "us-counties": admin.counties,
    "admin-boundaries": admin.admin,
    "natural-earth-vectors": natural_earth.load,
    "gmba-mountains": mountains.load,
    "rgi-glaciers": glaciers.load,
}


@pytest.fixture
def archives(static_fixtures, monkeypatch):
    """Serve every download from the local fixtures; record what was asked for."""
    calls: list[dict] = []

    def pick(url: str):
        name = url.rsplit("/", 1)[-1]
        for key, fixture in (
            ("admin_0_countries", "ne_countries_zip"),
            ("admin_1_states_provinces", "ne_states_zip"),
            ("lakes", "ne_lakes_zip"),
            ("_us_state_", "census_state_zip"),
            ("_us_county_", "census_county_zip"),
            ("GMBA_Inventory_v2.0_standard_basic", "gmba_basic_zip"),
            ("GMBA_Inventory_v2.0_broad_basic", "gmba_basic_zip"),
            ("GMBA_Inventory_v2.0_standard_300", "gmba_300_zip"),
            ("GMBA_Inventory_v2.0_standard.zip", "gmba_all_zip"),
            ("RGI2000-v7.0-regions", "rgi7_regions_zip"),
            ("RGI2000-v7.0-G-02", "rgi7_g02_zip"),
            ("RGI2000-v7.0-C-02", "rgi7_c02_zip"),
            ("rgi60_regions", "rgi6_regions_zip"),
            ("rgi60.regions", "rgi6_regions_zip"),
            ("02_rgi60_WesternCanadaUS", "rgi6_02_zip"),
            ("nsidc0770_02.rgi60", "rgi6_02_zip"),
            ("geoBoundaries-", "geoboundaries_geojson"),
        ):
            if key in name:
                return static_fixtures[fixture]
        raise AssertionError(f"no fixture for {url}")

    def fetch(url, fname=None, **kwargs):
        calls.append({"url": url, "fname": fname, "subdir": kwargs.get("subdir")})
        return pick(url)

    def download(urls, subdir, **kwargs):
        calls.append({"url": urls[0], "subdir": subdir, "earthdata": True})
        return [pick(urls[0])]

    monkeypatch.setattr(providers.raster_http, "fetch", fetch)
    monkeypatch.setattr(providers.earthdata, "download", download)
    monkeypatch.setattr(glaciers, "ensure_source", lambda product, source: {})
    return calls


@pytest.fixture
def geoboundaries_api(monkeypatch):
    """The gbOpen API, answering one ADM1 record (or a 404 for level 5)."""
    asked: list[str] = []

    class Response:
        def __init__(self, url):
            self.status_code = 404 if "/ADM5/" in url else 200
            self.text = "Not found" if self.status_code == 404 else "{...}"

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "boundaryLicense": "Public Domain",
                "boundarySource": "United States Census Bureau",
                "gjDownloadURL": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
                "releaseData/gbOpen/USA/ADM1/geoBoundaries-USA-ADM1.geojson",
                "simplifiedGeometryGeoJSON": "https://github.com/wmgeolab/geoBoundaries/"
                "raw/9469f09/releaseData/gbOpen/USA/ADM1/geoBoundaries-USA-ADM1_simplified.geojson",
            }

    def get(url, timeout=None, **kwargs):
        asked.append(url)
        return Response(url)

    monkeypatch.setattr("requests.get", get)
    return asked


class TestCatalogEntries:
    def test_all_seven_registered_from_the_theme(self):
        for product_id, loader in PRODUCTS.items():
            product = catalog.get(product_id)
            assert product.theme == "boundaries"
            assert product.resolve_loader() is loader
            assert product.examples and product.examples[0].startswith("boundaries/")
        assert set(catalog.list(theme="boundaries").index) == set(PRODUCTS)
        assert catalog.validate_all() == []

    def test_only_rgi_nsidc_needs_an_account(self):
        for product_id in PRODUCTS:
            for src in catalog.get(product_id).sources:
                expected = ("earthdata",) if src.id == "nsidc" else ()
                assert src.requires == expected, (product_id, src.id)
        assert (
            catalog.get("rgi-glaciers").credential_free_sources[0].id == "oggm-mirror"
        )

    def test_theme_is_exported(self):
        assert "boundaries" in esd.__all__ and "boundaries" in catalog.KNOWN_THEMES
        assert set(esd.boundaries.__all__) == {
            "admin",
            "glaciers",
            "mountains",
            "natural_earth",
        }

    def test_every_source_has_a_probe(self):
        for product_id in PRODUCTS:
            for src in catalog.get(product_id).sources:
                assert src.health, (product_id, src.id)


class TestFinalizeFrame:
    def test_crs_and_z_and_provenance(self):
        frame_gdf = gpd.GeoDataFrame(
            {"a": [1]},
            geometry=[shapely.force_3d(shapely.box(0, 0, 1, 1), 5.0)],
            crs="EPSG:4269",
        )
        product = catalog.get("rgi-glaciers")
        done_gdf = contract.finalize_frame(frame_gdf, product, product.source(), x=1)
        assert done_gdf.crs.to_epsg() == 4326
        assert not done_gdf.geometry.has_z.any()
        assert (
            done_gdf.attrs["product_id"] == "rgi-glaciers" and done_gdf.attrs["x"] == 1
        )

    def test_missing_crs_is_set(self):
        frame_gdf = gpd.GeoDataFrame(geometry=[shapely.Point(0, 0)])
        product = catalog.get("countries")
        assert (
            contract.finalize_frame(frame_gdf, product, "natural-earth").crs.to_epsg()
            == 4326
        )


@pytest.mark.recorded
class TestCountries:
    def test_lead_columns_and_iso3_from_adm0_a3(self, archives):
        world_gdf = admin.countries()
        assert list(world_gdf.columns[:3]) == ["name", "iso3", "admin_level"]
        assert set(world_gdf["iso3"]) == {"USA", "CAN", "NOR"}
        # Natural Earth's ISO_A3 is -99 for Norway; iso3 still says NOR.
        assert world_gdf.set_index("iso3").loc["NOR", "ISO_A3"] == "-99"
        assert (world_gdf["admin_level"] == 0).all()
        assert world_gdf.attrs["scale"] == "1:110m"
        assert archives[-1]["url"].endswith(
            "/110m/cultural/ne_110m_admin_0_countries.zip"
        )
        assert archives[-1]["subdir"] == "boundaries/natural_earth"

    def test_aoi_defaults_to_10m_and_filters(self, archives):
        usa_gdf = admin.countries(RAINIER)
        assert usa_gdf["iso3"].tolist() == ["USA"] and usa_gdf.attrs["scale"] == "1:10m"
        assert admin.countries(iso3="nor")["name"].tolist() == ["Norway"]
        assert admin.countries(name=["Canada", "Norway"])["iso3"].tolist() == [
            "CAN",
            "NOR",
        ]

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="Invalid scale"):
            admin.countries(scale="5m")

    def test_geoboundaries_adm0(self, archives, geoboundaries_api):
        admin.countries(iso3="USA", source="geoboundaries")
        assert geoboundaries_api[-1].endswith("/USA/ADM0/")


@pytest.mark.recorded
class TestStates:
    def test_us_aoi_picks_the_census(self, archives):
        wa_gdf = admin.states(RAINIER)
        assert wa_gdf.attrs["source"] == "us-census"
        assert (
            wa_gdf["name"].tolist() == ["Washington"] and wa_gdf.crs.to_epsg() == 4326
        )
        assert wa_gdf.attrs["resolution"] == "1:5m" and wa_gdf.attrs["year"] == 2024
        assert "cb_2024_us_state_5m.zip" in archives[-1]["url"]

    def test_country_selects_the_source(self, archives):
        bc_gdf = admin.states(country="CAN", name="british columbia")
        assert bc_gdf.attrs["source"] == "natural-earth"
        assert bc_gdf[["name", "iso3", "admin_level"]].values.tolist() == [
            ["British Columbia", "CAN", 1]
        ]
        assert admin.states(country="USA").attrs["source"] == "us-census"

    def test_world_defaults_to_natural_earth(self, archives):
        assert admin.states().attrs["source"] == "natural-earth"

    def test_census_year_and_resolution(self, archives):
        admin.states(country="USA", resolution="500k", year=2023)
        assert archives[-1]["url"].endswith("/GENZ2023/shp/cb_2023_us_state_500k.zip")
        with pytest.raises(ValueError, match="Invalid resolution"):
            admin.states(country="USA", resolution="1m")

    def test_natural_earth_has_no_110m_states(self, archives):
        with pytest.raises(ValueError, match='"10m" and "50m"'):
            admin.states(source="natural-earth", scale="110m")

    def test_geoboundaries_route(self, archives, geoboundaries_api):
        units_gdf = admin.states(RAINIER, source="geoboundaries")
        assert units_gdf["name"].tolist() == ["Washington"]
        assert units_gdf.attrs["source"] == "geoboundaries"


@pytest.mark.recorded
class TestCounties:
    def test_aoi_and_state_filters(self, archives):
        assert set(admin.counties(RAINIER)["name"]) == {"Pierce", "Lewis"}
        wa_gdf = admin.counties(state="WA")
        assert set(wa_gdf["name"]) == {"Pierce", "Lewis", "King"}
        assert set(admin.counties(state="Oregon")["name"]) == {"Multnomah"}
        assert admin.counties(state=["wa"], name="King")["GEOID"].tolist() == ["53033"]
        assert (wa_gdf["admin_level"] == 2).all() and (wa_gdf["iso3"] == "USA").all()


@pytest.mark.recorded
class TestAdmin:
    def test_record_and_cache_name(self, archives, geoboundaries_api):
        units_gdf = admin.admin(country="USA", level=1)
        assert geoboundaries_api == [
            "https://www.geoboundaries.org/api/current/gbOpen/USA/ADM1/"
        ]
        assert archives[-1]["fname"] == "9469f09-geoBoundaries-USA-ADM1.geojson"
        assert archives[-1]["subdir"] == "boundaries/geoboundaries"
        assert units_gdf["boundaryLicense"].unique().tolist() == ["Public Domain"]
        assert list(units_gdf.columns[:3]) == ["name", "iso3", "admin_level"]
        assert units_gdf.attrs["admin_level"] == 1

    def test_simplified_and_inferred_country(self, archives, geoboundaries_api):
        admin.admin(RAINIER, level=1, simplified=True)
        assert geoboundaries_api[-1].endswith("/USA/ADM1/")
        assert archives[-1]["fname"].endswith("ADM1_simplified.geojson")

    def test_errors(self, archives, geoboundaries_api):
        with pytest.raises(ValueError, match="per country"):
            admin.admin(level=1)
        with pytest.raises(ValueError, match="between 0 and 5"):
            admin.admin(country="USA", level=6)
        with pytest.raises(ValueError, match="no ADM5"):
            admin.admin(country="USA", level=5)


@pytest.mark.recorded
class TestNaturalEarth:
    def test_url_lookup(self):
        assert natural_earth.url("lakes", "110m").endswith(
            "/110m/physical/ne_110m_lakes.zip"
        )
        assert natural_earth.url("admin_0_countries_ind").endswith(
            "/10m/cultural/ne_10m_admin_0_countries_ind.zip"
        )
        assert natural_earth.url("bathymetry_K_200", category="physical").endswith(
            "ne_10m_bathymetry_K_200.zip"
        )
        with pytest.raises(ValueError, match="Unknown layer"):
            natural_earth.url("bathymetry_K_200")
        with pytest.raises(ValueError, match="'10m' only"):
            natural_earth.url("roads", "50m")

    def test_load(self, archives):
        lakes_gdf = natural_earth.load(RAINIER)
        assert lakes_gdf["name"].tolist() == ["Mowich Lake"]
        assert (
            lakes_gdf.attrs["layer"] == "lakes" and lakes_gdf.attrs["scale"] == "1:10m"
        )
        with pytest.raises(ValueError, match="Invalid scale"):
            natural_earth.load(layer="lakes", scale="20m")


@pytest.mark.recorded
class TestMountains:
    def test_basic_default(self, archives):
        ranges_gdf = mountains.load(RAINIER)
        assert list(ranges_gdf.columns[:4]) == ["name", "gmba_id", "level", "path"]
        assert ranges_gdf["name"].tolist() == ["Mount Rainier Massif"]
        assert ranges_gdf["level"].tolist() == [7]
        assert archives[-1]["subdir"] == "boundaries/gmba"
        assert ranges_gdf.attrs["subset"] == "basic"

    def test_all_levels_and_filters(self, archives):
        nested_gdf = mountains.load(RAINIER, subset="all")
        assert set(nested_gdf["name"]) == {"Mount Rainier Massif", "Cascade Range"}
        assert mountains.load(RAINIER, subset="all", level=4)["name"].tolist() == [
            "Cascade Range"
        ]
        assert mountains.load(subset="all", name="far range")["gmba_id"].tolist() == [
            12159
        ]

    def test_urls_and_errors(self):
        assert mountains.url("300", "broad").endswith(
            "/broad/GMBA_Inventory_v2.0_broad_300.zip"
        )
        assert mountains.url("all").endswith(
            "/standard/GMBA_Inventory_v2.0_standard.zip"
        )
        with pytest.raises(ValueError, match="Invalid subset"):
            mountains.url("600")
        with pytest.raises(ValueError, match="Invalid extent"):
            mountains.url("basic", "narrow")


@pytest.mark.recorded
class TestGlaciers:
    def test_rgi7_default(self, archives):
        glaciers_gdf = glaciers.load(RAINIER)
        assert list(glaciers_gdf.columns[:4]) == [
            "rgi_id",
            "name",
            "area_km2",
            "o1region",
        ]
        assert glaciers_gdf["name"].tolist() == ["Emmons Glacier", "Winthrop Glacier"]
        assert glaciers_gdf["o1region"].tolist() == [2, 2]
        assert not glaciers_gdf.geometry.has_z.any()  # RGI 7.0's constant Z is dropped
        assert (
            glaciers_gdf.attrs["version"] == "7.0"
            and glaciers_gdf.attrs["regions"] == "2"
        )
        # The region lookup and the regional file both go through Earthdata Login.
        assert [c.get("earthdata") for c in archives] == [True, True]
        assert archives[-1]["url"].endswith(
            "regional_files/RGI2000-v7.0-G/RGI2000-v7.0-G-02_western_canada_usa.zip"
        )

    def test_complexes(self, archives):
        complexes_gdf = glaciers.load(RAINIER, product="complexes")
        assert complexes_gdf["rgi_id"].tolist() == ["RGI2000-v7.0-C-02-1"]
        assert complexes_gdf["name"].isna().all()

    def test_rgi6_both_sources(self, archives):
        mirror_gdf = glaciers.load(RAINIER, version="6.0", source="oggm-mirror")
        assert mirror_gdf["rgi_id"].tolist()[:2] == ["RGI60-02.1", "RGI60-02.2"]
        assert mirror_gdf["area_km2"].tolist()[0] == pytest.approx(10.594)
        assert "earthdata" not in archives[-1]
        assert archives[-1]["url"].startswith(glaciers.OGGM_RGI6)
        nsidc_gdf = glaciers.load(RAINIER, version=6)
        assert archives[-1]["url"].endswith("nsidc0770_02.rgi60.WesternCanadaUS.zip")
        assert len(nsidc_gdf) == len(mirror_gdf)

    def test_region_bypasses_the_lookup(self, archives):
        whole_gdf = glaciers.load(region=2)
        assert len(whole_gdf) == 3 and len(archives) == 1

    def test_regions(self, archives):
        regions_gdf = glaciers.regions()
        assert regions_gdf["o1region"].tolist() == [2, 8]
        assert glaciers.regions(RAINIER, version="6.0", source="oggm-mirror")[
            "name"
        ].tolist() == ["Western Canada and USA"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"version": "5.0"}, "Invalid RGI version"),
            ({"product": "centerlines"}, "Invalid product"),
            ({"version": "6.0", "product": "complexes"}, "glacier outlines only"),
            ({"source": "oggm-mirror"}, "RGI 6.0 only"),
            ({"aoi": None}, "whole-world"),
        ],
    )
    def test_errors(self, archives, kwargs, match):
        aoi = kwargs.pop("aoi", RAINIER)
        with pytest.raises(ValueError, match=match):
            glaciers.load(aoi, **kwargs)

    def test_url(self):
        assert glaciers.url(18, version="6.0", source="oggm-mirror") == (
            f"{glaciers.OGGM_RGI6}/18_rgi60_NewZealand.zip",
            "18_rgi60_NewZealand.shp",
        )
        with pytest.raises(ValueError, match="1-19"):
            glaciers.url(20)

    def test_missing_earthdata_names_the_mirror(self, static_fixtures, monkeypatch):
        from easysnowdata import auth

        def refuse(*names, **kwargs):
            raise auth.get("earthdata").error("No Earthdata Login found.")

        monkeypatch.setattr(auth, "ensure", refuse)
        with pytest.raises(CredentialError, match="oggm-mirror"):
            glaciers.load(RAINIER, version="6.0")


@pytest.mark.live
class TestLive:
    def test_countries_and_states(self):
        assert admin.countries(RAINIER)["iso3"].tolist() == ["USA"]
        wa_gdf = admin.states(RAINIER)
        assert (
            wa_gdf["name"].tolist() == ["Washington"]
            and wa_gdf.attrs["source"] == "us-census"
        )

    def test_counties(self):
        assert set(admin.counties(RAINIER)["name"]) == {"Pierce", "Lewis"}

    def test_geoboundaries(self):
        units_gdf = admin.admin(RAINIER, level=2, simplified=True)
        assert {"Pierce", "Lewis"} <= set(units_gdf["name"])

    def test_natural_earth(self):
        assert len(natural_earth.load(layer="coastline", scale="110m")) > 100

    def test_mountains(self):
        assert mountains.load(RAINIER, subset="300")["name"].tolist() == [
            "Cascade Range"
        ]

    def test_rgi6_mirror(self):
        glaciers_gdf = glaciers.load(RAINIER, version="6.0", source="oggm-mirror")
        assert len(glaciers_gdf) > 150
        assert glaciers_gdf["name"].str.startswith("Emmons").any()

    @pytest.mark.requires_earthaccess
    def test_rgi7_nsidc(self):
        glaciers_gdf = glaciers.load(RAINIER)
        assert len(glaciers_gdf) > 150 and glaciers_gdf.attrs["source"] == "nsidc"


class TestEarthdataProbe:
    @pytest.fixture
    def session(self, monkeypatch):
        import earthaccess

        from easysnowdata import auth

        state: dict = {"status": 206, "url": glaciers.NSIDC_RGI7 + "/x.zip"}

        class Response:
            def __init__(self):
                self.status_code = state["status"]
                self.url = state["url"]

            def close(self):
                pass

        class Session:
            def get(self, url, **kwargs):
                state["asked"] = (url, kwargs["headers"]["Range"])
                return Response()

        monkeypatch.setattr(auth.get("earthdata"), "ensure", lambda **kwargs: None)
        monkeypatch.setattr(
            earthaccess, "get_requests_https_session", lambda: Session()
        )
        return state

    def test_first_byte_through_the_login(self, session):
        from easysnowdata.catalog import health

        health.earthdata_https_first_byte(glaciers.NSIDC_RGI7 + "/x.zip")
        assert session["asked"][1] == "bytes=0-0"

    def test_landing_on_the_login_page_fails(self, session):
        from easysnowdata.catalog import health

        session["url"] = "https://urs.earthdata.nasa.gov/oauth/authorize?client_id=x"
        session["status"] = 200
        with pytest.raises(RuntimeError, match="Earthdata Login"):
            health.earthdata_https_first_byte(glaciers.NSIDC_RGI7 + "/x.zip")
