"""Sturm & Liston snow classification: catalog entry, both routes offline, live smoke."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from easysnowdata import auth, catalog, plotting
from easysnowdata.snow import snow_classification as sc

RAINIER = (-121.94, 46.72, -121.54, 46.99)


@pytest.fixture
def local_cog(static_fixtures, monkeypatch):
    """Point the hosted-COG route at the local fixture."""
    monkeypatch.setattr(sc, "HOSTED_COG_URL", str(static_fixtures["snow_class_cog"]))
    return static_fixtures["snow_class_cog"]


@pytest.fixture
def fake_earthdata(static_fixtures, monkeypatch, tmp_path):
    """An Earthdata session that serves the local fixture, plus a private cache."""
    import earthaccess

    provider = auth.get("earthdata")
    monkeypatch.setattr(provider, "detect", lambda: auth.Detection(True, "env:TEST"))
    monkeypatch.setattr(provider, "ensure", lambda **kw: True)
    monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path))
    calls: dict = {}

    class Response:
        def __init__(self, url):
            self.status_code = 404 if "nope" in url else 200
            self._body = (
                static_fixtures["snow_class_cog"].read_bytes()
                if self.status_code == 200
                else b""
            )

        def iter_content(self, chunk_size=1):
            for start in range(0, len(self._body), chunk_size):
                yield self._body[start : start + chunk_size]

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Session:
        def get(self, url, **kwargs):
            calls.setdefault("urls", []).append(url)
            return Response(url)

    monkeypatch.setattr(earthaccess, "get_requests_https_session", lambda: Session())
    return calls


class TestCatalogEntry:
    def test_registered_with_nsidc_as_the_default(self):
        product = catalog.get("snow-classification")
        assert product.resolve_loader() is sc.load
        assert [s.id for s in product.sources] == ["nsidc", "hosted-cog"]
        assert product.requires == ("earthdata",)
        assert [s.id for s in product.credential_free_sources] == ["hosted-cog"]
        assert [p.label for s in product.sources for p in s.health] == [
            "Sturm & Liston snow classification (NSIDC-0768)",
            "Sturm & Liston snow classification (Azure)",
        ]
        assert catalog.validate_all() == []

    def test_the_nsidc_probe_runs_without_credentials(self):
        probe = catalog.get("snow-classification").sources[0].health[0]
        assert probe.requires == ()

    def test_filenames(self):
        assert sc.filename() == "SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
        assert sc.filename("30arcmin") == "SnowClass_GL_50km_0.50degree_2021_v01.0.tif"
        assert sc.filename("10arcsec", "NA").startswith("SnowClass_NA_300m")
        with pytest.raises(ValueError, match="Unknown resolution"):
            sc.filename("1km")
        with pytest.raises(ValueError, match="region must be"):
            sc.filename("10arcsec", "EU")

    def test_missing_credentials_name_the_free_route(self, monkeypatch):
        monkeypatch.setattr(
            auth.get("earthdata"), "detect", lambda: auth.Detection(False)
        )
        with pytest.raises(auth.CredentialError) as excinfo:
            sc.load(RAINIER)
        message = str(excinfo.value)
        assert excinfo.value.provider == "earthdata"
        assert 'source="hosted-cog"' in message


@pytest.mark.recorded
class TestHostedCogRoute:
    def test_output_contract(self, local_cog):
        da = sc.load(RAINIER, source="hosted-cog")
        assert isinstance(da, xr.DataArray) and da.name == "snow_class"
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 9 and da.rio.encoded_nodata is None
        assert da.attrs["product_id"] == "snow-classification"
        assert da.attrs["source"] == "hosted-cog" and da.attrs["epoch"] == "2021"
        assert da.attrs["flag_values"] == list(range(1, 10))
        assert da.attrs["flag_meanings"].split()[0] == "Tundra"
        assert da.attrs["long_name"] == "seasonal snow class"
        assert str(da.time.values)[:4] == "2021"
        assert set(np.unique(da.values)) <= set(da.attrs["flag_values"])

    def test_masked_form(self, local_cog):
        da = sc.load(RAINIER, source="hosted-cog", mask=True)
        assert da.dtype == "float32" and da.rio.encoded_nodata == 9
        assert bool(da.isnull().any())

    def test_plotting_reads_the_flags(self, local_cog):
        ax = plotting.categorical(sc.load(RAINIER, source="hosted-cog"))
        assert [t.get_text() for t in ax.get_legend().get_texts()][0] == "Tundra"

    def test_chunks(self, local_cog):
        assert sc.load(RAINIER, source="hosted-cog", chunks=None).chunks is None
        da = sc.load(RAINIER, source="hosted-cog", chunks={"x": 16, "y": 16})
        assert max(max(sizes) for sizes in da.chunks) <= 16


@pytest.mark.recorded
class TestNsidcRoute:
    def test_file_is_downloaded_once_then_cached(self, fake_earthdata):
        da = sc.load(RAINIER)
        assert fake_earthdata["urls"] == [
            f"{sc.NSIDC_DIRECTORY}/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
        ]
        assert da.attrs["source"] == "nsidc"
        assert da.attrs["source_url"].startswith(sc.NSIDC_DIRECTORY)
        assert da.attrs["file"] == "SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif"
        assert da.dims == ("latitude", "longitude") and da.rio.nodata == 9
        sc.load(RAINIER)  # the second call reads the cached file
        assert len(fake_earthdata["urls"]) == 1

    def test_coarse_grid_picks_another_file(self, fake_earthdata):
        sc.load(RAINIER, resolution="2.5arcmin")
        assert fake_earthdata["urls"][0].endswith(
            "SnowClass_GL_05km_2.50arcmin_2021_v01.0.tif"
        )

    def test_missing_file_says_where_to_look(self, fake_earthdata, monkeypatch):
        monkeypatch.setattr(sc, "filename", lambda *a, **kw: "nope.tif")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            sc.load(RAINIER)


class TestProbe:
    @pytest.mark.parametrize(
        ("status", "location"),
        [(200, ""), (302, "https://urs.earthdata.nasa.gov/oauth/authorize")],
    )
    def test_alive(self, monkeypatch, status, location):
        monkeypatch.setattr(
            "requests.get", lambda *a, **kw: _FakeResponse(status, location)
        )
        sc._nsidc_directory_probe()

    def test_dead(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda *a, **kw: _FakeResponse(500, ""))
        with pytest.raises(RuntimeError, match="Unreachable"):
            sc._nsidc_directory_probe()


class _FakeResponse:
    def __init__(self, status_code, location):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}

    def close(self):
        pass


@pytest.mark.live
class TestLive:
    def test_hosted_cog_smoke(self):
        da = sc.load(RAINIER, source="hosted-cog")
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
        assert da.rio.crs.to_epsg() == 4326 and da.odc.crs.epsg == 4326
        assert da.rio.nodata == 9
        assert da.attrs["product_id"] == "snow-classification"
        assert da.attrs["flag_meanings"].split()[0] == "Tundra"
        assert set(np.unique(da.values)) <= set(range(1, 10))

    def test_the_nsidc_directory_is_up(self):
        sc._nsidc_directory_probe()

    @pytest.mark.requires_earthaccess
    def test_nsidc_smoke(self):
        da = sc.load(RAINIER)
        assert da.attrs["source"] == "nsidc" and da.rio.nodata == 9
        assert da.dims == ("latitude", "longitude") and da.dtype == "uint8"
