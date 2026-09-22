"""Unit tests for easysnowdata.auth — monkeypatched env and files, no network."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import types
from pathlib import Path

import ee
import pandas as pd
import pytest

import easysnowdata
from easysnowdata import auth, config
from easysnowdata.auth import CredentialError, Detection, Provider
from easysnowdata.auth import earthdata as ed
from easysnowdata.auth import earthengine as eeprov

ALL_VARS = [v for p in auth.PROVIDERS.values() for v in p.env_vars] + [
    "NETRC",
    "EASYSNOWDATA_QUIET",
    "EASYSNOWDATA_REGION",
    "EASYSNOWDATA_CACHE_DIR",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No credential env vars; HOME (and USERPROFILE on Windows) in a temp dir."""
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(tmp_path / "cache"))
    auth.reset()
    yield tmp_path
    auth.reset()


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_provider_names(self):
        assert list(auth.PROVIDERS) == [
            "earthdata",
            "earthengine",
            "planetary_computer",
            "planet",
            "nve",
        ]

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown auth provider"):
            auth.get("nope")

    def test_top_level_exports(self):
        assert easysnowdata.auth is auth
        assert easysnowdata.CredentialError is CredentialError

    def test_status_table(self, clean_env):
        table = auth.status()
        assert isinstance(table, pd.DataFrame)
        assert list(table.index) == list(auth.PROVIDERS)
        assert set(["configured", "how", "optional", "needed_by"]) <= set(table.columns)
        assert not table.loc["earthdata", "configured"]
        assert table.loc["planetary_computer", "how"] == "anonymous"

    def test_ensure_and_env_stack(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVE_API_KEY", "k")
        assert auth.ensure("nve") == {"nve": "k"}
        with auth.env("nve", "planetary_computer") as options:
            assert options["headers"]["X-API-Key"] == "k"
            assert options["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"

    def test_ensure_missing_raises_before_network(self, clean_env):
        with pytest.raises(CredentialError) as info:
            auth.ensure("nve")
        assert info.value.provider == "nve"
        assert "NVE_API_KEY" in str(info.value)
        assert info.value.docs_url.startswith("https://")

    def test_login_all_logs_failures(self, clean_env, caplog):
        with caplog.at_level(logging.WARNING, logger="easysnowdata"):
            auth.login(interactive=False)
        assert "NVE HydAPI" in caplog.text

    def test_login_by_name_propagates(self, clean_env):
        with pytest.raises(CredentialError):
            auth.login("nve", interactive=False)

    def test_credential_error_defaults(self):
        err = CredentialError("boom")
        assert err.provider is None and err.docs_url
        assert str(err) == "boom"

    def test_base_provider_defaults(self, clean_env):
        class Dummy(Provider):
            name = "dummy"
            title = "Dummy"

            def detect(self):
                return Detection(False)

        d = Dummy()
        with pytest.raises(CredentialError, match="no interactive login"):
            d.login()
        with pytest.raises(CredentialError):
            d.ensure()
        with d.env() as opts:
            assert opts == {}
        with pytest.raises(NotImplementedError):
            Provider().detect()
        err = d.error(alternatives=("hosted-cog",), cause="x")
        assert "hosted-cog" in str(err) and "Cause: x" in str(err)

    def test_base_provider_ensure_caches(self, clean_env):
        class Dummy(Provider):
            name = "dummy"

            def detect(self):
                return Detection(True, "env:X")

        d = Dummy()
        assert d.ensure() is True and d.configured
        d.reset()
        assert d._ensured is None


# ── import-time summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_line_contents(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        line = auth.summary_line()
        assert line.startswith(f"easysnowdata {easysnowdata.__version__}")
        assert "NASA Earthdata ✓ (EARTHDATA_TOKEN)" in line
        assert (
            "Earth Engine ✗" in line and "Planet ✗" in line and "NVE HydAPI ✗" in line
        )
        assert "compute: local (HTTPS access)" in line

    def test_quiet_suppresses(self, clean_env, monkeypatch, capsys, caplog):
        monkeypatch.setenv("EASYSNOWDATA_QUIET", "1")
        monkeypatch.setattr(config, "is_interactive", lambda: True)
        with caplog.at_level(logging.INFO, logger="easysnowdata"):
            auth._emit_import_summary()
        assert capsys.readouterr().out == "" and "credentials:" not in caplog.text

    def test_interactive_prints(self, clean_env, monkeypatch, capsys):
        monkeypatch.setattr(config, "is_interactive", lambda: True)
        auth._emit_import_summary()
        assert "credentials:" in capsys.readouterr().out

    def test_non_interactive_logs(self, clean_env, monkeypatch, capsys, caplog):
        monkeypatch.setattr(config, "is_interactive", lambda: False)
        with caplog.at_level(logging.INFO, logger="easysnowdata"):
            auth._emit_import_summary()
        assert capsys.readouterr().out == ""
        assert "credentials:" in caplog.text


# ── earthdata ─────────────────────────────────────────────────────────────────


class TestEarthdata:
    provider = auth.get("earthdata")

    def test_nothing_configured(self, clean_env):
        det = self.provider.detect()
        assert not det and det.how is None
        assert self.provider.strategy() is None

    def test_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        assert self.provider.detect().how == "env:EARTHDATA_TOKEN"
        assert self.provider.strategy() == "environment"

    def test_username_password(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHDATA_USERNAME", "u")
        assert not self.provider.detect()  # password missing
        monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
        assert self.provider.detect().how == "env:EARTHDATA_USERNAME"

    def test_netrc(self, clean_env):
        netrc = clean_env / ".netrc"
        netrc.write_text("machine example.com login a password b\n")
        assert not self.provider.detect()
        netrc.write_text("machine urs.earthdata.nasa.gov login me password secret\n")
        det = self.provider.detect()
        assert det.how == "netrc" and det.details.endswith(".netrc")
        assert self.provider.strategy() == "netrc"

    def test_netrc_env_var_and_windows_name(self, clean_env, monkeypatch):
        alt = clean_env / "creds"
        alt.write_text("machine urs.earthdata.nasa.gov login me password secret\n")
        monkeypatch.setenv("NETRC", str(alt))
        assert ed.netrc_path() == alt
        assert self.provider.detect().how == "netrc"
        monkeypatch.delenv("NETRC")
        (clean_env / "_netrc").write_text(
            "machine urs.earthdata.nasa.gov login me password secret\n"
        )
        assert ed.netrc_path() == clean_env / "_netrc"

    def test_malformed_netrc_is_not_configured(self, clean_env):
        (clean_env / ".netrc").write_text(
            "bogus toplevel token urs.earthdata.nasa.gov\n"
        )
        assert ed.netrc_has_edl() is False

    def test_gdal_options_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        opts = self.provider.gdal_options()
        assert opts["GDAL_HTTP_AUTH"] == "BEARER" and opts["GDAL_HTTP_BEARER"] == "abc"
        assert "GDAL_HTTP_NETRC" not in opts
        assert Path(opts["GDAL_HTTP_COOKIEFILE"]).is_relative_to(clean_env / "cache")

    def test_gdal_options_netrc(self, clean_env, monkeypatch):
        alt = clean_env / "creds"
        alt.write_text("machine urs.earthdata.nasa.gov login me password secret\n")
        monkeypatch.setenv("NETRC", str(alt))
        opts = self.provider.gdal_options()
        assert opts["GDAL_HTTP_NETRC"] == "YES"
        assert opts["GDAL_HTTP_NETRC_FILE"] == str(alt)
        with self.provider.env() as effective:
            assert effective["GDAL_HTTP_NETRC"] == "YES"
            assert effective["GDAL_HTTP_MAX_RETRY"] == "5"

    def test_gdal_options_netrc_beats_a_token(self, clean_env, monkeypatch):
        """ASF's redirect chain drops a bearer token; the URS login works everywhere."""
        (clean_env / ".netrc").write_text(
            "machine urs.earthdata.nasa.gov login me password secret\n"
        )
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        opts = self.provider.gdal_options()
        assert opts["GDAL_HTTP_NETRC"] == "YES"
        assert opts["GDAL_HTTP_NETRC_FILE"] == str(clean_env / ".netrc")
        assert "GDAL_HTTP_BEARER" not in opts

    def test_gdal_options_username_password_write_a_netrc(self, clean_env, monkeypatch):
        """The CI secrets have no netrc, so one is written to the cache for GDAL."""
        monkeypatch.setenv("EARTHDATA_USERNAME", "u")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
        opts = self.provider.gdal_options()
        written = Path(opts["GDAL_HTTP_NETRC_FILE"])
        assert written.is_relative_to(clean_env / "cache")
        assert (
            written.read_text() == "machine urs.earthdata.nasa.gov login u password p\n"
        )
        if os.name != "nt":  # Windows has no POSIX mode bits to check
            assert (written.stat().st_mode & 0o777) == 0o600
        assert "GDAL_HTTP_BEARER" not in opts
        monkeypatch.setenv("EARTHDATA_PASSWORD", "changed")
        assert (
            "password changed"
            in Path(self.provider.gdal_options()["GDAL_HTTP_NETRC_FILE"]).read_text()
        )

    @pytest.fixture
    def fake_earthaccess(self, monkeypatch):
        import earthaccess

        state = {"auth": types.SimpleNamespace(authenticated=False), "calls": []}

        def login(strategy, persist=False):
            state["calls"].append(strategy)
            state["auth"].authenticated = True
            earthaccess._store = object()
            return state["auth"]

        monkeypatch.setattr(earthaccess, "_auth", state["auth"])
        monkeypatch.setattr(earthaccess, "_store", None)
        monkeypatch.setattr(earthaccess, "login", login)
        monkeypatch.setattr(ed.time, "sleep", lambda s: None)
        return state

    def test_ensure_logs_in_once(self, clean_env, monkeypatch, fake_earthaccess):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        self.provider.ensure()
        self.provider.ensure()
        assert fake_earthaccess["calls"] == ["environment"]

    def test_ensure_without_store_logs_in_again(
        self, clean_env, monkeypatch, fake_earthaccess
    ):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        fake_earthaccess["auth"].authenticated = True  # half-initialised state
        self.provider.ensure()
        assert fake_earthaccess["calls"] == ["environment"]

    def test_ensure_without_credentials_raises(self, clean_env, fake_earthaccess):
        with pytest.raises(CredentialError) as info:
            self.provider.ensure()
        assert info.value.provider == "earthdata" and fake_earthaccess["calls"] == []

    def test_ensure_retries_then_wraps(self, clean_env, monkeypatch, fake_earthaccess):
        import earthaccess

        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")

        def failing(strategy, persist=False):
            fake_earthaccess["calls"].append(strategy)
            raise ConnectionError("Network is unreachable")

        monkeypatch.setattr(earthaccess, "login", failing)
        with pytest.raises(CredentialError, match="Network is unreachable"):
            self.provider.ensure()
        assert len(fake_earthaccess["calls"]) == 2

    def test_ensure_rejected_credentials(
        self, clean_env, monkeypatch, fake_earthaccess
    ):
        import earthaccess

        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")

        def rejected(strategy, persist=False):
            fake_earthaccess["calls"].append(strategy)
            return types.SimpleNamespace(authenticated=False)

        monkeypatch.setattr(earthaccess, "login", rejected)
        with pytest.raises(CredentialError, match="login failed"):
            self.provider.ensure()
        assert fake_earthaccess["calls"] == ["environment"]

    def test_login_interactive(self, clean_env, fake_earthaccess):
        self.provider.login(persist=False)
        assert fake_earthaccess["calls"] == ["interactive"]
        with pytest.raises(CredentialError):
            self.provider.reset()
            self.provider.login(interactive=False)

    def test_login_when_configured_ensures(
        self, clean_env, monkeypatch, fake_earthaccess
    ):
        monkeypatch.setenv("EARTHDATA_TOKEN", "abc")
        self.provider.login()
        assert fake_earthaccess["calls"] == ["environment"]


# ── earth engine ──────────────────────────────────────────────────────────────

SA_KEY = {
    "type": "service_account",
    "project_id": "sa-project",
    "client_email": "bot@sa-project.iam.gserviceaccount.com",
    "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-key\n-----END PRIVATE KEY-----\n",
}
OAUTH = {"refresh_token": "rt", "scopes": ["s"], "project": "oauth-project"}


class TestEarthEngine:
    provider = auth.get("earthengine")

    @pytest.fixture
    def fake_ee(self, monkeypatch):
        state = {"init": [], "initialized": False}

        def initialize(**kwargs):
            state["init"].append(kwargs)
            state["initialized"] = True

        monkeypatch.setattr(ee, "Initialize", initialize)
        monkeypatch.setattr(ee.data, "is_initialized", lambda: state["initialized"])
        monkeypatch.setattr(
            ee, "ServiceAccountCredentials", lambda email, key_data=None: ("sa", email)
        )
        return state

    def test_detect_nothing(self, clean_env):
        assert not self.provider.detect()
        assert self.provider.project_id() is None

    def test_detect_token_adc_file(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHENGINE_TOKEN", json.dumps(OAUTH))
        assert self.provider.detect().how == "env:EARTHENGINE_TOKEN"
        monkeypatch.delenv("EARTHENGINE_TOKEN")
        key = clean_env / "key.json"
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        assert not self.provider.detect()  # file must exist
        key.write_text(json.dumps(SA_KEY))
        assert self.provider.detect().how == "adc"
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
        creds = eeprov.credentials_path()
        creds.parent.mkdir(parents=True)
        creds.write_text(json.dumps(OAUTH))
        assert self.provider.detect().how == "file"
        assert self.provider.project_id() == "oauth-project"

    def test_decode_token_forms(self):
        raw = json.dumps(OAUTH)
        assert eeprov.decode_token(raw)["refresh_token"] == "rt"
        b64 = base64.encodebytes(raw.encode()).decode()
        assert eeprov.decode_token(b64)["project"] == "oauth-project"
        assert eeprov.decode_token(None) is None and eeprov.decode_token("  ") is None
        with pytest.raises(ValueError, match="neither JSON"):
            eeprov.decode_token("not json!")
        with pytest.raises(ValueError, match="JSON object"):
            eeprov.decode_token("[1, 2]")

    def test_credentials_from_token(self, clean_env, monkeypatch, fake_ee):
        assert eeprov.credentials_from_token(json.dumps(SA_KEY)) == (
            "sa",
            SA_KEY["client_email"],
        )
        creds = eeprov.credentials_from_token(json.dumps(OAUTH))
        assert creds.refresh_token == "rt" and creds.quota_project_id == "oauth-project"
        assert creds.client_id == ee.oauth.CLIENT_ID
        with pytest.raises(ValueError, match="neither a service-account"):
            eeprov.credentials_from_token('{"foo": 1}')
        assert eeprov.credentials_from_token() is None

    def test_project_precedence(self, clean_env, monkeypatch):
        monkeypatch.setenv("EARTHENGINE_TOKEN", json.dumps(OAUTH))
        assert self.provider.project_id() == "oauth-project"
        monkeypatch.setenv("EARTHENGINE_PROJECT", "alias-project")
        assert self.provider.project_id() == "alias-project"
        monkeypatch.setenv("EE_PROJECT_ID", "geemap-project")
        assert self.provider.project_id() == "geemap-project"
        monkeypatch.setenv("EARTHENGINE_TOKEN", "garbage")
        monkeypatch.delenv("EE_PROJECT_ID")
        monkeypatch.delenv("EARTHENGINE_PROJECT")
        assert self.provider.project_id() is None  # bad token is not fatal here

    def test_ensure_with_token_once(self, clean_env, monkeypatch, fake_ee):
        monkeypatch.setenv(
            "EARTHENGINE_TOKEN", base64.b64encode(json.dumps(SA_KEY).encode()).decode()
        )
        self.provider.ensure()
        self.provider.ensure()
        assert len(fake_ee["init"]) == 1
        call = fake_ee["init"][0]
        assert call["project"] == "sa-project"
        assert call["opt_url"] == eeprov.HIGH_VOLUME_URL
        assert call["credentials"] == ("sa", SA_KEY["client_email"])
        assert self.provider.xee_init_kwargs()["project"] == "sa-project"

    def test_ensure_kwargs_force_reinit(self, clean_env, monkeypatch, fake_ee):
        monkeypatch.setenv("EARTHENGINE_TOKEN", json.dumps(OAUTH))
        self.provider.ensure()
        self.provider.ensure(project="other", url="https://x")
        assert len(fake_ee["init"]) == 2
        assert (
            fake_ee["init"][1]["project"] == "other"
            and "opt_url" not in fake_ee["init"][1]
        )

    def test_ensure_reinitialises_after_reset_by_ee(
        self, clean_env, monkeypatch, fake_ee
    ):
        monkeypatch.setenv("EARTHENGINE_TOKEN", json.dumps(OAUTH))
        self.provider.ensure()
        fake_ee["initialized"] = False  # e.g. the user called ee.Reset()
        self.provider.ensure()
        assert len(fake_ee["init"]) == 2

    def test_ensure_file_without_project_raises(self, clean_env, fake_ee):
        creds = eeprov.credentials_path()
        creds.parent.mkdir(parents=True)
        creds.write_text(json.dumps({"refresh_token": "rt"}))
        with pytest.raises(CredentialError, match="Cloud project"):
            self.provider.ensure()
        assert fake_ee["init"] == []

    def test_ensure_adc_without_project_is_allowed(
        self, clean_env, monkeypatch, fake_ee
    ):
        key = clean_env / "key.json"
        key.write_text(json.dumps(SA_KEY))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        self.provider.ensure()
        assert (
            fake_ee["init"][0]["project"] is None
            and "credentials" not in fake_ee["init"][0]
        )

    def test_ensure_bad_token(self, clean_env, monkeypatch, fake_ee):
        monkeypatch.setenv("EARTHENGINE_TOKEN", "garbage")
        with pytest.raises(CredentialError, match="could not be decoded"):
            self.provider.ensure()

    def test_ensure_missing_raises(self, clean_env, fake_ee):
        with pytest.raises(CredentialError) as info:
            self.provider.ensure()
        assert info.value.provider == "earthengine"

    def test_login(self, clean_env, monkeypatch, fake_ee):
        called = []
        monkeypatch.setattr(ee, "Authenticate", lambda **kw: called.append(kw))
        with pytest.raises(CredentialError):
            self.provider.login(interactive=False)
        monkeypatch.setenv("EE_PROJECT_ID", "p")

        def authenticate(**kw):
            called.append(kw)
            eeprov.credentials_path().parent.mkdir(parents=True, exist_ok=True)
            eeprov.credentials_path().write_text(json.dumps(OAUTH))

        monkeypatch.setattr(ee, "Authenticate", authenticate)
        self.provider.login()
        assert called == [{}] and fake_ee["init"][0]["project"] == "p"


# ── planetary computer ────────────────────────────────────────────────────────


class TestPlanetaryComputer:
    provider = auth.get("planetary_computer")

    def test_anonymous_key_file(self, clean_env, monkeypatch):
        assert self.provider.detect().how == "anonymous"
        assert self.provider.optional
        settings = clean_env / ".planetarycomputer" / "settings.env"
        settings.parent.mkdir()
        settings.write_text("PC_SDK_SUBSCRIPTION_KEY=x\n")
        assert self.provider.detect().how.startswith("file:")
        monkeypatch.setenv("PC_SDK_SUBSCRIPTION_KEY", "k")
        assert self.provider.detect().how == "env:PC_SDK_SUBSCRIPTION_KEY"

    def test_ensure_sign_login(self, clean_env):
        import planetary_computer

        assert self.provider.ensure() is planetary_computer
        assert self.provider.sign is planetary_computer.sign_inplace
        with pytest.raises(CredentialError, match="no interactive login"):
            self.provider.login()


# ── planet ────────────────────────────────────────────────────────────────────


class TestPlanet:
    provider = auth.get("planet")

    def test_detect_env_routes(self, clean_env, monkeypatch):
        assert not self.provider.detect()
        monkeypatch.setenv("PL_AUTH_CLIENT_ID", "id")
        assert not self.provider.detect()
        monkeypatch.setenv("PL_AUTH_CLIENT_SECRET", "s")
        assert self.provider.detect().how == "env:PL_AUTH_CLIENT_ID"
        monkeypatch.setenv("PL_AUTH_API_KEY", "k")
        assert self.provider.detect().how == "env:PL_AUTH_API_KEY"
        monkeypatch.setenv("PL_API_KEY", "k")
        assert self.provider.detect().how == "env:PL_API_KEY"

    def test_detect_files(self, clean_env, monkeypatch):
        sessions = clean_env / ".planet"
        sessions.mkdir()
        assert not self.provider.detect()  # empty directory
        (sessions / "default.json").write_text("{}")
        det = self.provider.detect()
        assert det.how == "file:~/.planet/" and det.details == "profile default"
        monkeypatch.setenv("PL_AUTH_PROFILE", "research")
        assert self.provider.detect().details == "profile research"
        (clean_env / ".planet.json").write_text('{"key": "x"}')
        assert self.provider.detect().how == "file:~/.planet.json"

    @pytest.fixture
    def fake_planet(self, monkeypatch):
        calls = {"login": [], "clients": 0}

        class Auth:
            @staticmethod
            def from_user_default_session():
                return Auth()

            def user_login(self, **kw):
                calls["login"].append(kw)

        class Planet:
            def __init__(self, **kw):
                calls["clients"] += 1

        monkeypatch.setitem(
            sys.modules, "planet", types.SimpleNamespace(Auth=Auth, Planet=Planet)
        )
        return calls

    def test_ensure_without_credentials(self, clean_env, fake_planet):
        with pytest.raises(CredentialError) as info:
            self.provider.ensure()
        assert info.value.provider == "planet" and fake_planet["clients"] == 0

    def test_ensure_builds_one_client(self, clean_env, monkeypatch, fake_planet):
        monkeypatch.setenv("PL_API_KEY", "k")
        c1 = self.provider.ensure()
        assert self.provider.ensure() is c1 and fake_planet["clients"] == 1

    def test_login_runs_sdk_flow(self, clean_env, monkeypatch, fake_planet):
        with pytest.raises(CredentialError):
            self.provider.login(interactive=False)
        # the SDK flow writes a session; emulate that so ensure() succeeds afterwards
        (clean_env / ".planet").mkdir()
        (clean_env / ".planet" / "default.json").write_text("{}")
        self.provider.reset()
        self.provider.login()
        assert (
            fake_planet["login"] == [] and fake_planet["clients"] == 1
        )  # already configured
        (clean_env / ".planet" / "default.json").unlink()
        self.provider.reset()

        def user_login(self_, **kw):
            fake_planet["login"].append(kw)
            (clean_env / ".planet" / "default.json").write_text("{}")

        monkeypatch.setattr(sys.modules["planet"].Auth, "user_login", user_login)
        self.provider.login()
        assert fake_planet["login"] == [
            {"allow_open_browser": True, "allow_tty_prompt": True}
        ]

    def test_sdk_missing(self, clean_env, monkeypatch):
        monkeypatch.setitem(sys.modules, "planet", None)  # makes `import planet` fail
        monkeypatch.setenv("PL_API_KEY", "k")
        with pytest.raises(CredentialError, match="pip install planet"):
            self.provider.ensure()
        monkeypatch.delenv("PL_API_KEY")
        with pytest.raises(CredentialError, match="pip install planet"):
            self.provider.login()
        with pytest.raises(ImportError):
            self.provider._sdk()


# ── nve ───────────────────────────────────────────────────────────────────────


class TestNVE:
    provider = auth.get("nve")

    def test_detect_ensure_headers(self, clean_env, monkeypatch):
        assert not self.provider.detect()
        with pytest.raises(CredentialError):
            self.provider.ensure()
        monkeypatch.setenv("NVE_API_KEY", " key ")
        assert self.provider.detect().how == "env:NVE_API_KEY"
        assert self.provider.ensure() == "key"
        assert self.provider.headers()["X-API-Key"] == "key"
        with pytest.raises(CredentialError, match="no interactive login"):
            self.provider.login()


class TestEarthdataTokenFallback:
    """An expired EARTHDATA_TOKEN is dropped in favour of username/password."""

    provider = auth.get("earthdata")

    @pytest.fixture
    def urs(self, monkeypatch):
        import requests

        state = {"status": 401, "calls": 0}

        def get(self, url, headers=None, timeout=None):
            state["calls"] += 1
            assert self.trust_env is False  # or netrc basic auth masks the bearer
            assert url == ed.URS_TOKENS_URL and headers["Authorization"].startswith(
                "Bearer "
            )
            return types.SimpleNamespace(status_code=state["status"])

        monkeypatch.setattr(requests.Session, "get", get)
        return state

    @pytest.fixture
    def fake_earthaccess(self, monkeypatch):
        import earthaccess

        state = {"auth": types.SimpleNamespace(authenticated=False), "calls": []}

        def login(strategy, persist=False):
            state["calls"].append((strategy, os.environ.get("EARTHDATA_TOKEN")))
            state["auth"].authenticated = True
            state["auth"].token = {"access_token": "fresh"}
            earthaccess._store = object()
            return state["auth"]

        monkeypatch.setattr(earthaccess, "_auth", state["auth"])
        monkeypatch.setattr(earthaccess, "_store", None)
        monkeypatch.setattr(earthaccess, "login", login)
        monkeypatch.setattr(ed.time, "sleep", lambda s: None)
        return state

    def test_token_is_valid_reads_the_status(self, urs):
        assert ed.token_is_valid("t") is False
        urs["status"] = 200
        assert ed.token_is_valid("t") is True
        urs["status"] = 500
        assert ed.token_is_valid("t") is None

    def test_rejected_token_falls_back_to_the_password(
        self, clean_env, monkeypatch, urs, fake_earthaccess
    ):
        monkeypatch.setenv("EARTHDATA_TOKEN", "expired")
        monkeypatch.setenv("EARTHDATA_USERNAME", "u")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
        self.provider.ensure()
        # logged in with the environment strategy, and the token was gone by then
        assert fake_earthaccess["calls"] == [("environment", None)]
        assert "EARTHDATA_TOKEN" not in os.environ
        # GDAL reads log in at URS with the password, not with the rejected token
        opts = self.provider.gdal_options()
        assert "GDAL_HTTP_BEARER" not in opts
        assert "login u password p" in Path(opts["GDAL_HTTP_NETRC_FILE"]).read_text()

    def test_valid_token_is_kept(self, clean_env, monkeypatch, urs, fake_earthaccess):
        urs["status"] = 200
        monkeypatch.setenv("EARTHDATA_TOKEN", "good")
        self.provider.ensure()
        assert fake_earthaccess["calls"] == [("environment", "good")]
        assert self.provider.gdal_options()["GDAL_HTTP_BEARER"] == "good"

    def test_rejected_token_without_fallback_names_the_cause(
        self, clean_env, monkeypatch, urs, fake_earthaccess
    ):
        monkeypatch.setenv("EARTHDATA_TOKEN", "expired")
        monkeypatch.setattr(ed, "netrc_has_edl", lambda: False)
        with pytest.raises(CredentialError, match="rejected"):
            self.provider.ensure()
        assert fake_earthaccess["calls"] == []

    def test_unreachable_urs_gives_the_token_the_benefit_of_the_doubt(
        self, clean_env, monkeypatch, urs, fake_earthaccess
    ):
        urs["status"] = 503
        monkeypatch.setenv("EARTHDATA_TOKEN", "maybe")
        self.provider.ensure()
        assert fake_earthaccess["calls"] == [("environment", "maybe")]
