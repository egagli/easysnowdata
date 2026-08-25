"""Tests for Earth Engine credential handling via ``EARTHENGINE_TOKEN``.

The token may be a service-account key JSON or the ``~/.config/earthengine``
OAuth JSON, raw or base64-encoded. Parsing tests need no network; the
``TestEarthEngine`` tests need a real token and are skipped without one.
"""

from __future__ import annotations

import base64
import json
import os

import ee
import pytest

from easysnowdata.utils import _ee_credentials_from_token, initialize_earthengine

FAKE_OAUTH_TOKEN = {
    "client_id": "id",
    "client_secret": "secret",
    "refresh_token": "rt",
    "project": "my-project",
}


def create_ee_credentials():
    """Create Earth Engine credentials from the environment token."""
    if not os.getenv("EARTHENGINE_TOKEN"):
        raise pytest.skip("EARTHENGINE_TOKEN environment variable not set")
    return _ee_credentials_from_token()


@pytest.fixture
def ee_credentials():
    """Fixture to set up Earth Engine credentials."""
    return create_ee_credentials()


class TestTokenParsing:
    def test_raw_oauth_json(self):
        creds = _ee_credentials_from_token(json.dumps(FAKE_OAUTH_TOKEN))
        assert creds.refresh_token == "rt"
        assert creds.quota_project_id == "my-project"

    def test_base64_oauth_json(self):
        encoded = base64.b64encode(json.dumps(FAKE_OAUTH_TOKEN).encode()).decode()
        creds = _ee_credentials_from_token(encoded)
        assert creds.client_id == "id"

    def test_base64_with_line_wraps(self):
        encoded = base64.encodebytes(json.dumps(FAKE_OAUTH_TOKEN).encode()).decode()
        assert "\n" in encoded
        assert _ee_credentials_from_token(encoded).client_secret == "secret"

    def test_credentials_file_without_client_id(self):
        # Format written by recent earthengine-api versions (no client_id/secret)
        token = {"refresh_token": "rt", "scopes": ["s"], "project": "p"}
        creds = _ee_credentials_from_token(json.dumps(token))
        assert creds.client_id == ee.oauth.CLIENT_ID
        assert creds.refresh_token == "rt"

    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("EARTHENGINE_TOKEN", raising=False)
        assert _ee_credentials_from_token() is None

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError):
            _ee_credentials_from_token('{"invalid": "token"}')
        with pytest.raises(ValueError):
            _ee_credentials_from_token("not json, not base64!")

    def test_missing_token_skips(self, monkeypatch):
        monkeypatch.delenv("EARTHENGINE_TOKEN", raising=False)
        with pytest.raises(pytest.skip.Exception):
            create_ee_credentials()


class TestEarthEngine:
    def test_ee_initialization(self, ee_credentials):
        """Test Earth Engine initializes successfully"""
        initialize_earthengine()
        assert ee.data.is_initialized()

    def test_ee_api_connection(self, ee_credentials):
        """Test Earth Engine API connection works"""
        initialize_earthengine()
        response = ee.String("Greetings from the Earth Engine servers!").getInfo()
        assert isinstance(response, str)
        assert "Greetings" in response
