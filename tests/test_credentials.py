# """Tests for `easysnowdata` package."""
import pytest

import ee
import json
import os
import google.oauth2.credentials


def create_ee_credentials():
    """Create Earth Engine credentials from environment token"""
    if not os.getenv("EARTHENGINE_TOKEN"):
        raise pytest.skip("EARTHENGINE_TOKEN environment variable not set")
    
    stored = json.loads(os.getenv("EARTHENGINE_TOKEN"))
    return google.oauth2.credentials.Credentials(
        None,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=stored["client_id"],
        client_secret=stored["client_secret"],
        refresh_token=stored["refresh_token"],
        quota_project_id=stored["project"],
    )

@pytest.fixture
def ee_credentials():
    """Fixture to setup Earth Engine credentials"""
    return create_ee_credentials()

class TestEarthEngine:
    def test_ee_initialization(self, ee_credentials):
        """Test Earth Engine initializes successfully"""
        ee.Initialize(credentials=ee_credentials)
        assert ee.data._credentials is not None
    
    def test_ee_api_connection(self, ee_credentials):
        """Test Earth Engine API connection works"""
        ee.Initialize(credentials=ee_credentials)
        response = ee.String("Greetings from the Earth Engine servers!").getInfo()
        assert isinstance(response, str)
        assert "Greetings" in response
    
    def test_missing_token(self, monkeypatch):
        """Test handling of missing environment variable"""
        monkeypatch.delenv("EARTHENGINE_TOKEN", raising=False)
        with pytest.raises(pytest.skip.Exception):
            create_ee_credentials()
    
    def test_invalid_token(self, monkeypatch):
        """Test handling of invalid token"""
        monkeypatch.setenv("EARTHENGINE_TOKEN", '{"invalid": "token"}')
        with pytest.raises(KeyError):
            create_ee_credentials()
            

# https://github.com/gee-community/ee-initialize-github-actions?tab=readme-ov-file#4-initialize-to-earth-engine-in-your-test-file
# import ee
# import json
# import os
# import google.oauth2.credentials

# stored = json.loads(os.getenv("EARTHENGINE_TOKEN"))
# credentials = google.oauth2.credentials.Credentials(
#     None,
#     token_uri="https://oauth2.googleapis.com/token",
#     client_id=stored["client_id"],
#     client_secret=stored["client_secret"],
#     refresh_token=stored["refresh_token"],
#     quota_project_id=stored["project"],
# )

# ee.Initialize(credentials=credentials)

# print(ee.String("Greetings from the Earth Engine servers!").getInfo())
