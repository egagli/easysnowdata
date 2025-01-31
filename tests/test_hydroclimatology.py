# #!/usr/bin/env python

# """Tests for `easysnowdata` package."""


# import unittest

# from easysnowdata import easysnowdata


# class TestEasysnowdata(unittest.TestCase):
#     """Tests for `easysnowdata` package."""

#     def setUp(self):
#         """Set up test fixtures, if any."""

#     def tearDown(self):
#         """Tear down test fixtures, if any."""

#     def test_000_something(self):
#         """Test something."""


# https://github.com/gee-community/ee-initialize-github-actions?tab=readme-ov-file#4-initialize-to-earth-engine-in-your-test-file
import ee
import json
import os
import google.oauth2.credentials

stored = json.loads(os.getenv("EARTHENGINE_TOKEN"))
credentials = google.oauth2.credentials.Credentials(
    None,
    token_uri="https://oauth2.googleapis.com/token",
    client_id=stored["client_id"],
    client_secret=stored["client_secret"],
    refresh_token=stored["refresh_token"],
    quota_project_id=stored["project"],
)

ee.Initialize(credentials=credentials)

print(ee.String("Greetings from the Earth Engine servers!").getInfo())
