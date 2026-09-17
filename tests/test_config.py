"""Unit tests for easysnowdata.config (region detection, quiet, cache dir)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from easysnowdata import config


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for var in (
        "EASYSNOWDATA_QUIET",
        "EASYSNOWDATA_REGION",
        "EASYSNOWDATA_CACHE_DIR",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        *config._AWS_EXECUTION_HINTS,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "_DMI_DIR", tmp_path / "dmi")
    config.region(refresh=True)
    yield tmp_path
    config.region(refresh=True)


class TestFlags:
    @pytest.mark.parametrize(
        "value, expected",
        [("1", True), ("true", True), ("YES", True), ("0", False), ("", False)],
    )
    def test_quiet(self, clean, monkeypatch, value, expected):
        monkeypatch.setenv("EASYSNOWDATA_QUIET", value)
        assert config.quiet() is expected

    def test_is_interactive_repl_and_ipython(self, monkeypatch):
        monkeypatch.setattr(sys, "ps1", ">>> ", raising=False)
        assert config.is_interactive()
        monkeypatch.delattr(sys, "ps1", raising=False)
        monkeypatch.setitem(
            sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: object())
        )
        assert config.is_interactive()
        monkeypatch.setitem(
            sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: None)
        )
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: False))
        assert not config.is_interactive()
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: True))
        assert config.is_interactive()
        monkeypatch.setattr(sys, "stdout", object())
        assert not config.is_interactive()

    def test_cache_dir_override(self, clean, monkeypatch):
        monkeypatch.setenv("EASYSNOWDATA_CACHE_DIR", str(clean / "c"))
        path = config.cache_dir("a", "b")
        assert path == clean / "c" / "a" / "b" and path.is_dir()

    def test_cache_dir_default_is_platform_cache(self, clean):
        import pooch

        assert config.cache_dir() == Path(pooch.os_cache("easysnowdata"))


class TestRegion:
    def test_default_is_local(self, clean):
        reg = config.region()
        assert not reg.on_aws and not reg.direct_s3
        assert reg.describe() == "local (HTTPS access)" == str(reg)
        assert config.access_mode() == "https"

    def test_override(self, clean, monkeypatch):
        monkeypatch.setenv("EASYSNOWDATA_REGION", "us-west-2")
        reg = config.region(refresh=True)
        assert reg.on_aws and reg.direct_s3 and reg.source == "EASYSNOWDATA_REGION"
        assert reg.describe() == "AWS us-west-2 (direct S3 access)"
        assert config.access_mode(reg) == "direct"
        monkeypatch.setenv("EASYSNOWDATA_REGION", "local")
        assert not config.region(refresh=True).on_aws

    def test_env_hints(self, clean, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        assert not config.region(refresh=True).on_aws  # a laptop with boto config
        monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_ECS_FARGATE")
        reg = config.region(refresh=True)
        assert reg.on_aws and reg.region == "eu-west-1" and reg.source == "env"
        assert reg.describe() == "AWS eu-west-1 (HTTPS access)"

    def test_dmi_without_region(self, clean):
        dmi = clean / "dmi"
        dmi.mkdir()
        (dmi / "product_uuid").write_text("EC2A1B2C-...\n")
        reg = config.region(refresh=True)
        assert reg.on_aws and reg.region is None and reg.source == "dmi"
        assert reg.describe() == "AWS (region unknown) (HTTPS access)"
        (dmi / "product_uuid").write_text("nope")
        (dmi / "sys_vendor").write_text("Amazon EC2\n")
        assert config.region(refresh=True).source == "dmi"
        (dmi / "sys_vendor").write_text("QEMU")
        (dmi / "board_asset_tag").write_text("i-0123456789abcdef0")
        assert config.region(refresh=True).source == "dmi"
        (dmi / "board_asset_tag").write_text("none")
        assert not config.region(refresh=True).on_aws

    def test_probe_uses_imds_once(self, clean, monkeypatch):
        calls = []

        class Resp:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            config, "_region_from_imds", lambda: calls.append(1) or "us-west-2"
        )
        reg = config.region(probe=True)
        assert reg.direct_s3 and reg.source == "imds"
        config.region(probe=True)
        assert calls == [1]  # cached

    def test_probe_failure_stays_local(self, clean, monkeypatch):
        monkeypatch.setattr(config, "_region_from_imds", lambda: None)
        assert not config.region(probe=True).on_aws

    def test_probe_not_run_with_override(self, clean, monkeypatch):
        monkeypatch.setenv("EASYSNOWDATA_REGION", "local")
        monkeypatch.setattr(
            config, "_region_from_imds", lambda: pytest.fail("should not probe")
        )
        assert not config.region(probe=True, refresh=True).on_aws

    def test_imds_request_flow(self, clean, monkeypatch):
        import requests

        seen = []

        class Resp:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            requests,
            "put",
            lambda url, headers, timeout: seen.append(("put", url)) or Resp("tok"),
        )
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, headers, timeout: (
                seen.append(("get", headers["X-aws-ec2-metadata-token"]))
                or Resp("us-west-2\n")
            ),
        )
        assert config._region_from_imds() == "us-west-2"
        assert seen == [("put", config.IMDS_TOKEN_URL), ("get", "tok")]

        def boom(*a, **k):
            raise requests.ConnectionError("blocked")

        monkeypatch.setattr(requests, "put", boom)
        assert config._region_from_imds() is None
