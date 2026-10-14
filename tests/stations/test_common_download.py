"""Offline tests for ``_common.download_file`` (ranged, resumable, verified)."""

import hashlib

import pytest
import requests

from easysnowdata.stations.clients import _common
from easysnowdata.stations.clients._common import download_file


class _Err(Exception):
    pass


class _Resp:
    def __init__(self, body, status, fail_after=None):
        self.body, self.status_code, self.fail_after = body, status, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size=1):
        sent = 0
        for i in range(0, len(self.body), chunk_size):
            if self.fail_after is not None and sent >= self.fail_after:
                raise requests.exceptions.ChunkedEncodingError("connection reset")
            chunk = self.body[i : i + chunk_size]
            sent += len(chunk)
            yield chunk


class _RangeSession:
    """Serves ``body`` honouring ``Range``; the first request can be cut off."""

    def __init__(self, body, honour_range=True, cut_first_after=None):
        self.body = body
        self.honour_range = honour_range
        self.cut = cut_first_after
        self.ranges = []

    def get(self, url, headers=None, stream=False, timeout=None):
        rng = (headers or {}).get("Range")
        self.ranges.append(rng)
        fail, self.cut = self.cut, None
        if rng and self.honour_range:
            start, _, end = rng.removeprefix("bytes=").partition("-")
            stop = int(end) + 1 if end else len(self.body)
            return _Resp(self.body[int(start) : stop], 206, fail)
        return _Resp(self.body, 200, fail)


@pytest.fixture(autouse=True)
def _small_parallel_threshold(monkeypatch):
    monkeypatch.setattr(_common, "PARALLEL_DOWNLOAD_MIN_BYTES", 1000)
    monkeypatch.setattr(_common.time, "sleep", lambda s: None)


BODY = bytes(range(256)) * 40  # 10,240 bytes
MD5 = "md5:" + hashlib.md5(BODY).hexdigest()


def test_parallel_ranges_reassemble_and_verify(tmp_path):
    session = _RangeSession(BODY)
    dest = download_file(
        session, "u", tmp_path / "f.bin", error_cls=_Err, size=len(BODY), md5=MD5
    )
    assert dest.read_bytes() == BODY
    assert len(session.ranges) == 8 and all(
        r.startswith("bytes=") for r in session.ranges
    )
    assert not list(tmp_path.glob("f.bin.part*"))


def test_dropped_connection_resumes_from_partial_range(tmp_path):
    session = _RangeSession(BODY, cut_first_after=0)
    dest = download_file(
        session,
        "u",
        tmp_path / "f.bin",
        error_cls=_Err,
        size=len(BODY),
        md5=MD5,
        parts=1,
    )
    assert dest.read_bytes() == BODY


def test_existing_part_files_are_resumed_not_refetched(tmp_path):
    first = tmp_path / "f.bin.part00"
    first.write_bytes(BODY[:1280])  # range 0 of 8 already complete
    session = _RangeSession(BODY)
    download_file(
        session, "u", tmp_path / "f.bin", error_cls=_Err, size=len(BODY), md5=MD5
    )
    assert "bytes=0-1279" not in session.ranges
    assert (tmp_path / "f.bin").read_bytes() == BODY


def test_md5_mismatch_raises_and_leaves_nothing(tmp_path):
    with pytest.raises(_Err, match="md5"):
        download_file(
            _RangeSession(BODY),
            "u",
            tmp_path / "f.bin",
            error_cls=_Err,
            size=len(BODY),
            md5="md5:" + "0" * 32,
        )
    assert not (tmp_path / "f.bin").exists()


def test_server_ignoring_range_raises_for_parallel(tmp_path):
    with pytest.raises(_Err, match="Range"):
        download_file(
            _RangeSession(BODY, honour_range=False),
            "u",
            tmp_path / "f.bin",
            error_cls=_Err,
            size=len(BODY),
            max_retries=1,
        )


def test_small_file_single_stream(tmp_path):
    body = b"x" * 100
    session = _RangeSession(body)
    download_file(session, "u", tmp_path / "s.bin", error_cls=_Err, size=len(body))
    assert (tmp_path / "s.bin").read_bytes() == body
    assert session.ranges == [None]
