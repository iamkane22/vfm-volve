from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vfm.data.source import IntegrityError, fetch


@pytest.fixture
def remote(tmp_path: Path) -> tuple[str, str]:
    payload = tmp_path / "remote.bin"
    payload.write_bytes(b"volve" * 1000)
    return payload.as_uri(), hashlib.sha256(payload.read_bytes()).hexdigest()


def test_downloads_and_verifies(tmp_path: Path, remote: tuple[str, str]) -> None:
    url, digest = remote
    dest = fetch(tmp_path / "out" / "file.bin", url=url, sha256=digest)
    assert dest.read_bytes() == b"volve" * 1000


def test_rejects_checksum_mismatch_and_leaves_nothing(
    tmp_path: Path, remote: tuple[str, str]
) -> None:
    url, _ = remote
    dest = tmp_path / "out" / "file.bin"
    with pytest.raises(IntegrityError):
        fetch(dest, url=url, sha256="0" * 64)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


def test_reuses_valid_cached_file(tmp_path: Path, remote: tuple[str, str]) -> None:
    url, digest = remote
    dest = fetch(tmp_path / "file.bin", url=url, sha256=digest)
    mtime = dest.stat().st_mtime_ns
    fetch(dest, url="file:///does/not/exist", sha256=digest)
    assert dest.stat().st_mtime_ns == mtime
