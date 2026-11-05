"""Download of the Volve daily production workbook with integrity check.

The workbook is part of the Volve data set released by Equinor under the Equinor Open
Data Licence. The official distribution sits behind an interactive download page, so we
fetch a public copy of the workbook and pin its SHA-256: every run analyses exactly the
same bytes, and a changed upstream file fails loudly instead of silently shifting results.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

VOLVE_XLSX_URL = (
    "https://raw.githubusercontent.com/ashrafalaghbari/GA-TCN-LSTM/main/datasets/raw_data.xlsx"
)
VOLVE_XLSX_SHA256 = "266124c6653573ff7b4358c7f026e200a0b3973c7187d2b08f4ef64e41f775f7"

_CHUNK = 1 << 16


class IntegrityError(RuntimeError):
    """Raised when a downloaded file does not match its pinned checksum."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    dest: Path,
    url: str = VOLVE_XLSX_URL,
    sha256: str = VOLVE_XLSX_SHA256,
    *,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` atomically and verify its checksum.

    An existing file with the right checksum is reused unless ``force`` is set.
    """
    if dest.exists() and not force:
        if sha256_of(dest) == sha256:
            logger.info("Using cached %s", dest)
            return dest
        logger.warning("Checksum mismatch for cached %s, downloading again", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, tmp, _CHUNK)

    actual = sha256_of(tmp_path)
    if actual != sha256:
        tmp_path.unlink(missing_ok=True)
        raise IntegrityError(f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}")

    tmp_path.replace(dest)
    logger.info("Downloaded %s (%d bytes)", dest, dest.stat().st_size)
    return dest
