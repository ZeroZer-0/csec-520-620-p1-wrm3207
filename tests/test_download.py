import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("download_dataset", ROOT / "data" / "download_dataset.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

GOOD = b"a,b,Label\r\n1,2,0\r\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    pinned = tmp_path / "dataset.sha256"
    pinned.write_text(hashlib.sha256(GOOD).hexdigest() + "  dataset.csv\n")
    monkeypatch.setattr(dl, "PINNED", pinned)
    monkeypatch.setattr(dl, "OUTPUT_PATH", tmp_path / "dataset.csv")
    calls = []

    def fake_gdown(payload):
        mod = types.ModuleType("gdown")
        mod.download = lambda url, out, quiet=False: calls.append(url) or Path(out).write_bytes(payload)
        monkeypatch.setitem(sys.modules, "gdown", mod)

    return tmp_path / "dataset.csv", calls, fake_gdown


def test_valid_existing_file_is_kept_and_nothing_is_downloaded(env):
    path, calls, fake = env
    path.write_bytes(GOOD)
    fake(b"should never be used")
    dl.main()
    assert calls == [] and path.read_bytes() == GOOD


def test_wrong_existing_file_is_rejected_not_accepted(env):
    path, calls, fake = env
    path.write_bytes(GOOD + b"extra")
    fake(GOOD)
    with pytest.raises(ValueError, match="SHA-256"):
        dl.main()
    assert calls == []


def test_missing_file_is_downloaded_and_verified(env):
    path, calls, fake = env
    fake(GOOD)
    dl.main()
    assert len(calls) == 1 and path.read_bytes() == GOOD


def test_corrupt_download_fails_verification(env):
    path, calls, fake = env
    fake(GOOD[:-3])                                  # truncated
    with pytest.raises(ValueError, match="SHA-256"):
        dl.main()


def test_force_replaces_an_existing_file(env):
    path, calls, fake = env
    path.write_bytes(b"old")
    fake(GOOD)
    dl.main(force=True)
    assert path.read_bytes() == GOOD and len(calls) == 1
