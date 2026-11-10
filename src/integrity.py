from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path, expected: str) -> None:
    got = sha256_file(path)
    if got.lower() != expected.strip().lower():
        raise ValueError(
            f"{path} has SHA-256 {got}, expected {expected}. The file is not the pinned dataset "
            "(wrong version, partial download, or edited). Delete it and run `make data`, or rebuild it "
            "with `python data/build_dataset.py`."
        )


def read_pinned(sha_file) -> str:
    return Path(sha_file).read_text().split()[0]
