import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.integrity import read_pinned, verify_sha256  # noqa: E402

DATA_DIR = ROOT / "data"
FILE_ID = "1dU1-pbVHLUXKUhWT-Ep-ZBOn7O0AUASM"
OUTPUT_PATH = DATA_DIR / "dataset.csv"
PINNED = DATA_DIR / "dataset.sha256"


def main(force: bool = False):
    expected = read_pinned(PINNED)
    if OUTPUT_PATH.exists() and not force:
        verify_sha256(OUTPUT_PATH, expected)
        print(f"{OUTPUT_PATH} already present and matches the pinned SHA-256; skipping download.")
        return
    import gdown
    print(f"Downloading dataset.csv to {OUTPUT_PATH}...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", str(OUTPUT_PATH), quiet=False)
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError("download failed: no file was written")
    verify_sha256(OUTPUT_PATH, expected)
    print("Downloaded and verified. Run 'make reproduce' to train and evaluate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download even if a file exists")
    try:
        main(ap.parse_args().force)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
