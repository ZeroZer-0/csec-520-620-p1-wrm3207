from pathlib import Path
import gdown

DATA_DIR = Path(__file__).parent
FILE_ID = "1dU1-pbVHLUXKUhWT-Ep-ZBOn7O0AUASM"
OUTPUT_PATH = DATA_DIR / "dataset.csv"


def main():
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists, skipping download.")
        return
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading dataset.csv to {OUTPUT_PATH}...")
    gdown.download(url, str(OUTPUT_PATH), quiet=False)
    print("Done. Run 'make reproduce' to train and evaluate.")


if __name__ == "__main__":
    main()