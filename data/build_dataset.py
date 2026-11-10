import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.integrity import read_pinned, sha256_file  # noqa: E402

SEED = 42  
MAX_ROWS_PER_MALICIOUS_FILE = 150_000  
BENIGN_FILE = "BenignTraffic.pcap.csv"
MALICIOUS_FILES = {
    "DDoS-ICMP_Flood.pcap.csv": "DDoS",
    "DDoS-SYN_Flood.pcap.csv": "DDoS",
    "DDoS-TCP_Flood.pcap.csv": "DDoS",
    "DDoS-UDP_Flood.pcap.csv": "DDoS",
    "DictionaryBruteForce.pcap.csv": "Brute Force",
    "DNS_Spoofing.pcap.csv": "Spoofing",
    "Mirai-greeth_flood.pcap.csv": "Mirai",
    "Mirai-udpplain.pcap.csv": "Mirai",
    "MITM-ArpSpoofing.pcap.csv": "Spoofing",
    "Recon-OSScan.pcap.csv": "Recon",
    "Recon-PortScan.pcap.csv": "Recon",
    "SqlInjection.pcap.csv": "Web-based",
    "XSS.pcap.csv": "Web-based",
}
LINE_ENDING = "\r\n"


def load_capped(path: Path, cap: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=seed)
    return df


def assemble(data_dir: Path) -> pd.DataFrame:
    """The shuffled table with Label, source and family columns. Raises if an input file is missing."""
    missing = [f for f in [BENIGN_FILE, *MALICIOUS_FILES] if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing input file(s) in {data_dir}: {missing}")

    print(f"Loading benign: {BENIGN_FILE}")
    benign = pd.read_csv(data_dir / BENIGN_FILE)
    benign["Label"], benign["source"], benign["family"] = 0, BENIGN_FILE, "Benign"
    print(f"  {len(benign)} rows")

    parts = []
    for fname, family in MALICIOUS_FILES.items():
        df = load_capped(data_dir / fname, MAX_ROWS_PER_MALICIOUS_FILE, SEED)
        df["Label"], df["source"], df["family"] = 1, fname, family
        print(f"  {fname}: using {len(df)} rows")
        parts.append(df)

    full = pd.concat([benign, *parts], ignore_index=True)
    print(f"Total malicious rows: {sum(len(p) for p in parts)}")

    numeric_cols = full.select_dtypes(include=[np.number]).columns
    full[numeric_cols] = full[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(full)
    full = full.dropna(subset=numeric_cols)
    print(f"Dropped {before - len(full)} rows containing inf/NaN values")
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Final dataset: {len(full)} rows "
          f"({(full['Label'] == 0).sum()} benign, {(full['Label'] == 1).sum()} malicious)")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    full = assemble(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.out_dir / "dataset.csv"
    full.drop(columns=["source", "family"]).to_csv(dataset, index=False, lineterminator=LINE_ENDING)
    full[["source", "family"]].to_csv(args.out_dir / "dataset_sources.csv", index=False, lineterminator="\n")
    print(f"Saved {dataset} and dataset_sources.csv")

    got, pinned = sha256_file(dataset), read_pinned(HERE / "dataset.sha256")
    if got != pinned:
        sys.exit(f"SHA-256 {got} does not match the pinned {pinned}; this is not the reported dataset.")
    print("SHA-256 matches data/dataset.sha256")


if __name__ == "__main__":
    main()
