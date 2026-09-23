import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "dataset.csv"

SEED = 42 # Used to provide both randomness in selection while being reproducable
MAX_ROWS_PER_MALICIOUS_FILE = 150_000 # Caps large attack categories to keep the malicious:benign ratio low
BENIGN_FILE = "BenignTraffic.pcap.csv"

MALICIOUS_FILES = [
    "DDoS-ICMP_Flood.pcap.csv",
    "DDoS-SYN_Flood.pcap.csv",
    "DDoS-TCP_Flood.pcap.csv",
    "DDoS-UDP_Flood.pcap.csv",
    "DictionaryBruteForce.pcap.csv",
    "DNS_Spoofing.pcap.csv",
    "Mirai-greeth_flood.pcap.csv",
    "Mirai-udpplain.pcap.csv",
    "MITM-ArpSpoofing.pcap.csv",
    "Recon-OSScan.pcap.csv",
    "Recon-PortScan.pcap.csv",
    "SqlInjection.pcap.csv",
    "XSS.pcap.csv",
]


def load_capped(path: Path, cap: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=seed)
    return df


def main():
    print(f"Loading benign: {BENIGN_FILE}")
    benign_df = pd.read_csv(DATA_DIR / BENIGN_FILE)
    benign_df["Label"] = 0
    print(f"  {len(benign_df)} rows")

    malicious_parts = []
    for fname in MALICIOUS_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        df = load_capped(path, MAX_ROWS_PER_MALICIOUS_FILE, SEED)
        df["Label"] = 1
        print(f"  {fname}: using {len(df)} rows")
        malicious_parts.append(df)

    malicious_df = pd.concat(malicious_parts, ignore_index=True)
    print(f"Total malicious rows: {len(malicious_df)}")

    full_df = pd.concat([benign_df, malicious_df], ignore_index=True)

    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_cols] = full_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    before = len(full_df)
    full_df = full_df.dropna(subset=numeric_cols)
    after = len(full_df)
    print(f"Dropped {before - after} rows containing inf/NaN values")

    full_df = full_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Final dataset: {len(full_df)} rows "
          f"({(full_df['Label']==0).sum()} benign, {(full_df['Label']==1).sum()} malicious)")

    full_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()