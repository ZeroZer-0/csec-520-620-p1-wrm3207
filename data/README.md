# Data

**Do not commit datasets to Git.** They are large, may be license-restricted, and bloat history.

Instead:

1. Document exactly where the data comes from (name, version, URL, date accessed).
2. Provide a script or one-liner to download it into this folder.
3. Reference the file(s) via `config.yaml` (`data.source: csv`, `data.csv_path: data/your_file.csv`).

# Dataset: CICIoT2023 (binary subset)

## Source

Original dataset: **CICIoT2023**, Canadian Institute for Cybersecurity (CIC),
University of New Brunswick.
Paper: E. C. P. Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark
for Large-Scale Attacks in IoT Environment," *Sensors*, vol. 23, no. 13,
p. 5941, 2023. https://doi.org/10.3390/s23135941

Files were originally downloaded from the CIC file server (CSV/CSV folder,
per-attack-category files):
https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

Note: accessing the CIC file server requires submitting a short access
request form (name/email); the form did not appear to validate the
submitted information. The server also blocks scripted
download requests. Becuase of that a fully automated download script for
the original source can't be reailably done. Becuase of that, the final
assembled dataset (`dataset.csv`) has been made available separately see
"Reproducing" below.

## Files used

The benign file and 13 attack files were downloaded, the main unparted CSV for each.
That covers six of the dataset's seven attack categories. **No DoS file was
downloaded**, so the DoS category and 20 of the 33 attack types in Table 3 of the
paper are absent. Several files are also only part of the paper's row counts
(for example ICMP flood 268,010 vs 7,200,504 and benign 362,361 vs 1,098,195);
brute force, OS scan, port scan, SQL injection and XSS match the paper.

| File | Rows (raw) | Category |
|---|---|---|
| BenignTraffic.pcap.csv | 362,361 | Benign |
| DDoS-ICMP_Flood.pcap.csv | 268,010 | DDoS |
| DDoS-SYN_Flood.pcap.csv | 266,024 | DDoS |
| DDoS-TCP_Flood.pcap.csv | 265,776 | DDoS |
| DDoS-UDP_Flood.pcap.csv | 266,605 | DDoS |
| DictionaryBruteForce.pcap.csv | 13,064 | Brute force |
| DNS_Spoofing.pcap.csv | 178,898 | Spoofing |
| Mirai-greeth_flood.pcap.csv | 33,597 | Mirai |
| Mirai-udpplain.pcap.csv | 36,481 | Mirai |
| MITM-ArpSpoofing.pcap.csv | 247,706 | Spoofing |
| Recon-OSScan.pcap.csv | 98,259 | Recon |
| Recon-PortScan.pcap.csv | 82,284 | Recon |
| SqlInjection.pcap.csv | 5,245 | Web-based |
| XSS.pcap.csv | 3,846 | Web-based |

The table has 39 numeric features. Against the 46 non-timestamp features in Table 4
of the paper, flow_duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance
and Weight are missing and an IGMP indicator is present. I did not investigate why.

## Assembly

`python data/build_dataset.py [--data-dir DIR] [--out-dir DIR]` combines the files
into a binary (benign vs malicious) dataset. It raises an error if any declared
input file is missing and exits non-zero if the result does not match the pinned
SHA-256.

1. All benign rows are kept and labelled `0`.
2. Each malicious file is capped at 150,000 randomly chosen rows (seed 42). The
   cap is per file, not per category. Only six files exceed it (four DDoS floods,
   DNS spoofing, ARP spoofing); the rest are used in full.
3. Rows with infinite or missing values (52) are dropped.
4. Malicious rows are labelled `1`, concatenated with benign, shuffled (seed 42)
   and written with CRLF line endings.
5. A sidecar `data/dataset_sources.csv` records the source file and family of each
   row (git-ignored). It is used for per-source analysis only, never as a feature.

Result: 1,535,085 rows, 362,342 benign and 1,172,743 malicious (about 1:3.2).
Repeated feature vectors are common (see `report/evidence/duplicate_audit.csv`),
which is why the project splits by feature vector.

## Pinned identity

`data/dataset.sha256` holds the SHA-256 of the expected `dataset.csv`:
`4b24f7311f790014f83ecdd80a1b7312e87275d3b181e3f8c483796dfc2d7d39`.
`config.yaml` (`data.sha256`) makes `src/data.py` check it on every load, and the
downloader checks it before and after downloading. An existing file with another
hash is an error (`--force` re-downloads).

# Reproducing

1. `python data/download_dataset.py` downloads the pre-assembled `data/dataset.csv`
   from the Google Drive link in that script and verifies the hash.
2. `config.yaml` has `data.source: csv`, `data.csv_path: data/dataset.csv`, `data.target: Label`.
3. `make reproduce`.

The CIC server needs an access form and blocks scripted downloads, so the source
files cannot be fetched automatically. Anyone with the 14 raw files can run
`data/build_dataset.py` and will get a byte-identical `dataset.csv`.

Raw per-category files and `dataset.csv` are git-ignored and not committed.
