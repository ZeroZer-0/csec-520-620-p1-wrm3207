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

Only the main unparted  CSV file was downloaded per category (not all 33+ attack
subtypes):

| File | Rows (raw) | Category |
|---|---|---|
| BenignTraffic.pcap.csv | 362,361 | Benign |
| DDoS-ICMP_Flood.pcap.csv | 268,010 | DDoS |
| DDoS-SYN_Flood.pcap.csv | 266,024 | DDoS |
| DDoS-TCP_Flood.pcap.csv | 265,776 | DDoS |
| DDoS-UDP_Flood.pcap.csv | 266,605 | DDoS |
| DictionaryBruteForce.pcap.csv | 13,064 | Brute Force |
| DNS_Spoofing.pcap.csv | 178,898 | Spoofing |
| Mirai-greeth_flood.pcap.csv | 33,597 | Mirai |
| Mirai-udpplain.pcap.csv | 36,481 | Mirai |
| MITM-ArpSpoofing.pcap.csv | 247,706 | Spoofing |
| Recon-OSScan.pcap.csv | 98,259 | Recon |
| Recon-PortScan.pcap.csv | 82,284 | Recon |
| SqlInjection.pcap.csv | 5,245 | Web-based |
| XSS.pcap.csv | 3,846 | Web-based |

## Assembly

`data/build_dataset.py` combines these files into a single binary
(benign vs. malicious) dataset:

1. All benign rows are kept and labeled `0`.
2. Each malicious file is capped at 150,000 rows (random sample, seed=42) to
   prevent the largest attack categories from being overly baised twords.
   Any files under the cap are used in full. This avoids the unbalanced malicous:benign ratio built into the dataset (40:1) while keeping a usable ratio.
3. Any row that contains infinite or NaN values (from ratio-based features such
   as flow rate on near-zero-duration flows) are dropped.
4. All malicious rows are labeled `1`, then concatenated with benign, and lastly shuffled
   (seed=42), before being written to `data/dataset.csv`.

Resulting dataset: 1,535,085 rows - 362,342 benign / 1,172,743 malicious (~1:3.2 ratio).

# Reproducing

1. Run `python data/download_dataset.py` downloads the pre-assembled `data/dataset.csv`
   from the google drive linked in that script. 
2. `config.yaml` is already set to `data.source: csv`,
   `data.csv_path: data/dataset.csv`, `data.target: Label`.
3. `make reproduce`.

Due to the download constraints mentioned above a proper scirpt that downloads the source files and creates the dataset could not be made. To rectafy this I have uploaded the end product to the mentioned google drive and included download_dataset.py to download that data directly. However if you wish to do it properly or simply want to verify this dataset.csv can also be created following the files used and assembly steps provided. 

Raw per-category files and `dataset.csv` are git-ignored and not committed.