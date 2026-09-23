# Submission Manifest

Fill this in before submitting. The grading agent reads it first.

## Team
- Willis Martin (RIT ID: wrm3207) - sole contributor: dataset
  assembly, model implementation, hyperparameter tuning, evaluation, report

## Course level
- [x] 520 (undergraduate)
- [ ] 620 (graduate)

## Problem
Binary classification of IoT network traffic as benign or malicious,
using a deep neural network trained on flow-level features, to detect
intrusions/attacks against IoT devices.

## Dataset
- Name / source / version / URL: CICIoT2023, Canadian Institute for
  Cybersecurity (CIC), University of New Brunswick.
  https://www.unb.ca/cic/datasets/iotdataset-2023.html
  (original per-category files downloaded via https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/browse.php?p=CSV%2FCSV;
  the assembled binary dataset is mirrored for automated download - see below)
- How to obtain it (command or steps): The pre-assembled `data/dataset.csv`
  is downloaded automatically as part of `make reproduce` (via
  `data/download_dataset.py`, which uses `gdown`). No manual steps are
  required. See `data/README.md` for full provenance details, including
  how `dataset.csv` was originally constructed from the 13 per-category
  CICIoT2023 CSV files.
- Target column: `Label` (binary: 0 = benign, 1 = malicious)

## How to reproduce
```bash
make setup
make reproduce
```
`make reproduce` automatically downloads `data/dataset.csv` if it is not
already present (via `make data` / `data/download_dataset.py`), then
trains and evaluates. No manual data download step is required.

## Claimed results
State the headline numbers you report in the paper (must match `results/metrics.json`):
- Accuracy: 0.8879
- Precision / Recall / F1: 0.9709 / 0.8796 / 0.9230
- ROC-AUC: 0.9555

## Report
See `report\CSEC_520___Cyber_Analytics___ML___Project_1___Willis_Martin.pdf`.
Source files used are aslo present in the report folder.
Overleaf: https://www.overleaf.com/read/sbhnrrtpxsyf#5ead6a

## AI-use acknowledgment
Used Claude (Anthropic) throughout this project for assisting with coding
questions and debugging (e.g., PyTorch/CUDA environment setup, sklearn
parameter errors, YAML config issues), formatting assistance for the
Overleaf/IEEE report, and building the `build_dataset.py` and
`download_dataset.py` scripts, including introducing and setting up
`gdown` for automated dataset download.