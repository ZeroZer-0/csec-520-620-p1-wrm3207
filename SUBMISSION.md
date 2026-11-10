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
  `data/download_dataset.py`, which uses `gdown`) and checked against the
  pinned SHA-256 in `data/dataset.sha256`; a file with a different hash is
  rejected, both by the downloader and by `src/data.py`. See
  `data/README.md` for provenance, including how `dataset.csv` is built
  from the benign file and 13 attack files (six of the dataset's seven
  attack categories, no DoS) with `data/build_dataset.py`.
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
Numbers on the grouped test partition (identical feature vectors are kept
in one partition, 80/10/10 train/val/test, hyperparameters chosen on
validation only). They are generated from `results/metrics.json`; do not
edit by hand.

<!-- claimed-results:start -->
- Accuracy: 0.8917
- Precision / Recall / F1: 0.9499 / 0.9061 / 0.9275
- ROC-AUC: 0.9560
- Split: grouped (identical feature vectors never straddle partitions), 1,227,810 train / 153,528 validation / 153,747 test rows, seed 42
<!-- claimed-results:end -->

## Revision note (after grader feedback on the first submission)
The first submission scored 83/100. Changes in this version:
- Evaluation protocol: grouped train/val/test split, scaler fit on train only,
  model selection on the validation partition only (`tools/sweep.py`), test
  scored once by `make reproduce`. The original random 80/20 protocol is kept
  as a labelled comparison, not as the headline.
- Duplicates: `tools/audit_duplicates.py` measures repeated feature vectors by
  source file and the accuracy cap from conflicting labels.
- Paper comparison: exact Table 6 values for the two-class DNN (accuracy
  0.9944, precision 0.9476, recall 0.9333, F1 0.9403), a note that the
  averaging is not stated, and the difference in class priors.
- Coverage claims corrected: six of seven attack categories, no DoS files,
  13 of 33 attack types, per-file (not per-category) cap, partial files.
- `data/build_dataset.py` fails on any missing input and reproduces the pinned
  SHA-256; `data/download_dataset.py` verifies the dataset identity.
- Tests: 40 tests covering splitting, scaling, duplicates, metrics, the
  builder, the downloader and the report tools.
- Extra evidence (`report/evidence/`): seed repeats, per-source errors,
  leave-one-family-out runs, validation sweep. Commands are listed in
  `report/evidence/README.md`.

## Report
See `report\CSEC_520___Cyber_Analytics___ML___Project_1___Willis_Martin.pdf`.
Source files used are also present in the report folder.
Overleaf: https://www.overleaf.com/read/sbhnrrtpxsyf#5ead6a

## AI-use acknowledgment
An AI assistant (Claude) helped write scripts, provided coding suggestions, assisted with Overleaf formatting, find relevant papers. Grammarly was also used to correct spelling and grammar in the report.
