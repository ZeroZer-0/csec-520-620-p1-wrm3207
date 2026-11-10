"""Small helpers shared by the scripts in tools/ (none of this is part of `make reproduce`)."""
import copy
import csv
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# P1_EVIDENCE_DIR redirects the output (used for smoke runs so they do not touch the committed evidence).
EVIDENCE = Path(os.environ.get("P1_EVIDENCE_DIR", ROOT / "report" / "evidence"))
SOURCES_CSV = ROOT / "data" / "dataset_sources.csv"


def load_config():
    """Read config.yaml, the same file `make reproduce` uses."""
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def config_variant(base, seed=None, strategy=None, **train_overrides):
    """Deep copy of the config with a different seed / split strategy / training settings."""
    cfg = copy.deepcopy(base)
    if seed is not None:
        cfg["seed"] = seed
    if strategy is not None:
        cfg["data"]["split"]["strategy"] = strategy
        if strategy == "random":      # the original protocol had no validation partition
            cfg["data"]["split"].update(val_size=0.0, test_size=0.2)
    cfg["train"].update(train_overrides)
    return cfg


def load_sources():
    """Row-aligned source file / attack family for data/dataset.csv (built by data/build_dataset.py)."""
    import pandas as pd
    if not SOURCES_CSV.exists():
        sys.exit(f"{SOURCES_CSV} not found. Run `python data/build_dataset.py` first (needs the raw per-attack CSVs).")
    return pd.read_csv(SOURCES_CSV)


def write_rows(path, rows):
    """Write a list of dicts as CSV. The header is every key in order of first appearance; missing cells are blank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print("Wrote", path)


def read_rows(path):
    """Rows of a CSV written by write_rows, as dicts of strings ([] if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
