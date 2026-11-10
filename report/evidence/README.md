# Evidence files

Every number in the report comes from one of these files or from `results/`.
`report/make_report_assets.py` turns them into `report/numbers.tex` and
`report/tables.tex`, so the report cannot drift from the data. Run everything
from the repository root with the virtual environment active.

| File | Produced by | What it holds |
|---|---|---|
| `duplicate_audit.csv` | `python tools/audit_duplicates.py` | Repeated feature vectors overall and per source file and class, conflicting-label vectors, accuracy cap |
| `sweep_validation.csv` | `python tools/sweep.py` | 12-run grid (5 architectures x 2 learning rates, plus weighted loss), validation metrics only |
| `sweep_epochs.csv` | `python tools/sweep.py` | Per-epoch validation curve for each configuration |
| `seed_runs.csv` | `python tools/seed_runs.py` | Frozen config over seeds, grouped and original random split, with seen/unseen-in-train test subsets |
| `per_source.csv` | `python tools/per_source.py` | Correct rate and share of errors per source file and family (needs `results/test_predictions.npz`) |
| `holdout_family.csv` | `python tools/holdout_family.py` | Baseline plus leave-one-family-out runs |

The sweep never scores the test partition. The test partition is scored by
`make reproduce` and by the seed and hold-out tools, and none of those feed back
into any choice.

## Order

```bash
python tools/audit_duplicates.py
python tools/sweep.py                 # about an hour on a GPU
# freeze config.yaml from the validation winner, then:
make reproduce
python tools/seed_runs.py
python tools/per_source.py
python tools/holdout_family.py
python report/make_report_assets.py
```

`data/dataset_sources.csv` (written by `data/build_dataset.py`, git-ignored) is
needed by `per_source.py`, `holdout_family.py` and the duplicate audit's
per-source columns.
