import numpy as np
import pandas as pd
import pytest

import _common
import holdout_family
import per_source
import seed_runs
import sweep


def test_write_rows_unions_columns(tmp_path):
    _common.write_rows(tmp_path / "x.csv", [{"a": 1}, {"a": 2, "b": 3}])
    df = pd.read_csv(tmp_path / "x.csv")
    assert list(df.columns) == ["a", "b"] and df["b"].isna().tolist() == [True, False]


def test_config_variant_random_means_original_protocol_and_leaves_base_alone():
    base = _common.load_config()
    legacy = _common.config_variant(base, seed=3, strategy="random", epochs=1)
    assert legacy["data"]["split"] == {"strategy": "random", "val_size": 0.0, "test_size": 0.2}
    assert legacy["seed"] == 3 and legacy["train"]["epochs"] == 1
    assert base["data"]["split"]["strategy"] == "grouped" and base["seed"] == 42


def test_subset_metrics_handles_empty_and_single_class_subsets():
    y, p = np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.4])
    assert seed_runs.subset(y, p, np.zeros(4, dtype=bool))["accuracy"] is None
    only_malicious = seed_runs.subset(y, p, np.array([True, True, False, False]))
    assert only_malicious["recall"] == 0.5 and only_malicious["roc_auc"] is None
    both = seed_runs.subset(y, p, np.ones(4, dtype=bool))
    assert both["roc_auc"] == 0.75 and both["n"] == 4


def test_sweep_grid_is_the_declared_twelve_runs():
    runs = sweep.grid()
    names = [sweep.name_of(*r) for r in runs]
    assert len(runs) == 12 and len(set(names)) == 12
    assert sorted(n for n in names if "_weighted" in n) == ["h128-64_lr0.001_weighted", "h128-64_lr0.003_weighted"]


def test_holdout_rates_per_family():
    class S:
        idx_test = np.array([0, 1, 2, 3])
        y_test = np.array([0, 1, 1, 1])
    families = np.array(["Benign", "DDoS", "DDoS", "Recon"])
    res = {"splits": S, "prob_test": np.array([0.1, 0.9, 0.2, 0.8])}
    r = holdout_family.rates(res, families)
    assert r["DDoS"] == (0.5, 2) and r["Recon"] == (1.0, 1) and r["Benign"] == (1.0, 1)


def test_per_source_attributes_errors(tmp_path, monkeypatch):
    sources = pd.DataFrame({"source": ["b.csv", "a.csv", "a.csv", "c.csv"],
                            "family": ["Benign", "FamA", "FamA", "FamC"]})
    np.savez(tmp_path / "p.npz", row=np.array([0, 1, 2, 3]), y_true=np.array([0, 1, 1, 1]),
             prob=np.array([0.1, 0.9, 0.2, 0.1]))
    monkeypatch.setattr(per_source, "load_sources", lambda: sources)
    monkeypatch.setattr(per_source, "EVIDENCE", tmp_path)
    monkeypatch.setattr("sys.argv", ["per_source.py", "--predictions", str(tmp_path / "p.npz")])
    per_source.main()
    out = pd.read_csv(tmp_path / "per_source.csv").set_index(["scope", "name"])
    assert out.loc[("source", "a.csv"), "correct_rate"] == 0.5
    assert out.loc[("source", "c.csv"), "share_of_all_errors"] == 0.5
    assert out.loc[("family", "Benign"), "errors"] == 0
