import numpy as np
import pandas as pd
import pytest
from conftest import make_cfg

from src.data import load_data, read_table, vector_groups
from src.integrity import sha256_file


def partitions(s):
    return s.idx_train, s.idx_val, s.idx_test


def test_grouped_split_never_shares_a_feature_vector(dup_csv):
    cfg = make_cfg(dup_csv, "grouped")
    s = load_data(cfg)
    g = vector_groups(read_table(cfg)[0])
    tr, va, te = (set(g[i]) for i in partitions(s))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert s.info["test_rows_with_vector_in_train"] == 0
    assert not s.test_seen_in_train.any()


def test_random_split_does_leak_repeats(dup_csv):
    """Same data, original protocol: repeated vectors do straddle train and test."""
    s = load_data(make_cfg(dup_csv, "random", val=0.0, test=0.2))
    assert s.info["test_fraction_with_vector_in_train"] > 0.5
    assert s.test_seen_in_train.sum() == s.info["test_rows_with_vector_in_train"]


def test_partitions_cover_every_row_once_with_sensible_sizes(dup_csv):
    s = load_data(make_cfg(dup_csv, "grouped"))
    rows = np.concatenate(partitions(s))
    assert sorted(rows) == list(range(3000))
    assert abs(len(s.idx_test) / 3000 - 0.1) < 0.03 and abs(len(s.idx_val) / 3000 - 0.1) < 0.03
    base = pd.read_csv(dup_csv)["Label"].mean()
    for part in ("train", "val", "test"):
        assert abs(s.info["malicious_fraction"][part] - base) < 0.08


def test_conflicting_vector_stays_in_one_partition(dup_csv):
    cfg = make_cfg(dup_csv, "grouped")
    s = load_data(cfg)
    X, y, _ = read_table(cfg)
    g = vector_groups(X)
    both = [k for k in np.unique(g) if 0 < y[g == k].mean() < 1]
    assert both, "fixture should contain vectors with both labels"
    where = np.empty(len(y), dtype=int)
    for p, idx in enumerate(partitions(s)):
        where[idx] = p
    assert all(len(set(where[g == k])) == 1 for k in both)


def test_scaler_uses_training_rows_only(dup_csv):
    s = load_data(make_cfg(dup_csv, "grouped"))
    assert np.abs(s.X_train.mean(axis=0)).max() < 1e-4
    assert np.abs(s.X_train.std(axis=0) - 1).max() < 1e-3
    assert np.abs(s.X_test.mean(axis=0)).max() > 1e-4       


def test_split_is_deterministic_and_seed_dependent(dup_csv):
    a, b = load_data(make_cfg(dup_csv, seed=3)), load_data(make_cfg(dup_csv, seed=3))
    c = load_data(make_cfg(dup_csv, seed=4))
    assert np.array_equal(a.idx_test, b.idx_test) and not np.array_equal(a.idx_test, c.idx_test)


def test_split_seed_can_differ_from_run_seed(dup_csv):
    cfg = make_cfg(dup_csv, seed=3)
    cfg["data"]["split"]["seed"] = 9
    assert np.array_equal(load_data(cfg).idx_test, load_data(make_cfg(dup_csv, seed=9)).idx_test)


def test_exclude_from_train_keeps_test_and_refits_scaler(dup_csv):
    cfg = make_cfg(dup_csv, "grouped")
    full = load_data(cfg)
    drop = np.zeros(3000, dtype=bool)
    drop[full.idx_train[: len(full.idx_train) // 2]] = True
    cut = load_data(cfg, exclude_from_train=drop)
    assert len(cut.idx_train) == len(full.idx_train) - drop[full.idx_train].sum()
    assert np.array_equal(cut.idx_test, full.idx_test)
    assert np.abs(cut.X_train.mean(axis=0)).max() < 1e-4


def test_unknown_strategy_and_non_numeric_columns_fail_loudly(dup_csv, tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        load_data(make_cfg(dup_csv, "stratified"))
    df = pd.read_csv(dup_csv)
    df["proto"] = "tcp"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="non-numeric"):
        load_data(make_cfg(bad))


def test_pinned_checksum_is_enforced(dup_csv):
    good = sha256_file(dup_csv)
    assert load_data(make_cfg(dup_csv, sha256=good)).info["n_train"] > 0
    with pytest.raises(ValueError, match="SHA-256"):
        load_data(make_cfg(dup_csv, sha256="0" * 64))
    with open(dup_csv, "ab") as f:                 
        f.write(b"1,1,1,1,1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_data(make_cfg(dup_csv, sha256=good))
