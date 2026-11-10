from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .integrity import verify_sha256


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray      
    idx_val: np.ndarray
    idx_test: np.ndarray
    feature_names: list
    info: dict = field(default_factory=dict)
    test_seen_in_train: np.ndarray | None = None   


def read_table(cfg: dict):
    d = cfg["data"]
    if d["source"] == "synthetic":
        X, y = make_classification(
            n_samples=d["n_samples"], n_features=d["n_features"],
            n_informative=max(2, d["n_features"] // 2), n_classes=2, random_state=cfg["seed"])
        return X, y.astype("int64"), [f"f{i}" for i in range(X.shape[1])]
    if d.get("sha256"):
        verify_sha256(d["csv_path"], d["sha256"])
    df = pd.read_csv(d["csv_path"])
    y = df[d["target"]].to_numpy().astype("int64")
    feats = df.drop(columns=[d["target"]])
    non_numeric = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns would be silently dropped: {non_numeric}")
    return feats.to_numpy(dtype="float64"), y, list(feats.columns)


def vector_groups(X: np.ndarray) -> np.ndarray:
    h = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    return np.unique(h, return_inverse=True)[1]


def _random_split(y, test_size, val_size, seed):
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    if val_size == 0:
        return trainval, np.empty(0, dtype=int), test
    train, val = train_test_split(trainval, test_size=val_size / (1 - test_size),
                                  random_state=seed, stratify=y[trainval])
    return train, val, test


def _grouped_split(y, groups, test_size, val_size, seed):
    n_groups = groups.max() + 1
    count = np.bincount(groups, minlength=n_groups)
    positives = np.bincount(groups, weights=y, minlength=n_groups)
    g_label = (2 * positives >= count).astype(int)
    g_idx = np.arange(n_groups)
    g_trainval, g_test = train_test_split(g_idx, test_size=test_size, random_state=seed, stratify=g_label)
    if val_size == 0:
        g_train, g_val = g_trainval, np.empty(0, dtype=int)
    else:
        g_train, g_val = train_test_split(g_trainval, test_size=val_size / (1 - test_size),
                                          random_state=seed, stratify=g_label[g_trainval])
    part = np.empty(n_groups, dtype=int)
    part[g_train], part[g_val], part[g_test] = 0, 1, 2
    row_part = part[groups]
    return tuple(np.flatnonzero(row_part == p) for p in (0, 1, 2))


def load_data(cfg: dict, exclude_from_train: np.ndarray | None = None) -> Splits:
    X, y, names = read_table(cfg)
    sp = cfg["data"]["split"]
    seed, test_size, val_size = sp.get("seed", cfg["seed"]), sp["test_size"], sp["val_size"]
    groups = vector_groups(X)
    if sp["strategy"] == "grouped":
        tr, va, te = _grouped_split(y, groups, test_size, val_size, seed)
    elif sp["strategy"] == "random":
        tr, va, te = _random_split(y, test_size, val_size, seed)
    else:
        raise ValueError(f"unknown split.strategy {sp['strategy']!r} (use 'grouped' or 'random')")
    if exclude_from_train is not None:
        tr = tr[~exclude_from_train[tr]]

    scaler = StandardScaler().fit(X[tr])

    def prep(rows):
        if len(rows) == 0:                      # no validation partition (legacy 80/20 mode)
            return np.empty((0, X.shape[1]), dtype="float32")
        return scaler.transform(X[rows]).astype("float32")

    seen = np.isin(groups[te], groups[tr])
    info = {
        "strategy": sp["strategy"], "seed": seed,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "malicious_fraction": {k: float(y[r].mean()) if len(r) else None
                               for k, r in (("train", tr), ("val", va), ("test", te))},
        "test_rows_with_vector_in_train": int(seen.sum()),
        "test_fraction_with_vector_in_train": float(seen.mean()),
    }
    return Splits(prep(tr), prep(va), prep(te), y[tr], y[va], y[te], tr, va, te, names, info, seen)
