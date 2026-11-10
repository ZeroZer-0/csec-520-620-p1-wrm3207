"""Smoke tests: the loader and metrics behave on the synthetic source."""
import numpy as np
from src.data import load_data
from src.evaluate import compute_metrics

CFG = {
    "seed": 0,
    "data": {"source": "synthetic", "n_samples": 400, "n_features": 10,
             "split": {"strategy": "random", "val_size": 0.2, "test_size": 0.2}},
}


def test_load_data_shapes():
    s = load_data(CFG)
    assert s.X_train.shape[1] == s.X_val.shape[1] == s.X_test.shape[1] == 10
    assert (len(s.y_train), len(s.y_val), len(s.y_test)) == (240, 80, 80)


def test_metrics_range():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y, p)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["accuracy"] == 1.0
