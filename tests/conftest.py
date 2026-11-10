import numpy as np
import pandas as pd
import pytest


def make_cfg(csv_path, strategy="grouped", seed=0, sha256=None, val=0.1, test=0.1, **train):
    cfg = {
        "seed": seed,
        "data": {"source": "csv", "csv_path": str(csv_path), "target": "Label",
                 "split": {"strategy": strategy, "val_size": val, "test_size": test}},
        "model": {"hidden_sizes": [8, 4], "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 64, "lr": 0.01, "weight_decay": 0.0, "use_class_weights": False},
        "output": {"dir": "unused"},
    }
    if sha256:
        cfg["data"]["sha256"] = sha256
    cfg["train"].update(train)
    return cfg


@pytest.fixture
def dup_csv(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(600, 4))
    labels = (rng.random(600) < 0.75).astype(int)
    X = np.repeat(base, 5, axis=0)
    y = np.repeat(labels, 5)
    y[:20 * 5:5] = 1 - y[:20 * 5:5]
    df = pd.DataFrame(X, columns=list("abcd"))
    df["Label"] = y
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    path = tmp_path / "dup.csv"
    df.to_csv(path, index=False)
    return path
