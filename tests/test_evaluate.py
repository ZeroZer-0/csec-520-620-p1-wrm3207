import numpy as np

from src.evaluate import compute_metrics, detail_metrics, save_confusion_matrix, save_roc_curve

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])    # one false positive (0.6), one miss (0.4)


def test_metrics_values_and_positive_class():
    m = compute_metrics(Y, P)
    assert m["accuracy"] == 6 / 8
    assert m["precision"] == 3 / 4 and m["recall"] == 3 / 4          # malicious is the positive class
    assert m["f1"] == 0.75


def test_threshold_moves_the_operating_point_not_auc():
    lo, hi = compute_metrics(Y, P, 0.35), compute_metrics(Y, P, 0.65)
    assert lo["recall"] > hi["recall"] and lo["precision"] < hi["precision"]
    assert lo["roc_auc"] == hi["roc_auc"]


def test_detail_metrics_confusion_and_per_class():
    d = detail_metrics(Y, P)
    assert d["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert d["per_class"]["benign"]["recall"] == 0.75           # specificity
    assert d["per_class"]["malicious"]["support"] == 4
    assert 0 <= d["average_precision"] <= 1


def test_macro_average_and_majority_baseline():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])                # 75% benign
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.9, 0.2])
    d = detail_metrics(y, p)
    assert d["majority_class_accuracy"] == 0.75
    assert d["macro_average"]["recall"] == (5 / 6 + 1 / 2) / 2 == d["balanced_accuracy"]
    assert d["macro_average"]["precision"] == (5 / 6 + 1 / 2) / 2        # precision: benign 5/6, malicious 1/2


def test_figures_are_written(tmp_path):
    save_confusion_matrix(Y, P, tmp_path / "cm.png")
    save_roc_curve(Y, P, tmp_path / "roc.png")
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "roc.png").stat().st_size > 0
