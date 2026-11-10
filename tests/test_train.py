"""End-to-end behaviour of src/train.py on a tiny CSV (CPU, a couple of epochs)."""
import json

import numpy as np
import pytest
from conftest import make_cfg

torch = pytest.importorskip("torch")
from grading import grade  # noqa: E402
from src import train  # noqa: E402


def test_run_writes_results_and_a_valid_metrics_file(dup_csv, tmp_path):
    out = tmp_path / "out"
    res = train.run(make_cfg(dup_csv), out_dir=str(out), log=lambda m: None)
    for name in ("metrics.json", "validation_metrics.json", "training_log.json", "test_predictions.npz",
                 "confusion_matrix.png", "roc_curve.png", "model.pt"):
        assert (out / name).exists(), name
    m = json.loads((out / "metrics.json").read_text())
    five = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert five <= set(m)
    top_level_numbers = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    assert set(top_level_numbers) == five and all(0 <= v <= 1 for v in top_level_numbers.values())
    assert m["details"]["confusion"]["tp"] + m["details"]["confusion"]["fn"] == m["details"]["per_class"]["malicious"]["support"]
    assert m["split"]["strategy"] == "grouped" and m["split"]["test_rows_with_vector_in_train"] == 0
    log = json.loads((out / "training_log.json").read_text())
    assert [r["epoch"] for r in log["history"]] == [1, 2] and "val_f1" in log["history"][0]
    assert log["run_info"]["seed"] == 0 and res["val_metrics"]["f1"] == log["history"][-1]["val_f1"]


def test_grading_harness_accepts_the_metrics_file(dup_csv, tmp_path, monkeypatch):
    train.run(make_cfg(dup_csv), out_dir=str(tmp_path / "results"), log=lambda m: None)
    monkeypatch.setattr(grade, "ROOT", tmp_path)
    assert grade.check_metrics()["status"] == "pass"


def test_saved_predictions_match_the_test_rows(dup_csv, tmp_path):
    out = tmp_path / "out"
    res = train.run(make_cfg(dup_csv), out_dir=str(out), log=lambda m: None)
    p = np.load(out / "test_predictions.npz")
    assert np.array_equal(p["row"], res["splits"].idx_test) and np.allclose(p["prob"], res["prob_test"])


def test_test_partition_is_only_scored_when_asked(dup_csv, monkeypatch):
    seen = []
    real = train.predict_proba
    monkeypatch.setattr(train, "predict_proba", lambda model, X, device: seen.append(id(X)) or real(model, X, device))
    res = train.run(make_cfg(dup_csv), score_test=False, log=lambda m: None)
    assert res["metrics"] is None and res["prob_test"] is None
    assert id(res["splits"].X_test) not in seen and id(res["splits"].X_val) in seen
    seen.clear()
    res = train.run(make_cfg(dup_csv), score_test=True, log=lambda m: None)
    assert id(res["splits"].X_test) in seen and res["metrics"] is not None


def test_legacy_random_mode_has_no_validation_partition(dup_csv):
    res = train.run(make_cfg(dup_csv, "random", val=0.0, test=0.2), log=lambda m: None)
    assert res["val_metrics"] is None and "val_f1" not in res["history"][0] and res["metrics"] is not None


def test_class_weights_follow_inverse_frequency_and_config_is_required():
    y = np.array([0] * 25 + [1] * 75)
    loss = train.make_loss({"train": {"use_class_weights": True}}, y, "cpu")
    assert np.allclose(loss.weight.numpy(), [2.0, 100 / 150])
    assert train.make_loss({"train": {"use_class_weights": False}}, y, "cpu").weight is None
    with pytest.raises(KeyError):                         # no silent default hidden in the code
        train.make_loss({"train": {}}, y, "cpu")


def test_same_seed_same_result_on_cpu(dup_csv):
    a = train.run(make_cfg(dup_csv, seed=5), log=lambda m: None)["metrics"]
    b = train.run(make_cfg(dup_csv, seed=5), log=lambda m: None)["metrics"]
    assert a == pytest.approx(b, abs=1e-6)
