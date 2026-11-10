from __future__ import annotations
import argparse, json, os, platform, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .data import load_data
from .model import MLP
from .evaluate import (compute_metrics, detail_metrics, save_confusion_matrix, save_roc_curve)

EVAL_BATCH = 65536


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loss(cfg: dict, y_train, device):
    if not cfg["train"]["use_class_weights"]:
        return nn.CrossEntropyLoss()
    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device):
    model.eval()
    out = []
    for i in range(0, len(X), EVAL_BATCH):
        xb = torch.as_tensor(X[i:i + EVAL_BATCH], device=device)
        out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def mean_cross_entropy(model, X: np.ndarray, y: np.ndarray, device):
    model.eval()
    total = 0.0
    for i in range(0, len(X), EVAL_BATCH):
        xb = torch.as_tensor(X[i:i + EVAL_BATCH], device=device)
        yb = torch.as_tensor(y[i:i + EVAL_BATCH], device=device)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
    return total / len(X)


def train_model(cfg: dict, splits, device, log=print):
    t = cfg["train"]
    Xtr = torch.tensor(splits.X_train, device=device)
    ytr = torch.tensor(splits.y_train, device=device)
    model = MLP(Xtr.shape[1], tuple(cfg["model"]["hidden_sizes"]), dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = make_loss(cfg, splits.y_train, device)
    bs, history = t["batch_size"], []
    have_val = len(splits.y_val) > 0

    for epoch in range(t["epochs"]):
        model.train()
        perm = torch.randperm(Xtr.size(0))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, Xtr.size(0), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        row = {"epoch": epoch + 1, "train_loss": epoch_loss / n_batches}
        if have_val:
            prob = predict_proba(model, splits.X_val, device)
            row.update({"val_loss": mean_cross_entropy(model, splits.X_val, splits.y_val, device)})
            row.update({f"val_{k}": v for k, v in compute_metrics(splits.y_val, prob).items()})
        history.append(row)
        extra = f" - val F1 {row['val_f1']:.4f}, AUC {row['val_roc_auc']:.4f}" if have_val else ""
        log(f"Epoch {epoch + 1}/{t['epochs']} - avg loss: {row['train_loss']:.4f}{extra}")
    return model, history


def run(cfg: dict, out_dir: str | None = None, score_test: bool = True, log=print, splits=None) -> dict:
    set_seed(cfg["seed"])
    started = time.time()
    if splits is None:
        splits = load_data(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"split: {splits.info}")
    model, history = train_model(cfg, splits, device, log)

    res = {"splits": splits, "history": history, "model": model, "metrics": None, "details": None,
           "val_metrics": None, "prob_test": None}
    if len(splits.y_val):
        prob_val = predict_proba(model, splits.X_val, device)
        res["val_metrics"] = compute_metrics(splits.y_val, prob_val)
    if score_test:
        prob = predict_proba(model, splits.X_test, device)
        res["prob_test"] = prob
        res["metrics"] = compute_metrics(splits.y_test, prob)
        res["details"] = detail_metrics(splits.y_test, prob)
    res["seconds"] = time.time() - started
    res["run_info"] = {
        "seconds": round(res["seconds"], 1), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "seed": cfg["seed"],
    }
    if out_dir:
        write_results(cfg, res, out_dir, log)
    return res


def write_results(cfg: dict, res: dict, out: str, log=print):
    os.makedirs(out, exist_ok=True)
    s = res["splits"]
    metrics = dict(res["metrics"], details=res["details"], split=s.info)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if res["val_metrics"]:
        with open(os.path.join(out, "validation_metrics.json"), "w") as f:
            json.dump(res["val_metrics"], f, indent=2)
    with open(os.path.join(out, "training_log.json"), "w") as f:
        json.dump({"run_info": res["run_info"], "config": cfg, "history": res["history"]}, f, indent=2)
    np.savez_compressed(os.path.join(out, "test_predictions.npz"),
                        row=s.idx_test, y_true=s.y_test, prob=res["prob_test"])
    save_confusion_matrix(s.y_test, res["prob_test"], os.path.join(out, "confusion_matrix.png"))
    save_roc_curve(s.y_test, res["prob_test"], os.path.join(out, "roc_curve.png"))
    torch.save(res["model"].state_dict(), os.path.join(out, "model.pt"))
    log(f"Results written to {out} ({res['seconds']:.0f}s on {res['run_info']['gpu'] or 'cpu'})")
    log(json.dumps(metrics, indent=2))


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg, out_dir=cfg["output"]["dir"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
