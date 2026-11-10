from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: works in CI and on the cluster
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

CLASS_NAMES = ("benign", "malicious")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_prob = P(class 1). The five metrics the course asks for."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def detail_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    per_class = {}
    for label, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=label, zero_division=0)),
            "support": int((np.asarray(y_true) == label).sum()),
        }
    macro = {k: float(np.mean([per_class[c][k] for c in CLASS_NAMES])) for k in ("precision", "recall", "f1")}
    support = [per_class[c]["support"] for c in CLASS_NAMES]
    return {
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": per_class,
        # Unweighted mean over the two classes. A paper that does not state its averaging may be using this.
        "macro_average": macro,
        "balanced_accuracy": macro["recall"],
        "majority_class_accuracy": max(support) / sum(support),   # accuracy of always predicting one class
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def save_confusion_matrix(y_true, y_prob, path, threshold: float = 0.5):
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_roc_curve(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
