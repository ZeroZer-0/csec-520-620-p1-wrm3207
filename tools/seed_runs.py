import argparse

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from _common import EVIDENCE, config_variant, load_config, read_rows, write_rows

TEST = ("accuracy", "precision", "recall", "f1", "roc_auc")


def subset(y, prob, mask):
    y, prob = y[mask], prob[mask]
    pred = (prob >= 0.5).astype(int)
    out = dict(n=int(len(y)), malicious_share=float(y.mean()) if len(y) else None,
               accuracy=None, precision=None, recall=None, f1=None, roc_auc=None)
    if len(y):
        out.update(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                   recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))
        if len(set(y)) == 2:
            out["roc_auc"] = roc_auc_score(y, prob)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--strategies", default="grouped,random")
    ap.add_argument("--vary", default="both", help="both and/or init, comma separated")
    ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs from config.yaml")
    args = ap.parse_args()

    from src.data import load_data
    from src.train import run

    base = load_config()
    out = EVIDENCE / "seed_runs.csv"
    rows = read_rows(out)       # earlier invocations are kept; a re-run of the same (vary, strategy, seed) replaces its row
    for vary in args.vary.split(","):
        for strategy in args.strategies.split(","):
            shared = None
            for seed in [int(s) for s in args.seeds.split(",")]:
                cfg = config_variant(base, seed=seed, strategy=strategy, **({"epochs": args.epochs} if args.epochs else {}))
                if vary == "init":
                    cfg["data"]["split"]["seed"] = base["seed"]       # keep the split fixed
                    shared = shared or load_data(cfg)
                print(f"=== vary={vary} strategy={strategy} seed={seed}", flush=True)
                res = run(cfg, score_test=True, splits=shared, log=lambda m: print(m, flush=True))
                s = res["splits"]
                row = dict(vary=vary, strategy=strategy, seed=seed, n_train=s.info["n_train"],
                           n_test=s.info["n_test"],
                           test_fraction_with_vector_in_train=s.info["test_fraction_with_vector_in_train"],
                           **{f"test_{k}": res["metrics"][k] for k in TEST},
                           **{f"val_{k}": (res["val_metrics"] or {}).get(k) for k in TEST},
                           seconds=round(res["seconds"], 1))
                for part, mask in (("seen", s.test_seen_in_train), ("unseen", ~s.test_seen_in_train)):
                    for k, v in subset(s.y_test, res["prob_test"], mask).items():
                        row[f"{part}_{k}"] = v
                key = lambda r: (str(r["vary"]), str(r["strategy"]), str(r["seed"]))
                rows = [r for r in rows if key(r) != key(row)] + [row]
                write_rows(out, rows)
    for vary, strategy in sorted({(r["vary"], r["strategy"]) for r in rows}):
        sel = [r for r in rows if r["vary"] == vary and r["strategy"] == strategy]
        vals = {k: np.array([float(r[f"test_{k}"]) for r in sel]) for k in TEST}
        stats = {k: (v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan")) for k, v in vals.items()}
        print(vary, strategy, " ".join(f"{k} {m:.4f}±{sd:.4f}" for k, (m, sd) in stats.items()))


if __name__ == "__main__":
    main()
