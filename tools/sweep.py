import argparse

from _common import EVIDENCE, config_variant, load_config, read_rows, write_rows

ARCHS = ([128, 64], [256, 128, 64], [256, 128], [64, 64], [128, 64, 32])
LRS = (0.001, 0.003)
METRICS = ("val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_loss")


def grid():
    runs = [(h, lr, False) for h in ARCHS for lr in LRS]
    runs += [([128, 64], lr, True) for lr in LRS]
    return runs


def name_of(h, lr, weighted):
    return f"h{'-'.join(map(str, h))}_lr{lr}_{'weighted' if weighted else 'unweighted'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs from config.yaml")
    ap.add_argument("--only", default=None, help="run only grid entries whose name contains this text")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    ap.add_argument("--resume", action="store_true", help="skip runs already present in the output CSV")
    args = ap.parse_args()

    from src.data import load_data
    from src.train import run

    base = load_config()
    epochs = args.epochs or base["train"]["epochs"]
    summary_path = EVIDENCE / f"sweep_validation{args.suffix}.csv"
    epochs_path = EVIDENCE / f"sweep_epochs{args.suffix}.csv"
    summary = read_rows(summary_path) if args.resume else []
    per_epoch = read_rows(epochs_path) if args.resume else []
    done = {r["name"] for r in summary}

    splits = load_data(base)           
    print("split:", splits.info, flush=True)
    for h, lr, weighted in grid():
        name = name_of(h, lr, weighted)
        if (args.only and args.only not in name) or name in done:
            continue
        cfg = config_variant(base, epochs=epochs, lr=lr, use_class_weights=weighted)
        cfg["model"]["hidden_sizes"] = h
        print(f"=== {name} ({epochs} epochs)", flush=True)
        res = run(cfg, score_test=False, splits=splits, log=lambda m: print(m, flush=True))
        summary.append(dict(name=name, hidden_sizes=str(h), lr=lr, use_class_weights=weighted, epochs=epochs,
                            **{k: res["history"][-1][k] for k in METRICS}, seconds=round(res["seconds"], 1)))
        per_epoch += [dict(name=name, **{k: row[k] for k in ("epoch", "train_loss", *METRICS)})
                      for row in res["history"]]
        write_rows(summary_path, summary)
        write_rows(epochs_path, per_epoch)

    best = max(summary, key=lambda r: float(r["val_f1"]))
    print(f"\nBest by validation F1: {best['name']} (F1 {float(best['val_f1']):.4f}, "
          f"AUC {float(best['val_roc_auc']):.4f}, recall {float(best['val_recall']):.4f})")


if __name__ == "__main__":
    main()
