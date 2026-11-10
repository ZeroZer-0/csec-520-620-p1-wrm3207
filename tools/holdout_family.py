import argparse

import numpy as np

from _common import EVIDENCE, config_variant, load_config, load_sources, write_rows


def rates(res, families):
    """Per-family correct rate on the test rows (malicious: detection rate; benign: specificity)."""
    s = res["splits"]
    fam = families[s.idx_test]
    ok = (res["prob_test"] >= 0.5).astype(int) == s.y_test
    return {f: (float(ok[fam == f].mean()), int((fam == f).sum())) for f in np.unique(fam)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="hold out just this family")
    ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs from config.yaml")
    args = ap.parse_args()

    from src.data import load_data
    from src.train import run

    base = load_config()
    if args.epochs:
        base = config_variant(base, epochs=args.epochs)
    families = load_sources()["family"].to_numpy()
    log = lambda m: print(m, flush=True)

    print("=== baseline (nothing held out)", flush=True)
    baseline = rates(run(base, splits=load_data(base), log=log), families)
    rows = []
    for fam in sorted(set(families) - {"Benign"}):
        if args.only and fam != args.only:
            continue
        print(f"=== holding out {fam}", flush=True)
        splits = load_data(base, exclude_from_train=(families == fam))
        held = rates(run(base, splits=splits, log=log), families)
        rows.append(dict(
            family=fam, n_test_rows=held[fam][1], n_train_rows_removed=int((families == fam).sum()),
            detection_rate_seen_in_training=baseline[fam][0], detection_rate_held_out=held[fam][0],
            benign_specificity_baseline=baseline["Benign"][0], benign_specificity_held_out=held["Benign"][0]))
        write_rows(EVIDENCE / "holdout_family.csv", rows)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
