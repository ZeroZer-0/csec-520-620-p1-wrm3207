import argparse

import numpy as np
import pandas as pd

from _common import EVIDENCE, ROOT, load_sources, write_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(ROOT / "results" / "test_predictions.npz"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    p = np.load(args.predictions)
    src = load_sources().iloc[p["row"]].reset_index(drop=True)
    src["y"], src["pred"] = p["y_true"], (p["prob"] >= args.threshold).astype(int)
    src["correct"] = src["y"] == src["pred"]
    errors = int((~src["correct"]).sum())
    rows = []
    for scope, col in (("family", "family"), ("source", "source")):
        for name, g in src.groupby(col):
            rows.append(dict(scope=scope, name=name, label=int(g["y"].iloc[0]), n_test_rows=len(g),
                             correct_rate=float(g["correct"].mean()),
                             errors=int((~g["correct"]).sum()),
                             share_of_all_errors=float((~g["correct"]).sum() / errors) if errors else 0.0))
    write_rows(EVIDENCE / "per_source.csv", rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
