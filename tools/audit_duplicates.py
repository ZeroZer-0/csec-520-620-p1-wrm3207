import numpy as np
import pandas as pd

from _common import EVIDENCE, ROOT, SOURCES_CSV, load_config, write_rows
from src.data import _random_split, read_table, vector_groups

SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def main():
    cfg = load_config()
    X, y, _ = read_table(cfg)
    g = vector_groups(X)
    n_groups = g.max() + 1
    size = np.bincount(g, minlength=n_groups)
    pos = np.bincount(g, weights=y, minlength=n_groups)
    neg = size - pos
    rows = []
    add = lambda section, name, value: rows.append(dict(section=section, name=name, value=value))

    add("overall", "rows", len(y))
    add("overall", "distinct_feature_vectors", int(n_groups))
    add("overall", "rows_that_repeat_an_earlier_vector", int(len(y) - n_groups))
    add("overall", "largest_group_size", int(size.max()))
    conflict = (pos > 0) & (neg > 0)
    add("label_conflict", "vectors_with_both_labels", int(conflict.sum()))
    add("label_conflict", "rows_in_those_vectors", int(size[conflict].sum()))
    forced = int(np.minimum(pos, neg).sum())
    add("label_conflict", "min_rows_any_classifier_must_misclassify", forced)
    add("label_conflict", "accuracy_upper_bound", 1 - forced / len(y))
    for label, name in ((0, "benign"), (1, "malicious")):
        m = y == label
        add("by_class", f"{name}_rows", int(m.sum()))
        add("by_class", f"{name}_distinct_vectors", int(len(np.unique(g[m]))))

    if SOURCES_CSV.exists():
        src = pd.read_csv(SOURCES_CSV)
        assert len(src) == len(y), "dataset_sources.csv is not aligned with dataset.csv"
        for source, idx in src.groupby("source").indices.items():
            n, d = len(idx), len(np.unique(g[idx]))
            add("by_source", f"{source}_rows", n)
            add("by_source", f"{source}_repeat_share", (n - d) / n)
    else:
        print("data/dataset_sources.csv not found: skipping per-source rows")

    for seed in SEEDS:                                   # the original 80/20 stratified row split
        tr, _, te = _random_split(y, 0.2, 0.0, seed)
        seen = np.isin(g[te], g[tr])
        add("random_80_20_split", f"seed_{seed}_test_rows_with_twin_in_train", int(seen.sum()))
        add("random_80_20_split", f"seed_{seed}_test_fraction_with_twin_in_train", float(seen.mean()))
        add("random_80_20_split", f"seed_{seed}_benign_fraction_with_twin", float(seen[y[te] == 0].mean()))
        add("random_80_20_split", f"seed_{seed}_malicious_fraction_with_twin", float(seen[y[te] == 1].mean()))
    write_rows(EVIDENCE / "duplicate_audit.csv", rows)
    for r in rows[:12]:
        print(f"{r['section']:15s} {r['name']:45s} {r['value']}")


if __name__ == "__main__":
    main()
