import argparse
import shutil
import csv
import json
import os
import re
import statistics as st
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = Path(os.environ.get("P1_RESULTS_DIR", ROOT / "results"))
EVIDENCE = Path(os.environ.get("P1_EVIDENCE_DIR", HERE / "evidence"))

PAPER = {"accuracy": 0.994422814, "precision": 0.947579486, "recall": 0.933277496, "f1": 0.940305998}
ORIGINAL = {"accuracy": 0.88790, "precision": 0.97090, "recall": 0.87960, "f1": 0.92300, "roc_auc": 0.95550}

macros, tables = {}, {}


def tex(s):
    return str(s).replace("\\", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def f(x, d=4):
    return "n/a" if x in (None, "") else f"{float(x):.{d}f}"


def pct(x, d=1):
    return "n/a" if x in (None, "") else f"{100 * float(x):.{d}f}"


def num(n):
    return f"{int(n):,}".replace(",", "{,}")


def put(name, val):
    macros[name] = val


def rows_of(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def missing(what, cmd):
    return rf"\fbox{{[missing: {tex(what)}; run {tex(cmd)}]}}"


def table(caption, label, spec, header, body, star=False, size=r"\footnotesize"):
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]\centering\caption{{{caption}}}\label{{{label}}}{size}\setlength{{\tabcolsep}}{{3pt}}",
             rf"\begin{{tabular}}{{{spec}}}\toprule", header + r" \\ \midrule"]
    lines += [row + r" \\" for row in body]
    lines += [rf"\bottomrule\end{{tabular}}\end{{{env}}}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--no-submission", action="store_true", help="do not touch SUBMISSION.md")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    M = json.loads((RESULTS / "metrics.json").read_text())
    D, S = M["details"], M["split"]
    log = json.loads((RESULTS / "training_log.json").read_text())
    val = json.loads((RESULTS / "validation_metrics.json").read_text())

    # ---- main result (grouped test partition) ------------------------------------------------------
    for k, n in (("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec"), ("f1", "FOne"), ("roc_auc", "Auc")):
        put(f"Main{n}", f(M[k]))
        put(f"Val{n}", f(val[k]))
    put("MainAP", f(D["average_precision"]))
    for k, n in (("precision", "Prec"), ("recall", "Rec"), ("f1", "FOne")):
        put(f"MainMacro{n}", f(D["macro_average"][k]))
        put(f"MainBenign{n}", f(D["per_class"]["benign"][k]))
    put("MainBalAcc", f(D["balanced_accuracy"]))
    put("MainMajority", f(D["majority_class_accuracy"]))
    for k in ("tn", "fp", "fn", "tp"):
        put(f"Conf{k.upper()}", num(D["confusion"][k]))
    for k, n in (("n_train", "NTrain"), ("n_val", "NVal"), ("n_test", "NTest")):
        put(n, num(S[k]))
    put("TestMalPct", pct(S["malicious_fraction"]["test"]))
    put("TestSeenInTrain", str(S["test_rows_with_vector_in_train"]))
    put("RunSeconds", f"{log['run_info']['seconds']:.0f}")
    put("RunDevice", tex(log["run_info"]["gpu"] or "CPU"))
    put("ConfigEpochs", str(cfg["train"]["epochs"]))
    put("ConfigSeed", str(cfg["seed"]))
    for k, n in (("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec"), ("f1", "FOne"), ("roc_auc", "Auc")):
        put(f"Orig{n}", f(ORIGINAL[k]))
    for k, n in (("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec"), ("f1", "FOne")):
        put(f"Paper{n}", f(PAPER[k]))
    # Paper class prior (Table 3 benign share) and what its Table 6 numbers imply about the averaging.
    b = 1_098_195 / 46_686_579
    put("PaperMajority", f(1 - b))
    put("PaperMargin", f"{100 * (PAPER['accuracy'] - (1 - b)):.1f}")
    put("OurMargin", f"{100 * (M['accuracy'] - D['majority_class_accuracy']):.1f}")
    put("PaperMalPosAccBound", f(1 - (1 - b) * (1 - PAPER["recall"])))
    r_sum = 2 * PAPER["recall"]                       # macro recall: r0 + r1
    r0 = (r_sum * (1 - b) - PAPER["accuracy"]) / ((1 - b) - b)
    put("PaperImpliedBenignRec", f(r0))
    put("PaperImpliedMalRec", f(r_sum - r0))

    # ---- duplicate audit -------------------------------------------------------------------------------
    audit = rows_of(EVIDENCE / "duplicate_audit.csv")
    A = {(r["section"], r["name"]): r["value"] for r in audit} if audit else {}
    if audit:
        put("DupRows", num(A["overall", "rows"]))
        put("DupDistinct", num(A["overall", "distinct_feature_vectors"]))
        put("DupRepeats", num(A["overall", "rows_that_repeat_an_earlier_vector"]))
        put("DupConflictVecs", num(A["label_conflict", "vectors_with_both_labels"]))
        put("DupConflictRows", num(A["label_conflict", "rows_in_those_vectors"]))
        put("DupAccBound", f(A["label_conflict", "accuracy_upper_bound"]))
        put("DupSeenOrig", num(A["random_80_20_split", "seed_42_test_rows_with_twin_in_train"]))
        put("DupSeenOrigPct", pct(A["random_80_20_split", "seed_42_test_fraction_with_twin_in_train"], 2))
        put("DupSeenBenignPct", pct(A["random_80_20_split", "seed_42_benign_fraction_with_twin"]))
        put("DupSeenMalPct", pct(A["random_80_20_split", "seed_42_malicious_fraction_with_twin"]))
        fr = [float(v) for (s, n), v in A.items() if s == "random_80_20_split" and n.endswith("test_fraction_with_twin_in_train")]
        put("DupSeenMin", pct(min(fr), 1)); put("DupSeenMax", pct(max(fr), 1))

    # ---- seed runs: split protocols ---------------------------------------------------------------------
    seeds = rows_of(EVIDENCE / "seed_runs.csv")
    if seeds:
        def sel(vary, strategy):
            return [r for r in seeds if r["vary"] == vary and r["strategy"] == strategy]
        body = []
        for vary, strategy, label in (("both", "grouped", "grouped split"), ("both", "random", "original random split"),
                                      ("init", "grouped", "grouped, fixed split"), ("init", "random", "random, fixed split")):
            rs = sel(vary, strategy)
            if not rs:
                continue
            cells = []
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                v = [float(r[f"test_{k}"]) for r in rs]
                cells.append(f"{st.mean(v):.4f}$\\pm${(st.stdev(v) if len(v) > 1 else 0):.4f}")
            body.append(f"{label} & {len(rs)} & " + " & ".join(cells))
            put(f"Seeds{vary.title()}{strategy.title()}N", str(len(rs)))
            for k, n in (("f1", "FOne"), ("accuracy", "Acc"), ("recall", "Rec")):
                v = [float(r[f"test_{k}"]) for r in rs]
                put(f"Seeds{vary.title()}{strategy.title()}{n}Mean", f"{st.mean(v):.4f}")
                put(f"Seeds{vary.title()}{strategy.title()}{n}Sd", f"{(st.stdev(v) if len(v) > 1 else 0):.4f}")
        cap = "Test metrics over seeds (mean $\\pm$ sd). The seed changes the split and the initialisation."
        if any(sel("init", s) for s in ("grouped", "random")):
            cap = ("Test metrics over seeds (mean $\\pm$ sd). Rows without ``fixed split'' change the split and the "
                   "initialisation with the seed; ``fixed split'' rows change only the initialisation and batch order.")
        tables["tabSeeds"] = table(
            cap,
            "tab:seeds", "@{}lrccccc@{}", "Protocol & Runs & Acc & Prec & Rec & F1 & AUC", body)
        # seen / unseen breakdown of the original random split
        body, rr = [], sel("both", "random")
        if rr:
            def avg(key):
                vals = [float(r[key]) for r in rr if r.get(key) not in ("", None)]
                return st.mean(vals) if vals else None
            for part, label in (("seen", "twin in training"), ("unseen", "no twin in training")):
                body.append(f"{label} & {pct(avg(part + '_n') / avg('n_test'), 1)}\\% & {f(avg(part + '_accuracy'))} & "
                            f"{f(avg(part + '_precision'))} & {f(avg(part + '_recall'))} & {f(avg(part + '_f1'))}")
            body.append(f"all test rows & 100\\% & {f(avg('test_accuracy'))} & {f(avg('test_precision'))} & "
                        f"{f(avg('test_recall'))} & {f(avg('test_f1'))}")
            for part in ("seen", "unseen"):
                put(f"Orig{part.title()}Acc", f(avg(part + "_accuracy")))
                put(f"Orig{part.title()}Rec", f(avg(part + "_recall")))
                put(f"Orig{part.title()}FOne", f(avg(part + "_f1")))
                put(f"Orig{part.title()}Share", pct(avg(part + "_n") / avg("n_test"), 1))
            tables["tabSeen"] = table(
                "Original random 80/20 split, test rows grouped by whether the row's exact feature vector also occurs in "
                f"training (mean over {len(rr)} seeds).",
                "tab:seen", "@{}lrcccc@{}", "Test rows & Share & Acc & Prec & Rec & F1", body)
    for name, cmd in (("tabSeeds", "tools/seed_runs.py"), ("tabSeen", "tools/seed_runs.py")):
        tables.setdefault(name, missing("seed_runs.csv", cmd))

    # ---- sweep on validation -------------------------------------------------------------------------
    sweep = rows_of(EVIDENCE / "sweep_validation.csv")
    if sweep:
        chosen = (str(cfg["model"]["hidden_sizes"]), float(cfg["train"]["lr"]), bool(cfg["train"]["use_class_weights"]))
        body = []
        for r in sorted(sweep, key=lambda r: -float(r["val_f1"])):
            is_chosen = (r["hidden_sizes"], float(r["lr"]), r["use_class_weights"] == "True") == chosen
            arch = "/".join(re.findall(r"\d+", r["hidden_sizes"]))
            mark = r"$\ast$" if is_chosen else ""
            body.append(f"{arch}{mark} & {r['lr']} & {'weighted' if r['use_class_weights'] == 'True' else 'plain'} & "
                        f"{f(r['val_precision'])} & {f(r['val_recall'])} & {f(r['val_f1'])} & {f(r['val_roc_auc'])}")
        top = max(sweep, key=lambda r: float(r["val_f1"]))
        put("SweepRuns", str(len(sweep)))
        put("SweepBestF", f(top["val_f1"]))
        put("SweepBestArch", "/".join(re.findall(r"\d+", top["hidden_sizes"])))
        put("SweepBestLr", top["lr"])
        put("SweepBestWeights", "weighted" if top["use_class_weights"] == "True" else "unweighted")
        put("SweepChosenIsBest", "yes" if (top["hidden_sizes"], float(top["lr"]), top["use_class_weights"] == "True") == chosen else "no")
        put("SweepEpochs", str(sweep[0]["epochs"]))
        plain1 = [float(r["val_f1"]) for r in sweep if r["use_class_weights"] != "True" and float(r["lr"]) == 0.001]
        plain3 = [float(r["val_f1"]) for r in sweep if r["use_class_weights"] != "True" and float(r["lr"]) == 0.003]
        weighted = [r for r in sweep if r["use_class_weights"] == "True" and float(r["lr"]) == 0.001]
        put("SweepLrOneMin", f(min(plain1))); put("SweepLrOneMax", f(max(plain1)))
        put("SweepLrThreeMax", f(max(plain3)))
        put("SweepSpread", f(max(plain1) - min(plain1)))
        if weighted:
            put("SweepWeightedF", f(weighted[0]["val_f1"])); put("SweepWeightedPrec", f(weighted[0]["val_precision"]))
            put("SweepWeightedRec", f(weighted[0]["val_recall"]))
        tables["tabSweep"] = table(
            f"Validation sweep ({sweep[0]['epochs']} epochs each), sorted by validation F1. $\\ast$ marks the configuration "
            "in config.yaml. The test partition was not scored.",
            "tab:sweep", "@{}lcccccc@{}", "Hidden & LR & Loss & Prec & Rec & F1 & AUC", body)
    tables.setdefault("tabSweep", missing("sweep_validation.csv", "tools/sweep.py"))

    # ---- per source ------------------------------------------------------------------------------------
    per = rows_of(EVIDENCE / "per_source.csv")
    if per:
        src = [r for r in per if r["scope"] == "source"]
        fam = [r for r in per if r["scope"] == "family"]
        repeat = {n[:-len("_repeat_share")]: float(v) for (s, n), v in A.items() if s == "by_source" and n.endswith("_repeat_share")}
        families = {"BenignTraffic.pcap.csv": "Benign"}
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "data" / "build_dataset.py")
            bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
            families.update(bd.MALICIOUS_FILES)
        except Exception:
            pass
        body = []
        for r in sorted(src, key=lambda r: (families.get(r["name"], "~"), r["name"])):
            label = "specificity" if r["label"] == "0" else "detection"
            body.append(f"{tex(r['name'].replace('.pcap.csv', ''))} & {tex(families.get(r['name'], ''))} & {num(r['n_test_rows'])} & "
                        f"{f(r['correct_rate'], 3)} ({label}) & {pct(r['share_of_all_errors'])}\\% & {pct(repeat.get(r['name']), 1)}\\%")
        tables["tabSource"] = table(
            "Grouped test partition by source file: fraction classified correctly (detection rate for attacks, specificity for benign), "
            "share of all test errors, and share of that file's rows in the dataset that repeat an earlier feature vector.",
            "tab:source", "@{}llrccr@{}", "File & Family & Rows & Correct & Errors & Repeats", body, star=True)
        for r in fam:
            key = re.sub(r"[^A-Za-z]", "", r["name"])
            put(f"Fam{key}Rate", f(r["correct_rate"], 3))
            put(f"Fam{key}Err", pct(r["share_of_all_errors"]))
    tables.setdefault("tabSource", missing("per_source.csv", "tools/per_source.py"))

    # ---- hold-out family -------------------------------------------------------------------------------
    ho = rows_of(EVIDENCE / "holdout_family.csv")
    if ho:
        body = [f"{tex(r['family'])} & {num(r['n_test_rows'])} & {f(r['detection_rate_seen_in_training'], 3)} & "
                f"{f(r['detection_rate_held_out'], 3)} & {f(r['benign_specificity_baseline'], 3)} & {f(r['benign_specificity_held_out'], 3)}"
                for r in ho]
        for r in ho:
            key = re.sub(r"[^A-Za-z]", "", r["family"])
            put(f"Hold{key}Seen", f(r["detection_rate_seen_in_training"], 3))
            put(f"Hold{key}Out", f(r["detection_rate_held_out"], 3))
        tables["tabHoldout"] = table(
            "Detection rate on each attack family's test rows when the family is in training versus withheld from training, "
            "with benign specificity for both models (same seed and split).",
            "tab:holdout", "@{}lrcccc@{}", "Family & Rows & Seen & Held out & Spec. seen & Spec. held out", body)
    tables.setdefault("tabHoldout", missing("holdout_family.csv", "tools/holdout_family.py"))

    # ---- main comparison table ---------------------------------------------------------------------------
    orig_run = next((r for r in (seeds or []) if r["vary"] == "both" and r["strategy"] == "random" and r["seed"] in ("42", "42.0")), None)
    def orig(k):
        return f(orig_run[f"test_{k}"]) if orig_run else "n/a"
    mac = D["macro_average"]
    body = [
        f"Accuracy & {f(M['accuracy'])} & {orig('accuracy')} & {f(ORIGINAL['accuracy'])} & {f(D['balanced_accuracy'])}\\textsuperscript{{b}} & {f(PAPER['accuracy'])}",
        f"Precision & {f(M['precision'])} & {orig('precision')} & {f(ORIGINAL['precision'])} & {f(mac['precision'])} & {f(PAPER['precision'])}",
        f"Recall & {f(M['recall'])} & {orig('recall')} & {f(ORIGINAL['recall'])} & {f(mac['recall'])} & {f(PAPER['recall'])}",
        f"F1 & {f(M['f1'])} & {orig('f1')} & {f(ORIGINAL['f1'])} & {f(mac['f1'])} & {f(PAPER['f1'])}",
        f"ROC-AUC & {f(M['roc_auc'])} & {orig('roc_auc')} & {f(ORIGINAL['roc_auc'])} & -- & --",
        f"Always-malicious accuracy & {f(D['majority_class_accuracy'])} & -- & -- & -- & 0.9765\\textsuperscript{{c}}",
    ]
    tables["tabMain"] = table(
        "Test results. Ours: grouped split, malicious as positive class, threshold 0.5. ``Original'': same code on the original "
        "random 80/20 split (a fresh run of the original protocol / the submitted numbers). Macro: unweighted mean over the two classes "
        "(\\textsuperscript{b} balanced accuracy). Paper: two-class DNN, Table 6 of Neto et al.; \\textsuperscript{c} computed from its Table 3 "
        "benign share (1{,}098{,}195 of 46{,}686{,}579 rows).",
        "tab:main", "@{}lccccc@{}", "Metric & Ours & Orig.\\ (rerun) & Orig.\\ (submitted) & Ours, macro & Paper DNN", body, star=True)

    # ---- write ----------------------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "numbers.tex").write_text(
        "% generated by make_report_assets.py, do not edit\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    (args.out_dir / "tables.tex").write_text(
        "% generated by make_report_assets.py, do not edit\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(tables.items())))
    for png in ("confusion_matrix.png", "roc_curve.png"):
        if (RESULTS / png).exists():
            shutil.copy2(RESULTS / png, args.out_dir / png)
    print(f"wrote {args.out_dir/'numbers.tex'} ({len(macros)} macros) and tables.tex ({len(tables)} tables)")

    if not args.no_submission:
        block = (f"- Accuracy: {f(M['accuracy'])}\n- Precision / Recall / F1: {f(M['precision'])} / {f(M['recall'])} / {f(M['f1'])}\n"
                 f"- ROC-AUC: {f(M['roc_auc'])}\n- Split: {S['strategy']} (identical feature vectors never straddle partitions), "
                 f"{S['n_train']:,} train / {S['n_val']:,} validation / {S['n_test']:,} test rows, seed {S['seed']}\n")
        sub = ROOT / "SUBMISSION.md"
        text = sub.read_text()
        new = re.sub(r"(<!-- claimed-results:start -->\n).*?(<!-- claimed-results:end -->)",
                     lambda m: m.group(1) + block + m.group(2), text, flags=re.S)
        if new == text and "<!-- claimed-results:start -->" not in text:
            print("SUBMISSION.md has no claimed-results markers; left unchanged")
        else:
            sub.write_text(new)
            print("updated the claimed-results block in SUBMISSION.md")


if __name__ == "__main__":
    main()
