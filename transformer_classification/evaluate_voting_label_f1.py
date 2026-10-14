"""
Computes per-label micro F1 and macro F1 for note-level experiments
under the voting threshold strategy (0% to 50%, step 5%).

Metrics (sklearn-style, per label):
  - Macro F1: F1 computed per code, then averaged (unweighted) over codes
               that appear in at least one gold annotation.
  - Micro F1: global TP / FP / FN aggregated across all codes and patients,
               then F1 from those totals.

Output (new files, existing files untouched):
  <experiments_base>/<exp>/test/voting_label_f1_{bin_thr}.json
  <experiments_base>/voting_label_f1_summary.json

Usage:
    python evaluate_voting_label_f1.py --experiments_config configs/experiments.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

PRED_FILE_MAP = {
    "0_5": "predictions.csv",
    "0_4": "predictions_0_4.csv",
    "0_6": "predictions_0_6.csv",
}

VOTING_THRESHOLDS = [round(i / 100, 2) for i in range(0, 55, 5)]

AGE_BIN_EDGES  = [0, 8, 13, 18, np.inf]
AGE_BIN_LABELS = ["<8", "8-12", "13-17", "18+"]


def load_experiments_config(config_path):
    """
    Loads a JSON config of the form:
    {
      "experiments_base": "/path/to/Experiments_data",
      "experiments": [
        {"name": "...", "test_csv": "...", "labels_start_index": 15}
      ]
    }
    Only note-level experiments make sense here (voting across a patient's notes).
    See configs/experiments.example.json for a template.
    """
    with open(config_path) as fh:
        config = json.load(fh)
    return config["experiments_base"], config["experiments"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_code(code):
    if not isinstance(code, str):
        return None
    code = code.strip().upper()
    if not code.startswith("F"):
        return None
    if len(code) > 3 and code[3] != ".":
        code = code[:3] + "." + code[3:]
    return code


def to_main_code(code):
    return code[:3]


def load_vote_data(test_csv_path, predictions_csv_path, labels_start_index):
    """Same loader as evaluate_voting_threshold.py."""
    print(f"  Loading test CSV: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path, low_memory=False)
    test_df.columns = [c.lower() for c in test_df.columns]

    print(f"  Loading predictions: {predictions_csv_path}")
    pred_df = pd.read_csv(predictions_csv_path, low_memory=False)
    pred_df.columns = [c.lower() for c in pred_df.columns]

    if len(test_df) != len(pred_df):
        print(f"  WARNING: test rows ({len(test_df)}) != pred rows ({len(pred_df)})")

    label_cols = list(test_df.columns[labels_start_index:])

    gold_by_pat  = defaultdict(set)
    votes_by_pat = defaultdict(lambda: defaultdict(int))
    notes_by_pat = defaultdict(int)

    for i, (_, test_row) in enumerate(test_df.iterrows()):
        if i >= len(pred_df):
            break
        pid = str(test_row.get("idpaciente", i))
        notes_by_pat[pid] += 1

        for col in label_cols:
            if str(test_row.get(col, 0)) in ("1", "1.0"):
                code = normalize_code(col)
                if code:
                    gold_by_pat[pid].add(code)

        pred_row = pred_df.iloc[i]
        for col in pred_df.columns:
            if str(pred_row.get(col, 0)) in ("1", "1.0"):
                code = normalize_code(col)
                if code:
                    votes_by_pat[pid][code] += 1

    all_pids = sorted(set(gold_by_pat) | set(votes_by_pat) | set(notes_by_pat))
    return {
        pid: {
            "gold_full":   gold_by_pat[pid],
            "vote_counts": dict(votes_by_pat[pid]),
            "n_notes":     notes_by_pat[pid],
        }
        for pid in all_pids
    }


def apply_voting_thr(vote_counts, n_notes, voting_thr):
    if n_notes == 0:
        return set()
    return {code for code, count in vote_counts.items()
            if count / n_notes >= voting_thr}


# ── Label-level metrics ───────────────────────────────────────────────────────

def compute_label_metrics(records, voting_thr, use_main_codes):
    """
    Returns macro F1, micro F1, micro precision, micro recall.

    Macro: per-label F1 averaged over labels with gold support.
    Micro: global TP/FP/FN aggregated across all patients and labels.
    """
    # Build (gold, pred) sets per patient
    gold_pred = {}
    for pid, r in records.items():
        pred_full = apply_voting_thr(r["vote_counts"], r["n_notes"], voting_thr)
        gold_full = r["gold_full"]
        if use_main_codes:
            gold = {to_main_code(c) for c in gold_full}
            pred = {to_main_code(c) for c in pred_full}
        else:
            gold = gold_full
            pred = pred_full
        gold_pred[pid] = (gold, pred)

    # All labels with gold support
    gold_labels = set()
    for gold, _ in gold_pred.values():
        gold_labels.update(gold)

    # Per-label TP, FP, FN  (only labels present in gold)
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    for gold, pred in gold_pred.values():
        for label in gold_labels:
            in_gold = label in gold
            in_pred = label in pred
            if in_gold and in_pred:
                label_tp[label] += 1
            elif not in_gold and in_pred:
                label_fp[label] += 1
            elif in_gold and not in_pred:
                label_fn[label] += 1

    # Macro P / R / F1
    label_precs, label_recs, label_f1s = [], [], []
    for label in gold_labels:
        tp, fp, fn = label_tp[label], label_fp[label], label_fn[label]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        label_precs.append(prec)
        label_recs.append(rec)
        label_f1s.append(f1)
    macro_prec = float(np.mean(label_precs)) if label_precs else 0.0
    macro_rec  = float(np.mean(label_recs))  if label_recs  else 0.0
    macro_f1   = float(np.mean(label_f1s))   if label_f1s   else 0.0

    # Micro P / R / F1 — aggregate TP/FP/FN across all patients (not just gold labels,
    # to correctly penalise false positives for codes outside the gold label set)
    total_tp = sum(len(g & p) for g, p in gold_pred.values())
    total_fp = sum(len(p - g) for g, p in gold_pred.values())
    total_fn = sum(len(g - p) for g, p in gold_pred.values())

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0.0)

    return {
        "macro_precision": macro_prec,
        "macro_recall":    macro_rec,
        "macro_f1":        macro_f1,
        "micro_precision": float(micro_prec),
        "micro_recall":    float(micro_rec),
        "micro_f1":        float(micro_f1),
        "n_gold_labels":   len(gold_labels),
    }


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(exp_name, code_level, all_bin_results):
    """
    all_bin_results: {bin_thr: {vt_key: metrics_dict}}
    Prints one sub-table per binarization threshold, all metrics.
    """
    bin_thrs = sorted(all_bin_results.keys())
    vt_keys  = [f"{int(vt*100)}%" for vt in VOTING_THRESHOLDS]

    for b in bin_thrs:
        print(f"\n{'='*72}")
        print(f"  {exp_name}  |  {code_level}  |  bin={b}")
        print(f"{'='*72}")
        print(f"  {'voting_thr':>11}  {'macro_P':>8}  {'macro_R':>8}  {'macro_F1':>8}"
              f"  {'micro_P':>8}  {'micro_R':>8}  {'micro_F1':>8}")
        print(f"  {'-'*66}")
        for vt_key in vt_keys:
            if b not in all_bin_results or vt_key not in all_bin_results[b]:
                continue
            m = all_bin_results[b][vt_key]
            print(f"  {vt_key:>11}"
                  f"  {m['macro_precision']:>8.4f}"
                  f"  {m['macro_recall']:>8.4f}"
                  f"  {m['macro_f1']:>8.4f}"
                  f"  {m['micro_precision']:>8.4f}"
                  f"  {m['micro_recall']:>8.4f}"
                  f"  {m['micro_f1']:>8.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_config", required=True,
                        help="Path to a JSON file describing note-level experiments "
                             "(see configs/experiments.example.json)")
    parser.add_argument("--bin_thr", default=None, choices=["0_4", "0_5", "0_6"],
                        help="Run only for this binarization threshold")
    args = parser.parse_args()
    bin_thrs_to_run = [args.bin_thr] if args.bin_thr else ["0_4", "0_5", "0_6"]

    experiments_base, note_experiments = load_experiments_config(args.experiments_config)

    all_results = {}

    for exp in note_experiments:
        all_results[exp["name"]] = {}
        loaded_records = {}  # cache: bin_thr -> records

        for bin_thr in bin_thrs_to_run:
            exp_dir   = os.path.join(experiments_base, exp["name"], "test")
            pred_file = os.path.join(exp_dir, PRED_FILE_MAP[bin_thr])

            if not os.path.exists(pred_file):
                print(f"[SKIP] {exp['name']} bin={bin_thr}: {pred_file} not found")
                continue

            print(f"\n[{exp['name']}]  binarization_thr={bin_thr}")
            records = load_vote_data(exp["test_csv"], pred_file, exp["labels_start_index"])
            loaded_records[bin_thr] = records
            print(f"  Loaded {len(records)} patients")

            thr_results = {}
            for vt in VOTING_THRESHOLDS:
                vt_key = f"{int(vt * 100)}%"
                thr_results[vt_key] = {
                    "voting_threshold": vt,
                    "full_codes": compute_label_metrics(records, vt, use_main_codes=False),
                    "main_codes": compute_label_metrics(records, vt, use_main_codes=True),
                }

            all_results[exp["name"]][bin_thr] = thr_results

            out_path = os.path.join(exp_dir, f"voting_label_f1_{bin_thr}.json")
            with open(out_path, "w") as fh:
                json.dump(thr_results, fh, indent=2)
            print(f"  Saved -> {out_path}")

        # Print tables per code level
        for code_level in ["full_codes", "main_codes"]:
            all_bin = {
                b: {vk: all_results[exp["name"]][b][vk][code_level]
                    for vk in all_results[exp["name"]][b]}
                for b in all_results[exp["name"]]
            }
            print_table(exp["name"], code_level, all_bin)

    summary_path = os.path.join(experiments_base, "voting_label_f1_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
