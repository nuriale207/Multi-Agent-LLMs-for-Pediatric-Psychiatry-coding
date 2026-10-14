"""
Evaluates note-level predictions using a voting threshold strategy.

For each patient, a code is assigned if it is predicted in at least
(voting_threshold * 100)% of that patient's notes. Voting threshold is
swept from 0% to 50% in steps of 5% (0% = union, the current baseline).

Results are computed for each combination of:
  - experiment  (note-level only, as listed in --experiments_config)
  - binarization threshold  (0.4, 0.5, 0.6)
  - voting threshold  (0%, 5%, 10%, ..., 50%)

New output files (existing files are never overwritten):
  <experiments_base>/<exp>/test/voting_threshold_{bin_thr}.json
  <experiments_base>/voting_threshold_summary.json

Usage:
    python evaluate_voting_threshold.py --experiments_config configs/experiments.json
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

VOTING_THRESHOLDS = [round(i / 100, 2) for i in range(0, 55, 5)]  # 0.0, 0.05, ..., 0.50

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

def compute_patient_f1(gold_codes, pred_codes):
    gold_set, pred_set = set(gold_codes), set(pred_codes)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


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


def get_age_bin(age):
    try:
        a = float(age)
    except (ValueError, TypeError):
        return None
    for i in range(len(AGE_BIN_EDGES) - 1):
        if AGE_BIN_EDGES[i] <= a < AGE_BIN_EDGES[i + 1]:
            return AGE_BIN_LABELS[i]
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_vote_data(test_csv_path, predictions_csv_path, labels_start_index):
    """
    Loads note-level data and accumulates per-patient vote counts.

    Returns dict {patient_id: {sex, age_bin, gold_full, vote_counts, n_notes}}
    where vote_counts = {code: n_notes_that_predicted_it}.
    """
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
    sex_by_pat   = {}
    age_by_pat   = {}

    for i, (_, test_row) in enumerate(test_df.iterrows()):
        if i >= len(pred_df):
            break

        pid     = str(test_row.get("idpaciente", i))
        sex_val = str(test_row.get("sexo", "unknown"))
        age_val = test_row.get("age", None)

        if pid not in sex_by_pat:
            sex_by_pat[pid] = sex_val
            age_by_pat[pid] = age_val

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
    records = {}
    for pid in all_pids:
        records[pid] = {
            "sex":         sex_by_pat.get(pid, "unknown"),
            "age_bin":     get_age_bin(age_by_pat.get(pid)),
            "gold_full":   gold_by_pat[pid],
            "vote_counts": dict(votes_by_pat[pid]),
            "n_notes":     notes_by_pat[pid],
        }
    return records


# ── Evaluation ────────────────────────────────────────────────────────────────

def apply_voting_thr(vote_counts, n_notes, voting_thr):
    """Predicted codes for a patient at a given voting threshold."""
    if n_notes == 0:
        return set()
    # vote_counts only contains codes predicted >= 1 time, so count/n_notes >= 0 is
    # always True → T=0.0 naturally gives the union of all predicted codes.
    return {code for code, count in vote_counts.items()
            if count / n_notes >= voting_thr}


def stats_dict(scores):
    if not scores:
        return {"n_patients": 0, "mean_f1": None, "std_f1": None}
    return {
        "n_patients": len(scores),
        "mean_f1": float(np.mean(scores)),
        "std_f1":  float(np.std(scores)),
    }


def evaluate_at_voting_thr(records, voting_thr, use_main_codes):
    """
    Computes per-patient F1 at a given voting threshold and aggregates
    overall + by sex + by age_bin + by sex_and_age_bin.
    """
    patient_rows = []
    for pid, r in records.items():
        pred_full = apply_voting_thr(r["vote_counts"], r["n_notes"], voting_thr)
        gold_full = r["gold_full"]

        if use_main_codes:
            gold = {to_main_code(c) for c in gold_full}
            pred = {to_main_code(c) for c in pred_full}
        else:
            gold = gold_full
            pred = pred_full

        patient_rows.append({
            "sex":     r["sex"],
            "age_bin": r["age_bin"],
            "f1":      compute_patient_f1(gold, pred),
        })

    all_f1 = [d["f1"] for d in patient_rows]

    sex_groups   = defaultdict(list)
    age_groups   = defaultdict(list)
    combo_groups = defaultdict(list)

    for d in patient_rows:
        s, ab = d["sex"], d["age_bin"]
        valid_sex = s not in ("0", "", None, "unknown")

        if valid_sex:
            sex_groups[s].append(d["f1"])
        if ab is not None:
            age_groups[ab].append(d["f1"])
        if valid_sex and ab is not None:
            combo_groups[f"{s}_{ab}"].append(d["f1"])

    return {
        "overall":            stats_dict(all_f1),
        "by_sex":             {k: stats_dict(v) for k, v in sorted(sex_groups.items())},
        "by_age_bin":         {k: stats_dict(v) for k, v in sorted(age_groups.items())},
        "by_sex_and_age_bin": {k: stats_dict(v) for k, v in sorted(combo_groups.items())},
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.4f}" if v is not None else " N/A  "


def print_summary_table(exp_name, bin_thr, results):
    print(f"\n{'='*70}")
    print(f"  {exp_name}  |  binarization={bin_thr}")
    print(f"{'='*70}")
    header = f"  {'voting_thr':>11}  {'full_codes':>12}  {'main_codes':>12}"
    print(header)
    print(f"  {'-'*40}")
    for vt_key, data in results.items():
        fc = data["full_codes"]["overall"]["mean_f1"]
        mc = data["main_codes"]["overall"]["mean_f1"]
        print(f"  {vt_key:>11}  {fmt(fc):>12}  {fmt(mc):>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_config", required=True,
                        help="Path to a JSON file describing note-level experiments "
                             "(see configs/experiments.example.json)")
    args = parser.parse_args()

    experiments_base, note_experiments = load_experiments_config(args.experiments_config)

    all_results = {}

    for exp in note_experiments:
        all_results[exp["name"]] = {}

        for bin_thr in ["0_4", "0_5", "0_6"]:
            exp_dir   = os.path.join(experiments_base, exp["name"], "test")
            pred_file = os.path.join(exp_dir, PRED_FILE_MAP[bin_thr])

            if not os.path.exists(pred_file):
                print(f"[SKIP] {exp['name']} bin={bin_thr}: {pred_file} not found")
                continue

            print(f"\n[{exp['name']}]  binarization_thr={bin_thr}")
            records = load_vote_data(exp["test_csv"], pred_file, exp["labels_start_index"])
            print(f"  Loaded {len(records)} patients")

            thr_results = {}
            for vt in VOTING_THRESHOLDS:
                vt_key = f"{int(vt * 100)}%"
                thr_results[vt_key] = {
                    "voting_threshold": vt,
                    "full_codes": evaluate_at_voting_thr(records, vt, use_main_codes=False),
                    "main_codes": evaluate_at_voting_thr(records, vt, use_main_codes=True),
                }

            all_results[exp["name"]][bin_thr] = thr_results
            print_summary_table(exp["name"], bin_thr, thr_results)

            out_path = os.path.join(exp_dir, f"voting_threshold_{bin_thr}.json")
            with open(out_path, "w") as fh:
                json.dump(thr_results, fh, indent=2)
            print(f"\n  Saved -> {out_path}")

    summary_path = os.path.join(experiments_base, "voting_threshold_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
