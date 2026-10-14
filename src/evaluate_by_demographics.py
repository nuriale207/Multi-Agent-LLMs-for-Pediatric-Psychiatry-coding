"""
Evaluates predictions grouped by patient age bin and sex.

For note-level experiments, predictions are aggregated per patient (union of codes).
For patient-level experiments, each row is already one patient.

Expects the test CSV to contain (case-insensitive) columns `idpaciente`, `sexo`
and `age`, plus one binary (0/1) column per diagnostic code starting at
`--labels_start_index`.

Usage:
    python evaluate_by_demographics.py \
        --test_csv /path/to/test.csv \
        --predictions_csv /path/to/predictions.csv \
        --labels_start_index 15 \
        --output_file results_demographics.json

    # Batch mode over several experiments/thresholds, see configs/experiments.example.json:
    python evaluate_by_demographics.py --batch --experiments_config configs/experiments.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

AGE_BIN_EDGES = [0, 8, 13, 18, np.inf]
AGE_BIN_LABELS = ["<8", "8-12", "13-17", "18+"]

THRESHOLDS = ["0_5", "0_4", "0_6"]
PRED_FILE_MAP = {
    "0_5": "predictions.csv",
    "0_4": "predictions_0_4.csv",
    "0_6": "predictions_0_6.csv",
}


def load_experiments_config(config_path):
    """
    Loads a JSON config of the form:
    {
      "experiments_base": "/path/to/Experiments_data",
      "experiments": [
        {"name": "...", "test_csv": "...", "patient_level": false, "labels_start_index": 15}
      ]
    }
    See configs/experiments.example.json for a template.
    """
    with open(config_path) as fh:
        config = json.load(fh)
    return config["experiments_base"], config["experiments"]


def compute_patient_f1(gold_codes, pred_codes):
    gold_set = set(gold_codes)
    pred_set = set(pred_codes)

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

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
    """Truncate a normalized full code to its 3-character main code (e.g. F32.1 → F32)."""
    return code[:3]


def age_bin(age):
    try:
        a = float(age)
    except (ValueError, TypeError):
        return None
    for i in range(len(AGE_BIN_EDGES) - 1):
        if AGE_BIN_EDGES[i] <= a < AGE_BIN_EDGES[i + 1]:
            return AGE_BIN_LABELS[i]
    return None


def load_data(test_csv_path, predictions_csv_path, labels_start_index, patient_level):
    """
    Returns a list of dicts, one per patient:
        {patient_id, sex, age, age_bin, gold_codes, pred_codes}
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

    gold_by_patient = defaultdict(set)
    pred_by_patient = defaultdict(set)
    sex_by_patient = {}
    age_by_patient = {}

    for i, (_, test_row) in enumerate(test_df.iterrows()):
        if i >= len(pred_df):
            break

        patient_id = str(test_row.get("idpaciente", i))
        sex = str(test_row.get("sexo", "unknown"))
        age_val = test_row.get("age", None)

        # Store demographics (first occurrence wins)
        if patient_id not in sex_by_patient:
            sex_by_patient[patient_id] = sex
            age_by_patient[patient_id] = age_val

        # Gold codes
        for col in label_cols:
            if str(test_row.get(col, 0)) in ("1", "1.0"):
                code = normalize_code(col)
                if code:
                    gold_by_patient[patient_id].add(code)

        # Predicted codes
        pred_row = pred_df.iloc[i]
        for col in pred_df.columns:
            if str(pred_row.get(col, 0)) in ("1", "1.0"):
                code = normalize_code(col)
                if code:
                    pred_by_patient[patient_id].add(code)

    all_patients = sorted(set(gold_by_patient) | set(pred_by_patient))
    records = []
    for pid in all_patients:
        gold_full = list(gold_by_patient[pid])
        pred_full = list(pred_by_patient[pid])
        records.append({
            "patient_id": pid,
            "sex": sex_by_patient.get(pid, "unknown"),
            "age": age_by_patient.get(pid),
            "age_bin": age_bin(age_by_patient.get(pid)),
            "gold_codes": gold_full,
            "pred_codes": pred_full,
            "gold_main_codes": list({to_main_code(c) for c in gold_full}),
            "pred_main_codes": list({to_main_code(c) for c in pred_full}),
        })
    return records


def group_stats(records, key_fn, gold_key, pred_key):
    groups = defaultdict(list)
    for r in records:
        k = key_fn(r)
        if k is not None:
            f1 = compute_patient_f1(r[gold_key], r[pred_key])
            groups[k].append(f1)
    result = {}
    for k, scores in sorted(groups.items()):
        result[str(k)] = {
            "n_patients": len(scores),
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
        }
    return result


def build_section(records, key_fn, gold_key, pred_key):
    all_f1 = [compute_patient_f1(r[gold_key], r[pred_key]) for r in records]
    return {
        "overall": {
            "n_patients": len(all_f1),
            "mean_f1": float(np.mean(all_f1)) if all_f1 else 0.0,
            "std_f1": float(np.std(all_f1)) if all_f1 else 0.0,
        },
        "by_sex": group_stats(records, lambda r: r["sex"] if r["sex"] not in ("0", "", None) else None, gold_key, pred_key),
        "by_age_bin": group_stats(records, lambda r: r["age_bin"], gold_key, pred_key),
        "by_sex_and_age_bin": group_stats(
            records,
            lambda r: f"{r['sex']}_{r['age_bin']}"
            if r["sex"] not in ("0", "", None) and r["age_bin"] is not None
            else None,
            gold_key, pred_key,
        ),
    }


def evaluate_file(test_csv, predictions_csv, labels_start_index, patient_level):
    records = load_data(test_csv, predictions_csv, labels_start_index, patient_level)
    return {
        "full_codes": build_section(records, None, "gold_codes", "pred_codes"),
        "main_codes": build_section(records, None, "gold_main_codes", "pred_main_codes"),
    }


def print_results(results, indent=2):
    pad = " " * indent
    for code_level, sections in results.items():
        print(f"\n{pad}=== {code_level} ===")
        for section, data in sections.items():
            print(f"\n{pad}  {section}:")
            if "n_patients" in data:
                d = data
                print(f"{pad}    n={d['n_patients']}  mean_F1={d['mean_f1']:.4f}  std={d['std_f1']:.4f}")
            else:
                for group, d in data.items():
                    print(f"{pad}    {group}: n={d['n_patients']}  mean_F1={d['mean_f1']:.4f}  std={d['std_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions grouped by age and sex")
    parser.add_argument("--test_csv", help="Test CSV path")
    parser.add_argument("--predictions_csv", help="Predictions CSV path")
    parser.add_argument("--labels_start_index", type=int, default=15)
    parser.add_argument("--patient_level", action="store_true",
                        help="Set if the test CSV already has one row per patient")
    parser.add_argument("--output_file", default=None, help="Save results as JSON")
    parser.add_argument("--batch", action="store_true",
                        help="Run over all experiments listed in --experiments_config")
    parser.add_argument("--experiments_config", default=None,
                        help="Path to a JSON file describing experiments for --batch mode "
                             "(see configs/experiments.example.json)")
    args = parser.parse_args()

    if args.batch:
        if not args.experiments_config:
            parser.error("--batch requires --experiments_config <path.json>")
        experiments_base, experiments = load_experiments_config(args.experiments_config)

        all_results = {}
        for exp in experiments:
            exp_dir = os.path.join(experiments_base, exp["name"], "test")
            all_results[exp["name"]] = {}
            for thr in THRESHOLDS:
                pred_file = os.path.join(exp_dir, PRED_FILE_MAP[thr])
                if not os.path.exists(pred_file):
                    print(f"[SKIP] {exp['name']} thr={thr}: {pred_file} not found")
                    continue
                print(f"\n[{exp['name']}]  threshold={thr}")
                results = evaluate_file(
                    exp["test_csv"],
                    pred_file,
                    exp["labels_start_index"],
                    exp.get("patient_level", False),
                )
                print_results(results)
                all_results[exp["name"]][thr] = results

                # Per-experiment per-threshold output file
                out_path = os.path.join(exp_dir, f"demographics_{thr}.json")
                with open(out_path, "w") as fh:
                    json.dump(results, fh, indent=2)
                print(f"  Saved -> {out_path}")

        summary_path = os.path.join(experiments_base, "demographics_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nBatch summary saved -> {summary_path}")
        return

    if not args.test_csv or not args.predictions_csv:
        parser.error("Provide --test_csv and --predictions_csv, or use --batch --experiments_config")

    results = evaluate_file(
        args.test_csv,
        args.predictions_csv,
        args.labels_start_index,
        args.patient_level,
    )
    print_results(results)

    if args.output_file:
        with open(args.output_file, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
