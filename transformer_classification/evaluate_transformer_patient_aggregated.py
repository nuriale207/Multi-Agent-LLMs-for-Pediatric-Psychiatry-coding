"""
Evaluates note-level transformer predictions at the patient level by aggregating
(union) all codes predicted across a patient's notes.

Usage:
    python evaluate_transformer_patient_aggregated.py \
        --test_csv /path/to/test_notes.csv \
        --predictions_csv /path/to/test/predictions.csv \
        --csv_sep , \
        --labels_start_index 15 \
        --output_file results_patient.json
"""

import argparse
import json
from collections import defaultdict

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def evaluate_codes(gold_dict, extracted_dict):
    def normalize_and_filter(codes):
        normalized = []
        if not isinstance(codes, list):
            codes = [codes] if codes else []
        for code in codes:
            if not isinstance(code, str):
                continue
            code = code.strip()
            if not code.upper().startswith('F'):
                continue
            if len(code) > 3 and code[3] != '.':
                code = code[:3] + '.' + code[3:]
            normalized.append(code.upper())
        if not normalized:
            return ['O']
        return list(set(normalized))

    def truncate_to_main(codes):
        main_codes = set()
        for code in codes:
            if code == 'O':
                main_codes.add('O')
            else:
                main_codes.add(code[:3])
        return list(main_codes)

    all_keys = set(gold_dict.keys()).union(set(extracted_dict.keys()))

    gold_full_all, pred_full_all = [], []
    gold_main_all, pred_main_all = [], []

    for key in all_keys:
        g_norm = normalize_and_filter(gold_dict.get(key, ['O']))
        e_norm = normalize_and_filter(extracted_dict.get(key, ['O']))
        gold_full_all.append(g_norm)
        pred_full_all.append(e_norm)
        gold_main_all.append(truncate_to_main(g_norm))
        pred_main_all.append(truncate_to_main(e_norm))

    mlb_full = MultiLabelBinarizer()
    mlb_full.fit(gold_full_all + pred_full_all)
    y_true_full = mlb_full.transform(gold_full_all)
    y_pred_full = mlb_full.transform(pred_full_all)

    p_mic_f, r_mic_f, f1_mic_f, _ = precision_recall_fscore_support(y_true_full, y_pred_full, average='micro', zero_division=0)
    p_mac_f, r_mac_f, f1_mac_f, _ = precision_recall_fscore_support(y_true_full, y_pred_full, average='macro', zero_division=0)

    mlb_main = MultiLabelBinarizer()
    mlb_main.fit(gold_main_all + pred_main_all)
    y_true_main = mlb_main.transform(gold_main_all)
    y_pred_main = mlb_main.transform(pred_main_all)

    p_mic_m, r_mic_m, f1_mic_m, _ = precision_recall_fscore_support(y_true_main, y_pred_main, average='micro', zero_division=0)
    p_mac_m, r_mac_m, f1_mac_m, _ = precision_recall_fscore_support(y_true_main, y_pred_main, average='macro', zero_division=0)

    return {
        "full_codes": {
            "micro": {"precision": p_mic_f, "recall": r_mic_f, "f1_score": f1_mic_f},
            "macro": {"precision": p_mac_f, "recall": r_mac_f, "f1_score": f1_mac_f}
        },
        "main_codes": {
            "micro": {"precision": p_mic_m, "recall": r_mic_m, "f1_score": f1_mic_m},
            "macro": {"precision": p_mac_m, "recall": r_mac_m, "f1_score": f1_mac_m}
        }
    }


def aggregate_by_patient(test_df, pred_df, labels_start_index):
    test_df = test_df.copy()
    test_df.columns = [c.lower() for c in test_df.columns]
    pred_df.columns = [c.lower() for c in pred_df.columns]

    label_cols = list(test_df.columns[labels_start_index:])

    gold_patient = defaultdict(set)
    pred_patient = defaultdict(set)

    for i, (_, row) in enumerate(test_df.iterrows()):
        patient_id = str(row['idpaciente'])

        # Gold: active label columns for this note
        for col in label_cols:
            if str(row.get(col, 0)) in ('1', '1.0'):
                gold_patient[patient_id].add(col)

        # Pred: active label columns for this note
        pred_row = pred_df.iloc[i]
        for col in label_cols:
            if col in pred_df.columns and str(pred_row.get(col, 0)) in ('1', '1.0'):
                pred_patient[patient_id].add(col)

    # Convert sets to lists, use ['O'] for patients with no codes
    all_patients = set(gold_patient.keys()) | set(pred_patient.keys())
    gold_dict = {pid: list(gold_patient[pid]) if gold_patient[pid] else ['O'] for pid in all_patients}
    pred_dict = {pid: list(pred_patient[pid]) if pred_patient[pid] else ['O'] for pid in all_patients}

    return gold_dict, pred_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate note-level predictions aggregated to patient level")
    parser.add_argument('--test_csv', required=True, help="Note-level test CSV")
    parser.add_argument('--predictions_csv', required=True, help="predictions.csv from test output dir")
    parser.add_argument('--csv_sep', default=',')
    parser.add_argument('--labels_start_index', type=int, default=15)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    print(f"Loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv, sep=args.csv_sep, low_memory=False)

    print(f"Loading predictions: {args.predictions_csv}")
    pred_df = pd.read_csv(args.predictions_csv, low_memory=False)

    if len(test_df) != len(pred_df):
        print(f"WARNING: test rows ({len(test_df)}) != prediction rows ({len(pred_df)})")

    print("Aggregating notes by patient...")
    gold_dict, pred_dict = aggregate_by_patient(test_df, pred_df, args.labels_start_index)
    print(f"Total patients: {len(gold_dict)}")

    print("Running evaluation...")
    results = evaluate_codes(gold_dict, pred_dict)

    print("\n=== RESULTS (patient-level aggregation) ===")
    for level, metrics in results.items():
        print(f"\n{level}:")
        for avg, vals in metrics.items():
            print(f"  {avg}: P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1_score']:.4f}")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({k: {a: {m: float(v) for m, v in mv.items()} for a, mv in lv.items()} for k, lv in results.items()}, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
