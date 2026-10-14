"""
Converts transformer test predictions to the dict format used by the agent evaluation script
and computes the same metrics for fair comparison.

Usage:
    python evaluate_transformer_agent_format.py \
        --test_csv /path/to/test.csv \
        --predictions_csv /path/to/test/predictions.csv \
        --id_column IDPACIENTE \
        --output_file results.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support


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


def build_dicts(test_df, pred_df, id_column, labels_start_index):
    label_cols = list(test_df.columns[labels_start_index:])

    gold_dict = {}
    pred_dict = {}

    for i, (_, row) in enumerate(test_df.iterrows()):
        # Unique key: id + position to handle repeated IDs
        key = f"{row[id_column]}_{i}"

        gold_codes = [col for col in label_cols if str(row.get(col, 0)) in ('1', '1.0')]
        gold_dict[key] = gold_codes if gold_codes else ['O']

        pred_row = pred_df.iloc[i]
        pred_codes = [col for col in label_cols if col in pred_df.columns and str(pred_row.get(col, 0)) in ('1', '1.0')]
        pred_dict[key] = pred_codes if pred_codes else ['O']

    return gold_dict, pred_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate transformer predictions using agent-compatible metrics")
    parser.add_argument('--test_csv', required=True, help="Path to the test split CSV")
    parser.add_argument('--predictions_csv', required=True, help="Path to predictions.csv from the test output dir")
    parser.add_argument('--id_column', default='IDPACIENTE', help="Column to use as instance ID (default: IDPACIENTE)")
    parser.add_argument('--csv_sep', default=',', help="Separator of the test CSV (default: ,)")
    parser.add_argument('--labels_start_index', type=int, default=15, help="Column index where labels start (default: 15)")
    parser.add_argument('--output_file', default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    print(f"Loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv, sep=args.csv_sep, low_memory=False)
    test_df.columns = [c.lower() for c in test_df.columns]
    id_col = args.id_column.lower()

    print(f"Loading predictions: {args.predictions_csv}")
    pred_df = pd.read_csv(args.predictions_csv, low_memory=False)
    pred_df.columns = [c.lower() for c in pred_df.columns]

    if len(test_df) != len(pred_df):
        print(f"WARNING: test rows ({len(test_df)}) != prediction rows ({len(pred_df)})")

    print(f"Building gold/pred dicts using id_column='{id_col}', labels_start_index={args.labels_start_index}")
    gold_dict, pred_dict = build_dicts(test_df, pred_df, id_col, args.labels_start_index)

    print("Running evaluation...")
    results = evaluate_codes(gold_dict, pred_dict)

    print("\n=== RESULTS ===")
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
