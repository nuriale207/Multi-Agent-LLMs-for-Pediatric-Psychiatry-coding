"""
Re-binarizes predictions_raw.csv with a custom threshold.

Usage:
    python rebinarize_predictions.py \
        --raw_predictions_csv /path/to/predictions_raw.csv \
        --threshold 0.4 \
        --output_csv /path/to/predictions_0_4.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_predictions_csv', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.raw_predictions_csv)
    binarized = (df >= args.threshold).astype(int)
    binarized.to_csv(args.output_csv, index=False)
    print(f"Saved binarized predictions (threshold={args.threshold}) to {args.output_csv}")


if __name__ == '__main__':
    main()
