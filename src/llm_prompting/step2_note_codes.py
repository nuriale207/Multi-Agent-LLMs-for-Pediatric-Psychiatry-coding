"""
Step 2 — Note-level code assignment.

Regimes:
  - top1      : keep only the highest-scoring code per note
  - threshold : keep every retrieved code with similarity > Th_sim

Usage:
    python step2_note_codes.py --input dev_step1_icd_standard.csv --regime top1
    python step2_note_codes.py --input dev_step1_icd_standard.csv --regime threshold --th-sim 0.8

Output: <input>_step2_<top1|threshold0.8>.csv with a `note_codes` column (JSON list).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def top1(top_codes: list[dict]) -> list[str]:
    return [top_codes[0]["code"]] if top_codes else []


def threshold(top_codes: list[dict], th_sim: float) -> list[str]:
    return [c["code"] for c in top_codes if c["similarity"] > th_sim]


def main():
    parser = argparse.ArgumentParser(description="Step 2: note-level code assignment")
    parser.add_argument("--input", default="dev_step1_icd_standard.csv")
    parser.add_argument("--regime", choices=["top1", "threshold"], default="threshold")
    parser.add_argument("--th-sim", type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    top_codes = df["top_codes"].apply(json.loads)

    if args.regime == "top1":
        df["note_codes"] = top_codes.apply(top1)
        tag = "top1"
    else:
        df["note_codes"] = top_codes.apply(lambda tc: threshold(tc, args.th_sim))
        tag = f"threshold{args.th_sim}"

    df["note_codes"] = df["note_codes"].apply(json.dumps)
    output = f"{Path(args.input).with_suffix('')}_step2_{tag}.csv"
    df.to_csv(output, index=False)
    print(f"Saved → {output} ({len(df)} notes)")


if __name__ == "__main__":
    main()
