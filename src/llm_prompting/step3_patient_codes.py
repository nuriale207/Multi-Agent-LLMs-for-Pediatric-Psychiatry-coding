"""
Step 3 — Patient-level code assignment.

Methods:
  - prevalence : keep a code if it appears in >= Th_prev of the patient's notes
  - agentic    : Physician + Patient Safety Advocate review the candidate codes;
                 if either disagrees, the Code Reconciler decides the final list

Usage:
    python step3_patient_codes.py --input dev_step1_icd_standard_step2_threshold0.8.csv --method prevalence --th-prev 0.0
    python step3_patient_codes.py --input dev_step1_icd_standard_step2_threshold0.8.csv --method agentic

Output: <input>_step3_<prevalence0.0|agentic>.csv, one row per patient with
`gold_codes` (union of F codes in COD_CIE_10) and `final_codes` (JSON lists).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

PHYSICIAN_SYSTEM_PROMPT = "<physician_agent_system_prompt>"
PATIENT_SYSTEM_PROMPT = "<patient_safety_advocate_agent_system_prompt>"
RECONCILER_SYSTEM_PROMPT = "<code_reconciler_agent_system_prompt>"
REVIEWER_USER_PROMPT = "<reviewer_user_prompt>"  # fields: {patient_id}, {codes}, {passages}
RECONCILER_USER_PROMPT = "<reconciler_user_prompt>"  # fields: {patient_id}, {codes}, {physician_section}, {patient_section}, {passages}


# ─────────────────────────────────────────────────────────────────────────────
# Prevalence criterion
# ─────────────────────────────────────────────────────────────────────────────

def prevalence(note_codes: pd.Series, th_prev: float) -> list[str]:
    counts = {}
    for codes in note_codes:
        for code in set(codes):
            counts[code] = counts.get(code, 0) + 1
    return [code for code, n in counts.items() if n / len(note_codes) >= th_prev]


# ─────────────────────────────────────────────────────────────────────────────
# Agentic role-play
# ─────────────────────────────────────────────────────────────────────────────

def build_context(pid, group: pd.DataFrame) -> dict:
    """Unique candidate codes (first-appearance order) + the passages they came from."""
    codes, passages = [], []
    for _, row in group.iterrows():
        text = str(row["ENTRADA_ANONIMIZADA"]).strip()
        if not row["note_codes"] or text.lower() in ("", "nan", "none"):
            continue
        codes += [c for c in row["note_codes"] if c not in codes]
        passages.append(f"[{str(row['FECHA'])[:10]}] (top_codes=[{', '.join(row['note_codes'])}])\n{text}")
    return {"IDPACIENTE": pid, "codes": codes, "passages": passages}


def reviewer_msg(ctx: dict) -> str:
    return REVIEWER_USER_PROMPT.format(
        patient_id=ctx["IDPACIENTE"],
        codes="\n".join(f"  - {c}" for c in ctx["codes"]),
        passages="\n\n".join(ctx["passages"]),
    )


def reconciler_msg(ctx: dict, physician: Optional[dict], patient: Optional[dict]) -> str:
    section = lambda title, review: f"{title}:\n{json.dumps(review, ensure_ascii=False, indent=2)}\n\n" if review else ""
    return RECONCILER_USER_PROMPT.format(
        patient_id=ctx["IDPACIENTE"],
        codes="\n".join(f"  - {c}" for c in ctx["codes"]),
        physician_section=section("Physician review", physician),
        patient_section=section("Patient-safety review", patient),
        passages="\n\n".join(ctx["passages"]),
    )


def disagreed(review: dict) -> bool:
    return "parse_error" not in review and bool(review.get("disagreement", False))


def agentic(contexts: list[dict], batch_size: int) -> dict:
    """Returns {IDPACIENTE: row dict} for patients with at least one candidate code."""
    from llm_utils import run_batch, parse_json

    msgs = [reviewer_msg(c) for c in contexts]
    print("[Physician agent]")
    physician = [parse_json(r) for r in run_batch(PHYSICIAN_SYSTEM_PROMPT, msgs, batch_size)]
    print("[Patient Safety Advocate agent]")
    patient = [parse_json(r) for r in run_batch(PATIENT_SYSTEM_PROMPT, msgs, batch_size)]

    # Only disagreeing reviews are passed to the reconciler
    to_reconcile = [i for i in range(len(contexts)) if disagreed(physician[i]) or disagreed(patient[i])]
    print(f"[Code Reconciler agent] {len(to_reconcile)}/{len(contexts)} patients need reconciliation")
    responses = run_batch(RECONCILER_SYSTEM_PROMPT, [
        reconciler_msg(contexts[i],
                       physician[i] if disagreed(physician[i]) else None,
                       patient[i] if disagreed(patient[i]) else None)
        for i in to_reconcile
    ], batch_size)
    reconciler = {i: parse_json(r) for i, r in zip(to_reconcile, responses)}

    results = {}
    for i, ctx in enumerate(contexts):
        rec = reconciler.get(i)
        reconciled = rec is not None and "final_codes" in rec
        results[ctx["IDPACIENTE"]] = {
            "original_codes": ctx["codes"],
            "physician_review": json.dumps(physician[i], ensure_ascii=False),
            "patient_review": json.dumps(patient[i], ensure_ascii=False),
            "reconciler_review": json.dumps(rec, ensure_ascii=False) if rec else None,
            "final_codes": rec["final_codes"] if reconciled else ctx["codes"],
            "resolution": "reconciled" if reconciled else "accepted_as_is",
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3: patient-level code assignment")
    parser.add_argument("--input", default="dev_step1_icd_standard_step2_threshold0.8.csv")
    parser.add_argument("--method", choices=["prevalence", "agentic"], default="prevalence")
    parser.add_argument("--th-prev", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df = df.sort_values(["IDPACIENTE", "FECHA"]).reset_index(drop=True)
    df["note_codes"] = df["note_codes"].apply(json.loads)
    df["COD_CIE_10"] = df["COD_CIE_10"].apply(json.loads)

    groups = list(df.groupby("IDPACIENTE"))
    gold = {pid: sorted({c for codes in g["COD_CIE_10"] for c in codes if c.upper().startswith("F")})
            for pid, g in groups}

    if args.method == "prevalence":
        rows = [{"IDPACIENTE": pid, "final_codes": prevalence(g["note_codes"], args.th_prev)} for pid, g in groups]
        tag = f"prevalence{args.th_prev}"
    else:
        contexts = [build_context(pid, g) for pid, g in groups]
        results = agentic([c for c in contexts if c["codes"]], args.batch_size)
        rows = [{"IDPACIENTE": pid, **results.get(pid, {"final_codes": [], "resolution": "no_candidates"})}
                for pid, _ in groups]
        tag = "agentic"

    out = pd.DataFrame(rows)
    out.insert(1, "gold_codes", out["IDPACIENTE"].map(gold))
    for col in ("gold_codes", "final_codes", "original_codes"):
        if col in out:
            out[col] = out[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

    output = f"{Path(args.input).with_suffix('')}_step3_{tag}.csv"
    out.to_csv(output, index=False)
    print(f"Saved → {output} ({len(out)} patients)")


if __name__ == "__main__":
    main()
