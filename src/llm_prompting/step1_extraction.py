"""
Step 1 — Clinical information extraction + code retrieval.

1. The Clinical Information Extractor Agent extracts diagnoses from each note.
2. The first extracted diagnosis is matched against a reference, either:
   - icd_standard : ICD-10 code/description table (ICDstandard.csv), TF-IDF char n-grams
   - guidelines   : Chapter V coding guidelines (guidelines.txt), TF-IDF passage retrieval
                    + Relevant Passage Retrieval Agent that picks the code

Usage:
    python step1_extraction.py --input dev.csv --reference ICDstandard.csv
    python step1_extraction.py --input dev.csv --reference guidelines.txt

Input columns: IDPACIENTE, FECHA, ENTRADA_ANONIMIZADA (or text_sp), COD_CIE_10
Output: <input>_step1_<icd_standard|guidelines>.csv, one row per note with a
`top_codes` column: JSON list of {"code", "similarity"} sorted by similarity.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_utils import run_batch, parse_json

EXTRACTOR_SYSTEM_PROMPT = """You are a clinical information extraction assistant. Your task is to extract diagnostic information from a single clinical note that belongs to a longitudinal patient record.

IMPORTANT CONTEXT: You will only see ONE note at a time. Each note is part of a larger series. Extract only what is explicitly stated or directly inferable from THIS note. Do not invent, assume, or carry over information not present in the current text.

Output your extraction strictly in the following JSON structure:

{"diagnoses": [
    {
        "name": "<diagnosis name>",
        "icd_hint": "<ICD-10 code-related information if mentioned or inferrable, else null>",
        "status": "<new | ongoing | resolved | suspected | ruled out>"
    }
    ]
}

Rules:
- Use null for any field not found in the note. Do not guess. Fill all the fields, even if no information is found, but use null for those.
- Output only valid JSON. Begin your response with {" and include no prose before or after the JSON block."""

EXTRACTOR_USER_PROMPT = """Extract diagnose information from the following clinical note. Follow the system instructions exactly.

--- BEGIN NOTE ---
{note}
--- END NOTE ---"""  # fields: {note}

PASSAGE_SYSTEM_PROMPT = """You are an expert clinical coding assistant specializing in the ICD-10 classification system.
Your job is to analyze the provided patient diagnosis record and use the attached reference text passages to determine the most exact, accurate ICD-10 code(s).

Guidelines:
- Rely strictly on the provided context passages. Do not invent codes.
- Select codes with the highest level of detail/specificity possible (e.g., matching subcategories down to the decimal point if supported by text).
- Respond in a structured JSON markdown format containing the final assigned code, description, and clinical reasoning.
"""

PASSAGE_USER_PROMPT = """Patient Diagnosis to Code: "{diagnosis_name}" (Status: {status})

Reference Passages from ICD-10 Documentation:
{retrieved_context}

Based on the diagnosis and the reference text above, provide the most accurate ICD-10 code assignment.
Return your response strictly in JSON format as follows:
{{
    "assigned_code": "CODE_HERE",
    "description": "DESCRIPTION_HERE",
    "reasoning": "Brief explanation mapping clinical details to the code."
}}
"""  # fields: {diagnosis_name}, {status}, {retrieved_context}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Clinical information extraction
# ─────────────────────────────────────────────────────────────────────────────

def parse_icd_codes(raw) -> list[str]:
    """COD_CIE_10 is stored as a string list, e.g. "['F32.1', 'Z00']"."""
    if pd.isna(raw) or str(raw).strip() in ("", "nan", "None"):
        return []
    raw = str(raw).strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return [c.strip() for c in re.split(r"[;,\s]+", raw) if c.strip()]


def extract_diagnoses(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    text_col = "text_sp" if "text_sp" in df.columns else "ENTRADA_ANONIMIZADA"
    notes = df[text_col].astype(str).tolist()
    responses = run_batch(
        EXTRACTOR_SYSTEM_PROMPT,
        [EXTRACTOR_USER_PROMPT.format(note=n) for n in notes],
        batch_size=batch_size, max_length=2048,
    )
    return pd.DataFrame({
        "IDPACIENTE": df["IDPACIENTE"],
        "FECHA": df["FECHA"],
        "COD_CIE_10": df["COD_CIE_10"].apply(parse_icd_codes).apply(json.dumps, ensure_ascii=False),
        "ENTRADA_ANONIMIZADA": notes,
        "extraction": [json.dumps(parse_json(r), ensure_ascii=False) for r in responses],
    })


def first_diagnosis(extraction: str):
    """Return (name, status) of the first non-null diagnosis, or (None, None)."""
    try:
        data = json.loads(extraction)
    except (json.JSONDecodeError, TypeError):
        return None, None
    for diag in (data or {}).get("diagnoses") or []:
        if not isinstance(diag, dict):
            continue
        name = diag.get("name")
        if name is None or str(name).strip().lower() in ("", "null"):
            continue
        status = diag.get("status")
        status = "unknown" if status is None or str(status).lower() == "null" else str(status)
        return str(name).strip(), status
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 2a. ICD standard definitions (structured table)
# ─────────────────────────────────────────────────────────────────────────────

class ICDStandardRetriever:
    def __init__(self, path: str):
        df = pd.read_csv(path)
        code_col = next((c for c in df.columns if "code" in c.lower()), df.columns[0])
        desc_col = next((c for c in df.columns if "desc" in c.lower() or "title" in c.lower()), df.columns[1])
        df = df[df[code_col].astype(str).str.upper().str.startswith("F")]  # Chapter V only
        self.codes = df[code_col].astype(str).str.strip().tolist()
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), lowercase=True)
        self.vectors = self.vectorizer.fit_transform(df[desc_col].astype(str).str.strip())
        print(f"Loaded {len(self.codes)} ICD-10 Chapter V codes")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.vectors)[0]
        return [{"code": self.codes[i], "similarity": float(sims[i])}
                for i in np.argsort(sims)[::-1][:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Chapter V guidelines (unstructured text)
# ─────────────────────────────────────────────────────────────────────────────

class GuidelinesRetriever:
    def __init__(self, path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        content = re.sub(r"\n{3,}", "\n\n", Path(path).read_text(encoding="utf-8"))
        self.passages = []
        for p in (p.strip() for p in content.split("\n\n")):
            if not p:
                continue
            if len(p) <= chunk_size:
                self.passages.append(p)
            else:
                words = p.split()
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    self.passages.append(" ".join(words[i:i + chunk_size]))
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
        self.vectors = self.vectorizer.fit_transform(self.passages)
        print(f"Loaded {len(self.passages)} guideline passages")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.vectors)[0]
        return [{"text": self.passages[i], "similarity": float(sims[i])}
                for i in np.argsort(sims)[::-1][:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1: diagnosis extraction + code retrieval")
    parser.add_argument("--input", default="dev.csv")
    parser.add_argument("--reference", default="ICDstandard.csv",
                        help="ICDstandard.csv (ICD standard definitions) or guidelines.txt (Chapter V guidelines)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    framework = "guidelines" if args.reference.endswith(".txt") else "icd_standard"
    stem = Path(args.input).with_suffix("")

    # Extraction does not depend on the reference, so it is cached and reused
    extractions_path = f"{stem}_extractions.csv"
    if Path(extractions_path).exists():
        df = pd.read_csv(extractions_path)
    else:
        df = pd.read_csv(args.input)
        df["FECHA"] = pd.to_datetime(df["FECHA"])
        df = df.sort_values(["IDPACIENTE", "FECHA"]).reset_index(drop=True)
        df = extract_diagnoses(df, args.batch_size)
        df.to_csv(extractions_path, index=False)

    df["diagnosis_name"], df["diagnosis_status"] = zip(*df["extraction"].apply(first_diagnosis))
    has_diag = df["diagnosis_name"].notna()

    if framework == "icd_standard":
        retriever = ICDStandardRetriever(args.reference)
        df["top_codes"] = [retriever.retrieve(n) if ok else [] for n, ok in zip(df["diagnosis_name"], has_diag)]
    else:
        retriever = GuidelinesRetriever(args.reference)
        hits = {i: retriever.retrieve(df.at[i, "diagnosis_name"]) for i in df.index[has_diag]}
        contexts = {i: "\n\n".join(f"--- Passages Block #{j + 1} ---\n{h['text']}" for j, h in enumerate(hs))
                    for i, hs in hits.items()}
        responses = run_batch(
            PASSAGE_SYSTEM_PROMPT,
            [PASSAGE_USER_PROMPT.format(diagnosis_name=df.at[i, "diagnosis_name"],
                                        status=df.at[i, "diagnosis_status"],
                                        retrieved_context=contexts[i]) for i in hits],
            batch_size=2,
        )
        df["top_codes"] = [[] for _ in range(len(df))]
        for i, raw in zip(hits, responses):
            code = parse_json(raw).get("assigned_code")
            if code:
                df.at[i, "top_codes"] = [{"code": str(code).strip(), "similarity": hits[i][0]["similarity"]}]

    df["top_codes"] = df["top_codes"].apply(json.dumps)
    output = f"{stem}_step1_{framework}.csv"
    df.drop(columns=["extraction"]).to_csv(output, index=False)
    print(f"Saved → {output} ({len(df)} notes)")


if __name__ == "__main__":
    main()
