# Multi-agent LLM prompting pipeline (Qwen2.5-7B-Instruct)

An alternative to the fine-tuned transformer classifiers in
[`../transformer_classification/`](../transformer_classification/README.md)
(see the [top-level README](../README.md)): instead of training a model, this
pipeline prompts **Qwen2.5-7B-Instruct** in a chain of roles ("agents") to
extract diagnoses from a note, map them to ICD-10 Chapter V ("F") codes, and
aggregate codes across a patient's notes — optionally with further LLM agents
reviewing and reconciling disagreements.

**The exact prompt text for every agent is included directly in the code**
(also reported in the paper's appendix) as the constants below — no
placeholders to fill in:

| Constant                              | File                    | Agent                           |
|---------------------------------------|-------------------------|----------------------------------|
| `EXTRACTOR_SYSTEM_PROMPT` / `_USER_PROMPT` | `step1_extraction.py`   | Clinical Information Extractor   |
| `PASSAGE_SYSTEM_PROMPT` / `_USER_PROMPT`   | `step1_extraction.py`   | Relevant Passage Retrieval Agent |
| `PHYSICIAN_SYSTEM_PROMPT`             | `step3_patient_codes.py`| Physician (reviewer)            |
| `PATIENT_SYSTEM_PROMPT`               | `step3_patient_codes.py`| Patient Safety Advocate (reviewer) |
| `RECONCILER_SYSTEM_PROMPT`            | `step3_patient_codes.py`| Code Reconciler                 |
| `REVIEWER_USER_PROMPT` / `RECONCILER_USER_PROMPT` | `step3_patient_codes.py` | shared by the three agents above |

## Pipeline overview

```
step1_extraction.py  →  step2_note_codes.py  →  step3_patient_codes.py
 (per-note codes)        (filter per note)        (per-patient codes)
```

### Step 1 — `step1_extraction.py`: extraction + code retrieval

1. The **Clinical Information Extractor** agent reads each note and returns a
   structured list of diagnoses (name + status) as JSON.
2. The first extracted diagnosis is matched against a code reference, using
   one of two frameworks (`--reference`):
   - **`icd_standard`**: a CSV table of ICD-10 code/description pairs
     (`ICDstandard.csv`-like file, one `code` column and one `description`
     column — column names are auto-detected). Matching is TF-IDF
     (character n-grams) cosine similarity between the diagnosis name and
     each code's description; Chapter V ("F"-prefixed) codes only.
   - **`guidelines`**: a free-text file of ICD-10 Chapter V coding guidelines
     (`guidelines.txt`-like file), chunked into passages and retrieved by
     TF-IDF (word n-grams); a **Relevant Passage Retrieval Agent** then reads
     the top passages and picks the final code.

```bash
cd llm_prompting
python step1_extraction.py --input dev.csv --reference ICDstandard.csv
# or
python step1_extraction.py --input dev.csv --reference guidelines.txt
```

Input CSV columns: `IDPACIENTE`, `FECHA`, `ENTRADA_ANONIMIZADA` (or `text_sp`),
`COD_CIE_10` (gold codes, stored as a Python-list-like string, e.g.
`"['F32.1', 'Z00']"`). Output: `<input>_step1_<framework>.csv`, one row per
note with a `top_codes` column (JSON list of `{"code", "similarity"}`, sorted
by similarity).

Diagnosis extraction (step 1.1) does not depend on the reference and is
cached to `<input>_extractions.csv`, so switching `--reference` on the same
input does not re-run the LLM extraction.

### Step 2 — `step2_note_codes.py`: note-level code assignment

Turns the ranked `top_codes` from step 1 into a definite set of codes per
note (`--regime`):

- `top1`: keep only the single highest-similarity code.
- `threshold`: keep every code with similarity above `--th-sim` (default 0.8).

```bash
python step2_note_codes.py --input dev_step1_icd_standard.csv --regime threshold --th-sim 0.8
```

### Step 3 — `step3_patient_codes.py`: patient-level code assignment

Aggregates a patient's note-level codes into a final per-patient code list
(`--method`):

- **`prevalence`**: keep a code if it appears in at least `--th-prev` (0-1)
  fraction of the patient's notes (`0.0` = union across all notes).
- **`agentic`**: three further LLM agents review the candidate codes against
  the patient's note text — a **Physician** and a **Patient Safety Advocate**
  independently flag agreement/disagreement with the candidate list; if
  either disagrees, a **Code Reconciler** agent decides the final list.

```bash
python step3_patient_codes.py --input dev_step1_icd_standard_step2_threshold0.8.csv --method prevalence --th-prev 0.0
# or
python step3_patient_codes.py --input dev_step1_icd_standard_step2_threshold0.8.csv --method agentic
```

Output: `<input>_step3_<method>.csv`, one row per patient, with `gold_codes`
(ground-truth union of "F" codes from `COD_CIE_10`) and `final_codes` — ready
to compare with e.g. `sklearn.metrics.precision_recall_fscore_support` or the
evaluation scripts in [`../transformer_classification/`](../transformer_classification/)
(format the two columns as `note_codes`-style JSON lists and adapt an
evaluation script's loader, or compute metrics directly from
`gold_codes`/`final_codes`).

## `llm_utils.py`

Shared helpers used by steps 1 and 3:

- `load_model()`: loads `Qwen/Qwen2.5-7B-Instruct` once (bfloat16,
  `device_map="auto"`) and caches it at module level.
- `run_batch(system_prompt, user_messages, batch_size, max_length,
  max_new_tokens)`: applies the chat template, batches generation, and
  returns the decoded responses (greedy decoding, `do_sample=False`).
- `parse_json(text)`: best-effort JSON parsing of a model response (falls
  back to extracting the first `{...}` block, then to an error dict).

Requires a CUDA GPU (bfloat16 + `device_map="auto"`); `Qwen2.5-7B-Instruct`
needs roughly 16GB of GPU memory at bfloat16.

## Installation

Use a **separate environment** from the fine-tuned-transformer pipeline — see
[`requirements-prompting.txt`](requirements-prompting.txt):

```bash
conda create -n psychiatry_prompting python=3.10
conda activate psychiatry_prompting
pip install torch --index-url <your-cuda-specific-index>
pip install -r requirements-prompting.txt
```
