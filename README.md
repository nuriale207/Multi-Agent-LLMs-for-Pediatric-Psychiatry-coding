# Multi-agent LLMs and fine-tuned transformers for pediatric psychiatry coding

Code to predict psychiatric ICD-10 ("F" chapter) diagnostic codes from
free-text clinical notes, accompanying the paper:

> _Add full citation here once available (authors, title, venue, year, DOI)._

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

This repository contains **only the code**: it does not include the trained
model weights, the clinical data used in the paper, or the exact LLM prompt
text (reported in the paper's appendix), all of which are either private or
live in the manuscript itself — see [Data availability](#data-availability)
and [`src/llm_prompting/README.md`](src/llm_prompting/README.md). What it
does provide is two complete, dataset-agnostic pipelines that reproduce the
paper's two approaches on **any** dataset that follows the same CSV format:

- **`src/`** — fine-tuned transformer classifiers (BERT / Longformer):
  train, test, re-binarize, and evaluate (micro/macro F1, patient
  aggregation, demographic breakdown, note-voting). Documented below.
- **`src/llm_prompting/`** — a multi-agent prompting pipeline built on
  **Qwen2.5-7B-Instruct**: an extraction agent pulls diagnoses out of each
  note, retrieval matches them to ICD-10 codes, and (optionally) further
  agents review and reconcile the patient-level code list. See
  [`src/llm_prompting/README.md`](src/llm_prompting/README.md).

## Repository structure

```
.
├── src/
│   ├── Train_transformer.py                     # train / test a transformer classifier
│   ├── rebinarize_predictions.py                 # re-binarize raw probabilities at a new threshold
│   ├── evaluate_transformer_agent_format.py      # micro/macro P/R/F1, full & main (3-char) code level
│   ├── evaluate_transformer_patient_aggregated.py# same metrics, notes aggregated per patient
│   ├── evaluate_by_demographics.py               # per-patient F1 broken down by sex / age bin
│   ├── evaluate_voting_label_f1.py                # per-label F1 under a note-voting strategy
│   ├── evaluate_voting_threshold.py               # per-patient F1 under a note-voting strategy
│   ├── utils/                                     # data loading/tokenization used by Train_transformer.py
│   └── llm_prompting/                             # Qwen2.5 multi-agent prompting pipeline (see its own README)
│       ├── llm_utils.py
│       ├── step1_extraction.py
│       ├── step2_note_codes.py
│       ├── step3_patient_codes.py
│       └── README.md
├── configs/
│   └── experiments.example.json                  # template for the batch-mode evaluation scripts
├── scripts/
│   └── run_pipeline_example.sh                   # end-to-end example: train -> test -> evaluate
├── requirements.txt                               # fine-tuned transformer pipeline (src/)
├── requirements-prompting.txt                     # Qwen2.5 prompting pipeline (src/llm_prompting/)
└── LICENSE
```

The rest of this README documents the **fine-tuned transformer pipeline**
(`src/`). For the **Qwen2.5 prompting pipeline**, see
[`src/llm_prompting/README.md`](src/llm_prompting/README.md) — it uses a
separate environment (`requirements-prompting.txt`) since Qwen2.5 needs a
much newer `transformers` than the fine-tuning pipeline.

## Installation

```bash
conda create -n psychiatry_classification python=3.9
conda activate psychiatry_classification

# Install PyTorch separately, matching your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch --index-url <your-cuda-specific-index>

pip install -r requirements.txt
```

The pinned versions in `requirements.txt` match the environment used to
produce the paper's results. Newer versions of `transformers`/`pandas` may
require minor adjustments.

## Expected data format

All scripts read CSV files with:

- A **text column** (name configurable via `--text_column`, default `text`)
  containing the clinical note.
- A block of **one-hot label columns**, one per diagnostic code, starting at
  a configurable column index (`--labels_default_start_index` /
  `--labels_start_index`, default `15` to match the paper's layout — adjust
  it to wherever your label columns start). Column names are the diagnostic
  codes themselves (e.g. `F32.1`); values are `0`/`1`.
- Codes are expected to be psychiatric ICD-10 "F" codes; the evaluation
  scripts normalize them (uppercase, insert the decimal point if missing) and
  can optionally truncate to the 3-character "main code" (e.g. `F32.1` →
  `F32`).

The demographic/voting analysis scripts (`evaluate_by_demographics.py`,
`evaluate_voting_label_f1.py`, `evaluate_voting_threshold.py`) additionally
expect these exact column names (case-insensitive):

| Column       | Meaning                                   |
|--------------|--------------------------------------------|
| `idpaciente` | Patient identifier, used to group notes    |
| `sexo`       | Patient sex (any non-empty value is a group) |
| `age`        | Patient age in years (binned as `<8`, `8-12`, `13-17`, `18+`) |

If your dataset uses different column names for these three fields, rename
the columns before running those scripts (the core training/evaluation
scripts do not need them).

## Data availability

The clinical notes used in the paper come from two restricted-access
sources and are **not** included in this repository:

- Psychiatric clinical notes for minors from a public health system
  (Basque Health Service / Osakidetza), subject to a data use agreement and
  institutional/ethics approval. Not redistributable.
- Discharge notes derived from **MIMIC-III**, available only to credentialed
  researchers via [PhysioNet](https://physionet.org/).

To replicate the paper's experiments, you need access to one of these
(or an equivalent) dataset, reformatted to match the
[expected data format](#expected-data-format) above. The code itself works
with any dataset that follows that format.

The exact system/user prompts used by the Qwen2.5 agents in
`src/llm_prompting/` are reported in the paper's appendix rather than
duplicated here; see [`src/llm_prompting/README.md`](src/llm_prompting/README.md)
for which placeholder constant each one fills in.

## Usage

### 1. Train a model

```bash
python src/Train_transformer.py \
  --training_data_dir /path/to/train.csv \
  --eval_data_dir /path/to/dev.csv \
  --test_data_dir /path/to/test.csv \
  --labels_default_start_index 15 \
  --model_id PlanTL-GOB-ES/bsc-bio-ehr-es \
  --model_type bert \
  --model_path model_id \
  --output_dir ./Experiments_data/ \
  --experiment_name my_experiment \
  --max_len 512 \
  --batch_size 16 \
  --epochs 25 \
  --warmup_steps 500 \
  --grad_accum 2 \
  --mode train \
  --text_column text \
  --csv_sep ","
```

### 2. Test / generate predictions

Same command with `--mode test` and `--model_path` pointing at the saved
checkpoint (`<output_dir>/<experiment_name>/model/checkpoint-<step>`).
Writes `predictions.csv` (binarized at 0.5), `predictions_raw.csv` (raw
sigmoid probabilities) and `evaluation.txt` under
`<output_dir>/<experiment_name>/test/`.

### 3. Re-binarize at other thresholds

```bash
python src/rebinarize_predictions.py \
  --raw_predictions_csv Experiments_data/my_experiment/test/predictions_raw.csv \
  --threshold 0.4 \
  --output_csv Experiments_data/my_experiment/test/predictions_0_4.csv
```

### 4. Compute metrics

```bash
# Micro/macro P/R/F1 at full-code and main-code level
python src/evaluate_transformer_agent_format.py \
  --test_csv /path/to/test.csv \
  --predictions_csv Experiments_data/my_experiment/test/predictions.csv \
  --id_column idpaciente \
  --labels_start_index 15 \
  --output_file Experiments_data/my_experiment/test/evaluation_agent_format_0_5.json

# Same metrics, notes aggregated per patient (union of predicted codes)
python src/evaluate_transformer_patient_aggregated.py \
  --test_csv /path/to/test.csv \
  --predictions_csv Experiments_data/my_experiment/test/predictions.csv \
  --labels_start_index 15 \
  --output_file Experiments_data/my_experiment/test/evaluation_patient_aggregated_0_5.json

# Per-patient F1 broken down by sex / age bin
python src/evaluate_by_demographics.py \
  --test_csv /path/to/test.csv \
  --predictions_csv Experiments_data/my_experiment/test/predictions.csv \
  --labels_start_index 15 \
  --output_file Experiments_data/my_experiment/test/demographics_0_5.json
```

### 5. Batch analysis across several experiments/thresholds

`evaluate_by_demographics.py --batch`, `evaluate_voting_label_f1.py` and
`evaluate_voting_threshold.py` operate over a list of experiments described
in a JSON config, instead of a single CSV pair. Copy
`configs/experiments.example.json`, point it at your own experiment
directories and test CSVs, then run e.g.:

```bash
python src/evaluate_by_demographics.py --batch --experiments_config configs/experiments.json
python src/evaluate_voting_label_f1.py --experiments_config configs/experiments.json
python src/evaluate_voting_threshold.py --experiments_config configs/experiments.json
```

Each experiment entry's `test_csv` must be a **note-level** CSV (several rows
per patient) for the voting scripts to make sense; `evaluate_by_demographics.py`
also accepts patient-level CSVs via `"patient_level": true`.

See `scripts/run_pipeline_example.sh` for a full end-to-end example (train →
test → rebinarize → evaluate) with plain bash variables. If you submit jobs
through SLURM or another scheduler, wrap that script's body in your own job
header.

## Paper experiment configurations

The four experiments reported in the paper used the following configurations
(swap `--training_data_dir`/`--eval_data_dir`/`--test_data_dir` for your own
data in the same format):

| Experiment              | `--model_id`                                  | `--model_type` | `--text_column`      | granularity | `--max_len` | `--batch_size` | `--epochs` | `--warmup_steps` | `--grad_accum` |
|-------------------------|------------------------------------------------|----------------|------------------------|-------------|------------:|---------------:|-----------:|-----------------:|---------------:|
| BERT (clinical, 25 ep)  | `PlanTL-GOB-ES/bsc-bio-ehr-es`                | `bert`         | note text              | per-note    | 512         | 16             | 25         | 500               | 2              |
| BERT (clinical, +15 ep) | `PlanTL-GOB-ES/bsc-bio-ehr-es`                | `bert`         | note text              | per-note    | 512         | 16             | 15*        | 0*                | 2              |
| Clinical-Longformer     | `yikuan8/Clinical-Longformer`                 | `longformer`   | note text              | per-patient | 4096        | 2              | 50         | 100               | 8              |
| Spanish Longformer      | `mrm8488/longformer-base-4096-spanish`        | `longformer`   | note text              | per-patient | 4096        | 2              | 50         | 100               | 8              |

\* The second BERT run continues training from the first run's checkpoint
(25 epochs already trained, 15 more on top, no warmup since the optimizer
state is already warmed up).

`--labels_default_start_index` was `15` in all four experiments (adjust to
match your own CSV layout). Evaluation was run at binarization thresholds
0.4, 0.5 and 0.6 (see `rebinarize_predictions.py`).

## License

Released under the [MIT License](LICENSE).
