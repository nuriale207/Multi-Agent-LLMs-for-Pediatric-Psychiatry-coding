#!/bin/bash
# End-to-end example: train -> test -> rebinarize at extra thresholds -> evaluate.
#
# This is a plain bash template, independent of any job scheduler. If you run
# jobs through SLURM (or similar), wrap this script's body in your own
# #SBATCH header and submit it as usual.
#
# Edit the variables below to point at your own data, then run:
#   bash scripts/run_pipeline_example.sh
set -euo pipefail

# ── Variables ─────────────────────────────────────────────────────────────────
TRAINING_DATA_DIR="/path/to/train.csv"
EVAL_DATA_DIR="/path/to/dev.csv"
TEST_DATA_DIR="/path/to/test.csv"

LABELS_START_INDEX=15        # first column index holding one-hot diagnostic codes
TEXT_COLUMN="text"           # column holding the clinical note text
ID_COLUMN="idpaciente"       # column identifying the patient
CSV_SEP=","

MODEL_ID="PlanTL-GOB-ES/bsc-bio-ehr-es"   # swap for yikuan8/Clinical-Longformer,
                                          # mrm8488/longformer-base-4096-spanish, etc.
MODEL_TYPE="bert"
MAX_LEN=512
BATCH_SIZE=16
EPOCHS=25
WARMUP_STEPS=500
GRAD_ACCUM=2

OUTPUT_DIR="./Experiments_data/"
EXPERIMENT_NAME="my_experiment"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Train ─────────────────────────────────────────────────────────────────────
python "$SRC_DIR/Train_transformer.py" \
  --training_data_dir "$TRAINING_DATA_DIR" \
  --eval_data_dir "$EVAL_DATA_DIR" \
  --test_data_dir "$TEST_DATA_DIR" \
  --labels_default_start_index "$LABELS_START_INDEX" \
  --model_id "$MODEL_ID" \
  --model_type "$MODEL_TYPE" \
  --model_path "model_id" \
  --output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --max_len "$MAX_LEN" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --grad_accum "$GRAD_ACCUM" \
  --mode "train" \
  --text_column "$TEXT_COLUMN" \
  --csv_sep "$CSV_SEP"

# ── Test (loads the last checkpoint saved by the training run above) ─────────
TEST_MODEL_PATH=$(ls -d "${OUTPUT_DIR}${EXPERIMENT_NAME}/model/checkpoint-"* | sort -V | tail -1)

python "$SRC_DIR/Train_transformer.py" \
  --training_data_dir "$TRAINING_DATA_DIR" \
  --eval_data_dir "$EVAL_DATA_DIR" \
  --test_data_dir "$TEST_DATA_DIR" \
  --labels_default_start_index "$LABELS_START_INDEX" \
  --model_id "$MODEL_ID" \
  --model_type "$MODEL_TYPE" \
  --model_path "$TEST_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --max_len "$MAX_LEN" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --grad_accum "$GRAD_ACCUM" \
  --mode "test" \
  --text_column "$TEXT_COLUMN" \
  --csv_sep "$CSV_SEP"

TEST_DIR="${OUTPUT_DIR}${EXPERIMENT_NAME}/test"
RAW_PREDS="${TEST_DIR}/predictions_raw.csv"

# ── Re-binarize at additional thresholds ─────────────────────────────────────
python "$SRC_DIR/rebinarize_predictions.py" \
  --raw_predictions_csv "$RAW_PREDS" --threshold 0.4 --output_csv "${TEST_DIR}/predictions_0_4.csv"

python "$SRC_DIR/rebinarize_predictions.py" \
  --raw_predictions_csv "$RAW_PREDS" --threshold 0.6 --output_csv "${TEST_DIR}/predictions_0_6.csv"

# ── Evaluate (agent-compatible metrics, at the default 0.5 threshold) ───────
python "$SRC_DIR/evaluate_transformer_agent_format.py" \
  --test_csv "$TEST_DATA_DIR" \
  --predictions_csv "${TEST_DIR}/predictions.csv" \
  --id_column "$ID_COLUMN" \
  --csv_sep "$CSV_SEP" \
  --labels_start_index "$LABELS_START_INDEX" \
  --output_file "${TEST_DIR}/evaluation_agent_format_0_5.json"

# ── Evaluate aggregated at the patient level (union of codes across notes) ──
python "$SRC_DIR/evaluate_transformer_patient_aggregated.py" \
  --test_csv "$TEST_DATA_DIR" \
  --predictions_csv "${TEST_DIR}/predictions.csv" \
  --csv_sep "$CSV_SEP" \
  --labels_start_index "$LABELS_START_INDEX" \
  --output_file "${TEST_DIR}/evaluation_patient_aggregated_0_5.json"

echo "Done. Results written under ${TEST_DIR}"
