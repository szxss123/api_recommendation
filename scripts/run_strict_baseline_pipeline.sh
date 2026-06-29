#!/usr/bin/env bash
set -euo pipefail

# Run from the project root.
#
# Stage 1: select collaborative baseline hyperparameters on validation.
# Stage 2: run the fixed baselines on the test set and build the main table.

CONFIG="${CONFIG:-configs/strict_baselines.yaml}"
VALIDATION_DIR="${VALIDATION_DIR:-outputs/strict_baselines/validation}"
TEST_DIR="${TEST_DIR:-outputs/strict_baselines/test}"

python -u scripts/search_strict_baseline_hparams.py \
  --config "${CONFIG}" \
  --output_dir "${VALIDATION_DIR}"

python -u scripts/run_strict_baseline_main.py \
  --config "${CONFIG}" \
  --selected "${VALIDATION_DIR}/selected_hyperparameters.json" \
  --output_dir "${TEST_DIR}"

echo
echo "Finished."
echo "Main report: ${TEST_DIR}/baseline_main_table.md"
echo "CSV table:   ${TEST_DIR}/baseline_main_table_formatted.csv"
