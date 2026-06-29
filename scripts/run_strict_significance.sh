#!/usr/bin/env bash
set -euo pipefail

# Export per-Mashup metrics for the three ablation methods and run paired
# statistical significance analysis.
#
# Run from the project root:
#   bash scripts/run_strict_significance.sh
#
# Optional:
#   DEVICE=cuda BOOTSTRAP_SAMPLES=20000 bash scripts/run_strict_significance.sh

DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/statistical_analysis/runs}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/statistical_analysis/summary}"
ANALYZER="${ANALYZER:-scripts/analyze_strict_coldstart.py}"

mkdir -p "${OUTPUT_ROOT}" "${SUMMARY_DIR}" \
  "outputs/statistical_analysis/logs"

config_for_seed() {
  local seed="$1"
  if [[ "${seed}" == "0" && -f "configs/finetune_mtfm_cold_strict_zscore_seed0.yaml" ]]; then
    echo "configs/finetune_mtfm_cold_strict_zscore_seed0.yaml"
  else
    echo "configs/finetune_mtfm_cold_strict_seed${seed}.yaml"
  fi
}

checkpoint_for_seed() {
  local seed="$1"
  echo "outputs/checkpoints/mtfm_cold_strict_weighted_log_seed${seed}_best.pt"
}

run_one() {
  local method_dir="$1"
  local mode="$2"
  local lambda="$3"
  local seed="$4"
  local config="$5"
  local checkpoint="$6"
  local out_dir="${OUTPUT_ROOT}/${method_dir}/seed${seed}"

  mkdir -p "${out_dir}"

  echo "=================================================================="
  echo "method=${method_dir} seed=${seed} mode=${mode} lambda=${lambda}"
  echo "=================================================================="

  python -u "${ANALYZER}" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output_dir "${out_dir}" \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --split test \
    --ranking_mode "${mode}" \
    --fusion_lambda "${lambda}" \
    --popularity_penalty 0 \
    2>&1 | tee \
    "outputs/statistical_analysis/logs/${method_dir}_seed${seed}.log"

  test -f "${out_dir}/per_mashup_metrics.csv"
  test -f "${out_dir}/rankings_topk.csv"
}

for seed in 0 1 2
do
  config="$(config_for_seed "${seed}")"
  checkpoint="$(checkpoint_for_seed "${seed}")"

  if [[ ! -f "${config}" ]]; then
    echo "[ERROR] Missing config: ${config}"
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Missing checkpoint: ${checkpoint}"
    exit 1
  fi

  run_one graph_only zscore 0.0 \
    "${seed}" "${config}" "${checkpoint}"

  run_one bge_only bge_only 1.0 \
    "${seed}" "${config}" "${checkpoint}"

  run_one graph_bge_zscore zscore 0.25 \
    "${seed}" "${config}" "${checkpoint}"
done

python -u scripts/analyze_paired_significance.py \
  --root "${OUTPUT_ROOT}" \
  --output_dir "${SUMMARY_DIR}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap_seed 2026 \
  --alternative two-sided

echo
echo "Finished."
echo "Main result: ${SUMMARY_DIR}/significance_report.md"
echo "CSV result:  ${SUMMARY_DIR}/significance_results.csv"
