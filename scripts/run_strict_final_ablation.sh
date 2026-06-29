#!/usr/bin/env bash
set -euo pipefail

# Run the final strict cold-start ablation on the same test split:
#   1) Graph-only
#   2) BGE-only
#   3) Graph+BGE z-score fusion (lambda=0.25)
#
# Usage:
#   cd ~/code/api_recommendation
#   bash scripts/run_strict_final_ablation.sh
#
# Optional:
#   FORCE=1 bash scripts/run_strict_final_ablation.sh
#   DEVICE=cuda bash scripts/run_strict_final_ablation.sh

DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/final_strict_ablation}"

ANALYZER="scripts/analyze_strict_coldstart.py"
SUMMARY_SCRIPT="scripts/summarize_strict_final_results.py"

if [[ ! -f "${ANALYZER}" ]]; then
  echo "[ERROR] Missing ${ANALYZER}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

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
  local method="$1"
  local seed="$2"
  local config="$3"
  local checkpoint="$4"
  local out_dir="${OUTPUT_ROOT}/${method}/seed${seed}"

  if [[ -f "${out_dir}/overall_metrics.csv" && "${FORCE}" != "1" ]]; then
    echo "[SKIP] ${method} seed${seed}: ${out_dir}/overall_metrics.csv already exists"
    return
  fi

  mkdir -p "${out_dir}"

  local mode=""
  local lambda=""
  case "${method}" in
    graph_only)
      mode="zscore"
      lambda="0.0"
      ;;
    bge_only)
      mode="bge_only"
      lambda="1.0"
      ;;
    graph_bge_zscore)
      mode="zscore"
      lambda="0.25"
      ;;
    *)
      echo "[ERROR] Unknown method: ${method}"
      exit 1
      ;;
  esac

  echo "=================================================================="
  echo "Running ${method}, seed=${seed}, mode=${mode}, lambda=${lambda}"
  echo "Config: ${config}"
  echo "Checkpoint: ${checkpoint}"
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
    2>&1 | tee "${OUTPUT_ROOT}/logs/${method}_seed${seed}.log"
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

  run_one graph_only "${seed}" "${config}" "${checkpoint}"
  run_one bge_only "${seed}" "${config}" "${checkpoint}"

  # Reuse the three main-model results if they were already produced.
  old_main="outputs/analysis/strict_test_zscore_lam0.25_seed${seed}"
  new_main="${OUTPUT_ROOT}/graph_bge_zscore/seed${seed}"
  if [[ -f "${old_main}/overall_metrics.csv" && ! -f "${new_main}/overall_metrics.csv" && "${FORCE}" != "1" ]]; then
    mkdir -p "$(dirname "${new_main}")"
    cp -a "${old_main}" "${new_main}"
    echo "[REUSE] ${old_main} -> ${new_main}"
  else
    run_one graph_bge_zscore "${seed}" "${config}" "${checkpoint}"
  fi
done

python -u "${SUMMARY_SCRIPT}" \
  --root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_ROOT}/summary"

echo
echo "Final summary:"
echo "  ${OUTPUT_ROOT}/summary/ablation_table.csv"
echo "  ${OUTPUT_ROOT}/summary/overall_mean_std.csv"
echo "  ${OUTPUT_ROOT}/summary/group_mean_std.csv"
echo "  ${OUTPUT_ROOT}/summary/final_report.md"
