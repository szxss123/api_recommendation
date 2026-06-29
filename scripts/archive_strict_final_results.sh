#!/usr/bin/env bash
set -euo pipefail

# Archive final strict cold-start outputs, code, configs and a checksum manifest.
#
# Usage:
#   cd ~/code/api_recommendation
#   bash scripts/archive_strict_final_results.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/final_strict_ablation}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archived_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TARGET="${ARCHIVE_ROOT}/strict_coldstart_final_${STAMP}"

mkdir -p "${TARGET}/outputs" "${TARGET}/code" "${TARGET}/configs"

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] Missing output directory: ${OUTPUT_ROOT}"
  exit 1
fi

cp -a "${OUTPUT_ROOT}/." "${TARGET}/outputs/"

for file in \
  scripts/analyze_strict_coldstart.py \
  scripts/run_strict_final_ablation.sh \
  scripts/summarize_strict_final_results.py \
  src/trainers/finetune_trainer_strict_inductive.py
do
  if [[ -f "${file}" ]]; then
    cp "${file}" "${TARGET}/code/"
  fi
done

for file in \
  configs/finetune_mtfm_cold_strict_zscore_seed0.yaml \
  configs/finetune_mtfm_cold_strict_seed0.yaml \
  configs/finetune_mtfm_cold_strict_seed1.yaml \
  configs/finetune_mtfm_cold_strict_seed2.yaml
do
  if [[ -f "${file}" ]]; then
    cp "${file}" "${TARGET}/configs/"
  fi
done

cat > "${TARGET}/CHECKPOINTS.txt" <<'EOF'
Expected checkpoints:
outputs/checkpoints/mtfm_cold_strict_weighted_log_seed0_best.pt
outputs/checkpoints/mtfm_cold_strict_weighted_log_seed1_best.pt
outputs/checkpoints/mtfm_cold_strict_weighted_log_seed2_best.pt

Checkpoint binaries are intentionally not copied into this archive.
EOF

(
  cd "${TARGET}"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt
  find . -type f | sort > MANIFEST.txt
)

tarball="${TARGET}.tar.gz"
tar -czf "${tarball}" -C "${ARCHIVE_ROOT}" "$(basename "${TARGET}")"

echo "Archive directory: ${TARGET}"
echo "Archive tarball:   ${tarball}"
