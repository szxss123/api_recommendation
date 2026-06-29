#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_scf_final_significance.py \
  --config configs/scf_final_significance.yaml \
  2>&1 | tee outputs/scf_final_significance.log

echo
echo "Report:"
echo "  outputs/scf_final_significance/scf_final_significance_report.md"
