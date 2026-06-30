#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_scf_group_significance.py \
  --config configs/scf_group_significance.yaml \
  2>&1 | tee outputs/scf_group_significance.log

echo
echo "Report:"
echo "  results/strict_coldstart/final_scf/group_significance/group_significance_report.md"
