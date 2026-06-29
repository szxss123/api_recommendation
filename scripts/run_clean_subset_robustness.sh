#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_clean_subset_robustness.py \
  --config configs/strict_clean_subset_analysis.yaml \
  2>&1 | tee outputs/strict_clean_subset_analysis.log

echo
echo "Report:"
echo "  outputs/strict_clean_subset_analysis/clean_subset_report.md"
