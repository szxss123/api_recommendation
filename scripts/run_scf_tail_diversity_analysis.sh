#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_scf_tail_diversity.py \
  --config configs/scf_tail_diversity.yaml \
  2>&1 | tee outputs/scf_tail_diversity_analysis.log

echo
echo "Report:"
echo "  outputs/scf_tail_diversity/scf_tail_diversity_report.md"
