#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_final_scf_cases.py \
  --config configs/final_scf_case_study.yaml \
  2>&1 | tee outputs/final_scf_case_study.log

echo
echo "Final case-study report:"
echo "  results/strict_coldstart/final_scf/case_studies/case_study_analysis_cn.md"
