#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/analyze_new_baseline_significance.py \
  --baseline_per_mashup \
    outputs/strict_baselines/test/new_baseline_per_mashup_metrics.csv \
  --graph_bge_source \
    outputs/statistical_analysis/runs \
  --output_dir \
    outputs/strict_baseline_significance \
  --bootstrap_samples 10000 \
  --random_seed 2026 \
  2>&1 | tee outputs/strict_baseline_significance.log
