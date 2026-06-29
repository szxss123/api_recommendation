#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/export_inductive_lightgcn_rankings.py \
  --config configs/scf_tail_diversity.yaml \
  2>&1 | tee outputs/export_inductive_lightgcn_rankings.log
