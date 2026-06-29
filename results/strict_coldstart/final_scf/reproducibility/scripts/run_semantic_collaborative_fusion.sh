#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/run_semantic_collaborative_fusion.py \
  --config configs/strict_semantic_collaborative_fusion.yaml \
  2>&1 | tee outputs/strict_semantic_collaborative_fusion.log

echo
echo "Report:"
echo "  outputs/strict_semantic_collaborative_fusion/semantic_collaborative_fusion_report.md"
