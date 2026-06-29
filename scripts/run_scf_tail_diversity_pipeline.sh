#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_export_inductive_lightgcn_rankings.sh
bash scripts/run_scf_tail_diversity_analysis.sh
