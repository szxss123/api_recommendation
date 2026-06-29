#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_strict_leakage_audit.sh
bash scripts/run_new_baseline_significance.sh

echo
echo "Completed:"
echo "  outputs/strict_leakage_audit/leakage_audit_report.md"
echo "  outputs/strict_baseline_significance/new_baseline_significance_report.md"
