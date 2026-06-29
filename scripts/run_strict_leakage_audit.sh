#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u scripts/audit_strict_coldstart_leakage.py \
  --config configs/strict_leakage_audit.yaml \
  2>&1 | tee outputs/strict_leakage_audit.log
