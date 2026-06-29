#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/szx2025/code/api_recommendation}"
DEST="$PROJECT_ROOT/results/strict_coldstart/final_scf"

test -d "$DEST" || {
  echo "ERROR: frozen directory not found: $DEST"
  exit 1
}

cd "$DEST"
sha256sum -c metadata/SHA256SUMS.txt

python - <<'PY'
from pathlib import Path
import json

root = Path(".")
spec = json.loads(
    (root / "metadata" / "final_model_spec.json").read_text(encoding="utf-8")
)

assert spec["final_method_name"] == "SCF-LightGCN+BGE"
assert spec["lightgcn_weight"] == 0.35
assert spec["bge_weight"] == 0.65
assert spec["popularity_weight"] == 0.0
assert spec["test_tuning_prohibited"] is True

print("Model specification: PASSED")
PY

echo "Frozen result verification: PASSED"
