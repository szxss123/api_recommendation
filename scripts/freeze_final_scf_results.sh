#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/szx2025/code/api_recommendation}"
cd "$PROJECT_ROOT"

DEST="results/strict_coldstart/final_scf"

echo "[1/8] Checking project root..."
test -d .git || {
  echo "ERROR: $PROJECT_ROOT is not a Git repository."
  exit 1
}

required_files=(
  "configs/strict_semantic_collaborative_fusion.yaml"
  "configs/scf_final_significance.yaml"
  "configs/strict_clean_subset_analysis.yaml"
  "configs/scf_tail_diversity.yaml"

  "scripts/run_semantic_collaborative_fusion.py"
  "scripts/run_semantic_collaborative_fusion.sh"
  "scripts/analyze_scf_final_significance.py"
  "scripts/run_scf_final_significance.sh"
  "scripts/analyze_clean_subset_robustness.py"
  "scripts/run_clean_subset_robustness.sh"
  "scripts/analyze_scf_tail_diversity.py"
  "scripts/run_scf_tail_diversity_analysis.sh"

  "outputs/strict_baselines/test/baseline_main_table.md"
  "outputs/strict_baselines/test/baseline_results_by_seed.csv"

  "outputs/strict_semantic_collaborative_fusion/selected_fusion_weights.json"
  "outputs/strict_semantic_collaborative_fusion/fusion_validation_leaderboard.csv"
  "outputs/strict_semantic_collaborative_fusion/scf_results_by_seed.csv"
  "outputs/strict_semantic_collaborative_fusion/scf_per_mashup_metrics.csv"
  "outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed0.csv"
  "outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed1.csv"
  "outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed2.csv"
  "outputs/strict_semantic_collaborative_fusion/semantic_collaborative_fusion_report.md"

  "outputs/scf_final_significance/scf_final_significance_results.csv"
  "outputs/scf_final_significance/scf_final_significance_report.md"
  "outputs/scf_final_significance/scf_final_significance_summary.json"

  "outputs/strict_clean_subset_analysis/subset_definitions.csv"
  "outputs/strict_clean_subset_analysis/clean_subset_main_table_mean_std.csv"
  "outputs/strict_clean_subset_analysis/clean_subset_significance.csv"
  "outputs/strict_clean_subset_analysis/clean_subset_report.md"

  "outputs/scf_tail_diversity/diversity_metrics_mean_std.csv"
  "outputs/scf_tail_diversity/group_exposure_mean_std.csv"
  "outputs/scf_tail_diversity/group_accuracy_mean_std.csv"
  "outputs/scf_tail_diversity/accuracy_diversity_tradeoff_summary.csv"
  "outputs/scf_tail_diversity/scf_tail_diversity_report.md"

  "outputs/strict_leakage_audit/audit_summary.json"
  "outputs/strict_leakage_audit/leakage_audit_report.md"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "MISSING: $file"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "ERROR: Required files are missing. Freeze aborted."
  exit 1
fi

echo "[2/8] Checking final diversity report correction..."
if grep -q "Unseen recall remaining zero" \
  outputs/scf_tail_diversity/scf_tail_diversity_report.md; then
  echo "ERROR: The diversity report still contains the obsolete Unseen-recall statement."
  echo "Replace it with the corrected report before freezing."
  exit 1
fi

echo "[3/8] Recreating frozen result directory..."
rm -rf "$DEST"
mkdir -p \
  "$DEST/model" \
  "$DEST/baseline" \
  "$DEST/main" \
  "$DEST/significance" \
  "$DEST/robustness" \
  "$DEST/diversity" \
  "$DEST/audit" \
  "$DEST/reproducibility/configs" \
  "$DEST/reproducibility/scripts" \
  "$DEST/metadata"

echo "[4/8] Copying final model and results..."

cp \
  outputs/strict_semantic_collaborative_fusion/selected_fusion_weights.json \
  outputs/strict_semantic_collaborative_fusion/fusion_validation_leaderboard.csv \
  "$DEST/model/"

cp \
  outputs/strict_baselines/test/baseline_main_table.md \
  outputs/strict_baselines/test/baseline_results_by_seed.csv \
  "$DEST/baseline/"

cp \
  outputs/strict_semantic_collaborative_fusion/scf_results_by_seed.csv \
  outputs/strict_semantic_collaborative_fusion/scf_per_mashup_metrics.csv \
  outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed0.csv \
  outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed1.csv \
  outputs/strict_semantic_collaborative_fusion/scf_rankings_top10_seed2.csv \
  outputs/strict_semantic_collaborative_fusion/semantic_collaborative_fusion_report.md \
  "$DEST/main/"

cp \
  outputs/scf_final_significance/scf_final_significance_results.csv \
  outputs/scf_final_significance/scf_final_significance_report.md \
  outputs/scf_final_significance/scf_final_significance_summary.json \
  "$DEST/significance/"

cp \
  outputs/strict_clean_subset_analysis/subset_definitions.csv \
  outputs/strict_clean_subset_analysis/clean_subset_main_table_mean_std.csv \
  outputs/strict_clean_subset_analysis/clean_subset_significance.csv \
  outputs/strict_clean_subset_analysis/clean_subset_report.md \
  "$DEST/robustness/"

cp \
  outputs/scf_tail_diversity/diversity_metrics_mean_std.csv \
  outputs/scf_tail_diversity/group_exposure_mean_std.csv \
  outputs/scf_tail_diversity/group_accuracy_mean_std.csv \
  outputs/scf_tail_diversity/accuracy_diversity_tradeoff_summary.csv \
  outputs/scf_tail_diversity/scf_tail_diversity_report.md \
  "$DEST/diversity/"

cp \
  outputs/strict_leakage_audit/audit_summary.json \
  outputs/strict_leakage_audit/leakage_audit_report.md \
  "$DEST/audit/"

cp \
  configs/strict_semantic_collaborative_fusion.yaml \
  configs/scf_final_significance.yaml \
  configs/strict_clean_subset_analysis.yaml \
  configs/scf_tail_diversity.yaml \
  "$DEST/reproducibility/configs/"

cp \
  scripts/run_semantic_collaborative_fusion.py \
  scripts/run_semantic_collaborative_fusion.sh \
  scripts/analyze_scf_final_significance.py \
  scripts/run_scf_final_significance.sh \
  scripts/analyze_clean_subset_robustness.py \
  scripts/run_clean_subset_robustness.sh \
  scripts/analyze_scf_tail_diversity.py \
  scripts/run_scf_tail_diversity_analysis.sh \
  "$DEST/reproducibility/scripts/"

echo "[5/8] Writing immutable model specification..."
python - <<'PY'
from pathlib import Path
import json
from datetime import datetime, timezone

dest = Path("results/strict_coldstart/final_scf")
weights_path = dest / "model" / "selected_fusion_weights.json"
weights = json.loads(weights_path.read_text(encoding="utf-8"))

spec = {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "final_method_name": "SCF-LightGCN+BGE",
    "primary_task": "strict new-Mashup cold-start API recommendation",
    "secondary_observation": (
        "partial zero-shot capability for APIs unseen in training interactions"
    ),
    "score_formula": (
        "0.35 * z(Inductive-LightGCN score) + "
        "0.65 * z(BGE Mashup-API score)"
    ),
    "lightgcn_weight": 0.35,
    "bge_weight": 0.65,
    "popularity_weight": 0.0,
    "normalization": "per-Mashup row-wise z-score over the full API catalog",
    "weight_selection_split": "validation only",
    "weight_selection_seed": 0,
    "weight_selection_rule": "NDCG@10 > MAP@10 > Recall@10",
    "test_seeds": [0, 1, 2],
    "test_tuning_prohibited": True,
    "source_selected_weights": weights,
    "reporting_constraints": [
        "Do not claim significant superiority over Inductive LightGCN on the strict-clean subset.",
        "Describe Unseen APIs as unseen in training interactions, not necessarily unseen to the pretrained text encoder.",
        "Do not attribute SCF gains to the heterogeneous graph branch; the final SCF contains LightGCN and BGE only."
    ]
}

(dest / "metadata" / "final_model_spec.json").write_text(
    json.dumps(spec, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

echo "[6/8] Recording Git and software environment..."
{
  echo "frozen_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo
  echo "[git status --short]"
  git status --short
} > "$DEST/metadata/git_state.txt"

python - <<'PY' > "$DEST/metadata/environment.txt"
import platform
import sys

print("python:", sys.version.replace("\n", " "))
print("platform:", platform.platform())

for name in ["torch", "numpy", "pandas", "scipy", "sklearn", "yaml"]:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
PY

echo "[7/8] Writing README and SHA-256 manifest..."
cat > "$DEST/README.md" <<'EOF'
# Frozen Final SCF Experiment

## Final method

`SCF-LightGCN+BGE`

\[
s(m,a)=0.35\,z(s_{\mathrm{LightGCN}}(m,a))
      +0.65\,z(s_{\mathrm{BGE}}(m,a))
\]

The fusion weights were selected on seed-0 validation only and frozen before
test evaluation.

## Primary full-test result

- Recall@10: approximately 0.7739
- NDCG@10: approximately 0.7158
- MAP@10: approximately 0.6505

## Reporting constraints

1. The primary task is strict new-Mashup cold start.
2. Non-zero Unseen-API results mean unseen in training interactions.
3. On the strict-clean subset, SCF has higher mean scores than Inductive
   LightGCN, but the paired Holm-corrected tests are not significant.
4. Do not change the 0.35/0.65 weights after test evaluation.
5. The final SCF does not contain an active Popularity or Graph-only branch.

## Directory structure

- `model/`: validation-selected weights and leaderboard
- `baseline/`: formal baseline main table
- `main/`: final SCF metrics and rankings
- `significance/`: paired significance tests
- `robustness/`: strict-clean subset analysis
- `diversity/`: long-tail, exposure, novelty and diversity analysis
- `audit/`: duplicate and lexical-cue audit
- `reproducibility/`: exact configs and scripts
- `metadata/`: immutable model specification, environment and checksums
EOF

(
  cd "$DEST"
  find . -type f \
    ! -path "./metadata/SHA256SUMS.txt" \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > metadata/SHA256SUMS.txt
)

echo "[8/8] Verifying frozen package..."
(
  cd "$DEST"
  sha256sum -c metadata/SHA256SUMS.txt
)

echo
echo "Freeze completed successfully."
echo "Frozen directory:"
echo "  $PROJECT_ROOT/$DEST"
echo
echo "File count:"
find "$DEST" -type f | wc -l
echo
echo "Directory size:"
du -sh "$DEST"
