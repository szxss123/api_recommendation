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
