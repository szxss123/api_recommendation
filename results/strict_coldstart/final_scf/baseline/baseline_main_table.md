# Strict New-Mashup Cold-Start Baseline Main Table

All methods use the same Mashup-disjoint train/validation/test split, the same candidate API catalog, all-API ranking, and identical Recall/NDCG/HitRate/MAP definitions.

Vanilla BPR-MF and LightGCN cannot directly represent unseen Mashups. The table therefore labels their fair cold-start adaptations as `Inductive BPR-MF` and `Inductive LightGCN`: each unseen Mashup embedding is projected from its BGE-nearest training Mashups, while the collaborative model is trained only on training interactions.

## Main results (mean ± population std over seeds 0/1/2)

| Method | Recall@5 | NDCG@5 | HitRate@5 | MAP@5 | Recall@10 | NDCG@10 | HitRate@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.3921 ± 0.0000 | 0.3602 ± 0.0000 | 0.5459 ± 0.0000 | 0.3107 ± 0.0000 | 0.4821 ± 0.0000 | 0.3910 ± 0.0000 | 0.6413 ± 0.0000 | 0.3230 ± 0.0000 |
| TF-IDF | 0.3008 ± 0.0000 | 0.2608 ± 0.0000 | 0.4018 ± 0.0000 | 0.2230 ± 0.0000 | 0.3682 ± 0.0000 | 0.2857 ± 0.0000 | 0.4894 ± 0.0000 | 0.2334 ± 0.0000 |
| BM25 | 0.3378 ± 0.0000 | 0.2974 ± 0.0000 | 0.4541 ± 0.0000 | 0.2563 ± 0.0000 | 0.4113 ± 0.0000 | 0.3236 ± 0.0000 | 0.5416 ± 0.0000 | 0.2672 ± 0.0000 |
| Category-Jaccard | 0.0994 ± 0.0000 | 0.0751 ± 0.0000 | 0.1459 ± 0.0000 | 0.0583 ± 0.0000 | 0.1401 ± 0.0000 | 0.0894 ± 0.0000 | 0.2085 ± 0.0000 | 0.0635 ± 0.0000 |
| BGE | 0.2545 ± 0.0000 | 0.2214 ± 0.0000 | 0.3435 ± 0.0000 | 0.1904 ± 0.0000 | 0.3152 ± 0.0000 | 0.2435 ± 0.0000 | 0.4261 ± 0.0000 | 0.1994 ± 0.0000 |
| Inductive BPR-MF | 0.6523 ± 0.0023 | 0.6506 ± 0.0016 | 0.8111 ± 0.0022 | 0.6019 ± 0.0013 | 0.6983 ± 0.0017 | 0.6652 ± 0.0001 | 0.8375 ± 0.0008 | 0.6084 ± 0.0006 |
| Inductive LightGCN | 0.6682 ± 0.0004 | 0.6601 ± 0.0004 | 0.8247 ± 0.0010 | 0.6098 ± 0.0005 | 0.7199 ± 0.0037 | 0.6769 ± 0.0012 | 0.8565 ± 0.0018 | 0.6171 ± 0.0008 |
| Graph-only | 0.3921 ± 0.0000 | 0.3599 ± 0.0004 | 0.5459 ± 0.0000 | 0.3104 ± 0.0004 | 0.4813 ± 0.0040 | 0.3899 ± 0.0012 | 0.6474 ± 0.0025 | 0.3218 ± 0.0004 |
| Graph+BGE | 0.5722 ± 0.0084 | 0.5347 ± 0.0296 | 0.7437 ± 0.0081 | 0.4743 ± 0.0351 | 0.6420 ± 0.0041 | 0.5591 ± 0.0266 | 0.7992 ± 0.0017 | 0.4855 ± 0.0336 |

## Selection protocol

- TF-IDF, BM25, BGE, Category-Jaccard and Popularity are deterministic.
- Collaborative hyperparameters are selected on seed-0 validation only.
- The same selected hyperparameters are used for all test seeds.
- Test results are not used for parameter selection.
