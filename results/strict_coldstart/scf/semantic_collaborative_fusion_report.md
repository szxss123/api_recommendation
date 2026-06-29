# Semantic-Collaborative Fusion under Strict New-Mashup Cold Start

Fusion weights are selected once on seed-0 validation only. The test set is never used for selecting weights.

## Selected weights

- Inductive LightGCN: 0.35
- Direct BGE: 0.65
- Popularity prior: 0.00
- Normalization: per-Mashup row z-score

## Full test-set main table

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
| SCF-LightGCN+BGE+Popularity | 0.7149 ± 0.0021 | 0.6964 ± 0.0003 | 0.8740 ± 0.0017 | 0.6418 ± 0.0005 | 0.7739 ± 0.0024 | 0.7158 ± 0.0004 | 0.9121 ± 0.0028 | 0.6505 ± 0.0003 |

## Strict-clean robustness subset

| Method | Mashups | Recall@5 | NDCG@5 | HitRate@5 | MAP@5 | Recall@10 | NDCG@10 | HitRate@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCF-LightGCN+BGE+Popularity | 879 | 0.6661 ± 0.0039 | 0.6443 ± 0.0012 | 0.8233 ± 0.0019 | 0.5943 ± 0.0010 | 0.7260 ± 0.0067 | 0.6633 ± 0.0027 | 0.8699 ± 0.0063 | 0.6018 ± 0.0016 |

## Interpretation rule

Keep the fusion only if validation selection yields a non-trivial combination and the test/strict-clean results improve or remain competitive with Inductive LightGCN. Do not change weights after seeing test results.
