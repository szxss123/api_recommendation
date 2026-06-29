# Strict Cold-Start Final Ablation

## Overall results (mean ± population std over 3 seeds)

| Method | Recall@5 | NDCG@5 | HitRate@5 | MAP@5 | Recall@10 | NDCG@10 | HitRate@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.3921 | 0.3602 | 0.5459 | 0.3107 | 0.4821 | 0.3910 | 0.6413 | 0.3230 |
| Graph-only | 0.3921 ± 0.0000 | 0.3599 ± 0.0004 | 0.5459 ± 0.0000 | 0.3104 ± 0.0004 | 0.4813 ± 0.0040 | 0.3899 ± 0.0012 | 0.6474 ± 0.0025 | 0.3218 ± 0.0004 |
| BGE-only | 0.2545 ± 0.0000 | 0.2214 ± 0.0000 | 0.3435 ± 0.0000 | 0.1904 ± 0.0000 | 0.3152 ± 0.0000 | 0.2435 ± 0.0000 | 0.4261 ± 0.0000 | 0.1994 ± 0.0000 |
| Graph+BGE (z-score, λ=0.25) | 0.5722 ± 0.0084 | 0.5347 ± 0.0296 | 0.7437 ± 0.0081 | 0.4743 ± 0.0351 | 0.6420 ± 0.0041 | 0.5591 ± 0.0266 | 0.7992 ± 0.0017 | 0.4855 ± 0.0336 |

## Selected main model

`Graph+BGE (z-score, λ=0.25)`

Fusion formula:

```text
final_score = 0.75 × z(graph_score) + 0.25 × z(BGE cosine)
```

## Relative improvement over Popularity

| Metric | Popularity | Main model | Relative improvement |
| --- | --- | --- | --- |
| Recall@5 | 0.3921 | 0.5722 | 45.95% |
| NDCG@5 | 0.3602 | 0.5347 | 48.43% |
| HitRate@5 | 0.5459 | 0.7437 | 36.23% |
| MAP@5 | 0.3107 | 0.4743 | 52.63% |
| Recall@10 | 0.4821 | 0.6420 | 33.16% |
| NDCG@10 | 0.3910 | 0.5591 | 43.01% |
| HitRate@10 | 0.6413 | 0.7992 | 24.61% |
| MAP@10 | 0.3230 | 0.4855 | 50.32% |

## Main-model Head/Middle/Tail/Unseen results

| Group | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- |
| Head | 0.6612 ± 0.0101 | 0.6090 ± 0.0346 | 0.7403 ± 0.0033 | 0.6375 ± 0.0311 | 0.5618 ± 0.0392 |
| Middle | 0.0612 ± 0.0147 | 0.0403 ± 0.0160 | 0.0956 ± 0.0115 | 0.0515 ± 0.0150 | 0.0379 ± 0.0160 |
| Tail | 0.0121 ± 0.0086 | 0.0053 ± 0.0037 | 0.0121 ± 0.0086 | 0.0053 ± 0.0037 | 0.0031 ± 0.0022 |
| Unseen | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Interpretation

- Graph-only is expected to remain close to the popularity ranking.
- BGE-only measures semantic cold-start capability without graph scores.
- The standardized Graph+BGE model is the final main configuration.
- Middle and Tail results should be reported separately.
- Unseen-API performance must not be described as solved when it remains zero.
