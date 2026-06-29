# Final SCF Paired Significance Analysis

Statistical unit: one test Mashup. Each metric is first averaged over seeds 0/1/2 for each Mashup. The analysis uses paired Wilcoxon signed-rank tests, 10,000 paired bootstrap confidence intervals, Holm-Bonferroni correction, and rank-biserial effect sizes.

## Source coverage

| method | num_seeds | num_unique_mashups | num_rows |
| --- | --- | --- | --- |
| Graph+BGE | 3 | 1645 | 4935 |
| Inductive BPR-MF | 3 | 1645 | 4935 |
| Inductive LightGCN | 3 | 1645 | 4935 |
| SCF-LightGCN+BGE | 3 | 1645 | 4935 |

## Full test set

| method_a | method_b | metric | num_mashups | mean_difference | relative_improvement_percent | bootstrap_ci_low | bootstrap_ci_high | p_holm | rank_biserial | significance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCF-LightGCN+BGE | Inductive LightGCN | Recall@5 | 1645 | 0.0467 | 6.98% | 0.0348 | 0.0586 | <0.001 | 0.3587 | *** |
| SCF-LightGCN+BGE | Inductive LightGCN | NDCG@5 | 1645 | 0.0362 | 5.49% | 0.0272 | 0.0456 | <0.001 | 0.2826 | *** |
| SCF-LightGCN+BGE | Inductive LightGCN | MAP@5 | 1645 | 0.0320 | 5.25% | 0.0230 | 0.0413 | <0.001 | 0.2674 | *** |
| SCF-LightGCN+BGE | Inductive LightGCN | Recall@10 | 1645 | 0.0539 | 7.49% | 0.0411 | 0.0669 | <0.001 | 0.3907 | *** |
| SCF-LightGCN+BGE | Inductive LightGCN | NDCG@10 | 1645 | 0.0389 | 5.75% | 0.0301 | 0.0483 | <0.001 | 0.2927 | *** |
| SCF-LightGCN+BGE | Inductive LightGCN | MAP@10 | 1645 | 0.0334 | 5.42% | 0.0246 | 0.0426 | <0.001 | 0.2626 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | Recall@5 | 1645 | 0.0626 | 9.59% | 0.0493 | 0.0761 | <0.001 | 0.4258 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | NDCG@5 | 1645 | 0.0457 | 7.03% | 0.0357 | 0.0562 | <0.001 | 0.3118 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | MAP@5 | 1645 | 0.0399 | 6.63% | 0.0298 | 0.0503 | <0.001 | 0.2914 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | Recall@10 | 1645 | 0.0756 | 10.82% | 0.0612 | 0.0900 | <0.001 | 0.4688 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | NDCG@10 | 1645 | 0.0506 | 7.60% | 0.0404 | 0.0612 | <0.001 | 0.3282 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | MAP@10 | 1645 | 0.0421 | 6.91% | 0.0318 | 0.0523 | <0.001 | 0.2896 | *** |
| SCF-LightGCN+BGE | Graph+BGE | Recall@5 | 1645 | 0.1427 | 24.94% | 0.1271 | 0.1584 | <0.001 | 0.7286 | *** |
| SCF-LightGCN+BGE | Graph+BGE | NDCG@5 | 1645 | 0.1617 | 30.24% | 0.1477 | 0.1760 | <0.001 | 0.7543 | *** |
| SCF-LightGCN+BGE | Graph+BGE | MAP@5 | 1645 | 0.1675 | 35.32% | 0.1539 | 0.1814 | <0.001 | 0.7734 | *** |
| SCF-LightGCN+BGE | Graph+BGE | Recall@10 | 1645 | 0.1319 | 20.55% | 0.1155 | 0.1487 | <0.001 | 0.6652 | *** |
| SCF-LightGCN+BGE | Graph+BGE | NDCG@10 | 1645 | 0.1567 | 28.02% | 0.1439 | 0.1697 | <0.001 | 0.7427 | *** |
| SCF-LightGCN+BGE | Graph+BGE | MAP@10 | 1645 | 0.1650 | 33.98% | 0.1515 | 0.1783 | <0.001 | 0.7621 | *** |

## Strict-clean subset

| method_a | method_b | metric | num_mashups | mean_difference | relative_improvement_percent | bootstrap_ci_low | bootstrap_ci_high | p_holm | rank_biserial | significance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCF-LightGCN+BGE | Inductive LightGCN | Recall@5 | 879 | 0.0213 | 3.31% | 0.0064 | 0.0370 | 0.4920 | 0.1320 | ns |
| SCF-LightGCN+BGE | Inductive LightGCN | NDCG@5 | 879 | 0.0087 | 1.37% | -0.0023 | 0.0198 | 1.0000 | 0.0103 | ns |
| SCF-LightGCN+BGE | Inductive LightGCN | MAP@5 | 879 | 0.0058 | 0.98% | -0.0048 | 0.0167 | 1.0000 | 0.0066 | ns |
| SCF-LightGCN+BGE | Inductive LightGCN | Recall@10 | 879 | 0.0298 | 4.28% | 0.0136 | 0.0466 | 0.1949 | 0.1760 | ns |
| SCF-LightGCN+BGE | Inductive LightGCN | NDCG@10 | 879 | 0.0118 | 1.82% | 0.0011 | 0.0231 | 1.0000 | 0.0462 | ns |
| SCF-LightGCN+BGE | Inductive LightGCN | MAP@10 | 879 | 0.0073 | 1.23% | -0.0029 | 0.0177 | 1.0000 | 0.0203 | ns |
| SCF-LightGCN+BGE | Inductive BPR-MF | Recall@5 | 879 | 0.0401 | 6.41% | 0.0235 | 0.0576 | 0.0019 | 0.2746 | ** |
| SCF-LightGCN+BGE | Inductive BPR-MF | NDCG@5 | 879 | 0.0220 | 3.54% | 0.0090 | 0.0352 | 0.2006 | 0.1406 | ns |
| SCF-LightGCN+BGE | Inductive BPR-MF | MAP@5 | 879 | 0.0178 | 3.09% | 0.0053 | 0.0305 | 0.2283 | 0.1330 | ns |
| SCF-LightGCN+BGE | Inductive BPR-MF | Recall@10 | 879 | 0.0490 | 7.24% | 0.0305 | 0.0679 | <0.001 | 0.2821 | *** |
| SCF-LightGCN+BGE | Inductive BPR-MF | NDCG@10 | 879 | 0.0253 | 3.96% | 0.0127 | 0.0382 | 0.1057 | 0.1488 | ns |
| SCF-LightGCN+BGE | Inductive BPR-MF | MAP@10 | 879 | 0.0190 | 3.26% | 0.0068 | 0.0315 | 0.2283 | 0.1244 | ns |
| SCF-LightGCN+BGE | Graph+BGE | Recall@5 | 879 | 0.1032 | 18.34% | 0.0832 | 0.1239 | <0.001 | 0.6312 | *** |
| SCF-LightGCN+BGE | Graph+BGE | NDCG@5 | 879 | 0.1386 | 27.40% | 0.1210 | 0.1560 | <0.001 | 0.7116 | *** |
| SCF-LightGCN+BGE | Graph+BGE | MAP@5 | 879 | 0.1453 | 32.35% | 0.1279 | 0.1629 | <0.001 | 0.7476 | *** |
| SCF-LightGCN+BGE | Graph+BGE | Recall@10 | 879 | 0.0897 | 14.09% | 0.0685 | 0.1108 | <0.001 | 0.5219 | *** |
| SCF-LightGCN+BGE | Graph+BGE | NDCG@10 | 879 | 0.1320 | 24.84% | 0.1153 | 0.1485 | <0.001 | 0.7051 | *** |
| SCF-LightGCN+BGE | Graph+BGE | MAP@10 | 879 | 0.1418 | 30.82% | 0.1252 | 0.1586 | <0.001 | 0.7378 | *** |

## Naming

The validation-selected popularity weight is zero. The final model should therefore be reported as `SCF-LightGCN+BGE`, not as a three-branch model containing an active popularity component.
