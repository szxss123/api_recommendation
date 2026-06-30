# SCF Head/Middle/Tail/Unseen Group Significance Analysis

Statistical unit: one eligible test Mashup containing at least one ground-truth API in the target group. Each per-Mashup metric is first averaged over seeds 0/1/2. Paired Wilcoxon signed-rank tests, 10,000 paired bootstrap confidence intervals, Holm-Bonferroni correction, and rank-biserial effect sizes are then applied.

The primary p-value is `p_holm_subset_global`, which corrects all 48 group/metric/comparison tests within each subset as one family.

## Coverage

| method | seed | num_mashups | num_rows | minimum_rank | maximum_rank |
| --- | --- | --- | --- | --- | --- |
| Graph+BGE | 0 | 1645 | 16450 | 1 | 10 |
| Graph+BGE | 1 | 1645 | 16450 | 1 | 10 |
| Graph+BGE | 2 | 1645 | 16450 | 1 | 10 |
| Inductive LightGCN | 0 | 1645 | 16450 | 1 | 10 |
| Inductive LightGCN | 1 | 1645 | 16450 | 1 | 10 |
| Inductive LightGCN | 2 | 1645 | 16450 | 1 | 10 |
| SCF-LightGCN+BGE | 0 | 1645 | 16450 | 1 | 10 |
| SCF-LightGCN+BGE | 1 | 1645 | 16450 | 1 | 10 |
| SCF-LightGCN+BGE | 2 | 1645 | 16450 | 1 | 10 |

Strict-clean Mashups available: 879.

## Subset: full

### SCF-LightGCN+BGE vs Inductive LightGCN

| group | metric | num_mashups | mean_a | mean_b | mean_difference | bootstrap_ci_low | bootstrap_ci_high | p_holm_subset_global | rank_biserial | significance_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | Recall@5 | 1487 | 0.7538 | 0.7512 | 0.0027 | -0.0072 | 0.0128 | 1.0000 | -0.0252 | ns |
| Head | NDCG@5 | 1487 | 0.7358 | 0.7341 | 0.0016 | -0.0061 | 0.0095 | 1.0000 | -0.0229 | ns |
| Head | MAP@5 | 1487 | 0.6855 | 0.6854 | 0.0002 | -0.0080 | 0.0083 | 1.0000 | -0.0193 | ns |
| Head | Recall@10 | 1487 | 0.8028 | 0.8048 | -0.0020 | -0.0117 | 0.0079 | 0.9631 | -0.0888 | ns |
| Head | NDCG@10 | 1487 | 0.7525 | 0.7525 | 0.0000 | -0.0072 | 0.0070 | 1.0000 | -0.0416 | ns |
| Head | MAP@10 | 1487 | 0.6936 | 0.6942 | -0.0006 | -0.0084 | 0.0073 | 1.0000 | -0.0386 | ns |
| Middle | Recall@5 | 237 | 0.4229 | 0.2396 | 0.1834 | 0.1355 | 0.2333 | <0.001 | 0.8915 | *** |
| Middle | NDCG@5 | 237 | 0.2950 | 0.1525 | 0.1425 | 0.1096 | 0.1760 | <0.001 | 0.8705 | *** |
| Middle | MAP@5 | 237 | 0.2477 | 0.1212 | 0.1265 | 0.0968 | 0.1579 | <0.001 | 0.8642 | *** |
| Middle | Recall@10 | 237 | 0.5260 | 0.3203 | 0.2057 | 0.1571 | 0.2567 | <0.001 | 0.8734 | *** |
| Middle | NDCG@10 | 237 | 0.3299 | 0.1799 | 0.1500 | 0.1202 | 0.1811 | <0.001 | 0.8564 | *** |
| Middle | MAP@10 | 237 | 0.2628 | 0.1328 | 0.1300 | 0.1009 | 0.1599 | <0.001 | 0.8314 | *** |
| Tail | Recall@5 | 110 | 0.3182 | 0.1167 | 0.2015 | 0.1258 | 0.2803 | <0.001 | 0.8788 | *** |
| Tail | NDCG@5 | 110 | 0.2213 | 0.0830 | 0.1383 | 0.0873 | 0.1902 | <0.001 | 0.8538 | *** |
| Tail | MAP@5 | 110 | 0.1824 | 0.0699 | 0.1124 | 0.0674 | 0.1612 | <0.001 | 0.8026 | *** |
| Tail | Recall@10 | 110 | 0.4106 | 0.1561 | 0.2545 | 0.1742 | 0.3379 | <0.001 | 0.9175 | *** |
| Tail | NDCG@10 | 110 | 0.2528 | 0.0971 | 0.1556 | 0.1054 | 0.2090 | <0.001 | 0.8807 | *** |
| Tail | MAP@10 | 110 | 0.1979 | 0.0758 | 0.1220 | 0.0777 | 0.1705 | <0.001 | 0.8344 | *** |
| Unseen | Recall@5 | 144 | 0.2422 | 0.0000 | 0.2422 | 0.1763 | 0.3125 | <0.001 | 1.0000 | *** |
| Unseen | NDCG@5 | 144 | 0.1523 | 0.0000 | 0.1523 | 0.1076 | 0.1999 | <0.001 | 1.0000 | *** |
| Unseen | MAP@5 | 144 | 0.1208 | 0.0000 | 0.1208 | 0.0826 | 0.1633 | <0.001 | 1.0000 | *** |
| Unseen | Recall@10 | 144 | 0.3463 | 0.0000 | 0.3463 | 0.2692 | 0.4257 | <0.001 | 1.0000 | *** |
| Unseen | NDCG@10 | 144 | 0.1881 | 0.0000 | 0.1881 | 0.1432 | 0.2360 | <0.001 | 1.0000 | *** |
| Unseen | MAP@10 | 144 | 0.1360 | 0.0000 | 0.1360 | 0.0978 | 0.1776 | <0.001 | 1.0000 | *** |

### SCF-LightGCN+BGE vs Graph+BGE

| group | metric | num_mashups | mean_a | mean_b | mean_difference | bootstrap_ci_low | bootstrap_ci_high | p_holm_subset_global | rank_biserial | significance_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | Recall@5 | 1487 | 0.7538 | 0.6612 | 0.0926 | 0.0782 | 0.1068 | <0.001 | 0.5918 | *** |
| Head | NDCG@5 | 1487 | 0.7358 | 0.6090 | 0.1268 | 0.1139 | 0.1400 | <0.001 | 0.6774 | *** |
| Head | MAP@5 | 1487 | 0.6855 | 0.5478 | 0.1378 | 0.1243 | 0.1515 | <0.001 | 0.7026 | *** |
| Head | Recall@10 | 1487 | 0.8028 | 0.7403 | 0.0625 | 0.0484 | 0.0768 | <0.001 | 0.4260 | *** |
| Head | NDCG@10 | 1487 | 0.7525 | 0.6375 | 0.1150 | 0.1029 | 0.1273 | <0.001 | 0.6414 | *** |
| Head | MAP@10 | 1487 | 0.6936 | 0.5618 | 0.1319 | 0.1185 | 0.1451 | <0.001 | 0.6793 | *** |
| Middle | Recall@5 | 237 | 0.4229 | 0.0612 | 0.3617 | 0.3044 | 0.4198 | <0.001 | 1.0000 | *** |
| Middle | NDCG@5 | 237 | 0.2950 | 0.0403 | 0.2547 | 0.2115 | 0.2981 | <0.001 | 0.9896 | *** |
| Middle | MAP@5 | 237 | 0.2477 | 0.0333 | 0.2144 | 0.1741 | 0.2549 | <0.001 | 0.9779 | *** |
| Middle | Recall@10 | 237 | 0.5260 | 0.0956 | 0.4304 | 0.3734 | 0.4895 | <0.001 | 0.9979 | *** |
| Middle | NDCG@10 | 237 | 0.3299 | 0.0515 | 0.2784 | 0.2385 | 0.3197 | <0.001 | 0.9889 | *** |
| Middle | MAP@10 | 237 | 0.2628 | 0.0379 | 0.2250 | 0.1870 | 0.2640 | <0.001 | 0.9707 | *** |
| Tail | Recall@5 | 110 | 0.3182 | 0.0121 | 0.3061 | 0.2288 | 0.3864 | <0.001 | 1.0000 | *** |
| Tail | NDCG@5 | 110 | 0.2213 | 0.0053 | 0.2160 | 0.1583 | 0.2775 | <0.001 | 1.0000 | *** |
| Tail | MAP@5 | 110 | 0.1824 | 0.0031 | 0.1792 | 0.1257 | 0.2387 | <0.001 | 1.0000 | *** |
| Tail | Recall@10 | 110 | 0.4106 | 0.0121 | 0.3985 | 0.3121 | 0.4894 | <0.001 | 1.0000 | *** |
| Tail | NDCG@10 | 110 | 0.2528 | 0.0053 | 0.2475 | 0.1867 | 0.3107 | <0.001 | 1.0000 | *** |
| Tail | MAP@10 | 110 | 0.1979 | 0.0031 | 0.1947 | 0.1402 | 0.2540 | <0.001 | 1.0000 | *** |
| Unseen | Recall@5 | 144 | 0.2422 | 0.0000 | 0.2422 | 0.1755 | 0.3134 | <0.001 | 1.0000 | *** |
| Unseen | NDCG@5 | 144 | 0.1523 | 0.0000 | 0.1523 | 0.1069 | 0.1998 | <0.001 | 1.0000 | *** |
| Unseen | MAP@5 | 144 | 0.1208 | 0.0000 | 0.1208 | 0.0827 | 0.1629 | <0.001 | 1.0000 | *** |
| Unseen | Recall@10 | 144 | 0.3463 | 0.0000 | 0.3463 | 0.2722 | 0.4222 | <0.001 | 1.0000 | *** |
| Unseen | NDCG@10 | 144 | 0.1881 | 0.0000 | 0.1881 | 0.1435 | 0.2354 | <0.001 | 1.0000 | *** |
| Unseen | MAP@10 | 144 | 0.1360 | 0.0000 | 0.1360 | 0.0977 | 0.1785 | <0.001 | 1.0000 | *** |

### Interpretation checklist

- A positive mean difference favors SCF.
- A bootstrap confidence interval entirely above zero supports a positive average improvement.
- The formal significance claim requires `p_holm_subset_global < 0.05`.
- Unseen conclusions must report the eligible Mashup count; small samples limit generalization.

## Subset: strict_clean

### SCF-LightGCN+BGE vs Inductive LightGCN

| group | metric | num_mashups | mean_a | mean_b | mean_difference | bootstrap_ci_low | bootstrap_ci_high | p_holm_subset_global | rank_biserial | significance_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | Recall@5 | 808 | 0.7124 | 0.7192 | -0.0068 | -0.0202 | 0.0069 | 0.0930 | -0.1467 | ns |
| Head | NDCG@5 | 808 | 0.6901 | 0.7004 | -0.0103 | -0.0202 | -0.0001 | 0.0347 | -0.2133 | * |
| Head | MAP@5 | 808 | 0.6423 | 0.6531 | -0.0108 | -0.0208 | -0.0009 | 0.0347 | -0.2109 | * |
| Head | Recall@10 | 808 | 0.7642 | 0.7748 | -0.0106 | -0.0241 | 0.0031 | 0.0455 | -0.2009 | * |
| Head | NDCG@10 | 808 | 0.7073 | 0.7184 | -0.0111 | -0.0203 | -0.0016 | 0.0259 | -0.2102 | * |
| Head | MAP@10 | 808 | 0.6498 | 0.6606 | -0.0108 | -0.0204 | -0.0012 | 0.0335 | -0.2045 | * |
| Middle | Recall@5 | 107 | 0.2016 | 0.0955 | 0.1060 | 0.0498 | 0.1694 | 0.0241 | 0.7826 | * |
| Middle | NDCG@5 | 107 | 0.1248 | 0.0629 | 0.0618 | 0.0297 | 0.0976 | 0.0155 | 0.7513 | * |
| Middle | MAP@5 | 107 | 0.0957 | 0.0510 | 0.0447 | 0.0184 | 0.0738 | 0.0257 | 0.7090 | * |
| Middle | Recall@10 | 107 | 0.2752 | 0.1690 | 0.1062 | 0.0413 | 0.1747 | 0.0347 | 0.6460 | * |
| Middle | NDCG@10 | 107 | 0.1498 | 0.0880 | 0.0618 | 0.0297 | 0.0962 | 0.0143 | 0.6397 | * |
| Middle | MAP@10 | 107 | 0.1058 | 0.0611 | 0.0447 | 0.0193 | 0.0729 | 0.0192 | 0.6115 | * |
| Tail | Recall@5 | 53 | 0.1792 | 0.0377 | 0.1415 | 0.0629 | 0.2327 | 0.0347 | 1.0000 | * |
| Tail | NDCG@5 | 53 | 0.1247 | 0.0253 | 0.0994 | 0.0452 | 0.1604 | 0.0347 | 1.0000 | * |
| Tail | MAP@5 | 53 | 0.1025 | 0.0210 | 0.0816 | 0.0353 | 0.1360 | 0.0347 | 1.0000 | * |
| Tail | Recall@10 | 53 | 0.2925 | 0.0597 | 0.2327 | 0.1289 | 0.3428 | 0.0156 | 1.0000 | * |
| Tail | NDCG@10 | 53 | 0.1627 | 0.0330 | 0.1297 | 0.0730 | 0.1948 | 0.0143 | 1.0000 | * |
| Tail | MAP@10 | 53 | 0.1203 | 0.0237 | 0.0966 | 0.0498 | 0.1526 | 0.0143 | 1.0000 | * |
| Unseen | Recall@5 | 73 | 0.1233 | 0.0000 | 0.1233 | 0.0548 | 0.2055 | 0.0347 | 1.0000 | * |
| Unseen | NDCG@5 | 73 | 0.0794 | 0.0000 | 0.0794 | 0.0328 | 0.1340 | 0.0455 | 1.0000 | * |
| Unseen | MAP@5 | 73 | 0.0646 | 0.0000 | 0.0646 | 0.0240 | 0.1126 | 0.0455 | 1.0000 | * |
| Unseen | Recall@10 | 73 | 0.2068 | 0.0000 | 0.2068 | 0.1233 | 0.3027 | 0.0035 | 1.0000 | ** |
| Unseen | NDCG@10 | 73 | 0.1076 | 0.0000 | 0.1076 | 0.0584 | 0.1633 | 0.0143 | 1.0000 | * |
| Unseen | MAP@10 | 73 | 0.0763 | 0.0000 | 0.0763 | 0.0362 | 0.1262 | 0.0143 | 1.0000 | * |

### SCF-LightGCN+BGE vs Graph+BGE

| group | metric | num_mashups | mean_a | mean_b | mean_difference | bootstrap_ci_low | bootstrap_ci_high | p_holm_subset_global | rank_biserial | significance_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | Recall@5 | 808 | 0.7124 | 0.6342 | 0.0782 | 0.0579 | 0.0984 | <0.001 | 0.5253 | *** |
| Head | NDCG@5 | 808 | 0.6901 | 0.5627 | 0.1274 | 0.1096 | 0.1453 | <0.001 | 0.6671 | *** |
| Head | MAP@5 | 808 | 0.6423 | 0.5031 | 0.1392 | 0.1214 | 0.1568 | <0.001 | 0.7096 | *** |
| Head | Recall@10 | 808 | 0.7642 | 0.7195 | 0.0447 | 0.0255 | 0.0645 | <0.001 | 0.3114 | *** |
| Head | NDCG@10 | 808 | 0.7073 | 0.5932 | 0.1142 | 0.0978 | 0.1309 | <0.001 | 0.6363 | *** |
| Head | MAP@10 | 808 | 0.6498 | 0.5170 | 0.1328 | 0.1153 | 0.1507 | <0.001 | 0.6889 | *** |
| Middle | Recall@5 | 107 | 0.2016 | 0.0062 | 0.1953 | 0.1252 | 0.2701 | <0.001 | 1.0000 | *** |
| Middle | NDCG@5 | 107 | 0.1248 | 0.0029 | 0.1219 | 0.0769 | 0.1729 | <0.001 | 1.0000 | *** |
| Middle | MAP@5 | 107 | 0.0957 | 0.0018 | 0.0938 | 0.0557 | 0.1384 | <0.001 | 1.0000 | *** |
| Middle | Recall@10 | 107 | 0.2752 | 0.0234 | 0.2518 | 0.1786 | 0.3302 | <0.001 | 0.9844 | *** |
| Middle | NDCG@10 | 107 | 0.1498 | 0.0084 | 0.1414 | 0.0972 | 0.1897 | <0.001 | 0.9892 | *** |
| Middle | MAP@10 | 107 | 0.1058 | 0.0040 | 0.1018 | 0.0630 | 0.1454 | <0.001 | 0.9892 | *** |
| Tail | Recall@5 | 53 | 0.1792 | 0.0126 | 0.1667 | 0.0818 | 0.2610 | 0.0335 | 1.0000 | * |
| Tail | NDCG@5 | 53 | 0.1247 | 0.0059 | 0.1189 | 0.0556 | 0.1915 | 0.0347 | 1.0000 | * |
| Tail | MAP@5 | 53 | 0.1025 | 0.0037 | 0.0988 | 0.0408 | 0.1690 | 0.0347 | 1.0000 | * |
| Tail | Recall@10 | 53 | 0.2925 | 0.0126 | 0.2799 | 0.1667 | 0.4025 | 0.0060 | 1.0000 | ** |
| Tail | NDCG@10 | 53 | 0.1627 | 0.0059 | 0.1569 | 0.0876 | 0.2330 | 0.0143 | 1.0000 | * |
| Tail | MAP@10 | 53 | 0.1203 | 0.0037 | 0.1166 | 0.0583 | 0.1859 | 0.0143 | 1.0000 | * |
| Unseen | Recall@5 | 73 | 0.1233 | 0.0000 | 0.1233 | 0.0548 | 0.2055 | 0.0347 | 1.0000 | * |
| Unseen | NDCG@5 | 73 | 0.0794 | 0.0000 | 0.0794 | 0.0341 | 0.1344 | 0.0455 | 1.0000 | * |
| Unseen | MAP@5 | 73 | 0.0646 | 0.0000 | 0.0646 | 0.0247 | 0.1140 | 0.0455 | 1.0000 | * |
| Unseen | Recall@10 | 73 | 0.2068 | 0.0000 | 0.2068 | 0.1233 | 0.3027 | 0.0035 | 1.0000 | ** |
| Unseen | NDCG@10 | 73 | 0.1076 | 0.0000 | 0.1076 | 0.0582 | 0.1646 | 0.0143 | 1.0000 | * |
| Unseen | MAP@10 | 73 | 0.0763 | 0.0000 | 0.0763 | 0.0358 | 0.1262 | 0.0143 | 1.0000 | * |

### Interpretation checklist

- A positive mean difference favors SCF.
- A bootstrap confidence interval entirely above zero supports a positive average improvement.
- The formal significance claim requires `p_holm_subset_global < 0.05`.
- Unseen conclusions must report the eligible Mashup count; small samples limit generalization.

## Reporting rules

- Report group metrics only over Mashups that contain at least one ground-truth API in that group.
- Use `interaction-unseen API` or `API unseen in training interactions`; do not claim the pretrained text encoder never saw the concept.
- If strict-clean results are numerically positive but not Holm-significant, describe them as a positive trend rather than a statistically significant improvement.
