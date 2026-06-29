# SCF Long-Tail, Diversity, and Popularity-Bias Analysis

All values are mean ± population standard deviation over seeds 0/1/2.
The primary comparison is among Graph+BGE, Inductive LightGCN, and SCF-LightGCN+BGE.

## Subset: full

### Diversity and popularity bias

#### K=5

| Method | Coverage | Long-tail coverage | Personalization | Gini | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- |
| Graph+BGE | 0.1018 ± 0.0222 | 0.0215 ± 0.0110 | 0.7835 ± 0.0944 | 0.9821 ± 0.0086 | 472.3708 ± 123.9667 | 6.0845 ± 0.7072 |
| Inductive LightGCN | 0.3001 ± 0.0020 | 0.2265 ± 0.0032 | 0.8941 ± 0.0019 | 0.9507 ± 0.0009 | 368.4348 ± 4.1241 | 6.9353 ± 0.0213 |
| SCF-LightGCN+BGE | 0.5203 ± 0.0039 | 0.4679 ± 0.0018 | 0.9471 ± 0.0003 | 0.8737 ± 0.0012 | 271.8124 ± 1.9756 | 8.3898 ± 0.0245 |

#### K=10

| Method | Coverage | Long-tail coverage | Personalization | Gini | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- |
| Graph+BGE | 0.1305 ± 0.0180 | 0.0382 ± 0.0082 | 0.7534 ± 0.0876 | 0.9754 ± 0.0085 | 342.5535 ± 75.2898 | 6.6681 ± 0.5348 |
| Inductive LightGCN | 0.4602 ± 0.0028 | 0.4339 ± 0.0022 | 0.8889 ± 0.0002 | 0.9225 ± 0.0015 | 239.4594 ± 2.9542 | 7.7301 ± 0.0202 |
| SCF-LightGCN+BGE | 0.7308 ± 0.0027 | 0.7093 ± 0.0017 | 0.9516 ± 0.0009 | 0.8135 ± 0.0015 | 160.8054 ± 0.9348 | 9.3011 ± 0.0200 |

### Recommendation exposure by API group

#### K=5 slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Graph+BGE | 0.9922 ± 0.0061 | 0.0073 ± 0.0058 | 0.0004 ± 0.0003 | 0.0000 ± 0.0000 |
| Inductive LightGCN | 0.9325 ± 0.0005 | 0.0508 ± 0.0011 | 0.0167 ± 0.0009 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | 0.7596 ± 0.0046 | 0.1484 ± 0.0039 | 0.0783 ± 0.0002 | 0.0138 ± 0.0005 |

#### K=10 slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Graph+BGE | 0.9907 ± 0.0059 | 0.0090 ± 0.0058 | 0.0003 ± 0.0001 | 0.0000 ± 0.0000 |
| Inductive LightGCN | 0.8994 ± 0.0026 | 0.0740 ± 0.0029 | 0.0266 ± 0.0010 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | 0.6677 ± 0.0016 | 0.1938 ± 0.0028 | 0.1212 ± 0.0013 | 0.0173 ± 0.0003 |

### Accuracy by ground-truth API group

#### K=5

| Method | Group | Mashups | Recall | NDCG | MAP |
| --- | --- | --- | --- | --- | --- |
| Graph+BGE | Head | 1487 | 0.6612 ± 0.0101 | 0.6090 ± 0.0346 | 0.5478 ± 0.0408 |
| Graph+BGE | Middle | 237 | 0.0612 ± 0.0147 | 0.0403 ± 0.0160 | 0.0333 ± 0.0165 |
| Graph+BGE | Tail | 110 | 0.0121 ± 0.0086 | 0.0053 ± 0.0037 | 0.0031 ± 0.0022 |
| Graph+BGE | Unseen | 144 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Inductive LightGCN | Head | 1487 | 0.7512 ± 0.0014 | 0.7341 ± 0.0010 | 0.6854 ± 0.0011 |
| Inductive LightGCN | Middle | 237 | 0.2396 ± 0.0035 | 0.1525 ± 0.0026 | 0.1212 ± 0.0022 |
| Inductive LightGCN | Tail | 110 | 0.1167 ± 0.0043 | 0.0830 ± 0.0012 | 0.0699 ± 0.0013 |
| Inductive LightGCN | Unseen | 144 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | Head | 1487 | 0.7538 ± 0.0017 | 0.7358 ± 0.0002 | 0.6855 ± 0.0002 |
| SCF-LightGCN+BGE | Middle | 237 | 0.4229 ± 0.0026 | 0.2950 ± 0.0014 | 0.2477 ± 0.0027 |
| SCF-LightGCN+BGE | Tail | 110 | 0.3182 ± 0.0021 | 0.2213 ± 0.0030 | 0.1824 ± 0.0035 |
| SCF-LightGCN+BGE | Unseen | 144 | 0.2422 ± 0.0013 | 0.1523 ± 0.0037 | 0.1208 ± 0.0047 |

#### K=10

| Method | Group | Mashups | Recall | NDCG | MAP |
| --- | --- | --- | --- | --- | --- |
| Graph+BGE | Head | 1487 | 0.7403 ± 0.0033 | 0.6375 ± 0.0311 | 0.5618 ± 0.0392 |
| Graph+BGE | Middle | 237 | 0.0956 ± 0.0115 | 0.0515 ± 0.0150 | 0.0379 ± 0.0160 |
| Graph+BGE | Tail | 110 | 0.0121 ± 0.0086 | 0.0053 ± 0.0037 | 0.0031 ± 0.0022 |
| Graph+BGE | Unseen | 144 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Inductive LightGCN | Head | 1487 | 0.8048 ± 0.0034 | 0.7525 ± 0.0012 | 0.6942 ± 0.0012 |
| Inductive LightGCN | Middle | 237 | 0.3203 ± 0.0090 | 0.1799 ± 0.0042 | 0.1328 ± 0.0028 |
| Inductive LightGCN | Tail | 110 | 0.1561 ± 0.0081 | 0.0971 ± 0.0032 | 0.0758 ± 0.0020 |
| Inductive LightGCN | Unseen | 144 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | Head | 1487 | 0.8028 ± 0.0023 | 0.7525 ± 0.0007 | 0.6936 ± 0.0004 |
| SCF-LightGCN+BGE | Middle | 237 | 0.5260 ± 0.0079 | 0.3299 ± 0.0043 | 0.2628 ± 0.0035 |
| SCF-LightGCN+BGE | Tail | 110 | 0.4106 ± 0.0093 | 0.2528 ± 0.0052 | 0.1979 ± 0.0042 |
| SCF-LightGCN+BGE | Unseen | 144 | 0.3463 ± 0.0033 | 0.1881 ± 0.0036 | 0.1360 ± 0.0046 |

## Subset: strict_clean

### Diversity and popularity bias

#### K=5

| Method | Coverage | Long-tail coverage | Personalization | Gini | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- |
| Graph+BGE | 0.0799 ± 0.0210 | 0.0119 ± 0.0093 | 0.7837 ± 0.0873 | 0.9830 ± 0.0083 | 472.4530 ± 118.6072 | 6.1385 ± 0.7253 |
| Inductive LightGCN | 0.1961 ± 0.0035 | 0.1107 ± 0.0037 | 0.8838 ± 0.0017 | 0.9621 ± 0.0011 | 384.2355 ± 3.7224 | 6.9185 ± 0.0239 |
| SCF-LightGCN+BGE | 0.3688 ± 0.0014 | 0.3080 ± 0.0044 | 0.9331 ± 0.0010 | 0.9006 ± 0.0016 | 298.6579 ± 1.6661 | 8.3592 ± 0.0326 |

#### K=10

| Method | Coverage | Long-tail coverage | Personalization | Gini | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- |
| Graph+BGE | 0.1073 ± 0.0199 | 0.0215 ± 0.0070 | 0.7543 ± 0.0854 | 0.9760 ± 0.0084 | 338.0759 ± 73.8730 | 6.7108 ± 0.5587 |
| Inductive LightGCN | 0.3252 ± 0.0072 | 0.2540 ± 0.0097 | 0.8757 ± 0.0040 | 0.9385 ± 0.0017 | 242.9613 ± 2.5984 | 7.7337 ± 0.0182 |
| SCF-LightGCN+BGE | 0.5624 ± 0.0076 | 0.5306 ± 0.0088 | 0.9397 ± 0.0020 | 0.8447 ± 0.0016 | 172.1780 ± 1.3965 | 9.2912 ± 0.0282 |

### Recommendation exposure by API group

#### K=5 slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Graph+BGE | 0.9924 ± 0.0077 | 0.0071 ± 0.0074 | 0.0005 ± 0.0004 | 0.0000 ± 0.0000 |
| Inductive LightGCN | 0.9420 ± 0.0008 | 0.0452 ± 0.0020 | 0.0128 ± 0.0014 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | 0.7599 ± 0.0048 | 0.1435 ± 0.0045 | 0.0842 ± 0.0007 | 0.0124 ± 0.0008 |

#### K=10 slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Graph+BGE | 0.9907 ± 0.0066 | 0.0091 ± 0.0064 | 0.0003 ± 0.0001 | 0.0000 ± 0.0000 |
| Inductive LightGCN | 0.9078 ± 0.0030 | 0.0695 ± 0.0035 | 0.0226 ± 0.0016 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | 0.6640 ± 0.0025 | 0.1907 ± 0.0039 | 0.1292 ± 0.0019 | 0.0160 ± 0.0004 |

### Accuracy by ground-truth API group

#### K=5

| Method | Group | Mashups | Recall | NDCG | MAP |
| --- | --- | --- | --- | --- | --- |
| Graph+BGE | Head | 808 | 0.6342 ± 0.0231 | 0.5627 ± 0.0583 | 0.5031 ± 0.0668 |
| Graph+BGE | Middle | 107 | 0.0062 ± 0.0044 | 0.0029 ± 0.0021 | 0.0018 ± 0.0013 |
| Graph+BGE | Tail | 53 | 0.0126 ± 0.0089 | 0.0059 ± 0.0042 | 0.0037 ± 0.0027 |
| Graph+BGE | Unseen | 73 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Inductive LightGCN | Head | 808 | 0.7192 ± 0.0017 | 0.7004 ± 0.0015 | 0.6531 ± 0.0014 |
| Inductive LightGCN | Middle | 107 | 0.0955 ± 0.0064 | 0.0629 ± 0.0026 | 0.0510 ± 0.0010 |
| Inductive LightGCN | Tail | 53 | 0.0377 ± 0.0000 | 0.0253 ± 0.0021 | 0.0210 ± 0.0030 |
| Inductive LightGCN | Unseen | 73 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | Head | 808 | 0.7124 ± 0.0039 | 0.6901 ± 0.0012 | 0.6423 ± 0.0010 |
| SCF-LightGCN+BGE | Middle | 107 | 0.2016 ± 0.0114 | 0.1248 ± 0.0019 | 0.0957 ± 0.0025 |
| SCF-LightGCN+BGE | Tail | 53 | 0.1792 ± 0.0160 | 0.1247 ± 0.0040 | 0.1025 ± 0.0013 |
| SCF-LightGCN+BGE | Unseen | 73 | 0.1233 ± 0.0000 | 0.0794 ± 0.0003 | 0.0646 ± 0.0003 |

#### K=10

| Method | Group | Mashups | Recall | NDCG | MAP |
| --- | --- | --- | --- | --- | --- |
| Graph+BGE | Head | 808 | 0.7195 ± 0.0052 | 0.5932 ± 0.0512 | 0.5170 ± 0.0635 |
| Graph+BGE | Middle | 107 | 0.0234 ± 0.0066 | 0.0084 ± 0.0027 | 0.0040 ± 0.0014 |
| Graph+BGE | Tail | 53 | 0.0126 ± 0.0089 | 0.0059 ± 0.0042 | 0.0037 ± 0.0027 |
| Graph+BGE | Unseen | 73 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Inductive LightGCN | Head | 808 | 0.7748 ± 0.0010 | 0.7184 ± 0.0015 | 0.6606 ± 0.0016 |
| Inductive LightGCN | Middle | 107 | 0.1690 ± 0.0055 | 0.0880 ± 0.0011 | 0.0611 ± 0.0001 |
| Inductive LightGCN | Tail | 53 | 0.0597 ± 0.0168 | 0.0330 ± 0.0066 | 0.0237 ± 0.0037 |
| Inductive LightGCN | Unseen | 73 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SCF-LightGCN+BGE | Head | 808 | 0.7642 ± 0.0067 | 0.7073 ± 0.0025 | 0.6498 ± 0.0015 |
| SCF-LightGCN+BGE | Middle | 107 | 0.2752 ± 0.0141 | 0.1498 ± 0.0075 | 0.1058 ± 0.0055 |
| SCF-LightGCN+BGE | Tail | 53 | 0.2925 ± 0.0077 | 0.1627 ± 0.0015 | 0.1203 ± 0.0016 |
| SCF-LightGCN+BGE | Unseen | 73 | 0.2068 ± 0.0000 | 0.1076 ± 0.0001 | 0.0763 ± 0.0001 |

## Interpretation

- Higher coverage, long-tail coverage, personalization, entropy, and novelty indicate broader exposure.
- Lower Gini and lower average training frequency indicate weaker popularity concentration.
- Group-specific accuracy measures whether the method actually retrieves Head/Middle/Tail/Unseen ground-truth APIs; exposure alone does not imply useful long-tail recommendation.
- SCF improves both ranking accuracy and recommendation diversity: it expands catalog and long-tail coverage, reduces Gini concentration and average popularity, and substantially lowers Head exposure while keeping Head accuracy nearly unchanged.
- Graph+BGE and Inductive LightGCN have zero Unseen recall, whereas SCF obtains non-zero Unseen recall through its direct BGE branch. This provides evidence of partial zero-shot new-API cold-start capability, although Unseen performance remains lower than Head, Middle, and Tail performance.
