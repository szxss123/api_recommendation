# Strict Cold-Start Coverage and Diversity Analysis

All values are mean ± population standard deviation over seeds 0/1/2.

## Main diversity and popularity-bias metrics

### K=5

| Method | Coverage | Unique-list ratio | Personalization | Gini | Norm. entropy | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0030 ± 0.0000 | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 | 0.9970 ± 0.0000 | 0.2173 ± 0.0000 | 770.4000 ± 0.0000 | 4.4311 ± 0.0000 |
| Graph-only | 0.0030 ± 0.0000 | 0.0008 ± 0.0003 | 0.0000 ± 0.0000 | 0.9970 ± 0.0000 | 0.2173 ± 0.0000 | 770.4000 ± 0.0000 | 4.4311 ± 0.0000 |
| BGE-only | 0.7644 ± 0.0000 | 0.9982 ± 0.0000 | 0.9934 ± 0.0000 | 0.6713 ± 0.0000 | 0.8823 ± 0.0000 | 12.0635 ± 0.0000 | 11.8206 ± 0.0000 |
| Graph+BGE | 0.1018 ± 0.0222 | 0.9394 ± 0.0431 | 0.7835 ± 0.0944 | 0.9821 ± 0.0086 | 0.4666 ± 0.0660 | 472.3708 ± 123.9667 | 6.0845 ± 0.7072 |

### K=10

| Method | Coverage | Unique-list ratio | Personalization | Gini | Norm. entropy | Avg. popularity | Novelty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0061 ± 0.0000 | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 | 0.9939 ± 0.0000 | 0.3109 ± 0.0000 | 492.7000 ± 0.0000 | 5.2404 ± 0.0000 |
| Graph-only | 0.0063 ± 0.0003 | 0.0022 ± 0.0006 | 0.0303 ± 0.0428 | 0.9939 ± 0.0000 | 0.3140 ± 0.0044 | 489.8459 ± 2.5872 | 5.2775 ± 0.0357 |
| BGE-only | 0.8889 ± 0.0000 | 1.0000 ± 0.0000 | 0.9886 ± 0.0000 | 0.6329 ± 0.0000 | 0.8983 ± 0.0000 | 11.2245 ± 0.0000 | 11.8746 ± 0.0000 |
| Graph+BGE | 0.1305 ± 0.0180 | 1.0000 ± 0.0000 | 0.7534 ± 0.0876 | 0.9754 ± 0.0085 | 0.5245 ± 0.0464 | 342.5535 ± 75.2898 | 6.6681 ± 0.5348 |

## Exposure by API popularity group

### K=5 recommendation-slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Popularity | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Graph-only | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BGE-only | 0.2317 ± 0.0000 | 0.2941 ± 0.0000 | 0.3459 ± 0.0000 | 0.1283 ± 0.0000 |
| Graph+BGE | 0.9922 ± 0.0061 | 0.0073 ± 0.0058 | 0.0004 ± 0.0003 | 0.0000 ± 0.0000 |

### K=10 recommendation-slot ratio

| Method | Head | Middle | Tail | Unseen |
| --- | --- | --- | --- | --- |
| Popularity | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Graph-only | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BGE-only | 0.2217 ± 0.0000 | 0.2942 ± 0.0000 | 0.3578 ± 0.0000 | 0.1264 ± 0.0000 |
| Graph+BGE | 0.9907 ± 0.0059 | 0.0090 ± 0.0058 | 0.0003 ± 0.0001 | 0.0000 ± 0.0000 |

## Interpretation

- Higher catalog coverage indicates that more distinct APIs are exposed.
- Higher unique-list ratio and personalization indicate less identical recommendation across Mashups.
- Lower Gini indicates a less concentrated recommendation distribution.
- Higher normalized entropy indicates more even catalog exposure.
- Lower average training frequency and higher novelty indicate weaker popularity bias.
- Head/Middle/Tail/Unseen exposure must be interpreted together with Recall/NDCG/MAP; diversity alone is not sufficient.
