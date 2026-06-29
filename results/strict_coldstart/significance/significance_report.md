# Strict Cold-Start Paired Significance Analysis

Statistical unit: one test Mashup.

For every method, each Mashup metric is first averaged over seeds 0/1/2. The paired Wilcoxon test and paired bootstrap are then performed over the same Mashups.

- Wilcoxon alternative: `two-sided`
- Bootstrap samples: `10000`
- Multiple-comparison correction: Holm-Bonferroni
- `*`: p<0.05, `**`: p<0.01, `***`: p<0.001

## Results

| Comparison | Metric | Main | Baseline | Mean diff. | 95% CI | Holm p | Effect r_rb | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Graph+BGE vs Popularity | Recall@5 | 0.5722 | 0.3921 | +0.1801 | [+0.1625, +0.1979] | <0.001 | 0.7503 | *** |
| Graph+BGE vs Popularity | NDCG@5 | 0.5347 | 0.3602 | +0.1745 | [+0.1577, +0.1913] | <0.001 | 0.6147 | *** |
| Graph+BGE vs Popularity | MAP@5 | 0.4743 | 0.3107 | +0.1635 | [+0.1465, +0.1805] | <0.001 | 0.5586 | *** |
| Graph+BGE vs Popularity | Recall@10 | 0.6420 | 0.4821 | +0.1599 | [+0.1435, +0.1770] | <0.001 | 0.7736 | *** |
| Graph+BGE vs Popularity | NDCG@10 | 0.5591 | 0.3910 | +0.1681 | [+0.1529, +0.1836] | <0.001 | 0.6270 | *** |
| Graph+BGE vs Popularity | MAP@10 | 0.4855 | 0.3230 | +0.1625 | [+0.1463, +0.1792] | <0.001 | 0.5640 | *** |
| Graph+BGE vs Graph-only | Recall@5 | 0.5722 | 0.3921 | +0.1801 | [+0.1626, +0.1976] | <0.001 | 0.7503 | *** |
| Graph+BGE vs Graph-only | NDCG@5 | 0.5347 | 0.3599 | +0.1747 | [+0.1580, +0.1918] | <0.001 | 0.6162 | *** |
| Graph+BGE vs Graph-only | MAP@5 | 0.4743 | 0.3104 | +0.1639 | [+0.1474, +0.1811] | <0.001 | 0.5599 | *** |
| Graph+BGE vs Graph-only | Recall@10 | 0.6420 | 0.4813 | +0.1606 | [+0.1446, +0.1774] | <0.001 | 0.7755 | *** |
| Graph+BGE vs Graph-only | NDCG@10 | 0.5591 | 0.3899 | +0.1692 | [+0.1540, +0.1843] | <0.001 | 0.6271 | *** |
| Graph+BGE vs Graph-only | MAP@10 | 0.4855 | 0.3218 | +0.1638 | [+0.1471, +0.1799] | <0.001 | 0.5650 | *** |
| Graph+BGE vs BGE-only | Recall@5 | 0.5722 | 0.2545 | +0.3177 | [+0.2901, +0.3452] | <0.001 | 0.6033 | *** |
| Graph+BGE vs BGE-only | NDCG@5 | 0.5347 | 0.2214 | +0.3133 | [+0.2880, +0.3389] | <0.001 | 0.6266 | *** |
| Graph+BGE vs BGE-only | MAP@5 | 0.4743 | 0.1904 | +0.2839 | [+0.2599, +0.3079] | <0.001 | 0.6173 | *** |
| Graph+BGE vs BGE-only | Recall@10 | 0.6420 | 0.3152 | +0.3268 | [+0.2975, +0.3550] | <0.001 | 0.6039 | *** |
| Graph+BGE vs BGE-only | NDCG@10 | 0.5591 | 0.2435 | +0.3156 | [+0.2904, +0.3405] | <0.001 | 0.6296 | *** |
| Graph+BGE vs BGE-only | MAP@10 | 0.4855 | 0.1994 | +0.2861 | [+0.2622, +0.3097] | <0.001 | 0.6188 | *** |

## Interpretation rules

- A positive mean difference favors Graph+BGE.
- A 95% bootstrap confidence interval excluding zero supports a stable paired improvement.
- Positive rank-biserial correlation favors Graph+BGE.
- Use Holm-adjusted p-values in the paper.
