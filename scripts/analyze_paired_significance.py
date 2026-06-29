#!/usr/bin/env python3
"""
Paired statistical significance analysis for strict cold-start recommendation.

Unit of analysis
----------------
A test Mashup is the statistical unit. For each method and Mashup, metrics are
first averaged over seeds. Paired tests are then conducted over the same set
of Mashups.

Comparisons
-----------
Graph+BGE vs Popularity
Graph+BGE vs Graph-only
Graph+BGE vs BGE-only

Outputs
-------
significance_results.csv
bootstrap_results.csv
method_metric_summary.csv
significance_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


METHOD_DIRS = {
    "Graph-only": "graph_only",
    "BGE-only": "bge_only",
    "Graph+BGE": "graph_bge_zscore",
}

DEFAULT_METRICS = [
    "Recall@5",
    "NDCG@5",
    "MAP@5",
    "Recall@10",
    "NDCG@10",
    "MAP@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/statistical_analysis/runs"),
        help="Root containing method/seedN/per_mashup_metrics.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/statistical_analysis/summary"),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help="Wilcoxon alternative for main-minus-baseline differences.",
    )
    return parser.parse_args()


def load_method_seed(
    root: Path,
    method_name: str,
    method_dir: str,
    seed: int,
) -> pd.DataFrame:
    path = root / method_dir / f"seed{seed}" / "per_mashup_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"method", "mashup_id", "num_positives"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    ours = frame[frame["method"] == "Ours"].copy()
    if ours.empty:
        raise ValueError(f"No method='Ours' rows found in {path}")

    if ours["mashup_id"].duplicated().any():
        raise ValueError(f"Duplicate Mashup IDs in {path}")

    ours["method_name"] = method_name
    ours["seed"] = seed
    return ours


def load_popularity_seed(root: Path, seed: int) -> pd.DataFrame:
    # Popularity is included in every analyzer output. Read it from the main
    # model run so that its Mashup ordering and split are guaranteed identical.
    path = (
        root
        / METHOD_DIRS["Graph+BGE"]
        / f"seed{seed}"
        / "per_mashup_metrics.csv"
    )
    frame = pd.read_csv(path)
    pop = frame[frame["method"] == "Popularity"].copy()
    if pop.empty:
        raise ValueError(f"No method='Popularity' rows found in {path}")
    if pop["mashup_id"].duplicated().any():
        raise ValueError(f"Duplicate Popularity Mashup IDs in {path}")
    pop["method_name"] = "Popularity"
    pop["seed"] = seed
    return pop


def average_over_seeds(frames: Iterable[pd.DataFrame], metrics: List[str]) -> pd.DataFrame:
    merged = pd.concat(list(frames), ignore_index=True)
    for metric in metrics:
        if metric not in merged.columns:
            raise ValueError(f"Missing metric column: {metric}")

    counts = merged.groupby("mashup_id")["seed"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 3:
        raise ValueError(
            "Every Mashup must have exactly 3 seeds. "
            f"Observed counts: {counts.value_counts().to_dict()}"
        )

    result = (
        merged.groupby("mashup_id", as_index=False)
        .agg(
            num_positives=("num_positives", "first"),
            **{metric: (metric, "mean") for metric in metrics},
        )
        .sort_values("mashup_id")
        .reset_index(drop=True)
    )
    return result


def holm_adjust(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni family-wise error correction.
    """
    n = len(p_values)
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0

    for rank, original_index in enumerate(order):
        multiplier = n - rank
        candidate = min(1.0, multiplier * float(p_values[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_bootstrap(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    n = differences.size
    if n == 0:
        raise ValueError("No paired observations")

    chunk_size = min(1000, samples)
    means: List[np.ndarray] = []
    completed = 0

    while completed < samples:
        current = min(chunk_size, samples - completed)
        indices = rng.integers(0, n, size=(current, n))
        means.append(differences[indices].mean(axis=1))
        completed += current

    bootstrap_means = np.concatenate(means)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    probability_positive = float(np.mean(bootstrap_means > 0.0))
    return (
        float(differences.mean()),
        float(lower),
        float(upper),
        probability_positive,
    )


def significance_mark(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    averaged: Dict[str, pd.DataFrame] = {}

    for method_name, method_dir in METHOD_DIRS.items():
        seed_frames = [
            load_method_seed(root, method_name, method_dir, seed)
            for seed in (0, 1, 2)
        ]
        averaged[method_name] = average_over_seeds(
            seed_frames,
            args.metrics,
        )

    popularity_frames = [
        load_popularity_seed(root, seed)
        for seed in (0, 1, 2)
    ]
    averaged["Popularity"] = average_over_seeds(
        popularity_frames,
        args.metrics,
    )

    reference_ids = averaged["Graph+BGE"]["mashup_id"].tolist()
    for method_name, frame in averaged.items():
        if frame["mashup_id"].tolist() != reference_ids:
            raise ValueError(
                f"Mashup alignment differs for {method_name}"
            )

    summary_rows = []
    for method_name, frame in averaged.items():
        row = {
            "method": method_name,
            "num_mashups": len(frame),
        }
        for metric in args.metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std_across_mashups"] = float(
                frame[metric].std(ddof=0)
            )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_dir / "method_metric_summary.csv",
        index=False,
    )

    comparisons = [
        ("Graph+BGE", "Popularity"),
        ("Graph+BGE", "Graph-only"),
        ("Graph+BGE", "BGE-only"),
    ]

    rng = np.random.default_rng(args.bootstrap_seed)
    result_rows = []
    bootstrap_rows = []

    for main_name, baseline_name in comparisons:
        main_frame = averaged[main_name]
        baseline_frame = averaged[baseline_name]

        for metric in args.metrics:
            main_values = main_frame[metric].to_numpy(dtype=float)
            baseline_values = baseline_frame[metric].to_numpy(dtype=float)
            differences = main_values - baseline_values

            nonzero_count = int(np.count_nonzero(differences))
            if nonzero_count == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    correction=False,
                    alternative=args.alternative,
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            mean_diff, ci_low, ci_high, prob_positive = paired_bootstrap(
                differences,
                args.bootstrap_samples,
                rng,
            )

            row = {
                "comparison": f"{main_name} vs {baseline_name}",
                "main_method": main_name,
                "baseline_method": baseline_name,
                "metric": metric,
                "num_mashups": int(differences.size),
                "main_mean": float(main_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "mean_difference": mean_diff,
                "relative_improvement_percent": (
                    (main_values.mean() / baseline_values.mean() - 1.0) * 100.0
                    if baseline_values.mean() != 0
                    else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial(differences),
                "positive_difference_ratio": float(
                    np.mean(differences > 0.0)
                ),
                "zero_difference_ratio": float(
                    np.mean(differences == 0.0)
                ),
                "bootstrap_samples": args.bootstrap_samples,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "bootstrap_probability_positive": prob_positive,
            }
            result_rows.append(row)
            bootstrap_rows.append(
                {
                    "comparison": row["comparison"],
                    "metric": metric,
                    "mean_difference": mean_diff,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "bootstrap_probability_positive": prob_positive,
                    "bootstrap_samples": args.bootstrap_samples,
                }
            )

    adjusted = holm_adjust([row["p_value"] for row in result_rows])
    for row, adjusted_p in zip(result_rows, adjusted):
        row["p_value_holm"] = adjusted_p
        row["significance"] = significance_mark(adjusted_p)
        row["ci_excludes_zero"] = bool(
            row["ci_95_low"] > 0.0 or row["ci_95_high"] < 0.0
        )

    results_df = pd.DataFrame(result_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    results_df.to_csv(
        output_dir / "significance_results.csv",
        index=False,
    )
    bootstrap_df.to_csv(
        output_dir / "bootstrap_results.csv",
        index=False,
    )

    report_table = pd.DataFrame(
        {
            "Comparison": results_df["comparison"],
            "Metric": results_df["metric"],
            "Main": results_df["main_mean"].map(lambda x: f"{x:.4f}"),
            "Baseline": results_df["baseline_mean"].map(lambda x: f"{x:.4f}"),
            "Mean diff.": results_df["mean_difference"].map(
                lambda x: f"{x:+.4f}"
            ),
            "95% CI": [
                f"[{low:+.4f}, {high:+.4f}]"
                for low, high in zip(
                    results_df["ci_95_low"],
                    results_df["ci_95_high"],
                )
            ],
            "Holm p": results_df["p_value_holm"].map(
                lambda x: "<0.001" if x < 0.001 else f"{x:.4g}"
            ),
            "Effect r_rb": results_df["rank_biserial"].map(
                lambda x: f"{x:.4f}"
            ),
            "Sig.": results_df["significance"],
        }
    )

    report = [
        "# Strict Cold-Start Paired Significance Analysis",
        "",
        "Statistical unit: one test Mashup.",
        "",
        "For every method, each Mashup metric is first averaged over seeds "
        "0/1/2. The paired Wilcoxon test and paired bootstrap are then "
        "performed over the same Mashups.",
        "",
        f"- Wilcoxon alternative: `{args.alternative}`",
        f"- Bootstrap samples: `{args.bootstrap_samples}`",
        "- Multiple-comparison correction: Holm-Bonferroni",
        "- `*`: p<0.05, `**`: p<0.01, `***`: p<0.001",
        "",
        "## Results",
        "",
        markdown_table(report_table),
        "",
        "## Interpretation rules",
        "",
        "- A positive mean difference favors Graph+BGE.",
        "- A 95% bootstrap confidence interval excluding zero supports a "
        "stable paired improvement.",
        "- Positive rank-biserial correlation favors Graph+BGE.",
        "- Use Holm-adjusted p-values in the paper.",
        "",
    ]
    (output_dir / "significance_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nPaired significance results:")
    print(report_table.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
