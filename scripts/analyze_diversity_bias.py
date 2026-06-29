#!/usr/bin/env python3
"""
Coverage, popularity-bias, concentration and personalization analysis for the
strict cold-start recommendation experiment.

Input layout
------------
outputs/statistical_analysis/runs/
├── graph_only/seed0..2/
├── bge_only/seed0..2/
└── graph_bge_zscore/seed0..2/

Each seed directory must contain:
- rankings_topk.csv
- api_groups.csv

Methods
-------
- Popularity
- Graph-only
- BGE-only
- Graph+BGE

Metrics
-------
- Catalog Coverage@K
- Seen Catalog Coverage@K
- Long-tail Catalog Coverage@K
- Unique Lists@K
- Unique List Ratio@K
- Average Pairwise Jaccard@K
- Personalization@K
- Gini@K
- Normalized Entropy@K
- Average Recommendation Popularity@K
- Average log(1+Popularity)@K
- Novelty@K
- Head/Middle/Tail/Unseen exposure ratios
- Per-group catalog coverage
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


METHODS = {
    "Popularity": ("graph_bge_zscore", "Popularity"),
    "Graph-only": ("graph_only", "Ours"),
    "BGE-only": ("bge_only", "Ours"),
    "Graph+BGE": ("graph_bge_zscore", "Ours"),
}

GROUPS = ("Head", "Middle", "Tail", "Unseen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/statistical_analysis/runs"),
        help="Directory containing the per-method, per-seed analyzer outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/diversity_analysis/summary"),
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10],
    )
    return parser.parse_args()


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient over all candidate APIs.

    0 means perfectly uniform recommendation frequency.
    Values closer to 1 mean highly concentrated recommendations.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("Gini input must be one-dimensional")
    if np.any(x < 0):
        raise ValueError("Gini input must be non-negative")
    total = float(x.sum())
    if total == 0.0:
        return 0.0

    x = np.sort(x)
    n = x.size
    ranks = np.arange(1, n + 1, dtype=float)
    value = np.sum((2.0 * ranks - n - 1.0) * x) / (n * total)
    return float(value)


def normalized_entropy(counts: np.ndarray) -> float:
    """
    Shannon entropy normalized by log(number of candidate APIs).

    0 means all recommendations go to one API.
    1 means recommendation exposure is perfectly uniform over the catalog.
    """
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total == 0.0 or counts.size <= 1:
        return 0.0

    probabilities = counts[counts > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(counts.size)


def build_ranked_lists(frame: pd.DataFrame, k: int) -> Dict[int, Tuple[int, ...]]:
    topk = frame[frame["rank"] <= k].copy()
    topk = topk.sort_values(["mashup_id", "rank"])

    lists: Dict[int, Tuple[int, ...]] = {}
    for mashup_id, group in topk.groupby("mashup_id", sort=True):
        api_ids = tuple(int(value) for value in group["api_id"].tolist())
        if len(api_ids) != k:
            raise ValueError(
                f"Mashup {mashup_id} has {len(api_ids)} recommendations "
                f"at K={k}; expected {k}"
            )
        if len(set(api_ids)) != len(api_ids):
            raise ValueError(
                f"Mashup {mashup_id} contains duplicate APIs at K={k}"
            )
        lists[int(mashup_id)] = api_ids

    if not lists:
        raise ValueError(f"No ranking lists found for K={k}")
    return lists


def exact_pairwise_jaccard(
    ranked_lists: Mapping[int, Sequence[int]],
    num_apis: int,
) -> Tuple[float, float]:
    """
    Compute exact average pairwise Jaccard similarity and personalization.

    Personalization is defined as:
        1 - average pairwise Jaccard similarity

    This is exact rather than sampled. With 1,645 Mashups the dense pairwise
    intersection matrix is small enough for this analysis.
    """
    mashup_ids = sorted(ranked_lists)
    row_indices: List[int] = []
    col_indices: List[int] = []

    for row_index, mashup_id in enumerate(mashup_ids):
        for api_id in ranked_lists[mashup_id]:
            row_indices.append(row_index)
            col_indices.append(int(api_id))

    data = np.ones(len(row_indices), dtype=np.int16)
    matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(mashup_ids), num_apis),
        dtype=np.int16,
    )

    intersections = (matrix @ matrix.T).toarray().astype(float)
    row_sizes = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(float)
    unions = row_sizes[:, None] + row_sizes[None, :] - intersections

    upper = np.triu_indices(len(mashup_ids), k=1)
    if upper[0].size == 0:
        return 0.0, 0.0

    similarities = np.divide(
        intersections[upper],
        unions[upper],
        out=np.zeros_like(intersections[upper], dtype=float),
        where=unions[upper] > 0,
    )
    avg_jaccard = float(similarities.mean())
    personalization = 1.0 - avg_jaccard
    return avg_jaccard, personalization


def load_seed_data(
    root: Path,
    method_name: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method_dir, source_method = METHODS[method_name]
    run_dir = root / method_dir / f"seed{seed}"

    rankings_path = run_dir / "rankings_topk.csv"
    api_groups_path = run_dir / "api_groups.csv"

    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)
    if not api_groups_path.exists():
        raise FileNotFoundError(api_groups_path)

    rankings = pd.read_csv(rankings_path)
    api_groups = pd.read_csv(api_groups_path)

    required_rankings = {
        "method",
        "mashup_id",
        "rank",
        "api_id",
        "is_positive",
        "api_group",
    }
    missing_rankings = required_rankings - set(rankings.columns)
    if missing_rankings:
        raise ValueError(
            f"{rankings_path} missing columns: {sorted(missing_rankings)}"
        )

    required_groups = {"api_id", "train_frequency", "group"}
    missing_groups = required_groups - set(api_groups.columns)
    if missing_groups:
        raise ValueError(
            f"{api_groups_path} missing columns: {sorted(missing_groups)}"
        )

    rankings = rankings[rankings["method"] == source_method].copy()
    if rankings.empty:
        raise ValueError(
            f"No method={source_method!r} rows in {rankings_path}"
        )

    api_groups = api_groups.sort_values("api_id").reset_index(drop=True)
    expected_api_ids = np.arange(len(api_groups))
    actual_api_ids = api_groups["api_id"].to_numpy(dtype=int)
    if not np.array_equal(actual_api_ids, expected_api_ids):
        raise ValueError(
            "api_groups.csv must contain contiguous API IDs from 0 to N-1"
        )

    rankings["mashup_id"] = rankings["mashup_id"].astype(int)
    rankings["rank"] = rankings["rank"].astype(int)
    rankings["api_id"] = rankings["api_id"].astype(int)
    return rankings, api_groups


def analyze_method_seed(
    rankings: pd.DataFrame,
    api_groups: pd.DataFrame,
    method_name: str,
    seed: int,
    ks: Sequence[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    num_apis = len(api_groups)
    train_frequency = api_groups["train_frequency"].to_numpy(dtype=float)
    api_group = api_groups["group"].astype(str).to_numpy()
    total_train_interactions = float(train_frequency.sum())

    group_catalog_sizes = {
        group: int(np.sum(api_group == group))
        for group in GROUPS
    }
    seen_catalog_size = int(np.sum(train_frequency > 0))
    long_tail_catalog_size = (
        group_catalog_sizes["Middle"] + group_catalog_sizes["Tail"]
    )

    metric_rows: List[Dict] = []
    exposure_rows: List[Dict] = []
    frequency_rows: List[Dict] = []

    for k in ks:
        ranked_lists = build_ranked_lists(rankings, k)
        mashup_ids = sorted(ranked_lists)
        flat_api_ids = np.asarray(
            [
                api_id
                for mashup_id in mashup_ids
                for api_id in ranked_lists[mashup_id]
            ],
            dtype=int,
        )

        counts = np.bincount(flat_api_ids, minlength=num_apis).astype(float)
        unique_recommended = int(np.count_nonzero(counts))
        total_slots = int(flat_api_ids.size)
        unique_lists = len(set(ranked_lists.values()))
        num_mashups = len(ranked_lists)

        avg_jaccard, personalization = exact_pairwise_jaccard(
            ranked_lists,
            num_apis,
        )

        smoothed_probability = (
            (train_frequency + 1.0)
            / (total_train_interactions + num_apis)
        )
        api_novelty = -np.log2(smoothed_probability)

        seen_unique = int(
            np.count_nonzero((counts > 0) & (train_frequency > 0))
        )
        long_tail_unique = int(
            np.count_nonzero(
                (counts > 0)
                & np.isin(api_group, ["Middle", "Tail"])
            )
        )

        metric_rows.append(
            {
                "method": method_name,
                "seed": seed,
                "K": k,
                "num_mashups": num_mashups,
                "catalog_size": num_apis,
                "total_recommendation_slots": total_slots,
                "unique_recommended_apis": unique_recommended,
                "catalog_coverage": unique_recommended / num_apis,
                "seen_catalog_coverage": (
                    seen_unique / seen_catalog_size
                    if seen_catalog_size > 0
                    else 0.0
                ),
                "long_tail_catalog_coverage": (
                    long_tail_unique / long_tail_catalog_size
                    if long_tail_catalog_size > 0
                    else 0.0
                ),
                "unique_lists": unique_lists,
                "unique_list_ratio": unique_lists / num_mashups,
                "avg_pairwise_jaccard": avg_jaccard,
                "personalization": personalization,
                "gini": gini_coefficient(counts),
                "normalized_entropy": normalized_entropy(counts),
                "effective_catalog_size": float(
                    math.exp(
                        normalized_entropy(counts) * math.log(num_apis)
                    )
                ),
                "avg_train_frequency": float(
                    train_frequency[flat_api_ids].mean()
                ),
                "avg_log1p_train_frequency": float(
                    np.log1p(train_frequency[flat_api_ids]).mean()
                ),
                "novelty": float(api_novelty[flat_api_ids].mean()),
            }
        )

        for group in GROUPS:
            group_mask = api_group[flat_api_ids] == group
            group_slots = int(np.sum(group_mask))
            catalog_mask = api_group == group
            group_unique = int(
                np.count_nonzero((counts > 0) & catalog_mask)
            )
            group_catalog_size = group_catalog_sizes[group]

            exposure_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "K": k,
                    "group": group,
                    "slot_count": group_slots,
                    "slot_ratio": (
                        group_slots / total_slots
                        if total_slots > 0
                        else 0.0
                    ),
                    "unique_recommended_apis": group_unique,
                    "group_catalog_size": group_catalog_size,
                    "group_catalog_coverage": (
                        group_unique / group_catalog_size
                        if group_catalog_size > 0
                        else 0.0
                    ),
                }
            )

        for api_id in range(num_apis):
            frequency_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "K": k,
                    "api_id": api_id,
                    "api_group": api_group[api_id],
                    "train_frequency": int(train_frequency[api_id]),
                    "recommendation_frequency": int(counts[api_id]),
                    "recommendation_share": (
                        counts[api_id] / total_slots
                        if total_slots > 0
                        else 0.0
                    ),
                }
            )

    return metric_rows, exposure_rows, frequency_rows


def population_std(series: pd.Series) -> float:
    return float(np.std(series.to_numpy(dtype=float), ddof=0))


def aggregate_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "method",
        "K",
        "num_mashups",
        "catalog_size",
        "total_recommendation_slots",
    ]
    value_columns = [
        column
        for column in frame.columns
        if column not in identity + ["seed"]
    ]

    rows: List[Dict] = []
    for (method, k), subset in frame.groupby(
        ["method", "K"],
        sort=False,
    ):
        row = {
            "method": method,
            "K": int(k),
            "seeds": int(subset["seed"].nunique()),
            "num_mashups": int(subset["num_mashups"].iloc[0]),
            "catalog_size": int(subset["catalog_size"].iloc[0]),
            "total_recommendation_slots": int(
                subset["total_recommendation_slots"].iloc[0]
            ),
        }
        for column in value_columns:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_std"] = population_std(subset[column])
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_exposure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    numeric_columns = [
        "slot_count",
        "slot_ratio",
        "unique_recommended_apis",
        "group_catalog_size",
        "group_catalog_coverage",
    ]

    for (method, k, group), subset in frame.groupby(
        ["method", "K", "group"],
        sort=False,
    ):
        row = {
            "method": method,
            "K": int(k),
            "group": group,
            "seeds": int(subset["seed"].nunique()),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_std"] = population_std(subset[column])
        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


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


def build_report(
    metric_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> str:
    report: List[str] = [
        "# Strict Cold-Start Coverage and Diversity Analysis",
        "",
        "All values are mean ± population standard deviation over seeds 0/1/2.",
        "",
        "## Main diversity and popularity-bias metrics",
        "",
    ]

    for k in sorted(metric_summary["K"].unique()):
        subset = metric_summary[metric_summary["K"] == k]
        table = pd.DataFrame(
            {
                "Method": subset["method"],
                "Coverage": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["catalog_coverage_mean"],
                        subset["catalog_coverage_std"],
                    )
                ],
                "Unique-list ratio": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["unique_list_ratio_mean"],
                        subset["unique_list_ratio_std"],
                    )
                ],
                "Personalization": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["personalization_mean"],
                        subset["personalization_std"],
                    )
                ],
                "Gini": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["gini_mean"],
                        subset["gini_std"],
                    )
                ],
                "Norm. entropy": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["normalized_entropy_mean"],
                        subset["normalized_entropy_std"],
                    )
                ],
                "Avg. popularity": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["avg_train_frequency_mean"],
                        subset["avg_train_frequency_std"],
                    )
                ],
                "Novelty": [
                    format_mean_std(mean, std)
                    for mean, std in zip(
                        subset["novelty_mean"],
                        subset["novelty_std"],
                    )
                ],
            }
        )
        report += [
            f"### K={k}",
            "",
            markdown_table(table),
            "",
        ]

    report += [
        "## Exposure by API popularity group",
        "",
    ]

    for k in sorted(exposure_summary["K"].unique()):
        subset = exposure_summary[exposure_summary["K"] == k]
        pivot_rows: List[Dict] = []
        for method in METHODS:
            method_rows = subset[subset["method"] == method]
            row: Dict[str, str] = {"Method": method}
            for group in GROUPS:
                group_row = method_rows[method_rows["group"] == group].iloc[0]
                row[group] = format_mean_std(
                    float(group_row["slot_ratio_mean"]),
                    float(group_row["slot_ratio_std"]),
                )
            pivot_rows.append(row)

        report += [
            f"### K={k} recommendation-slot ratio",
            "",
            markdown_table(pd.DataFrame(pivot_rows)),
            "",
        ]

    report += [
        "## Interpretation",
        "",
        "- Higher catalog coverage indicates that more distinct APIs are exposed.",
        "- Higher unique-list ratio and personalization indicate less identical "
        "recommendation across Mashups.",
        "- Lower Gini indicates a less concentrated recommendation distribution.",
        "- Higher normalized entropy indicates more even catalog exposure.",
        "- Lower average training frequency and higher novelty indicate weaker "
        "popularity bias.",
        "- Head/Middle/Tail/Unseen exposure must be interpreted together with "
        "Recall/NDCG/MAP; diversity alone is not sufficient.",
        "",
    ]
    return "\n".join(report)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(set(args.ks)) != len(args.ks):
        raise ValueError("--ks contains duplicate values")
    if any(k <= 0 for k in args.ks):
        raise ValueError("All K values must be positive")

    metric_rows: List[Dict] = []
    exposure_rows: List[Dict] = []
    frequency_rows: List[Dict] = []

    reference_catalog: pd.DataFrame | None = None

    for method_name in METHODS:
        for seed in (0, 1, 2):
            rankings, api_groups = load_seed_data(
                root,
                method_name,
                seed,
            )

            if reference_catalog is None:
                reference_catalog = api_groups.copy()
            else:
                left = reference_catalog[
                    ["api_id", "train_frequency", "group"]
                ].reset_index(drop=True)
                right = api_groups[
                    ["api_id", "train_frequency", "group"]
                ].reset_index(drop=True)
                if not left.equals(right):
                    raise ValueError(
                        f"API catalog differs for {method_name}, seed={seed}"
                    )

            current_metrics, current_exposure, current_frequency = (
                analyze_method_seed(
                    rankings,
                    api_groups,
                    method_name,
                    seed,
                    args.ks,
                )
            )
            metric_rows.extend(current_metrics)
            exposure_rows.extend(current_exposure)
            frequency_rows.extend(current_frequency)

    metrics_by_seed = pd.DataFrame(metric_rows)
    exposure_by_seed = pd.DataFrame(exposure_rows)
    frequency_by_seed = pd.DataFrame(frequency_rows)

    metric_summary = aggregate_metric_rows(metrics_by_seed)
    exposure_summary = aggregate_exposure_rows(exposure_by_seed)

    metrics_by_seed.to_csv(
        output_dir / "diversity_metrics_by_seed.csv",
        index=False,
    )
    metric_summary.to_csv(
        output_dir / "diversity_metrics_mean_std.csv",
        index=False,
    )
    exposure_by_seed.to_csv(
        output_dir / "group_exposure_by_seed.csv",
        index=False,
    )
    exposure_summary.to_csv(
        output_dir / "group_exposure_mean_std.csv",
        index=False,
    )
    frequency_by_seed.to_csv(
        output_dir / "api_recommendation_frequency.csv",
        index=False,
    )

    report = build_report(metric_summary, exposure_summary)
    (output_dir / "diversity_report.md").write_text(
        report,
        encoding="utf-8",
    )

    display_columns = [
        "method",
        "K",
        "catalog_coverage_mean",
        "unique_list_ratio_mean",
        "personalization_mean",
        "gini_mean",
        "normalized_entropy_mean",
        "avg_train_frequency_mean",
        "novelty_mean",
    ]
    print("\nCoverage and diversity summary:")
    print(metric_summary[display_columns].to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
