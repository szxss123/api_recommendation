#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon

from strict_baseline_core import load_strict_data


METHODS = (
    "Graph+BGE",
    "Inductive LightGCN",
    "SCF-LightGCN+BGE",
)

COMPARISONS = (
    ("SCF-LightGCN+BGE", "Inductive LightGCN"),
    ("SCF-LightGCN+BGE", "Graph+BGE"),
)

GROUPS = ("Head", "Middle", "Tail", "Unseen")
KS = (5, 10)
METRICS = ("Recall", "NDCG", "MAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired group-specific significance analysis for SCF. "
            "The statistical unit is one eligible test Mashup."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scf_group_significance.yaml"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def resolve_path(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def normalize_method(value: str) -> str:
    text = str(value).strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())

    if compact.startswith("scf") or (
        "lightgcn" in compact and "bge" in compact
    ):
        return "SCF-LightGCN+BGE"
    if "lightgcn" in compact:
        return "Inductive LightGCN"
    if (
        "graphbge" in compact
        or ("graph" in compact and "bge" in compact)
        or "zscore" in compact
        or "fusion" in compact
    ):
        return "Graph+BGE"
    if compact == "ours":
        return "Ours"
    return text


def load_rankings(
    *,
    method: str,
    seed: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if method == "Graph+BGE":
        template = str(config["graph_bge_rankings_template"])
        source_label = "Ours"
    elif method == "Inductive LightGCN":
        template = str(config["lightgcn_rankings_template"])
        source_label = "Inductive LightGCN"
    elif method == "SCF-LightGCN+BGE":
        template = str(config["scf_rankings_template"])
        source_label = None
    else:
        raise ValueError(f"Unsupported method: {method}")

    path = resolve_path(
        template.format(seed=seed),
        f"{method} seed-{seed} ranking file",
    )
    frame = pd.read_csv(path)

    required = {"mashup_id", "rank", "api_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if "method" in frame.columns:
        normalized = frame["method"].map(normalize_method)
        if method == "Graph+BGE":
            candidate = frame[
                normalized.isin({"Graph+BGE", source_label})
            ].copy()
        else:
            candidate = frame[normalized == method].copy()
            if candidate.empty and method == "SCF-LightGCN+BGE":
                candidate = frame.copy()
        frame = candidate

    if frame.empty:
        raise ValueError(f"No {method} rows found in {path}")

    frame["mashup_id"] = frame["mashup_id"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["api_id"] = frame["api_id"].astype(int)
    frame = frame[frame["rank"] <= 10].copy()
    frame["method"] = method
    frame["seed"] = seed

    duplicate_rank = frame.duplicated(
        ["mashup_id", "rank"],
        keep=False,
    )
    if duplicate_rank.any():
        raise ValueError(
            f"Duplicate Mashup/rank rows found in {path}"
        )

    duplicate_api = frame.duplicated(
        ["mashup_id", "api_id"],
        keep=False,
    )
    if duplicate_api.any():
        raise ValueError(
            f"Duplicate API recommendations found in {path}"
        )

    return frame[
        ["method", "seed", "mashup_id", "rank", "api_id"]
    ].sort_values(["mashup_id", "rank"])


def load_api_groups(path: Path, num_apis: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"api_id", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    columns = ["api_id", "group"]
    if "train_frequency" in frame.columns:
        columns.append("train_frequency")

    frame = frame[columns].copy()
    frame["api_id"] = frame["api_id"].astype(int)
    frame["group"] = frame["group"].astype(str)

    if frame["api_id"].nunique() != num_apis:
        raise ValueError(
            f"API group count {frame['api_id'].nunique()} "
            f"does not match catalog size {num_apis}"
        )

    unknown_groups = sorted(set(frame["group"]) - set(GROUPS))
    if unknown_groups:
        raise ValueError(
            f"Unknown API groups: {unknown_groups}"
        )

    return frame.sort_values("api_id").reset_index(drop=True)


def positives_by_mashup_and_group(
    pairs: np.ndarray,
    group_by_api: Mapping[int, str],
) -> Dict[int, Dict[str, set[int]]]:
    result: Dict[int, Dict[str, set[int]]] = {}

    for mashup_id, api_id in pairs:
        mashup_id = int(mashup_id)
        api_id = int(api_id)
        group = str(group_by_api[api_id])

        result.setdefault(
            mashup_id,
            {name: set() for name in GROUPS},
        )[group].add(api_id)

    return result


def strict_clean_ids(
    audit_path: Path,
    threshold: float,
) -> set[int]:
    audit = pd.read_csv(audit_path)
    required = {
        "mashup_id",
        "exact_train_text_duplicate",
        "exact_train_name_duplicate",
        "direct_api_name_mention_count",
        "top1_cosine_similarity",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            f"{audit_path} is missing columns: {sorted(missing)}"
        )

    mask = (
        (audit["exact_train_text_duplicate"] == 0)
        & (audit["exact_train_name_duplicate"] == 0)
        & (audit["direct_api_name_mention_count"] == 0)
        & (audit["top1_cosine_similarity"] < threshold)
    )
    return set(
        audit.loc[mask, "mashup_id"].astype(int).tolist()
    )


def dcg(binary_hits: Sequence[int]) -> float:
    return float(
        sum(
            hit / math.log2(rank + 1.0)
            for rank, hit in enumerate(binary_hits, start=1)
        )
    )


def per_mashup_group_metrics(
    *,
    rankings: pd.DataFrame,
    positives: Mapping[int, Mapping[str, set[int]]],
    method: str,
    seed: int,
) -> List[Dict[str, Any]]:
    ranking_map: Dict[int, List[int]] = {
        int(mashup_id): [
            int(value)
            for value in group.sort_values("rank")["api_id"]
        ]
        for mashup_id, group in rankings.groupby(
            "mashup_id",
            sort=True,
        )
    }

    rows: List[Dict[str, Any]] = []

    for mashup_id, grouped_positives in positives.items():
        if mashup_id not in ranking_map:
            continue

        ranked = ranking_map[mashup_id]

        for group in GROUPS:
            relevant = grouped_positives[group]
            if not relevant:
                continue

            row: Dict[str, Any] = {
                "method": method,
                "seed": seed,
                "mashup_id": mashup_id,
                "group": group,
                "positive_count": len(relevant),
            }

            for k in KS:
                topk = ranked[:k]
                binary_hits = [
                    int(api_id in relevant)
                    for api_id in topk
                ]
                hit_count = int(sum(binary_hits))

                recall = hit_count / len(relevant)

                ideal_hits = [1] * min(len(relevant), k)
                ideal_dcg = dcg(ideal_hits)
                ndcg = (
                    dcg(binary_hits) / ideal_dcg
                    if ideal_dcg > 0
                    else 0.0
                )

                precision_sum = 0.0
                cumulative_hits = 0
                for rank, hit in enumerate(binary_hits, start=1):
                    if hit:
                        cumulative_hits += 1
                        precision_sum += cumulative_hits / rank
                average_precision = (
                    precision_sum / min(len(relevant), k)
                )

                row[f"Recall@{k}"] = recall
                row[f"NDCG@{k}"] = ndcg
                row[f"MAP@{k}"] = average_precision

            rows.append(row)

    return rows


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (
        (positive - negative) / denominator
        if denominator > 0
        else 0.0
    )


def paired_bootstrap_ci(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    count = len(differences)
    means = np.empty(samples, dtype=np.float64)

    cursor = 0
    batch_size = 256
    while cursor < samples:
        current = min(batch_size, samples - cursor)
        indices = rng.integers(
            0,
            count,
            size=(current, count),
        )
        means[cursor : cursor + current] = (
            differences[indices].mean(axis=1)
        )
        cursor += current

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (total - rank) * p_values[index],
        )
        running = max(running, candidate)
        adjusted[index] = running

    return adjusted


def significance_stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def analyze_subset(
    *,
    averaged: pd.DataFrame,
    subset_name: str,
    allowed_ids: set[int] | None,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    working = averaged.copy()
    if allowed_ids is not None:
        working = working[
            working["mashup_id"].isin(allowed_ids)
        ].copy()

    metric_names = [
        f"{metric}@{k}"
        for k in KS
        for metric in METRICS
    ]

    rows: List[Dict[str, Any]] = []

    for method_a, method_b in COMPARISONS:
        for group in GROUPS:
            group_frame = working[
                working["group"] == group
            ]

            for metric in metric_names:
                left = group_frame[
                    group_frame["method"] == method_a
                ][["mashup_id", metric]].rename(
                    columns={metric: "a"}
                )
                right = group_frame[
                    group_frame["method"] == method_b
                ][["mashup_id", metric]].rename(
                    columns={metric: "b"}
                )

                paired = left.merge(
                    right,
                    on="mashup_id",
                    how="inner",
                )
                if paired.empty:
                    continue

                differences = (
                    paired["a"].to_numpy(dtype=np.float64)
                    - paired["b"].to_numpy(dtype=np.float64)
                )

                if np.allclose(differences, 0):
                    statistic = 0.0
                    p_raw = 1.0
                else:
                    result = wilcoxon(
                        differences,
                        zero_method="wilcox",
                        alternative="two-sided",
                        correction=False,
                        method="auto",
                    )
                    statistic = float(result.statistic)
                    p_raw = float(result.pvalue)

                ci_low, ci_high = paired_bootstrap_ci(
                    differences,
                    bootstrap_samples,
                    rng,
                )

                rows.append(
                    {
                        "subset": subset_name,
                        "method_a": method_a,
                        "method_b": method_b,
                        "group": group,
                        "metric": metric,
                        "num_mashups": int(len(paired)),
                        "mean_a": float(paired["a"].mean()),
                        "mean_b": float(paired["b"].mean()),
                        "mean_difference": float(
                            differences.mean()
                        ),
                        "relative_improvement_percent": float(
                            differences.mean()
                            / max(float(paired["b"].mean()), 1e-12)
                            * 100.0
                        ),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_raw,
                        "rank_biserial": rank_biserial(
                            differences
                        ),
                        "method_a_win_rate": float(
                            (differences > 0).mean()
                        ),
                        "tie_rate": float(
                            (differences == 0).mean()
                        ),
                    }
                )

    result_frame = pd.DataFrame(rows)
    if result_frame.empty:
        return result_frame

    # Primary correction: all 48 tests within a subset form one family.
    result_frame["p_holm_subset_global"] = holm_adjust(
        result_frame["p_raw"]
    )

    # Secondary diagnostic correction: 24 tests within each comparison.
    result_frame["p_holm_comparison"] = np.nan
    for _, indices in result_frame.groupby(
        ["method_a", "method_b"]
    ).groups.items():
        index_list = list(indices)
        result_frame.loc[
            index_list,
            "p_holm_comparison",
        ] = holm_adjust(
            result_frame.loc[index_list, "p_raw"]
        )

    result_frame["significance_global"] = (
        result_frame["p_holm_subset_global"].map(
            significance_stars
        )
    )
    result_frame["ci_excludes_zero"] = (
        (result_frame["bootstrap_ci_low"] > 0)
        | (result_frame["bootstrap_ci_high"] < 0)
    )

    return result_frame


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in frame.iterrows():
        values = [
            str(row[column])
            .replace("|", r"\|")
            .replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def build_report(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    strict_clean_count: int,
) -> str:
    report: List[str] = [
        "# SCF Head/Middle/Tail/Unseen Group Significance Analysis",
        "",
        "Statistical unit: one eligible test Mashup containing at least one "
        "ground-truth API in the target group. Each per-Mashup metric is first "
        "averaged over seeds 0/1/2. Paired Wilcoxon signed-rank tests, 10,000 "
        "paired bootstrap confidence intervals, Holm-Bonferroni correction, "
        "and rank-biserial effect sizes are then applied.",
        "",
        "The primary p-value is `p_holm_subset_global`, which corrects all "
        "48 group/metric/comparison tests within each subset as one family.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        f"Strict-clean Mashups available: {strict_clean_count}.",
        "",
    ]

    for subset_name in ("full", "strict_clean"):
        subset = results[
            results["subset"] == subset_name
        ]
        if subset.empty:
            continue

        report.extend(
            [
                f"## Subset: {subset_name}",
                "",
            ]
        )

        for method_a, method_b in COMPARISONS:
            comparison = subset[
                (subset["method_a"] == method_a)
                & (subset["method_b"] == method_b)
            ].copy()
            if comparison.empty:
                continue

            report.extend(
                [
                    f"### {method_a} vs {method_b}",
                    "",
                ]
            )

            display = comparison[
                [
                    "group",
                    "metric",
                    "num_mashups",
                    "mean_a",
                    "mean_b",
                    "mean_difference",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "p_holm_subset_global",
                    "rank_biserial",
                    "significance_global",
                ]
            ].copy()

            for column in (
                "mean_a",
                "mean_b",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "rank_biserial",
            ):
                display[column] = display[column].map(
                    lambda value: f"{float(value):.4f}"
                )

            display["p_holm_subset_global"] = display[
                "p_holm_subset_global"
            ].map(format_p)

            report.extend(
                [
                    markdown_table(display),
                    "",
                ]
            )

        report.extend(
            [
                "### Interpretation checklist",
                "",
                "- A positive mean difference favors SCF.",
                "- A bootstrap confidence interval entirely above zero supports "
                "a positive average improvement.",
                "- The formal significance claim requires "
                "`p_holm_subset_global < 0.05`.",
                "- Unseen conclusions must report the eligible Mashup count; "
                "small samples limit generalization.",
                "",
            ]
        )

    report.extend(
        [
            "## Reporting rules",
            "",
            "- Report group metrics only over Mashups that contain at least one "
            "ground-truth API in that group.",
            "- Use `interaction-unseen API` or `API unseen in training "
            "interactions`; do not claim the pretrained text encoder never saw "
            "the concept.",
            "- If strict-clean results are numerically positive but not "
            "Holm-significant, describe them as a positive trend rather than a "
            "statistically significant improvement.",
            "",
        ]
    )

    return "\n".join(report)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = resolve_path(
        str(config["reference_config"]),
        "strict reference configuration",
    )
    data = load_strict_data(reference_config)

    api_groups_path = resolve_path(
        str(config["api_groups_source"]),
        "API group metadata",
    )
    api_groups = load_api_groups(
        api_groups_path,
        data.num_apis,
    )
    group_by_api = (
        api_groups.set_index("api_id")["group"].to_dict()
    )

    positives = positives_by_mashup_and_group(
        data.test_pairs,
        group_by_api,
    )

    all_rows: List[Dict[str, Any]] = []
    source_coverage_rows: List[Dict[str, Any]] = []

    for method in METHODS:
        for seed in (0, 1, 2):
            rankings = load_rankings(
                method=method,
                seed=seed,
                config=config,
            )

            source_coverage_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "num_mashups": int(
                        rankings["mashup_id"].nunique()
                    ),
                    "num_rows": int(len(rankings)),
                    "minimum_rank": int(
                        rankings["rank"].min()
                    ),
                    "maximum_rank": int(
                        rankings["rank"].max()
                    ),
                }
            )

            all_rows.extend(
                per_mashup_group_metrics(
                    rankings=rankings,
                    positives=positives,
                    method=method,
                    seed=seed,
                )
            )

    by_seed = pd.DataFrame(all_rows)
    by_seed.to_csv(
        output_dir / "group_per_mashup_metrics_by_seed.csv",
        index=False,
    )

    metric_columns = [
        f"{metric}@{k}"
        for k in KS
        for metric in METRICS
    ]

    averaged = (
        by_seed.groupby(
            [
                "method",
                "mashup_id",
                "group",
                "positive_count",
            ],
            as_index=False,
        )[metric_columns]
        .mean()
    )
    averaged.to_csv(
        output_dir / "group_per_mashup_seed_averaged.csv",
        index=False,
    )

    audit_path = resolve_path(
        str(config["audit_csv"]),
        "strict leakage audit CSV",
    )
    clean_ids = strict_clean_ids(
        audit_path,
        float(config["strict_clean_cosine_threshold"]),
    )

    rng = np.random.default_rng(int(config["random_seed"]))

    full_results = analyze_subset(
        averaged=averaged,
        subset_name="full",
        allowed_ids=None,
        bootstrap_samples=int(config["bootstrap_samples"]),
        rng=rng,
    )
    clean_results = analyze_subset(
        averaged=averaged,
        subset_name="strict_clean",
        allowed_ids=clean_ids,
        bootstrap_samples=int(config["bootstrap_samples"]),
        rng=rng,
    )

    results = pd.concat(
        [full_results, clean_results],
        ignore_index=True,
    )
    results.to_csv(
        output_dir / "group_significance_results.csv",
        index=False,
    )

    coverage = pd.DataFrame(source_coverage_rows)
    coverage.to_csv(
        output_dir / "group_significance_source_coverage.csv",
        index=False,
    )

    eligible_rows: List[Dict[str, Any]] = []
    for subset_name, allowed_ids in (
        ("full", None),
        ("strict_clean", clean_ids),
    ):
        working = averaged
        if allowed_ids is not None:
            working = working[
                working["mashup_id"].isin(allowed_ids)
            ]
        for group in GROUPS:
            group_frame = working[
                (working["method"] == "SCF-LightGCN+BGE")
                & (working["group"] == group)
            ]
            eligible_rows.append(
                {
                    "subset": subset_name,
                    "group": group,
                    "eligible_mashups": int(
                        group_frame["mashup_id"].nunique()
                    ),
                    "mean_positive_apis_per_mashup": float(
                        group_frame["positive_count"].mean()
                    )
                    if not group_frame.empty
                    else 0.0,
                }
            )

    eligible = pd.DataFrame(eligible_rows)
    eligible.to_csv(
        output_dir / "group_eligible_mashup_counts.csv",
        index=False,
    )

    report = build_report(
        results,
        coverage,
        len(clean_ids),
    )
    (output_dir / "group_significance_report.md").write_text(
        report,
        encoding="utf-8",
    )

    summary = {
        "comparisons": [
            {"method_a": a, "method_b": b}
            for a, b in COMPARISONS
        ],
        "groups": list(GROUPS),
        "metrics": metric_columns,
        "subsets": ["full", "strict_clean"],
        "bootstrap_samples": int(
            config["bootstrap_samples"]
        ),
        "holm_primary_family": (
            "all 48 tests within each subset"
        ),
        "strict_clean_mashups": len(clean_ids),
    }
    (
        output_dir / "group_significance_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    key = results[
        results["metric"].isin(
            {"Recall@10", "NDCG@10", "MAP@10"}
        )
    ][
        [
            "subset",
            "method_a",
            "method_b",
            "group",
            "metric",
            "num_mashups",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_holm_subset_global",
            "rank_biserial",
            "significance_global",
        ]
    ].copy()

    print(eligible.to_string(index=False))
    print()
    print(key.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
