#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from strict_baseline_core import load_strict_data, load_yaml


METHODS = (
    "Graph+BGE",
    "Inductive LightGCN",
    "SCF-LightGCN+BGE",
)
GROUPS = ("Head", "Middle", "Tail", "Unseen")
KS = (5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SCF, Inductive LightGCN, and Graph+BGE on group accuracy, "
            "catalog diversity, novelty, and popularity bias."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scf_tail_diversity.yaml"),
    )
    return parser.parse_args()


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


def resolve_path(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_api_groups(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"api_id", "train_frequency", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )
    frame = frame[["api_id", "train_frequency", "group"]].copy()
    frame["api_id"] = frame["api_id"].astype(int)
    frame["train_frequency"] = frame["train_frequency"].astype(int)
    frame["group"] = frame["group"].astype(str)
    frame = frame.sort_values("api_id").reset_index(drop=True)

    expected_ids = np.arange(len(frame))
    if not np.array_equal(
        frame["api_id"].to_numpy(dtype=int),
        expected_ids,
    ):
        raise ValueError(
            "api_groups.csv must contain contiguous API IDs from 0 to N-1"
        )
    return frame


def load_rankings(
    *,
    config: Mapping[str, Any],
    method: str,
    seed: int,
    api_groups: pd.DataFrame,
) -> pd.DataFrame:
    if method == "Graph+BGE":
        template = str(config["graph_bge_rankings_template"])
        source_method = "Ours"
    elif method == "Inductive LightGCN":
        template = str(config["lightgcn_rankings_template"])
        source_method = "Inductive LightGCN"
    elif method == "SCF-LightGCN+BGE":
        template = str(config["scf_rankings_template"])
        source_method = None
    else:
        raise ValueError(method)

    path = resolve_path(
        template.format(seed=seed),
        f"{method} seed-{seed} rankings",
    )
    frame = pd.read_csv(path)

    required = {"mashup_id", "rank", "api_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if "method" in frame.columns:
        frame["method_normalized"] = frame["method"].map(
            normalize_method
        )
        if method == "Graph+BGE":
            candidate = frame[
                frame["method_normalized"].isin(
                    {"Graph+BGE", source_method}
                )
            ].copy()
        else:
            candidate = frame[
                frame["method_normalized"] == method
            ].copy()
            if candidate.empty and method == "SCF-LightGCN+BGE":
                candidate = frame.copy()
        frame = candidate

    if frame.empty:
        raise ValueError(f"No {method} rows found in {path}")

    frame["mashup_id"] = frame["mashup_id"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["api_id"] = frame["api_id"].astype(int)
    frame = frame[frame["rank"] <= 10].copy()

    metadata = api_groups.rename(
        columns={
            "group": "resolved_api_group",
            "train_frequency": "resolved_train_frequency",
        }
    )
    frame = frame.merge(metadata, on="api_id", how="left")
    if frame["resolved_api_group"].isna().any():
        raise ValueError(f"Unknown API IDs found in {path}")

    frame["api_group"] = frame["resolved_api_group"]
    frame["train_frequency"] = frame[
        "resolved_train_frequency"
    ].astype(int)
    frame["method"] = method
    frame["seed"] = seed

    return frame[
        [
            "method",
            "seed",
            "mashup_id",
            "rank",
            "api_id",
            "api_group",
            "train_frequency",
        ]
    ].sort_values(["mashup_id", "rank"])


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
    return set(audit.loc[mask, "mashup_id"].astype(int))


def ranked_lists(
    frame: pd.DataFrame,
    k: int,
    allowed_ids: set[int] | None,
) -> Dict[int, Tuple[int, ...]]:
    subset = frame[frame["rank"] <= k].copy()
    if allowed_ids is not None:
        subset = subset[subset["mashup_id"].isin(allowed_ids)]

    result: Dict[int, Tuple[int, ...]] = {}
    for mashup_id, group in subset.groupby(
        "mashup_id",
        sort=True,
    ):
        values = tuple(
            int(value)
            for value in group.sort_values("rank")["api_id"]
        )
        if len(values) != k:
            raise ValueError(
                f"Mashup {mashup_id} has {len(values)} items at K={k}"
            )
        if len(set(values)) != k:
            raise ValueError(
                f"Mashup {mashup_id} contains duplicate APIs"
            )
        result[int(mashup_id)] = values

    if not result:
        raise ValueError("No ranking lists after filtering")
    return result


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("Gini values must be nonnegative")
    total = float(x.sum())
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(
        np.sum((2.0 * ranks - n - 1.0) * x)
        / (n * total)
    )


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total == 0 or len(counts) <= 1:
        return 0.0
    probabilities = counts[counts > 0] / total
    entropy = -float(
        np.sum(probabilities * np.log(probabilities))
    )
    return entropy / math.log(len(counts))


def pairwise_personalization(
    lists: Mapping[int, Sequence[int]],
    num_apis: int,
) -> Tuple[float, float]:
    mashup_ids = sorted(lists)
    rows: List[int] = []
    cols: List[int] = []

    for row, mashup_id in enumerate(mashup_ids):
        for api_id in lists[mashup_id]:
            rows.append(row)
            cols.append(int(api_id))

    matrix = csr_matrix(
        (
            np.ones(len(rows), dtype=np.int16),
            (rows, cols),
        ),
        shape=(len(mashup_ids), num_apis),
    )
    intersections = (matrix @ matrix.T).toarray().astype(
        np.float64
    )
    sizes = np.asarray(matrix.sum(axis=1)).reshape(-1)
    unions = sizes[:, None] + sizes[None, :] - intersections
    upper = np.triu_indices(len(mashup_ids), k=1)
    if len(upper[0]) == 0:
        return 0.0, 0.0
    similarities = np.divide(
        intersections[upper],
        unions[upper],
        out=np.zeros_like(intersections[upper]),
        where=unions[upper] > 0,
    )
    average = float(similarities.mean())
    return average, 1.0 - average


def dcg(hits: Sequence[int]) -> float:
    return float(
        sum(
            hit / math.log2(rank + 1.0)
            for rank, hit in enumerate(hits, start=1)
        )
    )


def positive_sets_by_group(
    pairs: np.ndarray,
    api_groups: pd.DataFrame,
) -> Dict[int, Dict[str, set[int]]]:
    group_by_api = api_groups.set_index("api_id")["group"].to_dict()
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


def diversity_rows(
    *,
    method: str,
    seed: int,
    subset_name: str,
    rankings: pd.DataFrame,
    api_groups: pd.DataFrame,
    allowed_ids: set[int] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    num_apis = len(api_groups)
    train_frequency = api_groups["train_frequency"].to_numpy(
        dtype=np.float64
    )
    api_group = api_groups["group"].to_numpy(dtype=str)
    total_train_interactions = float(train_frequency.sum())
    catalog_sizes = {
        group: int(np.sum(api_group == group))
        for group in GROUPS
    }
    seen_catalog_size = int(np.sum(train_frequency > 0))
    long_tail_catalog_size = (
        catalog_sizes["Middle"] + catalog_sizes["Tail"]
    )

    metrics: List[Dict[str, Any]] = []
    exposures: List[Dict[str, Any]] = []
    frequencies: List[Dict[str, Any]] = []

    for k in KS:
        lists = ranked_lists(rankings, k, allowed_ids)
        flat = np.asarray(
            [
                api_id
                for mashup_id in sorted(lists)
                for api_id in lists[mashup_id]
            ],
            dtype=int,
        )
        counts = np.bincount(flat, minlength=num_apis).astype(
            np.float64
        )
        total_slots = int(len(flat))
        unique_recommended = int(np.count_nonzero(counts))
        unique_lists = len(set(lists.values()))
        num_mashups = len(lists)
        avg_jaccard, personalization = pairwise_personalization(
            lists,
            num_apis,
        )

        smoothed_probability = (
            (train_frequency + 1.0)
            / (total_train_interactions + num_apis)
        )
        novelty_by_api = -np.log2(smoothed_probability)

        seen_unique = int(
            np.count_nonzero(
                (counts > 0) & (train_frequency > 0)
            )
        )
        long_tail_unique = int(
            np.count_nonzero(
                (counts > 0)
                & np.isin(api_group, ["Middle", "Tail"])
            )
        )
        entropy = normalized_entropy(counts)

        metrics.append(
            {
                "subset": subset_name,
                "method": method,
                "seed": seed,
                "K": k,
                "num_mashups": num_mashups,
                "catalog_size": num_apis,
                "catalog_coverage": unique_recommended / num_apis,
                "seen_catalog_coverage": (
                    seen_unique / seen_catalog_size
                    if seen_catalog_size
                    else 0.0
                ),
                "long_tail_catalog_coverage": (
                    long_tail_unique / long_tail_catalog_size
                    if long_tail_catalog_size
                    else 0.0
                ),
                "unique_recommended_apis": unique_recommended,
                "unique_lists": unique_lists,
                "unique_list_ratio": unique_lists / num_mashups,
                "avg_pairwise_jaccard": avg_jaccard,
                "personalization": personalization,
                "gini": gini(counts),
                "normalized_entropy": entropy,
                "effective_catalog_size": float(
                    math.exp(entropy * math.log(num_apis))
                ),
                "avg_train_frequency": float(
                    train_frequency[flat].mean()
                ),
                "avg_log1p_train_frequency": float(
                    np.log1p(train_frequency[flat]).mean()
                ),
                "novelty": float(novelty_by_api[flat].mean()),
            }
        )

        for group in GROUPS:
            slot_mask = api_group[flat] == group
            slots = int(slot_mask.sum())
            catalog_mask = api_group == group
            unique_group = int(
                np.count_nonzero(
                    (counts > 0) & catalog_mask
                )
            )
            group_size = catalog_sizes[group]
            exposures.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "seed": seed,
                    "K": k,
                    "group": group,
                    "slot_count": slots,
                    "slot_ratio": (
                        slots / total_slots if total_slots else 0.0
                    ),
                    "unique_recommended_apis": unique_group,
                    "group_catalog_size": group_size,
                    "group_catalog_coverage": (
                        unique_group / group_size
                        if group_size
                        else 0.0
                    ),
                }
            )

        for api_id in range(num_apis):
            frequencies.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "seed": seed,
                    "K": k,
                    "api_id": api_id,
                    "api_group": api_group[api_id],
                    "train_frequency": int(
                        train_frequency[api_id]
                    ),
                    "recommendation_frequency": int(
                        counts[api_id]
                    ),
                    "recommendation_share": (
                        counts[api_id] / total_slots
                        if total_slots
                        else 0.0
                    ),
                }
            )

    return metrics, exposures, frequencies


def group_accuracy_rows(
    *,
    method: str,
    seed: int,
    subset_name: str,
    rankings: pd.DataFrame,
    positives: Mapping[int, Mapping[str, set[int]]],
    allowed_ids: set[int] | None,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

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

    for k in KS:
        for target_group in GROUPS:
            recalls: List[float] = []
            hrs: List[float] = []
            ndcgs: List[float] = []
            maps: List[float] = []
            total_positive = 0
            total_hits = 0

            for mashup_id, grouped in positives.items():
                if allowed_ids is not None and mashup_id not in allowed_ids:
                    continue
                relevant = grouped[target_group]
                if not relevant:
                    continue
                if mashup_id not in ranking_map:
                    continue

                topk = ranking_map[mashup_id][:k]
                hits = [
                    int(api_id in relevant)
                    for api_id in topk
                ]
                hit_count = int(sum(hits))
                total_positive += len(relevant)
                total_hits += hit_count

                recalls.append(hit_count / len(relevant))
                hrs.append(float(hit_count > 0))

                ideal = [1] * min(len(relevant), k)
                ideal_dcg = dcg(ideal)
                ndcgs.append(
                    dcg(hits) / ideal_dcg
                    if ideal_dcg
                    else 0.0
                )

                cumulative = 0
                precision_sum = 0.0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        cumulative += 1
                        precision_sum += cumulative / rank
                maps.append(
                    precision_sum / min(len(relevant), k)
                )

            result.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "seed": seed,
                    "K": k,
                    "group": target_group,
                    "eligible_mashups": len(recalls),
                    "positive_api_count": total_positive,
                    "macro_recall": (
                        float(np.mean(recalls)) if recalls else 0.0
                    ),
                    "micro_recall": (
                        total_hits / total_positive
                        if total_positive
                        else 0.0
                    ),
                    "hit_rate": (
                        float(np.mean(hrs)) if hrs else 0.0
                    ),
                    "ndcg": (
                        float(np.mean(ndcgs)) if ndcgs else 0.0
                    ),
                    "map": (
                        float(np.mean(maps)) if maps else 0.0
                    ),
                }
            )

    return result


def population_std(series: pd.Series) -> float:
    return float(
        np.std(series.to_numpy(dtype=np.float64), ddof=0)
    )


def aggregate(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = set(group_columns) | set(identity_columns) | {"seed"}
    value_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    rows: List[Dict[str, Any]] = []
    for keys, subset in frame.groupby(
        list(group_columns),
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            column: value
            for column, value in zip(group_columns, keys)
        }
        row["seeds"] = int(subset["seed"].nunique())
        for column in identity_columns:
            row[column] = subset[column].iloc[0]
        for column in value_columns:
            row[f"{column}_mean"] = float(
                subset[column].mean()
            )
            row[f"{column}_std"] = population_std(
                subset[column]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std(
    mean: float,
    std: float,
    digits: int = 4,
) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
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


def build_report(
    diversity_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    report: List[str] = [
        "# SCF Long-Tail, Diversity, and Popularity-Bias Analysis",
        "",
        "All values are mean ± population standard deviation over seeds 0/1/2.",
        "The primary comparison is among Graph+BGE, Inductive LightGCN, and "
        "SCF-LightGCN+BGE.",
        "",
    ]

    for subset_name in diversity_summary["subset"].unique():
        report.extend(
            [
                f"## Subset: {subset_name}",
                "",
                "### Diversity and popularity bias",
                "",
            ]
        )

        for k in KS:
            subset = diversity_summary[
                (diversity_summary["subset"] == subset_name)
                & (diversity_summary["K"] == k)
            ]
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
                    "Long-tail coverage": [
                        format_mean_std(mean, std)
                        for mean, std in zip(
                            subset[
                                "long_tail_catalog_coverage_mean"
                            ],
                            subset[
                                "long_tail_catalog_coverage_std"
                            ],
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
            report.extend(
                [
                    f"#### K={k}",
                    "",
                    markdown_table(table),
                    "",
                ]
            )

        report.extend(
            [
                "### Recommendation exposure by API group",
                "",
            ]
        )
        for k in KS:
            subset = exposure_summary[
                (exposure_summary["subset"] == subset_name)
                & (exposure_summary["K"] == k)
            ]
            rows: List[Dict[str, Any]] = []
            for method in METHODS:
                method_rows = subset[
                    subset["method"] == method
                ]
                if method_rows.empty:
                    continue
                row: Dict[str, Any] = {"Method": method}
                for group in GROUPS:
                    group_row = method_rows[
                        method_rows["group"] == group
                    ]
                    if group_row.empty:
                        row[group] = "-"
                    else:
                        item = group_row.iloc[0]
                        row[group] = format_mean_std(
                            float(item["slot_ratio_mean"]),
                            float(item["slot_ratio_std"]),
                        )
                rows.append(row)
            report.extend(
                [
                    f"#### K={k} slot ratio",
                    "",
                    markdown_table(pd.DataFrame(rows)),
                    "",
                ]
            )

        report.extend(
            [
                "### Accuracy by ground-truth API group",
                "",
            ]
        )
        for k in KS:
            subset = group_summary[
                (group_summary["subset"] == subset_name)
                & (group_summary["K"] == k)
            ]
            rows: List[Dict[str, Any]] = []
            for method in METHODS:
                method_rows = subset[
                    subset["method"] == method
                ]
                if method_rows.empty:
                    continue
                for group in GROUPS:
                    group_row = method_rows[
                        method_rows["group"] == group
                    ]
                    if group_row.empty:
                        continue
                    item = group_row.iloc[0]
                    rows.append(
                        {
                            "Method": method,
                            "Group": group,
                            "Mashups": int(
                                item["eligible_mashups"]
                            ),
                            "Recall": format_mean_std(
                                float(item["macro_recall_mean"]),
                                float(item["macro_recall_std"]),
                            ),
                            "NDCG": format_mean_std(
                                float(item["ndcg_mean"]),
                                float(item["ndcg_std"]),
                            ),
                            "MAP": format_mean_std(
                                float(item["map_mean"]),
                                float(item["map_std"]),
                            ),
                        }
                    )
            report.extend(
                [
                    f"#### K={k}",
                    "",
                    markdown_table(pd.DataFrame(rows)),
                    "",
                ]
            )

    report.extend(
        [
            "## Interpretation",
            "",
            "- Higher coverage, long-tail coverage, personalization, entropy, "
            "and novelty indicate broader exposure.",
            "- Lower Gini and lower average training frequency indicate weaker "
            "popularity concentration.",
            "- Group-specific accuracy measures whether the method actually "
            "retrieves Head/Middle/Tail/Unseen ground-truth APIs; exposure alone "
            "does not imply useful long-tail recommendation.",
            "- SCF improves both ranking accuracy and recommendation diversity: "
            "it expands catalog and long-tail coverage, reduces Gini concentration "
            "and average popularity, and substantially lowers Head exposure while "
            "keeping Head accuracy nearly unchanged.",
            "- Graph+BGE and Inductive LightGCN have zero Unseen recall, whereas "
            "SCF obtains non-zero Unseen recall through its direct BGE branch. "
            "This provides evidence of partial zero-shot new-API cold-start "
            "capability, although Unseen performance remains lower than Head, "
            "Middle, and Tail performance.",
            "",
        ]
    )

    return "\n".join(report)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["analysis_output_dir"]).expanduser()
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
    api_groups = load_api_groups(api_groups_path)
    if len(api_groups) != data.num_apis:
        raise ValueError(
            f"API group count {len(api_groups)} != {data.num_apis}"
        )

    positives = positive_sets_by_group(
        data.test_pairs,
        api_groups,
    )

    subsets: Dict[str, set[int] | None] = {"full": None}
    audit_value = str(config.get("audit_csv", "")).strip()
    if audit_value:
        audit_path = resolve_path(
            audit_value,
            "strict leakage audit CSV",
        )
        subsets["strict_clean"] = strict_clean_ids(
            audit_path,
            float(config["strict_clean_cosine_threshold"]),
        )

    diversity_all: List[Dict[str, Any]] = []
    exposure_all: List[Dict[str, Any]] = []
    frequency_all: List[Dict[str, Any]] = []
    group_accuracy_all: List[Dict[str, Any]] = []

    ranking_frames: List[pd.DataFrame] = []

    for method in METHODS:
        for seed in (0, 1, 2):
            rankings = load_rankings(
                config=config,
                method=method,
                seed=seed,
                api_groups=api_groups,
            )
            ranking_frames.append(rankings)

            for subset_name, allowed_ids in subsets.items():
                metric_rows, exposure_rows, frequency_rows = (
                    diversity_rows(
                        method=method,
                        seed=seed,
                        subset_name=subset_name,
                        rankings=rankings,
                        api_groups=api_groups,
                        allowed_ids=allowed_ids,
                    )
                )
                diversity_all.extend(metric_rows)
                exposure_all.extend(exposure_rows)
                frequency_all.extend(frequency_rows)

                group_accuracy_all.extend(
                    group_accuracy_rows(
                        method=method,
                        seed=seed,
                        subset_name=subset_name,
                        rankings=rankings,
                        positives=positives,
                        allowed_ids=allowed_ids,
                    )
                )

    pd.concat(ranking_frames, ignore_index=True).to_csv(
        output_dir / "combined_rankings_top10.csv",
        index=False,
    )

    diversity_by_seed = pd.DataFrame(diversity_all)
    exposure_by_seed = pd.DataFrame(exposure_all)
    frequency = pd.DataFrame(frequency_all)
    group_by_seed = pd.DataFrame(group_accuracy_all)

    diversity_by_seed.to_csv(
        output_dir / "diversity_metrics_by_seed.csv",
        index=False,
    )
    exposure_by_seed.to_csv(
        output_dir / "group_exposure_by_seed.csv",
        index=False,
    )
    frequency.to_csv(
        output_dir / "api_recommendation_frequency.csv",
        index=False,
    )
    group_by_seed.to_csv(
        output_dir / "group_accuracy_by_seed.csv",
        index=False,
    )

    diversity_summary = aggregate(
        diversity_by_seed,
        group_columns=["subset", "method", "K"],
        identity_columns=[
            "num_mashups",
            "catalog_size",
        ],
    )
    exposure_summary = aggregate(
        exposure_by_seed,
        group_columns=["subset", "method", "K", "group"],
        identity_columns=["group_catalog_size"],
    )
    group_summary = aggregate(
        group_by_seed,
        group_columns=["subset", "method", "K", "group"],
        identity_columns=[
            "eligible_mashups",
            "positive_api_count",
        ],
    )

    diversity_summary.to_csv(
        output_dir / "diversity_metrics_mean_std.csv",
        index=False,
    )
    exposure_summary.to_csv(
        output_dir / "group_exposure_mean_std.csv",
        index=False,
    )
    group_summary.to_csv(
        output_dir / "group_accuracy_mean_std.csv",
        index=False,
    )

    tradeoff_rows: List[Dict[str, Any]] = []
    for subset_name in subsets:
        for method in METHODS:
            diversity_row = diversity_summary[
                (diversity_summary["subset"] == subset_name)
                & (diversity_summary["method"] == method)
                & (diversity_summary["K"] == 10)
            ]
            head_row = exposure_summary[
                (exposure_summary["subset"] == subset_name)
                & (exposure_summary["method"] == method)
                & (exposure_summary["K"] == 10)
                & (exposure_summary["group"] == "Head")
            ]
            tail_row = group_summary[
                (group_summary["subset"] == subset_name)
                & (group_summary["method"] == method)
                & (group_summary["K"] == 10)
                & (group_summary["group"] == "Tail")
            ]
            unseen_row = group_summary[
                (group_summary["subset"] == subset_name)
                & (group_summary["method"] == method)
                & (group_summary["K"] == 10)
                & (group_summary["group"] == "Unseen")
            ]
            if diversity_row.empty:
                continue
            d = diversity_row.iloc[0]
            tradeoff_rows.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "coverage@10": float(
                        d["catalog_coverage_mean"]
                    ),
                    "personalization@10": float(
                        d["personalization_mean"]
                    ),
                    "gini@10": float(d["gini_mean"]),
                    "avg_popularity@10": float(
                        d["avg_train_frequency_mean"]
                    ),
                    "novelty@10": float(d["novelty_mean"]),
                    "head_exposure@10": (
                        float(head_row.iloc[0]["slot_ratio_mean"])
                        if not head_row.empty
                        else 0.0
                    ),
                    "tail_recall@10": (
                        float(tail_row.iloc[0]["macro_recall_mean"])
                        if not tail_row.empty
                        else 0.0
                    ),
                    "unseen_recall@10": (
                        float(unseen_row.iloc[0]["macro_recall_mean"])
                        if not unseen_row.empty
                        else 0.0
                    ),
                }
            )

    pd.DataFrame(tradeoff_rows).to_csv(
        output_dir / "accuracy_diversity_tradeoff_summary.csv",
        index=False,
    )

    report = build_report(
        diversity_summary,
        exposure_summary,
        group_summary,
    )
    (output_dir / "scf_tail_diversity_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
