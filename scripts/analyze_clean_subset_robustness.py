#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon


METRICS = [
    "Recall@5",
    "NDCG@5",
    "HitRate@5",
    "MAP@5",
    "Recall@10",
    "NDCG@10",
    "HitRate@10",
    "MAP@10",
]

REPORT_METRICS = [
    "Recall@5",
    "NDCG@5",
    "MAP@5",
    "Recall@10",
    "NDCG@10",
    "MAP@10",
]

COMPARISONS = [
    ("Inductive LightGCN", "Graph+BGE"),
    ("Inductive BPR-MF", "Graph+BGE"),
    ("Inductive LightGCN", "Inductive BPR-MF"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate all formal baselines on de-duplicated and conservative "
            "clean test subsets without retraining any model."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strict_clean_subset_analysis.yaml"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def normalize_method(value: str) -> str:
    text = str(value).strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())

    if "lightgcn" in compact:
        return "Inductive LightGCN"
    if "bpr" in compact and "mf" in compact:
        return "Inductive BPR-MF"
    if (
        "graphbge" in compact
        or ("graph" in compact and "bge" in compact)
        or "fusion" in compact
        or "zscore" in compact
    ):
        return "Graph+BGE"
    if "graphonly" in compact:
        return "Graph-only"
    if compact in {"bge", "bgeonly"}:
        return "BGE"
    if "popularity" in compact:
        return "Popularity"
    if "tfidf" in compact:
        return "TF-IDF"
    if "bm25" in compact:
        return "BM25"
    if "categoryjaccard" in compact:
        return "Category-Jaccard"
    return text


def infer_seed(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def standardize_frame(frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    frame = frame.copy()
    if "mashup_id" not in frame.columns:
        raise ValueError(f"{source_path} does not contain mashup_id")

    path_method = normalize_method(source_path.as_posix())
    if "method" not in frame.columns:
        frame["method"] = path_method
    else:
        normalized = frame["method"].map(normalize_method)
        generic = normalized.astype(str).str.strip().str.lower().isin(
            {"ours", "model", "main", "method"}
        )
        if path_method in {
            "Graph+BGE",
            "Graph-only",
            "BGE",
            "Popularity",
            "Inductive BPR-MF",
            "Inductive LightGCN",
        }:
            normalized.loc[generic] = path_method
        frame["method"] = normalized

    if "seed" not in frame.columns:
        seed = infer_seed(source_path)
        frame["seed"] = 0 if seed is None else seed

    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_path} is missing metric columns: {missing}"
        )

    return frame[["method", "seed", "mashup_id", *METRICS]]


def load_graph_methods(path: Path) -> pd.DataFrame:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("per_mashup_metrics.csv"))
    else:
        raise FileNotFoundError(path)

    loaded: List[pd.DataFrame] = []
    errors: List[str] = []
    for candidate in candidates:
        try:
            frame = standardize_frame(pd.read_csv(candidate), candidate)
            frame = frame[
                frame["method"].isin({"Graph-only", "Graph+BGE"})
            ].copy()
            if not frame.empty:
                loaded.append(frame)
        except Exception as error:
            errors.append(f"{candidate}: {error}")

    if not loaded:
        raise RuntimeError(
            "No Graph-only or Graph+BGE per-Mashup files were found under "
            f"{path}.\n" + "\n".join(errors[:10])
        )

    result = pd.concat(loaded, ignore_index=True)
    return result.drop_duplicates(
        subset=["method", "seed", "mashup_id"],
        keep="last",
    )


def build_subsets(
    audit: pd.DataFrame,
    cosine_thresholds: Sequence[float],
) -> Dict[str, np.ndarray]:
    required = {
        "mashup_id",
        "exact_train_text_duplicate",
        "exact_train_name_duplicate",
        "top1_cosine_similarity",
        "direct_api_name_mention_count",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            f"Audit CSV is missing columns: {sorted(missing)}"
        )

    no_exact = (
        (audit["exact_train_text_duplicate"] == 0)
        & (audit["exact_train_name_duplicate"] == 0)
    )
    no_mention = audit["direct_api_name_mention_count"] == 0

    subsets: Dict[str, np.ndarray] = {
        "full": audit["mashup_id"].to_numpy(dtype=np.int64),
        "deduplicated": audit.loc[
            no_exact, "mashup_id"
        ].to_numpy(dtype=np.int64),
        "deduplicated_no_api_mention": audit.loc[
            no_exact & no_mention, "mashup_id"
        ].to_numpy(dtype=np.int64),
    }

    for threshold in cosine_thresholds:
        key = f"deduplicated_cosine_lt_{threshold:.2f}".replace(".", "")
        mask = no_exact & (audit["top1_cosine_similarity"] < threshold)
        subsets[key] = audit.loc[mask, "mashup_id"].to_numpy(
            dtype=np.int64
        )

    strict_threshold = float(min(cosine_thresholds))
    strict_key = (
        f"strict_clean_cosine_lt_{strict_threshold:.2f}".replace(".", "")
    )
    strict_mask = (
        no_exact
        & no_mention
        & (audit["top1_cosine_similarity"] < strict_threshold)
    )
    subsets[strict_key] = audit.loc[
        strict_mask, "mashup_id"
    ].to_numpy(dtype=np.int64)

    return subsets


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_bootstrap_ci(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    n = len(differences)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 256
    cursor = 0
    while cursor < samples:
        current = min(batch_size, samples - cursor)
        indices = rng.integers(0, n, size=(current, n))
        means[cursor : cursor + current] = differences[indices].mean(axis=1)
        cursor += current
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p[index])
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


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            str(row[column]).replace("|", r"\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = Path(config["audit_csv"])
    baseline_path = Path(config["baseline_per_mashup_csv"])
    graph_source = Path(config["graph_method_source"])

    audit = pd.read_csv(audit_path)
    baseline = standardize_frame(
        pd.read_csv(baseline_path),
        baseline_path,
    )
    graph_methods = load_graph_methods(graph_source)

    combined = pd.concat([baseline, graph_methods], ignore_index=True)
    combined["method"] = combined["method"].map(normalize_method)
    combined = combined.drop_duplicates(
        subset=["method", "seed", "mashup_id"],
        keep="last",
    )

    method_order = list(config["method_order"])
    combined = combined[combined["method"].isin(method_order)].copy()
    combined.to_csv(
        output_dir / "combined_per_mashup_metric_source.csv",
        index=False,
    )

    subsets = build_subsets(
        audit,
        [float(value) for value in config["cosine_thresholds"]],
    )

    subset_rows = []
    membership_rows = []
    total = len(audit)
    for name, mashup_ids in subsets.items():
        subset_rows.append(
            {
                "subset": name,
                "num_mashups": int(len(mashup_ids)),
                "retained_rate": float(len(mashup_ids) / total),
            }
        )
        membership_rows.extend(
            {
                "subset": name,
                "mashup_id": int(mashup_id),
            }
            for mashup_id in mashup_ids
        )

    subset_definition = pd.DataFrame(subset_rows)
    subset_definition.to_csv(
        output_dir / "subset_definitions.csv",
        index=False,
    )
    pd.DataFrame(membership_rows).to_csv(
        output_dir / "subset_membership.csv",
        index=False,
    )

    by_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    formatted_rows: List[Dict[str, Any]] = []

    order_map = {
        method: index for index, method in enumerate(method_order)
    }

    for subset_name, mashup_ids in subsets.items():
        allowed = set(int(value) for value in mashup_ids)
        subset_frame = combined[
            combined["mashup_id"].isin(allowed)
        ].copy()

        for (method, seed), group in subset_frame.groupby(
            ["method", "seed"]
        ):
            row: Dict[str, Any] = {
                "subset": subset_name,
                "method": method,
                "seed": int(seed),
                "num_mashups": int(group["mashup_id"].nunique()),
            }
            for metric in METRICS:
                row[metric] = float(group[metric].mean())
            by_seed_rows.append(row)

        for method in method_order:
            method_seed_rows = [
                row
                for row in by_seed_rows
                if row["subset"] == subset_name
                and row["method"] == method
            ]
            if not method_seed_rows:
                continue
            method_seed = pd.DataFrame(method_seed_rows)
            summary: Dict[str, Any] = {
                "subset": subset_name,
                "method": method,
                "num_mashups": int(len(allowed)),
                "num_seeds": int(method_seed["seed"].nunique()),
            }
            formatted: Dict[str, Any] = {
                "Subset": subset_name,
                "Method": method,
                "Mashups": int(len(allowed)),
            }
            for metric in METRICS:
                mean = float(method_seed[metric].mean())
                std = float(method_seed[metric].std(ddof=0))
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
                if metric in REPORT_METRICS:
                    formatted[metric] = format_mean_std(mean, std)
            summary_rows.append(summary)
            formatted_rows.append(formatted)

    by_seed = pd.DataFrame(by_seed_rows)
    by_seed["_order"] = by_seed["method"].map(order_map)
    by_seed = by_seed.sort_values(
        ["subset", "_order", "seed"]
    ).drop(columns="_order")
    by_seed.to_csv(
        output_dir / "clean_subset_metrics_by_seed.csv",
        index=False,
    )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["_order"] = summary_frame["method"].map(order_map)
    summary_frame = summary_frame.sort_values(
        ["subset", "_order"]
    ).drop(columns="_order")
    summary_frame.to_csv(
        output_dir / "clean_subset_main_table_mean_std.csv",
        index=False,
    )

    formatted_frame = pd.DataFrame(formatted_rows)
    formatted_frame["_order"] = formatted_frame["Method"].map(order_map)
    formatted_frame = formatted_frame.sort_values(
        ["Subset", "_order"]
    ).drop(columns="_order")
    formatted_frame.to_csv(
        output_dir / "clean_subset_main_table_formatted.csv",
        index=False,
    )

    # Seed-average each Mashup before paired testing.
    seed_averaged = (
        combined.groupby(["method", "mashup_id"], as_index=False)[
            REPORT_METRICS
        ]
        .mean()
    )

    rng = np.random.default_rng(int(config["random_seed"]))
    significance_rows: List[Dict[str, Any]] = []

    for subset_name, mashup_ids in subsets.items():
        allowed = set(int(value) for value in mashup_ids)
        subset_averaged = seed_averaged[
            seed_averaged["mashup_id"].isin(allowed)
        ].copy()

        subset_test_indices: List[int] = []
        for method_a, method_b in COMPARISONS:
            for metric in REPORT_METRICS:
                left = subset_averaged[
                    subset_averaged["method"] == method_a
                ][["mashup_id", metric]].rename(
                    columns={metric: "a"}
                )
                right = subset_averaged[
                    subset_averaged["method"] == method_b
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

                if np.allclose(differences, 0.0):
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
                    int(config["bootstrap_samples"]),
                    rng,
                )

                significance_rows.append(
                    {
                        "subset": subset_name,
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "num_mashups": int(len(paired)),
                        "mean_a": float(paired["a"].mean()),
                        "mean_b": float(paired["b"].mean()),
                        "mean_difference": float(
                            differences.mean()
                        ),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_raw,
                        "rank_biserial": rank_biserial(differences),
                        "method_a_win_rate": float(
                            (differences > 0).mean()
                        ),
                        "tie_rate": float(
                            (differences == 0).mean()
                        ),
                    }
                )
                subset_test_indices.append(
                    len(significance_rows) - 1
                )

        if subset_test_indices:
            adjusted = holm_adjust(
                [
                    significance_rows[index]["p_raw"]
                    for index in subset_test_indices
                ]
            )
            for index, p_holm in zip(
                subset_test_indices,
                adjusted,
            ):
                significance_rows[index]["p_holm"] = float(p_holm)
                significance_rows[index]["significance"] = (
                    significance_stars(float(p_holm))
                )
                significance_rows[index]["ci_excludes_zero"] = bool(
                    significance_rows[index]["bootstrap_ci_low"] > 0
                    or significance_rows[index]["bootstrap_ci_high"] < 0
                )

    significance = pd.DataFrame(significance_rows)
    significance.to_csv(
        output_dir / "clean_subset_significance.csv",
        index=False,
    )

    # Compare each clean subset against the full-set mean.
    full = summary_frame[
        summary_frame["subset"] == "full"
    ].set_index("method")
    delta_rows = []
    for _, row in summary_frame.iterrows():
        if row["subset"] == "full":
            continue
        method = row["method"]
        if method not in full.index:
            continue
        delta = {
            "subset": row["subset"],
            "method": method,
        }
        for metric in REPORT_METRICS:
            delta[f"{metric}_delta_vs_full"] = float(
                row[f"{metric}_mean"]
                - full.loc[method, f"{metric}_mean"]
            )
        delta_rows.append(delta)

    pd.DataFrame(delta_rows).to_csv(
        output_dir / "robustness_delta_vs_full.csv",
        index=False,
    )

    report = [
        "# Clean-Subset Robustness Analysis",
        "",
        "No model is retrained. Existing per-Mashup predictions are filtered "
        "using the audit results and then re-aggregated. The full test set "
        "remains the primary result; clean subsets are robustness checks.",
        "",
        "## Subset sizes",
        "",
        markdown_table(
            subset_definition.assign(
                retained_rate=subset_definition["retained_rate"].map(
                    lambda value: f"{float(value):.2%}"
                )
            )
        ),
        "",
    ]

    for subset_name in subset_definition["subset"]:
        table = formatted_frame[
            formatted_frame["Subset"] == subset_name
        ].drop(columns=["Subset"])
        report.extend(
            [
                f"## {subset_name}",
                "",
                markdown_table(table),
                "",
            ]
        )

        sig = significance[
            (significance["subset"] == subset_name)
            & (
                (
                    significance["method_a"]
                    == "Inductive LightGCN"
                )
                & (
                    significance["method_b"]
                    == "Graph+BGE"
                )
            )
        ][
            [
                "metric",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "p_holm",
                "rank_biserial",
                "significance",
            ]
        ].copy()

        if not sig.empty:
            for column in (
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "rank_biserial",
            ):
                sig[column] = sig[column].map(
                    lambda value: f"{float(value):.4f}"
                )
            sig["p_holm"] = sig["p_holm"].map(
                lambda value: (
                    "<0.001"
                    if float(value) < 0.001
                    else f"{float(value):.4f}"
                )
            )
            report.extend(
                [
                    "### Inductive LightGCN vs Graph+BGE",
                    "",
                    markdown_table(sig),
                    "",
                ]
            )

    report.extend(
        [
            "## Interpretation",
            "",
            "- `deduplicated` removes exact normalized text duplicates and "
            "exact Mashup-identifier duplicates.",
            "- cosine-filtered subsets additionally remove highly similar "
            "test-to-train Mashups.",
            "- `deduplicated_no_api_mention` removes cases whose test text "
            "directly mentions at least one ground-truth API identifier.",
            "- the strict-clean subset combines de-duplication, no direct API "
            "mention, and the most conservative cosine threshold.",
            "- direct API mentions are not automatically leakage when names "
            "and descriptions are valid inference-time inputs; this subset "
            "measures dependence on that strong lexical cue.",
            "",
        ]
    )

    (output_dir / "clean_subset_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(subset_definition.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
