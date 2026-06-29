#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


METRICS = [
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
            "Run paired Mashup-level significance tests for the new formal baselines."
        )
    )
    parser.add_argument(
        "--baseline_per_mashup",
        type=Path,
        default=Path(
            "outputs/strict_baselines/test/"
            "new_baseline_per_mashup_metrics.csv"
        ),
    )
    parser.add_argument(
        "--graph_bge_source",
        type=Path,
        default=Path("outputs/statistical_analysis/runs"),
        help=(
            "Either a Graph+BGE per-Mashup CSV or a directory containing "
            "per_mashup_metrics.csv files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/strict_baseline_significance"),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def normalize_method(value: str) -> str:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)

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
    if "bge" == compact or "bgeonly" in compact:
        return "BGE"
    return str(value).strip()


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

    # analyze_strict_coldstart.py writes the evaluated model as "Ours" in
    # every run. The actual ablation identity is encoded in the run directory,
    # e.g. graph_bge_zscore/seed0, graph_only/seed0, or bge_only/seed0.
    # Resolve generic labels from the source path before filtering methods.
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
            "Inductive BPR-MF",
            "Inductive LightGCN",
        }:
            normalized.loc[generic] = path_method
        frame["method"] = normalized

    if "seed" not in frame.columns:
        seed = infer_seed(source_path)
        if seed is None:
            seed = 0
        frame["seed"] = seed

    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_path} is missing metric columns: {missing}"
        )

    columns = ["method", "seed", "mashup_id", *METRICS]
    return frame[columns]


def load_graph_bge_source(path: Path) -> pd.DataFrame:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("per_mashup_metrics.csv"))
        if not candidates:
            candidates = sorted(path.rglob("*.csv"))
    else:
        raise FileNotFoundError(path)

    loaded = []
    errors = []
    for candidate in candidates:
        try:
            frame = pd.read_csv(candidate)
            standardized = standardize_frame(frame, candidate)
            graph = standardized[
                standardized["method"] == "Graph+BGE"
            ].copy()
            if not graph.empty:
                loaded.append(graph)
        except Exception as error:
            errors.append(f"{candidate}: {error}")

    if not loaded:
        details = "\n".join(errors[:10])
        raise RuntimeError(
            "No Graph+BGE per-Mashup metrics were found under "
            f"{path}.\nChecked files:\n{details}"
        )

    result = pd.concat(loaded, ignore_index=True)
    return result.drop_duplicates(
        subset=["method", "seed", "mashup_id"],
        keep="last",
    )


def average_over_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "mashup_id"], as_index=False)[METRICS]
        .mean()
        .sort_values(["method", "mashup_id"])
    )


def paired_vectors(
    averaged: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = averaged[averaged["method"] == method_a][
        ["mashup_id", metric]
    ].rename(columns={metric: "a"})
    right = averaged[averaged["method"] == method_b][
        ["mashup_id", metric]
    ].rename(columns={metric: "b"})
    merged = left.merge(right, on="mashup_id", how="inner")
    if merged.empty:
        raise RuntimeError(
            f"No paired Mashups for {method_a} vs {method_b}"
        )
    return (
        merged["a"].to_numpy(dtype=np.float64),
        merged["b"].to_numpy(dtype=np.float64),
        merged["mashup_id"].to_numpy(dtype=np.int64),
    )


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


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = standardize_frame(
        pd.read_csv(args.baseline_per_mashup),
        args.baseline_per_mashup,
    )
    baseline = baseline[
        baseline["method"].isin(
            {"Inductive BPR-MF", "Inductive LightGCN"}
        )
    ].copy()

    graph_bge = load_graph_bge_source(args.graph_bge_source)
    combined = pd.concat([baseline, graph_bge], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["method", "seed", "mashup_id"],
        keep="last",
    )
    combined.to_csv(
        output_dir / "paired_metric_source.csv",
        index=False,
    )

    coverage_rows = []
    for method, group in combined.groupby("method"):
        coverage_rows.append(
            {
                "method": method,
                "num_seeds": int(group["seed"].nunique()),
                "num_unique_mashups": int(group["mashup_id"].nunique()),
                "num_rows": int(len(group)),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("method")
    coverage.to_csv(output_dir / "source_coverage.csv", index=False)

    averaged = average_over_seeds(combined)
    averaged.to_csv(
        output_dir / "per_mashup_seed_averaged.csv",
        index=False,
    )

    rng = np.random.default_rng(args.random_seed)
    rows: List[Dict[str, object]] = []

    for method_a, method_b in COMPARISONS:
        for metric in METRICS:
            values_a, values_b, mashup_ids = paired_vectors(
                averaged,
                method_a,
                method_b,
                metric,
            )
            differences = values_a - values_b
            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=False,
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            ci_low, ci_high = paired_bootstrap_ci(
                differences,
                args.bootstrap_samples,
                rng,
            )
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "num_mashups": int(len(mashup_ids)),
                    "mean_a": float(values_a.mean()),
                    "mean_b": float(values_b.mean()),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                    "rank_biserial": rank_biserial(differences),
                    "method_a_win_rate": float((differences > 0).mean()),
                    "tie_rate": float((differences == 0).mean()),
                }
            )

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].to_numpy())
    result["significance"] = result["p_holm"].map(significance_stars)
    result["ci_excludes_zero"] = (
        (result["bootstrap_ci_low"] > 0)
        | (result["bootstrap_ci_high"] < 0)
    )
    result.to_csv(
        output_dir / "new_baseline_significance_results.csv",
        index=False,
    )

    display = result[
        [
            "method_a",
            "method_b",
            "metric",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_holm",
            "rank_biserial",
            "method_a_win_rate",
            "significance",
        ]
    ].copy()
    for column in (
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "rank_biserial",
        "method_a_win_rate",
    ):
        display[column] = display[column].map(
            lambda value: f"{float(value):.4f}"
        )
    display["p_holm"] = display["p_holm"].map(
        lambda value: "<0.001" if float(value) < 0.001 else f"{float(value):.4f}"
    )

    report = [
        "# Paired Significance Analysis for the New Formal Baselines",
        "",
        "Statistical unit: one test Mashup. Metrics are first averaged over "
        "seeds 0/1/2 for each Mashup. The analysis then applies paired Wilcoxon "
        "signed-rank tests, 10,000 paired bootstrap confidence intervals, "
        "Holm-Bonferroni correction, and rank-biserial effect sizes.",
        "",
        "A positive mean difference and positive rank-biserial value indicate "
        "that `method_a` outperforms `method_b`.",
        "",
        "## Source coverage",
        "",
        markdown_table(coverage),
        "",
        "## Results",
        "",
        markdown_table(display),
        "",
    ]
    (output_dir / "new_baseline_significance_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(coverage.to_string(index=False))
    print()
    print(display.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
