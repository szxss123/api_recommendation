#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    "MAP@5",
    "Recall@10",
    "NDCG@10",
    "MAP@10",
]

COMPARISONS = [
    ("SCF-LightGCN+BGE", "Inductive LightGCN"),
    ("SCF-LightGCN+BGE", "Inductive BPR-MF"),
    ("SCF-LightGCN+BGE", "Graph+BGE"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final paired significance analysis for the validation-selected "
            "semantic-collaborative fusion model."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scf_final_significance.yaml"),
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

    if compact.startswith("scf") or (
        "lightgcn" in compact and "bge" in compact
    ):
        return "SCF-LightGCN+BGE"
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
    return text


def standardize(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    required = {"mashup_id", "seed", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source} is missing columns: {sorted(missing)}"
        )

    if "method" not in frame.columns:
        raise ValueError(f"{source} does not contain method")
    frame["method"] = frame["method"].map(normalize_method)
    return frame[["method", "seed", "mashup_id", *METRICS]]


def strict_clean_ids(audit: pd.DataFrame, threshold: float) -> set[int]:
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
            f"Audit CSV is missing columns: {sorted(missing)}"
        )

    mask = (
        (audit["exact_train_text_duplicate"] == 0)
        & (audit["exact_train_name_duplicate"] == 0)
        & (audit["direct_api_name_mention_count"] == 0)
        & (audit["top1_cosine_similarity"] < threshold)
    )
    return set(audit.loc[mask, "mashup_id"].astype(int))


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
    count = len(differences)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 256
    cursor = 0

    while cursor < samples:
        current = min(batch_size, samples - cursor)
        indices = rng.integers(
            0,
            count,
            size=(current, count),
        )
        means[cursor : cursor + current] = differences[
            indices
        ].mean(axis=1)
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


def stars(value: float) -> str:
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
            str(row[column])
            .replace("|", r"\|")
            .replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paired_analysis(
    *,
    averaged: pd.DataFrame,
    subset_name: str,
    allowed_ids: set[int] | None,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if allowed_ids is not None:
        averaged = averaged[
            averaged["mashup_id"].isin(allowed_ids)
        ].copy()

    rows: List[Dict[str, Any]] = []

    for method_a, method_b in COMPARISONS:
        for metric in METRICS:
            left = averaged[
                averaged["method"] == method_a
            ][["mashup_id", metric]].rename(
                columns={metric: "a"}
            )
            right = averaged[
                averaged["method"] == method_b
            ][["mashup_id", metric]].rename(
                columns={metric: "b"}
            )

            paired = left.merge(
                right,
                on="mashup_id",
                how="inner",
            )
            if paired.empty:
                raise RuntimeError(
                    f"No paired data for {method_a} vs {method_b}"
                )

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
                    "rank_biserial": rank_biserial(differences),
                    "method_a_win_rate": float(
                        (differences > 0).mean()
                    ),
                    "tie_rate": float(
                        (differences == 0).mean()
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_raw"])
    frame["significance"] = frame["p_holm"].map(stars)
    frame["ci_excludes_zero"] = (
        (frame["bootstrap_ci_low"] > 0)
        | (frame["bootstrap_ci_high"] < 0)
    )
    return frame


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scf_path = Path(config["scf_per_mashup_csv"])
    prior_path = Path(config["prior_paired_metric_source_csv"])
    audit_path = Path(config["audit_csv"])

    scf = standardize(pd.read_csv(scf_path), scf_path)
    prior = standardize(pd.read_csv(prior_path), prior_path)

    prior = prior[
        prior["method"].isin(
            {
                "Inductive LightGCN",
                "Inductive BPR-MF",
                "Graph+BGE",
            }
        )
    ].copy()

    combined = pd.concat([scf, prior], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["method", "seed", "mashup_id"],
        keep="last",
    )
    combined.to_csv(
        output_dir / "scf_paired_metric_source.csv",
        index=False,
    )

    coverage = (
        combined.groupby("method")
        .agg(
            num_seeds=("seed", "nunique"),
            num_unique_mashups=("mashup_id", "nunique"),
            num_rows=("mashup_id", "size"),
        )
        .reset_index()
    )
    coverage.to_csv(
        output_dir / "scf_source_coverage.csv",
        index=False,
    )

    averaged = (
        combined.groupby(
            ["method", "mashup_id"],
            as_index=False,
        )[METRICS]
        .mean()
    )
    averaged.to_csv(
        output_dir / "scf_per_mashup_seed_averaged.csv",
        index=False,
    )

    audit = pd.read_csv(audit_path)
    clean_ids = strict_clean_ids(
        audit,
        float(config["strict_clean_cosine_threshold"]),
    )

    rng = np.random.default_rng(int(config["random_seed"]))

    full = paired_analysis(
        averaged=averaged,
        subset_name="full",
        allowed_ids=None,
        bootstrap_samples=int(config["bootstrap_samples"]),
        rng=rng,
    )
    clean = paired_analysis(
        averaged=averaged,
        subset_name="strict_clean",
        allowed_ids=clean_ids,
        bootstrap_samples=int(config["bootstrap_samples"]),
        rng=rng,
    )

    results = pd.concat([full, clean], ignore_index=True)
    results.to_csv(
        output_dir / "scf_final_significance_results.csv",
        index=False,
    )

    display = results[
        [
            "subset",
            "method_a",
            "method_b",
            "metric",
            "num_mashups",
            "mean_difference",
            "relative_improvement_percent",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_holm",
            "rank_biserial",
            "significance",
        ]
    ].copy()

    for column in (
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "rank_biserial",
    ):
        display[column] = display[column].map(
            lambda value: f"{float(value):.4f}"
        )

    display["relative_improvement_percent"] = display[
        "relative_improvement_percent"
    ].map(lambda value: f"{float(value):.2f}%")
    display["p_holm"] = display["p_holm"].map(
        lambda value: (
            "<0.001"
            if float(value) < 0.001
            else f"{float(value):.4f}"
        )
    )

    report = [
        "# Final SCF Paired Significance Analysis",
        "",
        "Statistical unit: one test Mashup. Each metric is first averaged "
        "over seeds 0/1/2 for each Mashup. The analysis uses paired "
        "Wilcoxon signed-rank tests, 10,000 paired bootstrap confidence "
        "intervals, Holm-Bonferroni correction, and rank-biserial effect sizes.",
        "",
        "## Source coverage",
        "",
        markdown_table(coverage),
        "",
        "## Full test set",
        "",
        markdown_table(
            display[display["subset"] == "full"].drop(
                columns=["subset"]
            )
        ),
        "",
        "## Strict-clean subset",
        "",
        markdown_table(
            display[
                display["subset"] == "strict_clean"
            ].drop(columns=["subset"])
        ),
        "",
        "## Naming",
        "",
        "The validation-selected popularity weight is zero. The final model "
        "should therefore be reported as `SCF-LightGCN+BGE`, not as a "
        "three-branch model containing an active popularity component.",
        "",
    ]

    (output_dir / "scf_final_significance_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    summary = {
        "strict_clean_mashups": len(clean_ids),
        "comparisons": [
            {
                "method_a": a,
                "method_b": b,
            }
            for a, b in COMPARISONS
        ],
        "bootstrap_samples": int(config["bootstrap_samples"]),
    }
    (output_dir / "scf_final_significance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(coverage.to_string(index=False))
    print()
    print(display.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
