#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

METHODS = {
    "graph_only": "Graph-only",
    "bge_only": "BGE-only",
    "graph_bge_zscore": "Graph+BGE (z-score, λ=0.25)",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/final_strict_ablation"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(str(row[col]) for col in headers) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: List[Dict] = []
    group_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    composition_rows: List[Dict] = []
    popularity_row = None

    for method_dir, display_name in METHODS.items():
        for seed in (0, 1, 2):
            run_dir = root / method_dir / f"seed{seed}"
            overall_path = run_dir / "overall_metrics.csv"
            if not overall_path.exists():
                raise FileNotFoundError(overall_path)

            overall = pd.read_csv(overall_path)
            ours = overall[overall["method"] == "Ours"]
            if ours.empty:
                raise ValueError(f"No Ours row in {overall_path}")
            row = ours.iloc[0].to_dict()
            row.update(
                {
                    "method_key": method_dir,
                    "method": display_name,
                    "seed": seed,
                }
            )
            overall_rows.append(row)

            if popularity_row is None:
                pop = overall[overall["method"] == "Popularity"]
                if not pop.empty:
                    popularity_row = pop.iloc[0].to_dict()

            group_path = run_dir / "group_metrics.csv"
            if group_path.exists():
                group = pd.read_csv(group_path)
                group = group[group["method"] == "Ours"].copy()
                group["method_key"] = method_dir
                group["method"] = display_name
                group["seed"] = seed
                group_rows.extend(group.to_dict(orient="records"))

            overlap_path = run_dir / "overlap_summary.csv"
            if overlap_path.exists():
                overlap = pd.read_csv(overlap_path)
                overlap["method_key"] = method_dir
                overlap["method"] = display_name
                overlap["seed"] = seed
                overlap_rows.extend(overlap.to_dict(orient="records"))

            composition_path = run_dir / "recommendation_composition.csv"
            if composition_path.exists():
                comp = pd.read_csv(composition_path)
                comp = comp[comp["method"] == "Ours"].copy()
                comp["method_key"] = method_dir
                comp["method"] = display_name
                comp["seed"] = seed
                composition_rows.extend(comp.to_dict(orient="records"))

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_dir / "overall_by_seed.csv", index=False)

    mean_std_rows = []
    ablation_rows = []
    for method_key, display_name in METHODS.items():
        subset = overall_df[overall_df["method_key"] == method_key]
        summary = {
            "method_key": method_key,
            "method": display_name,
            "seeds": len(subset),
        }
        ablation = {"Method": display_name}
        for metric in METRICS:
            mean = float(subset[metric].mean())
            std = float(subset[metric].std(ddof=0))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            ablation[metric] = fmt(mean, std)
        mean_std_rows.append(summary)
        ablation_rows.append(ablation)

    if popularity_row is not None:
        pop_ablation = {"Method": "Popularity"}
        for metric in METRICS:
            pop_ablation[metric] = f"{float(popularity_row[metric]):.4f}"
        ablation_rows.insert(0, pop_ablation)

    mean_std_df = pd.DataFrame(mean_std_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    mean_std_df.to_csv(output_dir / "overall_mean_std.csv", index=False)
    ablation_df.to_csv(output_dir / "ablation_table.csv", index=False)

    if group_rows:
        group_df = pd.DataFrame(group_rows)
        group_df.to_csv(output_dir / "group_by_seed.csv", index=False)

        group_summary_rows = []
        for (method_key, method, group_name), subset in group_df.groupby(
            ["method_key", "method", "group"],
            sort=False,
        ):
            row = {
                "method_key": method_key,
                "method": method,
                "group": group_name,
                "seeds": len(subset),
                "eligible_mashups": int(subset["eligible_mashups"].iloc[0]),
                "test_positive_count": int(
                    subset["test_positive_count"].iloc[0]
                ),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(subset[metric].mean())
                row[f"{metric}_std"] = float(
                    subset[metric].std(ddof=0)
                )
            group_summary_rows.append(row)

        group_summary_df = pd.DataFrame(group_summary_rows)
        group_summary_df.to_csv(
            output_dir / "group_mean_std.csv",
            index=False,
        )
    else:
        group_summary_df = pd.DataFrame()

    if overlap_rows:
        overlap_df = pd.DataFrame(overlap_rows)
        overlap_df.to_csv(output_dir / "overlap_by_seed.csv", index=False)

        overlap_summary = (
            overlap_df.groupby(
                ["method_key", "method", "K"],
                as_index=False,
                sort=False,
            )
            .agg(
                avg_overlap_ratio_mean=("avg_overlap_ratio", "mean"),
                avg_overlap_ratio_std=(
                    "avg_overlap_ratio",
                    lambda x: float(np.std(x, ddof=0)),
                ),
                avg_jaccard_mean=("avg_jaccard", "mean"),
                exact_set_match_ratio_mean=(
                    "exact_set_match_ratio",
                    "mean",
                ),
                exact_order_match_ratio_mean=(
                    "exact_order_match_ratio",
                    "mean",
                ),
                top1_match_ratio_mean=("top1_match_ratio", "mean"),
            )
        )
        overlap_summary.to_csv(
            output_dir / "overlap_mean_std.csv",
            index=False,
        )

    if composition_rows:
        composition_df = pd.DataFrame(composition_rows)
        composition_df.to_csv(
            output_dir / "recommendation_composition_by_seed.csv",
            index=False,
        )

    report_parts = [
        "# Strict Cold-Start Final Ablation",
        "",
        "## Overall results (mean ± population std over 3 seeds)",
        "",
        markdown_table(ablation_df),
        "",
        "## Selected main model",
        "",
        "`Graph+BGE (z-score, λ=0.25)`",
        "",
        "Fusion formula:",
        "",
        "```text",
        "final_score = 0.75 × z(graph_score) + 0.25 × z(BGE cosine)",
        "```",
        "",
    ]

    if popularity_row is not None:
        main_row = mean_std_df[
            mean_std_df["method_key"] == "graph_bge_zscore"
        ].iloc[0]
        report_parts += [
            "## Relative improvement over Popularity",
            "",
        ]
        improvement_rows = []
        for metric in METRICS:
            base = float(popularity_row[metric])
            value = float(main_row[f"{metric}_mean"])
            improvement_rows.append(
                {
                    "Metric": metric,
                    "Popularity": f"{base:.4f}",
                    "Main model": f"{value:.4f}",
                    "Relative improvement": (
                        f"{(value / base - 1.0) * 100:.2f}%"
                        if base != 0
                        else "N/A"
                    ),
                }
            )
        report_parts += [
            markdown_table(pd.DataFrame(improvement_rows)),
            "",
        ]

    if not group_summary_df.empty:
        main_groups = group_summary_df[
            group_summary_df["method_key"] == "graph_bge_zscore"
        ].copy()
        group_table = pd.DataFrame(
            {
                "Group": main_groups["group"],
                "Recall@5": [
                    fmt(m, s)
                    for m, s in zip(
                        main_groups["Recall@5_mean"],
                        main_groups["Recall@5_std"],
                    )
                ],
                "NDCG@5": [
                    fmt(m, s)
                    for m, s in zip(
                        main_groups["NDCG@5_mean"],
                        main_groups["NDCG@5_std"],
                    )
                ],
                "Recall@10": [
                    fmt(m, s)
                    for m, s in zip(
                        main_groups["Recall@10_mean"],
                        main_groups["Recall@10_std"],
                    )
                ],
                "NDCG@10": [
                    fmt(m, s)
                    for m, s in zip(
                        main_groups["NDCG@10_mean"],
                        main_groups["NDCG@10_std"],
                    )
                ],
                "MAP@10": [
                    fmt(m, s)
                    for m, s in zip(
                        main_groups["MAP@10_mean"],
                        main_groups["MAP@10_std"],
                    )
                ],
            }
        )
        report_parts += [
            "## Main-model Head/Middle/Tail/Unseen results",
            "",
            markdown_table(group_table),
            "",
        ]

    report_parts += [
        "## Interpretation",
        "",
        "- Graph-only is expected to remain close to the popularity ranking.",
        "- BGE-only measures semantic cold-start capability without graph scores.",
        "- The standardized Graph+BGE model is the final main configuration.",
        "- Middle and Tail results should be reported separately.",
        "- Unseen-API performance must not be described as solved when it remains zero.",
        "",
    ]

    (output_dir / "final_report.md").write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )

    print("\nFinal ablation table:")
    print(ablation_df.to_string(index=False))
    print(f"\nSaved summary to: {output_dir}")


if __name__ == "__main__":
    main()
