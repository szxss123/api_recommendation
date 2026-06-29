#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from strict_baseline_core import (
    METRICS,
    deterministic_baseline_scores,
    evaluate_score_matrix,
    fit_inductive_collaborative,
    format_mean_std,
    load_strict_data,
    load_yaml,
    markdown_table,
    score_inductive_collaborative,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strict_baselines.yaml"),
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path(
            "outputs/strict_baselines/validation/"
            "selected_hyperparameters.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/strict_baselines/test"),
    )
    return parser.parse_args()


def load_existing_ablation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)

    required = {"method", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{path} must contain columns method and seed"
        )

    rename = {
        "Graph+BGE (z-score, λ=0.25)": "Graph+BGE",
        "Graph+BGE (z-score, λ=0.25)": "Graph+BGE",
        "BGE-only": "BGE",
    }
    frame["method"] = frame["method"].replace(rename)
    return frame


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_strict_data(Path(config["reference_config"]))
    test_ids = sorted(set(int(value) for value in data.test_pairs[:, 0]))
    val_ids = sorted(set(int(value) for value in data.val_pairs[:, 0]))

    by_seed_rows: List[Dict] = []
    per_mashup_frames = []

    deterministic_cache = {}
    for method in config["deterministic_methods"]:
        scores = deterministic_baseline_scores(method, data, test_ids)
        frame, metrics, _ = evaluate_score_matrix(
            method,
            scores,
            test_ids,
            data.test_pairs,
        )
        deterministic_cache[method] = metrics
        frame["seed"] = 0
        per_mashup_frames.append(frame)
        for seed in (0, 1, 2):
            by_seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **metrics,
                }
            )
        print(f"[Test] {method}: NDCG@10={metrics['NDCG@10']:.6f}")

    common = selected["collaborative_common"]
    display_names = {
        "bpr_mf": "Inductive BPR-MF",
        "lightgcn": "Inductive LightGCN",
    }

    for model_name in ("bpr_mf", "lightgcn"):
        chosen = selected["selected"][model_name]
        for seed in (0, 1, 2):
            print(f"\nTraining {display_names[model_name]}, seed={seed}")
            metadata, train_ids, train_latent, item_latent, val_metrics = (
                fit_inductive_collaborative(
                    model_name=model_name,
                    data=data,
                    val_ids=val_ids,
                    val_pairs=data.val_pairs,
                    seed=seed,
                    latent_dim=int(chosen["latent_dim"]),
                    knn_k=int(chosen["knn_k"]),
                    learning_rate=float(chosen["learning_rate"]),
                    regularization=float(chosen["regularization"]),
                    epochs=int(common["epochs"]),
                    eval_every=int(common["eval_every"]),
                    patience=int(common["patience"]),
                    projection_temperature=float(
                        common["projection_temperature"]
                    ),
                    lightgcn_layers=int(
                        chosen.get("lightgcn_layers", 1)
                    ),
                    device_name=str(common["device"]),
                )
            )
            scores = score_inductive_collaborative(
                data,
                test_ids,
                train_ids,
                train_latent,
                item_latent,
                int(chosen["knn_k"]),
                float(common["projection_temperature"]),
            )
            frame, metrics, _ = evaluate_score_matrix(
                display_names[model_name],
                scores,
                test_ids,
                data.test_pairs,
            )
            frame["seed"] = seed
            per_mashup_frames.append(frame)
            by_seed_rows.append(
                {
                    "method": display_names[model_name],
                    "seed": seed,
                    "best_epoch": metadata["best_epoch"],
                    "val_NDCG@10": val_metrics["NDCG@10"],
                    **metrics,
                }
            )
            print(
                f"[Test] {display_names[model_name]} seed={seed}: "
                f"NDCG@10={metrics['NDCG@10']:.6f}"
            )

    existing_path = Path(config["existing_ablation_by_seed"])
    existing = load_existing_ablation(existing_path)
    keep_methods = set(config["existing_methods_to_merge"])
    existing = existing[existing["method"].isin(keep_methods)].copy()

    # Verify deterministic BGE/Popularity against the existing implementation.
    for method in ("Popularity", "BGE"):
        old = existing[existing["method"] == method]
        if old.empty:
            continue
        for metric in METRICS:
            difference = abs(float(old[metric].mean()) - deterministic_cache[method][metric])
            if difference > float(config["verification_tolerance"]):
                raise RuntimeError(
                    f"{method} mismatch for {metric}: "
                    f"new={deterministic_cache[method][metric]}, "
                    f"existing={old[metric].mean()}"
                )

    # Avoid duplicate Popularity/BGE rows because the new deterministic run
    # already supplied them.
    existing = existing[
        ~existing["method"].isin({"Popularity", "BGE"})
    ]
    for row in existing.to_dict(orient="records"):
        by_seed_rows.append(row)

    external_path = str(config.get("external_results_csv", "")).strip()
    if external_path:
        external = pd.read_csv(external_path)
        required = {"method", "seed", *METRICS}
        missing = required - set(external.columns)
        if missing:
            raise ValueError(
                f"External result CSV missing columns: {sorted(missing)}"
            )
        by_seed_rows.extend(external.to_dict(orient="records"))

    by_seed = pd.DataFrame(by_seed_rows)
    method_order = config["table_method_order"]
    order_map = {method: index for index, method in enumerate(method_order)}
    by_seed["_order"] = by_seed["method"].map(
        lambda method: order_map.get(method, 999)
    )
    by_seed = by_seed.sort_values(["_order", "seed"]).drop(columns="_order")
    by_seed.to_csv(output_dir / "baseline_results_by_seed.csv", index=False)

    pd.concat(per_mashup_frames, ignore_index=True).to_csv(
        output_dir / "new_baseline_per_mashup_metrics.csv",
        index=False,
    )

    summary_rows = []
    formatted_rows = []
    for method in method_order:
        subset = by_seed[by_seed["method"] == method]
        if subset.empty:
            continue
        summary = {"method": method, "seeds": int(subset["seed"].nunique())}
        formatted = {"Method": method}
        for metric in METRICS:
            mean = float(subset[metric].mean())
            std = float(subset[metric].std(ddof=0))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            formatted[metric] = format_mean_std(mean, std)
        summary_rows.append(summary)
        formatted_rows.append(formatted)

    summary_frame = pd.DataFrame(summary_rows)
    formatted_frame = pd.DataFrame(formatted_rows)
    summary_frame.to_csv(
        output_dir / "baseline_main_table_mean_std.csv",
        index=False,
    )
    formatted_frame.to_csv(
        output_dir / "baseline_main_table_formatted.csv",
        index=False,
    )

    report = [
        "# Strict New-Mashup Cold-Start Baseline Main Table",
        "",
        "All methods use the same Mashup-disjoint train/validation/test split, "
        "the same candidate API catalog, all-API ranking, and identical "
        "Recall/NDCG/HitRate/MAP definitions.",
        "",
        "Vanilla BPR-MF and LightGCN cannot directly represent unseen Mashups. "
        "The table therefore labels their fair cold-start adaptations as "
        "`Inductive BPR-MF` and `Inductive LightGCN`: each unseen Mashup "
        "embedding is projected from its BGE-nearest training Mashups, while "
        "the collaborative model is trained only on training interactions.",
        "",
        "## Main results (mean ± population std over seeds 0/1/2)",
        "",
        markdown_table(formatted_frame),
        "",
        "## Selection protocol",
        "",
        "- TF-IDF, BM25, BGE, Category-Jaccard and Popularity are deterministic.",
        "- Collaborative hyperparameters are selected on seed-0 validation only.",
        "- The same selected hyperparameters are used for all test seeds.",
        "- Test results are not used for parameter selection.",
        "",
    ]
    (output_dir / "baseline_main_table.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nFinal baseline main table:")
    print(formatted_frame.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
