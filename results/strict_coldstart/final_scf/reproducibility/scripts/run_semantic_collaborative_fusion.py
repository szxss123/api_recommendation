#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from strict_baseline_core import (
    METRICS,
    bge_scores,
    evaluate_score_matrix,
    fit_inductive_collaborative,
    format_mean_std,
    load_strict_data,
    load_yaml,
    markdown_table,
    popularity_scores,
    score_inductive_collaborative,
)


MODEL_NAME = "SCF-LightGCN+BGE+Popularity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-selected semantic-collaborative fusion for strict "
            "new-Mashup cold start."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strict_semantic_collaborative_fusion.yaml"),
    )
    return parser.parse_args()


def resolve_existing_path(
    candidates: Sequence[str],
    description: str,
) -> Path:
    checked: List[Path] = []
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        checked.append(path)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n  - "
        + "\n  - ".join(str(path) for path in checked)
    )


def row_zscore(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, eps)


def weight_grid(step: float) -> Iterable[Tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("fusion_grid_step must be in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError(
            "fusion_grid_step must divide 1.0 exactly, e.g. 0.05 or 0.10"
        )
    for lightgcn_units in range(units + 1):
        for bge_units in range(units - lightgcn_units + 1):
            popularity_units = units - lightgcn_units - bge_units
            yield (
                lightgcn_units / units,
                bge_units / units,
                popularity_units / units,
            )


def fuse_scores(
    lightgcn: np.ndarray,
    bge: np.ndarray,
    popularity: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError("Fusion weights must sum to 1")
    return (
        alpha * row_zscore(lightgcn)
        + beta * row_zscore(bge)
        + gamma * row_zscore(popularity)
    ).astype(np.float32)


def selected_lightgcn_settings(
    selected_payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    common = dict(selected_payload["collaborative_common"])
    chosen = dict(selected_payload["selected"]["lightgcn"])
    return common, chosen


def train_lightgcn_scores(
    *,
    data,
    target_ids: Sequence[int],
    target_pairs: np.ndarray,
    validation_ids: Sequence[int],
    seed: int,
    common: Mapping[str, Any],
    chosen: Mapping[str, Any],
):
    metadata, train_ids, train_latent, item_latent, validation_metrics = (
        fit_inductive_collaborative(
            model_name="lightgcn",
            data=data,
            val_ids=validation_ids,
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
            lightgcn_layers=int(chosen.get("lightgcn_layers", 1)),
            device_name=str(common["device"]),
        )
    )

    target_scores = score_inductive_collaborative(
        data,
        target_ids,
        train_ids,
        train_latent,
        item_latent,
        int(chosen["knn_k"]),
        float(common["projection_temperature"]),
    )
    return (
        metadata,
        validation_metrics,
        train_ids,
        train_latent,
        item_latent,
        target_scores,
    )


def evaluate_subset_from_per_mashup(
    frame: pd.DataFrame,
    allowed_ids: set[int],
) -> Dict[str, float]:
    subset = frame[frame["mashup_id"].isin(allowed_ids)]
    if subset.empty:
        raise RuntimeError("The requested clean subset is empty")
    return {
        metric: float(subset[metric].mean())
        for metric in METRICS
    }


def strict_clean_ids(audit_path: Path, threshold: float) -> set[int]:
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
            f"{audit_path} is missing audit columns: {sorted(missing)}"
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


def save_rankings(
    *,
    output_path: Path,
    seed: int,
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, set[int]],
) -> None:
    rows: List[Dict[str, Any]] = []
    for mashup_id in sorted(rankings):
        relevant = positives[mashup_id]
        for rank, api_id in enumerate(rankings[mashup_id], start=1):
            rows.append(
                {
                    "method": MODEL_NAME,
                    "seed": seed,
                    "mashup_id": int(mashup_id),
                    "rank": rank,
                    "api_id": int(api_id),
                    "is_positive": int(api_id in relevant),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def positives_by_mashup(pairs: np.ndarray) -> Dict[int, set[int]]:
    result: Dict[int, set[int]] = {}
    for mashup_id, api_id in pairs:
        result.setdefault(int(mashup_id), set()).add(int(api_id))
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = resolve_existing_path(
        [str(config["reference_config"])],
        "reference strict configuration",
    )
    selected_path = resolve_existing_path(
        list(config["selected_hyperparameter_candidates"]),
        "selected LightGCN hyperparameters",
    )
    baseline_by_seed_path = resolve_existing_path(
        [str(config["baseline_results_by_seed"])],
        "formal baseline by-seed results",
    )

    data = load_strict_data(reference_config)
    selected_payload = json.loads(
        selected_path.read_text(encoding="utf-8")
    )
    common, chosen = selected_lightgcn_settings(selected_payload)

    validation_ids = sorted(
        set(int(value) for value in data.val_pairs[:, 0])
    )
    test_ids = sorted(
        set(int(value) for value in data.test_pairs[:, 0])
    )

    print("Training seed-0 LightGCN for validation-only weight selection...")
    (
        seed0_metadata,
        _,
        seed0_train_ids,
        seed0_train_latent,
        seed0_item_latent,
        validation_lightgcn,
    ) = train_lightgcn_scores(
        data=data,
        target_ids=validation_ids,
        target_pairs=data.val_pairs,
        validation_ids=validation_ids,
        seed=0,
        common=common,
        chosen=chosen,
    )

    validation_bge = bge_scores(data, validation_ids)
    validation_popularity = popularity_scores(data, validation_ids)

    leaderboard_rows: List[Dict[str, Any]] = []
    best_tuple = None
    best_weights = None

    for alpha, beta, gamma in weight_grid(
        float(config["fusion_grid_step"])
    ):
        fused = fuse_scores(
            validation_lightgcn,
            validation_bge,
            validation_popularity,
            alpha,
            beta,
            gamma,
        )
        _, metrics, _ = evaluate_score_matrix(
            MODEL_NAME,
            fused,
            validation_ids,
            data.val_pairs,
        )
        leaderboard_rows.append(
            {
                "lightgcn_weight": alpha,
                "bge_weight": beta,
                "popularity_weight": gamma,
                **metrics,
            }
        )

        selection_tuple = (
            metrics["NDCG@10"],
            metrics["MAP@10"],
            metrics["Recall@10"],
            alpha,
            -gamma,
        )
        if best_tuple is None or selection_tuple > best_tuple:
            best_tuple = selection_tuple
            best_weights = (alpha, beta, gamma)

    validation_leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [
            "NDCG@10",
            "MAP@10",
            "Recall@10",
            "lightgcn_weight",
            "popularity_weight",
        ],
        ascending=[False, False, False, False, True],
    )
    validation_leaderboard.to_csv(
        output_dir / "fusion_validation_leaderboard.csv",
        index=False,
    )

    assert best_weights is not None
    alpha, beta, gamma = best_weights
    selected_weights = {
        "model": MODEL_NAME,
        "selection_split": "validation",
        "selection_seed": 0,
        "selection_rule": "NDCG@10 > MAP@10 > Recall@10",
        "normalization": "per-Mashup row z-score",
        "lightgcn_weight": alpha,
        "bge_weight": beta,
        "popularity_weight": gamma,
        "fusion_grid_step": float(config["fusion_grid_step"]),
        "lightgcn_hyperparameters": chosen,
        "lightgcn_common": common,
        "seed0_best_epoch": int(seed0_metadata["best_epoch"]),
        "selected_hyperparameter_file": str(selected_path),
    }
    (output_dir / "selected_fusion_weights.json").write_text(
        json.dumps(selected_weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Selected validation weights: "
        f"LightGCN={alpha:.2f}, BGE={beta:.2f}, Popularity={gamma:.2f}"
    )

    deterministic_test_bge = bge_scores(data, test_ids)
    deterministic_test_popularity = popularity_scores(data, test_ids)
    test_positives = positives_by_mashup(data.test_pairs)

    per_mashup_frames: List[pd.DataFrame] = []
    result_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []

    audit_path_value = str(config.get("audit_csv", "")).strip()
    clean_ids: set[int] | None = None
    if audit_path_value:
        audit_path = resolve_existing_path(
            [audit_path_value],
            "leakage-audit Mashup CSV",
        )
        clean_ids = strict_clean_ids(
            audit_path,
            float(config["strict_clean_cosine_threshold"]),
        )
        print(f"Strict-clean Mashups: {len(clean_ids)}")

    seed0_test_lightgcn = score_inductive_collaborative(
        data,
        test_ids,
        seed0_train_ids,
        seed0_train_latent,
        seed0_item_latent,
        int(chosen["knn_k"]),
        float(common["projection_temperature"]),
    )

    for seed in (0, 1, 2):
        if seed == 0:
            metadata = seed0_metadata
            test_lightgcn = seed0_test_lightgcn
        else:
            print(f"Training LightGCN seed={seed}...")
            (
                metadata,
                _,
                _,
                _,
                _,
                test_lightgcn,
            ) = train_lightgcn_scores(
                data=data,
                target_ids=test_ids,
                target_pairs=data.test_pairs,
                validation_ids=validation_ids,
                seed=seed,
                common=common,
                chosen=chosen,
            )

        fused_test = fuse_scores(
            test_lightgcn,
            deterministic_test_bge,
            deterministic_test_popularity,
            alpha,
            beta,
            gamma,
        )
        per_mashup, metrics, rankings = evaluate_score_matrix(
            MODEL_NAME,
            fused_test,
            test_ids,
            data.test_pairs,
        )
        per_mashup["seed"] = seed
        per_mashup_frames.append(per_mashup)

        result_rows.append(
            {
                "method": MODEL_NAME,
                "seed": seed,
                "best_epoch": int(metadata["best_epoch"]),
                **metrics,
            }
        )

        save_rankings(
            output_path=output_dir
            / f"scf_rankings_top10_seed{seed}.csv",
            seed=seed,
            rankings=rankings,
            positives=test_positives,
        )

        if clean_ids is not None:
            clean_metrics = evaluate_subset_from_per_mashup(
                per_mashup,
                clean_ids,
            )
            clean_rows.append(
                {
                    "method": MODEL_NAME,
                    "subset": (
                        "strict_clean_cosine_lt_"
                        + str(config["strict_clean_cosine_threshold"]).replace(
                            ".", ""
                        )
                    ),
                    "seed": seed,
                    "num_mashups": len(clean_ids),
                    **clean_metrics,
                }
            )

        print(
            f"Seed {seed}: "
            f"R@10={metrics['Recall@10']:.4f}, "
            f"NDCG@10={metrics['NDCG@10']:.4f}, "
            f"MAP@10={metrics['MAP@10']:.4f}"
        )

    scf_by_seed = pd.DataFrame(result_rows)
    scf_by_seed.to_csv(
        output_dir / "scf_results_by_seed.csv",
        index=False,
    )
    pd.concat(per_mashup_frames, ignore_index=True).to_csv(
        output_dir / "scf_per_mashup_metrics.csv",
        index=False,
    )

    if clean_rows:
        clean_by_seed = pd.DataFrame(clean_rows)
        clean_by_seed.to_csv(
            output_dir / "scf_strict_clean_results_by_seed.csv",
            index=False,
        )

    existing = pd.read_csv(baseline_by_seed_path)
    combined = pd.concat([existing, scf_by_seed], ignore_index=True)
    combined.to_csv(
        output_dir / "baseline_plus_scf_by_seed.csv",
        index=False,
    )

    method_order = list(config["table_method_order"])
    order_map = {
        method: index for index, method in enumerate(method_order)
    }
    summary_rows: List[Dict[str, Any]] = []
    formatted_rows: List[Dict[str, Any]] = []

    for method in method_order:
        subset = combined[combined["method"] == method]
        if subset.empty:
            continue
        summary: Dict[str, Any] = {
            "method": method,
            "num_seeds": int(subset["seed"].nunique()),
        }
        formatted: Dict[str, Any] = {"Method": method}
        for metric in METRICS:
            mean = float(subset[metric].mean())
            std = float(subset[metric].std(ddof=0))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            formatted[metric] = format_mean_std(mean, std)
        summary_rows.append(summary)
        formatted_rows.append(formatted)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["_order"] = summary_frame["method"].map(order_map)
    summary_frame = summary_frame.sort_values("_order").drop(
        columns="_order"
    )
    summary_frame.to_csv(
        output_dir / "baseline_plus_scf_mean_std.csv",
        index=False,
    )

    formatted_frame = pd.DataFrame(formatted_rows)
    formatted_frame["_order"] = formatted_frame["Method"].map(order_map)
    formatted_frame = formatted_frame.sort_values("_order").drop(
        columns="_order"
    )
    formatted_frame.to_csv(
        output_dir / "baseline_plus_scf_formatted.csv",
        index=False,
    )

    report = [
        "# Semantic-Collaborative Fusion under Strict New-Mashup Cold Start",
        "",
        "Fusion weights are selected once on seed-0 validation only. "
        "The test set is never used for selecting weights.",
        "",
        "## Selected weights",
        "",
        f"- Inductive LightGCN: {alpha:.2f}",
        f"- Direct BGE: {beta:.2f}",
        f"- Popularity prior: {gamma:.2f}",
        "- Normalization: per-Mashup row z-score",
        "",
        "## Full test-set main table",
        "",
        markdown_table(formatted_frame),
        "",
    ]

    if clean_rows:
        clean_frame = pd.DataFrame(clean_rows)
        clean_summary = {
            metric: (
                float(clean_frame[metric].mean()),
                float(clean_frame[metric].std(ddof=0)),
            )
            for metric in METRICS
        }
        clean_display = pd.DataFrame(
            [
                {
                    "Method": MODEL_NAME,
                    "Mashups": int(clean_frame["num_mashups"].iloc[0]),
                    **{
                        metric: format_mean_std(*clean_summary[metric])
                        for metric in METRICS
                    },
                }
            ]
        )
        report.extend(
            [
                "## Strict-clean robustness subset",
                "",
                markdown_table(clean_display),
                "",
            ]
        )

    report.extend(
        [
            "## Interpretation rule",
            "",
            "Keep the fusion only if validation selection yields a non-trivial "
            "combination and the test/strict-clean results improve or remain "
            "competitive with Inductive LightGCN. Do not change weights after "
            "seeing test results.",
            "",
        ]
    )

    (output_dir / "semantic_collaborative_fusion_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
