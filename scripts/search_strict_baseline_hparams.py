#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from strict_baseline_core import (
    deterministic_baseline_scores,
    evaluate_score_matrix,
    fit_inductive_collaborative,
    load_strict_data,
    load_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strict_baselines.yaml"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/strict_baselines/validation"),
    )
    return parser.parse_args()


def grid_rows(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = [
        "latent_dim",
        "knn_k",
        "learning_rate",
        "regularization",
    ]
    values = [section[key] for key in keys]
    rows = []
    for combination in itertools.product(*values):
        row = dict(zip(keys, combination))
        if "lightgcn_layers" in section:
            for layers in section["lightgcn_layers"]:
                rows.append({**row, "lightgcn_layers": layers})
        else:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = Path(config["reference_config"])
    data = load_strict_data(reference_config)
    val_ids = sorted(set(int(value) for value in data.val_pairs[:, 0]))

    leaderboard = []

    for method in config["deterministic_methods"]:
        scores = deterministic_baseline_scores(method, data, val_ids)
        _, metrics, _ = evaluate_score_matrix(
            method,
            scores,
            val_ids,
            data.val_pairs,
        )
        leaderboard.append(
            {
                "method": method,
                "configuration": "{}",
                **metrics,
            }
        )
        print(f"[Validation] {method}: NDCG@10={metrics['NDCG@10']:.6f}")

    common = config["collaborative_common"]
    selected = {}

    for model_name, section_name in (
        ("bpr_mf", "bpr_grid"),
        ("lightgcn", "lightgcn_grid"),
    ):
        candidates = grid_rows(config[section_name])
        model_rows = []

        for index, candidate in enumerate(candidates, start=1):
            print(
                f"\n[{model_name}] candidate {index}/{len(candidates)}: "
                f"{candidate}"
            )
            metadata, _, _, _, val_metrics = fit_inductive_collaborative(
                model_name=model_name,
                data=data,
                val_ids=val_ids,
                val_pairs=data.val_pairs,
                seed=int(config["selection_seed"]),
                latent_dim=int(candidate["latent_dim"]),
                knn_k=int(candidate["knn_k"]),
                learning_rate=float(candidate["learning_rate"]),
                regularization=float(candidate["regularization"]),
                epochs=int(common["epochs"]),
                eval_every=int(common["eval_every"]),
                patience=int(common["patience"]),
                projection_temperature=float(
                    common["projection_temperature"]
                ),
                lightgcn_layers=int(
                    candidate.get("lightgcn_layers", 1)
                ),
                device_name=str(common["device"]),
            )
            row = {
                "method": model_name,
                "configuration": json.dumps(candidate, sort_keys=True),
                "best_epoch": metadata["best_epoch"],
                **val_metrics,
            }
            model_rows.append(row)
            leaderboard.append(row)

        model_frame = pd.DataFrame(model_rows).sort_values(
            ["NDCG@10", "MAP@10", "Recall@10"],
            ascending=False,
        )
        best = model_frame.iloc[0]
        selected[model_name] = {
            **json.loads(best["configuration"]),
            "selected_best_epoch_seed0": int(best["best_epoch"]),
            "selection_metrics": {
                metric: float(best[metric])
                for metric in (
                    "Recall@5",
                    "NDCG@5",
                    "MAP@5",
                    "Recall@10",
                    "NDCG@10",
                    "MAP@10",
                )
            },
        }
        print(f"\nSelected {model_name}: {selected[model_name]}")

    leaderboard_frame = pd.DataFrame(leaderboard)
    leaderboard_frame.to_csv(
        output_dir / "validation_leaderboard.csv",
        index=False,
    )

    selected_payload = {
        "reference_config": str(reference_config),
        "selection_seed": int(config["selection_seed"]),
        "selection_rule": "NDCG@10 > MAP@10 > Recall@10",
        "collaborative_common": common,
        "selected": selected,
    }
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
