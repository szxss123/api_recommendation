#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from strict_baseline_core import (
    METRICS,
    evaluate_score_matrix,
    fit_inductive_collaborative,
    load_strict_data,
    load_yaml,
    score_inductive_collaborative,
)


METHOD_NAME = "Inductive LightGCN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the validation-selected Inductive LightGCN and export "
            "its full test Top-10 rankings for diversity/long-tail analysis."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scf_tail_diversity.yaml"),
    )
    return parser.parse_args()


def resolve_first_existing(
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


def positive_sets(pairs: np.ndarray) -> Dict[int, set[int]]:
    result: Dict[int, set[int]] = {}
    for mashup_id, api_id in pairs:
        result.setdefault(int(mashup_id), set()).add(int(api_id))
    return result


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
    return frame.sort_values("api_id").reset_index(drop=True)


def save_rankings(
    *,
    path: Path,
    seed: int,
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, set[int]],
    api_groups: pd.DataFrame,
) -> None:
    metadata = api_groups.set_index("api_id")
    rows: List[Dict[str, Any]] = []

    for mashup_id in sorted(rankings):
        relevant = positives[mashup_id]
        for rank, api_id in enumerate(rankings[mashup_id], start=1):
            rows.append(
                {
                    "method": METHOD_NAME,
                    "seed": seed,
                    "mashup_id": int(mashup_id),
                    "rank": rank,
                    "api_id": int(api_id),
                    "is_positive": int(api_id in relevant),
                    "api_group": str(metadata.loc[api_id, "group"]),
                    "train_frequency": int(
                        metadata.loc[api_id, "train_frequency"]
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["lightgcn_export_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = resolve_first_existing(
        [str(config["reference_config"])],
        "strict reference configuration",
    )
    selected_path = resolve_first_existing(
        list(config["selected_hyperparameter_candidates"]),
        "validation-selected LightGCN hyperparameters",
    )
    api_groups_path = resolve_first_existing(
        [str(config["api_groups_source"])],
        "API group metadata",
    )

    data = load_strict_data(reference_config)
    selected_payload = json.loads(
        selected_path.read_text(encoding="utf-8")
    )
    common = dict(selected_payload["collaborative_common"])
    chosen = dict(selected_payload["selected"]["lightgcn"])

    api_groups = load_api_groups(api_groups_path)
    if len(api_groups) != data.num_apis:
        raise ValueError(
            f"API group count {len(api_groups)} != catalog size {data.num_apis}"
        )

    validation_ids = sorted(
        set(int(value) for value in data.val_pairs[:, 0])
    )
    test_ids = sorted(
        set(int(value) for value in data.test_pairs[:, 0])
    )
    positives = positive_sets(data.test_pairs)

    result_rows: List[Dict[str, Any]] = []
    per_mashup_frames: List[pd.DataFrame] = []

    for seed in (0, 1, 2):
        print(f"Training {METHOD_NAME}, seed={seed}...")
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

        scores = score_inductive_collaborative(
            data,
            test_ids,
            train_ids,
            train_latent,
            item_latent,
            int(chosen["knn_k"]),
            float(common["projection_temperature"]),
        )
        per_mashup, metrics, rankings = evaluate_score_matrix(
            METHOD_NAME,
            scores,
            test_ids,
            data.test_pairs,
        )
        per_mashup["seed"] = seed
        per_mashup_frames.append(per_mashup)

        result_rows.append(
            {
                "method": METHOD_NAME,
                "seed": seed,
                "best_epoch": int(metadata["best_epoch"]),
                "validation_NDCG@10": float(
                    validation_metrics["NDCG@10"]
                ),
                **metrics,
            }
        )

        save_rankings(
            path=output_dir
            / f"inductive_lightgcn_rankings_top10_seed{seed}.csv",
            seed=seed,
            rankings=rankings,
            positives=positives,
            api_groups=api_groups,
        )

        print(
            f"seed={seed}: "
            f"R@10={metrics['Recall@10']:.4f}, "
            f"NDCG@10={metrics['NDCG@10']:.4f}, "
            f"MAP@10={metrics['MAP@10']:.4f}"
        )

    by_seed = pd.DataFrame(result_rows)
    by_seed.to_csv(
        output_dir / "inductive_lightgcn_results_by_seed.csv",
        index=False,
    )
    pd.concat(per_mashup_frames, ignore_index=True).to_csv(
        output_dir / "inductive_lightgcn_per_mashup_metrics.csv",
        index=False,
    )

    expected_path_value = str(
        config.get("expected_baseline_results_by_seed", "")
    ).strip()
    if expected_path_value:
        expected_path = resolve_first_existing(
            [expected_path_value],
            "existing formal baseline results",
        )
        expected = pd.read_csv(expected_path)
        expected = expected[
            expected["method"] == METHOD_NAME
        ].sort_values("seed")
        actual = by_seed.sort_values("seed")

        checks: List[Dict[str, Any]] = []
        tolerance = float(config.get("reproduction_tolerance", 0.005))
        for metric in METRICS:
            for seed in (0, 1, 2):
                old_rows = expected[expected["seed"] == seed]
                new_rows = actual[actual["seed"] == seed]
                if old_rows.empty or new_rows.empty:
                    continue
                old_value = float(old_rows.iloc[0][metric])
                new_value = float(new_rows.iloc[0][metric])
                difference = new_value - old_value
                checks.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "expected": old_value,
                        "reproduced": new_value,
                        "difference": difference,
                        "within_tolerance": (
                            abs(difference) <= tolerance
                        ),
                    }
                )

        check_frame = pd.DataFrame(checks)
        check_frame.to_csv(
            output_dir / "reproduction_check.csv",
            index=False,
        )
        if not check_frame.empty and not bool(
            check_frame["within_tolerance"].all()
        ):
            worst = check_frame.iloc[
                check_frame["difference"].abs().argmax()
            ]
            raise RuntimeError(
                "LightGCN reproduction differs from the formal baseline by "
                f"more than tolerance={tolerance}. Worst row: "
                f"{worst.to_dict()}"
            )

    metadata_payload = {
        "method": METHOD_NAME,
        "selected_hyperparameter_file": str(selected_path),
        "selected_lightgcn": chosen,
        "collaborative_common": common,
        "api_groups_source": str(api_groups_path),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
