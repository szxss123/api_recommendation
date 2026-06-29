#!/usr/bin/env python3
"""
Export strict cold-start ground-truth Mashup–API pairs for case analysis.

This script does not load a checkpoint and does not run model inference.
It reads the same processed split used by the strict-inductive trainer and
writes every positive API for every validation/test Mashup.

Output columns
--------------
mashup_id
api_id
api_group
train_frequency
split
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.finetune_trainer import _as_numpy_pairs
from src.trainers.finetune_trainer_strict_inductive import (
    load_strict_processed_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/case_studies/ground_truth.csv"),
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
    )
    parser.add_argument("--head_ratio", type=float, default=0.20)
    parser.add_argument("--middle_ratio", type=float, default=0.30)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def build_api_groups(
    train_pairs: np.ndarray,
    num_apis: int,
    head_ratio: float,
    middle_ratio: float,
):
    if not (0.0 < head_ratio < 1.0):
        raise ValueError("head_ratio must be in (0,1)")
    if not (0.0 <= middle_ratio < 1.0):
        raise ValueError("middle_ratio must be in [0,1)")
    if head_ratio + middle_ratio >= 1.0:
        raise ValueError("head_ratio + middle_ratio must be < 1")

    frequency = Counter(int(api_id) for _, api_id in train_pairs)
    active = [
        api_id
        for api_id in range(num_apis)
        if frequency.get(api_id, 0) > 0
    ]
    active.sort(key=lambda api_id: (-frequency[api_id], api_id))

    head_count = max(1, int(math.ceil(len(active) * head_ratio)))
    middle_count = int(math.ceil(len(active) * middle_ratio))
    middle_end = min(len(active), head_count + middle_count)

    group_map = {}
    for index, api_id in enumerate(active):
        if index < head_count:
            group_map[api_id] = "Head"
        elif index < middle_end:
            group_map[api_id] = "Middle"
        else:
            group_map[api_id] = "Tail"

    for api_id in range(num_apis):
        if frequency.get(api_id, 0) == 0:
            group_map[api_id] = "Unseen"

    return group_map, frequency


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    processed_dir = config.get("processed_dir")
    if not processed_dir:
        raise ValueError("processed_dir is missing from config")

    (
        graph_train,
        graph_val,
        graph_test,
        train_pairs,
        val_pairs,
        test_pairs,
    ) = load_strict_processed_data(str(processed_dir))

    # The saved split files are dictionaries such as:
    # {"pos": positive_pairs, "neg": negative_pairs, ...}
    # Case-study ground truth must use only the positive pairs.
    if not isinstance(train_pairs, dict) or "pos" not in train_pairs:
        raise TypeError(
            "train_pairs.pt must be a dictionary containing key 'pos'"
        )
    if not isinstance(val_pairs, dict) or "pos" not in val_pairs:
        raise TypeError(
            "val_pairs.pt must be a dictionary containing key 'pos'"
        )
    if not isinstance(test_pairs, dict) or "pos" not in test_pairs:
        raise TypeError(
            "test_pairs.pt must be a dictionary containing key 'pos'"
        )

    train_pairs = _as_numpy_pairs(train_pairs["pos"])
    val_pairs = _as_numpy_pairs(val_pairs["pos"])
    test_pairs = _as_numpy_pairs(test_pairs["pos"])

    num_apis = int(graph_train["api"].num_nodes)
    group_map, frequency = build_api_groups(
        train_pairs,
        num_apis,
        args.head_ratio,
        args.middle_ratio,
    )

    target_pairs = val_pairs if args.split == "val" else test_pairs

    rows = []
    for mashup_id, api_id in target_pairs:
        mashup_id = int(mashup_id)
        api_id = int(api_id)
        rows.append(
            {
                "mashup_id": mashup_id,
                "api_id": api_id,
                "api_group": group_map[api_id],
                "train_frequency": int(frequency.get(api_id, 0)),
                "split": args.split,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pd.DataFrame(rows)
        .drop_duplicates(["mashup_id", "api_id"])
        .sort_values(["mashup_id", "api_id"])
        .reset_index(drop=True)
    )
    frame.to_csv(output, index=False)

    print(f"Saved {len(frame)} positive pairs to: {output}")
    print(f"Eligible Mashups: {frame['mashup_id'].nunique()}")
    print("\nPositive-pair group counts:")
    print(frame["api_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
