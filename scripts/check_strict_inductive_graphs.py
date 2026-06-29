#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Set, Tuple

import torch


M2C = ("mashup", "has_category", "category")
C2M = ("category", "category_of_mashup", "mashup")


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def counts(graph, ids: Set[int]) -> Tuple[int, int, int]:
    uses = 0
    out_count = 0
    in_count = 0

    uses_type = ("mashup", "uses", "api")
    if uses_type in graph.edge_types:
        uses = sum(
            int(x) in ids
            for x in graph[uses_type].edge_index[0].tolist()
        )
    if M2C in graph.edge_types:
        out_count = sum(
            int(x) in ids
            for x in graph[M2C].edge_index[0].tolist()
        )
    if C2M in graph.edge_types:
        in_count = sum(
            int(x) in ids
            for x in graph[C2M].edge_index[1].tolist()
        )
    return uses, out_count, in_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, required=True)
    args = parser.parse_args()

    d = args.processed_dir
    with (d / "metadata.pkl").open("rb") as file:
        metadata = pickle.load(file)

    split_ids = {
        "train": set(map(int, metadata["train_mashup_ids"])),
        "val": set(map(int, metadata["val_mashup_ids"])),
        "test": set(map(int, metadata["test_mashup_ids"])),
    }

    graphs = {
        "train": load(d / "graph_train.pt"),
        "val": load(d / "graph_val.pt"),
        "test": load(d / "graph_test.pt"),
    }

    print("Columns: uses | mashup->category | category->mashup")
    for graph_name, graph in graphs.items():
        print(f"\n[{graph_name} graph]")
        for split_name, ids in split_ids.items():
            print(f"{split_name:5s}: {counts(graph, ids)}")

    train_g = graphs["train"]
    val_g = graphs["val"]
    test_g = graphs["test"]

    assert counts(train_g, split_ids["val"])[1:] == (0, 0)
    assert counts(train_g, split_ids["test"])[1:] == (0, 0)

    assert counts(val_g, split_ids["val"])[1] == 0
    assert counts(val_g, split_ids["val"])[2] > 0
    assert counts(val_g, split_ids["test"])[1:] == (0, 0)

    assert counts(test_g, split_ids["test"])[1] == 0
    assert counts(test_g, split_ids["test"])[2] > 0
    assert counts(test_g, split_ids["val"])[1:] == (0, 0)

    for graph in graphs.values():
        assert counts(graph, split_ids["val"])[0] == 0
        assert counts(graph, split_ids["test"])[0] == 0

    print("\nSTRICT INDUCTIVE CHECKS: PASSED")


if __name__ == "__main__":
    main()
