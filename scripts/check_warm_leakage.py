from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import torch


Pair = Tuple[int, int]


def load_pairs(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "pos" not in obj:
            raise KeyError(f"{path} does not contain key 'pos'")
        arr = obj["pos"]
    else:
        arr = obj

    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr, dtype=np.int64)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{path} should have shape [num_pairs, 2], got {arr.shape}")

    return arr


def pairs_to_set(pairs: np.ndarray) -> Set[Pair]:
    return set((int(m), int(a)) for m, a in pairs)


def edge_index_to_set(edge_index) -> Set[Pair]:
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    edge_index = np.asarray(edge_index, dtype=np.int64)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index should have shape [2, num_edges], got {edge_index.shape}")

    src = edge_index[0]
    dst = edge_index[1]

    return set((int(s), int(d)) for s, d in zip(src, dst))


def print_pair_stats(name: str, pairs: np.ndarray) -> None:
    if len(pairs) == 0:
        print(f"\n[{name}] empty")
        return

    mashups = pairs[:, 0]
    apis = pairs[:, 1]

    unique_mashups = np.unique(mashups)
    unique_apis = np.unique(apis)

    counts = {}
    for m, a in pairs:
        m = int(m)
        counts[m] = counts.get(m, 0) + 1

    vals = list(counts.values())

    print(f"\n[{name}]")
    print(f"pos pairs: {len(pairs)}")
    print(f"unique mashups: {len(unique_mashups)}")
    print(f"unique apis: {len(unique_apis)}")
    print(f"avg positives per mashup: {float(np.mean(vals)):.4f}")
    print(f"min positives per mashup: {int(np.min(vals))}")
    print(f"max positives per mashup: {int(np.max(vals))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check warm-start dataset leakage.")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed_warm_bge_cooccur_m2_k50",
        help="Path to processed dataset directory.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)

    graph_path = processed_dir / "graph_data.pt"
    train_path = processed_dir / "train_pairs.pt"
    val_path = processed_dir / "val_pairs.pt"
    test_path = processed_dir / "test_pairs.pt"
    metadata_path = processed_dir / "metadata.pkl"

    required_files = [
        graph_path,
        train_path,
        val_path,
        test_path,
        metadata_path,
    ]

    print("=" * 80)
    print(f"Checking processed_dir: {processed_dir}")
    print("=" * 80)

    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        print(f"[OK] found {p.name}")

    train_pos = load_pairs(train_path)
    val_pos = load_pairs(val_path)
    test_pos = load_pairs(test_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    graph_data = torch.load(graph_path, map_location="cpu")

    print("\n" + "=" * 80)
    print("1. Metadata check")
    print("=" * 80)

    split_mode = metadata.get("split_mode", None)
    graph_uses_edges_source = metadata.get("graph_uses_edges_source", None)
    cooccur_source = metadata.get("cooccur_source", None)

    print("split_mode:", split_mode)
    print("graph_uses_edges_source:", graph_uses_edges_source)
    print("cooccur_source:", cooccur_source)
    print("add_api_cooccur:", metadata.get("add_api_cooccur", None))
    print("cooccur_min_count:", metadata.get("cooccur_min_count", None))
    print("cooccur_topk:", metadata.get("cooccur_topk", None))
    print("num_api_api_cooccur_edges:", metadata.get("num_api_api_cooccur_edges", None))
    print("num_graph_train_uses_edges:", metadata.get("num_graph_train_uses_edges", None))

    if split_mode == "interaction_loo":
        print("[PASS] split_mode is interaction_loo")
    else:
        print("[WARN] split_mode is not interaction_loo. This may not be warm-start data.")

    if graph_uses_edges_source == "train_pos_pairs":
        print("[PASS] graph uses edges are recorded as train_pos_pairs")
    else:
        print("[WARN] graph_uses_edges_source is not train_pos_pairs")

    if cooccur_source == "train":
        print("[PASS] API-API cooccur edges are recorded as train-only")
    else:
        print("[WARN] cooccur_source is not train. API-API cooccur may use val/test info.")

    print("\n" + "=" * 80)
    print("2. Pair statistics")
    print("=" * 80)

    print_pair_stats("train", train_pos)
    print_pair_stats("val", val_pos)
    print_pair_stats("test", test_pos)

    train_set = pairs_to_set(train_pos)
    val_set = pairs_to_set(val_pos)
    test_set = pairs_to_set(test_pos)

    train_m = set(int(x) for x in train_pos[:, 0].tolist()) if len(train_pos) else set()
    val_m = set(int(x) for x in val_pos[:, 0].tolist()) if len(val_pos) else set()
    test_m = set(int(x) for x in test_pos[:, 0].tolist()) if len(test_pos) else set()

    train_a = set(int(x) for x in train_pos[:, 1].tolist()) if len(train_pos) else set()
    val_a = set(int(x) for x in val_pos[:, 1].tolist()) if len(val_pos) else set()
    test_a = set(int(x) for x in test_pos[:, 1].tolist()) if len(test_pos) else set()

    print("\n" + "=" * 80)
    print("3. Split overlap check")
    print("=" * 80)

    print("train & val pair overlap:", len(train_set & val_set))
    print("train & test pair overlap:", len(train_set & test_set))
    print("val & test pair overlap:", len(val_set & test_set))

    print("\ntrain & val mashup overlap:", len(train_m & val_m))
    print("train & test mashup overlap:", len(train_m & test_m))
    print("val & test mashup overlap:", len(val_m & test_m))

    if len(train_m & test_m) > 0:
        print("[PASS] train/test mashup overlap > 0, warm-start behavior exists.")
    else:
        print("[WARN] train/test mashup overlap = 0, this looks like cold-start.")

    unseen_test_apis = test_a - train_a
    unseen_val_apis = val_a - train_a

    print("\nAPI coverage")
    print("train unique APIs:", len(train_a))
    print("val unique APIs:", len(val_a))
    print("test unique APIs:", len(test_a))
    print("val APIs not seen in train:", len(unseen_val_apis))
    print("test APIs not seen in train:", len(unseen_test_apis))
    print("val unseen API ratio:", len(unseen_val_apis) / max(len(val_a), 1))
    print("test unseen API ratio:", len(unseen_test_apis) / max(len(test_a), 1))

    print("\n" + "=" * 80)
    print("4. Graph edge leakage check")
    print("=" * 80)

    edge_type = ("mashup", "uses", "api")

    if edge_type not in graph_data.edge_types:
        raise KeyError(f"Graph does not contain edge type: {edge_type}")

    graph_uses_edge_index = graph_data[edge_type].edge_index
    graph_uses_set = edge_index_to_set(graph_uses_edge_index)

    print("graph mashup-uses-api edge count:", len(graph_uses_set))
    print("train_pos pair count:", len(train_set))

    missing_train_edges = train_set - graph_uses_set
    extra_graph_edges = graph_uses_set - train_set

    print("train pairs missing from graph:", len(missing_train_edges))
    print("extra graph edges not in train:", len(extra_graph_edges))

    if len(missing_train_edges) == 0 and len(extra_graph_edges) == 0:
        print("[PASS] graph mashup-uses-api edges exactly match train_pos_pairs.")
    else:
        print("[FAIL] graph mashup-uses-api edges do not exactly match train_pos_pairs.")
        if len(missing_train_edges) > 0:
            print("sample missing train edges:", list(missing_train_edges)[:10])
        if len(extra_graph_edges) > 0:
            print("sample extra graph edges:", list(extra_graph_edges)[:10])

    leaked_val_edges = graph_uses_set & val_set
    leaked_test_edges = graph_uses_set & test_set

    print("\nval positive edges leaked into graph:", len(leaked_val_edges))
    print("test positive edges leaked into graph:", len(leaked_test_edges))

    if len(leaked_val_edges) == 0 and len(leaked_test_edges) == 0:
        print("[PASS] no val/test positive edges leaked into mashup-uses-api graph.")
    else:
        print("[FAIL] val/test positive edges leaked into graph!")
        if len(leaked_val_edges) > 0:
            print("sample leaked val edges:", list(leaked_val_edges)[:10])
        if len(leaked_test_edges) > 0:
            print("sample leaked test edges:", list(leaked_test_edges)[:10])

    print("\n" + "=" * 80)
    print("5. Reverse edge check")
    print("=" * 80)

    rev_edge_type = ("api", "used_by", "mashup")

    if rev_edge_type in graph_data.edge_types:
        rev_set_raw = edge_index_to_set(graph_data[rev_edge_type].edge_index)
        rev_set = set((m, a) for a, m in rev_set_raw)

        print("reverse api-used_by-mashup edge count:", len(rev_set))

        missing_rev_edges = train_set - rev_set
        extra_rev_edges = rev_set - train_set

        print("train pairs missing from reverse graph:", len(missing_rev_edges))
        print("extra reverse graph edges not in train:", len(extra_rev_edges))

        if len(missing_rev_edges) == 0 and len(extra_rev_edges) == 0:
            print("[PASS] reverse edges exactly match train_pos_pairs.")
        else:
            print("[FAIL] reverse edges do not exactly match train_pos_pairs.")
            if len(missing_rev_edges) > 0:
                print("sample missing reverse edges:", list(missing_rev_edges)[:10])
            if len(extra_rev_edges) > 0:
                print("sample extra reverse edges:", list(extra_rev_edges)[:10])
    else:
        print("[WARN] graph does not contain reverse edge type:", rev_edge_type)

    print("\n" + "=" * 80)
    print("6. API-API cooccur edge check")
    print("=" * 80)

    cooccur_edge_type = ("api", "co_used_with", "api")

    if cooccur_edge_type in graph_data.edge_types:
        co_edge_index = graph_data[cooccur_edge_type].edge_index
        print("api-api cooccur edge shape:", tuple(co_edge_index.shape))
        print("api-api cooccur edge count:", int(co_edge_index.shape[1]))

        expected = metadata.get("num_api_api_cooccur_edges", None)
        if expected is not None:
            print("metadata num_api_api_cooccur_edges:", expected)
            if int(co_edge_index.shape[1]) == int(expected):
                print("[PASS] cooccur edge count matches metadata.")
            else:
                print("[WARN] cooccur edge count does not match metadata.")
    else:
        print("[WARN] graph does not contain API-API cooccur edge.")

    print("\n" + "=" * 80)
    print("7. Final conclusion")
    print("=" * 80)

    hard_fail = False

    if len(extra_graph_edges) > 0:
        hard_fail = True

    if len(leaked_val_edges) > 0 or len(leaked_test_edges) > 0:
        hard_fail = True

    if graph_uses_edges_source != "train_pos_pairs":
        hard_fail = True

    if cooccur_source != "train":
        print("[WARN] cooccur_source is not train. This may be acceptable for exploration, but not strict evaluation.")

    if hard_fail:
        print("[RESULT] Leakage risk detected. Please rebuild the dataset.")
    else:
        print("[RESULT] No mashup-API graph leakage detected.")
        print("         Warm-start setting looks valid if split_mode=interaction_loo and train/test mashup overlap > 0.")


if __name__ == "__main__":
    main()