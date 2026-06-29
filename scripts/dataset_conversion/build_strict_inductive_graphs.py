#!/usr/bin/env python3
"""
Build strict-inductive graph views from an existing Mashup-disjoint cold-start
processed directory.

The source directory must already contain:
  graph_data.pt
  train_pairs.pt / val_pairs.pt / test_pairs.pt
  metadata.pkl
  mashup_texts.json / api_texts.json
  mashup_text_emb.npy / api_text_emb.npy (when BGE is used)

The output directory contains:
  graph_train.pt
  graph_val.pt
  graph_test.pt
  graph_data.pt              # identical to graph_train.pt for safe fallback
  copied pair/mapping/text files

Strict-inductive rules
----------------------
Training graph:
  * only train Mashup metadata is visible;
  * val/test Mashup node features are zeroed;
  * val/test Mashup category edges are removed in both directions.

Validation graph:
  * train Mashups may write to Category nodes;
  * validation Mashups may only RECEIVE Category -> Mashup messages;
  * test Mashup features are zeroed and test category edges are absent.

Test graph:
  * train Mashups may write to Category nodes;
  * test Mashups may only RECEIVE Category -> Mashup messages;
  * validation Mashup features are zeroed and validation category edges are absent.

TF-IDF is re-fitted using ONLY train Mashup texts plus the known API catalogue,
then used to transform train/val/test Mashups. This avoids held-out Mashup text
affecting the vocabulary/IDF statistics used during training.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


MASHUP_TO_CATEGORY = ("mashup", "has_category", "category")
CATEGORY_TO_MASHUP = ("category", "category_of_mashup", "mashup")


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list):
        raise TypeError(f"{path} must contain a JSON list")
    return [str(value or "") for value in values]


def row_normalize_sparse(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return sp.diags(inverse).dot(matrix).tocsr()


def fit_strict_tfidf(
    mashup_texts: Sequence[str],
    api_texts: Sequence[str],
    train_mashup_ids: Sequence[int],
    max_features: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fit on train Mashups + all APIs (the API catalogue is assumed known), then
    transform all Mashups and APIs with the same vocabulary.
    """
    train_texts = [mashup_texts[idx] for idx in train_mashup_ids]
    fit_texts = train_texts + list(api_texts)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        lowercase=True,
    )
    vectorizer.fit(fit_texts)

    mashup_matrix = row_normalize_sparse(vectorizer.transform(mashup_texts))
    api_matrix = row_normalize_sparse(vectorizer.transform(api_texts))

    mashup_array = mashup_matrix.toarray().astype(np.float32)
    api_array = api_matrix.toarray().astype(np.float32)

    actual_dim = len(vectorizer.vocabulary_)
    if mashup_array.shape[1] < max_features:
        mashup_array = np.pad(
            mashup_array,
            ((0, 0), (0, max_features - mashup_array.shape[1])),
            mode="constant",
        )
        api_array = np.pad(
            api_array,
            ((0, 0), (0, max_features - api_array.shape[1])),
            mode="constant",
        )

    return (
        torch.from_numpy(mashup_array[:, :max_features]),
        torch.from_numpy(api_array[:, :max_features]),
        actual_dim,
    )


def filter_edge_store(store, mask: torch.Tensor) -> None:
    """
    Filter edge_index and any edge-level tensor whose first dimension equals
    the number of edges.
    """
    edge_count = int(store.edge_index.size(1))
    store.edge_index = store.edge_index[:, mask]

    for key, value in list(store.items()):
        if key == "edge_index" or not torch.is_tensor(value):
            continue
        if value.ndim >= 1 and int(value.size(0)) == edge_count:
            store[key] = value[mask]


def filter_mashup_category_edges(
    graph,
    *,
    outgoing_allowed: Set[int],
    incoming_allowed: Set[int],
) -> None:
    if MASHUP_TO_CATEGORY in graph.edge_types:
        store = graph[MASHUP_TO_CATEGORY]
        src = store.edge_index[0].cpu()
        allowed = torch.tensor(
            [int(x) in outgoing_allowed for x in src.tolist()],
            dtype=torch.bool,
        )
        filter_edge_store(store, allowed)

    if CATEGORY_TO_MASHUP in graph.edge_types:
        store = graph[CATEGORY_TO_MASHUP]
        dst = store.edge_index[1].cpu()
        allowed = torch.tensor(
            [int(x) in incoming_allowed for x in dst.tolist()],
            dtype=torch.bool,
        )
        filter_edge_store(store, allowed)


def zero_mashup_features(graph, hidden_ids: Set[int]) -> None:
    if not hidden_ids:
        return
    indices = torch.tensor(sorted(hidden_ids), dtype=torch.long)
    graph["mashup"].x[indices] = 0.0


def set_shared_features(
    graph,
    mashup_x: torch.Tensor,
    api_x: torch.Tensor,
) -> None:
    graph["mashup"].x = mashup_x.clone()
    graph["api"].x = api_x.clone()


def assert_uses_edges_train_only(graph, train_ids: Set[int]) -> None:
    etype = ("mashup", "uses", "api")
    if etype in graph.edge_types:
        src = set(int(x) for x in graph[etype].edge_index[0].tolist())
        bad = src - train_ids
        if bad:
            raise RuntimeError(
                f"Found non-train Mashup IDs in uses edges: {sorted(bad)[:10]}"
            )


def count_category_edges(graph, ids: Set[int]) -> Tuple[int, int]:
    outgoing = 0
    incoming = 0
    if MASHUP_TO_CATEGORY in graph.edge_types:
        outgoing = sum(
            int(x) in ids
            for x in graph[MASHUP_TO_CATEGORY].edge_index[0].tolist()
        )
    if CATEGORY_TO_MASHUP in graph.edge_types:
        incoming = sum(
            int(x) in ids
            for x in graph[CATEGORY_TO_MASHUP].edge_index[1].tolist()
        )
    return outgoing, incoming


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    filenames = [
        "train_pairs.pt",
        "val_pairs.pt",
        "test_pairs.pt",
        "node_mapping.pkl",
        "reverse_node_mapping.pkl",
        "metadata.pkl",
        "mashup_texts.json",
        "api_texts.json",
        "mashup_text_emb.npy",
        "api_text_emb.npy",
    ]
    for filename in filenames:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create strict-inductive train/val/test graph views."
    )
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--max_text_features",
        type=int,
        default=0,
        help="0 means infer from source graph Mashup x dimension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    required = [
        source_dir / "graph_data.pt",
        source_dir / "metadata.pkl",
        source_dir / "mashup_texts.json",
        source_dir / "api_texts.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Use --overwrite or choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_graph = torch_load(source_dir / "graph_data.pt")
    with (source_dir / "metadata.pkl").open("rb") as file:
        metadata: Dict[str, Any] = pickle.load(file)

    train_ids = set(int(x) for x in metadata["train_mashup_ids"])
    val_ids = set(int(x) for x in metadata["val_mashup_ids"])
    test_ids = set(int(x) for x in metadata["test_mashup_ids"])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Mashup split IDs overlap; source is not Mashup-disjoint.")

    num_mashups = int(base_graph["mashup"].x.size(0))
    all_ids = set(range(num_mashups))
    if train_ids | val_ids | test_ids != all_ids:
        missing_ids = all_ids - (train_ids | val_ids | test_ids)
        extra_ids = (train_ids | val_ids | test_ids) - all_ids
        raise RuntimeError(
            f"Split IDs do not cover graph Mashups. missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )

    assert_uses_edges_train_only(base_graph, train_ids)

    mashup_texts = load_json_list(source_dir / "mashup_texts.json")
    api_texts = load_json_list(source_dir / "api_texts.json")
    if len(mashup_texts) != num_mashups:
        raise ValueError(
            f"mashup_texts length={len(mashup_texts)}, graph nodes={num_mashups}"
        )
    if len(api_texts) != int(base_graph["api"].x.size(0)):
        raise ValueError("api_texts length does not match graph API node count.")

    max_features = (
        args.max_text_features
        if args.max_text_features > 0
        else int(base_graph["mashup"].x.size(1))
    )
    mashup_x, api_x, vocabulary_size = fit_strict_tfidf(
        mashup_texts,
        api_texts,
        sorted(train_ids),
        max_features,
    )

    # Training: held-out Mashups have neither metadata features nor category edges.
    graph_train = copy.deepcopy(base_graph)
    set_shared_features(graph_train, mashup_x, api_x)
    filter_mashup_category_edges(
        graph_train,
        outgoing_allowed=train_ids,
        incoming_allowed=train_ids,
    )
    zero_mashup_features(graph_train, val_ids | test_ids)

    # Validation: validation Mashups may receive messages, never write to categories.
    graph_val = copy.deepcopy(base_graph)
    set_shared_features(graph_val, mashup_x, api_x)
    filter_mashup_category_edges(
        graph_val,
        outgoing_allowed=train_ids,
        incoming_allowed=train_ids | val_ids,
    )
    zero_mashup_features(graph_val, test_ids)

    # Test: test Mashups may receive messages, never write to categories.
    graph_test = copy.deepcopy(base_graph)
    set_shared_features(graph_test, mashup_x, api_x)
    filter_mashup_category_edges(
        graph_test,
        outgoing_allowed=train_ids,
        incoming_allowed=train_ids | test_ids,
    )
    zero_mashup_features(graph_test, val_ids)

    for name, graph in [
        ("train", graph_train),
        ("val", graph_val),
        ("test", graph_test),
    ]:
        assert_uses_edges_train_only(graph, train_ids)
        torch.save(graph, output_dir / f"graph_{name}.pt")

    # Safe fallback: graph_data.pt is the leakage-free training graph.
    torch.save(graph_train, output_dir / "graph_data.pt")
    copy_support_files(source_dir, output_dir)

    strict_metadata = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "protocol": "strict_inductive_cold_start",
        "tfidf_fit_scope": "train_mashups_plus_all_apis",
        "tfidf_max_features": max_features,
        "tfidf_vocabulary_size": vocabulary_size,
        "train_mashup_count": len(train_ids),
        "val_mashup_count": len(val_ids),
        "test_mashup_count": len(test_ids),
        "graph_rules": {
            "train": "train Mashup bidirectional category edges only",
            "val": "train outgoing; train+val incoming category edges",
            "test": "train outgoing; train+test incoming category edges",
        },
    }
    with (output_dir / "strict_inductive_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(strict_metadata, file, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("Strict-inductive graph views created")
    print("=" * 88)
    print(f"Source:            {source_dir}")
    print(f"Output:            {output_dir}")
    print(f"Train/val/test:    {len(train_ids)}/{len(val_ids)}/{len(test_ids)} Mashups")
    print(f"Strict TF-IDF dim: {max_features} (vocabulary={vocabulary_size})")
    print()

    for name, graph, target_ids in [
        ("train", graph_train, train_ids),
        ("val", graph_val, val_ids),
        ("test", graph_test, test_ids),
    ]:
        tr_out, tr_in = count_category_edges(graph, train_ids)
        va_out, va_in = count_category_edges(graph, val_ids)
        te_out, te_in = count_category_edges(graph, test_ids)
        uses = int(
            graph[("mashup", "uses", "api")].edge_index.size(1)
            if ("mashup", "uses", "api") in graph.edge_types
            else 0
        )
        print(f"[{name}] uses={uses}")
        print(f"  train category outgoing/incoming: {tr_out}/{tr_in}")
        print(f"  val   category outgoing/incoming: {va_out}/{va_in}")
        print(f"  test  category outgoing/incoming: {te_out}/{te_in}")

    print()
    print("Expected:")
    print("  graph_train: val/test outgoing=0 and incoming=0")
    print("  graph_val:   val outgoing=0, val incoming>0, test incoming=0")
    print("  graph_test:  test outgoing=0, test incoming>0, val incoming=0")
    print("  all graphs:  uses edges contain train Mashups only")


if __name__ == "__main__":
    main()
