#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import HeteroData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MTFM ProgrammableWeb processed data for the current recommender."
    )
    parser.add_argument(
        "--unified_dir",
        type=Path,
        default=Path("data_unified/mtfm_pw"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed_mtfm_warm_bge_weighted_log_m2_k50"),
    )
    parser.add_argument(
        "--split_mode",
        choices=("interaction_loo", "mashup_random"),
        default="interaction_loo",
        help="interaction_loo is warm-start; mashup_random is cold-start.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_neg_per_pos_train", type=int, default=3)
    parser.add_argument("--num_neg_per_pos_eval", type=int, default=50)
    parser.add_argument(
        "--negative_mode",
        choices=("random", "hard", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)
    parser.add_argument("--max_text_features", type=int, default=512)
    parser.add_argument("--add_api_cooccur", action="store_true")
    parser.add_argument("--cooccur_min_count", type=int, default=2)
    parser.add_argument("--cooccur_topk", type=int, default=50)
    parser.add_argument(
        "--cooccur_weight_mode",
        choices=("binary", "count", "log", "norm", "pmi"),
        default="log",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def read_interactions(path: Path) -> np.ndarray:
    pairs: List[Tuple[int, int]] = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"mashup_id", "api_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            pair = (int(row["mashup_id"]), int(row["api_id"]))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    pairs.sort()
    return np.asarray(pairs, dtype=np.int64)


def normalize_categories(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    result: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def make_text(row: Dict[str, Any]) -> str:
    name = str(row.get("name", "")).strip()
    description = str(row.get("description", "")).strip()
    categories = " ".join(normalize_categories(row.get("categories", [])))
    return " ".join(part for part in (name, description, categories) if part).strip()


def row_normalize_sparse(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return sp.diags(inverse).dot(matrix).tocsr()


def safe_tfidf(texts: Sequence[str], max_features: int) -> torch.Tensor:
    texts = [str(text or "") for text in texts]
    if not texts:
        return torch.zeros((0, max_features), dtype=torch.float32)
    if not any(text.strip() for text in texts):
        return torch.zeros((len(texts), max_features), dtype=torch.float32)

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    matrix = row_normalize_sparse(matrix)
    array = matrix.toarray().astype(np.float32)

    if array.shape[1] < max_features:
        padding = np.zeros(
            (array.shape[0], max_features - array.shape[1]), dtype=np.float32
        )
        array = np.concatenate([array, padding], axis=1)

    return torch.from_numpy(array[:, :max_features])


def interaction_loo_split(
    all_positive: Dict[int, List[int]],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train: List[Tuple[int, int]] = []
    val: List[Tuple[int, int]] = []
    test: List[Tuple[int, int]] = []

    for mashup_id in sorted(all_positive):
        api_ids = sorted(set(int(api_id) for api_id in all_positive[mashup_id]))
        rng.shuffle(api_ids)

        if len(api_ids) == 1:
            train.append((mashup_id, api_ids[0]))
        elif len(api_ids) == 2:
            train.append((mashup_id, api_ids[0]))
            test.append((mashup_id, api_ids[1]))
        elif len(api_ids) >= 3:
            test.append((mashup_id, api_ids[0]))
            val.append((mashup_id, api_ids[1]))
            train.extend((mashup_id, api_id) for api_id in api_ids[2:])

    return (
        np.asarray(train, dtype=np.int64).reshape(-1, 2),
        np.asarray(val, dtype=np.int64).reshape(-1, 2),
        np.asarray(test, dtype=np.int64).reshape(-1, 2),
    )


def mashup_random_split(
    all_positive: Dict[int, List[int]],
    num_mashups: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    mashup_ids = np.arange(num_mashups, dtype=np.int64)
    rng.shuffle(mashup_ids)

    n_train = int(num_mashups * train_ratio)
    n_val = int(num_mashups * val_ratio)
    train_ids = set(mashup_ids[:n_train].tolist())
    val_ids = set(mashup_ids[n_train:n_train + n_val].tolist())
    test_ids = set(mashup_ids[n_train + n_val:].tolist())

    def collect(selected: set[int]) -> np.ndarray:
        pairs = [
            (mashup_id, api_id)
            for mashup_id in sorted(selected)
            for api_id in all_positive.get(mashup_id, [])
        ]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    return collect(train_ids), collect(val_ids), collect(test_ids)


def pairs_to_dict(pairs: np.ndarray) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = defaultdict(list)
    for mashup_id, api_id in np.asarray(pairs, dtype=np.int64):
        result[int(mashup_id)].append(int(api_id))
    return {key: sorted(set(values)) for key, values in result.items()}


def sample_negatives(
    *,
    mashup_id: int,
    anchor_api_id: int,
    all_true_apis: Sequence[int],
    num_apis: int,
    api_categories: Dict[int, List[int]],
    mashup_categories: Dict[int, List[int]],
    category_to_apis: Dict[int, set[int]],
    num_negatives: int,
    hard_negative_ratio: float,
    mode: str,
    rng: random.Random,
) -> List[int]:
    true_set = set(int(api_id) for api_id in all_true_apis)
    if len(true_set) >= num_apis or num_negatives <= 0:
        return []

    hard_pool: set[int] = set()
    if mode in {"hard", "hybrid"}:
        relevant_categories = set(api_categories.get(anchor_api_id, []))
        relevant_categories.update(mashup_categories.get(mashup_id, []))
        for category_id in relevant_categories:
            hard_pool.update(category_to_apis.get(category_id, set()))
        hard_pool.difference_update(true_set)

    if mode == "hard":
        hard_target = num_negatives
    elif mode == "hybrid":
        hard_target = int(round(num_negatives * hard_negative_ratio))
    elif mode == "random":
        hard_target = 0
    else:
        raise ValueError(f"Unsupported negative mode: {mode}")

    selected: List[int] = []
    selected_set: set[int] = set()

    if hard_target > 0 and hard_pool:
        hard_candidates = sorted(hard_pool)
        take = min(hard_target, len(hard_candidates))
        sampled = rng.sample(hard_candidates, take)
        selected.extend(sampled)
        selected_set.update(sampled)

    while len(selected) < num_negatives:
        api_id = rng.randrange(num_apis)
        if api_id in true_set or api_id in selected_set:
            continue
        selected.append(api_id)
        selected_set.add(api_id)

    return selected


def build_negative_pairs(
    *,
    positive_pairs: np.ndarray,
    all_positive: Dict[int, List[int]],
    num_apis: int,
    api_categories: Dict[int, List[int]],
    mashup_categories: Dict[int, List[int]],
    category_to_apis: Dict[int, set[int]],
    num_negatives: int,
    hard_negative_ratio: float,
    mode: str,
    seed: int,
    split_name: str,
) -> np.ndarray:
    rng = random.Random(seed)
    negative_pairs: List[Tuple[int, int]] = []

    print(
        f"[Neg-{split_name}] positives={len(positive_pairs)}, "
        f"num_neg_per_pos={num_negatives}, mode={mode}"
    )

    for index, (mashup_id, api_id) in enumerate(positive_pairs):
        if index % 2000 == 0:
            print(f"[Neg-{split_name}] {index}/{len(positive_pairs)}")

        negatives = sample_negatives(
            mashup_id=int(mashup_id),
            anchor_api_id=int(api_id),
            all_true_apis=all_positive[int(mashup_id)],
            num_apis=num_apis,
            api_categories=api_categories,
            mashup_categories=mashup_categories,
            category_to_apis=category_to_apis,
            num_negatives=num_negatives,
            hard_negative_ratio=hard_negative_ratio,
            mode=mode,
            rng=rng,
        )
        negative_pairs.extend((int(mashup_id), api_id) for api_id in negatives)

    return np.asarray(negative_pairs, dtype=np.int64).reshape(-1, 2)


def cooccur_weight(
    api_a: int,
    api_b: int,
    count: int,
    api_frequency: Counter[int],
    num_mashups: int,
    mode: str,
) -> float:
    if mode == "binary":
        return 1.0
    if mode == "count":
        return float(count)
    if mode == "log":
        return float(math.log1p(count))
    if mode == "norm":
        denominator = math.sqrt(
            max(api_frequency[api_a], 1) * max(api_frequency[api_b], 1)
        )
        return float(count / denominator)
    if mode == "pmi":
        score = math.log(
            (count * max(num_mashups, 1))
            / (max(api_frequency[api_a], 1) * max(api_frequency[api_b], 1))
            + 1e-12
        )
        return float(max(score, 0.0) + 1e-8)
    raise ValueError(f"Unsupported co-occurrence weight mode: {mode}")


def build_cooccur_edges(
    train_positive: Dict[int, List[int]],
    min_count: int,
    topk: int,
    weight_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pair_counter: Counter[Tuple[int, int]] = Counter()
    api_frequency: Counter[int] = Counter()
    valid_mashups = 0

    for api_ids in train_positive.values():
        unique_ids = sorted(set(api_ids))
        if not unique_ids:
            continue
        valid_mashups += 1
        api_frequency.update(unique_ids)
        for api_a, api_b in combinations(unique_ids, 2):
            pair_counter[(api_a, api_b)] += 1

    neighbors: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for (api_a, api_b), count in pair_counter.items():
        if count < min_count:
            continue
        weight = cooccur_weight(
            api_a,
            api_b,
            count,
            api_frequency,
            valid_mashups,
            weight_mode,
        )
        if weight <= 0:
            continue
        neighbors[api_a].append((api_b, count, weight))
        neighbors[api_b].append((api_a, count, weight))

    source: List[int] = []
    target: List[int] = []
    weights: List[float] = []

    for api_id in sorted(neighbors):
        items = sorted(
            neighbors[api_id],
            key=lambda item: (item[2], item[1], -item[0]),
            reverse=True,
        )
        if topk > 0:
            items = items[:topk]
        for neighbor_id, _, weight in items:
            source.append(api_id)
            target.append(neighbor_id)
            weights.append(weight)

    edge_index = (
        np.asarray([source, target], dtype=np.int64)
        if source
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_weight = (
        np.asarray(weights, dtype=np.float32)
        if weights
        else np.empty((0,), dtype=np.float32)
    )

    stats = {
        "raw_pair_count": len(pair_counter),
        "edge_count": int(edge_index.shape[1]),
        "weight_mode": weight_mode,
        "weight_min": float(edge_weight.min()) if len(edge_weight) else 0.0,
        "weight_max": float(edge_weight.max()) if len(edge_weight) else 0.0,
        "weight_mean": float(edge_weight.mean()) if len(edge_weight) else 0.0,
    }
    return edge_index, edge_weight, stats


def save_pairs(path: Path, positive: np.ndarray, negative: np.ndarray) -> None:
    torch.save(
        {
            "pos": np.asarray(positive, dtype=np.int64),
            "neg": np.asarray(negative, dtype=np.int64),
        },
        path,
    )


def validate_no_leakage(
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    train_neg: np.ndarray,
    all_positive: Dict[int, List[int]],
) -> None:
    train_set = set(map(tuple, train_pos.tolist()))
    val_set = set(map(tuple, val_pos.tolist()))
    test_set = set(map(tuple, test_pos.tolist()))

    if train_set & val_set:
        raise RuntimeError("Leakage: train and validation positive pairs overlap.")
    if train_set & test_set:
        raise RuntimeError("Leakage: train and test positive pairs overlap.")
    if val_set & test_set:
        raise RuntimeError("Leakage: validation and test positive pairs overlap.")

    for mashup_id, api_id in train_neg:
        if int(api_id) in set(all_positive[int(mashup_id)]):
            raise RuntimeError(
                f"Leakage: true API {(int(mashup_id), int(api_id))} sampled as train negative."
            )


def main() -> None:
    args = parse_args()
    unified_dir = args.unified_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apis = read_jsonl(unified_dir / "apis.jsonl")
    mashups = read_jsonl(unified_dir / "mashups.jsonl")
    interactions = read_interactions(unified_dir / "interactions.csv")

    apis.sort(key=lambda row: int(row["api_id"]))
    mashups.sort(key=lambda row: int(row["mashup_id"]))

    num_apis = len(apis)
    num_mashups = len(mashups)

    if [int(row["api_id"]) for row in apis] != list(range(num_apis)):
        raise ValueError("API IDs must be continuous and zero-based.")
    if [int(row["mashup_id"]) for row in mashups] != list(range(num_mashups)):
        raise ValueError("Mashup IDs must be continuous and zero-based.")
    if len(interactions):
        if interactions[:, 0].min() < 0 or interactions[:, 0].max() >= num_mashups:
            raise ValueError("interactions.csv contains invalid mashup IDs.")
        if interactions[:, 1].min() < 0 or interactions[:, 1].max() >= num_apis:
            raise ValueError("interactions.csv contains invalid API IDs.")

    all_positive: Dict[int, List[int]] = {
        mashup_id: [] for mashup_id in range(num_mashups)
    }
    for mashup_id, api_id in interactions:
        all_positive[int(mashup_id)].append(int(api_id))
    all_positive = {
        mashup_id: sorted(set(api_ids))
        for mashup_id, api_ids in all_positive.items()
        if api_ids
    }

    all_categories = sorted(
        {
            category
            for row in [*apis, *mashups]
            for category in normalize_categories(row.get("categories", []))
        }
    )
    category_to_id = {
        category: category_id for category_id, category in enumerate(all_categories)
    }

    api_categories: Dict[int, List[int]] = {}
    mashup_categories: Dict[int, List[int]] = {}
    category_to_apis: Dict[int, set[int]] = defaultdict(set)

    api_category_src: List[int] = []
    api_category_dst: List[int] = []
    mashup_category_src: List[int] = []
    mashup_category_dst: List[int] = []

    for row in apis:
        api_id = int(row["api_id"])
        category_ids = [
            category_to_id[category]
            for category in normalize_categories(row.get("categories", []))
        ]
        api_categories[api_id] = category_ids
        for category_id in category_ids:
            api_category_src.append(api_id)
            api_category_dst.append(category_id)
            category_to_apis[category_id].add(api_id)

    for row in mashups:
        mashup_id = int(row["mashup_id"])
        category_ids = [
            category_to_id[category]
            for category in normalize_categories(row.get("categories", []))
        ]
        mashup_categories[mashup_id] = category_ids
        for category_id in category_ids:
            mashup_category_src.append(mashup_id)
            mashup_category_dst.append(category_id)

    mashup_texts = [make_text(row) for row in mashups]
    api_texts = [make_text(row) for row in apis]

    # Mashup and API TF-IDF vectors must share the same vocabulary.
    # Fitting two independent vectorizers would make cosine/dot-product
    # retrieval between the two node types mathematically invalid.
    combined_texts = mashup_texts + api_texts
    combined_features = safe_tfidf(combined_texts, args.max_text_features)
    mashup_features = combined_features[:num_mashups]
    api_features = combined_features[num_mashups:]

    category_features = safe_tfidf(
        all_categories,
        min(128, args.max_text_features),
    )

    if args.split_mode == "interaction_loo":
        train_pos, val_pos, test_pos = interaction_loo_split(
            all_positive,
            args.seed,
        )
    else:
        train_pos, val_pos, test_pos = mashup_random_split(
            all_positive,
            num_mashups,
            args.train_ratio,
            args.val_ratio,
            args.seed,
        )

    train_neg = build_negative_pairs(
        positive_pairs=train_pos,
        all_positive=all_positive,
        num_apis=num_apis,
        api_categories=api_categories,
        mashup_categories=mashup_categories,
        category_to_apis=category_to_apis,
        num_negatives=args.num_neg_per_pos_train,
        hard_negative_ratio=args.hard_negative_ratio,
        mode=args.negative_mode,
        seed=args.seed + 11,
        split_name="train",
    )
    val_neg = build_negative_pairs(
        positive_pairs=val_pos,
        all_positive=all_positive,
        num_apis=num_apis,
        api_categories=api_categories,
        mashup_categories=mashup_categories,
        category_to_apis=category_to_apis,
        num_negatives=args.num_neg_per_pos_eval,
        hard_negative_ratio=args.hard_negative_ratio,
        mode=args.negative_mode,
        seed=args.seed + 29,
        split_name="val",
    )
    test_neg = build_negative_pairs(
        positive_pairs=test_pos,
        all_positive=all_positive,
        num_apis=num_apis,
        api_categories=api_categories,
        mashup_categories=mashup_categories,
        category_to_apis=category_to_apis,
        num_negatives=args.num_neg_per_pos_eval,
        hard_negative_ratio=args.hard_negative_ratio,
        mode=args.negative_mode,
        seed=args.seed + 47,
        split_name="test",
    )

    validate_no_leakage(
        train_pos,
        val_pos,
        test_pos,
        train_neg,
        all_positive,
    )

    train_edge = (
        np.asarray([train_pos[:, 0], train_pos[:, 1]], dtype=np.int64)
        if len(train_pos)
        else np.empty((2, 0), dtype=np.int64)
    )
    reverse_train_edge = (
        np.asarray([train_pos[:, 1], train_pos[:, 0]], dtype=np.int64)
        if len(train_pos)
        else np.empty((2, 0), dtype=np.int64)
    )

    cooccur_edge = np.empty((2, 0), dtype=np.int64)
    cooccur_weight_array = np.empty((0,), dtype=np.float32)
    cooccur_stats: Dict[str, Any] = {
        "raw_pair_count": 0,
        "edge_count": 0,
        "weight_mode": args.cooccur_weight_mode,
        "weight_min": 0.0,
        "weight_max": 0.0,
        "weight_mean": 0.0,
    }

    if args.add_api_cooccur:
        cooccur_edge, cooccur_weight_array, cooccur_stats = build_cooccur_edges(
            pairs_to_dict(train_pos),
            args.cooccur_min_count,
            args.cooccur_topk,
            args.cooccur_weight_mode,
        )

    graph = HeteroData()
    graph["mashup"].x = mashup_features
    graph["api"].x = api_features
    graph["category"].x = category_features

    graph[("mashup", "uses", "api")].edge_index = torch.from_numpy(train_edge)
    graph[("api", "used_by", "mashup")].edge_index = torch.from_numpy(
        reverse_train_edge
    )

    api_category_edge = (
        np.asarray([api_category_src, api_category_dst], dtype=np.int64)
        if api_category_src
        else np.empty((2, 0), dtype=np.int64)
    )
    mashup_category_edge = (
        np.asarray([mashup_category_src, mashup_category_dst], dtype=np.int64)
        if mashup_category_src
        else np.empty((2, 0), dtype=np.int64)
    )

    graph[("api", "has_category", "category")].edge_index = torch.from_numpy(
        api_category_edge
    )
    graph[("category", "category_of_api", "api")].edge_index = torch.from_numpy(
        api_category_edge[[1, 0], :]
    )
    graph[("mashup", "has_category", "category")].edge_index = torch.from_numpy(
        mashup_category_edge
    )
    graph[
        ("category", "category_of_mashup", "mashup")
    ].edge_index = torch.from_numpy(mashup_category_edge[[1, 0], :])

    graph[("api", "co_used_with", "api")].edge_index = torch.from_numpy(
        cooccur_edge
    )
    graph[("api", "co_used_with", "api")].edge_weight = torch.from_numpy(
        cooccur_weight_array
    )

    torch.save(graph, output_dir / "graph_data.pt")
    save_pairs(output_dir / "train_pairs.pt", train_pos, train_neg)
    save_pairs(output_dir / "val_pairs.pt", val_pos, val_neg)
    save_pairs(output_dir / "test_pairs.pt", test_pos, test_neg)

    mappings = {
        "mashup": {
            f"mashup_{int(row['mashup_id'])}": int(row["mashup_id"])
            for row in mashups
        },
        "api": {str(row["name"]): int(row["api_id"]) for row in apis},
        "category": category_to_id,
    }
    reverse_mappings = {
        "mashup": {
            int(row["mashup_id"]): str(row.get("name", ""))
            for row in mashups
        },
        "api": {
            int(row["api_id"]): str(row.get("name", ""))
            for row in apis
        },
        "category": {
            category_id: category
            for category, category_id in category_to_id.items()
        },
    }

    with (output_dir / "node_mapping.pkl").open("wb") as file:
        pickle.dump(mappings, file)
    with (output_dir / "reverse_node_mapping.pkl").open("wb") as file:
        pickle.dump(reverse_mappings, file)

    train_mashup_ids = sorted(set(train_pos[:, 0].tolist())) if len(train_pos) else []
    val_mashup_ids = sorted(set(val_pos[:, 0].tolist())) if len(val_pos) else []
    test_mashup_ids = sorted(set(test_pos[:, 0].tolist())) if len(test_pos) else []

    metadata = {
        "dataset": "mtfm_programmableweb",
        "num_mashups": num_mashups,
        "num_apis": num_apis,
        "num_categories": len(all_categories),
        "candidate_api_ids": list(range(num_apis)),
        "mashup_api_pos": all_positive,
        "mashup_category_ids": mashup_categories,
        "api_category_map": api_categories,
        "train_mashup_ids": train_mashup_ids,
        "val_mashup_ids": val_mashup_ids,
        "test_mashup_ids": test_mashup_ids,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "negative_mode": args.negative_mode,
        "hard_negative_ratio": args.hard_negative_ratio,
        "graph_uses_edges_source": "train_pos_pairs",
        "num_graph_train_uses_edges": int(train_edge.shape[1]),
        "add_api_cooccur": args.add_api_cooccur,
        "cooccur_source": "train",
        "cooccur_min_count": args.cooccur_min_count,
        "cooccur_topk": args.cooccur_topk,
        "cooccur_weight_mode": args.cooccur_weight_mode,
        "num_api_api_cooccur_edges": int(cooccur_edge.shape[1]),
        "cooccur_stats": cooccur_stats,
        "train_pos_count": int(len(train_pos)),
        "val_pos_count": int(len(val_pos)),
        "test_pos_count": int(len(test_pos)),
        "train_neg_count": int(len(train_neg)),
        "val_neg_count": int(len(val_neg)),
        "test_neg_count": int(len(test_neg)),
    }
    with (output_dir / "metadata.pkl").open("wb") as file:
        pickle.dump(metadata, file)

    with (output_dir / "mashup_texts.json").open("w", encoding="utf-8") as file:
        json.dump(mashup_texts, file, ensure_ascii=False)
    with (output_dir / "api_texts.json").open("w", encoding="utf-8") as file:
        json.dump(api_texts, file, ensure_ascii=False)

    print("=" * 80)
    print("MTFM processed dataset completed")
    print("=" * 80)
    print(f"Output:              {output_dir}")
    print(f"Split mode:          {args.split_mode}")
    print(f"Mashups/APIs:        {num_mashups}/{num_apis}")
    print(f"Categories:          {len(all_categories)}")
    print(f"Train positives:     {len(train_pos)}")
    print(f"Validation positives:{len(val_pos)}")
    print(f"Test positives:      {len(test_pos)}")
    print(f"Train negatives:     {len(train_neg)}")
    print(f"Co-occurrence edges: {cooccur_edge.shape[1]}")
    print(
        "Co-occurrence weight:"
        f" {args.cooccur_weight_mode}, "
        f"min={cooccur_stats['weight_min']:.6f}, "
        f"max={cooccur_stats['weight_max']:.6f}, "
        f"mean={cooccur_stats['weight_mean']:.6f}"
    )
    print("Leakage checks:      PASSED")
    print()
    for filename in (
        "graph_data.pt",
        "train_pairs.pt",
        "val_pairs.pt",
        "test_pairs.pt",
        "node_mapping.pkl",
        "reverse_node_mapping.pkl",
        "metadata.pkl",
        "mashup_texts.json",
        "api_texts.json",
    ):
        path = output_dir / filename
        print(f"[OK] {filename}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
