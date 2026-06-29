#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def as_positive_pairs(obj: Any) -> np.ndarray:
    if isinstance(obj, Mapping):
        if "pos" not in obj:
            raise KeyError("Pair dictionary does not contain key 'pos'")
        obj = obj["pos"]
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()
    array = np.asarray(obj, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected pair array [N,2], got {array.shape}")
    return array


def load_json_sequence(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        keys = sorted(value, key=lambda item: int(item))
        return [str(value[key]) for key in keys]
    raise TypeError(f"Unsupported JSON text container: {path}")


@dataclass
class StrictData:
    processed_dir: Path
    graph_test: Any
    train_pairs: np.ndarray
    val_pairs: np.ndarray
    test_pairs: np.ndarray
    mashup_text_emb: np.ndarray
    api_text_emb: np.ndarray
    mashup_texts: List[str]
    api_texts: List[str]
    num_mashups: int
    num_apis: int


def load_strict_data(reference_config: Path) -> StrictData:
    config = load_yaml(reference_config)
    processed_dir = Path(str(config["processed_dir"])).expanduser()
    if not processed_dir.is_absolute():
        processed_dir = (Path.cwd() / processed_dir).resolve()

    required = [
        "graph_test.pt",
        "train_pairs.pt",
        "val_pairs.pt",
        "test_pairs.pt",
        "mashup_text_emb.npy",
        "api_text_emb.npy",
        "mashup_texts.json",
        "api_texts.json",
    ]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed files:\n  - " + "\n  - ".join(missing)
        )

    graph_test = torch_load(processed_dir / "graph_test.pt")
    train_pairs = as_positive_pairs(torch_load(processed_dir / "train_pairs.pt"))
    val_pairs = as_positive_pairs(torch_load(processed_dir / "val_pairs.pt"))
    test_pairs = as_positive_pairs(torch_load(processed_dir / "test_pairs.pt"))

    mashup_text_emb = np.load(processed_dir / "mashup_text_emb.npy").astype(np.float32)
    api_text_emb = np.load(processed_dir / "api_text_emb.npy").astype(np.float32)
    mashup_texts = load_json_sequence(processed_dir / "mashup_texts.json")
    api_texts = load_json_sequence(processed_dir / "api_texts.json")

    num_mashups = int(mashup_text_emb.shape[0])
    num_apis = int(api_text_emb.shape[0])

    if len(mashup_texts) != num_mashups:
        raise ValueError("mashup_texts.json length does not match embeddings")
    if len(api_texts) != num_apis:
        raise ValueError("api_texts.json length does not match embeddings")

    return StrictData(
        processed_dir=processed_dir,
        graph_test=graph_test,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        mashup_text_emb=mashup_text_emb,
        api_text_emb=api_text_emb,
        mashup_texts=mashup_texts,
        api_texts=api_texts,
        num_mashups=num_mashups,
        num_apis=num_apis,
    )


def positives_by_mashup(pairs: np.ndarray) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = defaultdict(set)
    for mashup_id, api_id in pairs:
        result[int(mashup_id)].add(int(api_id))
    return dict(result)


def dcg(hits: Sequence[int]) -> float:
    return float(
        sum(
            hit / math.log2(rank + 1.0)
            for rank, hit in enumerate(hits, start=1)
        )
    )


def per_mashup_rows(
    method: str,
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, Set[int]],
    ks: Sequence[int] = (5, 10),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mashup_id in sorted(positives):
        relevant = positives[mashup_id]
        ranked = list(rankings[mashup_id])
        row: Dict[str, Any] = {
            "method": method,
            "mashup_id": mashup_id,
            "num_positives": len(relevant),
        }
        for k in ks:
            topk = ranked[:k]
            hits = [int(api_id in relevant) for api_id in topk]
            hit_count = sum(hits)
            row[f"Recall@{k}"] = hit_count / len(relevant)
            row[f"HitRate@{k}"] = float(hit_count > 0)

            ideal = [1] * min(len(relevant), k)
            denominator = dcg(ideal)
            row[f"NDCG@{k}"] = dcg(hits) / denominator if denominator else 0.0

            cumulative = 0
            precision_sum = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    cumulative += 1
                    precision_sum += cumulative / rank
            row[f"MAP@{k}"] = precision_sum / min(len(relevant), k)
        rows.append(row)
    return rows


def overall_from_per_mashup(frame: pd.DataFrame) -> Dict[str, float]:
    return {metric: float(frame[metric].mean()) for metric in METRICS}


def topk_rankings(
    scores: np.ndarray,
    mashup_ids: Sequence[int],
    k: int = 10,
) -> Dict[int, List[int]]:
    if scores.shape[0] != len(mashup_ids):
        raise ValueError("Score rows do not match Mashup IDs")
    k = min(k, scores.shape[1])
    partition = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    partition_scores = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1)
    topk = np.take_along_axis(partition, order, axis=1)
    return {
        int(mashup_id): [int(value) for value in topk[row]]
        for row, mashup_id in enumerate(mashup_ids)
    }


def evaluate_score_matrix(
    method: str,
    scores: np.ndarray,
    mashup_ids: Sequence[int],
    positive_pairs: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[int, List[int]]]:
    positives = positives_by_mashup(positive_pairs)
    expected_ids = sorted(positives)
    actual_ids = [int(value) for value in mashup_ids]
    if actual_ids != expected_ids:
        raise ValueError("Mashup IDs must match sorted positive-pair Mashup IDs")

    rankings = topk_rankings(scores, actual_ids, k=10)
    frame = pd.DataFrame(per_mashup_rows(method, rankings, positives))
    return frame, overall_from_per_mashup(frame), rankings


def normalize_rows(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def popularity_scores(data: StrictData, target_ids: Sequence[int]) -> np.ndarray:
    counts = np.bincount(data.train_pairs[:, 1], minlength=data.num_apis).astype(np.float32)
    api_ids = np.arange(data.num_apis, dtype=np.float32)
    # Deterministic tie-breaking in favor of lower API IDs.
    vector = counts - api_ids * 1e-9
    return np.broadcast_to(vector, (len(target_ids), data.num_apis)).copy()


def bge_scores(data: StrictData, target_ids: Sequence[int]) -> np.ndarray:
    query = normalize_rows(data.mashup_text_emb[np.asarray(target_ids)])
    api = normalize_rows(data.api_text_emb)
    return query @ api.T


def tfidf_scores(data: StrictData, target_ids: Sequence[int]) -> np.ndarray:
    train_mashup_ids = sorted(set(int(value) for value in data.train_pairs[:, 0]))
    fit_texts = [data.mashup_texts[idx] for idx in train_mashup_ids] + data.api_texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(fit_texts)
    api_matrix = vectorizer.transform(data.api_texts)
    query_matrix = vectorizer.transform([data.mashup_texts[idx] for idx in target_ids])
    return linear_kernel(query_matrix, api_matrix).astype(np.float32)


def bm25_scores(
    data: StrictData,
    target_ids: Sequence[int],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    vectorizer = CountVectorizer(lowercase=True, min_df=1)
    doc_tf = vectorizer.fit_transform(data.api_texts).astype(np.float32)
    query_tf = vectorizer.transform([data.mashup_texts[idx] for idx in target_ids])
    query_tf.data[:] = 1.0

    n_docs = doc_tf.shape[0]
    document_frequency = np.asarray((doc_tf > 0).sum(axis=0)).reshape(-1)
    idf = np.log((n_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)

    document_length = np.asarray(doc_tf.sum(axis=1)).reshape(-1)
    average_length = max(float(document_length.mean()), 1e-12)

    coo = doc_tf.tocoo()
    denominator = (
        coo.data
        + k1 * (1.0 - b + b * document_length[coo.row] / average_length)
    )
    weighted_data = (
        coo.data * (k1 + 1.0) / denominator * idf[coo.col]
    )
    weighted_docs = sparse.csr_matrix(
        (weighted_data, (coo.row, coo.col)),
        shape=doc_tf.shape,
    )
    return (query_tf @ weighted_docs.T).toarray().astype(np.float32)


def _edge_pairs(graph: Any, source: str, target: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for edge_type in graph.edge_types:
        src, _, dst = edge_type
        if src == source and dst == target:
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            pairs.extend(
                (int(s), int(t))
                for s, t in zip(edge_index[0], edge_index[1])
            )
        elif src == target and dst == source:
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            pairs.extend(
                (int(t), int(s))
                for s, t in zip(edge_index[0], edge_index[1])
            )
    return pairs


def category_jaccard_scores(data: StrictData, target_ids: Sequence[int]) -> np.ndarray:
    mashup_category_pairs = _edge_pairs(data.graph_test, "mashup", "category")
    api_category_pairs = _edge_pairs(data.graph_test, "api", "category")
    if not mashup_category_pairs or not api_category_pairs:
        raise RuntimeError("Could not find Mashup/API category edges in graph_test.pt")

    num_categories = int(data.graph_test["category"].num_nodes)
    m_rows = np.asarray([pair[0] for pair in mashup_category_pairs], dtype=np.int64)
    m_cols = np.asarray([pair[1] for pair in mashup_category_pairs], dtype=np.int64)
    a_rows = np.asarray([pair[0] for pair in api_category_pairs], dtype=np.int64)
    a_cols = np.asarray([pair[1] for pair in api_category_pairs], dtype=np.int64)

    mashup_matrix = sparse.csr_matrix(
        (np.ones(len(m_rows)), (m_rows, m_cols)),
        shape=(data.num_mashups, num_categories),
    )
    api_matrix = sparse.csr_matrix(
        (np.ones(len(a_rows)), (a_rows, a_cols)),
        shape=(data.num_apis, num_categories),
    )

    query = mashup_matrix[np.asarray(target_ids)]
    intersection = (query @ api_matrix.T).toarray()
    q_size = np.asarray(query.sum(axis=1)).reshape(-1, 1)
    a_size = np.asarray(api_matrix.sum(axis=1)).reshape(1, -1)
    union = q_size + a_size - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    ).astype(np.float32)


def sample_negatives(
    user_indices: np.ndarray,
    positive_sets: Sequence[Set[int]],
    num_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, num_items, size=len(user_indices), dtype=np.int64)
    invalid = np.fromiter(
        (
            int(negative in positive_sets[int(user)])
            for user, negative in zip(user_indices, negatives)
        ),
        dtype=bool,
        count=len(user_indices),
    )
    while invalid.any():
        negatives[invalid] = rng.integers(
            0,
            num_items,
            size=int(invalid.sum()),
            dtype=np.int64,
        )
        invalid = np.fromiter(
            (
                int(negative in positive_sets[int(user)])
                for user, negative in zip(user_indices, negatives)
            ),
            dtype=bool,
            count=len(user_indices),
        )
    return negatives


def project_unseen_users(
    query_text_emb: np.ndarray,
    train_text_emb: np.ndarray,
    train_latent: np.ndarray,
    knn_k: int,
    temperature: float,
    chunk_size: int = 512,
) -> np.ndarray:
    query = normalize_rows(query_text_emb.astype(np.float32))
    train = normalize_rows(train_text_emb.astype(np.float32))
    k = min(knn_k, train.shape[0])
    projected = np.empty((query.shape[0], train_latent.shape[1]), dtype=np.float32)

    for start in range(0, query.shape[0], chunk_size):
        end = min(query.shape[0], start + chunk_size)
        similarity = query[start:end] @ train.T
        indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        selected_similarity = np.take_along_axis(similarity, indices, axis=1)
        selected_similarity = selected_similarity / max(temperature, 1e-8)
        selected_similarity -= selected_similarity.max(axis=1, keepdims=True)
        weights = np.exp(selected_similarity)
        weights /= weights.sum(axis=1, keepdims=True)
        selected_latent = train_latent[indices]
        projected[start:end] = np.einsum("bk,bkd->bd", weights, selected_latent)
    return projected


class BPRMF(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, dim)
        self.item_embedding = torch.nn.Embedding(num_items, dim)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1)

    def embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight


class LightGCN(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, layers: int, adjacency):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.embedding = torch.nn.Embedding(num_users + num_items, dim)
        torch.nn.init.normal_(self.embedding.weight, std=0.1)
        self.register_buffer("adjacency", adjacency)

    def embeddings(self):
        values = [self.embedding.weight]
        current = self.embedding.weight
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            values.append(current)
        final = torch.stack(values, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]


def build_lightgcn_adjacency(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
):
    left = user_indices
    right = item_indices + num_users
    rows = np.concatenate([left, right])
    cols = np.concatenate([right, left])
    total_nodes = num_users + num_items
    degree = np.bincount(rows, minlength=total_nodes).astype(np.float32)
    values = 1.0 / np.sqrt(
        np.maximum(degree[rows], 1.0) * np.maximum(degree[cols], 1.0)
    )
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values_tensor,
        size=(total_nodes, total_nodes),
        device=device,
    ).coalesce()


def latent_score_matrix(
    user_latent: np.ndarray,
    item_latent: np.ndarray,
) -> np.ndarray:
    return user_latent.astype(np.float32) @ item_latent.astype(np.float32).T


def fit_inductive_collaborative(
    *,
    model_name: str,
    data: StrictData,
    val_ids: Sequence[int],
    val_pairs: np.ndarray,
    seed: int,
    latent_dim: int,
    knn_k: int,
    learning_rate: float,
    regularization: float,
    epochs: int,
    eval_every: int,
    patience: int,
    projection_temperature: float,
    lightgcn_layers: int = 1,
    device_name: str = "cuda",
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, float]]:
    set_seed(seed)
    device = torch.device(
        device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
    )

    train_mashup_ids = np.asarray(
        sorted(set(int(value) for value in data.train_pairs[:, 0])),
        dtype=np.int64,
    )
    global_to_local = {int(value): idx for idx, value in enumerate(train_mashup_ids)}
    local_users = np.asarray(
        [global_to_local[int(value)] for value in data.train_pairs[:, 0]],
        dtype=np.int64,
    )
    item_ids = data.train_pairs[:, 1].astype(np.int64)

    positive_sets: List[Set[int]] = [set() for _ in range(len(train_mashup_ids))]
    for user, item in zip(local_users, item_ids):
        positive_sets[int(user)].add(int(item))

    if model_name == "bpr_mf":
        model = BPRMF(len(train_mashup_ids), data.num_apis, latent_dim).to(device)
    elif model_name == "lightgcn":
        adjacency = build_lightgcn_adjacency(
            local_users,
            item_ids,
            len(train_mashup_ids),
            data.num_apis,
            device,
        )
        model = LightGCN(
            len(train_mashup_ids),
            data.num_apis,
            latent_dim,
            lightgcn_layers,
            adjacency,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    rng = np.random.default_rng(seed)
    users_tensor = torch.tensor(local_users, dtype=torch.long, device=device)
    positives_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)

    best_metric = (-float("inf"), -float("inf"), -float("inf"))
    best_state = None
    best_epoch = 0
    stale = 0
    best_val_metrics: Dict[str, float] = {}

    train_text = data.mashup_text_emb[train_mashup_ids]
    val_text = data.mashup_text_emb[np.asarray(val_ids)]

    for epoch in range(1, epochs + 1):
        model.train()
        negatives = sample_negatives(
            local_users,
            positive_sets,
            data.num_apis,
            rng,
        )
        negatives_tensor = torch.tensor(
            negatives,
            dtype=torch.long,
            device=device,
        )

        user_emb, item_emb = model.embeddings()
        u = user_emb[users_tensor]
        pos = item_emb[positives_tensor]
        neg = item_emb[negatives_tensor]

        positive_score = (u * pos).sum(dim=1)
        negative_score = (u * neg).sum(dim=1)
        ranking_loss = -F.logsigmoid(positive_score - negative_score).mean()
        reg_loss = (
            u.pow(2).sum(dim=1)
            + pos.pow(2).sum(dim=1)
            + neg.pow(2).sum(dim=1)
        ).mean()
        loss = ranking_loss + regularization * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every != 0 and epoch != epochs:
            continue

        model.eval()
        with torch.no_grad():
            train_latent_t, item_latent_t = model.embeddings()
            train_latent = train_latent_t.detach().cpu().numpy()
            item_latent = item_latent_t.detach().cpu().numpy()

        val_latent = project_unseen_users(
            val_text,
            train_text,
            train_latent,
            knn_k,
            projection_temperature,
        )
        val_scores = latent_score_matrix(val_latent, item_latent)
        _, val_metrics, _ = evaluate_score_matrix(
            model_name,
            val_scores,
            val_ids,
            val_pairs,
        )
        score_tuple = (
            val_metrics["NDCG@10"],
            val_metrics["MAP@10"],
            val_metrics["Recall@10"],
        )

        if score_tuple > best_metric:
            best_metric = score_tuple
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_val_metrics = val_metrics
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("No best state was selected")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_latent_t, item_latent_t = model.embeddings()
    train_latent = train_latent_t.detach().cpu().numpy()
    item_latent = item_latent_t.detach().cpu().numpy()

    metadata = {
        "model_name": model_name,
        "seed": seed,
        "latent_dim": latent_dim,
        "knn_k": knn_k,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "projection_temperature": projection_temperature,
        "lightgcn_layers": lightgcn_layers,
        "device": str(device),
    }
    return metadata, train_mashup_ids, train_latent, item_latent, best_val_metrics


def score_inductive_collaborative(
    data: StrictData,
    target_ids: Sequence[int],
    train_mashup_ids: np.ndarray,
    train_latent: np.ndarray,
    item_latent: np.ndarray,
    knn_k: int,
    projection_temperature: float,
) -> np.ndarray:
    target_latent = project_unseen_users(
        data.mashup_text_emb[np.asarray(target_ids)],
        data.mashup_text_emb[train_mashup_ids],
        train_latent,
        knn_k,
        projection_temperature,
    )
    return latent_score_matrix(target_latent, item_latent)


def deterministic_baseline_scores(
    method: str,
    data: StrictData,
    target_ids: Sequence[int],
) -> np.ndarray:
    if method == "Popularity":
        return popularity_scores(data, target_ids)
    if method == "TF-IDF":
        return tfidf_scores(data, target_ids)
    if method == "BM25":
        return bm25_scores(data, target_ids)
    if method == "BGE":
        return bge_scores(data, target_ids)
    if method == "Category-Jaccard":
        return category_jaccard_scores(data, target_ids)
    raise ValueError(f"Unknown deterministic baseline: {method}")


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    return "\n".join(lines)
