from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

Pair = Tuple[int, int]


# =========================
# Basic utils
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: Path, map_location="cpu"):
    """Compatible torch.load for PyTorch versions with weights_only default changes."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pairs(path: Path) -> Dict[str, np.ndarray]:
    obj = torch_load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path} should be a dict with keys pos/neg, got {type(obj)}")

    out: Dict[str, np.ndarray] = {}
    for key in ["pos", "neg"]:
        if key not in obj:
            raise KeyError(f"{path} missing key: {key}")
        arr = obj[key]
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{path}[{key}] should have shape [N, 2], got {arr.shape}")
        out[key] = arr
    return out


def pairs_to_user_items(pairs: np.ndarray) -> Dict[int, Set[int]]:
    d: Dict[int, Set[int]] = defaultdict(set)
    for m, a in pairs:
        d[int(m)].add(int(a))
    return dict(d)


def merge_user_items(*dicts: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for d in dicts:
        for m, items in d.items():
            out[int(m)].update(int(x) for x in items)
    return dict(out)


def normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def ensure_same_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate feature matrices so cosine/dot similarity is valid."""
    da = a.size(1)
    db = b.size(1)
    if da == db:
        return a, b
    d = max(da, db)
    if da < d:
        a = F.pad(a, (0, d - da))
    if db < d:
        b = F.pad(b, (0, d - db))
    return a, b


def get_node_x(graph_data, node_type: str) -> torch.Tensor:
    try:
        return graph_data[node_type].x.detach().cpu().float()
    except Exception as e:
        raise RuntimeError(f"Cannot read graph_data['{node_type}'].x") from e


# =========================
# Metrics and evaluation
# =========================

def _dcg(hit_list: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hit_list))


def _ap_at_k(hit_list: Sequence[int], num_relevant: int) -> float:
    if num_relevant <= 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for idx, rel in enumerate(hit_list):
        if rel:
            hit_count += 1
            precision_sum += hit_count / float(idx + 1)
    return precision_sum / float(min(num_relevant, len(hit_list)))


def evaluate_topk_from_scores(
    score_fn,
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    num_apis: int,
    ks: Sequence[int] = (5, 10),
    batch_size: int = 256,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    score_fn(mashup_ids_tensor) -> scores tensor [B, num_apis]
    Evaluates all-API ranking, then filters seen APIs by setting -inf.
    """
    max_k = max(ks)
    mashup_ids = sorted(test_pos_by_mashup.keys())

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})
    sums["MAP"] = 0.0

    valid_count = 0

    for start in range(0, len(mashup_ids), batch_size):
        batch_m = mashup_ids[start:start + batch_size]
        m_tensor = torch.tensor(batch_m, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = score_fn(m_tensor)
            if scores.device.type != device.split(":")[0]:
                scores = scores.to(device)
            scores = scores.clone()

        # filter seen APIs for each mashup
        for row, m in enumerate(batch_m):
            seen = filter_seen_by_mashup.get(int(m), set())
            if seen:
                seen_idx = torch.tensor(list(seen), dtype=torch.long, device=scores.device)
                seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < num_apis)]
                if seen_idx.numel() > 0:
                    scores[row, seen_idx] = -float("inf")

        topk = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu().numpy()

        for row, m in enumerate(batch_m):
            gt = test_pos_by_mashup.get(int(m), set())
            if not gt:
                continue
            valid_count += 1
            ranked = [int(x) for x in topk[row].tolist()]
            hit_full = [1 if a in gt else 0 for a in ranked]
            num_rel = len(gt)

            for k in ks:
                hit_k = hit_full[:k]
                hits = sum(hit_k)
                recall = hits / float(num_rel)
                hr = 1.0 if hits > 0 else 0.0
                dcg = _dcg(hit_k)
                ideal_hits = [1] * min(num_rel, k)
                idcg = _dcg(ideal_hits)
                ndcg = dcg / idcg if idcg > 0 else 0.0

                sums[f"Recall@{k}"] += recall
                sums[f"HitRate@{k}"] += hr
                sums[f"NDCG@{k}"] += ndcg

            sums["MAP"] += _ap_at_k(hit_full, num_rel)

    if valid_count == 0:
        raise RuntimeError("No valid test mashups to evaluate.")

    return {k: v / valid_count for k, v in sums.items()}


# =========================
# Baselines: non-parametric
# =========================

def run_popularity(
    train_pos: np.ndarray,
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    num_apis: int,
    ks: Sequence[int],
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    counts = torch.zeros(num_apis, dtype=torch.float32, device=device)
    if len(train_pos) > 0:
        api_ids = torch.tensor(train_pos[:, 1], dtype=torch.long, device=device)
        counts.scatter_add_(0, api_ids, torch.ones_like(api_ids, dtype=torch.float32))

    # A tiny deterministic tie-breaker avoids unstable topk order among zero-count items.
    counts = counts + torch.arange(num_apis, device=device, dtype=torch.float32) * 1e-12

    def score_fn(m_ids: torch.Tensor) -> torch.Tensor:
        return counts.unsqueeze(0).expand(m_ids.numel(), -1)

    return evaluate_topk_from_scores(
        score_fn=score_fn,
        test_pos_by_mashup=test_pos_by_mashup,
        filter_seen_by_mashup=filter_seen_by_mashup,
        num_apis=num_apis,
        ks=ks,
        batch_size=batch_size,
        device=device,
    )


def run_feature_retrieval(
    mashup_feat: torch.Tensor,
    api_feat: torch.Tensor,
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    num_apis: int,
    ks: Sequence[int],
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    mashup_feat, api_feat = ensure_same_dim(mashup_feat.float(), api_feat.float())
    mashup_feat = F.normalize(mashup_feat, dim=1).to(device)
    api_feat = F.normalize(api_feat, dim=1).to(device)

    def score_fn(m_ids: torch.Tensor) -> torch.Tensor:
        return mashup_feat[m_ids] @ api_feat.t()

    return evaluate_topk_from_scores(
        score_fn=score_fn,
        test_pos_by_mashup=test_pos_by_mashup,
        filter_seen_by_mashup=filter_seen_by_mashup,
        num_apis=num_apis,
        ks=ks,
        batch_size=batch_size,
        device=device,
    )


def load_text_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        values = json.load(f)
    if not isinstance(values, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return [str(value or "") for value in values]


def run_tfidf_text_retrieval(
    mashup_texts: Sequence[str],
    api_texts: Sequence[str],
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    num_apis: int,
    ks: Sequence[int],
    batch_size: int,
    device: str,
    max_features: int = 20000,
    ngram_max: int = 2,
    min_df: int = 1,
) -> Dict[str, float]:
    """
    Proper TF-IDF retrieval baseline.

    Mashup and API texts are fitted with ONE shared vocabulary.  The sparse
    TF-IDF matrices stay on CPU; each evaluation batch produces only a
    [batch_size, num_apis] dense score matrix.
    """
    if len(api_texts) != num_apis:
        raise ValueError(
            f"api_texts length {len(api_texts)} != num_apis {num_apis}"
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, max(1, int(ngram_max))),
        min_df=max(1, int(min_df)),
        max_features=max_features if max_features > 0 else None,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    all_texts = list(mashup_texts) + list(api_texts)
    all_matrix = vectorizer.fit_transform(all_texts)
    mashup_matrix = all_matrix[: len(mashup_texts)]
    api_matrix = all_matrix[len(mashup_texts) :].tocsr()
    api_matrix_t = api_matrix.transpose().tocsr()

    print(
        "[TF-IDF] shared vocabulary:",
        len(vectorizer.vocabulary_),
        "| ngram_range:",
        (1, max(1, int(ngram_max))),
        "| mashup matrix:",
        mashup_matrix.shape,
        "| api matrix:",
        api_matrix.shape,
    )

    def score_fn(m_ids: torch.Tensor) -> torch.Tensor:
        ids = m_ids.detach().cpu().numpy()
        scores = mashup_matrix[ids] @ api_matrix_t
        if hasattr(scores, "toarray"):
            scores = scores.toarray()
        scores = np.asarray(scores, dtype=np.float32)
        return torch.from_numpy(scores).to(device)

    return evaluate_topk_from_scores(
        score_fn=score_fn,
        test_pos_by_mashup=test_pos_by_mashup,
        filter_seen_by_mashup=filter_seen_by_mashup,
        num_apis=num_apis,
        ks=ks,
        batch_size=batch_size,
        device=device,
    )


# =========================
# BPR-MF
# =========================

class BPRMF(nn.Module):
    def __init__(self, num_mashups: int, num_apis: int, dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.user_emb = nn.Embedding(num_mashups, dim)
        self.item_emb = nn.Embedding(num_apis, dim)
        self.user_bias = nn.Embedding(num_mashups, 1)
        self.item_bias = nn.Embedding(num_apis, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.dropout(self.user_emb(users))
        i = self.dropout(self.item_emb(items))
        s = (u * i).sum(dim=-1)
        s = s + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
        return s

    def all_scores(self, users: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)
        scores = u @ self.item_emb.weight.t()
        scores = scores + self.user_bias(users) + self.item_bias.weight.t()
        return scores


def sample_negative_items(
    users_np: np.ndarray,
    num_apis: int,
    forbidden: Dict[int, Set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    neg = np.empty_like(users_np, dtype=np.int64)
    for idx, m in enumerate(users_np):
        m = int(m)
        bad = forbidden.get(m, set())
        # Usually positives per mashup are tiny, so rejection sampling is fine.
        while True:
            a = int(rng.integers(0, num_apis))
            if a not in bad:
                neg[idx] = a
                break
    return neg


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score).mean()


def run_bpr_mf(
    train_pos: np.ndarray,
    val_pos_by_mashup: Dict[int, Set[int]],
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    forbidden_by_mashup: Dict[int, Set[int]],
    num_mashups: int,
    num_apis: int,
    ks: Sequence[int],
    batch_size: int,
    eval_batch_size: int,
    device: str,
    seed: int,
    dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = BPRMF(num_mashups, num_apis, dim=dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_pos = np.asarray(train_pos, dtype=np.int64)
    best_val = -1e18
    best_state = None
    bad_epochs = 0

    def score_fn(m_ids: torch.Tensor) -> torch.Tensor:
        return model.all_scores(m_ids)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_pos))
        losses = []

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            batch = train_pos[idx]
            users_np = batch[:, 0]
            pos_np = batch[:, 1]
            neg_np = sample_negative_items(users_np, num_apis, forbidden_by_mashup, rng)

            users = torch.tensor(users_np, dtype=torch.long, device=device)
            pos = torch.tensor(pos_np, dtype=torch.long, device=device)
            neg = torch.tensor(neg_np, dtype=torch.long, device=device)

            opt.zero_grad()
            pos_score = model.score(users, pos)
            neg_score = model.score(users, neg)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        val_metrics = evaluate_topk_from_scores(
            score_fn=score_fn,
            test_pos_by_mashup=val_pos_by_mashup,
            filter_seen_by_mashup=filter_seen_by_mashup,
            num_apis=num_apis,
            ks=ks,
            batch_size=eval_batch_size,
            device=device,
        )
        val_key = val_metrics.get("NDCG@10", val_metrics.get(f"NDCG@{max(ks)}"))
        print(f"[BPR-MF] epoch={epoch:03d} loss={np.mean(losses):.4f} val_NDCG@10={val_key:.4f}")

        if val_key > best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[BPR-MF] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return evaluate_topk_from_scores(
        score_fn=score_fn,
        test_pos_by_mashup=test_pos_by_mashup,
        filter_seen_by_mashup=filter_seen_by_mashup,
        num_apis=num_apis,
        ks=ks,
        batch_size=eval_batch_size,
        device=device,
    )


# =========================
# LightGCN baseline
# =========================

class LightGCN(nn.Module):
    def __init__(self, num_mashups: int, num_apis: int, dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_mashups = num_mashups
        self.num_apis = num_apis
        self.num_layers = num_layers
        self.user_emb = nn.Embedding(num_mashups, dim)
        self.item_emb = nn.Embedding(num_apis, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        xs = [x0]
        x = x0
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            xs.append(x)
        out = torch.stack(xs, dim=0).mean(dim=0)
        users, items = out[:self.num_mashups], out[self.num_mashups:]
        return users, items

    def score(self, users: torch.Tensor, items: torch.Tensor, user_z: torch.Tensor, item_z: torch.Tensor) -> torch.Tensor:
        return (user_z[users] * item_z[items]).sum(dim=-1)


def build_lightgcn_adj(train_pos: np.ndarray, num_mashups: int, num_apis: int, device: str) -> torch.Tensor:
    total = num_mashups + num_apis
    users = train_pos[:, 0].astype(np.int64)
    items = train_pos[:, 1].astype(np.int64) + num_mashups

    row = np.concatenate([users, items])
    col = np.concatenate([items, users])
    idx = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)

    deg = np.zeros(total, dtype=np.float32)
    for r in row:
        deg[int(r)] += 1.0
    for c in col:
        # row already contains both directions, degree counted through row is enough;
        # no action needed here, kept for readability.
        pass
    deg = np.maximum(deg, 1.0)

    vals_np = 1.0 / np.sqrt(deg[row] * deg[col])
    vals = torch.tensor(vals_np, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, size=(total, total)).coalesce().to(device)
    return adj


def run_lightgcn(
    train_pos: np.ndarray,
    val_pos_by_mashup: Dict[int, Set[int]],
    test_pos_by_mashup: Dict[int, Set[int]],
    filter_seen_by_mashup: Dict[int, Set[int]],
    forbidden_by_mashup: Dict[int, Set[int]],
    num_mashups: int,
    num_apis: int,
    ks: Sequence[int],
    batch_size: int,
    eval_batch_size: int,
    device: str,
    seed: int,
    dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = LightGCN(num_mashups, num_apis, dim=dim, num_layers=num_layers).to(device)
    adj = build_lightgcn_adj(train_pos, num_mashups, num_apis, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1e18
    best_state = None
    bad_epochs = 0

    def make_score_fn():
        user_z, item_z = model.propagate(adj)
        def score_fn(m_ids: torch.Tensor) -> torch.Tensor:
            return user_z[m_ids] @ item_z.t()
        return score_fn

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_pos))
        losses = []
        user_z, item_z = model.propagate(adj)

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            batch = train_pos[idx]
            users_np = batch[:, 0]
            pos_np = batch[:, 1]
            neg_np = sample_negative_items(users_np, num_apis, forbidden_by_mashup, rng)

            users = torch.tensor(users_np, dtype=torch.long, device=device)
            pos = torch.tensor(pos_np, dtype=torch.long, device=device)
            neg = torch.tensor(neg_np, dtype=torch.long, device=device)

            opt.zero_grad()
            # Recompute graph embeddings per batch to keep computation graph valid after optimizer step.
            user_z, item_z = model.propagate(adj)
            pos_score = model.score(users, pos, user_z, item_z)
            neg_score = model.score(users, neg, user_z, item_z)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_topk_from_scores(
                score_fn=make_score_fn(),
                test_pos_by_mashup=val_pos_by_mashup,
                filter_seen_by_mashup=filter_seen_by_mashup,
                num_apis=num_apis,
                ks=ks,
                batch_size=eval_batch_size,
                device=device,
            )
        val_key = val_metrics.get("NDCG@10", val_metrics.get(f"NDCG@{max(ks)}"))
        print(f"[LightGCN] epoch={epoch:03d} loss={np.mean(losses):.4f} val_NDCG@10={val_key:.4f}")

        if val_key > best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[LightGCN] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return evaluate_topk_from_scores(
            score_fn=make_score_fn(),
            test_pos_by_mashup=test_pos_by_mashup,
            filter_seen_by_mashup=filter_seen_by_mashup,
            num_apis=num_apis,
            ks=ks,
            batch_size=eval_batch_size,
            device=device,
        )


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run API recommendation baselines on processed dataset.")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument(
        "--methods",
        type=str,
        default="popularity,tfidf,bge,bpr_mf,lightgcn",
        help="Comma separated: popularity,tfidf,bge,bpr_mf,lightgcn,all",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ks", type=str, default="5,10")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--filter_seen",
        type=str,
        default="train",
        choices=["train", "train_val", "none"],
        help="APIs filtered during evaluation. Use train to match most warm-start protocols.",
    )

    # Trainable baseline hyperparameters
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lightgcn_layers", type=int, default=2)

    # Standalone TF-IDF retrieval settings.  These are intentionally separate
    # from graph node feature dimensionality.
    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--tfidf_min_df", type=int, default=1)

    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def format_metrics(metrics: Dict[str, float]) -> str:
    order = ["Recall@5", "NDCG@5", "HitRate@5", "Recall@10", "NDCG@10", "HitRate@10", "MAP"]
    return " | ".join(f"{k}: {metrics[k]:.4f}" for k in order if k in metrics)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        args.device = "cpu"

    processed_dir = Path(args.processed_dir)
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "all" in methods:
        methods = ["popularity", "tfidf", "bge", "bpr_mf", "lightgcn"]

    print("=" * 80)
    print("Processed dir:", processed_dir)
    print("Methods:", methods)
    print("Device:", args.device)
    print("Seed:", args.seed)
    print("Ks:", ks)
    print("filter_seen:", args.filter_seen)
    print("=" * 80)

    train_pairs = load_pairs(processed_dir / "train_pairs.pt")
    val_pairs = load_pairs(processed_dir / "val_pairs.pt")
    test_pairs = load_pairs(processed_dir / "test_pairs.pt")
    train_pos = train_pairs["pos"]
    val_pos = val_pairs["pos"]
    test_pos = test_pairs["pos"]

    with open(processed_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    num_mashups = int(metadata.get("num_mashups"))
    num_apis = int(metadata.get("num_apis"))

    train_by_m = pairs_to_user_items(train_pos)
    val_by_m = pairs_to_user_items(val_pos)
    test_by_m = pairs_to_user_items(test_pos)

    if args.filter_seen == "train":
        filter_seen = train_by_m
    elif args.filter_seen == "train_val":
        filter_seen = merge_user_items(train_by_m, val_by_m)
    else:
        filter_seen = {}

    # For negative sampling during BPR/LightGCN training, avoid sampling any true API if metadata has full truth.
    raw_all_true = metadata.get("mashup_api_pos", None)
    if isinstance(raw_all_true, dict):
        forbidden_by_m: Dict[int, Set[int]] = {
            int(m): set(int(a) for a in apis) for m, apis in raw_all_true.items()
        }
    else:
        forbidden_by_m = merge_user_items(train_by_m, val_by_m, test_by_m)

    graph_data = None
    results: Dict[str, Dict[str, float]] = {}

    if "popularity" in methods:
        print("\n[Run] Popularity")
        metrics = run_popularity(
            train_pos=train_pos,
            test_pos_by_mashup=test_by_m,
            filter_seen_by_mashup=filter_seen,
            num_apis=num_apis,
            ks=ks,
            batch_size=args.eval_batch_size,
            device=args.device,
        )
        results["Popularity"] = metrics
        print("Popularity Test:", format_metrics(metrics))

    if "tfidf" in methods:
        print("\n[Run] TF-IDF retrieval from raw texts with a shared vocabulary")
        mashup_text_path = processed_dir / "mashup_texts.json"
        api_text_path = processed_dir / "api_texts.json"

        if mashup_text_path.exists() and api_text_path.exists():
            mashup_texts = load_text_list(mashup_text_path)
            api_texts = load_text_list(api_text_path)
            metrics = run_tfidf_text_retrieval(
                mashup_texts=mashup_texts,
                api_texts=api_texts,
                test_pos_by_mashup=test_by_m,
                filter_seen_by_mashup=filter_seen,
                num_apis=num_apis,
                ks=ks,
                batch_size=args.eval_batch_size,
                device=args.device,
                max_features=args.tfidf_max_features,
                ngram_max=args.tfidf_ngram_max,
                min_df=args.tfidf_min_df,
            )
        else:
            print(
                "[WARN] mashup_texts.json/api_texts.json not found; "
                "falling back to graph_data node x."
            )
            graph_data = graph_data or torch_load(
                processed_dir / "graph_data.pt", map_location="cpu"
            )
            mashup_x = get_node_x(graph_data, "mashup")
            api_x = get_node_x(graph_data, "api")
            metrics = run_feature_retrieval(
                mashup_feat=mashup_x,
                api_feat=api_x,
                test_pos_by_mashup=test_by_m,
                filter_seen_by_mashup=filter_seen,
                num_apis=num_apis,
                ks=ks,
                batch_size=args.eval_batch_size,
                device=args.device,
            )

        results["TF-IDF Retrieval"] = metrics
        print("TF-IDF Retrieval Test:", format_metrics(metrics))

    if "bge" in methods:
        print("\n[Run] BGE/text embedding retrieval")
        m_path = processed_dir / "mashup_text_emb.npy"
        a_path = processed_dir / "api_text_emb.npy"
        if not m_path.exists() or not a_path.exists():
            raise FileNotFoundError("Missing mashup_text_emb.npy or api_text_emb.npy")
        mashup_emb = torch.tensor(np.load(m_path), dtype=torch.float32)
        api_emb = torch.tensor(np.load(a_path), dtype=torch.float32)
        metrics = run_feature_retrieval(
            mashup_feat=mashup_emb,
            api_feat=api_emb,
            test_pos_by_mashup=test_by_m,
            filter_seen_by_mashup=filter_seen,
            num_apis=num_apis,
            ks=ks,
            batch_size=args.eval_batch_size,
            device=args.device,
        )
        results["BGE Retrieval"] = metrics
        print("BGE Retrieval Test:", format_metrics(metrics))

    if "bpr_mf" in methods:
        print("\n[Run] BPR-MF")
        metrics = run_bpr_mf(
            train_pos=train_pos,
            val_pos_by_mashup=val_by_m,
            test_pos_by_mashup=test_by_m,
            filter_seen_by_mashup=filter_seen,
            forbidden_by_mashup=forbidden_by_m,
            num_mashups=num_mashups,
            num_apis=num_apis,
            ks=ks,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed,
            dim=args.dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
        )
        results["BPR-MF"] = metrics
        print("BPR-MF Test:", format_metrics(metrics))

    if "lightgcn" in methods:
        print("\n[Run] LightGCN")
        metrics = run_lightgcn(
            train_pos=train_pos,
            val_pos_by_mashup=val_by_m,
            test_pos_by_mashup=test_by_m,
            filter_seen_by_mashup=filter_seen,
            forbidden_by_mashup=forbidden_by_m,
            num_mashups=num_mashups,
            num_apis=num_apis,
            ks=ks,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed,
            dim=args.dim,
            num_layers=args.lightgcn_layers,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        results["LightGCN"] = metrics
        print("LightGCN Test:", format_metrics(metrics))

    print("\n" + "=" * 80)
    print("Final results")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"{name}: {format_metrics(metrics)}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        metric_names = sorted({k for m in results.values() for k in m.keys()})
        preferred = ["Recall@5", "NDCG@5", "HitRate@5", "Recall@10", "NDCG@10", "HitRate@10", "MAP"]
        metric_names = [m for m in preferred if m in metric_names] + [m for m in metric_names if m not in preferred]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["method"] + metric_names)
            for name, metrics in results.items():
                writer.writerow([name] + [f"{metrics.get(m, 0.0):.6f}" for m in metric_names])
        print(f"Saved CSV to: {out}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {out}")


if __name__ == "__main__":
    main()
