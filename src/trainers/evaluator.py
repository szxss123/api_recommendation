
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np
import torch


def recall_at_k(y_true: Sequence[int], y_pred_ranked: Sequence[int], k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    topk = set(y_pred_ranked[:k])
    hits = len(set(y_true) & topk)
    return hits / float(len(set(y_true)))


def hitrate_at_k(y_true: Sequence[int], y_pred_ranked: Sequence[int], k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    topk = set(y_pred_ranked[:k])
    return 1.0 if len(set(y_true) & topk) > 0 else 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    relevances = np.asarray(relevances[:k], dtype=np.float32)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(y_true: Sequence[int], y_pred_ranked: Sequence[int], k: int) -> float:
    true_set = set(y_true)
    if len(true_set) == 0:
        return 0.0
    rel = [1 if a in true_set else 0 for a in y_pred_ranked[:k]]
    dcg = dcg_at_k(rel, k)
    ideal_rel = [1] * min(len(true_set), k)
    idcg = dcg_at_k(ideal_rel, k)
    return 0.0 if idcg == 0 else dcg / idcg


def average_precision(y_true: Sequence[int], y_pred_ranked: Sequence[int], k: int | None = None) -> float:
    true_set = set(y_true)
    if len(true_set) == 0:
        return 0.0
    ranked = y_pred_ranked if k is None else y_pred_ranked[:k]
    hits = 0
    ap_sum = 0.0
    for i, api_id in enumerate(ranked, start=1):
        if api_id in true_set:
            hits += 1
            ap_sum += hits / i
    denom = min(len(true_set), len(ranked))
    return 0.0 if denom == 0 else ap_sum / denom


def mean_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if len(metric_list) == 0:
        return {}
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


def build_ground_truth_dict(pos_pairs: np.ndarray) -> Dict[int, List[int]]:
    gt: Dict[int, List[int]] = {}
    for m_id, a_id in pos_pairs.tolist():
        gt.setdefault(int(m_id), []).append(int(a_id))
    return gt


def score_all_apis(
    mashup_emb: torch.Tensor,
    api_emb: torch.Tensor,
    scorer,
    batch_size: int = 1024,
) -> torch.Tensor:
    """
    Compute scores for all mashup-api pairs.
    scorer can be:
    - a callable: scorer(m_batch, a_batch) -> scores
    - None: defaults to dot product
    """
    device = mashup_emb.device
    num_mashup = mashup_emb.size(0)
    num_api = api_emb.size(0)

    if scorer is None:
        return mashup_emb @ api_emb.t()

    scores = []
    for start in range(0, num_mashup, batch_size):
        end = min(start + batch_size, num_mashup)
        m_batch = mashup_emb[start:end]
        batch_scores = []
        for a_start in range(0, num_api, batch_size):
            a_end = min(a_start + batch_size, num_api)
            a_batch = api_emb[a_start:a_end]

            # broadcast pair scoring
            m_expand = m_batch.unsqueeze(1).expand(-1, a_batch.size(0), -1)
            a_expand = a_batch.unsqueeze(0).expand(m_batch.size(0), -1, -1)
            s = scorer(m_expand, a_expand)
            if s.dim() == 3 and s.size(-1) == 1:
                s = s.squeeze(-1)
            batch_scores.append(s)
        scores.append(torch.cat(batch_scores, dim=1))
    return torch.cat(scores, dim=0)


def rank_apis_for_mashup(
    api_scores: np.ndarray,
    exclude_api_ids: Iterable[int] | None = None,
) -> List[int]:
    scores = api_scores.copy()
    if exclude_api_ids is not None:
        exclude_ids = list(exclude_api_ids)
        if exclude_ids:
            scores[exclude_ids] = -np.inf
    ranked = np.argsort(-scores)
    return ranked.tolist()


def evaluate_recommendation(
    mashup_emb: torch.Tensor,
    api_emb: torch.Tensor,
    test_pos_pairs: np.ndarray,
    train_pos_pairs: np.ndarray | None = None,
    scorer=None,
    ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """
    Evaluate recommendation on positive pairs.
    Known train APIs for each mashup are excluded during ranking.
    """
    gt_test = build_ground_truth_dict(test_pos_pairs)
    gt_train = build_ground_truth_dict(train_pos_pairs) if train_pos_pairs is not None else {}

    score_mat = score_all_apis(mashup_emb, api_emb, scorer=scorer).detach().cpu().numpy()

    metric_rows: List[Dict[str, float]] = []
    for m_id, true_api_ids in gt_test.items():
        exclude_ids = gt_train.get(m_id, [])
        ranked = rank_apis_for_mashup(score_mat[m_id], exclude_api_ids=exclude_ids)

        row: Dict[str, float] = {}
        for k in ks:
            row[f"Recall@{k}"] = recall_at_k(true_api_ids, ranked, k)
            row[f"NDCG@{k}"] = ndcg_at_k(true_api_ids, ranked, k)
            row[f"HitRate@{k}"] = hitrate_at_k(true_api_ids, ranked, k)
        row["MAP"] = average_precision(true_api_ids, ranked, k=None)
        metric_rows.append(row)

    return mean_metrics(metric_rows)


def format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "No metrics."
    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
