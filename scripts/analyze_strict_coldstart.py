#!/usr/bin/env python3
"""
Diagnose whether the strict cold-start model is genuinely different from the
Popularity baseline and whether gains come from Head/Middle/Tail APIs.

Outputs
-------
overall_metrics.csv
overlap_summary.csv
per_mashup_overlap.csv
per_mashup_metrics.csv
rankings_topk.csv
group_metrics.csv
recommendation_composition.csv
api_groups.csv
summary.json

The script uses the same strict-inductive trainer and test graph as training.
It supports the current dot scorer and contains a generic pairwise fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.trainers.finetune_trainer import _as_numpy_pairs, set_seed
from src.trainers.finetune_trainer_strict_inductive import (
    StrictInductiveRecommendationFinetuner,
    load_strict_processed_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlap and long-tail analysis for strict cold-start."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Overrides checkpoint_path in the YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/analysis/mtfm_cold_strict_seed0"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--api_chunk_size", type=int, default=2048)
    parser.add_argument("--head_ratio", type=float, default=0.20)
    parser.add_argument("--middle_ratio", type=float, default=0.30)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10],
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Use validation for parameter selection; use test only once.",
    )
    parser.add_argument(
        "--score_normalize",
        type=lambda value: str(value).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        default=None,
    )
    parser.add_argument("--text_score_weight", type=float, default=None)
    parser.add_argument("--popularity_penalty", type=float, default=None)
    parser.add_argument(
        "--ranking_mode",
        choices=("raw", "bge_only", "zscore"),
        default=None,
    )
    parser.add_argument("--fusion_lambda", type=float, default=None)
    parser.add_argument("--score_eps", type=float, default=None)
    return parser.parse_args()


def cfg(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return data


def instantiate_trainer(config: Dict[str, Any], device_override: Optional[str]):
    processed_dir = str(cfg(config, "processed_dir", ""))
    if not processed_dir:
        raise ValueError("processed_dir is missing from config.")

    (
        graph_train,
        graph_val,
        graph_test,
        train_pairs,
        val_pairs,
        test_pairs,
    ) = load_strict_processed_data(processed_dir)

    device = device_override or str(cfg(config, "device", "cuda"))

    trainer = StrictInductiveRecommendationFinetuner(
        graph_train=graph_train,
        graph_val=graph_val,
        graph_test=graph_test,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        processed_dir=processed_dir,
        hidden_dim=int(cfg(config, "hidden_dim", 128)),
        num_layers=int(cfg(config, "num_layers", 2)),
        scorer_name=str(cfg(config, "scorer", "dot")),
        scorer_dropout=float(cfg(config, "scorer_dropout", 0.1)),
        encoder_dropout=float(cfg(config, "encoder_dropout", 0.2)),
        lr=float(cfg(config, "lr", 5e-4)),
        weight_decay=float(cfg(config, "weight_decay", 5e-4)),
        batch_size=int(cfg(config, "batch_size", 32)),
        epochs=int(cfg(config, "epochs", 50)),
        device=device,
        checkpoint_path=str(cfg(config, "checkpoint_path", "outputs/checkpoint.pt")),
        use_text=bool(cfg(config, "use_text", True)),
        fusion_mode=str(cfg(config, "fusion_mode", "mlp")),
        text_dropout=float(cfg(config, "text_dropout", 0.2)),
        text_weight=float(cfg(config, "text_weight", 0.5)),
        align_weight=float(cfg(config, "align_weight", 0.05)),
        align_loss_type=str(cfg(config, "align_loss_type", "mse")),
        contrastive_temperature=float(
            cfg(config, "contrastive_temperature", 0.2)
        ),
        align_sample_size=int(cfg(config, "align_sample_size", 1024)),
        use_view_contrast=bool(cfg(config, "use_view_contrast", False)),
        view_contrast_weight=float(
            cfg(config, "view_contrast_weight", 0.0)
        ),
        view_contrast_temperature=float(
            cfg(config, "view_contrast_temperature", 0.2)
        ),
        view_contrast_sample_size=int(
            cfg(config, "view_contrast_sample_size", 1024)
        ),
        view_contrast_target=str(
            cfg(config, "view_contrast_target", "api")
        ),
        use_cooccur_edge_weight=bool(
            cfg(config, "use_cooccur_edge_weight", True)
        ),
        score_normalize=bool(cfg(config, "score_normalize", False)),
        text_score_weight=float(cfg(config, "text_score_weight", 0.0)),
        popularity_penalty=float(cfg(config, "popularity_penalty", 0.0)),
        ranking_mode=str(cfg(config, "ranking_mode", "raw")),
        fusion_lambda=float(cfg(config, "fusion_lambda", 0.5)),
        score_eps=float(cfg(config, "score_eps", 1e-8)),
    )

    return trainer, train_pairs, val_pairs, test_pairs


def _try_trainer_loader(trainer, checkpoint_path: Path) -> bool:
    """
    Try project-specific checkpoint methods first.
    """
    for method_name in ("load_checkpoint", "_load_checkpoint"):
        method = getattr(trainer, method_name, None)
        if method is None or not callable(method):
            continue
        try:
            signature = inspect.signature(method)
            if len(signature.parameters) == 0:
                method()
            else:
                method(str(checkpoint_path))
            print(f"[Checkpoint] loaded through trainer.{method_name}()")
            return True
        except Exception as exc:
            print(
                f"[Checkpoint] trainer.{method_name}() was not usable: "
                f"{type(exc).__name__}: {exc}"
            )
    return False


def _load_named_state(
    module: Optional[torch.nn.Module],
    checkpoint: Mapping[str, Any],
    direct_keys: Sequence[str],
    prefixes: Sequence[str],
) -> bool:
    if module is None:
        return True

    for key in direct_keys:
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            module.load_state_dict(state, strict=True)
            print(f"[Checkpoint] loaded {module.__class__.__name__} from '{key}'")
            return True

    combined_candidates: List[Mapping[str, Any]] = []
    for key in ("model_state_dict", "state_dict", "model"):
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            combined_candidates.append(state)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        combined_candidates.append(checkpoint)

    for state in combined_candidates:
        for prefix in prefixes:
            prefix_dot = prefix + "."
            extracted = {
                key[len(prefix_dot):]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix_dot)
            }
            if extracted:
                module.load_state_dict(extracted, strict=True)
                print(
                    f"[Checkpoint] loaded {module.__class__.__name__} "
                    f"from prefix '{prefix_dot}'"
                )
                return True

    # Dot scorers often have an empty state dict.
    if len(module.state_dict()) == 0:
        print(
            f"[Checkpoint] {module.__class__.__name__} has no trainable state; "
            "nothing to load."
        )
        return True

    return False


def load_checkpoint_flexible(trainer, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if _try_trainer_loader(trainer, checkpoint_path):
        return

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=trainer.device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    encoder_ok = _load_named_state(
        trainer.encoder,
        checkpoint,
        direct_keys=("encoder", "encoder_state_dict"),
        prefixes=("encoder",),
    )
    scorer_ok = _load_named_state(
        trainer.scorer,
        checkpoint,
        direct_keys=("scorer", "scorer_state_dict"),
        prefixes=("scorer",),
    )
    fusion_ok = _load_named_state(
        getattr(trainer, "text_fusion", None),
        checkpoint,
        direct_keys=("text_fusion", "text_fusion_state_dict"),
        prefixes=("text_fusion", "fusion"),
    )

    if not (encoder_ok and scorer_ok and fusion_ok):
        print("[Checkpoint] top-level keys:", list(checkpoint.keys())[:30])
        raise RuntimeError(
            "Could not map checkpoint states to encoder/scorer/text_fusion. "
            "Please send the printed checkpoint keys."
        )


def pairs_by_mashup(pairs: np.ndarray) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = defaultdict(set)
    for mashup_id, api_id in pairs:
        result[int(mashup_id)].add(int(api_id))
    return dict(result)


def dcg(hit_list: Sequence[int]) -> float:
    return sum(
        float(hit) / math.log2(index + 2)
        for index, hit in enumerate(hit_list)
    )


def metrics_for_rankings(
    rankings: Mapping[int, Sequence[int]],
    positives: Mapping[int, Set[int]],
    ks: Sequence[int],
    eligible_mashups: Optional[Set[int]] = None,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    for k in ks:
        sums[f"Recall@{k}"] = 0.0
        sums[f"NDCG@{k}"] = 0.0
        sums[f"HitRate@{k}"] = 0.0
        sums[f"MAP@{k}"] = 0.0

    count = 0
    for mashup_id, true_set in positives.items():
        if eligible_mashups is not None and mashup_id not in eligible_mashups:
            continue
        if not true_set:
            continue

        ranked = [int(value) for value in rankings[mashup_id]]
        count += 1

        for k in ks:
            topk = ranked[:k]
            hits = [1 if api_id in true_set else 0 for api_id in topk]
            hit_count = sum(hits)

            sums[f"Recall@{k}"] += hit_count / float(len(true_set))
            sums[f"HitRate@{k}"] += 1.0 if hit_count > 0 else 0.0

            ideal = [1] * min(len(true_set), k)
            idcg = dcg(ideal)
            sums[f"NDCG@{k}"] += dcg(hits) / idcg if idcg > 0 else 0.0

            precision_sum = 0.0
            cumulative_hits = 0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    cumulative_hits += 1
                    precision_sum += cumulative_hits / float(rank)
            # Match src/trainers/evaluator.py: AP@K denominator is min(|R|, K).
            sums[f"MAP@{k}"] += precision_sum / float(min(len(true_set), k))

    if count == 0:
        return {"eligible_mashups": 0, **{key: 0.0 for key in sums}}

    return {
        "eligible_mashups": count,
        **{key: value / count for key, value in sums.items()},
    }



def per_mashup_metric_rows(
    rankings_by_method: Mapping[str, Mapping[int, Sequence[int]]],
    positives: Mapping[int, Set[int]],
    ks: Sequence[int],
) -> List[Dict[str, Any]]:
    """
    Return one row per method and Mashup.

    These values use exactly the same Recall/NDCG/HitRate/AP@K definitions as
    metrics_for_rankings(), so averaging each metric column reproduces the
    corresponding overall metric.
    """
    rows: List[Dict[str, Any]] = []

    for method, rankings in rankings_by_method.items():
        for mashup_id in sorted(positives):
            true_set = positives[mashup_id]
            if not true_set:
                continue

            ranked = [int(value) for value in rankings[mashup_id]]
            row: Dict[str, Any] = {
                "method": method,
                "mashup_id": int(mashup_id),
                "num_positives": int(len(true_set)),
            }

            for k in ks:
                topk = ranked[:k]
                hits = [1 if api_id in true_set else 0 for api_id in topk]
                hit_count = int(sum(hits))

                row[f"Recall@{k}"] = hit_count / float(len(true_set))
                row[f"HitRate@{k}"] = 1.0 if hit_count > 0 else 0.0

                ideal = [1] * min(len(true_set), k)
                idcg = dcg(ideal)
                row[f"NDCG@{k}"] = (
                    dcg(hits) / idcg if idcg > 0 else 0.0
                )

                precision_sum = 0.0
                cumulative_hits = 0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        cumulative_hits += 1
                        precision_sum += cumulative_hits / float(rank)

                row[f"MAP@{k}"] = (
                    precision_sum / float(min(len(true_set), k))
                )

            rows.append(row)

    return rows


def ranking_rows(
    rankings_by_method: Mapping[str, Mapping[int, Sequence[int]]],
    positives: Mapping[int, Set[int]],
    group_map: Mapping[int, str],
    max_k: int,
) -> List[Dict[str, Any]]:
    """
    Save the actual Top-K lists for later diversity and case-study analysis.
    """
    rows: List[Dict[str, Any]] = []
    for method, rankings in rankings_by_method.items():
        for mashup_id in sorted(positives):
            true_set = positives[mashup_id]
            for rank, api_id in enumerate(
                rankings[mashup_id][:max_k],
                start=1,
            ):
                api_id = int(api_id)
                rows.append(
                    {
                        "method": method,
                        "mashup_id": int(mashup_id),
                        "rank": int(rank),
                        "api_id": api_id,
                        "is_positive": int(api_id in true_set),
                        "api_group": group_map[api_id],
                    }
                )
    return rows


def build_api_groups(
    train_pairs: np.ndarray,
    num_apis: int,
    head_ratio: float,
    middle_ratio: float,
) -> Tuple[Dict[int, str], Counter[int]]:
    if not (0.0 < head_ratio < 1.0):
        raise ValueError("head_ratio must be in (0,1)")
    if not (0.0 <= middle_ratio < 1.0):
        raise ValueError("middle_ratio must be in [0,1)")
    if head_ratio + middle_ratio >= 1.0:
        raise ValueError("head_ratio + middle_ratio must be < 1")

    frequency: Counter[int] = Counter(
        int(api_id) for _, api_id in train_pairs
    )
    active = [
        api_id for api_id in range(num_apis) if frequency.get(api_id, 0) > 0
    ]
    active.sort(key=lambda api_id: (-frequency[api_id], api_id))

    head_count = max(1, int(math.ceil(len(active) * head_ratio)))
    middle_count = int(math.ceil(len(active) * middle_ratio))
    middle_end = min(len(active), head_count + middle_count)

    group_map: Dict[int, str] = {}
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


def popularity_rankings(
    mashup_ids: Sequence[int],
    frequency: Counter[int],
    num_apis: int,
    known_by_mashup: Mapping[int, Set[int]],
    max_k: int,
) -> Dict[int, List[int]]:
    global_rank = sorted(
        range(num_apis),
        key=lambda api_id: (-frequency.get(api_id, 0), api_id),
    )
    result: Dict[int, List[int]] = {}
    for mashup_id in mashup_ids:
        seen = known_by_mashup.get(int(mashup_id), set())
        result[int(mashup_id)] = [
            api_id for api_id in global_rank if api_id not in seen
        ][:max_k]
    return result


def score_pairwise(
    scorer,
    mashup_batch: torch.Tensor,
    api_embeddings: torch.Tensor,
    api_chunk_size: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for api_start in range(0, api_embeddings.size(0), api_chunk_size):
        api_chunk = api_embeddings[
            api_start: api_start + api_chunk_size
        ]
        batch_size = mashup_batch.size(0)
        chunk_size = api_chunk.size(0)

        m = mashup_batch[:, None, :].expand(
            batch_size, chunk_size, mashup_batch.size(1)
        ).reshape(-1, mashup_batch.size(1))
        a = api_chunk[None, :, :].expand(
            batch_size, chunk_size, api_chunk.size(1)
        ).reshape(-1, api_chunk.size(1))

        values = scorer(m, a)
        if values.ndim > 1:
            values = values.reshape(-1)
        rows.append(values.reshape(batch_size, chunk_size))
    return torch.cat(rows, dim=1)


def model_rankings(
    trainer,
    graph,
    edge_weights,
    target_mashup_ids: Sequence[int],
    known_by_mashup: Mapping[int, Set[int]],
    max_k: int,
    batch_size: int,
    api_chunk_size: int,
    scorer_name: str,
) -> Dict[int, List[int]]:
    """
    Rank with the trainer's unified score-matrix implementation.

    api_chunk_size and scorer_name are retained in the signature for backward
    compatibility with existing commands; standardized fusion currently uses
    the dot scorer and evaluates all APIs at once per Mashup batch.
    """
    del api_chunk_size, scorer_name

    trainer.encoder.eval()
    trainer.scorer.eval()
    if trainer.text_fusion is not None:
        trainer.text_fusion.eval()

    with torch.no_grad():
        mashup_embeddings, api_embeddings, _, _ = (
            trainer._compute_embeddings_on_graph(
                graph,
                edge_weights,
                compute_aux=False,
                allow_training_aux=False,
            )
        )

    rankings: Dict[int, List[int]] = {}
    device = trainer.device

    for start in range(0, len(target_mashup_ids), batch_size):
        ids = [
            int(x)
            for x in target_mashup_ids[start:start + batch_size]
        ]
        idx = torch.tensor(ids, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = trainer.compute_ranking_scores(
                mashup_embeddings,
                api_embeddings,
                idx,
            ).clone()

            for row, mashup_id in enumerate(ids):
                seen = known_by_mashup.get(mashup_id, set())
                if seen:
                    seen_idx = torch.tensor(
                        sorted(seen),
                        dtype=torch.long,
                        device=scores.device,
                    )
                    scores[row, seen_idx] = -float("inf")

            topk = torch.topk(
                scores,
                k=max_k,
                dim=1,
            ).indices.detach().cpu().numpy()

        for row, mashup_id in enumerate(ids):
            rankings[mashup_id] = [
                int(api_id)
                for api_id in topk[row].tolist()
            ]

        print(
            f"[Ranking] "
            f"{min(start + batch_size, len(target_mashup_ids))}"
            f"/{len(target_mashup_ids)}"
        )

    return rankings


def overlap_analysis(
    ours: Mapping[int, Sequence[int]],
    popularity: Mapping[int, Sequence[int]],
    positives: Mapping[int, Set[int]],
    ks: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    for mashup_id in sorted(positives):
        row: Dict[str, Any] = {
            "mashup_id": mashup_id,
            "num_test_positives": len(positives[mashup_id]),
        }
        for k in ks:
            ours_list = list(ours[mashup_id][:k])
            pop_list = list(popularity[mashup_id][:k])
            ours_set = set(ours_list)
            pop_set = set(pop_list)
            intersection = ours_set & pop_set
            union = ours_set | pop_set

            row[f"overlap_count@{k}"] = len(intersection)
            row[f"overlap_ratio@{k}"] = len(intersection) / float(k)
            row[f"jaccard@{k}"] = (
                len(intersection) / float(len(union)) if union else 1.0
            )
            row[f"exact_set_match@{k}"] = int(ours_set == pop_set)
            row[f"exact_order_match@{k}"] = int(ours_list == pop_list)
            row[f"top1_match@{k}"] = int(
                bool(ours_list)
                and bool(pop_list)
                and ours_list[0] == pop_list[0]
            )

        rows.append(row)

    per_mashup = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for k in ks:
        summary_rows.append(
            {
                "K": k,
                "avg_overlap_count": per_mashup[
                    f"overlap_count@{k}"
                ].mean(),
                "avg_overlap_ratio": per_mashup[
                    f"overlap_ratio@{k}"
                ].mean(),
                "avg_jaccard": per_mashup[f"jaccard@{k}"].mean(),
                "exact_set_match_ratio": per_mashup[
                    f"exact_set_match@{k}"
                ].mean(),
                "exact_order_match_ratio": per_mashup[
                    f"exact_order_match@{k}"
                ].mean(),
                "top1_match_ratio": per_mashup[
                    f"top1_match@{k}"
                ].mean(),
            }
        )
    return per_mashup, pd.DataFrame(summary_rows)


def recommendation_composition(
    rankings: Mapping[int, Sequence[int]],
    group_map: Mapping[int, str],
    ks: Sequence[int],
    method: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mashup_ids = sorted(rankings)
    group_names = ["Head", "Middle", "Tail", "Unseen"]

    for k in ks:
        counts = Counter()
        unique = set()
        list_signatures = set()
        for mashup_id in mashup_ids:
            topk = tuple(int(x) for x in rankings[mashup_id][:k])
            list_signatures.add(topk)
            unique.update(topk)
            counts.update(group_map[api_id] for api_id in topk)

        total_slots = len(mashup_ids) * k
        for group_name in group_names:
            rows.append(
                {
                    "method": method,
                    "K": k,
                    "group": group_name,
                    "slot_count": counts[group_name],
                    "slot_ratio": (
                        counts[group_name] / float(total_slots)
                        if total_slots
                        else 0.0
                    ),
                    "unique_recommended_apis": len(unique),
                    "unique_list_count": len(list_signatures),
                }
            )
    return rows


def group_metric_rows(
    rankings_by_method: Mapping[str, Mapping[int, Sequence[int]]],
    test_positives: Mapping[int, Set[int]],
    group_map: Mapping[int, str],
    ks: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for group_name in ("Head", "Middle", "Tail", "Unseen"):
        group_positives: Dict[int, Set[int]] = {}
        total_positive_count = 0

        for mashup_id, true_set in test_positives.items():
            selected = {
                api_id
                for api_id in true_set
                if group_map[api_id] == group_name
            }
            if selected:
                group_positives[mashup_id] = selected
                total_positive_count += len(selected)

        eligible = set(group_positives)

        for method, rankings in rankings_by_method.items():
            metrics = metrics_for_rankings(
                rankings,
                group_positives,
                ks,
                eligible_mashups=eligible,
            )
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "eligible_mashups": metrics.pop("eligible_mashups"),
                    "test_positive_count": total_positive_count,
                    **metrics,
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(str(config["checkpoint_path"])).resolve()
    )
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_dir}")

    trainer, train_pairs_obj, val_pairs_obj, test_pairs_obj = (
        instantiate_trainer(config, args.device)
    )
    load_checkpoint_flexible(trainer, checkpoint_path)

    if args.score_normalize is not None:
        trainer.score_normalize = args.score_normalize
    if args.text_score_weight is not None:
        trainer.text_score_weight = args.text_score_weight
    if args.popularity_penalty is not None:
        trainer.popularity_penalty = args.popularity_penalty
    if args.ranking_mode is not None:
        trainer.ranking_mode = args.ranking_mode
    if args.fusion_lambda is not None:
        trainer.fusion_lambda = args.fusion_lambda
    if args.score_eps is not None:
        trainer.score_eps = args.score_eps

    if trainer.ranking_mode not in {"raw", "bge_only", "zscore"}:
        raise ValueError(
            "ranking_mode must be raw, bge_only, or zscore."
        )
    if not 0.0 <= trainer.fusion_lambda <= 1.0:
        raise ValueError("fusion_lambda must be in [0, 1].")
    if trainer.score_eps <= 0:
        raise ValueError("score_eps must be positive.")

    print(
        "[Analysis scoring] "
        f"split={args.split}, "
        f"mode={trainer.ranking_mode}, "
        f"lambda={trainer.fusion_lambda}, "
        f"normalize={trainer.score_normalize}, "
        f"alpha={trainer.text_score_weight}, "
        f"beta={trainer.popularity_penalty}"
    )

    train_pairs = _as_numpy_pairs(train_pairs_obj["pos"])
    val_pairs = _as_numpy_pairs(val_pairs_obj["pos"])
    test_pairs = _as_numpy_pairs(test_pairs_obj["pos"])

    if args.split == "val":
        target_pairs = val_pairs
        target_positives = pairs_by_mashup(val_pairs)
        known_by_mashup = pairs_by_mashup(train_pairs)
        ranking_graph = trainer.graph_val
        ranking_weights = trainer.edge_weight_val
    else:
        target_pairs = test_pairs
        target_positives = pairs_by_mashup(test_pairs)
        known_by_mashup = pairs_by_mashup(
            np.concatenate([train_pairs, val_pairs], axis=0)
        )
        ranking_graph = trainer.graph_test
        ranking_weights = trainer.edge_weight_test

    num_apis = int(ranking_graph["api"].x.size(0))
    target_mashup_ids = sorted(target_positives)
    max_k = max(args.ks)

    group_map, frequency = build_api_groups(
        train_pairs,
        num_apis,
        args.head_ratio,
        args.middle_ratio,
    )

    ours_rankings = model_rankings(
        trainer,
        ranking_graph,
        ranking_weights,
        target_mashup_ids,
        known_by_mashup,
        max_k,
        args.batch_size,
        args.api_chunk_size,
        str(cfg(config, "scorer", "dot")),
    )
    pop_rankings = popularity_rankings(
        target_mashup_ids,
        frequency,
        num_apis,
        known_by_mashup,
        max_k,
    )

    rankings_by_method = {
        "Ours": ours_rankings,
        "Popularity": pop_rankings,
    }

    # Overall metrics: useful for verifying the loaded checkpoint.
    overall_rows: List[Dict[str, Any]] = []
    for method, rankings in rankings_by_method.items():
        values = metrics_for_rankings(
            rankings,
            target_positives,
            args.ks,
        )
        overall_rows.append({"method": method, **values})
    overall_df = pd.DataFrame(overall_rows)

    per_mashup_metrics_df = pd.DataFrame(
        per_mashup_metric_rows(
            rankings_by_method,
            target_positives,
            args.ks,
        )
    )
    rankings_topk_df = pd.DataFrame(
        ranking_rows(
            rankings_by_method,
            target_positives,
            group_map,
            max_k,
        )
    )

    # Integrity check: the mean of per-Mashup metrics must reproduce the
    # overall metrics written by this script.
    for method in rankings_by_method:
        per_method = per_mashup_metrics_df[
            per_mashup_metrics_df["method"] == method
        ]
        overall_method = overall_df[
            overall_df["method"] == method
        ].iloc[0]
        for k in args.ks:
            for metric_name in (
                f"Recall@{k}",
                f"NDCG@{k}",
                f"HitRate@{k}",
                f"MAP@{k}",
            ):
                left = float(per_method[metric_name].mean())
                right = float(overall_method[metric_name])
                if not np.isclose(left, right, atol=1e-10, rtol=1e-8):
                    raise RuntimeError(
                        "Per-Mashup metric integrity check failed: "
                        f"method={method}, metric={metric_name}, "
                        f"per_mashup_mean={left}, overall={right}"
                    )

    per_mashup_df, overlap_df = overlap_analysis(
        ours_rankings,
        pop_rankings,
        target_positives,
        args.ks,
    )

    group_df = pd.DataFrame(
        group_metric_rows(
            rankings_by_method,
            target_positives,
            group_map,
            args.ks,
        )
    )

    composition_rows: List[Dict[str, Any]] = []
    for method, rankings in rankings_by_method.items():
        composition_rows.extend(
            recommendation_composition(
                rankings,
                group_map,
                args.ks,
                method,
            )
        )
    composition_df = pd.DataFrame(composition_rows)

    api_rows = []
    for api_id in range(num_apis):
        api_rows.append(
            {
                "api_id": api_id,
                "train_frequency": frequency.get(api_id, 0),
                "group": group_map[api_id],
            }
        )
    api_group_df = pd.DataFrame(api_rows).sort_values(
        ["group", "train_frequency", "api_id"],
        ascending=[True, False, True],
    )

    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    overlap_df.to_csv(output_dir / "overlap_summary.csv", index=False)
    per_mashup_df.to_csv(
        output_dir / "per_mashup_overlap.csv",
        index=False,
    )
    per_mashup_metrics_df.to_csv(
        output_dir / "per_mashup_metrics.csv",
        index=False,
    )
    rankings_topk_df.to_csv(
        output_dir / "rankings_topk.csv",
        index=False,
    )
    group_df.to_csv(output_dir / "group_metrics.csv", index=False)
    composition_df.to_csv(
        output_dir / "recommendation_composition.csv",
        index=False,
    )
    api_group_df.to_csv(output_dir / "api_groups.csv", index=False)

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_target_mashups": len(target_mashup_ids),
        "num_apis": num_apis,
        "head_ratio": args.head_ratio,
        "middle_ratio": args.middle_ratio,
        "ranking_mode": trainer.ranking_mode,
        "fusion_lambda": trainer.fusion_lambda,
        "score_normalize": trainer.score_normalize,
        "text_score_weight": trainer.text_score_weight,
        "popularity_penalty": trainer.popularity_penalty,
        "group_api_counts": dict(Counter(group_map.values())),
        "overall_metrics": overall_df.to_dict(orient="records"),
        "per_mashup_metric_rows": int(len(per_mashup_metrics_df)),
        "ranking_rows": int(len(rankings_topk_df)),
        "overlap_summary": overlap_df.to_dict(orient="records"),
        "group_metrics": group_df.to_dict(orient="records"),
        "recommendation_composition": composition_df.to_dict(
            orient="records"
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("Overall metrics")
    print("=" * 88)
    print(overall_df.to_string(index=False))

    print("\n" + "=" * 88)
    print("Ours vs Popularity overlap")
    print("=" * 88)
    print(overlap_df.to_string(index=False))

    print("\n" + "=" * 88)
    print("Head / Middle / Tail / Unseen metrics")
    print("=" * 88)
    display_columns = [
        "method",
        "group",
        "eligible_mashups",
        "test_positive_count",
        *[
            metric
            for k in args.ks
            for metric in (
                f"Recall@{k}",
                f"NDCG@{k}",
                f"HitRate@{k}",
                f"MAP@{k}",
            )
        ],
    ]
    print(group_df[display_columns].to_string(index=False))

    print("\nSaved files:")
    for filename in (
        "overall_metrics.csv",
        "overlap_summary.csv",
        "per_mashup_overlap.csv",
        "per_mashup_metrics.csv",
        "rankings_topk.csv",
        "group_metrics.csv",
        "recommendation_composition.csv",
        "api_groups.csv",
        "summary.json",
    ):
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
