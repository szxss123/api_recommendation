#!/usr/bin/env python3
"""
Strict-inductive trainer.

This file subclasses the project's existing weighted-cooccurrence trainer, so it
keeps the current encoder, scorer, BGE fusion, MSE/InfoNCE alignment and optional
view-contrast implementation.

Differences from the legacy trainer:
  * training uses graph_train.pt;
  * validation uses graph_val.pt;
  * testing uses graph_test.pt;
  * graph-text alignment and Mashup view contrast use TRAIN Mashups only;
  * val/test Mashup metadata cannot affect training Category/API representations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.trainers.finetune_trainer import (
    RecommendationFinetuner,
    _as_numpy_pairs,
    set_seed,
)

try:
    from src.trainers.evaluator import evaluate_recommendation, format_metrics
except Exception:
    from evaluator import evaluate_recommendation, format_metrics



def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_strict_args() -> argparse.Namespace:
    """
    Load the existing YAML config and support a few safe command-line
    overrides. This keeps the new scoring parameters independent of the base
    trainer's parser, so src/trainers/finetune_trainer.py does not need to be
    modified.
    """
    parser = argparse.ArgumentParser(
        description="Strict-inductive trainer with debiased ranking."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--score_normalize", type=_str2bool, default=None)
    parser.add_argument("--text_score_weight", type=float, default=None)
    parser.add_argument("--popularity_penalty", type=float, default=None)
    parser.add_argument(
        "--ranking_mode",
        choices=("raw", "bge_only", "zscore"),
        default=None,
    )
    parser.add_argument("--fusion_lambda", type=float, default=None)
    parser.add_argument("--score_eps", type=float, default=None)
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"{cli.config} must contain a YAML mapping.")

    defaults = {
        "processed_dir": "",
        "hidden_dim": 128,
        "num_layers": 2,
        "scorer": "dot",
        "scorer_dropout": 0.1,
        "encoder_dropout": 0.2,
        "batch_size": 32,
        "epochs": 50,
        "lr": 5e-4,
        "weight_decay": 5e-4,
        "device": "cuda",
        "checkpoint_path": "outputs/checkpoints/strict_best.pt",
        "use_text": True,
        "fusion_mode": "mlp",
        "text_dropout": 0.2,
        "text_weight": 0.5,
        "align_weight": 0.05,
        "align_loss_type": "mse",
        "contrastive_temperature": 0.2,
        "align_sample_size": 1024,
        "use_view_contrast": False,
        "view_contrast_weight": 0.0,
        "view_contrast_temperature": 0.2,
        "view_contrast_sample_size": 1024,
        "view_contrast_target": "api",
        "use_cooccur_edge_weight": True,
        "seed": 0,
        # Defaults preserve the legacy ranking unless the config enables them.
        "score_normalize": False,
        "text_score_weight": 0.0,
        "popularity_penalty": 0.0,
        "ranking_mode": "raw",
        "fusion_lambda": 0.5,
        "score_eps": 1e-8,
    }
    merged = {**defaults, **config}

    for key in (
        "seed",
        "device",
        "checkpoint_path",
        "epochs",
        "batch_size",
        "lr",
        "score_normalize",
        "text_score_weight",
        "popularity_penalty",
        "ranking_mode",
        "fusion_lambda",
        "score_eps",
    ):
        value = getattr(cli, key)
        if value is not None:
            merged[key] = value

    merged["config"] = str(cli.config)
    if not merged["processed_dir"]:
        raise ValueError("processed_dir is missing from the config.")
    return argparse.Namespace(**merged)

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_edge_weights(graph, enabled: bool, device: torch.device):
    if not enabled:
        return {}
    edge_type = ("api", "co_used_with", "api")
    if edge_type not in graph.edge_types:
        return {}
    store = graph[edge_type]
    if not hasattr(store, "edge_weight") or store.edge_weight is None:
        return {}
    return {edge_type: store.edge_weight.float().to(device)}


class StrictInductiveRecommendationFinetuner(RecommendationFinetuner):
    def __init__(
        self,
        *,
        graph_train,
        graph_val,
        graph_test,
        train_pairs,
        val_pairs,
        test_pairs,
        **kwargs,
    ) -> None:
        self.ranking_scorer_name = str(
            kwargs.get("scorer_name", "dot")
        ).lower()
        self.score_normalize = bool(
            kwargs.pop("score_normalize", False)
        )
        self.text_score_weight = float(
            kwargs.pop("text_score_weight", 0.0)
        )
        self.popularity_penalty = float(
            kwargs.pop("popularity_penalty", 0.0)
        )
        self.ranking_mode = str(
            kwargs.pop("ranking_mode", "raw")
        ).lower()
        self.fusion_lambda = float(
            kwargs.pop("fusion_lambda", 0.5)
        )
        self.score_eps = float(
            kwargs.pop("score_eps", 1e-8)
        )

        if self.ranking_mode not in {"raw", "bge_only", "zscore"}:
            raise ValueError(
                "ranking_mode must be one of: raw, bge_only, zscore."
            )
        if not 0.0 <= self.fusion_lambda <= 1.0:
            raise ValueError("fusion_lambda must be in [0, 1].")
        if self.score_eps <= 0:
            raise ValueError("score_eps must be positive.")
        if self.text_score_weight < 0:
            raise ValueError("text_score_weight must be non-negative.")
        if self.popularity_penalty < 0:
            raise ValueError("popularity_penalty must be non-negative.")

        # The parent trainer is initialized with the leakage-free training graph.
        super().__init__(
            graph_data=graph_train,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            **kwargs,
        )

        self.graph_train = self.graph_data
        self.graph_val = graph_val.to(self.device)
        self.graph_test = graph_test.to(self.device)

        self.edge_weight_train = collect_edge_weights(
            self.graph_train, self.use_cooccur_edge_weight, self.device
        )
        self.edge_weight_val = collect_edge_weights(
            self.graph_val, self.use_cooccur_edge_weight, self.device
        )
        self.edge_weight_test = collect_edge_weights(
            self.graph_test, self.use_cooccur_edge_weight, self.device
        )

        train_pos = _as_numpy_pairs(self.train_pairs["pos"])
        train_ids = np.unique(train_pos[:, 0])
        self.train_mashup_ids = torch.from_numpy(train_ids).long().to(self.device)

        num_apis = int(self.graph_train["api"].x.size(0))
        api_frequency = np.bincount(
            train_pos[:, 1].astype(np.int64),
            minlength=num_apis,
        ).astype(np.float32)
        popularity = np.log1p(api_frequency)
        if popularity.max() > popularity.min():
            popularity = (
                popularity - popularity.min()
            ) / (
                popularity.max() - popularity.min()
            )
        self.api_popularity = torch.from_numpy(
            popularity
        ).float().to(self.device)

        print("[Strict inductive]")
        print(f"train Mashups used for alignment: {len(train_ids)}")
        print("training graph: graph_train.pt")
        print("validation graph: graph_val.pt")
        print("test graph: graph_test.pt")
        print(
            "[Score calibration] "
            f"normalize={self.score_normalize}, "
            f"text_weight={self.text_score_weight}, "
            f"pop_penalty={self.popularity_penalty}, "
            f"mode={self.ranking_mode}, "
            f"lambda={self.fusion_lambda}"
        )

    def _compute_graph_text_align_loss(
        self,
        mashup_graph_emb: torch.Tensor,
        api_graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align only TRAIN Mashups during optimization.

        API text is allowed for all APIs because the API catalogue is assumed
        known before a new Mashup arrives.
        """
        if (not self.use_text) or self.align_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        assert self.text_fusion is not None
        assert self.mashup_text_emb is not None
        assert self.api_text_emb is not None

        train_idx = self.train_mashup_ids

        g_m = self.text_fusion.graph_proj(mashup_graph_emb[train_idx])
        t_m = self.text_fusion.text_proj(self.mashup_text_emb[train_idx])
        g_a = self.text_fusion.graph_proj(api_graph_emb)
        t_a = self.text_fusion.text_proj(self.api_text_emb)

        if self.align_loss_type == "mse":
            return F.mse_loss(g_m, t_m) + F.mse_loss(g_a, t_a)

        m_idx = self._sample_indices(g_m.size(0), self.align_sample_size)
        a_idx = self._sample_indices(g_a.size(0), self.align_sample_size)
        return (
            self._symmetric_infonce(
                g_m[m_idx], t_m[m_idx], self.contrastive_temperature
            )
            + self._symmetric_infonce(
                g_a[a_idx], t_a[a_idx], self.contrastive_temperature
            )
        )

    def _compute_view_contrast_loss(
        self,
        full_m: torch.Tensor,
        full_a: torch.Tensor,
        inter_m: torch.Tensor,
        inter_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Restrict Mashup view contrast to TRAIN Mashups.
        """
        if (not self.use_view_contrast) or self.view_contrast_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        losses = []
        if self.view_contrast_target in {"mashup", "both"}:
            full_train = full_m[self.train_mashup_ids]
            inter_train = inter_m[self.train_mashup_ids]
            local_idx = self._sample_indices(
                full_train.size(0), self.view_contrast_sample_size
            )
            losses.append(
                self._symmetric_infonce(
                    full_train[local_idx],
                    inter_train[local_idx],
                    self.view_contrast_temperature,
                )
            )

        if self.view_contrast_target in {"api", "both"}:
            api_idx = self._sample_indices(
                full_a.size(0), self.view_contrast_sample_size
            )
            losses.append(
                self._symmetric_infonce(
                    full_a[api_idx],
                    inter_a[api_idx],
                    self.view_contrast_temperature,
                )
            )

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return sum(losses)

    def _compute_embeddings_on_graph(
        self,
        graph,
        edge_weight_dict,
        *,
        compute_aux: bool,
        allow_training_aux: bool,
    ):
        full_dict = self.encoder(
            graph.x_dict,
            graph.edge_index_dict,
            edge_weight_dict=edge_weight_dict,
        )
        full_m_raw = full_dict["mashup"]
        full_a_raw = full_dict["api"]
        mashup_emb, api_emb = self._fuse_embeddings(full_m_raw, full_a_raw)

        if not compute_aux:
            zero = torch.tensor(0.0, device=self.device)
            return mashup_emb, api_emb, zero, zero

        if not allow_training_aux:
            raise RuntimeError("Auxiliary losses must only be computed on graph_train.")

        align_loss = self._compute_graph_text_align_loss(
            full_m_raw, full_a_raw
        )

        view_loss = torch.tensor(0.0, device=self.device)
        if self.use_view_contrast and self.view_contrast_weight > 0:
            inter_dict = self.encoder(
                self.graph_train.x_dict,
                self.interaction_edge_index_dict,
            )
            view_loss = self._compute_view_contrast_loss(
                full_m=full_m_raw,
                full_a=full_a_raw,
                inter_m=inter_dict["mashup"],
                inter_a=inter_dict["api"],
            )

        return mashup_emb, api_emb, align_loss, view_loss

    def _compute_embeddings(self, compute_aux: bool = True):
        # Inherited train() calls this method, therefore it always uses graph_train.
        return self._compute_embeddings_on_graph(
            self.graph_train,
            self.edge_weight_train,
            compute_aux=compute_aux,
            allow_training_aux=True,
        )


    def _graph_score_matrix(
        self,
        mashup_emb: torch.Tensor,
        api_emb: torch.Tensor,
        mashup_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute graph-model scores for selected Mashups against all APIs.
        The current strict experiments use the dot scorer.
        """
        if self.ranking_scorer_name not in {
            "dot", "dotproduct", "dot_product"
        }:
            raise RuntimeError(
                "Standardized fusion currently requires scorer: dot."
            )

        selected_m = mashup_emb[mashup_ids]
        selected_a = api_emb

        if self.score_normalize:
            selected_m = F.normalize(selected_m, p=2, dim=-1)
            selected_a = F.normalize(selected_a, p=2, dim=-1)

        return selected_m @ selected_a.T

    def _text_score_matrix(
        self,
        mashup_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.mashup_text_emb is None or self.api_text_emb is None:
            raise RuntimeError(
                "BGE text embeddings are unavailable in the processed directory."
            )

        text_m = F.normalize(
            self.mashup_text_emb[mashup_ids],
            p=2,
            dim=-1,
        )
        text_a = F.normalize(
            self.api_text_emb,
            p=2,
            dim=-1,
        )
        return text_m @ text_a.T

    def _row_zscore(self, scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False)
        return (scores - mean) / (std + self.score_eps)

    def compute_ranking_scores(
        self,
        mashup_emb: torch.Tensor,
        api_emb: torch.Tensor,
        mashup_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ranking modes
        -------------
        raw:
            graph_score + alpha * BGE_cosine - beta * popularity

        bge_only:
            BGE_cosine - beta * popularity

        zscore:
            (1-lambda) * z(graph_score)
            + lambda * z(BGE_cosine)
            - beta * popularity

        Row-wise z-score makes lambda comparable across random seeds even when
        graph-score scales differ.
        """
        if mashup_ids.dtype != torch.long:
            mashup_ids = mashup_ids.long()
        mashup_ids = mashup_ids.to(self.device)

        text_scores = None

        if self.ranking_mode == "bge_only":
            final_scores = self._text_score_matrix(mashup_ids)

        elif self.ranking_mode == "zscore":
            graph_scores = self._graph_score_matrix(
                mashup_emb,
                api_emb,
                mashup_ids,
            )
            text_scores = self._text_score_matrix(mashup_ids)
            final_scores = (
                (1.0 - self.fusion_lambda)
                * self._row_zscore(graph_scores)
                + self.fusion_lambda
                * self._row_zscore(text_scores)
            )

        else:
            graph_scores = self._graph_score_matrix(
                mashup_emb,
                api_emb,
                mashup_ids,
            )
            final_scores = graph_scores
            if self.text_score_weight > 0:
                text_scores = self._text_score_matrix(mashup_ids)
                final_scores = (
                    final_scores
                    + self.text_score_weight * text_scores
                )

        if self.popularity_penalty > 0:
            final_scores = (
                final_scores
                - self.popularity_penalty
                * self.api_popularity.unsqueeze(0)
            )

        return final_scores

    @staticmethod
    def _pairs_by_mashup(
        pairs: np.ndarray,
    ) -> Dict[int, set[int]]:
        result: Dict[int, set[int]] = {}
        for mashup_id, api_id in pairs:
            result.setdefault(int(mashup_id), set()).add(int(api_id))
        return result

    @staticmethod
    def _dcg(hits: list[int]) -> float:
        return sum(
            float(hit) / np.log2(index + 2.0)
            for index, hit in enumerate(hits)
        )

    def _metrics_from_rankings(
        self,
        rankings: Dict[int, list[int]],
        positives: Dict[int, set[int]],
        ks: tuple[int, ...] = (5, 10),
    ) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        for k in ks:
            sums[f"Recall@{k}"] = 0.0
            sums[f"NDCG@{k}"] = 0.0
            sums[f"HitRate@{k}"] = 0.0
            sums[f"MAP@{k}"] = 0.0

        count = 0
        for mashup_id, true_set in positives.items():
            if not true_set:
                continue
            ranked = rankings[mashup_id]
            count += 1

            for k in ks:
                topk = ranked[:k]
                hits = [
                    1 if api_id in true_set else 0
                    for api_id in topk
                ]
                hit_count = sum(hits)

                sums[f"Recall@{k}"] += (
                    hit_count / float(len(true_set))
                )
                sums[f"HitRate@{k}"] += (
                    1.0 if hit_count > 0 else 0.0
                )

                ideal_hits = [1] * min(len(true_set), k)
                idcg = self._dcg(ideal_hits)
                sums[f"NDCG@{k}"] += (
                    self._dcg(hits) / idcg if idcg > 0 else 0.0
                )

                precision_sum = 0.0
                cumulative_hits = 0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        cumulative_hits += 1
                        precision_sum += cumulative_hits / float(rank)

                sums[f"MAP@{k}"] += (
                    precision_sum
                    / float(min(len(true_set), k))
                )

        if count == 0:
            metrics = {key: 0.0 for key in sums}
        else:
            metrics = {
                key: value / count
                for key, value in sums.items()
            }

        # Preserve compatibility with the existing trainer display/checkpoint
        # logic, which historically used the unqualified key "MAP".
        metrics["MAP"] = metrics.get("MAP@10", 0.0)
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict[str, float]:
        self.encoder.eval()
        self.scorer.eval()
        if self.text_fusion is not None:
            self.text_fusion.eval()

        if split == "val":
            graph = self.graph_val
            weights = self.edge_weight_val
            pos_pairs = _as_numpy_pairs(self.val_pairs["pos"])
            known_pairs = _as_numpy_pairs(self.train_pairs["pos"])
        elif split == "test":
            graph = self.graph_test
            weights = self.edge_weight_test
            pos_pairs = _as_numpy_pairs(self.test_pairs["pos"])
            known_pairs = np.concatenate(
                [
                    _as_numpy_pairs(self.train_pairs["pos"]),
                    _as_numpy_pairs(self.val_pairs["pos"]),
                ],
                axis=0,
            )
        else:
            raise ValueError("split must be 'val' or 'test'")

        mashup_emb, api_emb, _, _ = self._compute_embeddings_on_graph(
            graph,
            weights,
            compute_aux=False,
            allow_training_aux=False,
        )

        positives = self._pairs_by_mashup(pos_pairs)
        known = self._pairs_by_mashup(known_pairs)
        mashup_ids = sorted(positives)

        id_tensor = torch.tensor(
            mashup_ids,
            dtype=torch.long,
            device=self.device,
        )
        scores = self.compute_ranking_scores(
            mashup_emb,
            api_emb,
            id_tensor,
        ).clone()

        for row, mashup_id in enumerate(mashup_ids):
            seen = known.get(mashup_id, set())
            if seen:
                seen_idx = torch.tensor(
                    sorted(seen),
                    dtype=torch.long,
                    device=scores.device,
                )
                scores[row, seen_idx] = -float("inf")

        max_k = 10
        topk = torch.topk(
            scores,
            k=max_k,
            dim=1,
        ).indices.detach().cpu().numpy()

        rankings = {
            mashup_id: [
                int(api_id)
                for api_id in topk[row].tolist()
            ]
            for row, mashup_id in enumerate(mashup_ids)
        }
        return self._metrics_from_rankings(
            rankings,
            positives,
            ks=(5, 10),
        )


def load_strict_processed_data(processed_dir: str):
    directory = Path(processed_dir)
    required = [
        "graph_train.pt",
        "graph_val.pt",
        "graph_test.pt",
        "train_pairs.pt",
        "val_pairs.pt",
        "test_pairs.pt",
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Strict-inductive files are missing:\n  - "
            + "\n  - ".join(missing)
        )

    return (
        torch_load(directory / "graph_train.pt"),
        torch_load(directory / "graph_val.pt"),
        torch_load(directory / "graph_test.pt"),
        torch_load(directory / "train_pairs.pt"),
        torch_load(directory / "val_pairs.pt"),
        torch_load(directory / "test_pairs.pt"),
    )


def main() -> None:
    args = parse_strict_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("Using args:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    (
        graph_train,
        graph_val,
        graph_test,
        train_pairs,
        val_pairs,
        test_pairs,
    ) = load_strict_processed_data(args.processed_dir)

    trainer = StrictInductiveRecommendationFinetuner(
        graph_train=graph_train,
        graph_val=graph_val,
        graph_test=graph_test,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        processed_dir=args.processed_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        scorer_name=args.scorer,
        scorer_dropout=args.scorer_dropout,
        encoder_dropout=args.encoder_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        checkpoint_path=str(checkpoint_path),
        use_text=args.use_text,
        fusion_mode=args.fusion_mode,
        text_dropout=args.text_dropout,
        text_weight=args.text_weight,
        align_weight=args.align_weight,
        align_loss_type=args.align_loss_type,
        contrastive_temperature=args.contrastive_temperature,
        align_sample_size=args.align_sample_size,
        use_view_contrast=args.use_view_contrast,
        view_contrast_weight=args.view_contrast_weight,
        view_contrast_temperature=args.view_contrast_temperature,
        view_contrast_sample_size=args.view_contrast_sample_size,
        view_contrast_target=args.view_contrast_target,
        use_cooccur_edge_weight=args.use_cooccur_edge_weight,
        score_normalize=args.score_normalize,
        text_score_weight=args.text_score_weight,
        popularity_penalty=args.popularity_penalty,
        ranking_mode=args.ranking_mode,
        fusion_lambda=args.fusion_lambda,
        score_eps=args.score_eps,
    )
    trainer.train()


if __name__ == "__main__":
    main()
