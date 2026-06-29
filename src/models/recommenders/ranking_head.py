from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class DotProductScorer(nn.Module):
    """Simple dot-product scorer for mashup/API embedding pairs."""

    def forward(self, mashup_emb: torch.Tensor, api_emb: torch.Tensor) -> torch.Tensor:
        return (mashup_emb * api_emb).sum(dim=-1)


class BilinearScorer(nn.Module):
    """Bilinear scorer: score(m, a) = m^T W a."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, mashup_emb: torch.Tensor, api_emb: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(mashup_emb, self.weight)
        return (projected * api_emb).sum(dim=-1)


class PairFeatureScorer(nn.Module):
    """
    Pairwise MLP scorer.

    For each mashup/API pair, build the feature vector:
        [m || a || m * a || |m-a|]
    and map it to a scalar score.

    Works for both shapes:
    - [B, D]
    - [B, K, D]
    because nn.Linear broadcasts over leading dims.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * 4

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mashup_emb: torch.Tensor, api_emb: torch.Tensor) -> torch.Tensor:
        feat = torch.cat(
            [
                mashup_emb,
                api_emb,
                mashup_emb * api_emb,
                torch.abs(mashup_emb - api_emb),
            ],
            dim=-1,
        )
        out = self.mlp(feat)
        return out.squeeze(-1)


def build_scorer(
    scorer_name: str,
    hidden_dim: int,
    mlp_hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.1,
) -> nn.Module:
    scorer_name = scorer_name.lower()
    if scorer_name in {"dot", "dot_product", "dotproduct"}:
        return DotProductScorer()
    if scorer_name in {"bilinear", "biaffine"}:
        return BilinearScorer(hidden_dim)
    if scorer_name in {"mlp", "pair_mlp", "pair_feature"}:
        return PairFeatureScorer(
            hidden_dim=hidden_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported scorer: {scorer_name}")
