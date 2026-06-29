from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv

try:
    from src.models.recommenders.ranking_head import build_scorer
except Exception:
    from ranking_head import build_scorer

try:
    from src.trainers.evaluator import evaluate_recommendation, format_metrics
except Exception:
    from evaluator import evaluate_recommendation, format_metrics


# -----------------------------
# Utils
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_numpy_pairs(obj: Any) -> np.ndarray:
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()
    arr = np.asarray(obj, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pair array should have shape [N, 2], got {arr.shape}")
    return arr


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Using --config requires PyYAML. Install with: pip install pyyaml") from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config should be a YAML dict, got: {type(cfg)}")
    return cfg


# -----------------------------
# Model modules
# -----------------------------


class HeteroGraphEncoder(nn.Module):
    """
    Heterogeneous graph encoder.

    1. Project each node type to hidden_dim.
    2. Apply multiple HeteroConv layers.
    3. Fallback to previous hidden state for node types that receive no messages in a view.

    This fallback is important for dual-view training: the interaction view only contains
    mashup-api edges, so category/developer/company may have no incoming edges.
    """

    def __init__(
        self,
        metadata: Tuple[list[str], list[tuple[str, str, str]]],
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_cooccur_edge_weight: bool = True,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_cooccur_edge_weight = use_cooccur_edge_weight

        node_types, edge_types = metadata
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_dim)
            for node_type in node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs = {}
            for edge_type in edge_types:
                # Only the API-API co-occurrence relation is homogeneous and can use edge_weight.
                # Other relations are bipartite or side-information relations, and use SAGEConv.
                if (
                    self.use_cooccur_edge_weight
                    and edge_type == ("api", "co_used_with", "api")
                ):
                    convs[edge_type] = GCNConv(
                        hidden_dim,
                        hidden_dim,
                        add_self_loops=False,
                        normalize=True,
                    )
                else:
                    convs[edge_type] = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

            conv = HeteroConv(convs, aggr="sum")
            self.convs.append(conv)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        h_dict = {
            node_type: F.dropout(
                F.relu(self.input_proj[node_type](x)),
                p=self.dropout,
                training=self.training,
            )
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            if edge_weight_dict:
                out_dict = conv(h_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            else:
                out_dict = conv(h_dict, edge_index_dict)
            new_h_dict = {}
            for node_type, old_h in h_dict.items():
                h = out_dict.get(node_type, None)
                if h is None:
                    h = old_h
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
                new_h_dict[node_type] = h
            h_dict = new_h_dict

        return h_dict


class TextFusion(nn.Module):
    """
    Fuse graph embedding and text embedding.

    fusion_mode:
    - mlp:          MLP([graph_proj || text_proj])
    - weighted_sum: text_weight * graph_proj + (1-text_weight) * text_proj
    """

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        hidden_dim: int,
        fusion_mode: str = "mlp",
        dropout: float = 0.1,
        text_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.text_weight = text_weight

        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        if fusion_mode == "mlp":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif fusion_mode == "weighted_sum":
            self.fusion_mlp = None
        else:
            raise ValueError("fusion_mode must be one of: mlp, weighted_sum")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        g = self.graph_proj(graph_emb)
        t = self.text_proj(text_emb)
        if self.fusion_mode == "mlp":
            return self.fusion_mlp(torch.cat([g, t], dim=-1))
        return self.text_weight * g + (1.0 - self.text_weight) * t


# -----------------------------
# Trainer
# -----------------------------


class RecommendationFinetuner:
    def __init__(
        self,
        graph_data,
        train_pairs: Dict[str, np.ndarray],
        val_pairs: Dict[str, np.ndarray],
        test_pairs: Dict[str, np.ndarray],
        processed_dir: str = "data/processed",
        hidden_dim: int = 128,
        num_layers: int = 2,
        scorer_name: str = "dot",
        scorer_dropout: float = 0.1,
        encoder_dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 200,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        use_text: bool = False,
        fusion_mode: str = "mlp",
        text_dropout: float = 0.1,
        text_weight: float = 0.5,
        align_weight: float = 0.0,
        align_loss_type: str = "mse",
        contrastive_temperature: float = 0.2,
        align_sample_size: int = 1024,
        use_view_contrast: bool = False,
        view_contrast_weight: float = 0.0,
        view_contrast_temperature: float = 0.2,
        view_contrast_sample_size: int = 1024,
        view_contrast_target: str = "both",
        use_cooccur_edge_weight: bool = True,
    ) -> None:
        self.graph_data = graph_data
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.test_pairs = test_pairs
        self.processed_dir = Path(processed_dir)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.checkpoint_path = checkpoint_path

        self.use_text = use_text
        self.fusion_mode = fusion_mode
        self.text_weight = text_weight
        self.align_weight = align_weight
        self.align_loss_type = align_loss_type
        self.contrastive_temperature = contrastive_temperature
        self.align_sample_size = align_sample_size

        self.use_view_contrast = use_view_contrast
        self.view_contrast_weight = view_contrast_weight
        self.view_contrast_temperature = view_contrast_temperature
        self.view_contrast_sample_size = view_contrast_sample_size
        self.view_contrast_target = view_contrast_target
        self.use_cooccur_edge_weight = use_cooccur_edge_weight

        if self.align_loss_type not in {"mse", "infonce"}:
            raise ValueError("align_loss_type must be one of: mse, infonce")
        if self.view_contrast_target not in {"mashup", "api", "both"}:
            raise ValueError("view_contrast_target must be one of: mashup, api, both")

        metadata = graph_data.metadata()
        input_dims = {ntype: int(graph_data[ntype].x.size(-1)) for ntype in graph_data.node_types}

        self.encoder = HeteroGraphEncoder(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=encoder_dropout,
            use_cooccur_edge_weight=use_cooccur_edge_weight,
        ).to(self.device)

        self.scorer = build_scorer(
            scorer_name=scorer_name,
            hidden_dim=hidden_dim,
            dropout=scorer_dropout,
        ).to(self.device)

        self.text_fusion: Optional[TextFusion] = None
        self.mashup_text_emb: Optional[torch.Tensor] = None
        self.api_text_emb: Optional[torch.Tensor] = None

        if self.use_text:
            self._load_text_embeddings()
            assert self.mashup_text_emb is not None and self.api_text_emb is not None
            text_dim = int(self.mashup_text_emb.size(-1))
            if int(self.api_text_emb.size(-1)) != text_dim:
                raise ValueError("mashup_text_emb and api_text_emb should have the same dim")
            self.text_fusion = TextFusion(
                graph_dim=hidden_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                fusion_mode=fusion_mode,
                dropout=text_dropout,
                text_weight=text_weight,
            ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.scorer.parameters())
        if self.text_fusion is not None:
            params += list(self.text_fusion.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.graph_data = self.graph_data.to(self.device)
        self.edge_weight_dict = self._build_edge_weight_dict()
        self.interaction_edge_index_dict = self._build_interaction_edge_index_dict()

        self.best_state = None
        self.best_metrics = None
        self.best_score = -1.0

        print("[Trainer]")
        print(f"device={self.device}")
        print(f"use_text={self.use_text}, fusion_mode={self.fusion_mode}, align_weight={self.align_weight}, align_loss_type={self.align_loss_type}")
        print(f"use_view_contrast={self.use_view_contrast}, view_contrast_weight={self.view_contrast_weight}, view_target={self.view_contrast_target}")
        print(f"use_cooccur_edge_weight={self.use_cooccur_edge_weight}")
        print("weighted edge types:", list(self.edge_weight_dict.keys()))
        print("interaction view edge types:", list(self.interaction_edge_index_dict.keys()))

    def _load_text_embeddings(self) -> None:
        mashup_path = self.processed_dir / "mashup_text_emb.npy"
        api_path = self.processed_dir / "api_text_emb.npy"
        if not mashup_path.exists():
            raise FileNotFoundError(f"Missing text embedding: {mashup_path}")
        if not api_path.exists():
            raise FileNotFoundError(f"Missing text embedding: {api_path}")

        mashup_arr = np.load(mashup_path).astype(np.float32)
        api_arr = np.load(api_path).astype(np.float32)
        self.mashup_text_emb = torch.from_numpy(mashup_arr).to(self.device)
        self.api_text_emb = torch.from_numpy(api_arr).to(self.device)

        print(f"Loaded mashup_text_emb: {tuple(self.mashup_text_emb.shape)}")
        print(f"Loaded api_text_emb: {tuple(self.api_text_emb.shape)}")

    def _build_edge_weight_dict(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Collect edge weights stored in graph_data.

        Currently only (api, co_used_with, api) is expected to have edge_weight.
        The encoder uses GCNConv for this relation and SAGEConv for other relations.
        """
        if not self.use_cooccur_edge_weight:
            return {}

        out: Dict[Tuple[str, str, str], torch.Tensor] = {}
        cooccur_etype = ("api", "co_used_with", "api")

        if cooccur_etype not in self.graph_data.edge_types:
            return out

        store = self.graph_data[cooccur_etype]
        if hasattr(store, "edge_weight"):
            w = store.edge_weight
            if w is not None:
                out[cooccur_etype] = w.float().to(self.device)
                print(
                    "Loaded API-API cooccur edge_weight:",
                    tuple(out[cooccur_etype].shape),
                    "min=", float(out[cooccur_etype].min().item()) if out[cooccur_etype].numel() else 0.0,
                    "max=", float(out[cooccur_etype].max().item()) if out[cooccur_etype].numel() else 0.0,
                    "mean=", float(out[cooccur_etype].mean().item()) if out[cooccur_etype].numel() else 0.0,
                )
        else:
            print("[WARN] API-API cooccur edge exists, but no edge_weight found. It will be treated as unweighted.")

        return out

    def _build_interaction_edge_index_dict(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build the interaction view used for view-level contrastive learning.

        Interaction view only keeps:
        - (mashup, uses, api)
        - (api, used_by, mashup)

        The final recommendation still uses the full enhanced graph.
        """
        keep_edge_types = {
            ("mashup", "uses", "api"),
            ("api", "used_by", "mashup"),
        }
        out = {}
        for etype, edge_index in self.graph_data.edge_index_dict.items():
            if etype in keep_edge_types:
                out[etype] = edge_index
        if not out:
            raise RuntimeError("Interaction view has no mashup-api edges. Check graph_data edge types.")
        return out

    def _build_train_loader(self) -> DataLoader:
        pos = _as_numpy_pairs(self.train_pairs["pos"])
        neg = _as_numpy_pairs(self.train_pairs["neg"])

        pos_y = np.ones((len(pos),), dtype=np.float32)
        neg_y = np.zeros((len(neg),), dtype=np.float32)

        all_pairs = np.concatenate([pos, neg], axis=0)
        all_y = np.concatenate([pos_y, neg_y], axis=0)

        m_idx = torch.from_numpy(all_pairs[:, 0]).long()
        a_idx = torch.from_numpy(all_pairs[:, 1]).long()
        y = torch.from_numpy(all_y).float()

        dataset = TensorDataset(m_idx, a_idx, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def _sample_indices(self, num_nodes: int, sample_size: int) -> torch.Tensor:
        if sample_size is None or sample_size <= 0 or num_nodes <= sample_size:
            return torch.arange(num_nodes, device=self.device)
        return torch.randperm(num_nodes, device=self.device)[:sample_size]

    def _symmetric_infonce(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if z1.size(0) <= 1:
            return torch.tensor(0.0, device=self.device)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = (z1 @ z2.t()) / temperature
        labels = torch.arange(logits.size(0), device=self.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_12 + loss_21)

    def _fuse_embeddings(
        self,
        mashup_graph_emb: torch.Tensor,
        api_graph_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_text:
            return mashup_graph_emb, api_graph_emb
        assert self.text_fusion is not None
        assert self.mashup_text_emb is not None and self.api_text_emb is not None
        mashup_emb = self.text_fusion(mashup_graph_emb, self.mashup_text_emb)
        api_emb = self.text_fusion(api_graph_emb, self.api_text_emb)
        return mashup_emb, api_emb

    def _compute_graph_text_align_loss(
        self,
        mashup_graph_emb: torch.Tensor,
        api_graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.use_text) or self.align_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        assert self.text_fusion is not None
        assert self.mashup_text_emb is not None and self.api_text_emb is not None

        g_m = self.text_fusion.graph_proj(mashup_graph_emb)
        t_m = self.text_fusion.text_proj(self.mashup_text_emb)
        g_a = self.text_fusion.graph_proj(api_graph_emb)
        t_a = self.text_fusion.text_proj(self.api_text_emb)

        if self.align_loss_type == "mse":
            return F.mse_loss(g_m, t_m) + F.mse_loss(g_a, t_a)

        m_idx = self._sample_indices(g_m.size(0), self.align_sample_size)
        a_idx = self._sample_indices(g_a.size(0), self.align_sample_size)
        return (
            self._symmetric_infonce(g_m[m_idx], t_m[m_idx], self.contrastive_temperature)
            + self._symmetric_infonce(g_a[a_idx], t_a[a_idx], self.contrastive_temperature)
        )

    def _compute_view_contrast_loss(
        self,
        full_m: torch.Tensor,
        full_a: torch.Tensor,
        inter_m: torch.Tensor,
        inter_a: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.use_view_contrast) or self.view_contrast_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        losses = []
        if self.view_contrast_target in {"mashup", "both"}:
            m_idx = self._sample_indices(full_m.size(0), self.view_contrast_sample_size)
            losses.append(
                self._symmetric_infonce(
                    full_m[m_idx],
                    inter_m[m_idx],
                    self.view_contrast_temperature,
                )
            )
        if self.view_contrast_target in {"api", "both"}:
            a_idx = self._sample_indices(full_a.size(0), self.view_contrast_sample_size)
            losses.append(
                self._symmetric_infonce(
                    full_a[a_idx],
                    inter_a[a_idx],
                    self.view_contrast_temperature,
                )
            )

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return sum(losses)

    def _compute_embeddings(
        self,
        compute_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
        - final mashup embedding used for recommendation
        - final api embedding used for recommendation
        - graph-text alignment loss
        - interaction/full view contrastive loss
        """
        full_dict = self.encoder(
            self.graph_data.x_dict,
            self.graph_data.edge_index_dict,
            edge_weight_dict=self.edge_weight_dict,
        )
        full_m_raw = full_dict["mashup"]
        full_a_raw = full_dict["api"]

        mashup_emb, api_emb = self._fuse_embeddings(full_m_raw, full_a_raw)

        if not compute_aux:
            zero = torch.tensor(0.0, device=self.device)
            return mashup_emb, api_emb, zero, zero

        align_loss = self._compute_graph_text_align_loss(full_m_raw, full_a_raw)

        view_loss = torch.tensor(0.0, device=self.device)
        if self.use_view_contrast and self.view_contrast_weight > 0:
            inter_dict = self.encoder(self.graph_data.x_dict, self.interaction_edge_index_dict)
            inter_m_raw = inter_dict["mashup"]
            inter_a_raw = inter_dict["api"]
            view_loss = self._compute_view_contrast_loss(
                full_m=full_m_raw,
                full_a=full_a_raw,
                inter_m=inter_m_raw,
                inter_a=inter_a_raw,
            )

        return mashup_emb, api_emb, align_loss, view_loss

    def _score_pairs(self, mashup_emb, api_emb, m_idx, a_idx) -> torch.Tensor:
        return self.scorer(mashup_emb[m_idx], api_emb[a_idx])

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict[str, float]:
        self.encoder.eval()
        self.scorer.eval()
        if self.text_fusion is not None:
            self.text_fusion.eval()

        mashup_emb, api_emb, _, _ = self._compute_embeddings(compute_aux=False)

        if split == "val":
            pos_pairs = _as_numpy_pairs(self.val_pairs["pos"])
            known_pairs = _as_numpy_pairs(self.train_pairs["pos"])
        elif split == "test":
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

        metrics = evaluate_recommendation(
            mashup_emb=mashup_emb,
            api_emb=api_emb,
            test_pos_pairs=pos_pairs,
            train_pos_pairs=known_pairs,
            scorer=self.scorer,
            ks=(5, 10),
        )
        return metrics

    def _save_best_state(self) -> None:
        self.best_state = {
            "encoder": copy.deepcopy(self.encoder.state_dict()),
            "scorer": copy.deepcopy(self.scorer.state_dict()),
        }
        if self.text_fusion is not None:
            self.best_state["text_fusion"] = copy.deepcopy(self.text_fusion.state_dict())
        if self.checkpoint_path is not None:
            torch.save(self.best_state, self.checkpoint_path)
            print(f"Saved best checkpoint to {self.checkpoint_path}")

    def _load_best_state(self) -> None:
        if self.best_state is None:
            return
        self.encoder.load_state_dict(self.best_state["encoder"])
        self.scorer.load_state_dict(self.best_state["scorer"])
        if self.text_fusion is not None and "text_fusion" in self.best_state:
            self.text_fusion.load_state_dict(self.best_state["text_fusion"])

    def train(self) -> Dict[str, float]:
        loader = self._build_train_loader()

        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.scorer.train()
            if self.text_fusion is not None:
                self.text_fusion.train()

            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_align_loss = 0.0
            epoch_view_loss = 0.0
            total = 0

            mashup_emb, api_emb, align_loss_full, view_loss_full = self._compute_embeddings(
                compute_aux=(self.align_weight > 0 or self.view_contrast_weight > 0)
            )

            for m_idx, a_idx, y in loader:
                m_idx = m_idx.to(self.device)
                a_idx = a_idx.to(self.device)
                y = y.to(self.device)

                logits = self._score_pairs(mashup_emb, api_emb, m_idx, a_idx)
                rec_loss = self.loss_fn(logits, y)
                loss = (
                    rec_loss
                    + self.align_weight * align_loss_full
                    + self.view_contrast_weight * view_loss_full
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = y.size(0)
                epoch_loss += float(loss.item()) * batch_size
                epoch_rec_loss += float(rec_loss.item()) * batch_size
                epoch_align_loss += float(align_loss_full.item()) * batch_size
                epoch_view_loss += float(view_loss_full.item()) * batch_size
                total += batch_size

                # Refresh embeddings after parameter update for next batch.
                mashup_emb, api_emb, align_loss_full, view_loss_full = self._compute_embeddings(
                    compute_aux=(self.align_weight > 0 or self.view_contrast_weight > 0)
                )

            avg_loss = epoch_loss / max(total, 1)
            avg_rec = epoch_rec_loss / max(total, 1)
            avg_align = epoch_align_loss / max(total, 1)
            avg_view = epoch_view_loss / max(total, 1)

            val_metrics = self.evaluate(split="val")
            val_score = val_metrics.get("NDCG@10", 0.0) + val_metrics.get("Recall@10", 0.0)

            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.6f} | rec={avg_rec:.6f} | "
                f"align={avg_align:.6f} | view={avg_view:.6f} | "
                f"val: {format_metrics(val_metrics)}"
            )

            if val_score > self.best_score:
                self.best_score = val_score
                self.best_metrics = val_metrics
                self._save_best_state()

        self._load_best_state()
        test_metrics = self.evaluate(split="test")
        print("Best val:", format_metrics(self.best_metrics or {}))
        print("Test:", format_metrics(test_metrics))
        return test_metrics


# -----------------------------
# IO and CLI
# -----------------------------


def load_processed_data(processed_dir: str):
    processed_dir = Path(processed_dir)
    graph_data = torch.load(processed_dir / "graph_data.pt", map_location="cpu")
    train_pairs = torch.load(processed_dir / "train_pairs.pt", map_location="cpu")
    val_pairs = torch.load(processed_dir / "val_pairs.pt", map_location="cpu")
    test_pairs = torch.load(processed_dir / "test_pairs.pt", map_location="cpu")
    return graph_data, train_pairs, val_pairs, test_pairs


def build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = defaults or {}

    def cfg(name: str, default):
        return defaults.get(name, default)

    parser = argparse.ArgumentParser(description="Finetune weighted-cooccur API recommendation model.")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=cfg("processed_dir", "data/processed"))
    parser.add_argument("--hidden_dim", type=int, default=cfg("hidden_dim", 128))
    parser.add_argument("--num_layers", type=int, default=cfg("num_layers", 2))
    parser.add_argument("--scorer", type=str, default=cfg("scorer", "dot"), choices=["mlp", "dot", "bilinear"])
    parser.add_argument("--scorer_dropout", type=float, default=cfg("scorer_dropout", 0.1))
    parser.add_argument("--encoder_dropout", type=float, default=cfg("encoder_dropout", 0.2))
    parser.add_argument("--batch_size", type=int, default=cfg("batch_size", 32))
    parser.add_argument("--epochs", type=int, default=cfg("epochs", 200))
    parser.add_argument("--lr", type=float, default=cfg("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=cfg("weight_decay", 1e-4))
    parser.add_argument("--device", type=str, default=cfg("device", "cuda"))
    parser.add_argument("--checkpoint_path", type=str, default=cfg("checkpoint_path", "outputs/checkpoints/recommender_best.pt"))
    parser.add_argument("--seed", type=int, default=cfg("seed", 0))

    parser.add_argument("--use_text", action="store_true", default=cfg("use_text", False))
    parser.add_argument("--fusion_mode", type=str, default=cfg("fusion_mode", "mlp"), choices=["mlp", "weighted_sum"])
    parser.add_argument("--text_dropout", type=float, default=cfg("text_dropout", 0.1))
    parser.add_argument("--text_weight", type=float, default=cfg("text_weight", 0.5))

    parser.add_argument("--align_weight", type=float, default=cfg("align_weight", 0.0))
    parser.add_argument("--align_loss_type", type=str, default=cfg("align_loss_type", "mse"), choices=["mse", "infonce"])
    parser.add_argument("--contrastive_temperature", type=float, default=cfg("contrastive_temperature", 0.2))
    parser.add_argument("--align_sample_size", type=int, default=cfg("align_sample_size", 1024))

    parser.add_argument("--use_view_contrast", action="store_true", default=cfg("use_view_contrast", False))
    parser.add_argument("--view_contrast_weight", type=float, default=cfg("view_contrast_weight", 0.0))
    parser.add_argument("--view_contrast_temperature", type=float, default=cfg("view_contrast_temperature", 0.2))
    parser.add_argument("--view_contrast_sample_size", type=int, default=cfg("view_contrast_sample_size", 1024))
    parser.add_argument("--view_contrast_target", type=str, default=cfg("view_contrast_target", "both"), choices=["mashup", "api", "both"])

    parser.add_argument("--use_cooccur_edge_weight", action="store_true", default=cfg("use_cooccur_edge_weight", True))

    return parser

def parse_args() -> argparse.Namespace:
    # Stage 1: read --config only.
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    known, _ = base_parser.parse_known_args()
    cfg = _load_yaml_config(known.config)

    # Stage 2: config values become defaults, CLI overrides config.
    parser = build_parser(defaults=cfg)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("Using args:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    graph_data, train_pairs, val_pairs, test_pairs = load_processed_data(args.processed_dir)

    trainer = RecommendationFinetuner(
        graph_data=graph_data,
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
    )
    trainer.train()


if __name__ == "__main__":
    main()
