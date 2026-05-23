"""GIN_PPI: GIN-based GNN for multi-label PPI edge classification.

Architecture (per protein → per edge):

    Inputs (precomputed once)
    ─────────────────────────
    x:         (N, esm_dim=320)     ← frozen ESM-2 mean-pooled per protein
    pathway_x: (N, pathway_dim=768) ← BioBERT pathway embedding per protein

    Fusion
    ──────
    cat([x ; pathway_x])              → (N, esm_dim + pathway_dim)
        ↓ self.fuse  (Linear)
    (N, gin_in_feature)

    Graph convolutions
    ──────────────────
    GIN layers × num_layers → optional JumpingKnowledge
        ↓ lin1 → ReLU → dropout → lin2
    (N, num_classes)

    Edge prediction
    ───────────────
    For each train_edge_id k:
        x1 = x[edge_index[0, k]]   # (batch, num_classes)
        x2 = x[edge_index[1, k]]   # (batch, num_classes)
        edge_feat = mul(x1, x2)    # or concat
        ↓ fc2
    (batch, num_classes)   ← logits, BCEWithLogitsLoss outside

Compared to the previous version: the entire sequence encoder
(Conv1d + BatchNorm + MaxPool + BiGRU + AvgPool + fc1) is gone.
ESM-2 already does that work, frozen, once at precompute time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge


class GIN_PPI(nn.Module):
    def __init__(
        self,
        esm_dim:        int  = 320,        # ESM-2 8M output dim (480/640/1280 for larger)
        pathway_dim:    int  = 768,        # BioBERT
        gin_in_feature: int  = 256,
        gin_hidden:     int  = 512,
        num_gin_layers: int  = 1,
        use_jk:         bool = False,
        train_eps:      bool = True,
        feature_fusion: str  = "mul",      # "mul" | "concat"
        num_classes:    int  = 7,
        dropout:        float = 0.5,
    ):
        super().__init__()
        if feature_fusion not in {"mul", "concat"}:
            raise ValueError(f"feature_fusion must be 'mul' or 'concat', got {feature_fusion!r}")

        self.esm_dim        = esm_dim
        self.pathway_dim    = pathway_dim
        self.feature_fusion = feature_fusion
        self.use_jk         = use_jk
        self.dropout        = dropout
        self.num_gin_layers = num_gin_layers

        # ---------- ESM + pathway fusion ----------
        self.fuse = nn.Linear(esm_dim + pathway_dim, gin_in_feature)

        # ---------- GIN stack ----------
        self.gin_convs = nn.ModuleList()
        for i in range(num_gin_layers):
            in_dim = gin_in_feature if i == 0 else gin_hidden
            self.gin_convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, gin_hidden),
                    nn.ReLU(),
                    nn.Linear(gin_hidden, gin_hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(gin_hidden),
                ),
                train_eps=train_eps,
            ))

        # ---------- post-GIN projection ----------
        if use_jk:
            self.jump = JumpingKnowledge("cat")
            self.lin1 = nn.Linear(num_gin_layers * gin_hidden, gin_hidden)
        else:
            self.lin1 = nn.Linear(gin_hidden, gin_hidden)
        self.lin2 = nn.Linear(gin_hidden, num_classes)

        # ---------- edge classifier ----------
        edge_in = num_classes * (2 if feature_fusion == "concat" else 1)
        self.fc2 = nn.Linear(edge_in, num_classes)

    def reset_parameters(self) -> None:
        for m in [self.fuse, self.lin1, self.lin2, self.fc2]:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for gin in self.gin_convs:
            gin.reset_parameters()
        if self.use_jk:
            self.jump.reset_parameters()

    def forward(
        self,
        x:             torch.Tensor,            # (N, esm_dim)
        pathway_x:     torch.Tensor,            # (N, pathway_dim)
        edge_index:    torch.Tensor,            # (2, 2E)
        train_edge_id: torch.Tensor | list,     # indices into [0, 2E)
    ) -> torch.Tensor:
        # Fusion of ESM + pathway → per-node feature for GIN
        node_feat = torch.cat([x, pathway_x], dim=1)   # (N, esm_dim + pathway_dim)
        node_feat = self.fuse(node_feat)               # (N, gin_in_feature)

        # GIN stack
        xs = []
        for gin in self.gin_convs:
            node_feat = gin(node_feat, edge_index)
            xs.append(node_feat)
        if self.use_jk:
            node_feat = self.jump(xs)

        # Post-GIN per-node projection → logits per node
        node_feat = F.relu(self.lin1(node_feat))
        node_feat = F.dropout(node_feat, p=self.dropout, training=self.training)
        node_feat = self.lin2(node_feat)               # (N, num_classes)

        # Edge prediction: gather endpoints of selected edges
        if not torch.is_tensor(train_edge_id):
            train_edge_id = torch.as_tensor(train_edge_id, dtype=torch.long, device=x.device)
        node_id = edge_index[:, train_edge_id]         # (2, batch)
        x1 = node_feat[node_id[0]]                     # (batch, num_classes)
        x2 = node_feat[node_id[1]]                     # (batch, num_classes)

        if self.feature_fusion == "concat":
            edge_feat = torch.cat([x1, x2], dim=1)     # (batch, 2*num_classes)
        else:
            edge_feat = torch.mul(x1, x2)              # (batch, num_classes)

        return self.fc2(edge_feat)                     # (batch, num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
