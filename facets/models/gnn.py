"""
Lightweight GNN for FACETS.

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointbert_extended import knn_query


class GeometryAwareGNN(nn.Module):
    """
    Single-layer edge-conv with 3D relative-position features.

    Input:
        tilde_z:  (B, N, d)   enhanced patch embeddings
        centers:  (B, N, 3)   patch centers

    Output:
        bar_z:      (B, N, d)    refined per-patch embeddings
        z_gnn:      (B, d)       mean-pooled global patch representation

    """

    def __init__(self, d: int, d_e: int = 64, k: int = 12):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.k = k

        # Edge encoder f_edge: R^3 -> R^{d_e}. Single linear layer.
        self.f_edge = nn.Linear(3, d_e, bias=True)

        # W_GNN: R^{d x (2d + d_e)}
        self.W_gnn = nn.Linear(2 * d + d_e, d, bias=True)

        # Xavier init
        nn.init.xavier_uniform_(self.f_edge.weight)
        nn.init.zeros_(self.f_edge.bias)
        nn.init.xavier_uniform_(self.W_gnn.weight)
        nn.init.zeros_(self.W_gnn.bias)

    # ------------------------------------------------------------------
    def forward(self, tilde_z: torch.Tensor, centers: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, d = tilde_z.shape
        k = min(self.k, N - 1)

        # Build k-NN graph on patch centers. Exclude self (distance-0 neighbor).
        with torch.no_grad():
            dist = torch.cdist(centers, centers, p=2.0)       # (B, N, N)
            _, sorted_idx = torch.topk(dist, k + 1, dim=-1, largest=False)
            nn_idx = sorted_idx[..., 1:k + 1]                 # (B, N, k) excl. self

        b_idx = torch.arange(B, device=centers.device).view(B, 1, 1).expand(-1, N, k)

        # Gather neighbor features and centers
        z_neigh = tilde_z[b_idx, nn_idx]                      # (B, N, k, d)
        c_neigh = centers[b_idx, nn_idx]                      # (B, N, k, 3)

        # Relative position: delta_c = c_j - c_i
        delta_c = c_neigh - centers.unsqueeze(2)              # (B, N, k, 3)
        e = self.f_edge(delta_c)                              # (B, N, k, d_e)

        # Difference-based term: tilde_z_j - tilde_z_i
        diff = z_neigh - tilde_z.unsqueeze(2)                 # (B, N, k, d)

        # Center term: replicate tilde_z_i
        z_i = tilde_z.unsqueeze(2).expand(-1, -1, k, -1)      # (B, N, k, d)

        # Concatenate [tilde_z_i ; tilde_z_j - tilde_z_i ; e_ji]
        msg_in = torch.cat([z_i, diff, e], dim=-1)            # (B, N, k, 2d+d_e)
        m = self.W_gnn(msg_in)                                # (B, N, k, d)

        # Mean aggregation + ReLU
        agg = m.mean(dim=2)                                   # (B, N, d)
        agg = F.relu(agg)

        # Residual connection
        bar_z = tilde_z + agg                                 # (B, N, d)

        # Global mean pool
        z_gnn = bar_z.mean(dim=1)                             # (B, d)

        return bar_z, z_gnn
