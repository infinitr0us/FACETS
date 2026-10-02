"""
Main FACETS framework.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointbert_extended import ExtendedPointBERT
from .gnn import GeometryAwareGNN


# ---------------------------------------------------------------------------
# MLP: g_mlp : R^{d_mini} -> R^{d}
# ---------------------------------------------------------------------------

class PointFeatureMLP(nn.Module):
    """
    Two-layer MLP with ReLU.
    Kaiming initialisation is used (fan-in, ReLU non-linearity).

    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Main framework
# ---------------------------------------------------------------------------

@dataclass
class FrameworkConfig:
    d: int = 768                # transformer hidden dim of Point-BERT
    d_mini: int = 256           # Mini-PointNet output dim
    d_bar: int = 512            # joint 3D-text embedding dim
    d_e: int = 64               # edge-encoding dim in the GNN
    k_gnn: int = 12             # GNN neighborhood size
    num_patches: int = 64       # N
    patch_size: int = 32        # K
    eps_zscore: float = 1e-6    # z-score denominator guard
    attn_layer_indices: Optional[list] = None  # None = all layers for beta


class ULIPADFramework(nn.Module):
    """
    Full framework: frozen Point-BERT + trainable head.

    """

    def __init__(self,
                 pointbert: ExtendedPointBERT,
                 P_ulip: torch.Tensor,        # original ULIP-2 projection (d_bar, 2d)
                 cfg: FrameworkConfig,
                 freeze_pointbert: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pointbert = pointbert
        if freeze_pointbert:
            for p in self.pointbert.parameters():
                p.requires_grad_(False)

        d, d_bar = cfg.d, cfg.d_bar

        # Handle possible transpose: we need (d_bar, 2d).
        if P_ulip.shape == (2 * d, d_bar):
            P_ulip = P_ulip.t().contiguous()
        assert P_ulip.shape == (d_bar, 2 * d), \
            f"P_ulip has shape {tuple(P_ulip.shape)} but need ({d_bar}, {2*d})"

        # Split into P^CLS and P^pool, register as buffers (frozen).
        P_cls  = P_ulip[:, :d].contiguous()
        P_pool = P_ulip[:, d:].contiguous()
        self.register_buffer('P_full', P_ulip)          # (d_bar, 2d) frozen
        self.register_buffer('P_cls',  P_cls)           # (d_bar, d)  frozen
        self.register_buffer('P_pool', P_pool)          # (d_bar, d)  frozen

        # Trainable P_hat, init from P_pool.
        self.P_hat = nn.Parameter(P_pool.clone())

        # g_mlp: d_mini -> d
        self.g_mlp = PointFeatureMLP(cfg.d_mini, d)

        # GNN
        self.gnn = GeometryAwareGNN(d=d, d_e=cfg.d_e, k=cfg.k_gnn)

        # Learnable temperature w_tau; tau = exp(w_tau)
        self.w_tau = nn.Parameter(torch.tensor(float(np.log(0.07))))

    # ------------------------------------------------------------------
    @property
    def tau(self) -> torch.Tensor:
        """
        Current temperature (log-parameterized). Bounded below for stability.

        """
        # Clamp from below so tau cannot collapse to 0 (causes NaNs).
        return torch.exp(self.w_tau.clamp(min=float(np.log(0.005))))

    # ------------------------------------------------------------------
    def _project_hat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project d-dim vector into joint space via P_hat, then L2-normalize.
        Normalization runs in float32, so an all-zero input (e.g. tilde_v when
        every w_{j,i} = 0) maps to 0, not NaN, on backends whose float16
        autocast would keep the norm in float16.

        """
        y = F.linear(x, self.P_hat)             # (..., d_bar)
        return F.normalize(y.float(), dim=-1)

    def _project_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project CLS d-dim vector via P_cls (frozen), then L2-normalize.

        """
        y = F.linear(x, self.P_cls)             # (..., d_bar)
        return F.normalize(y.float(), dim=-1)

    def _project_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project 2d-dim concatenated vector via P (frozen), L2-normalize.

        """
        y = F.linear(x, self.P_full)            # (..., d_bar)
        return F.normalize(y.float(), dim=-1)

    # ------------------------------------------------------------------
    def forward(self,
                pts: torch.Tensor,
                text_embed: torch.Tensor,      # (d_bar,) or (B, d_bar)
                return_point: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        """
        B, _, _ = pts.shape

        # Ensure text_embed has correct shape for broadcasting
        if text_embed.dim() == 1:
            t = text_embed.unsqueeze(0).expand(B, -1)           # (B, d_bar)
        else:
            t = text_embed                                       # (B, d_bar)
        t = F.normalize(t, dim=-1)

        # -------------------- 1. Point-BERT forward --------------------
        # Backbone is frozen; do not need gradients through it.
        with torch.set_grad_enabled(any(p.requires_grad for p in self.pointbert.parameters())):
            feats = self.pointbert(pts, keep_attn_grad=False)
        cls       = feats['cls']                     # (B, d)
        patches   = feats['patches']                 # (B, N, d)
        centers   = feats['centers']                 # (B, N, 3)
        point_feats = feats['point_feats']           # (B, N, K, d_mini)
        knn_idx   = feats['knn_idx']                 # (B, N, K)

        # -------------------- 2. Beta from holistic--local attention ---------------------
        beta_all = self.pointbert.get_beta(include_cls_self=True,
                                           layer_indices=self.cfg.attn_layer_indices)
        # beta_all: (B, N+1).  beta_0 = CLS self-attention; beta[1:] are patches.
        beta_0   = beta_all[:, 0]                    # (B,)
        beta     = beta_all[:, 1:]                   # (B, N)

        # -------------------- 3. Mu_z from patch linguistic grounding -----
        patch_proj = self._project_hat(patches)      # (B, N, d_bar), normalized
        cls_proj_for_mu = self._project_cls(cls)     # (B, d_bar), normalized
        #   mu_i = max(cos(P_hat z_i^L, t), 0)
        mu = torch.clamp((patch_proj * t.unsqueeze(1)).sum(dim=-1), min=0.)
        #   mu_0 = max(cos(P_CLS z_0^L, t), 0)
        mu_0 = torch.clamp((cls_proj_for_mu * t).sum(dim=-1), min=0.)        # (B,)

        # -------------------- 4. Pre-max-pool point features through g_mlp --
        # g_mlp: d_mini -> d, then project to joint space
        gv = self.g_mlp(point_feats)                 # (B, N, K, d)
        vj_proj = self._project_hat(gv)              # (B, N, K, d_bar), normalised

        # -------------------- 5. w_{j,i} from point geometric retention  --
        # w_{j,i} = max(cos(P_hat g_mlp(v_j), P_hat z_i^L), 0)
        # patch_proj: (B, N, d_bar); vj_proj: (B, N, K, d_bar)
        raw_cos = (vj_proj * patch_proj.unsqueeze(2)).sum(dim=-1)  # (B, N, K)
        w = torch.clamp(raw_cos, min=0.)                           # (B, N, K)

        # Consolidated-point representations (in d-space, NOT joint space)
        tilde_v = (w.unsqueeze(-1) * gv).sum(dim=2)                 # (B, N, d)

        # -------------------- 6. Mu_v from consolidated-point linguistic grounding -----
        tilde_v_proj = self._project_hat(tilde_v)                   # (B, N, d_bar)
        mu_v = torch.clamp((tilde_v_proj * t.unsqueeze(1)).sum(dim=-1), min=0.)

        # -------------------- 7. Combined patch embeddings -----------------
        # tilde_z_i = beta_i (mu_i z_i^L + mu_v_i tilde_v_i)
        tilde_z = beta.unsqueeze(-1) * (
            mu.unsqueeze(-1) * patches + mu_v.unsqueeze(-1) * tilde_v
        )                                                           # (B, N, d)

        # -------------------- 8. GNN in the Combiner --------------------------
        _, z_gnn = self.gnn(tilde_z, centers)                       # (B, d)

        # -------------------- 9. Weighted CLS ----------------------------
        bar_cls = (mu_0 * beta_0).unsqueeze(-1) * cls                # (B, d)

        # -------------------- 10. Final concatenation & projection -------
        z_final = torch.cat([bar_cls, z_gnn], dim=-1)               # (B, 2d)
        z_final_proj = self._project_full(z_final)                  # (B, d_bar)

        out: Dict[str, torch.Tensor] = {
            'z_final':      z_final,
            'z_final_proj': z_final_proj,
            'patch_proj':   patch_proj,
            'vj_proj':      vj_proj,
            'tilde_v':      tilde_v,
            'beta':         beta,
            'mu':           mu,
            'mu_v':         mu_v,
            'knn_idx':      knn_idx,
            'centers':      centers,
            # For loss computations:
            'patches':      patches,            # (B, N, d)
            'gv':           gv,                 # (B, N, K, d)
            'cls':          cls,
            'tau':          self.tau,
        }
        if return_point:
            # s_local: point vs. patch centroid (in P_hat space).
            K = gv.shape[2]
            mean_per_patch = gv.mean(dim=2, keepdim=True)            # (B, N, 1, d)
            loo_centroid = (mean_per_patch * K - gv) / (K - 1)       # (B, N, K, d)
            loo_centroid_proj = self._project_hat(loo_centroid)      # (B, N, K, d_bar)

            s_local = -(vj_proj * loo_centroid_proj).sum(dim=-1)     # (B, N, K)

            s_consist = -raw_cos                                      # (B, N, K)

            # Sum and within-patch z-score.
            s_point = s_local + s_consist                             # (B, N, K)
            s_mean = s_point.mean(dim=2, keepdim=True)                # (B, N, 1)
            s_std = s_point.std(dim=2, keepdim=True)
            s_point_norm = (s_point - s_mean) / (s_std + self.cfg.eps_zscore)
            out['s_local'] = s_local
            out['s_consist'] = s_consist
            out['s_point_raw'] = s_point
            out['s_point_norm'] = s_point_norm
        return out

    # ------------------------------------------------------------------
    # Parameter grouping helpers
    # ------------------------------------------------------------------
    def trainable_param_groups(self,
                               base_lr: float = 5e-4,
                               P_hat_lr: float = 5e-5,
                               weight_decay: float = 1e-2) -> list:
        """
        Parameter groups for the optimizer.

        """
        phat_params = [self.P_hat]
        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and n != 'P_hat']
        return [
            {'params': phat_params,  'lr': P_hat_lr,
             'weight_decay': weight_decay, 'name': 'P_hat'},
            {'params': other_params, 'lr': base_lr,
             'weight_decay': weight_decay, 'name': 'head'},
        ]
