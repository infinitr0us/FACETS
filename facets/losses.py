"""
Loss functions for FACETS.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.distributed.nn.functional import all_gather as dist_all_gather
except Exception:
    dist_all_gather = None


# ---------------------------------------------------------------------------
# Category-masked 3D <-> text contrastive loss
# ---------------------------------------------------------------------------

def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    if _dist_world_size() == 1:
        return x
    if dist_all_gather is None:
        raise RuntimeError(
            "Differentiable distributed all_gather is unavailable in this "
            "PyTorch build; upgrade PyTorch or disable DDP.")
    return torch.cat(dist_all_gather(x), dim=0)


@torch.no_grad()
def _gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    if _dist_world_size() == 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=0)

def category_masked_contrastive_loss(emb_3d: torch.Tensor,    # (B, d_bar)
                                     emb_text: torch.Tensor,  # (B, d_bar)
                                     labels: torch.Tensor,    # (B,)
                                     tau: torch.Tensor        # scalar tensor
                                     ) -> Dict[str, torch.Tensor]:
    """
    CLIP-style symmetric InfoNCE loss over V(b) = {k : k = b or y_k != y_b},
    i.e. same-category off-diagonal pairs are excluded from both softmaxes.

    """
    B = emb_3d.shape[0]
    emb_3d = F.normalize(emb_3d, dim=-1)
    emb_text = F.normalize(emb_text, dim=-1)
    labels = labels.long().view(-1)

    # (B, B) similarity matrix, rows = 3D, cols = text
    logits = (emb_3d @ emb_text.t()).float() / tau
    same_cat = labels.unsqueeze(0) == labels.unsqueeze(1)
    off_diag = ~torch.eye(B, dtype=torch.bool, device=labels.device)
    logits = logits.masked_fill(same_cat & off_diag, float('-inf'))

    targets = torch.arange(B, device=emb_3d.device)
    l_3d2t = F.cross_entropy(logits, targets)       # rows are 3D queries
    l_t23d = F.cross_entropy(logits.t(), targets)   # cols are text queries
    return {
        'l_3d2t': l_3d2t,
        'l_t23d': l_t23d,
    }


# ---------------------------------------------------------------------------
# Point-to-patch contrastive loss
# ---------------------------------------------------------------------------

def point_patch_contrastive_loss(vj_proj: torch.Tensor,    # (B, N, K, d_bar)
                                 patch_proj: torch.Tensor, # (B, N, d_bar)
                                 tau: torch.Tensor,
                                 sample_points_per_patch: int = 0
                                 ) -> torch.Tensor:
    """
    For each point j in patch i, the positive is patch i and negatives are
    the other N-1 patches in the SAME point cloud.

    """
    B, N, K, D = vj_proj.shape
    device = vj_proj.device
    vj = F.normalize(vj_proj, dim=-1)
    pz = F.normalize(patch_proj, dim=-1)

    if sample_points_per_patch and sample_points_per_patch < K:
        # Deterministic per-forward subsample; uniform random choice.
        idx = torch.randint(0, K, (sample_points_per_patch,), device=device)
        vj = vj[:, :, idx, :]
        K = sample_points_per_patch

    # sim[b, i, j, k] = cos( v_{b,i,j}, z_{b,k} )  -- note k over all patches
    # vj: (B, N, K, D); pz: (B, N, D)
    sim = torch.einsum('bikd, bnd -> bikn', vj, pz)                   # (B, N, K, N)
    logits = sim.float() / tau

    # Target: patch index = i for the query point in patch i.
    targets = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)

    # Cross-entropy over the last dim.
    loss = F.cross_entropy(
        logits.reshape(-1, N),        # (B*N*K, N)
        targets.reshape(-1),          # (B*N*K,)
    )
    return loss


# ---------------------------------------------------------------------------
# Intra-patch consistency
# ---------------------------------------------------------------------------

def intra_patch_consistency_loss(vj_proj: torch.Tensor,   # (B, N, K, d_bar)
                                 sample_pairs: int = 0
                                 ) -> torch.Tensor:
    """
    Mean pairwise cosine-distance within each patch.

    """
    B, N, K, D = vj_proj.shape
    v = F.normalize(vj_proj, dim=-1)

    if sample_pairs <= 0 or sample_pairs >= K * (K - 1):
        s = v.sum(dim=2)                          # (B, N, D)
        sum_sq = (s * s).sum(dim=-1)              # (B, N)
        pair_cos_sum = sum_sq - K                  # sum_{j,k} cos - diag

        # sum_{j!=k} cos = pair_cos_sum (diagonal is K, subtracted)
        mean_pair_cos = pair_cos_sum / (K * (K - 1))
        loss = (1.0 - mean_pair_cos).mean()
        return loss
    else:
        # Random pairs
        device = v.device
        j = torch.randint(0, K, (sample_pairs,), device=device)
        k = torch.randint(0, K, (sample_pairs,), device=device)

        # Re-draw where j == k
        same = (j == k)
        while same.any():
            k = torch.where(same, torch.randint(0, K, k.shape, device=device), k)
            same = (j == k)
        vj_s = v[:, :, j, :]                      # (B, N, P, D)
        vk_s = v[:, :, k, :]                      # (B, N, P, D)
        cos = (vj_s * vk_s).sum(dim=-1)           # (B, N, P)
        loss = (1.0 - cos.mean())
        return loss


# ---------------------------------------------------------------------------
# Combined loss wrapper
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    lambda_intra: float = 0.1
    point_patch_sample: int = 0     # 0 = use all K per patch; else K'


class TotalLoss(nn.Module):
    """
    Convenience module that takes the framework's outputs and per-sample
    text embeddings and returns the total training loss + a components dict.

    """

    def __init__(self, weights: LossWeights = LossWeights()):
        super().__init__()
        self.w = weights

    def forward(self,
                framework_out: Dict[str, torch.Tensor],
                text_embed: torch.Tensor,   # (B, d_bar) normalised
                tau: torch.Tensor,
                labels: torch.Tensor,       # (B,) category indices
                ) -> Dict[str, torch.Tensor]:
        z_final_proj = framework_out['z_final_proj']   # (B, d_bar)
        vj_proj      = framework_out['vj_proj']        # (B, N, K, d_bar)
        patch_proj   = framework_out['patch_proj']     # (B, N, d_bar)

        # Under DDP the contrastive loss runs over the global batch. The
        # all_gather backward sums the per-rank gradients of the local slice,
        # so after DDP averaging the gradients match a single process.
        cont = category_masked_contrastive_loss(
            _gather_with_grad(z_final_proj),
            _gather_no_grad(text_embed),
            _gather_no_grad(labels),
            tau)
        l_pp = point_patch_contrastive_loss(
            vj_proj, patch_proj, tau,
            sample_points_per_patch=self.w.point_patch_sample)
        l_intra = intra_patch_consistency_loss(vj_proj)

        total = cont['l_3d2t'] + cont['l_t23d'] + l_pp + \
                self.w.lambda_intra * l_intra

        return {
            'loss': total,
            'l_3d2t': cont['l_3d2t'].detach(),
            'l_t23d': cont['l_t23d'].detach(),
            'l_point_patch': l_pp.detach(),
            'l_intra': l_intra.detach(),
            'tau': tau.detach(),
        }
