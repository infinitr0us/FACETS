"""
Utilities: score interpolation and LR scheduler.

"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist


# ===========================================================================
# Point-score deduplication + full-resolution interpolation (RPRIL-style)
# ===========================================================================

@torch.no_grad()
def collect_unique_scored_points(s_point: torch.Tensor,     # (N, K)
                                 knn_idx: torch.Tensor,     # (N, K) idx into sub-cloud
                                 pc_sub:  torch.Tensor      # (M_sub, 3)
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a de-duplicated set of (coord, score) pairs from patch-points.

    Returns
    -------
    coords : (U, 3)  unique scored point coordinates
    scores : (U,)    corresponding mean scores

    """
    flat_idx = knn_idx.reshape(-1)              # (N*K,)
    flat_scores = s_point.reshape(-1)           # (N*K,)

    unique_idx, inverse = torch.unique(flat_idx, return_inverse=True)
    U = unique_idx.numel()
    device = flat_scores.device
    dtype = flat_scores.dtype
    sum_scores = torch.zeros(U, device=device, dtype=dtype)
    counts = torch.zeros(U, device=device, dtype=dtype)
    sum_scores.scatter_add_(0, inverse, flat_scores)
    counts.scatter_add_(0, inverse, torch.ones_like(flat_scores))
    mean_scores = sum_scores / counts.clamp(min=1)
    coords = pc_sub[unique_idx]                 # (U, 3)
    return coords, mean_scores


@torch.no_grad()
def knn_interpolate_to_targets(src_xyz: torch.Tensor,       # (U, 3)
                               src_scores: torch.Tensor,    # (U,)
                               tgt_xyz: torch.Tensor,       # (M_gt, 3)
                               k: int = 3,
                               eps: float = 1e-8,
                               chunk: int = 65536
                               ) -> torch.Tensor:
    """
    Inverse-distance-weighted kNN interpolation.

    This matches the benchmark practice in Real3D-AD and its
    downstream works (e.g. 3DKeyAD's "k=3 NN" interpolation).

    """
    if src_xyz.numel() == 0:
        return torch.zeros(tgt_xyz.shape[0], device=tgt_xyz.device,
                           dtype=src_scores.dtype)

    # Chunk over target points to cap memory: Real3D-AD full-res clouds
    # frequently exceed 200k points per sample.
    M_tgt = tgt_xyz.shape[0]
    out = torch.empty(M_tgt, device=tgt_xyz.device, dtype=src_scores.dtype)
    k_eff = min(k, src_xyz.shape[0])
    for start in range(0, M_tgt, chunk):
        end = min(start + chunk, M_tgt)
        d = torch.cdist(tgt_xyz[start:end], src_xyz, p=2.0)         # (m, U)
        d_k, idx_k = torch.topk(d, k_eff, dim=-1, largest=False)    # (m, k)
        w = 1.0 / (d_k + eps)
        w = w / w.sum(dim=-1, keepdim=True)
        out[start:end] = (w * src_scores[idx_k]).sum(dim=-1)
    return out


@torch.no_grad()
def score_test_sample(s_point_norm: torch.Tensor,   # (N, K)   per-patch-point scores
                      knn_idx:      torch.Tensor,   # (N, K)   idx into pc_sub
                      pc_sub:       torch.Tensor,   # (M_sub, 3)
                      pc_gt:        torch.Tensor,   # (M_gt, 3)
                      k: int = 3,
                      ) -> torch.Tensor:
    """
    End-to-end per-point scoring of one test sample's GT points.

    """
    src_xyz, src_scores = collect_unique_scored_points(
        s_point_norm, knn_idx, pc_sub)
    return knn_interpolate_to_targets(src_xyz, src_scores, pc_gt, k=k)


# ===========================================================================
# LR scheduler: warmup + cosine
# ===========================================================================

class WarmupCosineLR:
    """
    Per-step warmup + cosine-decay LR multiplier for ``torch.optim`` groups.

    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.0):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(1, total_steps)
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g['lr'] for g in self.opt.param_groups]
        self.step_num = 0
        self._apply()

    def _ratio(self, step: int) -> float:
        # ``step`` = number of optimizer steps already taken.
        if step < self.warmup:
            return (step + 1) / self.warmup
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_ratio + (1.0 - self.min_ratio) * cos

    def _apply(self):
        r = self._ratio(self.step_num)
        for g, b in zip(self.opt.param_groups, self.base_lrs):
            g['lr'] = b * r

    def step(self):
        self.step_num += 1
        self._apply()

    def state_dict(self):
        return {'step_num': self.step_num, 'base_lrs': self.base_lrs}

    def load_state_dict(self, sd):
        self.step_num = sd['step_num']
        self.base_lrs = sd['base_lrs']
        self._apply()


# ===========================================================================
# Seeding helper
# ===========================================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Distributed training helpers
# ===========================================================================

@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(requested_device: str = 'cuda') -> DistributedContext:
    """
    Initialize torch.distributed when launched by torchrun.

    Launch example:
        torchrun --standalone --nproc_per_node=4 -m facets.train ...

    Outside torchrun this returns a single-process context and does not call
    ``init_process_group``.

    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    enabled = world_size > 1

    wants_cuda = requested_device.startswith('cuda')
    use_cuda = wants_cuda and torch.cuda.is_available()
    if enabled and use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    elif use_cuda:
        device = torch.device(requested_device)
    else:
        device = torch.device('cpu')

    backend = 'nccl' if use_cuda and dist.is_nccl_available() else 'gloo'
    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ===========================================================================
# Exponential moving average over trainable parameters
# ===========================================================================

class TrainableEMA:
    """
    EMA of trainable model parameters.

    The frozen Point-BERT/text-aligned projection buffers should not be averaged:
    they are either fixed pretrained weights or deterministic buffers. We only
    track parameters with ``requires_grad=True`` at construction time.

    The decay is warmed up as min(decay, (1 + n) / (10 + n)) after n updates,
    so short runs do not keep a large share of the initial weights.

    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        n = self.num_updates
        decay = min(self.decay, (1 + n) / (10 + n))
        self.num_updates += 1
        params = dict(model.named_parameters())
        for name, avg in self.shadow.items():
            p = params[name]
            avg.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def state_dict(self) -> Dict[str, object]:
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, object], device: torch.device | str):
        self.decay = float(state.get('decay', self.decay))
        self.num_updates = int(state.get('num_updates', 0))
        self.shadow = {
            k: v.to(device=device).detach().clone()
            for k, v in state['shadow'].items()
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        params = dict(model.named_parameters())
        for name, avg in self.shadow.items():
            if name in params:
                params[name].copy_(avg.to(params[name].device))

    def merged_model_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Return a full model state_dict with EMA trainable params overlaid.

        """
        state = model.state_dict()
        for name, avg in self.shadow.items():
            if name in state:
                state[name] = avg.detach().cpu()
        return state
