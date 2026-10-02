"""
Extended Point-BERT backbone for FACETS.

This wraps the ULIP-2 Point-BERT so that, in a single forward pass, we can
extract:

  * CLS embedding at the last layer (z_0^L)
  * All patch embeddings at the last layer ({z_i^L}_{i=1...N})
  * Attention weights from every layer/head (for computing beta)
  * Pre-max-pooling Mini-PointNet features ({v_j}_{j in patch i})
  * Patch center coordinates c_i (for GNN graph construction)

"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# FPS + kNN
# -----------------------------------------------------------------------------

@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    FPS in pure PyTorch.

    Args:
        xyz: (B, N, 3) coordinates.
        npoint: number of centers to sample.
    Returns:
        centroids: (B, npoint) long indices.

    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    # Deterministic-ish start (first point) gives stable behavior across augmented views
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[batch_idx, farthest, :].unsqueeze(1)      # (B, 1, 3)
        dist = torch.sum((xyz - c) ** 2, dim=-1)          # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


@torch.no_grad()
def knn_query(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    k-NN in pure PyTorch. Returns (B, S, k) indices into xyz.

    """
    d = torch.cdist(new_xyz, xyz, p=2.0)                  # (B, S, N)
    _, idx = torch.topk(d, k, dim=-1, largest=False, sorted=False)
    return idx


class PointCloudGroupDivider(nn.Module):
    """
    Tokenize a point cloud into local patches via FPS + kNN.

    Returns:
        neighborhood: (B, G, K, 3)  — centred neighbors per patch
        center:       (B, G, 3)     — patch center coordinates
        knn_idx:      (B, G, K)     — indices of patch points into input cloud
                                      (useful for projecting per-point scores
                                       back to the original points)

    """

    def __init__(self, num_group: int = 64, group_size: int = 32):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.num_group)  # (B, G)

        b_idx = torch.arange(B, device=xyz.device).unsqueeze(1).expand(-1, self.num_group)
        center = xyz[b_idx, fps_idx]                          # (B, G, 3)

        knn_idx = knn_query(self.group_size, xyz, center)     # (B, G, K)
        b_idx_knn = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(
            -1, self.num_group, self.group_size)
        neighborhood = xyz[b_idx_knn, knn_idx]                # (B, G, K, 3)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, knn_idx


# -----------------------------------------------------------------------------
# Mini-PointNet
# -----------------------------------------------------------------------------

class MiniPointNetEncoder(nn.Module):
    """
    Mini-PointNet.

    """

    def __init__(self, encoder_dim: int = 256):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, encoder_dim, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(encoder_dim * 2, encoder_dim * 2, 1),
            nn.BatchNorm1d(encoder_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(encoder_dim * 2, encoder_dim, 1),
        )

    def forward(self, patches: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (B, G, K, 3) centred patch points.

        Returns:
            patch_tokens: (B, G, d_mini)  — after final max-pool (Point-BERT token)
            point_feats:  (B, G, K, d_mini) — per-point features before final max-pool

        """
        B, G, K, _ = patches.shape
        x = patches.reshape(B * G, K, 3).transpose(1, 2)           # (BG, 3, K)
        feat = self.first_conv(x)                                  # (BG, d_mini, K)
        feat_glob = feat.max(dim=2, keepdim=True).values           # (BG, d_mini, 1)
        feat_cat = torch.cat([feat_glob.expand(-1, -1, K), feat], dim=1)
        feat2 = self.second_conv(feat_cat)                         # (BG, d_mini, K)
        patch_tokens = feat2.max(dim=2).values                     # (BG, d_mini)

        # Reshape back to (B, G, ...)
        patch_tokens = patch_tokens.reshape(B, G, self.encoder_dim)
        point_feats = feat2.transpose(1, 2).reshape(
            B, G, K, self.encoder_dim).contiguous()
        return patch_tokens, point_feats


# -----------------------------------------------------------------------------
# Transformer block with attention capture
# -----------------------------------------------------------------------------

class MultiHeadAttentionWithCapture(nn.Module):
    """
    MHSA that caches the softmax attention matrix.

    We keep gradient detachment OPTIONAL because training the backbone
    requires live gradients, but during feature-extraction (backbone frozen)
    we detach to save memory. See ``enable_grad_through_attn``.

    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_weights: Optional[torch.Tensor] = None

        # If True, attn_weights will keep the autograd graph; else detached
        # for memory efficiency when backbone is frozen.
        self.keep_attn_grad: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Cache for beta computation
        self.attn_weights = attn if self.keep_attn_grad else attn.detach()
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm block; Point-BERT adds pos embed to the input of EVERY block.

    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithCapture(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + pos
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Extended Point-BERT
# -----------------------------------------------------------------------------

class ExtendedPointBERT(nn.Module):
    """
    Point-BERT that exposes everything FACETS needs.

    On ``forward(pts)``, returns a dict with keys:
        cls:          (B, d)        z_0^L
        patches:      (B, G, d)     {z_i^L}_{i=1..N}
        concat_f:     (B, 2d)       standard Point-BERT output [CLS; max-pool]
        centers:      (B, G, 3)     patch centers c_i
        patch_tokens: (B, G, d_mini) patch tokens from Mini-PointNet (pre reduce_dim)
        point_feats:  (B, G, K, d_mini) v_j (pre-max-pool per-point features)
        knn_idx:      (B, G, K)     patch-point indices into input pts

    """

    def __init__(self, num_group: int = 64, group_size: int = 32,
                 encoder_dim: int = 256, trans_dim: int = 384,
                 depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0.):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.trans_dim = trans_dim
        self.encoder_dim = encoder_dim
        self.depth = depth
        self.num_heads = num_heads

        self.group_divider = PointCloudGroupDivider(num_group, group_size)
        self.encoder = MiniPointNetEncoder(encoder_dim)
        self.reduce_dim = nn.Linear(encoder_dim, trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, trans_dim))

        # 3D -> pos-embed projection.
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(trans_dim, num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop=drop_rate,
                             attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(trans_dim)

        # Init CLS token/pos
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, pts: torch.Tensor, keep_attn_grad: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pts: (B, N_p, 3)
            keep_attn_grad: if True, attention matrices are NOT detached
                (needed only if you want gradients to flow through beta).

        Returns:
            dict with the keys documented in the class docstring.

        """
        # Keep/reset attn-grad flag on every block
        for blk in self.blocks:
            blk.attn.keep_attn_grad = keep_attn_grad

        neighborhood, center, knn_idx = self.group_divider(pts)

        # Mini-PointNet with pre-max-pool capture
        patch_tokens_raw, point_feats = self.encoder(neighborhood)

        # Project into transformer dim
        group_input_tokens = self.reduce_dim(patch_tokens_raw)

        B = pts.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm(x)
        cls = x[:, 0]
        patches = x[:, 1:]

        concat_f = torch.cat([cls, patches.max(dim=1).values], dim=-1)

        return {
            'cls': cls,                                     # (B, d)
            'patches': patches,                             # (B, G, d)
            'concat_f': concat_f,                           # (B, 2d)
            'centers': center,                              # (B, G, 3)
            'patch_tokens_raw': patch_tokens_raw,           # (B, G, d_mini)
            'point_feats': point_feats,                     # (B, G, K, d_mini)
            'knn_idx': knn_idx,                             # (B, G, K)
        }

    # ------------------------------------------------------------------
    def get_beta(self, include_cls_self: bool = False,
                 layer_indices: Optional[List[int]] = None
                 ) -> torch.Tensor:
        """
        Compute beta from cached attention weights.

        Args:
            include_cls_self: if True, returns shape (B, G+1) including
                the CLS self-attention beta_0; else (B, G).
            layer_indices: subset of block indices to average over. None = all.

        """
        L_total = len(self.blocks)
        if layer_indices is None:
            layer_indices = list(range(L_total))
        H = self.num_heads

        total = None
        for l in layer_indices:
            attn = self.blocks[l].attn.attn_weights  # (B, H, G+1, G+1)
            if attn is None:
                raise RuntimeError("No attention weights cached; call forward first.")

            # CLS -> everything: attn[:, :, 0, :]
            cls_to_all = attn[:, :, 0, :]                    # (B, H, G+1)
            head_sum = cls_to_all.sum(dim=1)                 # (B, G+1)
            total = head_sum if total is None else total + head_sum

        beta_all = total / (len(layer_indices) * H)          # (B, G+1)
        if include_cls_self:
            return beta_all
        return beta_all[:, 1:]                                # (B, G)


# =============================================================================
# Checkpoint loading
# =============================================================================

def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt
    else:
        sd = ckpt
    return {k.replace('module.', ''): v for k, v in sd.items()}


def load_pointbert_from_checkpoint(checkpoint_path: str,
                                   device: str = 'cuda',
                                   num_group: int = 64,
                                   group_size: int = 32,
                                   verbose: bool = True
                                   ) -> ExtendedPointBERT:
    """
    Load ULIP-2 Point-BERT weights into our Extended Point-BERT.

    """
    sd = _load_state_dict(checkpoint_path)

    # Detect encoder prefix (ULIP / ULIP-2 naming variations)
    prefix = ''
    for c in ['point_encoder.', 'pc_encoder.', 'encoder_3d.']:
        if any(k.startswith(c) for k in sd):
            prefix = c
            break
    if verbose:
        print(f"[ExtendedPointBERT] encoder prefix = '{prefix}'")

    pb = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    # Normalize double-nested blocks
    pb = {k.replace('blocks.blocks.', 'blocks.'): v for k, v in pb.items()}

    # The ULIP-2 ViT-G Point-BERT takes xyzrgb and is fed rgb = 0.4 for
    # colourless clouds; for the first 1x1 conv this folds into the bias.
    w_key, b_key = 'encoder.first_conv.0.weight', 'encoder.first_conv.0.bias'
    if w_key in pb and pb[w_key].shape[1] == 6:
        w = pb[w_key]
        pb[b_key] = pb[b_key] + 0.4 * w[:, 3:].sum(dim=(1, 2))
        pb[w_key] = w[:, :3].contiguous()
        if verbose:
            print("[ExtendedPointBERT] xyzrgb checkpoint: folded rgb = 0.4 "
                  "into the first conv bias")

    # Infer architecture
    trans_dim = pb.get('cls_token', torch.zeros(1, 1, 384)).shape[-1]
    encoder_dim = 256
    for k, v in pb.items():
        if 'reduce_dim' in k and 'weight' in k:
            encoder_dim = v.shape[1]; break
    blk_ids = set()
    for k in pb:
        if k.startswith('blocks.'):
            try: blk_ids.add(int(k.split('.')[1]))
            except: pass
    depth = (max(blk_ids) + 1) if blk_ids else 12
    num_heads = trans_dim // 64
    qkv_bias = 'blocks.0.attn.qkv.bias' in pb

    if verbose:
        print(f"[ExtendedPointBERT] trans_dim={trans_dim}  encoder_dim={encoder_dim}  "
              f"depth={depth}  num_heads={num_heads}  qkv_bias={qkv_bias}")

    model = ExtendedPointBERT(
        num_group=num_group, group_size=group_size,
        encoder_dim=encoder_dim, trans_dim=trans_dim,
        depth=depth, num_heads=num_heads, qkv_bias=qkv_bias,
    )
    msd = model.state_dict()
    loaded, skipped = 0, []
    for k in msd:
        if k in pb and msd[k].shape == pb[k].shape:
            msd[k] = pb[k]; loaded += 1
        elif not k.endswith('num_batches_tracked'):
            skipped.append(k)
    unused = sorted(set(pb) - set(msd))
    if skipped or unused:
        raise RuntimeError(
            f"Point-BERT checkpoint {checkpoint_path} does not match the "
            f"inferred model: missing/mismatched {skipped}, unused {unused}")
    model.load_state_dict(msd, strict=False)
    if verbose:
        print(f"[ExtendedPointBERT] loaded {loaded}/{len(msd)} params")

    return model.to(device).eval()


# =============================================================================
# Helper: extract ULIP-2 projection matrix P in R^{d_bar x 2d}
# =============================================================================

def load_ulip_projection(checkpoint_path: str) -> torch.Tensor:
    """
    Load P from ULIP-2 checkpoint.

    """
    sd = _load_state_dict(checkpoint_path)
    candidates = ['pc_projection', 'point_encoder.pc_projection',
                  'point_encoder.projection', 'point_encoder.fc']
    for name in candidates:
        if name in sd:
            P = sd[name]
            return P
    raise KeyError(
        f"No projection matrix found in checkpoint. Tried: {candidates}.\n"
        f"Available top-level keys (first 40): "
        f"{list(sd.keys())[:40]}"
    )
