"""
Training script for FACETS.

Usage
-----
    python -m facets.train \\
        --config configs/real3d_ad.json \\
        --ulip-checkpoint /path/to/ulip2_pointbert.pt \\
        --data-root /path/to/Real3D-AD \\
        --save-dir ./runs/real3d_v1

"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import Config
from .data import (build_train_dataset, PointCloudAugmentor, build_text_prompts)
from .losses import TotalLoss, LossWeights
from .models import (load_pointbert_from_checkpoint,
                     build_text_encoder_from_checkpoint,
                     ULIPADFramework)
from .models.framework import FrameworkConfig
from .models.pointbert_extended import load_ulip_projection
from .utils import (TrainableEMA, WarmupCosineLR, cleanup_distributed,
                    set_seed, setup_distributed)


# ---------------------------------------------------------------------------
# Text embedding pre-computation
# ---------------------------------------------------------------------------

def build_per_category_text_embeds(text_encoder,
                                   categories: List[str],
                                   use_ensemble: bool,
                                   fine_grained_map=None,
                                   device: str = 'cuda'
                                   ) -> torch.Tensor:
    """
    Encode every category's prompts and average -> (C, d_bar) normalized.

    """
    prompts = build_text_prompts(categories, use_ensemble=use_ensemble,
                                 fine_grained_map=fine_grained_map)
    embs = []
    for c in categories:
        raws = text_encoder.encode(prompts[c], normalize=True)   # (P, d_bar)
        avg = raws.mean(dim=0)
        avg = F.normalize(avg, dim=-1)
        embs.append(avg)
    return torch.stack(embs, dim=0).to(device)


def build_text_group_ids(categories: List[str],
                         use_ensemble: bool,
                         fine_grained_map=None) -> torch.Tensor:
    """
    Per-category index of its prompt set -> (C,). Categories with identical
    prompts (e.g. bottle0 / bottle1) share one text embedding and one index,
    which is the category label y used by the contrastive mask.

    """
    prompts = build_text_prompts(categories, use_ensemble=use_ensemble,
                                 fine_grained_map=fine_grained_map)
    keys = [tuple(sorted(prompts[c])) for c in categories]
    return torch.tensor([keys.index(k) for k in keys], dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    dist_ctx = setup_distributed(cfg.device)
    device = dist_ctx.device

    def log(*args, **kwargs):
        if dist_ctx.is_main:
            print(*args, **kwargs)

    if dist_ctx.is_main:
        os.makedirs(cfg.train.save_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.train.save_dir, 'config.json'))
    if dist_ctx.enabled:
        dist.barrier()
    set_seed(cfg.seed + dist_ctx.rank)
    if dist_ctx.enabled:
        log(f"[train] DDP enabled: world_size={dist_ctx.world_size}, "
            f"backend={dist_ctx.backend}, global_batch={cfg.train.batch_size}")

    # --- Data ---
    log("\n[train] building dataset")
    augmentor = PointCloudAugmentor(
        rotate_axis=cfg.data.rotate_axis,
        so3_prob=cfg.data.so3_prob,
        scale_lo=cfg.data.scale_lo,
        scale_hi=cfg.data.scale_hi,
    )
    train_ds = build_train_dataset(
        cfg.data.dataset, cfg.data.data_root,
        num_points=cfg.data.num_points,
        augmentor=augmentor,
        num_augmented_views=cfg.data.num_augmented_views,
        classes=cfg.data.classes,
    )
    categories = train_ds.classes
    if cfg.train.batch_size > 0:
        global_batch_size = cfg.train.batch_size
    else:
        global_batch_size = len(train_ds)
    if dist_ctx.enabled:
        if global_batch_size % dist_ctx.world_size != 0:
            raise ValueError(
                f"Global batch_size ({global_batch_size}) must be divisible "
                f"by DDP world_size ({dist_ctx.world_size}).")
        per_rank_batch_size = max(1, global_batch_size // dist_ctx.world_size)
        sampler = DistributedSampler(
            train_ds, num_replicas=dist_ctx.world_size, rank=dist_ctx.rank,
            shuffle=True, seed=cfg.seed, drop_last=False)
    else:
        per_rank_batch_size = global_batch_size
        sampler = None
    log(f"[train] per-rank batch size: {per_rank_batch_size}")

    loader = DataLoader(
        train_ds,
        batch_size=per_rank_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    # --- Models ---
    log("\n[train] loading Point-BERT")
    pointbert = load_pointbert_from_checkpoint(
        cfg.ulip_checkpoint, device=device,
        num_group=cfg.model.num_patches,
        group_size=cfg.model.patch_size,
        verbose=dist_ctx.is_main,
    )
    d = pointbert.trans_dim
    d_mini = pointbert.encoder_dim
    log(f"[train] d = {d}, d_mini = {d_mini}")

    log("[train] loading ULIP projection P")
    P_ulip = load_ulip_projection(cfg.ulip_checkpoint)

    # Determine d_bar from P_ulip's shape.
    if P_ulip.shape[-1] == 2 * d:
        d_bar = P_ulip.shape[0]
    else:
        d_bar = P_ulip.shape[1]    # (2d, d_bar) stored variant
        P_ulip = P_ulip.t()
    log(f"[train] d_bar = {d_bar}")

    log("[train] loading CLIP text encoder")
    text_encoder = build_text_encoder_from_checkpoint(
        ulip_ckpt_path=cfg.ulip_checkpoint,
        open_clip_variant=cfg.text.open_clip_variant,
        open_clip_pretrained=cfg.text.open_clip_pretrained,
        device=device,
        verbose=dist_ctx.is_main,
    )
    if text_encoder.embed_dim != d_bar:
        raise RuntimeError(
            f"Text encoder embed_dim ({text_encoder.embed_dim}) != "
            f"ULIP joint dim d_bar ({d_bar}). "
            f"The CLIP variant used by your ULIP-2 checkpoint must match. "
            f"Try setting open_clip_variant accordingly "
            f"(e.g. ViT-B/32 -> 512, ViT-L/14 -> 768, ViT-H/14 -> 1024, "
            f"ViT-bigG-14 -> 1280).")

    # --- Framework ---
    fw_cfg = FrameworkConfig(
        d=d, d_mini=d_mini, d_bar=d_bar,
        d_e=cfg.model.d_e, k_gnn=cfg.model.k_gnn,
        num_patches=cfg.model.num_patches,
        patch_size=cfg.model.patch_size,
        eps_zscore=cfg.model.eps_zscore,
        attn_layer_indices=cfg.model.attn_layer_indices,
    )
    framework = ULIPADFramework(
        pointbert=pointbert,
        P_ulip=P_ulip.to(device),
        cfg=fw_cfg,
        freeze_pointbert=True,
    ).to(device)
    n_train_params = sum(p.numel() for p in framework.parameters()
                         if p.requires_grad)
    n_frozen_params = sum(p.numel() for p in framework.parameters()
                          if not p.requires_grad)
    log(f"[train] trainable params: {n_train_params/1e6:.2f} M, "
        f"frozen params: {n_frozen_params/1e6:.2f} M")

    if dist_ctx.enabled:
        ddp_kwargs = {}
        if device.type == 'cuda':
            ddp_kwargs.update({
                'device_ids': [dist_ctx.local_rank],
                'output_device': dist_ctx.local_rank,
            })
        framework_ddp = DDP(
            framework,
            broadcast_buffers=False,
            find_unused_parameters=False,
            **ddp_kwargs,
        )
    else:
        framework_ddp = framework

    # --- Text embeddings ---
    fine_grained_map = None
    if cfg.text.fine_grained_captions_json:
        with open(cfg.text.fine_grained_captions_json, 'r') as f:
            fine_grained_map = json.load(f)
    cat_embeds = build_per_category_text_embeds(
        text_encoder, categories,
        use_ensemble=cfg.text.use_prompt_ensemble,
        fine_grained_map=fine_grained_map,
        device=device,
    )                                             # (C, d_bar)
    text_group = build_text_group_ids(
        categories, use_ensemble=cfg.text.use_prompt_ensemble,
        fine_grained_map=fine_grained_map,
    ).to(device)                                  # (C,)

    # The text tower (with the whole CLIP model) is only needed above.
    del text_encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Persist to disk for eval reuse
    if dist_ctx.is_main:
        torch.save({'categories': categories, 'embeds': cat_embeds.cpu()},
                   os.path.join(cfg.train.save_dir, 'cat_text_embeds.pt'))

    # --- Optimizer / scheduler ---
    param_groups = framework.trainable_param_groups(
        base_lr=cfg.train.base_lr,
        P_hat_lr=cfg.train.P_hat_lr,
        weight_decay=cfg.train.weight_decay,
    )
    optim = torch.optim.AdamW(param_groups)
    total_steps = cfg.train.epochs * max(1, len(loader))
    warmup_steps = cfg.train.warmup_epochs * max(1, len(loader))
    scheduler = WarmupCosineLR(optim, warmup_steps=warmup_steps,
                               total_steps=total_steps)

    total_loss_fn = TotalLoss(LossWeights(
        lambda_intra=cfg.train.lambda_intra,
        point_patch_sample=cfg.train.point_patch_sample,
    )).to(device)

    ema = TrainableEMA(framework, decay=cfg.train.ema_decay) \
        if cfg.train.use_ema else None
    if ema is not None:
        log(f"[train] EMA enabled for trainable head "
            f"(decay={cfg.train.ema_decay})")

    # --- AMP ---
    scaler = torch.amp.GradScaler(
        device.type, enabled=cfg.train.amp and device.type == 'cuda')

    # --- Training loop ---
    step = 0
    for epoch in range(cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        framework.train()

        # Keep Point-BERT BN/LN in eval mode regardless of framework.train()
        framework.pointbert.eval()
        for batch in loader:
            pts = batch['pc'].to(device, non_blocking=True)              # (B, Np, 3)
            label_idx = batch['label_idx'].to(device, non_blocking=True)  # (B,)
            text_batch = cat_embeds[label_idx]                            # (B, d_bar)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
                out = framework_ddp(pts, text_embed=text_batch, return_point=False)
                tau = out['tau']
                loss_out = total_loss_fn(out, text_batch, tau,
                                         labels=text_group[label_idx])
                loss = loss_out['loss']

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping (requires unscaling first when AMP is on)
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p for p in framework.parameters() if p.requires_grad],
                cfg.train.grad_clip_norm,
            )
            scale = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            # GradScaler skips the update on inf/NaN gradients (and lowers
            # the scale); only applied updates advance the EMA and the LR.
            if scaler.get_scale() >= scale:
                if ema is not None:
                    ema.update(framework)
                scheduler.step()
            step += 1

            if step % cfg.train.log_interval == 0:
                lr_head = optim.param_groups[1]['lr']
                lr_phat = optim.param_groups[0]['lr']
                log(f"[epoch {epoch+1}/{cfg.train.epochs} step {step}] "
                    f"loss={loss.item():.4f}  "
                    f"3d2t={loss_out['l_3d2t'].item():.3f}  "
                    f"t23d={loss_out['l_t23d'].item():.3f}  "
                    f"pp={loss_out['l_point_patch'].item():.3f}  "
                    f"intra={loss_out['l_intra'].item():.3f}  "
                    f"tau={loss_out['tau'].item():.3f}  "
                    f"lr(head/Phat)={lr_head:.2e}/{lr_phat:.2e}")

        # End of epoch: checkpoint
        if dist_ctx.is_main and ((epoch + 1) % 10 == 0 or
                                 (epoch + 1) == cfg.train.epochs):
            ckpt = {
                'epoch': epoch + 1,
                'framework': framework.state_dict(),
                'ema': ema.state_dict() if ema is not None else None,
                'config': cfg.__dict__,
                'categories': categories,
            }
            torch.save(ckpt, os.path.join(cfg.train.save_dir,
                                           f'ckpt_ep{epoch+1}.pt'))
            torch.save(ckpt, os.path.join(cfg.train.save_dir, 'ckpt_last.pt'))
            log(f"[train] saved checkpoint at epoch {epoch+1}")

    log("[train] done")
    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli_parser():
    ap = argparse.ArgumentParser(description='FACETS training')
    ap.add_argument('--config', type=str, default='',
                    help='Path to JSON config (optional).')
    ap.add_argument('--ulip-checkpoint', type=str, default='',
                    help='Path to ULIP-2 Point-BERT checkpoint.')
    ap.add_argument('--data-root', type=str, default='',
                    help='Root directory of the dataset.')
    ap.add_argument('--dataset', type=str, default=None,
                    choices=['real3d-ad', 'anomaly-shapenet'])
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--base-lr', type=float, default=None,
                    help='Head learning rate; P_hat uses base_lr / 10.')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--open-clip-variant', type=str, default=None)
    ap.add_argument('--prompt-ensemble', action='store_true',
                    help='Average text embeddings over the prompt ensemble.')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--ema-decay', type=float, default=None)
    return ap


def main():
    args = _build_cli_parser().parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    if args.ulip_checkpoint:
        cfg.ulip_checkpoint = args.ulip_checkpoint
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.dataset:
        cfg.data.dataset = args.dataset
    if args.save_dir:
        cfg.train.save_dir = args.save_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.base_lr is not None:
        cfg.train.base_lr = args.base_lr
        cfg.train.P_hat_lr = args.base_lr / 10
    if args.open_clip_variant:
        cfg.text.open_clip_variant = args.open_clip_variant
    if args.prompt_ensemble:
        cfg.text.use_prompt_ensemble = True
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_amp:
        cfg.train.amp = False
    if args.no_ema:
        cfg.train.use_ema = False
        cfg.train.eval_ema = False
    if args.ema_decay is not None:
        cfg.train.ema_decay = args.ema_decay

    assert cfg.ulip_checkpoint, 'Must provide --ulip-checkpoint'
    assert cfg.data.data_root, 'Must provide --data-root'
    train(cfg)


if __name__ == '__main__':
    main()
