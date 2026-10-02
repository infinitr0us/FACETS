"""
Evaluation script for FACETS.

Usage
-----
    python -m facets.evaluate \\
        --run-dir ./runs/real3d_v1 \\
        --ckpt ckpt_last.pt \\
        --data-root /path/to/Real3D-AD

"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import build_test_dataset
from .models import (load_pointbert_from_checkpoint,
                     build_text_encoder_from_checkpoint,
                     ULIPADFramework)
from .models.framework import FrameworkConfig
from .models.pointbert_extended import load_ulip_projection
from .utils import score_test_sample, set_seed


# ---------------------------------------------------------------------------
# AUROC helper
# ---------------------------------------------------------------------------

def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Binary AUROC via the Mann-Whitney U statistic (handles ties correctly).

    """
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same length")
    if np.isnan(y_score).any():
        y_score = np.nan_to_num(y_score, nan=-1e9)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_ranks = np.bincount(inv.ravel(), weights=ranks, minlength=counts.size)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    sum_pos_ranks = ranks[y_true == 1].sum()
    U = sum_pos_ranks - pos.size * (pos.size + 1) / 2
    return float(U / (pos.size * neg.size))


# ---------------------------------------------------------------------------
# Evaluation entry
# ---------------------------------------------------------------------------

def evaluate(cfg: Config, ckpt_path: str,
             test_data_root: str = None
             ) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = cfg.device
    point_mode = cfg.infer.point_auc_mode
    if point_mode not in ('pooled', 'per_sample'):
        raise ValueError(f"point_auc_mode must be 'pooled' or 'per_sample', "
                         f"got {point_mode!r}")

    # --- Data ---
    data_root = test_data_root or cfg.data.data_root
    print(f"\n[eval] test data root: {data_root}")
    test_ds = build_test_dataset(cfg.data.dataset, data_root,
                                 num_points=cfg.data.num_points,
                                 classes=cfg.data.classes)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=False)

    # --- Models ---
    pointbert = load_pointbert_from_checkpoint(
        cfg.ulip_checkpoint, device=device,
        num_group=cfg.model.num_patches,
        group_size=cfg.model.patch_size,
    )
    d, d_mini = pointbert.trans_dim, pointbert.encoder_dim

    P_ulip = load_ulip_projection(cfg.ulip_checkpoint)
    if P_ulip.shape[-1] != 2 * d:
        P_ulip = P_ulip.t()
    d_bar = P_ulip.shape[0]

    fw_cfg = FrameworkConfig(
        d=d, d_mini=d_mini, d_bar=d_bar,
        d_e=cfg.model.d_e, k_gnn=cfg.model.k_gnn,
        num_patches=cfg.model.num_patches,
        patch_size=cfg.model.patch_size,
        eps_zscore=cfg.model.eps_zscore,
        attn_layer_indices=cfg.model.attn_layer_indices,
    )
    framework = ULIPADFramework(
        pointbert=pointbert, P_ulip=P_ulip.to(device),
        cfg=fw_cfg, freeze_pointbert=True,
    ).to(device)

    # Load head weights from checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    msd = framework.state_dict()
    src = ckpt['framework'] if 'framework' in ckpt else ckpt
    loaded = 0
    for k in msd:
        if k in src and msd[k].shape == src[k].shape:
            msd[k] = src[k]; loaded += 1
    framework.load_state_dict(msd, strict=False)
    print(f"[eval] loaded {loaded}/{len(msd)} params from {ckpt_path}")

    if getattr(cfg.train, 'eval_ema', True) and ckpt.get('ema') is not None:
        shadow = ckpt['ema'].get('shadow', {})
        params = dict(framework.named_parameters())
        ema_loaded = 0
        with torch.no_grad():
            for name, value in shadow.items():
                if name in params and params[name].shape == value.shape:
                    params[name].copy_(value.to(device))
                    ema_loaded += 1
        print(f"[eval] overlaid EMA trainable params: "
              f"{ema_loaded}/{len(shadow)}")

    # --- Text embeds per category ---
    text_embeds_path = os.path.join(os.path.dirname(ckpt_path),
                                    'cat_text_embeds.pt')
    if os.path.isfile(text_embeds_path):
        te = torch.load(text_embeds_path, map_location=device, weights_only=False)
        categories = te['categories']
        cat_embeds = te['embeds'].to(device)
        print(f"[eval] loaded pre-computed text embeddings for "
              f"{len(categories)} categories")
    else:
        print(f"[eval] pre-computed text embeddings not found; recomputing")
        from .train import build_per_category_text_embeds
        text_encoder = build_text_encoder_from_checkpoint(
            ulip_ckpt_path=cfg.ulip_checkpoint,
            open_clip_variant=cfg.text.open_clip_variant,
            open_clip_pretrained=cfg.text.open_clip_pretrained,
            device=device,
        )
        categories = test_ds.classes
        cat_embeds = build_per_category_text_embeds(
            text_encoder, categories,
            use_ensemble=cfg.text.use_prompt_ensemble,
            fine_grained_map=None, device=device,
        )
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    framework.eval()
    per_cls_obj: Dict[str, Tuple[List[float], List[int]]] = {}
    per_cls_point_auc: Dict[str, List[float]] = {}
    per_cls_points: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    n = len(test_ds)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            pc_sub   = batch['pc_sub'].to(device, non_blocking=True)   # (1, Np, 3)
            gt_xyz   = batch['gt_xyz'][0].to(device)                   # (M_gt, 3)
            gt_full  = batch['gt_full'][0].numpy()                     # (M_gt,)
            has_gt   = bool(batch['has_gt'][0])
            is_anom  = int(batch['is_anomalous'][0])
            category = batch['category'][0]
            sample_id = batch['sample_id'][0]

            if category not in cat_to_idx:
                continue
            t = cat_embeds[cat_to_idx[category]].unsqueeze(0)          # (1, d_bar)

            out = framework(pc_sub, text_embed=t, return_point=True)
            z_proj = out['z_final_proj']                                # (1, d_bar)
            s_obj = -(z_proj * t).sum(dim=-1).item()

            # Log object-level score
            per_cls_obj.setdefault(category, ([], []))
            per_cls_obj[category][0].append(s_obj)
            per_cls_obj[category][1].append(is_anom)

            # --- Point-level ---
            pooled = point_mode == 'pooled'
            if has_gt and (pooled or 0 < gt_full.sum() < gt_full.size):
                full_scores = score_test_sample(
                    s_point_norm=out['s_point_norm'][0],                # (N, K)
                    knn_idx=out['knn_idx'][0],                          # (N, K)
                    pc_sub=pc_sub[0],
                    pc_gt=gt_xyz,
                    k=cfg.infer.knn_interp_k,
                ).cpu().numpy()
                if pooled:
                    sc, lb = per_cls_points.setdefault(category, ([], []))
                    sc.append(full_scores.astype(np.float32))
                    lb.append(gt_full.astype(np.uint8))
                else:
                    auc_p = roc_auc_score_np(gt_full, full_scores)
                    if not np.isnan(auc_p):
                        per_cls_point_auc.setdefault(category, []).append(auc_p)

            if (bi + 1) % 50 == 0 or bi == 0:
                print(f"[eval] {bi+1}/{n}  {category}/{sample_id}  "
                      f"s_obj={s_obj:+.4f}  anom={is_anom}  has_gt={has_gt}")

    # --- Report ---
    p_label = 'point_auc' if point_mode == 'pooled' else 'point_auc(avg)'
    print("\n" + "=" * 70)
    print(f"{'category':<20}  {'obj_auc':>8}  {p_label:>14}  "
          f"{'N_test':>6}")
    print("-" * 70)
    obj_scores_all, obj_labels_all = [], []
    per_cls_obj_auc: Dict[str, float] = {}
    per_cls_point_auc_mean: Dict[str, float] = {}
    for cls, (scores, labels) in per_cls_obj.items():
        o = roc_auc_score_np(np.array(labels), np.array(scores))
        if point_mode == 'pooled':
            sc, lb = per_cls_points.pop(cls, ([], []))
            p = roc_auc_score_np(np.concatenate(lb), np.concatenate(sc)) \
                if lb else float('nan')
        else:
            pts = per_cls_point_auc.get(cls, [])
            p = float(np.mean(pts)) if pts else float('nan')
        per_cls_obj_auc[cls] = o
        per_cls_point_auc_mean[cls] = p
        obj_scores_all.extend(scores)
        obj_labels_all.extend(labels)
        print(f"{cls:<20}  {o:>8.4f}  {p:>14.4f}  {len(scores):>6}")
    print("-" * 70)
    obj_mean = float(np.nanmean(list(per_cls_obj_auc.values())))
    point_mean = float(np.nanmean(list(per_cls_point_auc_mean.values())))
    print(f"{'MEAN (per-category)':<20}  {obj_mean:>8.4f}  {point_mean:>14.4f}")
    pooled_obj = roc_auc_score_np(np.array(obj_labels_all),
                                  np.array(obj_scores_all))
    print(f"{'POOLED OBJECT AUROC':<20}  {pooled_obj:>8.4f}")
    print("=" * 70 + "\n")

    return {
        'point_auc_mode': point_mode,
        'per_cls_obj_auc': per_cls_obj_auc,
        'per_cls_point_auc': per_cls_point_auc_mean,
        'obj_mean': obj_mean,
        'point_mean': point_mean,
        'pooled_obj_auc': pooled_obj,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='FACETS evaluation')
    ap.add_argument('--run-dir', type=str, required=True,
                    help='Training run directory containing config.json')
    ap.add_argument('--ckpt', type=str, default='ckpt_last.pt')
    ap.add_argument('--data-root', type=str, default=None,
                    help='Override test data root.')
    ap.add_argument('--raw-weights', action='store_true',
                    help='Evaluate raw checkpoint weights instead of EMA.')
    ap.add_argument('--point-auc', choices=['pooled', 'per_sample'],
                    default=None,
                    help="Point-level AUROC: 'pooled' over all points of a "
                         "category (default) or 'per_sample' mean over "
                         "anomalous samples.")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, 'config.json')
    cfg = Config.load(cfg_path)
    if args.raw_weights:
        cfg.train.eval_ema = False
    if args.point_auc:
        cfg.infer.point_auc_mode = args.point_auc
    ckpt_path = os.path.join(args.run_dir, args.ckpt)

    results = evaluate(cfg, ckpt_path, test_data_root=args.data_root)
    out_path = os.path.join(args.run_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'point_auc_mode': results['point_auc_mode'],
            'obj_mean': results['obj_mean'],
            'point_mean': results['point_mean'],
            'pooled_obj_auc': results['pooled_obj_auc'],
            'per_cls_obj_auc': results['per_cls_obj_auc'],
            'per_cls_point_auc': results['per_cls_point_auc'],
        }, f, indent=2)
    print(f"[eval] results written to {out_path}")


if __name__ == '__main__':
    main()
