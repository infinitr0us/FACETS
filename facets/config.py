"""
Configurations for FACETS training and evaluation.

"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json


@dataclass
class ModelConfig:
    d_mini: int = 256
    d_e: int = 64
    k_gnn: int = 12
    num_patches: int = 64        # N
    patch_size: int = 32         # K
    eps_zscore: float = 1e-6
    attn_layer_indices: Optional[List[int]] = None    # None = all layers


@dataclass
class DataConfig:
    dataset: str = 'real3d-ad'           # or 'anomaly-shapenet'
    data_root: str = ''
    num_points: int = 8192
    classes: Optional[List[str]] = None  # None = auto-discover

    # Augmentation
    # 'z', 'x', 'y', 'so3', 'mixed', or None. 'mixed' samples SO(3) with
    # probability so3_prob and z-axis rotation otherwise.
    rotate_axis: str = 'mixed'
    so3_prob: float = 0.5
    scale_lo: float = 0.9
    scale_hi: float = 1.1

    # Effective batch: number of augmented views per raw sample per epoch
    num_augmented_views: int = 4


@dataclass
class TextConfig:
    use_prompt_ensemble: bool = False
    open_clip_variant: str = 'ViT-bigG-14'
    open_clip_pretrained: str = 'laion2b_s39b_b160k'
    fine_grained_captions_json: Optional[str] = None   # path to dict json


@dataclass
class TrainConfig:
    epochs: int = 200
    warmup_epochs: int = 5
    base_lr: float = 5e-4
    P_hat_lr: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    batch_size: int = 128               # global batch under DDP, local otherwise
    point_patch_sample: int = 0
    lambda_intra: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.995
    eval_ema: bool = True
    num_workers: int = 4
    amp: bool = True
    log_interval: int = 10              # steps
    eval_interval: int = 10             # epochs
    save_dir: str = './runs/default'


@dataclass
class InferConfig:
    # kNN interpolation for point scores
    knn_interp_k: int = 3

    # Score to use for object-level AUROC:
    object_score_mode: str = 'cosine'

    # Point-level AUROC: 'pooled' = one AUROC over all points of all test
    # samples of a category (Real3D-AD protocol); 'per_sample' = mean of
    # per-sample AUROCs over anomalous samples.
    point_auc_mode: str = 'pooled'


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    # Paths
    ulip_checkpoint: str = ''            # required
    device: str = 'cuda'

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            d = json.load(f)
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                if isinstance(getattr(c, k), (ModelConfig, DataConfig,
                                              TextConfig, TrainConfig,
                                              InferConfig)):
                    sub = getattr(c, k)
                    for sk, sv in v.items():
                        if hasattr(sub, sk):
                            setattr(sub, sk, sv)
                else:
                    setattr(c, k, v)
        return c
