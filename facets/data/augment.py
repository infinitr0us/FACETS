"""
Augmentations for FACETS.

"""
from __future__ import annotations

from typing import Tuple
import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def random_rotation_matrix(axis: str = 'z') -> np.ndarray:
    """
    Return a 3x3 rotation matrix sampled uniformly on a single axis.

    """
    if axis == 'so3':
        # Uniform random rotation (Shoemake, 1992).
        u1, u2, u3 = np.random.random(3)
        q = np.array([
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ])
        x, y, z, w = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)
        return R

    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def random_scale(pc: np.ndarray, lo: float = 0.9, hi: float = 1.1
                 ) -> np.ndarray:
    s = np.random.uniform(lo, hi)
    return pc * s


# ---------------------------------------------------------------------------
# Composed augmentor
# ---------------------------------------------------------------------------

class PointCloudAugmentor:
    """
    Lightweight stateless augmentor usable in a DataLoader worker.

    Args:
        rotate_axis: 'z', 'x', 'y', 'so3' (full 3D), or 'mixed'. Mixed uses
            SO(3) with probability ``so3_prob`` and z-axis rotation otherwise.
        scale_lo / scale_hi: range of isotropic scale factors.

    """

    def __init__(self,
                 rotate_axis: str = 'z',
                 so3_prob: float = 0.5,
                 scale_lo: float = 0.9, scale_hi: float = 1.1):
        self.rotate_axis = rotate_axis
        self.so3_prob = so3_prob
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation pipeline to (N, 3) float32 points.

        """
        # 1. Rotation
        if self.rotate_axis is not None:
            axis = self.rotate_axis
            if axis == 'mixed':
                axis = 'so3' if np.random.random() < self.so3_prob else 'z'
            R = random_rotation_matrix(axis)
            pc = pc @ R.T

        # 2. Scaling
        if self.scale_hi > self.scale_lo:
            pc = random_scale(pc, self.scale_lo, self.scale_hi)
        return pc.astype(np.float32, copy=False)
