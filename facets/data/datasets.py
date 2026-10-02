"""
Dataset loaders for FACETS

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# ===========================================================================
# File I/O
# ===========================================================================

def _load_single_point_cloud(fpath: str) -> Optional[np.ndarray]:
    """
    Return an (N, 3) float32 array, or None on failure.

    """
    ext = os.path.splitext(fpath)[1].lower()
    try:
        if ext == '.npy':
            pc = np.load(fpath)
            return pc[:, :3].astype(np.float32) if pc.ndim == 2 and pc.shape[1] >= 3 else None
        if ext == '.pcd':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(fpath)
                return np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                return _parse_pcd_manual(fpath)
        if ext == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(fpath)
                return np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                try:
                    from plyfile import PlyData
                    ply = PlyData.read(fpath)
                    return np.stack([ply['vertex']['x'], ply['vertex']['y'],
                                     ply['vertex']['z']], axis=1).astype(np.float32)
                except ImportError:
                    return _parse_ply_manual(fpath)
        if ext == '.obj':
            return _parse_obj_vertices(fpath)
        if ext in ('.xyz', '.pts'):
            pc = np.loadtxt(fpath, dtype=np.float32)
            return pc[:, :3] if pc.ndim == 2 and pc.shape[1] >= 3 else None
    except Exception as e:
        print(f"[dataset] failed to load {fpath}: {e}")
    return None


def _parse_pcd_manual(fpath: str) -> Optional[np.ndarray]:
    with open(fpath, 'rb') as f:
        header_bytes = []
        while True:
            line = f.readline()
            if not line:
                return None
            header_bytes.append(line)
            if line.startswith(b'DATA'):
                data_offset = f.tell()
                break
        payload = f.read()

    header = [ln.decode('ascii', errors='ignore').strip() for ln in header_bytes]
    meta = {}
    for line in header:
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        meta[parts[0].upper()] = parts[1:]

    fields = meta.get('FIELDS', [])
    sizes = [int(x) for x in meta.get('SIZE', [])]
    types = meta.get('TYPE', [])
    counts = [int(x) for x in meta.get('COUNT', ['1'] * len(fields))]
    npts = int(meta.get('POINTS', meta.get('WIDTH', ['0']))[0])
    data_kind = meta.get('DATA', ['ascii'])[0].lower()
    if not fields or npts <= 0:
        return None

    if data_kind == 'ascii':
        text = payload.decode('ascii', errors='ignore')
        arr = np.loadtxt(text.splitlines(), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        field_to_col = {name: i for i, name in enumerate(fields)}
        return arr[:, [field_to_col['x'], field_to_col['y'], field_to_col['z']]].astype(np.float32)

    if data_kind != 'binary':
        return None

    dtype_fields = []
    for name, typ, size, count in zip(fields, types, sizes, counts):
        if typ == 'F' and size == 4:
            dt = '<f4'
        elif typ == 'F' and size == 8:
            dt = '<f8'
        elif typ == 'I' and size == 1:
            dt = '<i1'
        elif typ == 'I' and size == 2:
            dt = '<i2'
        elif typ == 'I' and size == 4:
            dt = '<i4'
        elif typ == 'U' and size == 1:
            dt = '<u1'
        elif typ == 'U' and size == 2:
            dt = '<u2'
        elif typ == 'U' and size == 4:
            dt = '<u4'
        else:
            return None
        dtype_fields.append((name, dt) if count == 1 else (name, dt, (count,)))

    arr = np.frombuffer(payload, dtype=np.dtype(dtype_fields), count=npts)
    if not all(k in arr.dtype.names for k in ('x', 'y', 'z')):
        return None
    return np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)


def _parse_ply_manual(fpath: str) -> Optional[np.ndarray]:
    with open(fpath, 'rb') as f:
        header_bytes = []
        while True:
            line = f.readline()
            if not line:
                return None
            header_bytes.append(line)
            if line.strip() == b'end_header':
                break
        payload = f.read()

    header = [ln.decode('ascii', errors='ignore').strip() for ln in header_bytes]
    if not header or header[0] != 'ply':
        return None
    fmt_line = next((ln for ln in header if ln.startswith('format ')), '')
    if 'binary_little_endian' not in fmt_line and 'ascii' not in fmt_line:
        return None

    vertex_count = 0
    vertex_props = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == 'element':
            in_vertex = parts[1] == 'vertex'
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) >= 3 and parts[0] == 'property':
            if parts[1] == 'list':
                continue
            vertex_props.append((parts[2], parts[1]))

    if vertex_count <= 0 or not vertex_props:
        return None

    if 'ascii' in fmt_line:
        text = payload.decode('ascii', errors='ignore')
        arr = np.loadtxt(text.splitlines()[:vertex_count], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        prop_to_col = {name: i for i, (name, _) in enumerate(vertex_props)}
        return arr[:, [prop_to_col['x'], prop_to_col['y'], prop_to_col['z']]].astype(np.float32)

    type_map = {
        'char': '<i1', 'int8': '<i1',
        'uchar': '<u1', 'uint8': '<u1',
        'short': '<i2', 'int16': '<i2',
        'ushort': '<u2', 'uint16': '<u2',
        'int': '<i4', 'int32': '<i4',
        'uint': '<u4', 'uint32': '<u4',
        'float': '<f4', 'float32': '<f4',
        'double': '<f8', 'float64': '<f8',
    }
    try:
        dtype = np.dtype([(name, type_map[typ]) for name, typ in vertex_props])
    except KeyError:
        return None
    arr = np.frombuffer(payload, dtype=dtype, count=vertex_count)
    if not all(k in arr.dtype.names for k in ('x', 'y', 'z')):
        return None
    return np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)


def _parse_obj_vertices(fpath: str) -> Optional[np.ndarray]:
    vs = []
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                p = line.strip().split()
                if len(p) >= 4:
                    vs.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(vs, dtype=np.float32) if vs else None


def _subsample(pc: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (subsampled_pc, index_into_original).

    """
    m = pc.shape[0]
    if m >= n:
        idx = np.random.choice(m, n, replace=False)
    else:
        extra = np.random.choice(m, n - m, replace=True)
        idx = np.concatenate([np.arange(m), extra], axis=0)
    return pc[idx], idx


def _unit_sphere(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize to unit sphere; return (pc_norm, centroid, scale).

    """
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    scale = float(np.sqrt((pc ** 2).sum(axis=1).max()))
    if scale > 0:
        pc = pc / scale
    return pc.astype(np.float32, copy=False), centroid.astype(np.float32), scale


# ===========================================================================
# GT parsing
# ===========================================================================

def _load_gt_for_test_sample(gt_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load per-point GT labels and optional GT coordinates.

    """
    ext = os.path.splitext(gt_path)[1].lower()
    try:
        if ext == '.npy':
            arr = np.load(gt_path)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                xyz = arr[:, :3].astype(np.float32)
                labels = (arr[:, -1] > 0).astype(np.int64)
                return labels, xyz
            return arr.astype(np.int64).flatten(), None
        if ext in ('.txt', '.xyz', '.pts'):
            with open(gt_path, 'r') as f:
                first = f.readline()
            delimiter = ',' if ',' in first else None
            data = np.loadtxt(gt_path, dtype=np.float32, delimiter=delimiter)
            if data.ndim == 1:
                return data.astype(np.int64), None
            if data.shape[1] >= 4:
                xyz = data[:, :3].astype(np.float32)
                labels = (data[:, -1] > 0).astype(np.int64)
                return labels, xyz
            return (data[:, -1] > 0).astype(np.int64), None
    except Exception as e:
        print(f"[dataset] GT load failed for {gt_path}: {e}")
    return None, None


def _find_gt_file(cls_root: str, sample_stem: str) -> Optional[str]:
    """
    Locate the GT file corresponding to a given test sample.

    """
    for gt_dir in ['gt', 'GT', 'ground_truth', 'test_gt']:
        d = os.path.join(cls_root, gt_dir)
        if not os.path.isdir(d):
            continue
        for ext in ['.txt', '.npy', '.pcd']:
            p = os.path.join(d, sample_stem + ext)
            if os.path.isfile(p):
                return p
    return None


def _looks_normal_sample(sample_stem_or_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(sample_stem_or_path))[0].lower()
    return 'good' in stem or 'positive' in stem


# ===========================================================================
# Train dataset
# ===========================================================================

@dataclass
class TrainSample:
    pc: np.ndarray
    category: str
    label_idx: int
    sample_id: str


class _BaseTrainDataset(Dataset):
    """
    Training dataset.

    """

    def __init__(self,
                 data_root: str,
                 classes: List[str],
                 num_points: int = 8192,
                 augmentor: Optional[Callable] = None,
                 num_augmented_views: int = 1,
                 train_subdirs: Tuple[str, ...] = ('train', 'train/good',
                                                   'training')):
        super().__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.augmentor = augmentor
        self.num_augmented_views = max(1, num_augmented_views)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: List[TrainSample] = []
        for cls in classes:
            d = self._find_train_dir(cls, train_subdirs)
            if d is None:
                print(f"[dataset] class '{cls}' has no train dir; skipping")
                continue
            files = sorted([f for f in os.listdir(d)
                            if f.lower().endswith(('.npy', '.pcd', '.ply',
                                                   '.xyz', '.pts', '.obj'))])
            if not files:
                print(f"[dataset] no files under {d}; skipping")
                continue
            for fn in files:
                pc = _load_single_point_cloud(os.path.join(d, fn))
                if pc is None:
                    continue
                pc, _, _ = _unit_sphere(pc)
                self.samples.append(TrainSample(
                    pc=pc, category=cls,
                    label_idx=self.class_to_idx[cls],
                    sample_id=os.path.splitext(fn)[0]))
        print(f"[dataset] loaded {len(self.samples)} training samples "
              f"across {len(classes)} classes (raw, unaugmented)")

    def _find_train_dir(self, cls: str,
                        candidates: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            p = os.path.join(self.data_root, cls, c)
            if os.path.isdir(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples) * self.num_augmented_views

    def __getitem__(self, idx: int):
        s = self.samples[idx % len(self.samples)]
        pc = s.pc
        if self.augmentor is not None:
            pc = self.augmentor(pc)

        # Subsample to num_points
        pc_sub, _ = _subsample(pc, self.num_points)
        return {
            'pc': torch.from_numpy(pc_sub).float(),   # (Np, 3)
            'label_idx': s.label_idx,
            'category': s.category,
            'sample_id': s.sample_id,
        }


# ===========================================================================
# Test dataset
# ===========================================================================

@dataclass
class TestSample:
    pc_path: str
    gt_path: Optional[str]
    is_anomalous: int
    category: str
    label_idx: int
    sample_id: str


class _BaseTestDataset(Dataset):
    """
    Test dataset: loads each test cloud and its per-point GT.

    """

    def __init__(self,
                 data_root: str,
                 classes: List[str],
                 num_points: int = 8192,
                 test_subdirs: Tuple[str, ...] = ('test', 'test/bad',
                                                   'testing')):
        super().__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[TestSample] = []

        for cls in classes:
            test_dir = self._find_test_dir(cls, test_subdirs)
            if test_dir is None:
                print(f"[dataset] class '{cls}' has no test dir; skipping")
                continue

            # Search recursively, as some datasets put different defect
            # types in subdirectories.
            test_files = []
            for root, _, fns in os.walk(test_dir):
                # Skip GT directories if they happen to sit under test_dir
                if os.path.basename(root).lower() in ('gt', 'ground_truth', 'test_gt'):
                    continue
                for f in fns:
                    # Template files are not test samples (Real3D-AD protocol).
                    if 'temp' in f.lower():
                        continue
                    if f.lower().endswith(('.npy', '.pcd', '.ply',
                                            '.xyz', '.pts', '.obj')):
                        test_files.append(os.path.join(root, f))
            test_files.sort()
            if not test_files:
                print(f"[dataset] no test files under {test_dir}; skipping")
                continue

            cls_root = os.path.join(self.data_root, cls)
            for fp in test_files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                # Labels follow the file name; GT is read for anomalous
                # samples only, normal samples get an all-zero mask.
                is_anom = int(not _looks_normal_sample(stem))
                gt_path = _find_gt_file(cls_root, stem) if is_anom else None
                self.samples.append(TestSample(
                    pc_path=fp,
                    gt_path=gt_path,
                    is_anomalous=is_anom,
                    category=cls,
                    label_idx=self.class_to_idx[cls],
                    sample_id=stem,
                ))
        n_anom = sum(s.is_anomalous for s in self.samples)
        print(f"[dataset] loaded {len(self.samples)} test samples "
              f"({n_anom} anomalous) across {len(classes)} classes")

    def _find_test_dir(self, cls: str,
                       candidates: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            p = os.path.join(self.data_root, cls, c)
            if os.path.isdir(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pc = _load_single_point_cloud(s.pc_path)
        if pc is None:
            raise RuntimeError(f"Failed to load point cloud: {s.pc_path}")
        pc_norm, centroid, scale = _unit_sphere(pc)
        if s.gt_path:
            gt, gt_xyz = _load_gt_for_test_sample(s.gt_path)
        elif not s.is_anomalous:
            gt, gt_xyz = np.zeros(pc_norm.shape[0], dtype=np.int64), None
        else:
            gt, gt_xyz = None, None
        if gt is not None:
            if gt_xyz is not None:
                if gt.shape[0] != gt_xyz.shape[0]:
                    print(f"[dataset] GT label/xyz mismatch for {s.sample_id} "
                          f"({gt.shape[0]} vs {gt_xyz.shape[0]}); dropping GT")
                    gt, gt_xyz = None, None
                else:
                    gt_xyz = gt_xyz - centroid.reshape(1, 3)
                    if scale > 0:
                        gt_xyz = gt_xyz / scale
                    gt_xyz = gt_xyz.astype(np.float32, copy=False)
            elif gt.shape[0] == pc_norm.shape[0]:
                gt_xyz = pc_norm
            else:
                print(f"[dataset] GT length mismatch for {s.sample_id} "
                      f"({gt.shape[0]} vs {pc_norm.shape[0]}); dropping GT")
                gt, gt_xyz = None, None

        pc_sub, sub_idx = _subsample(pc_norm, self.num_points)
        return {
            'pc_sub': torch.from_numpy(pc_sub).float(),
            'pc_full': torch.from_numpy(pc_norm).float(),
            'sub_idx': torch.from_numpy(sub_idx).long(),
            'gt_full': torch.from_numpy(gt).long() if gt is not None
                       else torch.zeros(pc_norm.shape[0], dtype=torch.long),
            'gt_xyz': torch.from_numpy(gt_xyz).float() if gt_xyz is not None
                      else torch.from_numpy(pc_norm).float(),
            'has_gt': gt is not None,
            'is_anomalous': s.is_anomalous,
            'label_idx': s.label_idx,
            'category': s.category,
            'sample_id': s.sample_id,
        }


# ===========================================================================
# Public class names and factory functions
# ===========================================================================

REAL3DAD_CLASSES = [
    'airplane', 'car', 'candybar', 'chicken', 'diamond', 'duck',
    'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees',
]

ANOMALYSHAPENET_CLASSES = [
    'ashtray0', 'bag0', 'basket0', 'bottle0', 'bottle1', 'bowl0',
    'bucket0', 'cap0', 'car0', 'clock0', 'cup0', 'cup1',
    'dishwasher0', 'eraser0', 'eyeglasses0', 'flowerpot0',
    'fork0', 'headphones0', 'helmet0', 'jar0', 'kettle0',
    'keyboard0', 'knife0', 'lamp0', 'laptop0', 'lighter0',
    'microphone0', 'monitor0', 'mouse0', 'mug0', 'pillow0',
    'plug0', 'pot0', 'scissors0', 'shelf0', 'sink0',
    'suitcase0', 'tablespoon0', 'teapot0', 'toothbrush0',
    'trashcan0', 'vase0',
]


def _discover_classes(data_root: str,
                      fallback: List[str]) -> List[str]:
    if not os.path.isdir(data_root):
        return fallback
    disc = sorted([d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))
                   and not d.startswith('.')
                   and d.lower() not in ('new', '__pycache__')])
    return disc if disc else fallback


def _resolve_dataset_root(data_root: str, kind: str) -> str:
    """
    Accept either the dataset root or the point-cloud subroot used locally.

    """
    if not os.path.isdir(data_root):
        return data_root
    candidates = []
    if kind == 'real3d':
        candidates = ['PCD', 'PLY']
    elif kind == 'shapenet':
        candidates = ['pcd', 'PCD']
    for c in candidates:
        p = os.path.join(data_root, c)
        if os.path.isdir(p):
            return p
    return data_root


def Real3DADTrainDataset(data_root: str, num_points: int = 8192,
                         augmentor=None, num_augmented_views: int = 1,
                         classes: Optional[List[str]] = None
                         ) -> _BaseTrainDataset:
    data_root = _resolve_dataset_root(data_root, 'real3d')
    if classes is None:
        classes = _discover_classes(data_root, REAL3DAD_CLASSES)
    return _BaseTrainDataset(data_root, classes, num_points,
                             augmentor, num_augmented_views)


def Real3DADTestDataset(data_root: str, num_points: int = 8192,
                        classes: Optional[List[str]] = None
                        ) -> _BaseTestDataset:
    data_root = _resolve_dataset_root(data_root, 'real3d')
    if classes is None:
        classes = _discover_classes(data_root, REAL3DAD_CLASSES)
    return _BaseTestDataset(data_root, classes, num_points)


def AnomalyShapeNetTrainDataset(data_root: str, num_points: int = 8192,
                                augmentor=None, num_augmented_views: int = 1,
                                classes: Optional[List[str]] = None
                                ) -> _BaseTrainDataset:
    data_root = _resolve_dataset_root(data_root, 'shapenet')
    if classes is None:
        classes = _discover_classes(data_root, ANOMALYSHAPENET_CLASSES)
    return _BaseTrainDataset(data_root, classes, num_points,
                             augmentor, num_augmented_views)


def AnomalyShapeNetTestDataset(data_root: str, num_points: int = 8192,
                               classes: Optional[List[str]] = None
                               ) -> _BaseTestDataset:
    data_root = _resolve_dataset_root(data_root, 'shapenet')
    if classes is None:
        classes = _discover_classes(data_root, ANOMALYSHAPENET_CLASSES)
    return _BaseTestDataset(data_root, classes, num_points)


def build_train_dataset(name: str, data_root: str, **kwargs):
    name = name.lower()
    if 'real3d' in name or name == 'real3d-ad':
        return Real3DADTrainDataset(data_root, **kwargs)
    if 'shapenet' in name or 'anomaly' in name:
        return AnomalyShapeNetTrainDataset(data_root, **kwargs)
    raise ValueError(f"Unknown dataset name: {name}")


def build_test_dataset(name: str, data_root: str, **kwargs):
    name = name.lower()
    if 'real3d' in name or name == 'real3d-ad':
        return Real3DADTestDataset(data_root, **kwargs)
    if 'shapenet' in name or 'anomaly' in name:
        return AnomalyShapeNetTestDataset(data_root, **kwargs)
    raise ValueError(f"Unknown dataset name: {name}")
