"""NYUv2 labeled RGB segmentation: official 795/654 split and NYU40 mapping.

Read individual HDF5 frames, never materialize the depth arrays or whole dataset.
The legacy pipeline key 'validation' refers to the official TEST split here.
"""
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
from PIL import Image

from ..io import canonical_hash, sha256_file

METADATA_REVISION = 'b92d15c506876dada1ac4086938e30ebc4deb49b'
METADATA_SHA256 = {
    'classMapping40.mat': 'ff177194cce9a2163517f1e1e6f1c64c50987162f542dedddc04bbf6ebe6d851',
    'splits.mat': '6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee',
}

CLASS_NAMES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
               'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk',
               'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
               'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper',
               'towel', 'shower curtain', 'box', 'whiteboard', 'person',
               'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag',
               'otherstructure', 'otherfurniture', 'otherprop')


class NYUSpec:
    name, title, classes, mask_column = 'nyuv2', 'NYUv2-40', 40, 'mask'
    train_rows, validation_rows = 795, 654

    def remap(self, mask):
        raw = np.asarray(mask)
        if raw.ndim != 2 or raw.dtype.kind not in 'ui' or not ((raw >= 0) & (raw <= 40)).all():
            raise ValueError('NYUv2 requires mapped NYU40 masks: 0=void, 1..40=classes')
        return np.where(raw == 0, 255, raw.astype(np.int16)-1).astype(np.uint8)


@lru_cache(maxsize=4)
def _handle(path, stamp, size, pid):
    # Each spawned/forked DataLoader process opens its own read-only handle.
    import h5py
    return h5py.File(path, 'r')


class NYURows:
    column_names = ('id', 'image', 'mask')

    def __init__(self, path, indices, mapping):
        self.path = str(path)
        self.indices = tuple(int(i) for i in indices)
        self.mapping = mapping
        stat = Path(path).stat()
        self.stamp, self.size = stat.st_mtime_ns, stat.st_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, str):
            if index != 'id':
                raise KeyError(index)
            return [f'nyuv2-{i+1:04d}' for i in self.indices]
        i = self.indices[index]
        data = _handle(self.path, self.stamp, self.size, os.getpid())
        image = np.asarray(data['images'][i]).transpose(2, 1, 0)
        raw = np.asarray(data['labels'][i]).T
        if raw.dtype.kind not in 'ui' or raw.max() >= len(self.mapping):
            raise ValueError(f'NYUv2 frame {i+1}: raw label is outside the class mapping')
        mask = self.mapping[raw]
        return {'id': f'nyuv2-{i+1:04d}', 'image': Image.fromarray(image),
                'mask': Image.fromarray(mask)}


def load_local(root=None, *, expected=(795, 654), fingerprint=False):
    import h5py
    from scipy.io import loadmat
    value = root or os.environ.get('MPSAE_NYUV2_ROOT')
    if not value:
        raise FileNotFoundError('Attach NYUv2 labeled MAT, splits.mat and classMapping40.mat; pass --nyuv2-root PATH')
    root = Path(value)
    files = {}
    for name in ('nyu_depth_v2_labeled.mat', 'splits.mat', 'classMapping40.mat'):
        matches = sorted(root.rglob(name)) if root.is_dir() else []
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected exactly one {name} under {root}; found {len(matches)}')
        files[name] = matches[0]
    if expected == (795, 654):
        for name, digest in METADATA_SHA256.items():
            if sha256_file(files[name]) != digest:
                raise ValueError(f'{name} does not match the pinned official split/NYU40 mapping; run prepare_nyuv2')
    mapping = np.asarray(loadmat(files['classMapping40.mat'])['mapClass']).ravel()
    if mapping.dtype.kind not in 'uif' or not np.isfinite(mapping).all() or not np.equal(mapping, np.floor(mapping)).all():
        raise ValueError('NYU40 mapping must contain integer class IDs')
    if set(mapping.tolist()) != set(range(1, 41)):
        raise ValueError('Expected NYU40 mapping with all 40 classes, not a 13-class mapping')
    mapping = np.concatenate([np.zeros(1, dtype=np.uint8), mapping.astype(np.uint8)])
    path = files['nyu_depth_v2_labeled.mat']
    with h5py.File(path, 'r') as handle:
        images, labels = handle['images'], handle['labels']
        if (images.ndim != 4 or images.shape[1] != 3 or images.dtype != np.uint8
                or labels.shape != (images.shape[0], *images.shape[2:]) or labels.dtype.kind not in 'ui'):
            raise ValueError('Unexpected official NYUv2 MAT layout; expected images [N,3,W,H], labels [N,W,H]')
        n = images.shape[0]
    split = loadmat(files['splits.mat'])
    indices = {}
    for internal, field, count in [('train', 'trainNdxs', expected[0]), ('validation', 'testNdxs', expected[1])]:
        raw = np.asarray(split[field]).ravel()
        if (len(raw) != count or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()
                or not ((raw >= 1) & (raw <= n)).all() or len(np.unique(raw)) != count):
            raise ValueError(f'Invalid NYUv2 official {field}: count, duplicate or out-of-range image IDs')
        indices[internal] = raw.astype(np.int64)-1
    if set(indices['train']) & set(indices['validation']):
        raise ValueError('NYUv2 train/test IDs overlap')
    if sum(expected) != n:
        raise ValueError('NYUv2 split must cover the entire labeled dataset')
    revision = None
    if fingerprint:
        revision = canonical_hash({'files': {name: sha256_file(p) for name, p in files.items()},
            'protocol': 'nyu40-rgb-official-train-test-v1', 'ignore': 0})
    return {s: NYURows(path, ids, mapping) for s, ids in indices.items()}, revision
