"""Official fine train/val only; read local PNGs or ZIPs without extraction."""
from functools import lru_cache
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import io
import os

import numpy as np
from PIL import Image

from ..io import canonical_hash, sha256_file

LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
CLASS_NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle')


class CitySpec:
    name, title, classes, mask_column = 'cityscapes', 'Cityscapes', 19, 'mask'
    train_rows, validation_rows = 2975, 500

    def remap(self, mask):
        raw = np.asarray(mask)
        if raw.ndim != 2 or not np.isin(raw, list(range(34))+[255]).all():
            raise ValueError('Cityscapes requires official grayscale labelIds masks (not RGB/trainIds)')
        out = np.full(raw.shape, 255, dtype=np.uint8)
        for target, original in enumerate(LABEL_IDS):
            out[raw == original] = target
        return out


@lru_cache(maxsize=4)
def _zip(path, pid):
    return ZipFile(path)


def _image(location):
    archive, name = location
    if archive:
        data = _zip(archive, os.getpid()).read(name)
        with Image.open(io.BytesIO(data)) as image:
            return image.copy()
    with Image.open(name) as image:
        return image.copy()


class CityRows:
    column_names = ('id', 'image', 'mask')

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index == 'id':
            return [row[0] for row in self.records]
        name, image, mask = self.records[index]
        return {'id': name, 'image': _image(image), 'mask': _image(mask)}


def load_local(root=None, *, expected=(2975, 500), fingerprint=False):
    value = root or os.environ.get('MPSAE_CITYSCAPES_ROOT')
    if not value:
        raise FileNotFoundError('Attach official Cityscapes fine images/labels and pass --cityscapes-root PATH')
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(root)
    maps = {'image': {}, 'mask': {}}
    archives = sorted(root.rglob('*.zip'))
    archives = [p for p in archives if p.name in ('leftImg8bit_trainvaltest.zip', 'gtFine_trainvaltest.zip')]
    used_archives = set()

    def add(name, archive):
        p = PurePosixPath(name.replace('\\', '/'))
        kind = 'image' if p.name.endswith('_leftImg8bit.png') else 'mask' if p.name.endswith('_gtFine_labelIds.png') else None
        if kind is None:
            return
        marker = 'leftImg8bit' if kind == 'image' else 'gtFine'
        if marker not in p.parts:
            return
        i = p.parts.index(marker)
        if i+3 != len(p.parts)-1 or p.parts[i+1] not in ('train', 'val'):
            return  # never open test, train_extra, coarse, or sequence images
        suffix = '_leftImg8bit.png' if kind == 'image' else '_gtFine_labelIds.png'
        key = (p.parts[i+1], p.name[:-len(suffix)])
        if key in maps[kind]:
            raise ValueError(f'duplicate Cityscapes file: {key}; supply ZIPs OR extracted folders')
        maps[kind][key] = (str(archive) if archive else None, name)
        if archive:
            used_archives.add(archive)

    for path in archives:
        with ZipFile(path) as z:
            for name in z.namelist():
                add(name, path)
    for suffix in ('*_leftImg8bit.png', '*_gtFine_labelIds.png'):
        for path in root.rglob(suffix):
            add(path.as_posix(), None)
    if maps['image'].keys() != maps['mask'].keys():
        raise ValueError('Cityscapes train/val image-mask pairs do not match')
    data = {}
    cities = []
    for split, public, count in [('train', 'train', expected[0]), ('val', 'validation', expected[1])]:
        keys = sorted(k for k in maps['image'] if k[0] == split)
        if len(keys) != count:
            raise ValueError(f'Cityscapes {split}: expected {count} official fine images, got {len(keys)}')
        cities.append({k[1].split('_')[0] for k in keys})
        data[public] = CityRows([(k[1], maps['image'][k], maps['mask'][k]) for k in keys])
    if cities[0] & cities[1]:
        raise ValueError('train and validation cities overlap')
    revision = None
    if fingerprint:
        hashes = [(p.name, sha256_file(p)) for p in sorted(used_archives)]
        for kind in maps:
            hashes.extend((f'{kind}/{split}/{name}', sha256_file(loc[1]))
                          for (split, name), loc in sorted(maps[kind].items()) if loc[0] is None)
        revision = canonical_hash({'files': hashes, 'label_ids': LABEL_IDS, 'splits': expected})
    return data, revision
