"""Dataset-specific semantic label contracts; internal ignore is always 255."""
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import numpy as np


@dataclass(frozen=True)
class SceneDataset:
    name: str
    title: str
    classes: int
    train_rows: int
    validation_rows: int
    mask_column: str
    reduce_zero: bool = False

    def remap(self, mask):
        raw = np.asarray(mask, dtype=np.int64)
        if raw.ndim != 2:
            raise ValueError('segmentation mask must be class IDs, not RGB')
        if self.reduce_zero:
            if not np.all((raw >= 0) & (raw <= self.classes)):
                raise ValueError('ADE20K raw masks must contain IDs 0..150')
            return np.where(raw == 0, 255, raw-1).astype(np.uint8)
        if not np.all(((raw >= 0) & (raw < self.classes)) | (raw == 255)):
            raise ValueError('VOC raw masks must contain IDs 0..20 or 255')
        return raw.astype(np.uint8)


VOC = SceneDataset('voc', 'Pascal VOC', 21, 1464, 1449, 'mask')
VOC_AUG = SceneDataset('voc', 'Pascal VOC', 21, 10582, 1449, 'mask')
ADE20K = SceneDataset('ade20k', 'ADE20K', 150, 20210, 2000, 'annotation', True)


def dataset_spec(name, config):
    if name == 'voc' and config in ('segmentation', 'segmentation_aug'):
        return VOC if config == 'segmentation' else VOC_AUG
    if name == 'ade20k' and config == 'scene_parsing':
        return ADE20K
    raise ValueError('Use voc/segmentation[_aug] or ade20k/scene_parsing')


def row_ids(dataset):
    """ADE Parquet has no ID column: use original filenames, never synthetic split IDs."""
    if 'id' in dataset.column_names:
        return list(dataset['id'])
    from datasets import Image
    images = dataset.select_columns(['image']).cast_column('image', Image(decode=False))
    result = []
    for row in images:
        value = row['image']
        path = value.get('path')
        if path:
            result.append(PurePosixPath(path.replace('\\', '/')).name)
        elif value.get('bytes') is not None:
            result.append(hashlib.sha256(value['bytes']).hexdigest())
        else:
            raise ValueError('image has no stable filename or bytes')
    return result


def load_splits(config, resolution, cache_dir):
    from datasets import load_dataset
    if config.dataset == 'voc':
        return load_dataset(config.dataset_repo, config.dataset_config,
                            revision=resolution['revision'], cache_dir=str(cache_dir))
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(config.dataset_repo, repo_type='dataset', revision=resolution['revision'])
    selected = {}
    for split in ('train', 'validation'):
        names = sorted(f for f in files if f.startswith(f'scene_parsing/{split}/') and f.endswith('.parquet'))
        if not names:
            raise ValueError('ADE20K revision must contain scene_parsing Parquet shards; see scene_ade20k.yaml')
        selected[split] = [f'hf://datasets/{config.dataset_repo}@{resolution["revision"]}/{name}' for name in names]
    # Do not download the unlabeled test split or the instance-segmentation configuration.
    return load_dataset('parquet', data_files=selected, cache_dir=str(cache_dir))
