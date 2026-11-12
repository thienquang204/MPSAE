"""Pinned Hugging Face VOC and reusable spatial features, independent of SAE K."""
from contextlib import nullcontext
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..benchmark.data import open_rgb
from ..benchmark.extract import _network
from ..config import BACKBONES
from ..io import atomic_json, canonical_hash, read_json, sha256_file
from .config import VERSION
from .core import label_distributions, resnet_spatial, sample_patches


def open_mask(value):
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as image:
                return image.copy()
        value = value.get('path')
    with Image.open(value) as image:
        return image.copy()


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


class ImageRows:
    def __init__(self, dataset, pending, size, normalize):
        self.dataset, self.pending, self.size = dataset, pending, size
        self.normalize = normalize

    def __len__(self):
        return len(self.pending)

    def __getitem__(self, position):
        from torchvision.transforms.functional import normalize, pil_to_tensor
        index = self.pending[position]
        row = self.dataset[index]
        image, mask = open_rgb(row['image']), open_mask(row['mask'])
        if image.size != mask.size:
            raise ValueError('image and segmentation mask are misaligned')
        truth = np.array(mask, dtype=np.int64)
        if truth.ndim != 2 or not np.all(((truth >= 0) & (truth <= 20)) | (truth == 255)):
            raise ValueError('HF masks must preserve VOC class IDs (not RGB)')
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        scaled = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        image = normalize(pil_to_tensor(image).float()/255, *self.normalize)
        return index, image, truth, np.array(scaled, dtype=np.int64)


def collate(rows):
    return [r[0] for r in rows], torch.stack([r[1] for r in rows]), [r[2] for r in rows], [r[3] for r in rows]


def prepare(config, workspace, weights, *, device):
    """One extraction shared by all SAE methods. Completed caches need no HF request."""
    from huggingface_hub import HfApi
    from ..embed.archive import library_versions

    workspace = Path(workspace)
    resolution_key = canonical_hash([config.dataset_repo, config.dataset_revision])[:16]
    resolution_path = workspace / 'dataset' / f'{resolution_key}.json'
    resolution = read_json(resolution_path)
    if resolution is None:
        commit = HfApi().dataset_info(config.dataset_repo, revision=config.dataset_revision).sha
        resolution = {'repo': config.dataset_repo, 'revision': commit}
        atomic_json(resolution_path, resolution)
    identity = {'version': VERSION, 'dataset': resolution, 'dataset_config': config.dataset_config,
                'backbone_weights': weights, 'backbone': config.backbone, 'image_size': config.image_size,
                'memory_patches': config.memory_patches, 'seed': config.seed,
                'precision': 'cuda_fp16_to_float32' if str(device).startswith('cuda') else 'cpu_float32',
                'extraction_batch': config.extraction_batch,
                'layer': 'layer4_pre_avgpool', 'stride': 32,
                'normalize': [list(x) for x in BACKBONES[config.backbone].normalize],
                'library_versions': library_versions(),
                'geometry': 'square_bilinear_image_nearest_mask_no_crop', 'augmentation': 'none'}
    directory = workspace / 'features' / canonical_hash(identity)[:20]
    completed = read_json(directory / 'manifest.json')
    if completed and completed.get('identity') == identity:
        for row in completed['files']:
            path = directory / row['file']
            if not path.is_file() or sha256_file(path) != row['sha256']:
                raise RuntimeError(f'Feature cache is incomplete/corrupt: {path}')
        return directory, completed

    from datasets import load_dataset
    from torch.utils.data import DataLoader

    data = load_dataset(config.dataset_repo, config.dataset_config, revision=resolution['revision'],
                        cache_dir=str(workspace / 'hf_datasets'))
    expected = {'train': config.plan()['support_images'], 'validation': 1449}
    ids = {}
    for split, count in expected.items():
        if len(data[split]) != count:
            raise ValueError(f'{split}: expected {count} VOC rows, got {len(data[split])}')
        ids[split] = list(data[split]['id'])
        if len(set(ids[split])) != count:
            raise ValueError(f'duplicate image IDs in {split}')
    if set(ids['train']) & set(ids['validation']):
        raise ValueError('support/query image IDs overlap')

    spec = BACKBONES[config.backbone]
    network, _, observed = _network(spec, workspace / 'backbone_cache', torch.device(device), None, weights)
    network.eval().requires_grad_(False)
    grid = (config.image_size//32,)*2
    files = []
    old_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        for split, count in expected.items():
            folder = directory / split
            pending = [i for i in range(count) if not (folder / f'{i:05d}.npz').is_file()]
            loader = DataLoader(ImageRows(data[split], pending, config.image_size, spec.normalize),
                                batch_size=config.extraction_batch, num_workers=config.workers,
                                collate_fn=collate, shuffle=False, pin_memory=str(device).startswith('cuda'))
            amp = torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
            with torch.inference_mode(), amp:
                for positions, images, truths, scaled in loader:
                    features = resnet_spatial(network, images.to(device)).float().cpu().numpy()
                    if features.shape[1:] != (*grid, 2048) or not np.isfinite(features).all():
                        raise ValueError('unexpected/nonfinite spatial features')
                    for index, feature, truth, mask in zip(positions, features, truths, scaled):
                        feature = feature.reshape(-1, 2048)
                        if split == 'train':
                            labels = label_distributions(mask, grid)
                            chosen = sample_patches(labels, config.plan()['patches_per_support_image'],
                                                    np.random.default_rng([config.seed, index]))
                            save_arrays(folder / f'{index:05d}.npz', features=feature[chosen],
                                        labels=labels[chosen], patch_indices=chosen)
                        else:
                            save_arrays(folder / f'{index:05d}.npz', features=feature,
                                        truth=truth.astype(np.uint8))
                    print(f'[scene extract] {split}: {positions[-1]+1}/{count}', flush=True)
            for index in range(count):
                path = folder / f'{index:05d}.npz'
                files.append({'file': str(path.relative_to(directory)).replace('\\', '/'),
                              'id': ids[split][index], 'sha256': sha256_file(path)})
    finally:
        torch.backends.cudnn.allow_tf32 = old_tf32
    manifest = {'complete': True, 'identity': identity, 'grid': grid, 'files': files,
                'weights_observed': observed, 'support_images': expected['train'], 'query_images': 1449}
    atomic_json(directory / 'manifest.json', manifest)
    return directory, manifest


def support_bank(directory, manifest):
    """Concatenate only sampled patches; never load all query features into RAM."""
    features, labels = [], []
    for index in range(manifest['support_images']):
        with np.load(Path(directory) / 'train' / f'{index:05d}.npz', allow_pickle=False) as data:
            features.append(data['features'])
            labels.append(data['labels'])
    return np.concatenate(features), np.concatenate(labels)
