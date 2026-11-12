"""A small, independent VOC evaluation campaign."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from ..config import Cell, METHODS, BACKBONES

STRIDES = {"ff2048": 32, "convnext_tiny": 32, "vit_b16": 16, "dinov2_vits14": 14}

VERSION = 'voc-spatial-transfer-v1'


@dataclass(frozen=True)
class SceneConfig:
    backbone: str = 'ff2048'
    methods: tuple[str, ...] = ('csr', 'csrv2', 'mpsae')
    train_k: int = 8
    inference_k: tuple[int, ...] = (8,)
    seed: int = 1227
    dataset_repo: str = 'shijli/voc2012'
    dataset_revision: str = 'main'
    dataset_config: str = 'segmentation'
    image_size: int = 512
    memory_patches: int = 65536
    neighbors: int = 30
    temperature: float = 0.02
    extraction_batch: int = 8
    encode_batch: int = 1024
    gallery_chunk: int = 4096
    query_block: int = 256
    workers: int = 4

    def __post_init__(self):
        if self.backbone not in STRIDES:
            raise ValueError('Unsupported spatial backbone')
        if not self.methods or len(set(self.methods)) != len(self.methods) or any(m not in METHODS for m in self.methods):
            raise ValueError('Choose distinct active methods: csr, csrv2, mpsae')
        for name in ('train_k', 'image_size', 'memory_patches', 'neighbors', 'extraction_batch',
                     'encode_batch', 'gallery_chunk', 'query_block'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.image_size % STRIDES[self.backbone]:
            raise ValueError('image_size must be divisible by backbone stride/patch size')
        if not self.inference_k or len(set(self.inference_k)) != len(self.inference_k) or any(
                type(k) is not int or not 1 <= k <= 8192 for k in self.inference_k):
            raise ValueError('inference_k must contain distinct integers in 1..8192')
        if self.train_k > 8192 or type(self.seed) is not int or self.seed < 0:
            raise ValueError('invalid training K or seed')
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError('workers must be nonnegative')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if self.dataset_config not in ('segmentation', 'segmentation_aug'):
            raise ValueError('dataset_config must be segmentation or segmentation_aug')

    def cells(self):
        return [Cell(self.backbone, method, self.train_k, self.seed) for method in self.methods]

    def plan(self):
        rows = 1464 if self.dataset_config == 'segmentation' else 10582
        patches = (self.image_size // STRIDES[self.backbone])**2
        per_image = min(patches, self.memory_patches // rows)
        if per_image < 1 or per_image * rows < self.neighbors:
            raise ValueError('memory_patches too small for equal-per-image sampling / neighbor count')
        return {'version': VERSION, 'config': asdict(self), 'training_runs': 0,
                'evaluations': 1 + len(self.methods) * len(self.inference_k),
                'support_images': rows, 'query_images': 1449,
                'patches_per_support_image': per_image, 'bank_rows_upper_bound': per_image * rows,
                'cache_bytes_upper_bound': (per_image*rows + 1449*patches)*BACKBONES[self.backbone].dim*4,
                'sparse_bank_bytes_per_k': {k: per_image*rows*k*8 for k in self.inference_k},
                'protocol': 'adapted Hummingbird decoder; frozen pooled-to-spatial SAE transfer',
                'differences_from_paper': [f'native spatial stride {STRIDES[self.backbone]}', 'bounded memory bank',
                    'one deterministic square resize, no augmentation', 'exact search, not ScaNN',
                    'existing pooled-feature SAE; not Hummingbird pretraining']}


@dataclass(frozen=True)
class SceneSuite:
    groups: tuple[SceneConfig, ...]

    def cells(self):
        return [cell for group in self.groups for cell in group.cells()]

    def plan(self):
        plans = [group.plan() for group in self.groups]
        return {'training_runs': 0, 'source_checkpoints': len(self.cells()),
                'evaluations': sum(p['evaluations'] for p in plans), 'experiments': plans}


def load_config(path):
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    experiments = data.pop('experiments', None)
    if experiments is not None:
        if not experiments:
            raise ValueError('experiments cannot be empty')
        groups = []
        for entry in experiments:
            values = {**data, **entry}
            for key in ('methods', 'inference_k'):
                if key in values:
                    values[key] = tuple(values[key])
            groups.append(SceneConfig(**values))
        if len({g.backbone for g in groups}) != len(groups):
            raise ValueError('use one scene group per backbone')
        result = SceneSuite(tuple(groups))
        result.plan()
        return result
    for key in ('methods', 'inference_k'):
        if key in data:
            data[key] = tuple(data[key])
    result = SceneConfig(**data)
    result.plan()
    return result
