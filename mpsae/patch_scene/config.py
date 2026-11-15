from dataclasses import asdict, dataclass, replace
import math

from ..config import TrainPreset
from ..io import canonical_hash

VERSION = 'patch-scene-v2.0'
ENCODERS = {
    'siglip2': ('google/siglip2-base-patch16-256', 16),
    'dinov2': ('facebook/dinov2-base', 14),
    'clip': ('openai/clip-vit-base-patch16', 16),
}
DATASETS = {
    'voc': ('shijli/voc2012', 'fba53aa390b729cb2bb848de59e1439deea3e88f', 'segmentation'),
    'ade20k': ('zhoubolei/scene_parse_150', 'e660d866a1351c70bcf07925d2600b60bd6e3bc3', 'scene_parsing'),
}
CITYSCAPES_HF = ('Chris1/cityscapes', 'f5245e5b4ce9ec8c1c00e17921e176c566914a7c', 'labelIds')
DATASETS['cityscapes'] = ('local-cityscapes', 'official-fine', 'fine')
DATASETS['nyuv2'] = ('local-nyuv2', 'official-labeled', 'nyu40')

# Historical registry entries remain for reading old receipts, not launching jobs.
SUPPORTED_DATASETS = ('voc', 'nyuv2', 'cityscapes')
SUPPORTED_ENCODERS = ('siglip2', 'dinov2')

METHODS = ('csr', 'csrv2', 'mpsae')


@dataclass(frozen=True)
class Config:
    cityscapes_source: str = 'official'
    seed: int = 1227
    k: int = 12
    inference_k: tuple[int, ...] | None = None
    evaluation_mode: str = 'sweep'
    train_k: tuple[int, ...] | None = None
    encoders: tuple[str, ...] | None = None
    hidden_dim: int = 8192
    image_size: int = 448
    image_width: int | None = None
    all_train_patches: bool = False
    train_patches: int = 262144
    support_patches: int = 65536
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 4e-5
    extraction_batch: int = 4
    workers: int = 2
    shard_images: int = 64
    query_block: int = 32
    neighbors: int = 30
    temperature: float = 0.02
    checkpoint_every: int = 128
    log_every: int = 20
    cache_dtype: str = 'float16'
    datasets: tuple[str, ...] | None = None
    early_stopping: bool = False
    early_min_epochs: int = 15
    early_patience: int = 3
    early_min_delta: float = 0.001

    def __post_init__(self):
        if self.cityscapes_source not in ('official', 'chris1'):
            raise ValueError('cityscapes_source must be official or chris1')
        if self.evaluation_mode not in ('sweep', 'matched'):
            raise ValueError('evaluation_mode must be sweep or matched')
        if type(self.all_train_patches) is not bool:
            raise ValueError('all_train_patches must be boolean')
        if self.image_width is not None and (type(self.image_width) is not int or self.image_width < 112 or self.image_width % 112):
            raise ValueError('image_width must be divisible by 14 and 16')
        if self.datasets is not None:
            if not isinstance(self.datasets, (list, tuple)) or not self.datasets or any(d not in DATASETS for d in self.datasets) or len(set(self.datasets)) != len(self.datasets):
                raise ValueError('datasets must be distinct registered dataset names')
            object.__setattr__(self, 'datasets', tuple(d for d in DATASETS if d in self.datasets))
        if type(self.early_stopping) is not bool:
            raise ValueError('early_stopping must be boolean')
        for key in ('early_min_epochs', 'early_patience'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'invalid {key}')
        if not math.isfinite(self.early_min_delta) or not 0 <= self.early_min_delta < 1:
            raise ValueError('early_min_delta must be in [0, 1)')
        if self.early_stopping and self.early_min_epochs > self.epochs:
            raise ValueError('early_min_epochs exceeds epochs')
        for key in ('seed', 'k', 'hidden_dim', 'image_size', 'train_patches',
                    'support_patches', 'epochs', 'batch_size', 'extraction_batch',
                    'workers', 'shard_images', 'query_block', 'neighbors',
                    'checkpoint_every', 'log_every'):
            x = getattr(self, key)
            minimum = 0 if key in ('seed', 'workers') or (key == 'train_patches' and self.all_train_patches) else 1
            if type(x) is not int or x < minimum:
                raise ValueError(f'invalid {key}')
        if self.hidden_dim != 8192 or self.k > self.hidden_dim:
            raise ValueError('This campaign uses 8192 latents; K must not exceed hidden_dim')
        for key in ('train_k', 'inference_k'):
            values = getattr(self, key)
            if values is None:
                continue
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f'{key} must be a nonempty list of distinct positive integers')
            if any(type(k) is not int or not 1 <= k <= self.hidden_dim for k in values):
                raise ValueError(f'{key} must contain integers in [1, hidden_dim]')
            if len(set(values)) != len(values):
                raise ValueError(f'{key} must not contain duplicates')
            object.__setattr__(self, key, tuple(sorted(values)))
        if self.encoders is not None:
            if (not isinstance(self.encoders, (list, tuple)) or not self.encoders
                    or any(e not in ENCODERS for e in self.encoders)
                    or len(set(self.encoders)) != len(self.encoders)):
                raise ValueError('encoders must be a nonempty list of distinct registered encoder names')
            object.__setattr__(self, 'encoders', tuple(e for e in ENCODERS if e in self.encoders))
        if self.evaluation_mode == 'matched' and self.inference_k is not None and self.inference_k != self.training_ks:
            raise ValueError('matched evaluation requires inference_k omitted or equal to train_k')
        if self.image_size % 112 or self.batch_size < 2:
            raise ValueError('image_size must be divisible by 14 and 16; batch_size >= 2')
        if not self.neighbors <= self.support_patches or (not self.all_train_patches and self.support_patches > self.train_patches):
            raise ValueError('require neighbors <= support_patches <= train_patches')
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.temperature)):
            raise ValueError('lr and temperature must be finite and positive')
        if self.cache_dtype not in ('float16', 'float32'):
            raise ValueError('cache_dtype must be float16 or float32')

    def preset(self):
        return TrainPreset(name=VERSION, hidden_dim=self.hidden_dim, epochs=self.epochs,
                           batch_size=self.batch_size, lr=self.lr, cl_coef=0.1)

    def identity(self):
        settings = asdict(self)
        if self.cityscapes_source == 'official':
            settings.pop('cityscapes_source')
        if self.evaluation_mode == 'sweep':
            settings.pop('evaluation_mode')  # keep historical campaign identities
        if self.image_width is None:
            settings.pop('image_width')
        if not self.all_train_patches:
            settings.pop('all_train_patches')
        if self.datasets is None:
            settings.pop('datasets')
        if not self.early_stopping:
            for key in ('early_stopping', 'early_min_epochs', 'early_patience', 'early_min_delta'):
                settings.pop(key)
        # Preserve the original K=12 campaign ID and all training/cache identities.
        settings.pop('inference_k')
        settings.pop('train_k')
        settings.pop('encoders')
        if self.training_ks != (self.k,):
            settings['train_k'] = list(self.training_ks)
        if self.encoder_names != tuple(ENCODERS):
            settings['encoders'] = list(self.encoder_names)
        if self.inference_ks != (self.k,):
            settings['inference_k'] = list(self.inference_ks)
        return {'version': VERSION, 'settings': settings}

    @property
    def image_hw(self):
        return (self.image_size, self.image_width or self.image_size)

    @property
    def dataset_names(self):
        return self.datasets or ('voc', 'ade20k')  # historical configs only

    @property
    def inference_ks(self):
        return self.inference_k or self.training_ks

    @property
    def training_ks(self):
        return self.train_k or (self.k,)

    @property
    def encoder_names(self):
        # None retains the historical three-encoder campaign identity.
        return self.encoders or tuple(ENCODERS)

    def for_training(self, k):
        if k not in self.training_ks:
            raise ValueError(f'train K={k} is not selected')
        return replace(self, k=k, train_k=None,
                       inference_k=(k,) if self.evaluation_mode == 'matched' else self.inference_ks)

    @property
    def evaluation_pairs(self):
        return tuple((t, k) for t in self.training_ks for k in self.inference_ks
                     if self.evaluation_mode != 'matched' or t == k)

    def source_identity(self):
        identity = self.identity()
        identity['settings'].pop('datasets', None)
        if self.early_stopping:
            # Training duration/stopping does not change extracted features.
            for key in ('early_stopping', 'early_min_epochs', 'early_patience', 'early_min_delta'):
                identity['settings'].pop(key, None)
            identity['settings']['epochs'] = 10
        identity['settings'].pop('inference_k', None)
        identity['settings'].pop('train_k', None)
        identity['settings'].pop('encoders', None)
        identity['settings'].pop('evaluation_mode', None)
        # Legacy source receipts were keyed by the K=12 campaign. Neither the
        # selected training budgets nor encoder subset should refresh revisions.
        identity['settings']['k'] = 12
        return identity

    def plan(self):
        cells = len(self.encoder_names)*len(self.dataset_names)
        fits = cells*len(METHODS)*len(self.training_ks)
        sparse = cells*len(METHODS)*len(self.evaluation_pairs)
        matched = cells*len(METHODS)*sum(t == k for t, k in self.evaluation_pairs)
        return {**self.identity(), 'campaign': canonical_hash(self.identity())[:20],
                'encoders': {e: ENCODERS[e] for e in self.encoder_names}, 'datasets': {d: (CITYSCAPES_HF if d == 'cityscapes' and self.cityscapes_source == 'chris1' else DATASETS[d]) for d in self.dataset_names}, 'methods': METHODS,
                'sae_fits': fits, 'sparse_evaluations': sparse, 'dense_evaluations': cells,
                'matched_evaluations': matched, 'cross_k_evaluations': sparse-matched,
                'total_evaluations': cells+sparse,
                'train_k': self.training_ks[0] if len(self.training_ks) == 1 else list(self.training_ks),
                'train_ks': list(self.training_ks),
                'inference_k': list(self.inference_ks),
                'evaluation_mode': self.evaluation_mode,
                'evaluation_pairs': [list(pair) for pair in self.evaluation_pairs],
                'evaluation_splits': {d: 'test' if d == 'nyuv2' else 'validation' for d in self.dataset_names},
                'backbone_trainable': False, 'figures': False,
                'training_data': 'ALL valid training patches, disk streaming' if self.all_train_patches else 'sampled bank',
                'input_hw': list(self.image_hw),
                'support_candidate_limit': self.train_patches if not self.all_train_patches else None,
                'protocol': 'dataset-specific patch SAE training; Hummingbird-style retrieval',
                'csrv2_supervision': 'dominant nonvoid training-mask class',
                'training_preset': asdict(self.preset())}


def load(path):
    """Load a canonical config or a simple relative alias, with cycle detection."""
    import yaml
    from pathlib import Path
    seen = set()
    def resolve(current):
        current = Path(current).resolve()
        if current in seen:
            raise ValueError('configuration alias cycle')
        seen.add(current)
        with current.open(encoding='utf-8') as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError('configuration must be a mapping')
        parent = values.pop('extends', None)
        if parent is not None:
            if not isinstance(parent, str) or Path(parent).is_absolute():
                raise ValueError('extends must be a relative config path')
            if values:
                raise ValueError('config aliases cannot override settings; edit the canonical config')
            return resolve(current.parent/parent)
        return values
    return Config(**resolve(path))


def require_supported(config):
    """Never silently launch retired datasets/encoders from historical configs."""
    if (('cityscapes' in config.dataset_names and config.cityscapes_source != 'chris1')
            or any(d not in SUPPORTED_DATASETS for d in config.dataset_names)
            or any(e not in SUPPORTED_ENCODERS for e in config.encoder_names)):
        raise ValueError('Supported segmentation: VOC, NYUv2 or Cityscapes with SigLIP2/DINOv2. '
                         'Use configs/patch_scene_voc.yaml or configs/patch_scene_nyuv2.yaml; '
                         'archived configurations are retained only for provenance/export.')
