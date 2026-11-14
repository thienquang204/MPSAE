from dataclasses import asdict, dataclass
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
METHODS = ('csr', 'csrv2', 'mpsae')


@dataclass(frozen=True)
class Config:
    seed: int = 1227
    k: int = 12
    inference_k: tuple[int, ...] | None = None
    hidden_dim: int = 8192
    image_size: int = 448
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

    def __post_init__(self):
        for key in ('seed', 'k', 'hidden_dim', 'image_size', 'train_patches',
                    'support_patches', 'epochs', 'batch_size', 'extraction_batch',
                    'workers', 'shard_images', 'query_block', 'neighbors',
                    'checkpoint_every', 'log_every'):
            x = getattr(self, key)
            if type(x) is not int or x < (0 if key in ('seed', 'workers') else 1):
                raise ValueError(f'invalid {key}')
        if self.k != 12 or self.hidden_dim != 8192:
            raise ValueError('SAE training is fixed at K=12 and 8192 latents; use inference_k for evaluation')
        if self.inference_k is not None:
            if not isinstance(self.inference_k, (list, tuple)) or not self.inference_k:
                raise ValueError('inference_k must be a nonempty list of distinct positive integers')
            if any(type(k) is not int or not 1 <= k <= self.hidden_dim for k in self.inference_k):
                raise ValueError('inference_k must contain integers in [1, hidden_dim]')
            if len(set(self.inference_k)) != len(self.inference_k):
                raise ValueError('inference_k must not contain duplicates')
            object.__setattr__(self, 'inference_k', tuple(sorted(self.inference_k)))
        if self.image_size % 112 or self.batch_size < 2:
            raise ValueError('image_size must be divisible by 14 and 16; batch_size >= 2')
        if not self.neighbors <= self.support_patches <= self.train_patches:
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
        # Preserve the original K=12 campaign ID and all training/cache identities.
        settings.pop('inference_k')
        if self.inference_ks != (self.k,):
            settings['inference_k'] = list(self.inference_ks)
        return {'version': VERSION, 'settings': settings}

    @property
    def inference_ks(self):
        return self.inference_k or (self.k,)

    def source_identity(self):
        identity = self.identity()
        identity['settings'].pop('inference_k', None)
        return identity

    def plan(self):
        return {**self.identity(), 'campaign': canonical_hash(self.identity())[:20],
                'encoders': ENCODERS, 'datasets': DATASETS, 'methods': METHODS,
                'sae_fits': 18, 'sparse_evaluations': 18*len(self.inference_ks), 'dense_evaluations': 6,
                'total_evaluations': 6+18*len(self.inference_ks), 'train_k': self.k,
                'inference_k': list(self.inference_ks),
                'backbone_trainable': False, 'figures': False,
                'protocol': 'dataset-specific patch SAE training; Hummingbird-style retrieval',
                'csrv2_supervision': 'dominant nonvoid training-mask class',
                'training_preset': asdict(self.preset())}


def load(path):
    import yaml
    with open(path, encoding='utf-8') as handle:
        return Config(**(yaml.safe_load(handle) or {}))
