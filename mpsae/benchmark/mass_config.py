"""Independent ImageNet MPSAE mass ablations; planning does not import torch."""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path

from .. import __version__
from ..config import BACKBONES, Cell, training_preset
from ..io import canonical_hash
from ..transport import solver_name


def integers(values, name, minimum=1):
    if (not isinstance(values, (tuple, list)) or not values
            or any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'{name} must contain distinct integers >= {minimum}')
    return tuple(sorted(values))


@dataclass(frozen=True)
class MassConfig:
    backbones: tuple = ('ff2048', 'vit_b16')
    train_k: dict = field(default_factory=lambda: {'ff2048': [8, 32], 'vit_b16': [12, 48]})
    masses: tuple = (.9, 1.)
    seeds: tuple = (1227,)
    evaluation_mode: str = 'matched'
    inference_k: dict = field(default_factory=dict)
    epochs: int | None = None
    lr: float | None = None
    ot_weight: float | None = None
    ot_eta: float = .2
    ot_microbatch: int = 32

    def __post_init__(self):
        if (not isinstance(self.backbones, (tuple, list)) or not self.backbones
                or any(b not in BACKBONES for b in self.backbones)
                or len(set(self.backbones)) != len(self.backbones)):
            raise ValueError(f'backbones must be distinct members of {list(BACKBONES)}')
        object.__setattr__(self, 'backbones', tuple(sorted(self.backbones)))
        if not isinstance(self.train_k, dict) or set(self.train_k) != set(self.backbones):
            raise ValueError('train_k must give budgets for every selected backbone, and no others')
        object.__setattr__(self, 'train_k', {
            b: integers(self.train_k[b], f'train_k.{b}') for b in self.backbones})
        if not isinstance(self.inference_k, dict) or set(self.inference_k) - set(self.backbones):
            raise ValueError('inference_k must map selected backbones to budgets')
        if self.evaluation_mode not in ('matched', 'sweep'):
            raise ValueError('evaluation_mode must be matched or sweep')
        if self.evaluation_mode == 'matched' and self.inference_k:
            raise ValueError('matched evaluation does not accept inference_k; use evaluation_mode: sweep')
        if self.evaluation_mode == 'sweep' and set(self.inference_k) != set(self.backbones):
            raise ValueError('sweep requires inference_k for every selected backbone')
        object.__setattr__(self, 'inference_k', {
            b: integers(v, f'inference_k.{b}') for b, v in self.inference_k.items()})
        if not isinstance(self.masses, (tuple, list)) or not self.masses:
            raise ValueError('masses must be nonempty')
        for s in self.masses:
            solver_name(s)
        if len(set(self.masses)) != len(self.masses):
            raise ValueError('duplicate masses')
        object.__setattr__(self, 'masses', tuple(sorted(float(s) for s in self.masses)))
        object.__setattr__(self, 'seeds', integers(self.seeds, 'seeds', minimum=0))
        if self.epochs is not None:
            integers([self.epochs], 'epochs')
        integers([self.ot_microbatch], 'ot_microbatch', minimum=2)
        for name in ('lr', 'ot_weight', 'ot_eta'):
            v = getattr(self, name)
            if v is None and name != 'ot_eta':
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for b in self.backbones:
            budgets = (*self.train_k[b], *self.inference_k.get(b, ()))
            if max(budgets) > BACKBONES[b].dim:
                raise ValueError(f'{b}: K must not exceed input dimension {BACKBONES[b].dim}')
        if self.ot_weight is not None and self.ot_weight != .5 and any(
                k > 64 for ks in self.train_k.values() for k in ks):
            raise ValueError('Existing MPSAE recipe fixes weight=0.5 for train K>64; '
                             'omit ot_weight or select 0.5 for those experiments')

    @classmethod
    def load(cls, path):
        import yaml
        payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or set(payload) - set(cls.__dataclass_fields__):
            raise ValueError('invalid or unknown retrieval mass config fields')
        return cls(**payload)

    def cells(self):
        return [(Cell(b, 'mpsae', k, seed), s) for b in self.backbones
                for k in self.train_k[b] for seed in self.seeds for s in self.masses]

    def budgets(self, cell):
        return (cell.k,) if self.evaluation_mode == 'matched' else self.inference_k[cell.backbone]

    def spec(self, cell, mass, *, smoke=False):
        original = BACKBONES[cell.backbone]
        changes = dict(ot_mass=mass, ot_eta=self.ot_eta, ot_microbatch=self.ot_microbatch)
        for name in ('epochs', 'lr', 'ot_weight'):
            if getattr(self, name) is not None:
                changes[name] = getattr(self, name)
        preset = training_preset('mpsae', replace(original.preset, **changes), k=cell.k)
        if not smoke:
            return replace(original, preset=preset)
        budgets = tuple(sorted(set((*self.train_k[cell.backbone], *self.budgets(cell)))))
        return replace(original, dim=max(8, max(budgets)), k_list=budgets,
                       train_k_list=self.train_k[cell.backbone], e2_anchors=(),
                       reference_dense_top1=None,
                       preset=replace(preset, hidden_dim=max(64, 4*max(budgets)), k_aux=8, epochs=2))

    def plan(self):
        settings = json.loads(json.dumps(asdict(self)))
        recipes = [{'backbone': c.backbone, 'train_k': c.k, 'seed': c.seed,
                    's': s, 'solver': solver_name(s), 'inference_k': list(self.budgets(c)),
                    'preset': asdict(self.spec(c, s).preset)} for c, s in self.cells()]
        identity = {'protocol': 'imagenet-mass-v1', 'code_version': __version__,
                    'config': settings, 'recipes': recipes}
        return {**identity, 'sweep_id': canonical_hash(identity)[:20],
                'fits': len(recipes), 'evaluations': sum(len(r['inference_k']) for r in recipes),
                'metric': 'ImageNet exact 1-NN top-1 (%)',
                'linear_probe': False, 'timing': False, 'downstream': False}
