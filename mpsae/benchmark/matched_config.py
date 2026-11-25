"""Torch-free, explicit matched-optimizer comparison; no changes to main presets."""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

from ..config import BACKBONES, METHODS, Cell, training_preset
from ..io import canonical_hash


@dataclass(frozen=True)
class MatchedConfig:
    backbone: str = 'ff2048'
    methods: tuple = METHODS
    seeds: tuple = (1227, 2026)
    k_values: tuple = (2, 8, 32, 128, 256, 512)
    batch_sizes: tuple = (1024, 4096)
    epochs: int = 10
    lr: float = 4e-5
    probe_k: int = 32
    probe_epochs: int = 10
    probe_batch_size: int = 4096
    probe_lr: float = 1e-3
    warmup: int = 5
    rounds: int = 5
    query_batch: int = 512
    gallery_chunk: int = 32768
    build_entry_cap: int = 4194304
    encode_batch_size: int = 2048

    def __post_init__(self):
        if self.backbone != 'ff2048':
            raise ValueError('This controlled study uses ResNet-50 / ff2048')
        for name in ('methods', 'seeds', 'k_values', 'batch_sizes'):
            values=getattr(self,name)
            if not isinstance(values,(tuple,list)) or not values or len(set(values))!=len(values):
                raise ValueError(f'{name} must contain distinct nonempty values')
            object.__setattr__(self,name,tuple(values))
        if set(self.methods)-set(METHODS):
            raise ValueError('Only CSR, CSRv2 and MPSAE are supported')
        for name in ('seeds','k_values','batch_sizes'):
            minimum=0 if name=='seeds' else 2 if name=='batch_sizes' else 1
            if any(type(v) is not int or v<minimum for v in getattr(self,name)):
                raise ValueError(f'Invalid {name}')
        if max(self.k_values)>BACKBONES[self.backbone].dim or self.probe_k not in self.k_values:
            raise ValueError('K must be valid for ResNet-50; probe_k must be a selected training K')
        for name in ('epochs','probe_epochs','probe_batch_size','warmup','rounds','query_batch',
                     'gallery_chunk','build_entry_cap','encode_batch_size'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('lr','probe_lr'):
            if isinstance(getattr(self,name),bool) or not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        if self.build_entry_cap<max(self.k_values):
            raise ValueError('build_entry_cap must hold at least one code row')

    @classmethod
    def load(cls,path):
        import yaml
        value=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value,dict) or set(value)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown or invalid matched-study settings')
        return cls(**value)

    def cells(self):
        return [(Cell(self.backbone,m,k,s),b) for b in self.batch_sizes
                for k in sorted(self.k_values) for s in self.seeds for m in self.methods]

    def spec(self,cell,batch,*,smoke=False):
        base=BACKBONES[self.backbone]
        p=training_preset(cell.method,replace(base.preset,epochs=self.epochs,lr=self.lr),k=cell.k)
        p=replace(p,batch_size=batch)
        if smoke:
            budgets=tuple(sorted(self.k_values))
            return replace(base,dim=max(8,max(budgets)),k_list=budgets,train_k_list=budgets,
                           e2_anchors=(),reference_dense_top1=None,
                           preset=replace(p,hidden_dim=max(64,16*max(budgets)),k_aux=8,epochs=1))
        return replace(base,preset=p)

    def plan(self):
        n=len(self.cells())
        probes=len(self.methods)*len(self.batch_sizes)*len(self.seeds)
        settings=asdict(self)
        return {'protocol':'matched-optimizer-imagenet-v1','config':settings,
                'study_id':canonical_hash(settings)[:20], 'sae_fits':n,'one_nn_evaluations':n,
                'probe_fits':probes,'accuracy_evaluations':n+probes,
                'timing_sessions':n+probes, 'train_inference_k':'matched only',
                'updates_per_fit':{str(b):math.ceil(1281167/b)*self.epochs for b in self.batch_sizes},
                'mpsae_ot_mass':.9,'mpsae_ot_weight':{'K<=64':2.,'K>64':.5},
                'fairness':'same optimizer, epochs, initialization and batch order within each batch-size group; different method losses retained',
                'recipes':[{'method':c.method,'k':c.k,'seed':c.seed,'batch_size':b,
                            'preset':asdict(self.spec(c,b).preset)} for c,b in self.cells()]}
