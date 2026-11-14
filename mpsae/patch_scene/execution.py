"""Independent GPU jobs with one shared feature extraction per encoder/dataset."""
from dataclasses import dataclass, replace

from ..io import atomic_json, read_json
from .config import METHODS


@dataclass(frozen=True)
class Task:
    stage: str
    encoder: str
    dataset: str
    method: str = 'dense'
    train_k: int | None = None

    @property
    def name(self):
        return f'{self.dataset}-{self.encoder}-{self.stage}-{self.method}-train{self.train_k}'


def preparation_tasks(encoders, datasets):
    # Start the larger dataset early; workers take the next ready job dynamically.
    return [Task('prepare', e, d) for d in sorted(datasets, key=lambda d: d != 'ade20k') for e in encoders]


def evaluation_tasks(config, preparation):
    e, d = preparation.encoder, preparation.dataset
    # Reuse the previous K=12 fits first, then start the larger new budgets.
    grid = sorted(config.training_ks, key=lambda k: (k != 12, -k))
    return [Task('evaluate', e, d)] + [Task('fit', e, d, m, k) for k in grid for m in METHODS]


def record_result(hub, campaign, result):
    """Only the coordinator writes campaign rows, tables and the shared ZIP."""
    from .report import collect_tables, row_key
    relative = f'campaigns/{campaign}/cells/{result["dataset"]}-{result["encoder"]}'
    folder = hub.pull(relative)
    rows = (read_json(folder/'rows.json') or {}).get('results', [])
    rows = [r for r in rows if row_key(r) != row_key(result)]
    rows.append(result)
    atomic_json(folder/'rows.json', {'results': rows})
    hub.push(relative)
    return collect_tables(hub, campaign)


class TaskRunner:
    """One process owns one GPU; retain at most one CPU feature bank between jobs."""
    def __init__(self, config, sources, hub, device, *, smoke=False):
        self.config, self.sources, self.hub, self.device = config, sources, hub, device
        self.smoke = smoke
        self.cached = None

    def prepare(self, task):
        from .features import prepare, save_npz
        self.cached = None
        if not self.smoke:
            prepare(self.config, task.encoder, task.dataset, self.sources, self.hub, self.device)
            return
        from .smoke import fixture
        manifest, x, y, selected = fixture(self.hub, self.config, task.encoder, task.dataset)
        relative = f'features/synthetic/{task.dataset}/{task.encoder}/bank'
        save_npz(self.hub.root/relative/'data.npz', features=x, labels=y, selected=selected)
        atomic_json(self.hub.root/relative/'manifest.json', manifest)
        self.hub.push(relative)

    def bank(self, task):
        key = task.encoder, task.dataset
        if self.cached is not None and self.cached[0] == key:
            return self.cached[1]
        # Release the previous bank before loading the next one.
        self.cached = None
        if self.smoke:
            import numpy as np
            relative = f'features/synthetic/{task.dataset}/{task.encoder}/bank'
            folder = self.hub.pull(relative)
            manifest = read_json(folder/'manifest.json')
            if manifest is None:
                raise RuntimeError('synthetic preparation must finish before fitting')
            with np.load(folder/'data.npz', allow_pickle=False) as data:
                bank = manifest, data['features'], data['labels'], data['selected']
        else:
            from .features import prepare, training_bank
            _, manifest = prepare(self.config, task.encoder, task.dataset, self.sources, self.hub, self.device)
            bank = (manifest, *training_bank(manifest, self.hub, self.config))
        self.cached = key, bank
        return bank

    def run(self, task, emit):
        if task.stage == 'prepare':
            return self.prepare(task)
        import torch
        from .train import fit
        from .evaluate import evaluate
        manifest, features, labels, selected = self.bank(task)
        config = self.config.for_training(task.train_k) if task.train_k is not None else self.config
        model, record = None, None
        try:
            if task.method != 'dense':
                preset = replace(config.preset(), hidden_dim=max(64, 4*config.k, max(config.inference_ks)),
                                 epochs=1, batch_size=16, k_aux=32, ot_microbatch=4) if self.smoke else None
                model, record = fit(config, task.method, features, labels, manifest['identity'],
                                    self.hub, self.device, preset=preset)
            grid = sorted(config.inference_ks, key=lambda k: (k != config.k, k)) if model is not None else (None,)
            for k in grid:
                print(f'[job {task.dataset}/{task.encoder}/{task.method}] train K={task.train_k}; inference K={k}', flush=True)
                result = evaluate(config, task.encoder, task.dataset, task.method, model, record,
                                  features, labels, selected, manifest, self.hub, self.device, inference_k=k)
                emit(result)
        finally:
            del model
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
