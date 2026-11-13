"""Real tiny SAE updates, interrupted resume, and segmentation using the adapted weights."""
from dataclasses import replace
from pathlib import Path
import copy
import json

import numpy as np
import pytest
import torch

from mpsae import methods
from mpsae.config import BACKBONES
from mpsae.io import atomic_json, canonical_hash, sha256_file
from mpsae.model import build_sae
from mpsae.scene import adapt
from mpsae.scene.config import Adaptation, SceneConfig, load_config


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    preset = replace(BACKBONES['ff2048'].preset, hidden_dim=8, batch_size=8,
                     ot_microbatch=4, ot_max_iters=100, ot_tol=1e-3, ot_max_refine_iters=2000)
    monkeypatch.setitem(BACKBONES, 'ff2048', replace(BACKBONES['ff2048'], dim=4, preset=preset,
                                                  k_list=(1, 2, 4), e2_anchors=(2,)))
    settings = Adaptation(enabled=True, epochs=2, batch_size=8, lr=1e-3,
                          checkpoint_every=2, log_every=1, residency='cpu')
    config = SceneConfig(train_k=2, inference_k=(1, 2, 4), neighbors=2,
                         adaptation=settings, workers=0)
    features = np.random.default_rng(3).normal(size=(24, 4)).astype(np.float32)
    labels = np.zeros((24, 21), dtype=np.float32)
    labels[np.arange(24), np.arange(24) % 3] = .75
    labels[np.arange(24), (np.arange(24)+1) % 3] = .25
    records = []
    for cell in config.cells():
        model = build_sae(4, 8, seed=cell.seed)
        model.stats_last_nonzero.fill_(777)  # must be reset at adaptation start
        objective = methods.describe(cell.method, cell.k, preset)
        identity = {'backbone': cell.backbone, 'method': cell.method, 'target_k': cell.k,
                    'seed': cell.seed, 'objective': objective,
                    'optimizer': {'name': 'Adam', 'lr': 4e-5, 'weight_decay': 1e-4,
                                  'eps': preset.adam_eps, 'betas': list(preset.adam_betas)}}
        signature = canonical_hash(identity)
        folder = tmp_path / 'source' / cell.method
        folder.mkdir(parents=True)
        path = folder / 'checkpoint.pt'
        torch.save({'identity': identity, 'signature': signature, 'model': model.state_dict(),
                    'next_step': 100, 'optimizer': {'must_not_be_loaded': True}}, path)
        atomic_json(folder / 'summary.json', {'complete': True, 'identity': identity,
                                             'signature': signature, 'steps': 100})
        records.append({'path': str(path), 'checkpoint_id': sha256_file(path),
            'signature': signature, 'training_identity': identity, 'cell': cell.__dict__,
            'backbone_weights': {'fixture': True}})
    return config, features, labels, records


@pytest.mark.parametrize('position', [0, 1, 2])
def test_actual_objective_changes_only_sae_and_preserves_source(tiny, tmp_path, monkeypatch, position):
    config, features, labels, records = tiny
    cell, record = config.cells()[position], records[position]
    original = torch.load(record['path'], weights_only=False)['model']
    calls = []
    real = methods.compute_step
    def step(method, model, x, **kwargs):
        if not calls:
            assert model.stats_last_nonzero.sum() == 0
        assert method == cell.method and kwargs['fixed_budget'] is True
        assert kwargs['k'] == cell.k
        if method == 'csrv2':
            assert set(kwargs['labels'].tolist()) <= {0, 1, 2}
        else:
            assert kwargs['labels'] is None
        calls.append(x.detach().numpy().copy())
        return real(method, model, x, **kwargs)
    monkeypatch.setattr(methods, 'compute_step', step)
    result = adapt.fit(config, cell, record, features, labels, {'split': 'train'}, tmp_path/'fit', device='cpu')
    model = adapt.load_adapted(result, device='cpu')
    assert result['steps'] == 6 and len(calls) == 6
    assert not torch.equal(model.encoder.weight, original['encoder.weight'])
    assert sha256_file(record['path']) == record['checkpoint_id']
    assert result['identity']['preset']['lr'] == 1e-3
    assert result['patch_labels']['mean_purity'] == .75
    for k in (1, 2, 4):
        z = model.encode(torch.from_numpy(features), k)
        assert (z != 0).sum(1).max() <= k
    monkeypatch.setattr(methods, 'compute_step', lambda *a, **kw: pytest.fail('completed fit should be reused'))
    assert adapt.fit(config, cell, record, features, labels, {'split': 'train'}, tmp_path/'fit', device='cpu') == result


@pytest.mark.parametrize('position', [0, 1, 2])
def test_resume_matches_uninterrupted(tiny, tmp_path, monkeypatch, position):
    config, features, labels, records = tiny
    cell, record = config.cells()[position], records[position]
    full = adapt.fit(config, cell, record, features, labels, {}, tmp_path/'full', device='cpu')
    real = methods.compute_step
    calls = []
    def interrupted(*a, **kw):
        calls.append(kw['step'])
        if len(calls) == 3:
            raise RuntimeError('simulated interruption')
        return real(*a, **kw)
    monkeypatch.setattr(methods, 'compute_step', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        adapt.fit(config, cell, record, features, labels, {}, tmp_path/'resumed', device='cpu')
    monkeypatch.setattr(methods, 'compute_step', real)
    resumed = adapt.fit(config, cell, record, features, labels, {}, tmp_path/'resumed', device='cpu')
    a, b = adapt.load_adapted(full, device='cpu'), adapt.load_adapted(resumed, device='cpu')
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    lines = [json.loads(s) for s in (Path(resumed['path']).parent/'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in lines] == list(range(1, 7))


def test_source_coefficients_and_new_identity(tiny, tmp_path):
    config, features, labels, records = tiny
    changed = copy.deepcopy(records[2])
    changed['training_identity']['objective']['ot']['weight'] = 1.25
    changed['training_identity']['objective']['ot']['max_refine_iters'] = 4321
    preset = adapt.source_preset(config.cells()[2], changed, config.adaptation)
    assert preset.ot_weight == 1.25 and preset.ot_max_refine_iters == 4321
    first = adapt.fit(config, config.cells()[0], records[0], features, labels, {}, tmp_path/'runs', device='cpu')
    config = replace(config, adaptation=replace(config.adaptation, epochs=1))
    second = adapt.fit(config, config.cells()[0], records[0], features, labels, {}, tmp_path/'runs', device='cpu')
    assert first['path'] != second['path'] and second['steps'] == 3


def test_label_and_batch_policy():
    labels = np.zeros((2, 21), np.float32)
    labels[0, :2] = .5
    labels[1, 2] = 1
    targets, stats = adapt.patch_targets(labels)
    assert targets.tolist() == [0, 2] and stats['mixed_fraction'] == .5
    assert adapt.batch_ranges(17, 8) == [(0, 8), (8, 17)]
    for invalid in (np.zeros((2, 21)), np.full((2, 21), np.nan), np.ones((2, 20))):
        with pytest.raises(ValueError):
            adapt.patch_targets(invalid)


@pytest.mark.parametrize('values', [dict(enabled='yes'), dict(batch_size=1), dict(epochs=0),
                                    dict(lr=float('nan')), dict(residency='automatic')])
def test_invalid_adaptation_settings(values):
    with pytest.raises(ValueError):
        Adaptation(**values)


def test_corrupt_completed_checkpoint_refused(tiny, tmp_path):
    config, features, labels, records = tiny
    record, cell = records[0], config.cells()[0]
    result = adapt.fit(config, cell, record, features, labels, {}, tmp_path/'fit', device='cpu')
    with open(result['path'], 'ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(RuntimeError, match='inconsistent'):
        adapt.fit(config, cell, record, features, labels, {}, tmp_path/'fit', device='cpu')
    with pytest.raises(RuntimeError, match='bytes changed'):
        adapt.load_adapted(result, device='cpu')


def test_frozen_config_identity_preserved():
    root = Path(__file__).resolve().parents[1]
    old = load_config(root/'configs/archive/scene_voc_frozen_multi.yaml')
    assert old.plan()['training_runs'] == 0 and old.plan()['evaluations'] == 82
    assert 'adaptation' not in old.groups[0].plan()['config']
    current = load_config(root/'configs/scene_voc.yaml')
    assert current.plan()['training_runs'] == 12 and current.plan()['evaluations'] == 160
    assert current.groups[0].adaptation.lr == 1e-5


def test_adapted_segmentation_and_resume_without_retraining(tiny, tmp_path, monkeypatch):
    from mpsae.scene import run as runner
    from mpsae.scene.data import save_arrays
    config, features, labels, records = tiny
    cache = tmp_path/'cache'
    save_arrays(cache/'validation'/'00000.npz', features=features[:2], truth=np.array([[0, 1]]))
    manifest = {'identity': {'fixture': True}, 'grid': [1, 2], 'support_images': 1, 'query_images': 1}
    monkeypatch.setattr(runner, 'source_records', lambda *a: records)
    monkeypatch.setattr(runner, 'prepare', lambda *a, **kw: (cache, manifest))
    monkeypatch.setattr(runner, 'support_bank', lambda *a: (features, labels))
    seen = []
    real_predict = runner.predict
    def predict(bank, target, queries, **kwargs):
        if isinstance(bank, dict):
            assert bank['values'].shape[1] == queries['values'].shape[1]
            assert len(bank['values']) == len(features)
        seen.append(True)
        return real_predict(bank, target, queries, **kwargs)
    monkeypatch.setattr(runner, 'predict', predict)
    publications = []
    result = runner.run(config, tmp_path/'scene', tmp_path, device='cpu', on_adapted=publications.append)
    assert len(result['results']) == 19 and len(seen) == 19 and len(publications) == 3
    assert sum(r['variant'] == 'adapted' for r in result['results']) == 9
    assert sum(r['variant'] == 'frozen' for r in result['results']) == 9
    for name in ('segmentation.pdf', 'retrieval.pdf', 'adaptation_gain.pdf', 'search_time.pdf', 'results.csv'):
        assert (Path(result['report'])/name).is_file()
    monkeypatch.setattr(methods, 'compute_step', lambda *a, **kw: pytest.fail('must not retrain'))
    monkeypatch.setattr(runner, 'predict', lambda *a, **kw: pytest.fail('must not rerun retrieval'))
    again = runner.run(config, tmp_path/'scene', tmp_path, device='cpu')
    assert again['results'] == result['results']
    payload = json.loads((Path(result['report'])/'results.json').read_text())
    assert len(payload['adaptations']) == 3
    from mpsae.scene.report import suite_report
    combined = suite_report([{'report': result['report']}], tmp_path/'scene', config.plan())
    merged = json.loads(combined.read_text())
    assert len(merged['results']) == 19 and len(merged['adaptations']) == 3
    assert {r['backbone'] for r in merged['results']} == {'ff2048'}


def test_prepare_never_trains_and_adapted_only_mode(tiny, tmp_path, monkeypatch):
    from mpsae.scene import run as runner
    from mpsae.scene.data import save_arrays
    config, features, labels, records = tiny
    config = replace(config, methods=('csr',), inference_k=(2,),
                     adaptation=replace(config.adaptation, compare_frozen=False))
    cache = tmp_path/'cache'
    save_arrays(cache/'validation'/'00000.npz', features=features[:2], truth=np.array([[0, 1]]))
    manifest = {'identity': {}, 'grid': [1, 2], 'support_images': 1, 'query_images': 1}
    monkeypatch.setattr(runner, 'source_records', lambda *a: records[:1])
    monkeypatch.setattr(runner, 'prepare', lambda *a, **kw: (cache, manifest))
    monkeypatch.setattr(runner, 'support_bank', lambda *a: pytest.fail('prepare cannot train'))
    assert runner.run(config, tmp_path/'scene', tmp_path, device='cpu', prepare_only=True)['complete']
    monkeypatch.setattr(runner, 'support_bank', lambda *a: (features, labels))
    result = runner.run(config, tmp_path/'scene', tmp_path, device='cpu')
    assert config.plan()['evaluations'] == 2
    assert {r['variant'] for r in result['results']} == {'dense', 'adapted'}


def test_validation_masks_are_opened_only_after_all_fits(tiny, tmp_path, monkeypatch):
    from mpsae.scene import run as runner
    config, features, labels, records = tiny
    # There is deliberately no validation directory: training must finish before scoring opens it.
    monkeypatch.setattr(runner, 'source_records', lambda *a: records)
    monkeypatch.setattr(runner, 'prepare', lambda *a, **kw: (tmp_path/'missing',
        {'identity': {}, 'grid': [1, 2], 'support_images': 1, 'query_images': 1}))
    monkeypatch.setattr(runner, 'support_bank', lambda *a: (features, labels))
    completed = []
    with pytest.raises(FileNotFoundError):
        runner.run(config, tmp_path/'scene', tmp_path, device='cpu', on_adapted=completed.append)
    assert len(completed) == 3 and all(r['complete'] for r in completed)


def test_publishing_includes_recoverable_checkpoints(tiny, tmp_path, monkeypatch):
    import huggingface_hub
    from types import SimpleNamespace
    from mpsae.scene.server import publish_adaptation, publish_results
    config, features, labels, records = tiny
    result = adapt.fit(config, config.cells()[0], records[0], features, labels, {}, tmp_path/'fit', device='cpu')
    calls = []
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        upload_folder=lambda **kw: calls.append(kw) or 'commit'))
    publish_adaptation(Path(result['path']).parent, 'test/models')
    assert set(calls[0]['allow_patterns']) == {'checkpoint.pt', 'config.json', 'summary.json', 'train.jsonl'}
    assert calls[0]['path_in_repo'].startswith('scene-v1/adaptations/')
    with pytest.raises(ValueError):
        publish_adaptation(Path(result['path']).parent, 'test/models', expected_checkpoint='wrong')
    report = tmp_path/'fit/reports/fixture/results.json'
    atomic_json(report, {'complete': True, 'adaptations': [{'signature': result['signature'],
        'checkpoint_id': result['checkpoint_id'], 'directory': f'adaptations/{result["signature"][:20]}'}]})
    publish_results(report, 'test/models')
    assert calls[-2]['path_in_repo'].startswith('scene-v1/adaptations/')
    assert calls[-1]['path_in_repo'] == 'scene-v1/reports/fixture'
    assert '*.pdf' in calls[-1]['allow_patterns'] and 'figure_notes.txt' in calls[-1]['allow_patterns']
