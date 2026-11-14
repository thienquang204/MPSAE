from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest
import torch

from mpsae.io import atomic_json, sha256_file
from mpsae.patch_scene.artifacts import Hub
from mpsae.patch_scene.config import Config
from mpsae.patch_scene.report import expected_keys, row_key, write_tables
from mpsae.patch_scene.scheduler import run_parallel, validate_devices
from mpsae.patch_scene.__main__ import run_cell


def test_training_grid_validation_and_legacy_fit_settings():
    config = Config(train_k=(48, 3, 12), inference_k=(3, 12, 48), encoders=('dinov2', 'siglip2'))
    assert config.training_ks == (3, 12, 48)
    assert config.encoder_names == ('siglip2', 'dinov2')
    assert config.source_identity() == Config().identity()
    assert config.for_training(12).preset() == Config().preset()
    assert config.for_training(3).k == 3
    with pytest.raises(ValueError, match='not selected'):
        config.for_training(8)
    for bad in ([], [0], [12, 12], [True], [8193], '3,12,48'):
        with pytest.raises(ValueError, match='train_k'):
            Config(train_k=bad)
    for bad in ([], ['unknown'], ['dinov2', 'dinov2'], 'dinov2'):
        with pytest.raises(ValueError, match='encoders'):
            Config(encoders=bad)


def test_gpu_devices_are_unique_and_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert validate_devices(['cuda:0', 'cuda:1']) == ['cuda:0', 'cuda:1']
    assert validate_devices(['cpu', 'cpu'], smoke=True) == ['cpu', 'cpu']
    for devices in ([], ['cuda:2'], ['cuda', 'cuda:0'], ['cpu']):
        with pytest.raises(ValueError):
            validate_devices(devices)


def test_excluded_clip_is_not_resolved(monkeypatch, tmp_path):
    import huggingface_hub
    from mpsae.patch_scene.features import resolve_sources
    calls = []
    class API:
        def model_info(self, repo):
            calls.append(repo)
            return type('Info', (), {'sha': 'model-revision'})()
        def dataset_info(self, repo, revision):
            return type('Info', (), {'sha': revision})()
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    config = Config(train_k=(3, 12, 48), inference_k=(3, 12, 48), encoders=('siglip2', 'dinov2'))
    sources = resolve_sources(config, Hub(tmp_path))
    assert len(calls) == 2 and not any('clip' in repo.lower() for repo in calls)
    assert set(sources['models']) == {'siglip2', 'dinov2'}


def test_matched_and_cross_k_tables_keep_checkpoints_separate(tmp_path):
    config = Config(train_k=(3, 12, 48), inference_k=(3, 12, 48), encoders=('siglip2', 'dinov2'))
    rows = []
    for d, e, m, t, k in expected_keys(config):
        value = .6 if m == 'dense' else .3 + t/1000 + k/10000 + {'csr': 0, 'csrv2': .01, 'mpsae': .02}[m]
        rows.append(dict(dataset=d, encoder=e, method=m, train_k=t, inference_k=k, k=k,
                         complete=True, seed=1227, miou=value, pixel_accuracy=value+.1))
    write_tables(rows, tmp_path, config)
    data = json.loads((tmp_path/'results.json').read_text())
    assert data['complete'] and data['completed'] == data['expected'] == 112
    assert data['main_table'] == 'matched_budgets' and data['train_ks'] == [3, 12, 48]
    main = (tmp_path/'results.tex').read_text()
    sweep = (tmp_path/'inference_sweep.tex').read_text()
    assert 'Matched budgets' in main and 'CLIP' not in main and 'CLIP' not in sweep
    assert r'$K_{\mathrm{train}}$' in sweep
    assert main.count(r'\textbf{') == 12 and sweep.count(r'\textbf{') == 36
    with (tmp_path/'comparisons.csv').open(newline='') as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 36
    chosen = [r for r in comparisons if r['dataset']=='voc' and r['encoder']=='dinov2' and r['inference_k']=='12']
    assert len(chosen) == 3
    assert len({r['mpsae_miou_percent'] for r in chosen}) == 3
    assert all(float(r['mpsae_gain_over_best_baseline_pp']) == pytest.approx(1) for r in chosen)


def test_parallel_grid_reuses_old_k12_and_matches_serial(tmp_path):
    torch.set_num_threads(1)
    old = Config(inference_k=(3, 12, 48), encoders=('dinov2',))
    config = replace(old, train_k=(3, 12, 48))
    hub = Hub(tmp_path/'parallel')
    def settings(cfg, target):
        campaign = cfg.plan()['campaign']
        atomic_json(target.root/f'campaigns/{campaign}/settings/config.json', cfg.plan())
        return campaign
    run_cell(old, 'dinov2', 'voc', {}, hub, 'cpu', settings(old, hub), smoke=True)
    old_paths = [*(hub.root/'fits').glob('*/checkpoint.pt'), *(hub.root/'evaluations').glob('*/result.json')]
    old_hashes = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in old_paths}
    assert len(old_paths) == 13  # 3 fits and 10 evaluations
    campaign = settings(config, hub)
    run_parallel(config, {}, hub, campaign, ['dinov2'], ['voc'], ['cpu', 'cpu'], smoke=True, cpu_threads=1)
    receipt = json.loads((hub.root/f'campaigns/{campaign}/table/results.json').read_text())
    assert receipt['completed'] == 28 and receipt['expected'] == 56  # only VOC selected
    assert len(list((hub.root/'fits').glob('*/checkpoint.pt'))) == 9
    assert len(list((hub.root/'evaluations').glob('*/result.json'))) == 28
    assert all((sha256_file(p), p.stat().st_mtime_ns) == old_hashes[p] for p in old_paths)
    status = json.loads((hub.root/f'campaigns/{campaign}/scheduler.json').read_text())
    assert status['stage'] == 'complete' and status['completed_jobs'] == status['total_jobs'] == 11
    assert len(status['devices']) == 2
    serial = Hub(tmp_path/'serial')
    run_cell(config, 'dinov2', 'voc', {}, serial, 'cpu', settings(config, serial), smoke=True)
    serial_rows = json.loads((serial.root/f'campaigns/{campaign}/table/results.json').read_text())['results']
    reference = {row_key(r): r for r in serial_rows}
    for row in receipt['results']:
        assert row['confusion'] == reference[row_key(row)]['confusion']
        assert row['mean_nnz'] == reference[row_key(row)]['mean_nnz']
        if row['method'] != 'dense':
            assert row['mean_nnz'] <= row['inference_k'] and row['gallery_mean_nnz'] <= row['inference_k']
    def weights(root):
        result = {}
        for p in (root/'fits').glob('*/checkpoint.pt'):
            record = torch.load(p, map_location='cpu', weights_only=False)
            result[(record['identity']['method'], record['identity']['k'])] = record['model']
        return result
    a, b = weights(hub.root), weights(serial.root)
    for key in a:
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    # Re-running the coordinator must reuse every checkpoint and evaluation.
    before = {p: p.stat().st_mtime_ns for p in (hub.root/'fits').glob('*/checkpoint.pt')}
    run_parallel(config, {}, hub, campaign, ['dinov2'], ['voc'], ['cpu', 'cpu'], smoke=True, cpu_threads=1)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())
    assert json.loads((hub.root/f'campaigns/{campaign}/table/results.json').read_text())['completed'] == 28
