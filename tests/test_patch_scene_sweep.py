"""Inference-grid regression checks: exact reuse, distinct budgets and honest tables."""
from dataclasses import replace
import csv
import json

import pytest
import torch

from mpsae.io import atomic_json, canonical_hash, sha256_file
from mpsae.patch_scene.config import Config, load
from mpsae.patch_scene.artifacts import Hub
from mpsae.patch_scene.report import expected_keys, write_tables


def test_shipped_configs_keep_old_campaign_and_extend_only_inference():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    old = load(root/'configs/patch_scene_kaggle_k12.yaml')
    new = load(root/'configs/patch_scene_kaggle.yaml')
    assert old.plan()['campaign'] == '54d9e25955a1f70142da'
    assert new.source_identity() == old.identity()
    assert new.preset() == old.preset()
    assert (new.k, new.inference_ks, len(expected_keys(new))) == (12, (3, 12, 48), 60)


def test_sweep_reuses_pinned_sources_without_network(monkeypatch, tmp_path):
    import huggingface_hub
    from mpsae.patch_scene.features import resolve_sources, feature_identity
    config = Config(inference_k=(3, 12, 48))
    sources = {'models': {'dinov2': 'old-model-sha'}, 'datasets': {'voc': 'old-data-sha'}}
    original = tmp_path/'campaigns'/Config().plan()['campaign']/'sources/sources.json'
    atomic_json(original, sources)
    def no_network(*args, **kwargs):
        raise AssertionError('An inference sweep must reuse the existing pinned revisions')
    monkeypatch.setattr(huggingface_hub, 'HfApi', no_network)
    assert resolve_sources(config, Hub(tmp_path)) == sources
    copied = tmp_path/'campaigns'/config.plan()['campaign']/'sources/sources.json'
    assert json.loads(copied.read_text()) == sources
    # This identity check is offline; do not require the dataset-download package.
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'fixed-test-version')
    assert feature_identity(config, 'dinov2', 'voc', sources, 'cpu') == feature_identity(
        Config(), 'dinov2', 'voc', sources, 'cpu')


def test_extended_cell_never_retrains_or_repeats_cached_evaluations(monkeypatch, tmp_path):
    from mpsae import methods
    from mpsae.patch_scene import evaluate as evaluator
    from mpsae.patch_scene.__main__ import run_cell
    torch.set_num_threads(1)
    hub = Hub(tmp_path)
    old = Config()
    new = replace(old, inference_k=(3, 12, 48))
    def run(config):
        campaign = canonical_hash(config.identity())[:20]
        atomic_json(tmp_path/f'campaigns/{campaign}/settings/config.json', config.plan())
        run_cell(config, 'dinov2', 'voc', {}, hub, 'cpu', campaign, smoke=True)
        return json.loads((tmp_path/f'campaigns/{campaign}/table/results.json').read_text())
    assert run(old)['completed'] == 4
    fits = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in (tmp_path/'fits').glob('*/checkpoint.pt')}
    cached = {str(p): (sha256_file(p), p.stat().st_mtime_ns) for p in (tmp_path/'evaluations').glob('*/result.json')}
    assert len(fits) == 3 and len(cached) == 4
    def no_training(*args, **kwargs):
        raise AssertionError('Changing inference K must not call the training loss')
    monkeypatch.setattr(methods, 'compute_step', no_training)
    query_calls = []
    original_queries = evaluator.queries
    def queries(*args, **kwargs):
        query_calls.append(1)
        yield from original_queries(*args, **kwargs)
    monkeypatch.setattr(evaluator, 'queries', queries)
    receipt = run(new)
    assert receipt['completed'] == 10 and receipt['expected'] == 60
    assert len(query_calls) == 6  # only K=3 and K=48 for three methods
    assert len(list((tmp_path/'fits').glob('*/checkpoint.pt'))) == 3
    for path, (digest, mtime) in {**fits, **cached}.items():
        from pathlib import Path
        assert sha256_file(path) == digest and Path(path).stat().st_mtime_ns == mtime
    assert sum(r['method'] == 'dense' for r in receipt['results']) == 1
    for row in receipt['results']:
        if row['method'] != 'dense':
            assert row['train_k'] == 12 and row['k'] == row['inference_k']
            assert row['mean_nnz'] <= row['inference_k']
            assert row['gallery_mean_nnz'] <= row['inference_k']
    # A second run is fully cached and cannot duplicate report rows.
    query_calls.clear()
    again = run(new)
    assert again['completed'] == 10 and query_calls == []


def _rows(config):
    rows = []
    for dataset, encoder, method, train_k, k in sorted(expected_keys(config)):
        # MPSAE wins at K=3, loses at K=48, and ties at K=12 (display precision).
        score = {'dense': .6, 'csr': .41, 'csrv2': .45, 'mpsae': .47}[method]
        if method == 'mpsae':
            score = {3: .47, 12: .45001, 48: .43}[k]
        rows.append(dict(dataset=dataset, encoder=encoder, method=method, train_k=train_k,
                         inference_k=k, k=k, seed=config.seed, complete=True,
                         miou=score, pixel_accuracy=score+.1))
    return rows


def test_paper_tables_deltas_ties_missing_and_completion(tmp_path):
    config = Config(inference_k=(3, 12, 48))
    rows = _rows(config)
    write_tables(rows, tmp_path, config)
    receipt = json.loads((tmp_path/'results.json').read_text())
    assert receipt['complete'] and receipt['completed'] == receipt['expected'] == 60
    assert receipt['missing'] == []
    table = (tmp_path/'results.tex').read_text()
    assert r'\toprule' in table and r'\bottomrule' in table and r'\hline' not in table
    assert r'45.00 & \textbf{47.00} & +2.00' in table
    assert r'\textbf{45.00} & \textbf{45.00} & +0.00' in table
    assert r'\textbf{45.00} & 43.00 & -2.00' in table
    assert r'\textbf{60.00}' not in table  # dense excluded from sparse ranking
    assert 'Partial results' not in table
    with (tmp_path/'comparisons.csv').open(newline='') as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 18
    for row in comparisons:
        if row['inference_k'] == '48':
            assert float(row['mpsae_gain_over_best_baseline_pp']) == pytest.approx(-2)
    # Remove one baseline: that row must not claim a winner or delta.
    partial = [r for r in rows if not (r['dataset'] == 'voc' and r['encoder'] == 'siglip2'
               and r['method'] == 'csr' and r['k'] == 3)]
    write_tables(partial, tmp_path, config)
    table = (tmp_path/'results.tex').read_text()
    assert 'Partial results (59/60 evaluations)' in table
    assert '60.00 & -- & 45.00 & 47.00 & --' in table
    assert not json.loads((tmp_path/'results.json').read_text())['complete']
    with pytest.raises(ValueError, match='duplicate'):
        write_tables(rows+[rows[0]], tmp_path, config)
    with pytest.raises(ValueError, match='outside'):
        write_tables([{**rows[0], 'seed': 42}], tmp_path, config)
    with pytest.raises(ValueError, match='invalid miou'):
        write_tables([{**rows[0], 'miou': float('nan')}], tmp_path, config)
