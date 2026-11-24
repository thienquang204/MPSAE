import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_exact_79_and_preserved_26_training_recipes():
    from mpsae.benchmark.ablation79 import build_plan
    from mpsae.benchmark.retrieval_studies import conditions
    from dataclasses import asdict
    from mpsae.io import canonical_hash
    plan = build_plan(ROOT/'configs/retrieval_studies.yaml', ROOT/'configs/linear_scene_voc.yaml')
    assert plan['evaluations'] == 79
    assert plan['retrieval']['study_conditions'] == {'mass': 16, 'optimizer': 10, 'extreme': 2}
    assert plan['retrieval']['study_evaluations'] == {'mass': 48, 'optimizer': 20, 'extreme': 4}
    assert plan['retrieval']['unique_fits'] == 28
    assert plan['segmentation']['evaluations'] == plan['linear_head_fits'] == 7
    assert plan['sae_fits_total'] == 34  # 26 old + at most 8 new.
    def recipes(name):
        settings = yaml.safe_load((ROOT/'configs'/name).read_text())
        return {canonical_hash({'cell': c.slug, 'preset': asdict(conf.spec(c, s).preset)})
                for _, conf in conditions(settings) for c, s in conf.cells()}
    assert recipes('retrieval_studies_26.yaml') < recipes('retrieval_studies.yaml')


def test_cross_k_extension_reuses_weights_and_matched_evaluations(tmp_path):
    from mpsae.benchmark.retrieval_studies import main
    torch.set_num_threads(1)
    settings = yaml.safe_load((ROOT/'configs/retrieval_studies_26.yaml').read_text())
    settings.update(train_k={'ff2048': [3]}, mass_values=[1.], optimizer_weights=[2.])
    config = tmp_path/'s.yaml'
    config.write_text(yaml.safe_dump(settings))
    args = ['--config', str(config), '--workspace', str(tmp_path/'work'), '--source-root', str(tmp_path/'source'),
            '--smoke', '--device', 'cpu', '--cpu-threads', '1']
    assert main(args) == 0
    root = tmp_path/'work/SMOKE_ONLY'
    paths = list((root/'fits').glob('*/checkpoint.pt')) + list((root/'evaluations').glob('*/result.json'))
    stamps = {p: p.stat().st_mtime_ns for p in paths}
    settings['inference'] = {'mass': {'ff2048': {3: [2,3,4]}}, 'optimizer': {'ff2048': {3: [2,3]}}}
    settings['extreme'] = dict(enabled=True, mass=1., weight=2., batch_size=1024, inference_k={'ff2048': [2,3]})
    config.write_text(yaml.safe_dump(settings))
    assert main(args) == 0
    assert all(p.stat().st_mtime_ns == t for p,t in stamps.items())
    assert len(list((root/'fits').glob('*/checkpoint.pt'))) == 3
    with ZipFile(root/'retrieval-studies-results.zip') as z:
        result = json.loads(z.read('results.json'))
        assert result['completed'] == result['expected'] == 7
        assert len(result['results']) == 7
        assert {'mass_inference.pdf','weight_accuracy.pdf','optimizer_inference.pdf',
                'k2_training_comparison.pdf','mass_heatmap.pdf','extreme_table.tex'} <= set(z.namelist())
        assert '$K_{inf}$' in z.read('mass_table.tex').decode()


def test_sparse_linear_matches_dense_logits_and_parameter_gradients():
    from mpsae.patch_scene.linear_probe import logits
    torch.manual_seed(12)
    values = torch.tensor([[.2,.9], [0.,0.], [.5,.2]])
    indices = torch.tensor([[1,4], [0,3], [2,1]])
    x = torch.zeros(3,6).scatter(1, indices, values)
    head = torch.nn.Linear(6,4)
    sparse = logits(head, values, indices)
    dense = head(x)
    assert torch.allclose(sparse, dense, atol=1e-7)
    targets = torch.rand(3,4); targets /= targets.sum(1,keepdim=True)
    loss = lambda out: -(targets * out.log_softmax(-1)).sum(1).mean()
    g1 = torch.autograd.grad(loss(sparse), tuple(head.parameters()))
    g2 = torch.autograd.grad(loss(dense), tuple(head.parameters()))
    for a,b in zip(g1,g2):
        assert torch.allclose(a,b,atol=1e-7)
    assert torch.equal(sparse[1], head.bias)  # Declared standard affine empty-code policy.


def test_linear_segmentation_full_smoke_and_resume(tmp_path):
    from mpsae.patch_scene.linear_probe import main
    torch.set_num_threads(1)
    args = ['--config', str(ROOT/'configs/linear_scene_voc.yaml'), '--workspace', str(tmp_path),
            '--smoke', '--device', 'cpu', '--cpu-threads', '1']
    assert main(args) == 0
    root = tmp_path/'SMOKE_ONLY'
    checkpoints = list((root/'fits').glob('*/checkpoint.pt')) + list((root/'linear-heads').glob('*/head.pt'))
    assert len(checkpoints) == 13
    stamps = {p:p.stat().st_mtime_ns for p in checkpoints}
    with ZipFile(root/'linear-segmentation-results.zip') as z:
        r = json.loads(z.read('results.json'))
        assert r['complete'] and r['completed'] == 7
        assert not r['plan']['scientific']
        for row in r['results']:
            assert 0 <= row['miou'] <= 1
            assert row['training_rows'] == 64
            assert len(row['history']) == 2
            if row['method'] != 'dense':
                assert row['mean_nnz'] <= row['train_k']
        assert {'linear_table.tex', 'linear_results.csv', 'voc_siglip2_linear_accuracy.pdf',
                'voc_siglip2_class_iou.pdf', 'voc_siglip2_probe_convergence.pdf'} <= set(z.namelist())
    assert main(args) == 0
    assert all(p.stat().st_mtime_ns == t for p,t in stamps.items())
    assert main(args+['--export-only']) == 0


def test_probe_head_rejects_void_training_targets(tmp_path):
    from mpsae.patch_scene.linear_probe import train_head, ProbeConfig
    with pytest.raises(ValueError, match='nonvoid'):
        train_head(np.ones((2,4),dtype='float32'), None, np.zeros((2,3),dtype='float32'),
                   4, {}, ProbeConfig(epochs=1), tmp_path, 'cpu')


def test_k2_histogram_counts_real_positive_entries():
    from mpsae.model import SparseAutoencoder
    from mpsae.evaluate.encode import encode_split
    model = SparseAutoencoder(4,4)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(4))
    x = np.array([[1,2,-1,-1], [-1,-2,-3,-4], [1,-1,-2,-3]], dtype='float32')
    encoded = encode_split(model, x, None, 2, device='cpu', batch_size=2)
    assert encoded['stats']['nnz_histogram'] == [1,1,1]
    assert encoded['stats']['nnz_mean'] == 1


def test_linear_head_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from mpsae.patch_scene import linear_probe as probe
    rng = np.random.default_rng(22)
    values = rng.normal(size=(24,4)).astype('float32')
    labels = np.eye(3, dtype='float32')[np.arange(24) % 3]
    settings = probe.ProbeConfig(epochs=3, batch_size=7)
    full, _ = probe.train_head(values, None, labels, 4, {'test': 1}, settings, tmp_path/'full', 'cpu')
    original = probe.save_torch
    def interrupt(path, payload):
        original(path, payload)
        if payload['next_epoch'] == 1:
            raise KeyboardInterrupt('after first durable epoch')
    monkeypatch.setattr(probe, 'save_torch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        probe.train_head(values, None, labels, 4, {'test': 1}, settings, tmp_path/'resume', 'cpu')
    monkeypatch.setattr(probe, 'save_torch', original)
    resumed, record = probe.train_head(values, None, labels, 4, {'test': 1}, settings, tmp_path/'resume', 'cpu')
    assert len(record['history']) == 3
    for a,b in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(a,b)


def test_combined_export_with_missing_stages_marks_pending(tmp_path):
    from mpsae.benchmark.ablation79 import build_plan, package
    plan = build_plan(ROOT/'configs/retrieval_studies.yaml', ROOT/'configs/linear_scene_voc.yaml')
    result = package(tmp_path, tmp_path/'segmentation', plan)
    assert result['completed'] == 0 and result['expected'] == 79 and not result['complete']
    with ZipFile(tmp_path/'ablation79-results.zip') as z:
        assert {'plan.json','status.json'} <= set(z.namelist())
