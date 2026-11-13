from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mpsae.patch_scene.config import Config
from mpsae.patch_scene.artifacts import Hub, seal, verify
from mpsae.patch_scene.features import storage_cast, save_npz, queries
from mpsae.patch_scene.retrieval import stable_topk, Search, upsample_prediction
from mpsae.patch_scene.train import fit
from mpsae.patch_scene.report import write_tables
from mpsae.patch_scene.evaluate import evaluate
from mpsae.patch_scene.smoke import fixture
from mpsae.config import TrainPreset


def test_plan_24_and_reject_sweeps():
    p = Config().plan()
    assert (p['sae_fits'], p['total_evaluations'], p['inference_k']) == (18, 24, [12])
    with pytest.raises(ValueError):
        Config(k=48)


@pytest.mark.parametrize('depth', [1, 3, 7])
def test_topk_matches_stable_full_sort_with_boundary_ties(depth):
    scores = torch.tensor([[0., 1., 0., 1., 1., -1., 0.], [0.]*7, [3., 2., 1., 0., -1., -2., -3.]])
    val, idx = stable_topk(scores, depth)
    expected = scores.argsort(dim=1, descending=True, stable=True)[:, :depth]
    assert torch.equal(idx, expected)
    assert torch.equal(val, scores.gather(1, expected))


def test_sparse_lookup_matches_dense_cosine_prediction():
    torch.manual_seed(3)
    raw = torch.randn(32, 17).relu()
    selected = raw.topk(3, dim=1)
    dense = torch.zeros_like(raw).scatter_(1, selected.indices, selected.values)
    dense[0] = 0
    selected.values[0] = 0
    labels = np.eye(3, dtype='float32')[np.arange(32) % 3]
    args = dict(device='cpu', dim=17, neighbors=5, temperature=.2, query_block=3)
    sparse = Search((selected.indices, selected.values), labels, sparse=True, **args)
    normal = Search(dense, labels, sparse=False, **args)
    a, empty_a = sparse.predict((selected.indices[:9], selected.values[:9]), (3, 3), (13, 17))
    b, empty_b = normal.predict(dense[:9], (3, 3), (13, 17))
    np.testing.assert_array_equal(a, b)
    assert empty_a == empty_b == 1


def test_tiled_upsample_matches_full_reference_and_void():
    torch.manual_seed(12)
    probs = torch.rand(12, 150).softmax(1)
    empty = torch.zeros(12, dtype=torch.bool)
    empty[3] = True
    probs[empty] = 0
    shape = (179, 203)
    actual = upsample_prediction(probs, empty, (3, 4), shape, tile_rows=11)
    expected = F.interpolate(probs.T.reshape(1, 150, 3, 4), size=shape,
                             mode='bilinear', align_corners=False).argmax(1)[0]
    mask = F.interpolate(empty.float().reshape(1, 1, 3, 4), size=shape, mode='nearest')[0, 0].bool()
    expected[mask] = 150
    np.testing.assert_array_equal(actual, expected.numpy())


def test_cache_precision_and_corruption(tmp_path):
    values = np.random.default_rng(1).normal(size=(10, 768)).astype('float32')
    stored, error = storage_cast(values, 'float16')
    assert error < 1e-5 and stored.dtype == np.float16
    save_npz(tmp_path/'data.npz', features=stored)
    seal(tmp_path)
    assert verify(tmp_path)
    (tmp_path/'data.npz').write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='Corrupt'):
        verify(tmp_path)


@pytest.mark.parametrize('method', ['csr', 'csrv2', 'mpsae'])
def test_training_resume_matches_uninterrupted(tmp_path, method):
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(17, 8)).astype('float32')
    labels = np.eye(3, dtype='float32')[np.arange(17) % 3]
    cfg = Config(checkpoint_every=1)
    preset = TrainPreset(name='test', hidden_dim=64, batch_size=8, epochs=2,
                         k_aux=16, ot_microbatch=4, ot_max_iters=40, ot_max_refine_iters=80)
    args = (cfg, method, x, labels, {'fixture': True})
    full, _ = fit(*args, Hub(tmp_path/'full'), 'cpu', preset=preset)
    partial = Hub(tmp_path/'resume')
    assert fit(*args, partial, 'cpu', preset=preset, stop_after=1) is None
    resumed, _ = fit(*args, partial, 'cpu', preset=preset)
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)


def test_dense_eval_resume_and_table_missing_cells(tmp_path):
    torch.set_num_threads(1)
    cfg = Config()
    hub = Hub(tmp_path)
    manifest, x, y, selected = fixture(hub, cfg)
    args = (cfg, 'dinov2', 'voc', 'dense', None, None, x, y, selected, manifest, hub, 'cpu')
    expected = evaluate(*args)
    assert expected['query_images'] == 2
    assert expected['miou'] == 1.0
    folder = next((tmp_path/'evaluations').iterdir())
    (folder/'result.json').unlink()
    # Simulates a completed progress snapshot restored before final report creation.
    seal(folder)
    actual = evaluate(*args)
    assert actual['confusion'] == expected['confusion']
    assert actual['mean_nnz'] == expected['mean_nnz']
    write_tables([actual], tmp_path/'table')
    table = (tmp_path/'table/results.tex').read_text()
    assert '100.00' in table and '--' in table
    assert not json.loads((tmp_path/'table/results.json').read_text())['complete']


def test_local_export_updates_zip_without_hub_calls(monkeypatch, tmp_path):
    from zipfile import ZipFile
    import huggingface_hub
    from mpsae.io import atomic_json
    from mpsae.patch_scene.report import collect_tables
    def no_network(*args, **kwargs):
        raise AssertionError('local export must not contact Hugging Face')
    monkeypatch.setattr(huggingface_hub, 'HfApi', no_network)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', no_network)
    hub = Hub(tmp_path)
    campaign = 'local-test'
    root = tmp_path/'campaigns'/campaign
    atomic_json(root/'settings/config.json', Config().plan())
    row = {'dataset': 'voc', 'encoder': 'dinov2', 'method': 'dense', 'k': None,
           'seed': 1227, 'miou': .5, 'pixel_accuracy': .75, 'complete': True}
    atomic_json(root/'cells/voc-dinov2/rows.json', {'results': [row]})
    # Unrelated large/private artifacts must not be swept into the download.
    atomic_json(tmp_path/'fits/checkpoint.pt', {'not': 'a real checkpoint'})
    atomic_json(tmp_path/'features/cache.npz', {'not': 'a real cache'})
    collect_tables(hub, campaign)
    archive = tmp_path/'patch-scene-results.zip'
    with ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {'results.csv', 'results.tex', 'results.json', 'config.json', 'README.txt'}
        result = json.loads(zipped.read('results.json'))
        assert result['completed'] == 1 and not result['complete']
        assert '50.00' in zipped.read('results.tex').decode()
    atomic_json(root/'cells/voc-dinov2/rows.json', {'results': [row, {**row, 'method': 'csr', 'k': 12}]})
    # Replace a previously sealed local cell, as the real runner does.
    seal(root/'cells/voc-dinov2')
    collect_tables(hub, campaign)
    with ZipFile(archive) as zipped:
        assert json.loads(zipped.read('results.json'))['completed'] == 2


def test_empty_local_export_is_explicit(tmp_path):
    from mpsae.patch_scene.report import collect_tables
    with pytest.raises(FileNotFoundError, match='No completed evaluations'):
        collect_tables(Hub(tmp_path), 'empty')
    assert not (tmp_path/'patch-scene-results.zip').exists()


def test_kaggle_notebook_has_no_token_or_publish_requirement():
    import ast
    notebook = Path(__file__).resolve().parents[1]/'notebooks/patch_scene_kaggle.ipynb'
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    ast.parse(code)
    assert 'UserSecretsClient' not in code and 'kaggle_secrets' not in code
    assert 'HF_TOKEN' not in code and '--publish' not in code and '--repo' not in code
    assert 'patch-scene-results.zip' in code and 'finally:' in code


def test_hub_restores_hashed_bundle(monkeypatch, tmp_path):
    import huggingface_hub
    source = tmp_path/'remote'
    source.mkdir()
    (source/'data.txt').write_text('checkpoint')
    seal(source)
    hub = Hub(tmp_path/'local')
    hub.repo, hub.revision = 'fake/repo', 'fixed-commit'
    hub.files = {'scene-patch-v2/fit/bundle.json'}
    calls = []
    def download(repo, filename, **kw):
        calls.append(kw)
        return str(source/Path(filename).name)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    restored = hub.pull('fit')
    assert (restored/'data.txt').read_text() == 'checkpoint'
    assert verify(restored)
    assert all(call['revision'] == 'fixed-commit' for call in calls)


def test_feature_shards_resume_without_reextracting_completed_images(monkeypatch, tmp_path):
    from PIL import Image
    from mpsae.patch_scene import features as module
    from mpsae.scene.datasets import VOC
    cfg = Config(image_size=112, train_patches=16, support_patches=8, neighbors=2,
                 extraction_batch=2, shard_images=2, workers=0)
    spec = replace(VOC, train_rows=5, validation_rows=3)
    data = {split: [{'id': f'{split}-{i}',
                    'image': Image.fromarray(np.full((17, 23, 3), 40+i*10, dtype='uint8')),
                    'mask': Image.fromarray(np.full((17, 23), i%3, dtype='uint8'))}
                   for i in range(count)] for split, count in [('train', 5), ('validation', 3)]}
    monkeypatch.setattr(module, 'feature_identity', lambda *a: {'test': True})
    monkeypatch.setattr(module, 'dataset_spec', lambda *a: spec)
    monkeypatch.setattr(module, 'load_splits', lambda *a: data)
    monkeypatch.setattr(module, 'row_ids', lambda rows: [r['id'] for r in rows])
    monkeypatch.setattr(module, 'load_encoder', lambda *a: (torch.nn.Identity(), ([.5]*3, [.5]*3)))
    calls = []
    def tokens(model, images, encoder):
        calls.append(len(images))
        return images.mean((1, 2, 3))[:, None, None].expand(len(images), 64, 768).contiguous()
    monkeypatch.setattr(module, 'patch_tokens', tokens)
    hub = Hub(tmp_path)
    sources = {'models': {'dinov2': 'model-sha'}, 'datasets': {'voc': 'data-sha'}}
    rel, manifest = module.prepare(cfg, 'dinov2', 'voc', sources, hub, 'cpu')
    assert sum(calls) == 8
    x, y, chosen = module.training_bank(manifest, hub, cfg)
    assert x.shape == (16, 768) and len(chosen) == 8
    assert [i for i, _, _ in module.queries(manifest, hub)] == [0, 1, 2]
    # Simulate a session ending before the final shard/index committed.
    last = tmp_path/rel/'validation/000002'
    for name in ('data.npz', 'metadata.json', 'bundle.json'):
        (last/name).unlink()
    for name in ('manifest.json', 'bundle.json'):
        (tmp_path/rel/'index'/name).unlink()
    calls.clear()
    _, resumed = module.prepare(cfg, 'dinov2', 'voc', sources, hub, 'cpu')
    assert sum(calls) == 1
    assert [i for i, _, _ in module.queries(resumed, hub)] == [0, 1, 2]
    np.testing.assert_array_equal(module.training_bank(resumed, hub, cfg)[0], x)


@pytest.mark.parametrize('encoder', ['clip', 'dinov2', 'siglip2'])
def test_real_transformers_spatial_adapter_without_download(encoder):
    transformers = pytest.importorskip('transformers')
    from mpsae.patch_scene.encoders import patch_tokens
    torch.set_num_threads(1)
    if encoder == 'clip':
        cfg = transformers.CLIPVisionConfig(hidden_size=768, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=12, image_size=16, patch_size=16)
        model = transformers.CLIPVisionModel(cfg)
        size, tokens = 32, 4
    elif encoder == 'dinov2':
        cfg = transformers.Dinov2Config(hidden_size=768, num_hidden_layers=1,
                    num_attention_heads=12, image_size=14, patch_size=14, mlp_ratio=1)
        model = transformers.Dinov2Model(cfg)
        size, tokens = 28, 4
    else:
        cfg = transformers.SiglipVisionConfig(hidden_size=768, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=12, image_size=16, patch_size=16)
        model = transformers.SiglipVisionModel(cfg)
        size, tokens = 32, 4
    model.eval()
    actual = patch_tokens(model, torch.rand(1, 3, size, size), encoder)
    assert actual.shape == (1, tokens, 768)
    assert torch.isfinite(actual).all()
