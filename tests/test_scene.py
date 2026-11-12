import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mpsae.scene.config import SceneConfig, load_config
from mpsae.scene.core import (CLASSES, confusion, encode, label_distributions, metrics,
                              predict, sample_patches)
from mpsae.scene.data import save_arrays, open_mask
from mpsae.model import SparseAutoencoder


def test_plan_no_training_and_expected_memory():
    plan = SceneConfig().plan()
    assert plan['training_runs'] == 0
    assert plan['evaluations'] == 4
    assert plan['bank_rows_upper_bound'] == 64416
    assert plan['patches_per_support_image'] == 44
    assert SceneConfig(inference_k=(2, 8, 16)).plan()['evaluations'] == 10


@pytest.mark.parametrize('kwargs', [dict(backbone='unsupported'), dict(image_size=500),
    dict(inference_k=(0,)), dict(inference_k=(8, 8)), dict(temperature=0),
    dict(memory_patches=12), dict(methods=('mpsaev2',)), dict(workers=-1)])
def test_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        SceneConfig(**kwargs).plan()


def test_soft_labels_and_ignore():
    labels = label_distributions(np.array([[0, 1], [255, 1]]), (1, 1))
    np.testing.assert_allclose(labels[0, :2], [1/3, 2/3])
    assert labels[0, 2:].sum() == 0
    assert label_distributions(np.full((2, 2), 255), (1, 1)).sum() == 0
    with pytest.raises(ValueError):
        label_distributions(np.zeros((2, 2, 3)), (1, 1))


def test_sampling_matches_paper_score_and_avoids_void():
    labels = np.zeros((5, 21))
    labels[:3, 0] = 1
    labels[3, 1] = 1
    expected_scores = np.array([3, 3, 3, 1, 0])*np.random.default_rng(9).random(5)
    expected_scores[-1] = np.inf
    actual = sample_patches(labels, 3, np.random.default_rng(9))
    np.testing.assert_array_equal(actual, np.argsort(expected_scores)[:3])
    assert len(sample_patches(labels, 5, np.random.default_rng(9))) == 4


def test_encoder_keeps_budget_and_does_not_update_model():
    sae = SparseAutoencoder(2, 4).eval()
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    state = {key: value.clone() for key, value in sae.state_dict().items()}
    features = np.array([[2., 1.], [1., 2.], [-3., -1.]], dtype=np.float32)
    code = encode(sae, features, 1, 'cpu', batch=2)
    assert code['values'].shape == (3, 1)
    np.testing.assert_array_equal(code['indices'].ravel(), [0, 1, 2])
    for key, value in sae.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_retrieval_soft_labels_and_sparse_dense_parity():
    gallery = np.eye(2, dtype=np.float32)
    labels = np.zeros((2, 21), dtype=np.float32)
    labels[0, [0, 1]], labels[1, 1] = [.25, .75], 1
    queries = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    prediction, stats = predict(gallery, labels, queries, dim=2, grid=(1, 2),
        output_size=(1, 2), device='cpu', neighbors=1, gallery_chunk=1, query_block=1)
    np.testing.assert_array_equal(prediction, [[1, 1]])
    compact = {'indices': np.array([[0], [1]], dtype=np.int32),
               'values': np.ones((2, 1), dtype=np.float32)}
    other, _ = predict(compact, labels, compact, dim=2, grid=(1, 2),
        output_size=(1, 2), device='cpu', neighbors=1)
    np.testing.assert_array_equal(other, prediction)
    assert stats['empty_query_patches'] == 0


def test_zero_code_abstains_and_counts_as_miss():
    labels = np.zeros((1, 21), dtype=np.float32)
    labels[0, 0] = 1
    prediction, info = predict(np.ones((1, 2), dtype=np.float32), labels,
        np.zeros((1, 2), dtype=np.float32), dim=2, grid=(1, 1), output_size=(2, 2),
        device='cpu', neighbors=1)
    assert np.all(prediction == CLASSES)
    result = metrics(confusion(prediction, np.array([[0, 0], [0, 255]])))
    assert result['miou'] == 0
    assert result['valid_pixels'] == result['abstained_pixels'] == 3
    assert info['empty_query_patches'] == 1


def test_iou_global_confusion_not_mean_image_iou():
    matrix = confusion(np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]))
    result = metrics(matrix)
    assert result['miou'] == pytest.approx(1/3)
    assert result['pixel_accuracy'] == .5
    assert result['classes_scored'] == 2


def test_uint8_mask_confusion_does_not_overflow():
    truth = np.array([[20, 19, 255]], dtype=np.uint8)
    result = metrics(confusion(np.array([[20, 19, 0]], dtype=np.uint8), truth))
    assert result['miou'] == 1
    assert result['valid_pixels'] == 2


def test_mask_palette_preserved(tmp_path):
    from PIL import Image
    path = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(path)
    np.testing.assert_array_equal(np.array(open_mask(path)), [[0, 1, 255]])
    np.testing.assert_array_equal(np.array(open_mask({'bytes': path.read_bytes()})), [[0, 1, 255]])


def test_native_resnet_spatial_mean_is_existing_pooled_feature():
    from torch import nn
    from torchvision.models import resnet50
    from mpsae.embed.mrl_ff2048 import apply_blurpool
    from mpsae.scene.core import resnet_spatial
    network = resnet50(weights=None).eval()
    apply_blurpool(network)
    network.fc = nn.Identity()
    with torch.inference_mode():
        image = torch.randn(1, 3, 64, 64)
        spatial = resnet_spatial(network, image)
        assert spatial.shape == (1, 2, 2, 2048)
        torch.testing.assert_close(spatial.mean((1, 2)), network(image))


def test_source_weights_cannot_be_mixed(monkeypatch):
    import mpsae.scene.run as module
    monkeypatch.setattr(module, 'checkpoint_record',
                        lambda cell, root: {'backbone_weights': cell.method})
    with pytest.raises(ValueError, match='same backbone'):
        module.source_records(SceneConfig(), '.')


def test_restore_downloads_selected_fits_and_sidecars_only(monkeypatch, tmp_path):
    import huggingface_hub
    import mpsae.scene.run as run_module
    from mpsae.scene.__main__ import restore
    from types import SimpleNamespace
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        dataset_info=lambda *a, **kw: SimpleNamespace(sha='pinned-commit', siblings=[
            SimpleNamespace(rfilename=f'direct-v1/runs/{cell.slug}/{name}')
            for cell in SceneConfig().cells() for name in ('checkpoint.pt', 'config.json', 'summary.json')])))
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download',
                        lambda *a, **kw: calls.append(kw) or str(tmp_path))
    monkeypatch.setattr(run_module, 'source_records', lambda *a: [])
    result = restore(SceneConfig(), repo='test/repo', revision='main', prefix='direct-v1', cache_dir='cache')
    patterns = calls[0]['allow_patterns']
    assert len([p for p in patterns if p.endswith('checkpoint.pt')]) == 3
    assert all(p.endswith(('.pt', '.json')) for p in patterns)
    assert result['revision'] == 'pinned-commit'
    with pytest.raises(ValueError):
        restore(SceneConfig(), repo='x/y', revision='main', prefix='../x', cache_dir='cache')


def test_run_end_to_end_cached_and_plot(tmp_path, monkeypatch):
    import mpsae.scene.run as module
    config = SceneConfig(methods=('csr',), inference_k=(1, 2), workers=0)
    directory = tmp_path / 'features'
    raw = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    labels = np.zeros((2, 21), dtype=np.float32)
    labels[0, 0], labels[1, 1] = 1, 1
    save_arrays(directory / 'train' / '00000.npz', features=raw, labels=labels)
    save_arrays(directory / 'validation' / '00000.npz', features=raw, truth=np.array([[0, 1]]))
    manifest = {'identity': {'fake': True}, 'grid': [1, 2], 'support_images': 1, 'query_images': 1}
    monkeypatch.setattr(module, 'prepare', lambda *a, **kw: (directory, manifest))
    record = dict(checkpoint_id='fake', signature='fake', cell=config.cells()[0].__dict__,
                  path=str(tmp_path / 'checkpoint.pt'), backbone_weights={})
    monkeypatch.setattr(module, 'source_records', lambda *a: [record])
    sae = SparseAutoencoder(2, 4).eval()
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    monkeypatch.setattr(module, 'load_fitted', lambda *a, **kw: sae)
    # Tiny fixtures intentionally use one neighbor.
    from dataclasses import replace
    config = replace(config, neighbors=1)
    result = module.run(config, tmp_path, tmp_path, device='cpu')
    assert len(result['results']) == 3
    assert all(r['miou'] == 1 for r in result['results'])
    for name in ('results.json', 'results.csv', 'segmentation.pdf', 'segmentation.png'):
        assert (Path(result['report']) / name).is_file()
    monkeypatch.setattr(module, 'load_fitted', lambda *a, **kw: pytest.fail('should reuse complete results'))
    second = module.run(config, tmp_path, tmp_path, device='cpu')
    assert second['results'] == result['results']
    # Remove aggregate point only: per-image entries must permit scoring resume without retrieval.
    (Path(result['report']) / 'csr_k1.json').unlink()
    monkeypatch.setattr(module, 'load_fitted', lambda *a, **kw: sae)
    monkeypatch.setattr(module, 'predict', lambda *a, **kw: pytest.fail('should reuse image scores'))
    assert module.run(config, tmp_path, tmp_path, device='cpu')['complete']
