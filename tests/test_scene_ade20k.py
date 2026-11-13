import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from mpsae.scene.config import SceneConfig, Adaptation, load_config
from mpsae.scene.datasets import ADE20K, VOC, load_splits, row_ids
from mpsae.scene.core import label_distributions, confusion, metrics, predict
from mpsae.scene.data import ImageRows, save_arrays
from mpsae.scene.adapt import patch_targets
from test_scene_adapt import tiny


def ade_config(**kw):
    return SceneConfig(dataset='ade20k', dataset_config='scene_parsing',
        dataset_repo='zhoubolei/scene_parse_150', **kw)


def test_plan_keeps_budget_and_dataset_separation():
    root = Path(__file__).resolve().parents[1]
    ade = load_config(root/'configs/scene_ade20k.yaml')
    voc = load_config(root/'configs/scene_voc.yaml')
    assert ade.plan()['training_runs'] == 12 and ade.plan()['evaluations'] == 160
    for a, v in zip(ade.groups, voc.groups):
        p = a.plan()
        assert (p['support_images'], p['query_images']) == (20210, 2000)
        assert p['patches_per_support_image'] == 3 and p['bank_rows_upper_bound'] == 60630
        assert a.inference_k == v.inference_k and a.adaptation == v.adaptation
        assert p['label_contract']['classes'] == 150
        assert 'dataset' not in v.plan()['config']


def test_mask_mapping_ignores_zero_and_preserves_last_class():
    raw = np.array([[0, 1, 150], [2, 15, 0]], np.uint8)
    mapped = ADE20K.remap(raw)
    np.testing.assert_array_equal(mapped, [[255, 0, 149], [1, 14, 255]])
    np.testing.assert_array_equal(raw, [[0, 1, 150], [2, 15, 0]])
    for value in (np.array([[151]]), np.array([[255]]), np.zeros((2, 2, 3))):
        with pytest.raises(ValueError):
            ADE20K.remap(value)
    np.testing.assert_array_equal(VOC.remap([[0, 20, 255]]), [[0, 20, 255]])


def test_patch_histograms_match_onehot_pooling():
    rng = np.random.default_rng(3)
    mask = rng.integers(0, 150, (32, 48), dtype=np.int64)
    mask[:4] = 255
    actual = label_distributions(mask, (4, 6), classes=150)
    valid = mask != 255
    onehot = torch.nn.functional.one_hot(torch.tensor(np.where(valid, mask, 0)), 150).float()
    onehot *= torch.tensor(valid)[..., None]
    reference = torch.nn.functional.adaptive_avg_pool2d(onehot.permute(2, 0, 1)[None], (4, 6))
    reference = reference[0].permute(1, 2, 0).reshape(-1, 150)
    reference /= reference.sum(1, keepdim=True).clamp_min(1e-12)
    np.testing.assert_allclose(actual, reference.numpy(), atol=1e-7)
    labels = label_distributions(ADE20K.remap([[0, 150], [150, 1]]), (1, 1), classes=150)
    np.testing.assert_allclose(labels[0, [0, 149]], [1/3, 2/3])
    targets, stats = patch_targets(labels, classes=150)
    assert targets.tolist() == [149] and len(stats['class_counts']) == 150
    assert 'background_fraction' not in stats


def test_metrics_match_official_ignore_and_all_class_average():
    truth = ADE20K.remap([[0, 1, 150, 150]])
    pred = np.array([[4, 0, 149, 150]])  # ignored, correct, correct, abstained
    matrix = confusion(pred, truth, classes=150)
    result = metrics(matrix, include_absent=True)
    assert matrix.shape == (150, 151)
    assert result['valid_pixels'] == 3 and result['abstained_pixels'] == 1
    assert result['pixel_accuracy'] == pytest.approx(2/3)
    assert result['class_iou'][0] == 1 and result['class_iou'][149] == .5
    assert result['miou'] == pytest.approx(1.5/150) and result['classes_scored'] == 150


def test_image_loader_uses_semantic_annotation_not_scene_category():
    row = {'image': Image.new('RGB', (2, 2)), 'annotation': Image.fromarray(
        np.array([[0, 1], [150, 2]], np.uint8)), 'scene_category': 999}
    _, image, truth, scaled = ImageRows([row], [0], 4, ((0, 0, 0), (1, 1, 1)), ADE20K)[0]
    assert image.shape == (3, 4, 4)
    assert set(np.unique(truth)) == {0, 1, 149, 255}
    assert set(np.unique(scaled)) == {0, 1, 149, 255}


def test_parquet_loader_pins_revision_and_omits_test(monkeypatch):
    import huggingface_hub
    calls = []
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: calls.append((a,k))))
    files = ['scene_parsing/train/0000.parquet', 'scene_parsing/validation/0000.parquet',
             'scene_parsing/test/0000.parquet', 'instance_segmentation/train/0000.parquet']
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(list_repo_files=lambda *a, **kw: files))
    load_splits(ade_config(), {'revision': 'pinned'}, 'cache')
    args, kw = calls[0]
    assert args == ('parquet',)
    assert set(kw['data_files']) == {'train', 'validation'}
    assert all('@pinned/scene_parsing/' in f for fs in kw['data_files'].values() for f in fs)
    assert 'trust_remote_code' not in kw


def test_sparse_retrieval_handles_150_classes_and_abstention():
    labels = np.zeros((2, 150), np.float32)
    labels[0, 0], labels[1, 149] = 1, 1
    gallery = {'indices': np.array([[0], [1]], np.int32), 'values': np.ones((2, 1), np.float32)}
    query = {'indices': np.array([[0], [1], [0]], np.int32), 'values': np.array([[1], [1], [0]], np.float32)}
    pred, _ = predict(gallery, labels, query, dim=2, grid=(1, 3), output_size=(1, 3),
                      device='cpu', neighbors=1, gallery_chunk=2, query_block=2)
    np.testing.assert_array_equal(pred, [[0, 149, 150]])


def test_end_to_end_ade_report_uses_150_class_metrics(tmp_path, monkeypatch):
    from mpsae.scene import run as runner
    from mpsae.model import SparseAutoencoder
    config = ade_config(methods=('csr',), inference_k=(1,), neighbors=1, workers=0)
    features = np.eye(2, dtype=np.float32)
    labels = np.zeros((2, 150), np.float32)
    labels[0, 0], labels[1, 149] = 1, 1
    cache = tmp_path/'cache'
    save_arrays(cache/'validation/00000.npz', features=features, truth=np.array([[0, 149]], np.uint8))
    manifest = {'identity': {'dataset_name': 'ade20k'}, 'grid': (1, 2), 'support_images': 1, 'query_images': 1}
    record = {'checkpoint_id':'x','signature':'y','cell':config.cells()[0].__dict__,
              'path':str(tmp_path/'checkpoint.pt'),'backbone_weights':{}}
    model = SparseAutoencoder(2, 4)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]))
    monkeypatch.setattr(runner, 'source_records', lambda *a:[record])
    monkeypatch.setattr(runner, 'prepare', lambda *a, **k:(cache,manifest))
    monkeypatch.setattr(runner, 'support_bank', lambda *a:(features,labels))
    monkeypatch.setattr(runner, 'load_fitted', lambda *a, **k:model)
    result = runner.run(config,tmp_path/'scene',tmp_path,device='cpu')
    assert len(result['results']) == 2
    assert all(r['classes_scored'] == 150 and r['dataset'] == 'ade20k' for r in result['results'])
    assert all(r['miou'] == pytest.approx(2/150) for r in result['results'])
    assert 'dataset' in (Path(result['report'])/'results.csv').read_text().splitlines()[0]
    image_score = json.loads((Path(result['report'])/'per_image/csr_k1/00000.json').read_text())
    assert 'confusion' not in image_score and len(image_score['confusion_entries']) == 2
    (Path(result['report'])/'csr_k1.json').unlink()
    monkeypatch.setattr(runner, 'predict', lambda *a, **k: pytest.fail('reuse per-image counts'))
    assert runner.run(config,tmp_path/'scene',tmp_path,device='cpu')['results'][-1]['miou'] == pytest.approx(2/150)


@pytest.mark.parametrize('position', [0, 1, 2])
def test_real_ade_adaptation_all_methods(tiny, tmp_path, position):
    from mpsae.scene.adapt import fit, load_adapted
    config, features, old_labels, records = tiny
    config = replace(config, dataset='ade20k', dataset_config='scene_parsing',
                     dataset_repo='zhoubolei/scene_parse_150')
    labels = np.zeros((len(features), 150), np.float32)
    labels[:, :2] = old_labels[:, :2]
    labels[:, 149] = old_labels[:, 2]
    result = fit(config, config.cells()[position], records[position], features, labels,
                 {'dataset': 'ade20k'}, tmp_path/'ade', device='cpu')
    assert len(result['patch_labels']['class_counts']) == 150
    assert result['patch_labels']['class_counts'][149] > 0
    assert 'background_fraction' not in result['patch_labels']
    model = load_adapted(result, device='cpu')
    assert (model.encode(torch.from_numpy(features), 2) != 0).sum(1).max() <= 2


def test_ids_do_not_hide_cross_split_duplicates(monkeypatch):
    class Rows(list):
        column_names = ['image']
        def select_columns(self, columns):
            assert columns == ['image']
            return self
        def cast_column(self, name, feature):
            assert name == 'image'
            return self
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(Image=lambda **kw:kw))
    rows = Rows([{'image': {'path': '/train/ADE_train_000001.jpg', 'bytes': b'x'}},
                 {'image': {'path': None, 'bytes': b'content'}}])
    ids = row_ids(rows)
    assert ids[0] == 'ADE_train_000001.jpg' and len(ids[1]) == 64
    assert ids == row_ids(rows)
