from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from mpsae.io import atomic_json
from mpsae.patch_scene.config import Config, load
from mpsae.patch_scene.artifacts import Hub
from mpsae.patch_scene.features import save_npz, training_bank
from mpsae.patch_scene.stream import ShardTraining
from mpsae.patch_scene.train import fit
from mpsae.config import TrainPreset


def source(tmp_path):
    hub = Hub(tmp_path)
    entries, xs, ys, offset = [], [], [], 0
    for i, size in enumerate([0, 1, 7, 9]):
        name = f'shard{i}'
        x = np.zeros((size, 8), dtype='float16')
        x[:, 0] = np.arange(offset, offset+size)
        y = np.eye(3, dtype='float32')[np.arange(offset, offset+size)%3]
        save_npz(hub.root/name/'data.npz', features=x, labels=y)
        atomic_json(hub.root/name/'metadata.json', {'training_rows': size})
        hub.push(name)
        entries.append({'path': name, 'split': 'train'})
        xs.append(x); ys.append(y); offset += size
    manifest = {'shards': entries, 'classes': 3, 'identity': {'test': 'full'}}
    return ShardTraining(manifest, hub), hub, manifest, np.concatenate(xs), np.concatenate(ys)


def test_stream_coverage_and_support_sampling(tmp_path):
    from mpsae.scene.core import sample_patches
    bank, hub, manifest, x, y = source(tmp_path)
    for epoch in [0, 1]:
        full = list(bank.batches(7, epoch, 8, 0, 'cpu'))
        assert [len(a) for _, a, _ in full] == [8, 9]
        assert sorted(torch.cat([a[:, 0] for _, a, _ in full]).tolist()) == list(range(17))
        suffix = list(bank.batches(7, epoch, 8, 1, 'cpu'))
        assert torch.equal(full[1][1], suffix[0][1])
    cfg = Config(all_train_patches=True, support_patches=8, neighbors=2)
    a, b, chosen = training_bank(manifest, hub, cfg)
    expected = sample_patches(y, 8, np.random.default_rng(cfg.seed))
    np.testing.assert_array_equal(a, x[expected])
    np.testing.assert_array_equal(b, y[expected])
    np.testing.assert_array_equal(training_bank(manifest, hub, cfg)[0], a)
    assert len(chosen) == 8 and bank.rows == 17


@pytest.mark.parametrize('method', ['csr', 'csrv2', 'mpsae'])
def test_stream_training_resume(tmp_path, method):
    torch.set_num_threads(1)
    bank, hub, manifest, x, y = source(tmp_path/'data')
    cfg = Config(k=3, all_train_patches=True)
    preset = TrainPreset(name='stream-test', hidden_dim=64, batch_size=8, epochs=2,
                         k_aux=32, ot_microbatch=4)
    args = (cfg, method, x[:8].astype('float32'), y[:8], manifest['identity'])
    a, record = fit(*args, hub, 'cpu', preset=preset, training_source=bank)
    other = Hub(tmp_path/'resume')
    assert fit(*args, other, 'cpu', preset=preset, training_source=bank, stop_after=1) is None
    b, resumed = fit(*args, other, 'cpu', preset=preset, training_source=bank)
    assert record['training_patches'] == 17 and record['steps'] == 4
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format='PNG')
    return buf.getvalue()


def test_city_zip_split_mapping_and_rectangle(tmp_path):
    from mpsae.patch_scene.cityscapes import load_local, CitySpec, LABEL_IDS
    from mpsae.scene.data import ImageRows
    image_path = tmp_path/'leftImg8bit_trainvaltest.zip'
    mask_path = tmp_path/'gtFine_trainvaltest.zip'
    with ZipFile(image_path, 'w') as images, ZipFile(mask_path, 'w') as masks:
        for split, city, count in [('train', 'aachen', 2), ('val', 'frankfurt', 1), ('test', 'ignored', 1)]:
            for i in range(count):
                name = f'{city}_000000_{i:06d}'
                images.writestr(f'leftImg8bit/{split}/{city}/{name}_leftImg8bit.png', png(np.zeros((16,32,3), dtype='uint8')))
                masks.writestr(f'gtFine/{split}/{city}/{name}_gtFine_labelIds.png', png(np.full((16,32), 7, dtype='uint8')))
    data, revision = load_local(tmp_path, expected=(2,1), fingerprint=True)
    assert len(data['train']) == 2 and len(data['validation']) == 1 and revision
    _, x, truth, small = ImageRows(data['train'], [0], (112,224), ([.5]*3,[.5]*3), CitySpec())[0]
    assert x.shape == (3,112,224) and truth.shape == (16,32) and small.shape == (112,224)
    assert not truth.any()
    np.testing.assert_array_equal(CitySpec().remap(np.array([LABEL_IDS])), np.arange(19)[None])
    assert CitySpec().remap(np.array([[0,1,9,255]])).tolist() == [[255]*4]


def test_city_config_and_tables(tmp_path):
    from mpsae.patch_scene.report import write_tables, expected_keys, export_results
    cfg = load(Path(__file__).resolve().parents[1]/'configs/patch_scene_cityscapes.yaml')
    assert cfg.all_train_patches and cfg.image_hw == (448,896)
    assert cfg.dataset_names == ('cityscapes',) and cfg.epochs == 10
    assert cfg.plan()['sae_fits'] == 18 and len(expected_keys(cfg)) == 56
    rows = [dict(dataset=d, encoder=e, method=m, train_k=t, inference_k=k, k=k,
                 seed=cfg.seed, complete=True, miou=.4, pixel_accuracy=.8)
            for d,e,m,t,k in expected_keys(cfg)]
    table = tmp_path/'campaigns/test/table'
    write_tables(rows, table, cfg)
    assert not (table/'table_ade20k.tex').exists()
    assert not (table/'table_voc.tex').exists()
    assert 'Cityscapes' in (table/'results.tex').read_text()
    assert json.loads((table/'results.json').read_text())['complete']
    with ZipFile(export_results(tmp_path,'test')) as z:
        assert 'table_cityscapes.tex' in z.namelist()
        assert not any('ade20k' in n for n in z.namelist())


def test_downloader_restarts_when_range_is_ignored(tmp_path):
    from mpsae.patch_scene.download_cityscapes import fetch
    import hashlib
    target = tmp_path/'test.zip'
    target.with_suffix('.zip.partial').write_bytes(b'old partial')
    payload = b'new complete archive'
    calls = []
    class Response:
        status_code = 200
        headers = {'Content-Length': str(len(payload))}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, **kwargs): yield payload
    class Session:
        def get(self, url, **kwargs):
            calls.append(kwargs)
            return Response()
    fetch(Session(), 3, target, hashlib.md5(payload).hexdigest())
    assert target.read_bytes() == payload
    assert calls[0]['headers']['Range'].startswith('bytes=')
    fetch(Session(), 3, target, hashlib.md5(payload).hexdigest())
    assert len(calls) == 1  # completed valid file is never downloaded twice


def test_city_source_is_local_and_no_other_dataset_is_requested(monkeypatch, tmp_path):
    import huggingface_hub
    from mpsae.patch_scene import cityscapes
    from mpsae.patch_scene.features import resolve_sources
    monkeypatch.setattr(cityscapes, 'load_local', lambda **kwargs: ({}, 'local-sha'))
    class API:
        def model_info(self, repo):
            return type('Info', (), {'sha': 'model-sha'})()
        def dataset_info(self, *args, **kwargs):
            raise AssertionError('Cityscapes run must not request any HF dataset')
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    config = Config(datasets=['cityscapes'], encoders=['siglip2', 'dinov2'])
    result = resolve_sources(config, Hub(tmp_path))
    assert result['datasets'] == {'cityscapes': 'local-sha'}


def test_full_city_extraction_training_and_evaluation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mpsae.patch_scene import cityscapes, features as module
    from mpsae.patch_scene.evaluate import evaluate
    torch.set_num_threads(1)
    spec = cityscapes.CitySpec()
    spec.train_rows, spec.validation_rows = 2, 1
    data = {split: [{'id': f'{split}-{i}', 'image': Image.fromarray(np.full((16,32,3), 80+i, dtype='uint8')),
                    'mask': Image.fromarray(np.full((16,32), 7, dtype='uint8'))}
                   for i in range(count)] for split, count in [('train',2), ('validation',1)]}
    monkeypatch.setattr(cityscapes, 'CitySpec', lambda: spec)
    monkeypatch.setattr(cityscapes, 'load_local', lambda **kw: (data, 'local-sha'))
    monkeypatch.setattr(module, 'row_ids', lambda rows: [r['id'] for r in rows])
    monkeypatch.setattr(module, 'feature_identity', lambda *args: {'test': 'city-full'})
    monkeypatch.setattr(module, 'load_encoder', lambda *args: (torch.nn.Identity(), ([.5]*3,[.5]*3)))
    monkeypatch.setattr(module, 'patch_tokens', lambda model, x, encoder:
                        x.mean((1,2,3))[:,None,None].expand(len(x),128,768).contiguous())
    config = Config(k=3, datasets=['cityscapes'], image_size=112, image_width=224,
                    train_patches=0, all_train_patches=True, support_patches=8, neighbors=2,
                    extraction_batch=1, shard_images=1, workers=0, epochs=1)
    hub = Hub(tmp_path)
    _, manifest = module.prepare(config,'dinov2','cityscapes',
                                 {'datasets':{'cityscapes':'local-sha'}, 'models':{'dinov2':'model-sha'}},hub,'cpu')
    bank = ShardTraining(manifest,hub)
    x,y,selected = training_bank(manifest,hub,config)
    assert bank.rows == 256 and x.shape == (8,768) and manifest['grid'] == (8,16)
    preset = TrainPreset(name='full-city-test', hidden_dim=64, batch_size=32, epochs=1,k_aux=32)
    model, record = fit(config,'csr',x,y,manifest['identity'],hub,'cpu',preset=preset,training_source=bank)
    row = evaluate(config,'dinov2','cityscapes','csr',model,record,x,y,selected,manifest,hub,'cpu',inference_k=3)
    assert row['query_images'] == 1 and row['valid_pixels'] == 16*32
    assert row['mean_nnz'] <= 3 and row['gallery_mean_nnz'] <= 3
    assert len(row['class_names']) == 19
    assert record['training_patches'] == 256 and record['steps'] == 8
