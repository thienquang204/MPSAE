from pathlib import Path
from dataclasses import replace
import json
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
import torch
from scipy.io import savemat

from mpsae.patch_scene.nyuv2 import load_local, NYUSpec
from mpsae.patch_scene.config import Config, load
from mpsae.patch_scene.artifacts import Hub
from mpsae.patch_scene.report import expected_keys, write_tables, export_results


def tiny_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    images = np.arange(4*3*32*24, dtype=np.uint8).reshape(4,3,32,24)
    labels = np.ones((4,32,24), dtype=np.uint16)
    labels[0,0,0], labels[0,1,0] = 0,40
    with h5py.File(root/'nyu_depth_v2_labeled.mat', 'w') as file:
        file['images'] = images
        file['labels'] = labels
        # Intentionally no depth arrays: RGB segmentation must not require them.
    savemat(root/'classMapping40.mat', {'mapClass': np.arange(1,41)[None]})
    savemat(root/'splits.mat', {'trainNdxs': np.array([[3],[1],[4]]), 'testNdxs': np.array([[2]])})
    return images,labels


def test_mat_orientation_split_labels_and_worker_serialization(tmp_path):
    import pickle
    from mpsae.scene.data import ImageRows
    images,labels = tiny_dataset(tmp_path)
    data,revision = load_local(tmp_path, expected=(3,1), fingerprint=True)
    assert revision and data['train']['id'] == ['nyuv2-0003','nyuv2-0001','nyuv2-0004']
    assert data['validation']['id'] == ['nyuv2-0002']
    row = data['train'][1]
    np.testing.assert_array_equal(np.asarray(row['image']), images[0].transpose(2,1,0))
    np.testing.assert_array_equal(np.asarray(row['mask']), labels[0].T)
    rows = pickle.loads(pickle.dumps(data['train']))
    assert rows[1]['id'] == 'nyuv2-0001'
    _,image,truth,small = ImageRows(rows,[1],(336,448),([.5]*3,[.5]*3),NYUSpec())[0]
    assert image.shape == (3,336,448) and truth.shape == (24,32)
    assert truth[0,0] == 255 and truth[0,1] == 39 and small.shape == (336,448)


@pytest.mark.parametrize('failure', ['overlap','duplicate','out_of_range','fractional','mapping13'])
def test_reject_bad_splits_and_mapping(tmp_path, failure):
    tiny_dataset(tmp_path)
    train = np.array([3,1,4],dtype=float)
    test = np.array([2])
    if failure == 'overlap': test[0] = 3
    if failure == 'duplicate': train[2] = 1
    if failure == 'out_of_range': train[2] = 5
    if failure == 'fractional': train[2] = 3.5
    if failure == 'mapping13':
        savemat(tmp_path/'classMapping40.mat', {'mapClass': np.arange(1,14)[None]})
    savemat(tmp_path/'splits.mat', {'trainNdxs':train[:,None], 'testNdxs':test[:,None]})
    with pytest.raises(ValueError):
        load_local(tmp_path, expected=(3,1))


def test_matched_grid_and_paper_export(tmp_path):
    cfg = load(Path(__file__).resolve().parents[1]/'configs/patch_scene_nyuv2.yaml')
    plan = cfg.plan()
    assert (plan['sae_fits'],plan['total_evaluations'],plan['cross_k_evaluations']) == (18,20,0)
    assert plan['evaluation_pairs'] == [[3,3],[12,12],[48,48]]
    assert cfg.all_train_patches and cfg.image_hw == (336,448) and cfg.epochs == 20
    assert cfg.early_stopping and cfg.early_min_epochs == 15
    assert cfg.early_patience == 3 and cfg.early_min_delta == .001
    for k in cfg.training_ks:
        assert cfg.for_training(k).inference_ks == (k,)
    with pytest.raises(ValueError):
        replace(cfg,inference_k=(3,12))
    sweep = replace(cfg,evaluation_mode='sweep')
    assert cfg.source_identity() == sweep.source_identity() and cfg.preset() == sweep.preset()
    assert cfg.identity() != sweep.identity() and sweep.plan()['total_evaluations'] == 56
    rows = [dict(dataset=d,encoder=e,method=m,train_k=t,inference_k=k,k=k,
                 seed=cfg.seed,complete=True,miou=.4,pixel_accuracy=.8)
            for d,e,m,t,k in expected_keys(cfg)]
    table = tmp_path/'campaigns/example/table'
    write_tables(rows,table,cfg)
    receipt = json.loads((table/'results.json').read_text())
    assert receipt['complete'] and receipt['completed'] == 20
    assert receipt['evaluation_splits'] == {'nyuv2':'test'}
    assert not list(table.glob('*sweep*'))
    assert 'RGB only' in (table/'results.tex').read_text()
    with ZipFile(export_results(tmp_path,'example')) as archive:
        assert 'table_nyuv2.tex' in archive.namelist()
        assert not any('sweep' in name or 'cityscapes' in name for name in archive.namelist())


def test_nyu_source_resolution_is_local(monkeypatch,tmp_path):
    import huggingface_hub
    from mpsae.patch_scene import nyuv2
    from mpsae.patch_scene.features import resolve_sources
    monkeypatch.setattr(nyuv2,'load_local',lambda **kw: ({},'nyu-sha'))
    class API:
        def model_info(self,repo): return type('Info',(),{'sha':'model-sha'})()
        def dataset_info(self,*args,**kw): raise AssertionError('No HF dataset should be requested')
    monkeypatch.setattr(huggingface_hub,'HfApi',API)
    cfg = Config(datasets=['nyuv2'],encoders=['siglip2','dinov2'])
    hub=Hub(tmp_path)
    assert resolve_sources(cfg,hub)['datasets'] == {'nyuv2':'nyu-sha'}
    monkeypatch.setattr(nyuv2,'load_local',lambda **kw: ({},'changed-sha'))
    with pytest.raises(RuntimeError,match='source changed'):
        resolve_sources(cfg,hub)


def test_full_nyu_extraction_fit_and_test_evaluation(monkeypatch,tmp_path):
    from mpsae.patch_scene import nyuv2, features as module
    from mpsae.patch_scene.stream import ShardTraining
    from mpsae.patch_scene.train import fit
    from mpsae.patch_scene.evaluate import evaluate
    from mpsae.config import TrainPreset
    torch.set_num_threads(1)
    tiny_dataset(tmp_path/'input')
    data,_=load_local(tmp_path/'input',expected=(3,1))
    spec=NYUSpec(); spec.train_rows,spec.validation_rows=3,1
    monkeypatch.setattr(nyuv2,'NYUSpec',lambda:spec)
    monkeypatch.setattr(nyuv2,'load_local',lambda **kw:(data,None))
    monkeypatch.setattr(module,'feature_identity',lambda *args:{'test':'nyu-full'})
    monkeypatch.setattr(module,'load_encoder',lambda *args:(torch.nn.Identity(),([.5]*3,[.5]*3)))
    monkeypatch.setattr(module,'patch_tokens',lambda model,x,encoder:
        (1+x.mean((1,2,3)))[:,None,None].expand(len(x),192,768).contiguous())
    cfg=Config(k=3,datasets=['nyuv2'],encoders=['dinov2'],evaluation_mode='matched',
        image_size=112,image_width=336,all_train_patches=True,train_patches=0,
        support_patches=8,neighbors=2,extraction_batch=2,shard_images=2,workers=0,epochs=1)
    hub=Hub(tmp_path/'work')
    _,manifest=module.prepare(cfg,'dinov2','nyuv2',{'datasets':{'nyuv2':'x'},'models':{'dinov2':'x'}},hub,'cpu')
    stream=ShardTraining(manifest,hub)
    assert stream.rows <= 3*192 and stream.rows > 2*192
    assert manifest['counts'] == {'train':3,'validation':1}
    x,y,selected=module.training_bank(manifest,hub,cfg)
    assert len(x) == 8 and stream.rows > len(x)
    preset=TrainPreset(name='nyu-test',hidden_dim=64,batch_size=32,epochs=1,k_aux=32)
    model,record=fit(cfg,'csr',x,y,manifest['identity'],hub,'cpu',preset=preset,training_source=stream)
    row=evaluate(cfg,'dinov2','nyuv2','csr',model,record,x,y,selected,manifest,hub,'cpu',inference_k=3)
    assert row['valid_pixels'] == 24*32 and row['query_images'] == 1
    assert row['mean_nnz'] <= 3 and row['gallery_mean_nnz'] <= 3
    assert row['classes_scored'] == 40 and len(row['class_names']) == 40
    assert row['input_modalities'] == ['RGB']
    assert record['training_patches'] == stream.rows


def test_immutable_cache_avoids_rehash_but_detects_corruption(monkeypatch,tmp_path):
    from mpsae.patch_scene import artifacts
    from mpsae.patch_scene.features import save_npz
    hub=Hub(tmp_path)
    folder=tmp_path/'features/a'
    save_npz(folder/'data.npz',features=np.ones((3,2)))
    hub.push('features/a')
    calls=[]
    original=artifacts.sha256_file
    def counted(p):
        calls.append(str(p)); return original(p)
    monkeypatch.setattr(artifacts,'sha256_file',counted)
    hub.pull('features/a'); first=len(calls)
    hub.pull('features/a')
    assert first > 0 and len(calls) == first
    (folder/'data.npz').write_bytes(b'corruption')
    with pytest.raises(RuntimeError,match='Corrupt artifact'):
        hub.pull('features/a')


def test_metadata_download_hash_and_reuse(monkeypatch,tmp_path):
    from mpsae.patch_scene import prepare_nyuv2 as module
    import hashlib,io
    payload=b'test metadata'
    monkeypatch.setitem(module.METADATA_SHA256,'splits.mat',hashlib.sha256(payload).hexdigest())
    calls=[]
    def fetch(*args,**kwargs):
        calls.append(1); return io.BytesIO(payload)
    monkeypatch.setattr(module.urllib.request,'urlopen',fetch)
    module.fetch_metadata('splits.mat',tmp_path/'splits.mat')
    module.fetch_metadata('splits.mat',tmp_path/'splits.mat')
    assert len(calls)==1
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'bad download'))
    with pytest.raises(ValueError,match='hash'):
        module.fetch_metadata('splits.mat',tmp_path/'other.mat')
    assert not (tmp_path/'other.mat').exists()
