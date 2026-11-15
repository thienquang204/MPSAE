from pathlib import Path
import json
import pytest
from mpsae.patch_scene.config import load, require_supported

ROOT = Path(__file__).resolve().parents[1]




def test_alias_cycle_and_overrides_fail(tmp_path):
    a,b=tmp_path/'a.yaml',tmp_path/'b.yaml'
    a.write_text('extends: b.yaml\n'); b.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError,match='cycle'): load(a)
    a.write_text('extends: b.yaml\nepochs: 99\n')
    with pytest.raises(ValueError,match='cannot override'): load(a)






def test_notebook_guide_share_bootstrap():
    nb=json.loads((ROOT/'notebooks/patch_scene_kaggle.ipynb').read_text())
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells) == 1
    assert ''.join(cells[0]['source']) in (ROOT/'docs/patch_scene_kaggle.md').read_text()


def test_hf_mirror_download_pinned_and_verified(monkeypatch,tmp_path):
    from mpsae.patch_scene import prepare_nyuv2 as module
    import huggingface_hub
    calls=[]
    path=tmp_path/'nyu_depth_v2_labeled.mat'; path.write_bytes(b'test')
    def download(**kwargs): calls.append(kwargs); return str(path)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',download)
    monkeypatch.setattr(module,'sha256_file',lambda p:module.LABELED_SHA256)
    assert module.download_labeled(tmp_path) == path
    assert calls[0]['revision'] == module.HF_REVISION and calls[0]['token'] is False
    monkeypatch.setattr(module,'sha256_file',lambda p:'bad')
    with pytest.raises(ValueError,match='checksum mismatch'): module.download_labeled(tmp_path)


def test_only_three_supported_campaigns_and_no_retired_pipeline():
    from dataclasses import replace
    from mpsae.patch_scene.config import SUPPORTED_DATASETS
    assert set(SUPPORTED_DATASETS) == {'voc', 'nyuv2', 'cityscapes'}
    for dataset, count in [('voc',56),('nyuv2',20),('cityscapes',20)]:
        config=load(ROOT/f'configs/patch_scene_{dataset}.yaml')
        require_supported(config)
        assert config.plan()['total_evaluations']==count
    nyu=load(ROOT/'configs/patch_scene_nyuv2.yaml')
    assert nyu==load(ROOT/'configs/patch_scene_kaggle.yaml')
    with pytest.raises(ValueError):
        require_supported(replace(nyu,datasets=('ade20k',)))
    assert not list((ROOT/'mpsae/scene').glob('*.py'))
    assert not list((ROOT/'configs/archive').glob('scene_*.yaml'))
    assert not list((ROOT/'configs/archive').glob('patch_scene_*.yaml'))
    from mpsae.segmentation.doctor import doctor
    assert callable(doctor)
