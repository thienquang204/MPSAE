from pathlib import Path
import json
import pytest
from mpsae.patch_scene.config import load, require_supported

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_configs_alias_and_supported_scope():
    nyu = load(ROOT/'configs/patch_scene_nyuv2.yaml')
    assert nyu == load(ROOT/'configs/patch_scene_kaggle.yaml')
    voc = load(ROOT/'configs/patch_scene_voc.yaml')
    old = load(ROOT/'configs/archive/patch_scene_kaggle_before_cityscapes.yaml')
    require_supported(nyu); require_supported(voc)
    assert voc.dataset_names == ('voc',) and voc.encoder_names == ('siglip2', 'dinov2')
    assert voc.preset() == old.preset()
    assert voc.source_identity() == old.source_identity()
    assert voc.training_ks == old.training_ks and voc.inference_ks == old.inference_ks
    assert voc.train_patches == old.train_patches and voc.image_hw == old.image_hw
    assert voc.plan()['total_evaluations'] == 56 and nyu.plan()['total_evaluations'] == 20
    with pytest.raises(ValueError, match='Supported segmentation'):
        require_supported(old)
    with pytest.raises(ValueError, match='Supported segmentation'):
        require_supported(load(ROOT/'configs/archive/patch_scene_cityscapes.yaml'))


def test_alias_cycle_and_overrides_fail(tmp_path):
    a,b=tmp_path/'a.yaml',tmp_path/'b.yaml'
    a.write_text('extends: b.yaml\n'); b.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError,match='cycle'): load(a)
    a.write_text('extends: b.yaml\nepochs: 99\n')
    with pytest.raises(ValueError,match='cannot override'): load(a)


def test_old_imports_reference_common_implementations():
    from mpsae.segmentation import core,data,training,datasets
    from mpsae.scene import core as oldcore,data as olddata,adapt as oldtrain,datasets as oldsets
    assert core.label_distributions is oldcore.label_distributions
    assert core.predict is oldcore.predict and core.metrics is oldcore.metrics
    assert data.ImageRows is olddata.ImageRows
    assert training.batch_ranges is oldtrain.batch_ranges
    assert training.patch_targets is oldtrain.patch_targets
    assert datasets.load_splits is oldsets.load_splits


def test_retired_cli_rejects_before_download(monkeypatch):
    import sys
    from mpsae.patch_scene import __main__ as patch
    monkeypatch.setattr(sys,'argv',['patch_scene','run','--config',str(ROOT/'configs/archive/patch_scene_cityscapes.yaml')])
    with pytest.raises(SystemExit) as error: patch.main()
    assert error.value.code == 2
    from mpsae.scene import __main__ as old
    monkeypatch.setattr(sys,'argv',['scene','bootstrap'])
    with pytest.raises(SystemExit) as error: old.main()
    assert error.value.code == 2


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
