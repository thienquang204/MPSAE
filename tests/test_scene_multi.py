from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mpsae.scene.config import SceneConfig, SceneSuite, load_config
from mpsae.scene.spatial import spatial_features, normalization, STRIDES
from mpsae.config import BACKBONES


def test_expanded_plan():
    config=load_config(Path(__file__).resolve().parents[1]/'configs/scene_voc.yaml')
    assert isinstance(config, SceneSuite)
    assert config.plan()['evaluations']==160
    assert config.plan()['training_runs']==12
    assert len(config.cells())==12
    assert [g.train_k for g in config.groups]==[8,12,12,12]


def test_resolution_override_only_for_transformers():
    from mpsae.scene.spatial import loading_spec
    for name in STRIDES:
        cfg=SceneConfig(backbone=name,image_size=448)
        expected=448 if name in ('vit_b16','dinov2_vits14') else BACKBONES[name].input_size
        assert loading_spec(cfg).input_size==expected


@pytest.mark.parametrize('name,side', [('vit_b16',32),('dinov2_vits14',28),('convnext_tiny',64)])
def test_real_timm_local_prelogits(name,side):
    timm=pytest.importorskip('timm')
    spec=BACKBONES[name]
    opts={} if name=='convnext_tiny' else {'img_size':side}
    old=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        network=timm.create_model(spec.source.removeprefix('timm/'),pretrained=False,**opts).eval()
        images=torch.randn(1,3,side,side)
        with torch.inference_mode():
            local=spatial_features(network,images,name)
            assert local.shape==(1,side//STRIDES[name],side//STRIDES[name],spec.dim)
            assert torch.isfinite(local).all()
            raw=network.forward_features(images)
            if name=='convnext_tiny':
                # Normalization/MLP must match the actual head on an individual cell.
                expected=network.forward_head(raw[:,:,:1,:1],pre_logits=True)
                torch.testing.assert_close(local[:,0,0],expected)
            else:
                expected=network.head_drop(network.fc_norm(raw[:,network.num_prefix_tokens:]))
                torch.testing.assert_close(local.flatten(1,2),expected)
            # No classifier dimension or CLS/register tokens leak into patch grids.
            assert local.shape[-1]==spec.dim
        mean,std=normalization(spec)
        assert len(mean)==len(std)==3 and min(std)>0
    finally:
        torch.set_num_threads(old)


def test_suite_sequential_and_records_all_reports(tmp_path,monkeypatch):
    import mpsae.scene.server as server
    from mpsae.scene.suite import run_suite
    cfg=SceneSuite((SceneConfig(),SceneConfig(backbone='vit_b16',image_size=448,train_k=12)))
    called=[]
    def fake(group,*a,**kw):
        called.append(group.backbone)
        return {'complete':True,'report':str(tmp_path/group.backbone)}
    monkeypatch.setattr(server,'bootstrap',fake)
    args=SimpleNamespace(workspace=str(tmp_path),command='bootstrap',repo='x/y',revision='main',
                         prefix='direct-v1',device='cpu',refresh_source=False,publish=True)
    result=run_suite(cfg,args)
    assert called==['ff2048','vit_b16']
    assert result['complete'] and len(result['reports'])==2
    assert (tmp_path/'latest_suite.json').exists()


def test_multi_restore_requests_all_backbone_sidecars(tmp_path,monkeypatch):
    import huggingface_hub
    import mpsae.scene.run as runner
    from mpsae.scene.__main__ import restore
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/scene_voc.yaml')
    names=[f'direct-v1/runs/{c.slug}/{f}' for c in cfg.cells()
           for f in ('checkpoint.pt','summary.json','config.json')]
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(dataset_info=lambda *a,**k:
        SimpleNamespace(sha='sha',siblings=[SimpleNamespace(rfilename=n) for n in names])))
    calls=[]
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:calls.append(k) or str(tmp_path))
    monkeypatch.setattr(runner,'source_records',lambda *a:[])
    restore(cfg,repo='x/y',revision='main',prefix='direct-v1',cache_dir=str(tmp_path))
    assert sum(p.endswith('checkpoint.pt') for p in calls[0]['allow_patterns'])==12
    for g in cfg.groups:
        assert f'direct-v1/embeddings/{g.backbone}/*.json' in calls[0]['allow_patterns']
