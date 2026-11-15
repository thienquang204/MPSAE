from dataclasses import replace
from pathlib import Path
import pytest
from mpsae.patch_scene.config import load
from mpsae.io import canonical_hash

ROOT=Path(__file__).resolve().parents[1]

def test_nyu_override_scope_and_cache_identity():
    config=load(ROOT/'configs/patch_scene_nyuv2.yaml')
    old=replace(config,siglip2_mpsae_ot_weight=None)
    for k in config.training_ks:
        c=config.for_training(k)
        assert c.preset('mpsae','siglip2').ot_weight==.5
        assert c.preset('mpsae','dinov2').ot_weight==2.
        assert c.preset('csr','siglip2')==old.for_training(k).preset()
        assert c.preset('csrv2','siglip2')==old.for_training(k).preset()
    assert config.identity()!=old.identity()
    assert config.source_identity()==old.source_identity()
    for dataset in ('voc','cityscapes'):
        c=load(ROOT/f'configs/patch_scene_{dataset}.yaml')
        assert c.preset('mpsae','siglip2').ot_weight==2.
    selected=replace(config,encoders=('siglip2',))
    assert selected.plan()['total_evaluations']==10
    assert selected.plan()['sae_fits']==9
    assert selected.plan()['mpsae_ot_weights']=={'siglip2':.5}

@pytest.mark.parametrize('weight',[0,-1,float('nan'),float('inf'),True])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        replace(load(ROOT/'configs/patch_scene_nyuv2.yaml'),siglip2_mpsae_ot_weight=weight)

def test_real_fit_records_override_and_separates_old_checkpoint(tmp_path):
    import numpy as np
    from mpsae.patch_scene.train import fit
    from mpsae.patch_scene.artifacts import Hub
    config=replace(load(ROOT/'configs/patch_scene_nyuv2.yaml'),epochs=1,
        early_stopping=False,all_train_patches=False,train_patches=32,
        support_patches=32,batch_size=4,k=3,train_k=None,encoders=('siglip2',))
    x=np.random.default_rng(9).normal(size=(4,6)).astype('float32')
    y=np.eye(2,dtype='float32')[[0,1,0,1]]
    hub=Hub(tmp_path)
    # Actual training path must derive the encoder-specific preset from feature identity.
    _,record=fit(config,'mpsae',x,y,{'encoder':'siglip2'},hub,'cpu')
    import json
    import torch
    state=torch.load(hub.root/record['path']/'checkpoint.pt',weights_only=False)
    assert state['identity']['preset']['ot_weight']==.5
    old_identity={**state['identity'],'preset':{**state['identity']['preset'],'ot_weight':2.}}
    assert canonical_hash(old_identity)!=canonical_hash(state['identity'])
