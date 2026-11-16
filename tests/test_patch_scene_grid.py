from pathlib import Path
from dataclasses import replace
import json
import pytest
from mpsae.patch_scene.config import load
from mpsae.patch_scene.grid import grid_configs,grid_plan,write_grid_tables
from mpsae.patch_scene.report import expected_keys

ROOT=Path(__file__).resolve().parents[1]

def test_weights_only_change_mpsae_fit_presets():
    for d in ('nyuv2','cityscapes'):
        config=load(ROOT/f'configs/patch_scene_{d}.yaml')
        grid=grid_configs(config,[.5,1.,2.])
        assert len({c.plan()['campaign'] for c in grid})==3
        assert grid_plan(config,grid)['sae_fits']==30
        assert grid_plan(config,grid)['total_evaluations']==32
        assert grid_plan(config,grid)['sparse_evaluations']==30
        assert grid_plan(config,grid)['matched_evaluations']==30
        for c,w in zip(grid,[.5,1.,2.]):
            assert c.source_identity()==config.source_identity()
            for e in config.encoder_names:
                for k in config.training_ks:
                    assert c.for_training(k).preset('mpsae',e).ot_weight==w
                    for m in ('csr','csrv2'):
                        assert c.for_training(k).preset(m,e)==config.for_training(k).preset(m,e)

@pytest.mark.parametrize('weights',[[0],[-1],[.5,.5],[float('nan')],[],[True]])
def test_invalid_grids(weights):
    with pytest.raises(ValueError):grid_configs(load(ROOT/'configs/patch_scene_nyuv2.yaml'),weights)

def test_grid_tables_deduplicate_baselines_and_show_missing(tmp_path):
    configs={d:load(ROOT/f'configs/patch_scene_{d}.yaml') for d in ('nyuv2','cityscapes')}
    receipts=[]
    for d,config in configs.items():
        for c in grid_configs(config,[.5,1.,2.]):
            rows=[dict(dataset=ds,encoder=e,method=m,train_k=t,inference_k=k,
                complete=True,identity={'method':m,'weight':c.mpsae_ot_weight if m=='mpsae' else None},
                miou=.4,pixel_accuracy=.8,mean_nnz=3,gallery_mean_nnz=3)
                for ds,e,m,t,k in expected_keys(c)]
            receipts.append((c,d,{'results':rows}))
    write_grid_tables(receipts[:1],configs,[.5,1.,2.],tmp_path)
    partial=json.loads((tmp_path/'grid_results.json').read_text())
    assert not partial['complete'] and partial['completed']==20 and partial['expected']==64
    assert '--' in (tmp_path/'grid_results.tex').read_text()
    write_grid_tables(receipts,configs,[.5,1.,2.],tmp_path)
    full=json.loads((tmp_path/'grid_results.json').read_text())
    assert full['complete'] and full['completed']==64
    assert sum(row['method']=='dense' for row in full['results'])==4
    assert sum(row['method']=='mpsae' for row in full['results'])==36
    assert 'mpsae_lambda_0.5' in (tmp_path/'grid_comparisons.csv').read_text()
    assert r'\lambda=2' in (tmp_path/'grid_results.tex').read_text()
    # Cached baselines must agree rather than silently replacing mismatched runs.
    next(row for row in receipts[1][2]['results'] if row['method']=='csr')['miou']=.9
    with pytest.raises(ValueError,match='Inconsistent'):
        write_grid_tables(receipts,configs,[.5,1.,2.],tmp_path)
