import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location('server_launcher',ROOT/'scripts/run_segmentation_server.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def test_plan_preserves_protocol_and_has_no_side_effects(tmp_path,capsys,monkeypatch):
    launcher=module()
    monkeypatch.setattr(launcher,'run',lambda *a,**kw:pytest.fail('plan launched subprocess'))
    launcher.main(['--plan','--workspace',str(tmp_path/'unused')])
    plans=json.loads(capsys.readouterr().out)
    assert set(plans)=={'nyuv2','cityscapes'}
    assert sum(p['total_evaluations'] for p in plans.values())==64
    assert plans['nyuv2']['mpsae_ot_weights']=={'siglip2':[.5,1.,2.],'dinov2':[.5,1.,2.]}
    assert plans['cityscapes']['mpsae_ot_weights']=={'siglip2':[.5,1.,2.],'dinov2':[.5,1.,2.]}
    assert all(p['settings']['support_patches']==65536 for p in plans.values())
    assert not (tmp_path/'unused').exists()

def test_encoder_subset_is_complete_campaign(capsys):
    module().main(['--plan','--datasets','nyuv2','--encoders','siglip2'])
    plan=json.loads(capsys.readouterr().out)['nyuv2']
    assert plan['total_evaluations']==16 and plan['sae_fits']==15


def test_resolved_config_roundtrip_keeps_float_learning_rate(tmp_path):
    from mpsae.patch_scene.config import load
    config=load(ROOT/'configs/patch_scene_nyuv2.yaml')
    path=tmp_path/'resolved.yaml'
    module().write_config(config,path)
    assert load(path)==config
    assert load(path).plan()==config.plan()

def test_export_never_selects_stale_campaign(tmp_path):
    launcher=module()
    (tmp_path/'patch-scene-results.zip').write_bytes(b'old unrelated result')
    assert launcher.export_local(tmp_path,tmp_path/'out','new-campaign') is None
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('args',[['--cpu-threads','0'],['--datasets','nyuv2','nyuv2'],['--encoders','siglip2','siglip2']])
def test_invalid_settings(args):
    with pytest.raises(SystemExit):module().main(args+['--plan'])
