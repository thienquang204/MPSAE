import csv
from dataclasses import asdict
from zipfile import ZipFile

import pytest
import torch
import yaml

from mpsae.benchmark.matched_config import MatchedConfig
from mpsae.benchmark.matched_crossk import main, selection, checkpoint
from mpsae.benchmark.matched_study import main as original_main
from mpsae.io import read_json, atomic_json


def test_counts_and_invalid_selections():
    c=MatchedConfig()
    cells,counts=selection(c,[8,32],[2,8,32,128])
    assert len(cells)==24
    assert counts==dict(checkpoints=24,sweep_1nn=96,existing_diagonal=24,
                        new_off_diagonal=72,new_training=0,combined_accuracy_results=156)
    for anchors,budgets in [([3],[8]),([8,8],[2]),([8],[0]),([8],[4096]),([],[8])]:
        with pytest.raises(ValueError):
            selection(c,anchors,budgets)


def test_completed_checkpoint_crossk_never_trains_and_resumes(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    config=MatchedConfig(seeds=(1227,),k_values=(2,8),probe_k=8,batch_sizes=(8,),
                         probe_epochs=1,query_batch=8,gallery_chunk=16,build_entry_cap=128,encode_batch_size=32)
    cfg=tmp_path/'study.yaml';cfg.write_text(yaml.safe_dump(asdict(config)))
    work=tmp_path/'work'
    assert original_main(['--config',str(cfg),'--workspace',str(work),'--smoke','--device','cpu','--cpu-threads','1'])==0
    root=work/'SMOKE_ONLY';source=root/'studies'/config.plan()['study_id']
    originals={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    stamps={p:p.stat().st_mtime_ns for p in (root/'fits').rglob('*.pt')}
    # Saved absolute paths may refer to another mount; only fit IDs should locate weights.
    for p in (source/'cells').glob('*/result.json'):
        r=read_json(p);r['fit_path']='/obsolete/mount';atomic_json(p,r)
    originals={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    def forbidden(*a,**kw):
        raise AssertionError('Inference must not train or fit a probe')
    import mpsae.train
    import mpsae.benchmark.matched_evaluate as ev
    monkeypatch.setattr(mpsae.train,'fit',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(ev,'probe',forbidden)
    args=['--workspace',str(work),'--smoke','--device','cpu','--cpu-threads','1',
          '--train-k','2','8','--inference-k','2','8','16']
    main(args)
    output=next(p for p in (root/'crossk').iterdir() if p.is_dir() and p.name!='cache')
    result=read_json(output/'results.json')
    assert result['complete']==result['expected']==18
    assert result['combined_accuracy_rows']==21
    assert sum(r['reused_diagonal'] for r in result['results'])==6
    assert all(0<=r['query_nnz']<=r['inference_k'] for r in result['results'])
    with (output/'reports/combined_accuracy.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert sum(r['metric']=='linear_probe' for r in rows)==3
    assert len({tuple(r[k] for k in ('method','batch_size','seed','train_k','inference_k','metric')) for r in rows})==21
    assert (output/'reports/top1.pdf').is_file()
    with ZipFile(output/'crossk-results.zip') as z:
        assert 'reports/combined_accuracy.csv' in z.namelist()
        assert not any(n.endswith('.pt') for n in z.namelist())
    monkeypatch.setattr(ev,'retrieval',forbidden)
    main(args)
    main(args+['--report-only'])
    assert originals=={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert stamps=={p:p.stat().st_mtime_ns for p in (root/'fits').rglob('*.pt')}
    assert not list((root/'crossk/cache/transient-codes').glob('*'))
    cell,b=config.cells()[0]
    row,summary,path=checkpoint(root,source,cell,b)
    (path/'checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        checkpoint(root,source,cell,b)


def test_missing_record_cannot_start_training(tmp_path):
    config=MatchedConfig()
    cell,b=config.cells()[0]
    with pytest.raises(FileNotFoundError,match='no training'):
        checkpoint(tmp_path,tmp_path/'source',cell,b)
    assert not list(tmp_path.rglob('checkpoint.pt'))
