"""Local-only orchestration; numerical work stays in the existing mpsae engines."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

from common import HERE, REPO, read_json, read_yaml, write_json


def paths_from(path=None):
    base=REPO
    values={}
    if path:
        path=Path(path).resolve()
        values=read_yaml(path)
        base=path.parent
    defaults={'imagenet':REPO/'work/direct-v1','retrieval':REPO/'work/retrieval-studies',
              'segmentation':REPO/'work/patch-scene-paper','output':REPO/'work/cvpr2027',
              'nyuv2_root':None}
    unknown=set(values)-set(defaults)-{'note'}
    if unknown:
        raise ValueError(f'Unknown workspace fields: {unknown}')
    result={k:(base/values.get(k,v)).resolve() if values.get(k,v) is not None else None
            for k,v in defaults.items()}
    if result['imagenet'].name!='direct-v1':
        raise ValueError('imagenet must point to the existing direct-v1 directory, not its parent')
    return result


def check_source_lock(folder=HERE, repo=REPO):
    lock=read_json(folder/'reference/source.lock.json')
    if not lock:
        raise ValueError('Missing scientific source lock')
    changed=[]
    for name,expected in lock['files'].items():
        path=repo/name
        # Git may convert line endings; this lock concerns source text.
        observed=hashlib.sha256(path.read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest() if path.is_file() else None
        if observed!=expected:
            changed.append(name)
    if changed:
        raise ValueError('Scientific implementation differs from the paper recipe snapshot: '+', '.join(changed))


def runtime(device):
    import torch
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).splitlines()
    except (OSError,subprocess.CalledProcessError):
        commit,dirty=None,None
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    return {'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
            'platform':platform.platform(),'git_commit':commit,'git_status':dirty,'torch':str(torch.__version__),
            'cuda':torch.version.cuda,'device':device,
            'gpu':torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
            'pip_freeze':freeze.stdout.splitlines(),'uploads':False}


def segmentation(job, workspace, device, evaluate_only=False):
    """Reuse TaskRunner with an explicit prohibition on SAE fitting in evaluate mode."""
    from mpsae.io import atomic_json, canonical_hash, exclusive_lock
    from mpsae.patch_scene.config import Config
    from mpsae.patch_scene.artifacts import Hub
    from mpsae.patch_scene.features import resolve_sources
    from mpsae.patch_scene.execution import TaskRunner, preparation_tasks, evaluation_tasks, record_result
    c=Config(**job['config'])
    campaign=canonical_hash(c.identity())[:20]
    with exclusive_lock(workspace/'.campaign.lock',timeout=0):
        hub=Hub(workspace)
        atomic_json(workspace/f'campaigns/{campaign}/settings/config.json',c.plan())
        sources=resolve_sources(c,hub)
        runner=TaskRunner(c,sources,hub,device,allow_training=not evaluate_only)
        emit=lambda row:record_result(hub,campaign,row)
        for preparation in preparation_tasks(c.encoder_names,c.dataset_names):
            runner.run(preparation,emit)
            for task in evaluation_tasks(c,preparation):
                runner.run(task,emit)


def execute(plan, paths, *, device='cuda:0', evaluate_only=False, download_embeddings=False,
            cpu_threads=4):
    check_source_lock()
    import torch
    torch.set_num_threads(cpu_threads)
    info=runtime(device)
    if any(j['runner']=='segmentation' and 'nyuv2' in j['config']['datasets'] for j in plan['jobs']):
        root=paths['nyuv2_root']
        if root is None:
            root=paths['segmentation']/'datasets/nyuv2'
            subprocess.run([sys.executable,'-u','-m','mpsae.patch_scene.prepare_nyuv2',
                            '--from-hf','--destination',str(root)],cwd=REPO,check=True)
        os.environ['MPSAE_NYUV2_ROOT']=str(root)
    if download_embeddings:
        backbones=sorted({b for j in plan['jobs'] if j['runner']!='segmentation' for b in j['config']['backbones']})
        if backbones:
            # Read-only dataset downloads; no artifact publication.
            from mpsae.embeddings_hub import Hub,DEFAULT_REPO,sync,link_campaign
            hub=Hub.connect(os.environ.get('MPSAE_EMBEDDINGS_REPO',DEFAULT_REPO))
            root=paths['imagenet'].parent
            pairs=[('imagenet',b) for b in backbones]
            sync(pairs,root/'embeddings-hub',root,hub,download_only=True)
            link_campaign(root/'embeddings-hub',paths['imagenet'],pairs)
    for job in plan['jobs']:
        folder=paths['output']/'launches'/job['job_id']
        write_json(folder/'job.json',job)
        write_json(folder/'runtime.json',{**info,'evaluate_only':evaluate_only})
        write_json(folder/'status.json',{'status':'running'})
        print(f"[paper] {','.join(job['studies'])}: {job['job_id']} ({'evaluate' if evaluate_only else 'train/evaluate'})",flush=True)
        try:
            if job['runner']=='imagenet':
                from mpsae.benchmark.config import Campaign
                from mpsae.benchmark.run import run_campaign
                c=Campaign.from_mapping(job['config'])
                if evaluate_only:
                    c=replace(c,execution={**c.execution,'train_missing':False})
                run_campaign(c,paths['imagenet'].parent,device=device,workers=cpu_threads,
                             evaluate_only=evaluate_only)
            elif job['runner']=='retrieval':
                from mpsae.benchmark.mass_sweep import argument_parser,run
                from mpsae.benchmark.mass_config import MassConfig
                args=argument_parser().parse_args([])
                args.workspace=paths['retrieval']; args.source_root=paths['imagenet']
                args.embeddings_dir=paths['imagenet']/'embeddings'; args.cache_dir=paths['imagenet']/'cache'
                args.device=device; args.cpu_threads=cpu_threads; args.evaluate_only=evaluate_only
                args.checkpoint_every=512; args.log_every=100; args.query_block=2048
                result=run(MassConfig(**job['config']),args)
                if not result['complete']:
                    raise RuntimeError('Retrieval job finished with incomplete evaluations')
            else:
                segmentation(job,paths['segmentation'],device,evaluate_only)
        except BaseException as error:
            write_json(folder/'status.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
            raise
        else:
            write_json(folder/'status.json',{'status':'complete'})

