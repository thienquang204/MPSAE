"""Run or export the 72 ImageNet + 7 linear segmentation evaluations locally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from ..io import atomic_json, read_json


def build_plan(retrieval_config, segmentation_config):
    import yaml
    from .retrieval_studies import plan_for
    from ..patch_scene.linear_probe import load, plan_for as linear_plan
    retrieval = plan_for(yaml.safe_load(Path(retrieval_config).read_text(encoding='utf-8')))
    scene, probe = load(segmentation_config)
    segmentation = linear_plan(scene, probe)
    return {'retrieval': retrieval, 'segmentation': segmentation,
            'evaluations': retrieval['unique_evaluations'] + segmentation['evaluations'],
            'sae_fits_total': retrieval['unique_fits'] + segmentation['max_sae_fits'],
            'linear_head_fits': segmentation['probe_fits']}


def package(root, segmentation_root, plan, *, smoke=False):
    rroot = root/('SMOKE_ONLY' if smoke else '')
    sroot = segmentation_root/('SMOKE_ONLY' if smoke else '')
    selected = [('retrieval', rroot/'studies'/plan['retrieval']['suite_id']/'retrieval-studies-results.zip',
                 plan['retrieval']['unique_evaluations']),
                ('segmentation', sroot/'linear-studies'/plan['segmentation']['campaign_id']/'linear-segmentation-results.zip',
                 plan['segmentation']['evaluations'])]
    rroot.mkdir(parents=True, exist_ok=True)
    archive = rroot/'ablation79-results.zip'
    status = {'expected': plan['evaluations'], 'completed': 0, 'scientific': not smoke, 'stages': {}}
    with ZipFile(archive.with_suffix('.partial'), 'w', ZIP_DEFLATED) as target:
        target.writestr('plan.json', json.dumps(plan, indent=2))
        for stage, source, expected in selected:
            done = 0
            if source.exists():
                with ZipFile(source) as bundle:
                    report = json.loads(bundle.read('results.json'))
                    done = report['completed']
                    for name in bundle.namelist():
                        target.writestr(f'{stage}/{name}', bundle.read(name))
            status['completed'] += done
            status['stages'][stage] = {'completed': done, 'expected': expected}
        status['complete'] = status['completed'] == status['expected']
        target.writestr('status.json', json.dumps(status, indent=2))
    os.replace(archive.with_suffix('.partial'), archive)
    atomic_json(rroot/'ablation79-status.json', status)
    print(f'[combined] {status["completed"]}/{status["expected"]}; download {archive}', flush=True)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('work/retrieval-studies'))
    p.add_argument('--source-root', type=Path)
    p.add_argument('--segmentation-workspace', type=Path, help='Existing VOC feature/SAE workspace, if available')
    p.add_argument('--retrieval-config', default='configs/retrieval_studies.yaml')
    p.add_argument('--segmentation-config', default='configs/linear_scene_voc.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--query-block', type=int, default=2048)
    p.add_argument('--gallery-chunk', type=int, default=32768)
    p.add_argument('--download-embeddings', action='store_true')
    p.add_argument('--only', choices=('all', 'retrieval', 'segmentation'), default='all')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--export-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    plan = build_plan(args.retrieval_config, args.segmentation_config)
    if args.plan:
        print(json.dumps(plan, indent=2)); return 0
    root = args.workspace.resolve()
    segmentation = (args.segmentation_workspace or root/'segmentation').resolve()
    shared = ['--device', args.device, '--cpu-threads', str(args.cpu_threads)]
    if args.smoke:
        shared.append('--smoke')
    if args.export_only:
        shared.append('--export-only')
    try:
        if args.only in ('all', 'retrieval'):
            command = [sys.executable, '-u', '-m', 'mpsae.benchmark.retrieval_studies',
                '--config', args.retrieval_config, '--workspace', str(root),
                '--source-root', str((args.source_root or root/'source').resolve()),
                '--query-block', str(args.query_block), '--gallery-chunk', str(args.gallery_chunk), *shared]
            if args.download_embeddings:
                command.append('--download-embeddings')
            print('>>>', ' '.join(command), flush=True)
            subprocess.run(command, check=True)
        if args.only in ('all', 'segmentation'):
            command = [sys.executable, '-u', '-m', 'mpsae.patch_scene.linear_probe',
                '--config', args.segmentation_config, '--workspace', str(segmentation), *shared]
            print('>>>', ' '.join(command), flush=True)
            subprocess.run(command, check=True)
    finally:
        package(root, segmentation, plan, smoke=args.smoke)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
