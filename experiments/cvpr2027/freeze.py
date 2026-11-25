#!/usr/bin/env python3
"""Maintainer command: explicitly freeze reviewed recipes and scientific source."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import HERE,REPO,digest,write_json
from registry import build_plan

SOURCES=('mpsae/config.py','mpsae/model.py','mpsae/methods.py','mpsae/losses.py',
         'mpsae/ot.py','mpsae/ot_balanced.py','mpsae/schedules.py','mpsae/train.py',
         'mpsae/runs.py','mpsae/evaluate/knn.py','mpsae/evaluate/probe.py',
         'mpsae/segmentation/core.py','mpsae/patch_scene/encoders.py',
         'mpsae/patch_scene/features.py','mpsae/patch_scene/train.py',
         'mpsae/patch_scene/retrieval.py','mpsae/patch_scene/evaluate.py')


def freeze():
    plan=build_plan(check_lock=False)
    conditions=[{k:v for k,v in r.items() if k not in ('studies','job_ids')} for r in plan['conditions']]
    write_json(HERE/'reference/recipes.lock.json',{
        'schema':1,'conditions':{r['condition_id']:digest(r) for r in conditions},
        'presets':{digest(r['preset']):r['preset'] for r in conditions if r['preset']},
        'protocols':{digest(r['protocol']):r['protocol'] for r in conditions},
        'counts':{k:plan[k] for k in ('unique_evaluations','unique_sae_fits','probe_fits')}})
    write_json(HERE/'reference/source.lock.json',{'schema':1,'normalization':'UTF-8 with LF',
        'files':{name:hashlib.sha256((REPO/name).read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()
                 for name in SOURCES}})
    print('Wrote reviewed recipe/source locks. Inspect and commit their diff with the recipe changes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',required=True,
                        help='acknowledge an intentional reviewed recipe/source change')
    parser.parse_args()
    freeze()
