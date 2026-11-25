"""Read original JSON receipts from workspaces or archives without loading weights.

Imports are content addressed. Archives are read in place, never extracted.
Ambiguous reruns are conflicts, not an invitation to choose the highest score.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from common import digest, read_json, write_json, write_text
from registry import condition_id, condition_key

NAMES = {'config.json', 'summary.json', 'eval_e1.json', 'eval_e2.json', 'rows.json',
         'results.json', 'result.json', 'plan.json', 'runtime.json', 'resolved.json', 'catalog.json', 'sources.json'}
SKIP = {'.git', '.venv', '.venv-segmentation', '__pycache__', 'features', 'embeddings', 'cache', 'imports', 'SMOKE_ONLY'}
MAX_FILE = 64 * 1024**2
MAX_TOTAL = 512 * 1024**2


def _safe_name(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or any(':' in part for part in p.parts):
        raise ValueError(f'Unsafe archive entry: {name}')
    return p.as_posix()


def documents(source):
    """Bounded metadata reads; no pickle, checkpoint, image, or executable imports."""
    source = Path(source).resolve()
    result, size = {}, 0

    def add(name, data):
        nonlocal size
        name = _safe_name(name)
        size += len(data)
        if len(data) > MAX_FILE or size > MAX_TOTAL:
            raise ValueError('Metadata import exceeds size limit; import individual campaigns')
        obj = json.loads(data.decode('utf-8-sig'))
        if name in result:
            raise ValueError(f'Duplicate archive entry: {name}')
        result[name] = obj

    if source.is_dir():
        for base, dirs, files in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv')
                             and not (Path(base)/d).is_symlink())
            for name in sorted(set(files) & NAMES):
                p = Path(base)/name
                if p.is_symlink():
                    continue
                if p.stat().st_size > MAX_FILE:
                    raise ValueError(f'Metadata file too large: {p}')
                add(p.relative_to(source).as_posix(), p.read_bytes())
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                name = _safe_name(member.filename)
                if PurePosixPath(name).name not in NAMES or set(PurePosixPath(name).parts) & SKIP:
                    continue
                if member.file_size > MAX_FILE or size + member.file_size > MAX_TOTAL:
                    raise ValueError('Archive metadata exceeds size limit')
                add(name, archive.read(member))
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            for member in archive:
                name = _safe_name(member.name)
                if PurePosixPath(name).name not in NAMES or set(PurePosixPath(name).parts) & SKIP:
                    continue
                if not member.isfile() or member.size > MAX_FILE or size + member.size > MAX_TOTAL:
                    raise ValueError(f'Unsupported or oversized metadata entry: {name}')
                add(name, archive.extractfile(member).read())
    else:
        raise ValueError('Import a workspace directory, ZIP, or tar archive containing original JSON receipts')
    return result


def nearby(docs, name, filename):
    parent = PurePosixPath(name).parent
    for _ in range(6):
        for candidate in (parent/filename, parent/'settings'/filename):
            if candidate.as_posix() in docs:
                return docs[candidate.as_posix()]
        if parent == parent.parent:
            break
        parent = parent.parent
    return {}


def training_fields(identity):
    p = identity.get('preset', {})
    obj = identity.get('objective', {})
    schedule = identity.get('schedule', {})
    ot = obj.get('ot', {})
    return dict(batch_size=schedule.get('batch_size', p.get('batch_size')),
                epochs=schedule.get('epochs', p.get('epochs')),
                lr=identity.get('optimizer', {}).get('lr', p.get('lr')),
                mass=ot.get('mass', p.get('ot_mass')),
                loss_weight=ot.get('weight', p.get('ot_weight')))


def _float(value):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f'Invalid numeric metric: {value}')
    return float(value)


def normalize(docs):
    summaries = {v.get('checkpoint_sha256'): v for name, v in docs.items()
                 if name.endswith('/summary.json') and isinstance(v, dict) and v.get('checkpoint_sha256')}
    for name, payload in sorted(docs.items()):
        if not isinstance(payload, dict) or payload.get('scientific') is False:
            continue
        file = PurePosixPath(name).name
        if file in ('eval_e1.json', 'eval_e2.json'):
            summary = nearby(docs, name, 'summary.json')
            cfg = nearby(docs, name, 'config.json')
            identity = summary.get('identity', cfg.get('identity', {}))
            if summary and not summary.get('complete'):
                continue
            signature = payload.get('signature')
            if identity and signature and digest(identity) != signature:
                raise ValueError(f'{name}: training identity checksum differs from evaluation signature')
            if signature and summary.get('signature') and signature != summary['signature']:
                raise ValueError(f'{name}: evaluation and training signatures differ')
            if signature and cfg.get('signature') and signature != cfg['signature']:
                raise ValueError(f'{name}: evaluation and config signatures differ')
            points = payload.get('points', []) if file == 'eval_e2.json' else [payload]
            for point in points:
                base = dict(dataset='imagenet', encoder=payload.get('backbone', identity.get('backbone')),
                            method=payload.get('method', identity.get('method')),
                            seed=payload.get('seed', identity.get('seed')),
                            train_k=payload.get('anchor_k', payload.get('k', identity.get('target_k'))),
                            inference_k=point.get('k'), **training_fields(identity))
                for task, metric in (('imagenet_1nn','top1'), ('imagenet_probe','probe_top1')):
                    if point.get(metric) is not None:
                        yield dict(base, task=task, value=point[metric], source=name,
                                   training_signature=signature, training_identity=identity,
                                   checkpoint_sha256=summary.get('checkpoint_sha256'),
                                   evaluation_identity=point.get('probe_settings') if task == 'imagenet_probe' else point.get('search_precision'),
                                   gallery_mean_nnz=point.get('nnz_mean'), query_mean_nnz=point.get('query_nnz_mean'),
                                   empty_queries=point.get('empty_queries'),
                                   query_nnz_histogram=point.get('query_nnz_histogram'),
                                   steps=summary.get('steps'), epochs_completed=identity.get('schedule',{}).get('epochs'),
                                   metadata_level='training_identity' if identity else 'unverified')
        elif file in ('results.json','rows.json','result.json'):
            rows = payload.get('results', [payload] if payload.get('complete') and 'dataset' in payload else [])
            for row in rows:
                if not isinstance(row, dict) or row.get('status', 'complete') != 'complete' or row.get('complete') is False:
                    continue
                if 'backbone' in row and 'top1_percent' in row:
                    identity = row.get('training_identity', {})
                    yield dict(task='imagenet_1nn', dataset='imagenet',encoder=row['backbone'],
                               method=row.get('method','mpsae'),seed=row.get('seed'),train_k=row['train_k'],
                               inference_k=row['inference_k'],**training_fields(identity),value=row['top1_percent'],
                               source=name, training_signature=row.get('signature'), training_identity=identity,
                               checkpoint_sha256=row.get('checkpoint_sha256'), evaluation_identity=row.get('evaluation_identity'),
                               gallery_mean_nnz=row.get('gallery_mean_nnz'),query_mean_nnz=row.get('query_mean_nnz'),
                               empty_queries=row.get('empty_queries'),query_nnz_histogram=row.get('query_nnz_histogram'),
                               steps=identity.get('schedule',{}).get('total_steps'),
                               metadata_level='training_identity' if identity else 'unverified')
                elif 'encoder' in row and 'miou' in row:
                    cfg_doc = nearby(docs, name, 'config.json')
                    cfg = cfg_doc.get('settings', {})
                    evaluation = row.get('identity', {})
                    summary = summaries.get(evaluation.get('fit'), {})
                    identity = summary.get('identity', {})
                    fields = training_fields(identity)
                    if not identity and cfg:
                        fields.update(batch_size=cfg.get('batch_size',1024),epochs=cfg.get('epochs',10),lr=cfg.get('lr',4e-5),
                                      mass=cfg.get('mpsae_ot_mass',.9),loss_weight=cfg.get('mpsae_ot_weight'))
                        if fields['loss_weight'] is None:
                            fields['loss_weight'] = cfg.get('siglip2_mpsae_ot_weight') if row['encoder']=='siglip2' else None
                        if fields['loss_weight'] is None:
                            fields['loss_weight'] = 2.0
                    yield dict(task='segmentation_nn',dataset=row['dataset'],encoder=row['encoder'],method=row['method'],
                               seed=row.get('seed'),train_k=row.get('train_k'),inference_k=row.get('inference_k',row.get('k')),
                               **fields,value=100*_float(row['miou']), pixel_accuracy_percent=100*_float(row['pixel_accuracy']) if row.get('pixel_accuracy') is not None else None,
                               source=name,training_signature=digest(identity) if identity else None,training_identity=identity,
                               checkpoint_sha256=evaluation.get('fit'),evaluation_identity=evaluation,
                               gallery_mean_nnz=row.get('gallery_mean_nnz'),query_mean_nnz=row.get('mean_nnz'),
                               empty_queries=row.get('empty_patches'),steps=summary.get('steps'),
                               epochs_completed=row.get('training',{}).get('epochs_completed',summary.get('epochs_completed')),
                               source_config=cfg,metadata_level='training_identity' if identity else 'campaign_config' if cfg else 'unverified')


def index_source(source, output, plan):
    docs = documents(source)
    source_id = digest(docs)[:20]
    output = Path(output)
    imported = output/'imports'/source_id
    write_json(imported/'receipts.json', docs)
    write_json(imported/'source.json', {'source': str(Path(source).resolve()), 'metadata_sha256':digest(docs)})
    expected = {r['condition_id']:r for r in plan['conditions']}
    candidates, excluded = [], []
    for row in normalize(docs):
        if 'synthetic' in json.dumps(row.get('training_identity', {})).lower() or 'synthetic' in json.dumps(row.get('evaluation_identity', {})).lower():
            excluded.append({'source':row['source'],'reason':'synthetic data'})
            continue
        if row['method'] != 'mpsae':
            row['mass'] = row['loss_weight'] = None
        if row['method'] == 'dense':
            for key in ('train_k','inference_k','batch_size','epochs','lr'):
                row[key] = None
        for key in ('mass','loss_weight','lr'):
            row[key] = _float(row.get(key))
        for key in ('seed','train_k','inference_k','batch_size','epochs'):
            if row.get(key) is not None:
                if type(row[key]) is bool or int(row[key]) != row[key]:
                    raise ValueError(f'{row["source"]}: invalid {key}')
                row[key] = int(row[key])
        cid = condition_id(row)
        value = _float(row['value'])
        if value is None or not 0 <= value <= 100:
            raise ValueError(f'{row["source"]}: percentage outside [0,100]')
        if cid not in expected or row['metadata_level']=='unverified':
            excluded.append({'source':row['source'], 'condition':condition_key(row),
                             'reason':'outside paper grid or missing training metadata'})
            continue
        recipe = expected[cid]
        identity=row.get('training_identity', {})
        signature=row.get('training_signature')
        if identity and signature and digest(identity)!=signature:
            raise ValueError(f'{row["source"]}: invalid training identity checksum')
        preset=recipe.get('preset') or {}
        mismatches=[]
        for key,preset_value in identity.get('preset', {}).items():
            if key=='name' or (key.startswith('ot_') and row['method']!='mpsae') or (key.startswith('cl_') and row['method']=='mpsae'):
                continue
            if key in preset and json.dumps(preset_value)!=json.dumps(preset[key]):
                mismatches.append('preset.'+key)
        for key,target in [('weight_decay','weight_decay'),('eps','adam_eps'),('betas','adam_betas')]:
            actual=identity.get('optimizer', {}).get(key)
            if actual is not None and json.dumps(actual)!=json.dumps(preset.get(target)):
                mismatches.append('optimizer.'+key)
        obj=identity.get('objective', {})
        if obj.get('hidden_dim',preset.get('hidden_dim'))!=preset.get('hidden_dim'):
            mismatches.append('hidden_dim')
        for key,target in [('eta','ot_eta'),('microbatch','ot_microbatch'),('max_iters','ot_max_iters'),
                           ('max_refine_iters','ot_max_refine_iters'),('tol','ot_tol')]:
            if key in obj.get('ot',{}) and obj['ot'][key]!=preset.get(target):
                mismatches.append('ot.'+key)
        for key,target in [('weight','cl_coef'),('temperature','cl_temperature')]:
            if key in obj.get('contrastive',{}) and obj['contrastive'][key]!=preset.get(target):
                mismatches.append('contrastive.'+key)
        for key,target in [('k_aux','k_aux'),('coef','aux_coef'),('dead_threshold','dead_threshold')]:
            if key in obj.get('aux',{}) and obj['aux'][key]!=preset.get(target):
                mismatches.append('aux.'+key)
        if 'decay_ratio' in obj.get('annealing',{}) and obj['annealing']['decay_ratio']!=preset.get('k_decay_ratio'):
            mismatches.append('annealing.decay_ratio')
        if mismatches:
            excluded.append({'source':row['source'],'reason':'training recipe differs: '+','.join(mismatches)})
            continue
        if row['method'] != 'dense':
            for field in ('gallery_mean_nnz','query_mean_nnz'):
                if row.get(field) is not None and not 0 <= float(row[field]) <= row['inference_k'] + 1e-5:
                    raise ValueError(f'{row["source"]}: {field} violates K sparsity')
        cfg = row.get('source_config', {})
        if cfg and isinstance(recipe['protocol'],dict):
            changed = [k for k,v in recipe['protocol'].items() if k in cfg and cfg[k] != v]
            if changed:
                excluded.append({'source':row['source'],'reason':'segmentation protocol differs: '+','.join(changed)})
                continue
        row.update(condition_id=cid,value=value,source_id=source_id,
                   metric=recipe['metric'],verification='indexed; not independently rerun')
        # Receipt identity separates reruns. Exact duplicate E1/E2 points merge.
        proof = row.get('checkpoint_sha256') or row.get('training_signature') or digest(row.get('evaluation_identity'))
        values = {k:row.get(k) for k in ('value','pixel_accuracy_percent','gallery_mean_nnz','query_mean_nnz','empty_queries','query_nnz_histogram')}
        row['candidate_id'] = digest({'condition':cid,'checkpoint':proof,'values':values})[:20]
        candidates.append(row)
    write_json(imported/'candidates.json', candidates)
    write_json(imported/'excluded.json', excluded)
    write_json(imported/'inventory.json',{'documents':len(docs),'candidates':len(candidates),
                                        'excluded':len(excluded),'source':str(Path(source).resolve())})
    return rebuild_index(output, plan)


def rebuild_index(output, plan):
    output = Path(output)
    lookup = {}
    for path in sorted((output/'imports').glob('*/candidates.json')):
        for row in read_json(path, []):
            entry = lookup.setdefault(row['candidate_id'], {**row,'sources':[]})
            entry['sources'].append({'import':row['source_id'],'file':row['source']})
    by_condition = {}
    for row in lookup.values():
        by_condition.setdefault(row['condition_id'],[]).append(row)
    selections = read_json(output/'selections.json', {})
    results = []
    for expected in plan['conditions']:
        cid = expected['condition_id']
        candidates = by_condition.get(cid, [])
        choice = selections.get(cid)
        if choice is not None and choice not in {c['candidate_id'] for c in candidates}:
            raise ValueError(f'Selection {cid}/{choice} does not exist in imported evidence')
        selected = next((c for c in candidates if c['candidate_id']==choice), None)
        if selected is None and len(candidates)==1:
            selected = candidates[0]
        status = 'complete' if selected else 'conflict' if candidates else 'missing'
        results.append({**expected,'status':status,'result':selected,
                        'candidates':[c['candidate_id'] for c in candidates]})
    index = {'schema':1,'paper':plan['paper'],'expected':len(results),
             'complete':sum(r['status']=='complete' for r in results),
             'conflicts':sum(r['status']=='conflict' for r in results),'results':results}
    write_json(output/'results/index.json',index)
    write_text(output/'results/provenance.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in lookup.values()))
    return index
