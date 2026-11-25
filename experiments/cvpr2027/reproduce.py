#!/usr/bin/env python3
"""Plan, index, run and reproduce the CVPR 2027 draft experiments locally."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import HERE,read_json,write_json
from registry import build_plan
from runners import paths_from


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','index','report','verify','run','evaluate','bundle','select'])
    p.add_argument('source',nargs='?',help='workspace / ZIP / tar for index')
    p.add_argument('--studies',nargs='+',default=['all'],help='catalog names; defaults to all')
    p.add_argument('--paths',type=Path,help='local paths YAML; relative entries resolve beside this file')
    p.add_argument('--output',type=Path,help='override output/index directory')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--download-embeddings',action='store_true')
    p.add_argument('--tables',nargs='+',choices=[str(i) for i in range(1,14)])
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--allow-partial',action='store_true',help='export clearly marked incomplete tables')
    p.add_argument('--condition',help='condition ID for select')
    p.add_argument('--candidate',help='candidate ID for select; explicit conflict resolution')
    return p


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if args.cpu_threads<1:
        p.error('--cpu-threads must be positive')
    if args.command=='report' and args.studies!=['all']:
        p.error('Choose report outputs with --tables; --studies selects training/evaluation jobs')
    paths=paths_from(args.paths)
    if args.output:
        paths['output']=args.output.resolve()
    # Always index against the full catalog, even when a run selects one study.
    full=build_plan()
    selected=build_plan(args.studies)
    if args.command=='plan':
        print(json.dumps({k:v for k,v in selected.items() if k not in ('conditions','jobs')},indent=2))
        print(f"Jobs: {len(selected['jobs'])}; this is a plan, not evidence of completed runs.")
        return 0
    if args.command=='index':
        if not args.source:
            p.error('index requires a directory, ZIP, or tar path')
        from ingest import index_source
        status=index_source(args.source,paths['output'],full)
        print(f"Indexed: {status['complete']}/{status['expected']}; conflicts: {status['conflicts']}")
        print(paths['output']/'results/index.json')
        return 0
    if args.command=='select':
        if not args.condition or not args.candidate:
            p.error('select requires --condition and --candidate from results/index.json')
        from ingest import rebuild_index
        index=rebuild_index(paths['output'],full)
        row=next((r for r in index['results'] if r['condition_id']==args.condition),None)
        if not row or args.candidate not in row['candidates']:
            p.error('candidate does not belong to that condition')
        target=paths['output']/'selections.json'
        selections=read_json(target,{})
        selections[args.condition]=args.candidate
        write_json(target,selections)
        rebuild_index(paths['output'],full)
        return 0
    if args.command in ('run','evaluate'):
        from runners import execute
        from ingest import index_source
        write_json(paths['output']/'selected_plan.json',selected)
        try:
            execute(selected,paths,device=args.device,evaluate_only=args.command=='evaluate',
                    download_embeddings=args.download_embeddings,cpu_threads=args.cpu_threads)
        finally:
            # Preserve indexable partial results after an interrupted campaign.
            for kind in {j['runner'] for j in selected['jobs']}:
                path=paths[{'imagenet':'imagenet','retrieval':'retrieval','segmentation':'segmentation'}[kind]]
                if path.is_dir():
                    index_source(path,paths['output'],full)
        return 0
    if args.command=='bundle':
        from reporting import bundle
        print(bundle(paths['output']))
        return 0
    from ingest import rebuild_index
    from registry import table_conditions
    index=rebuild_index(paths['output'],full)
    ids={r['condition_id'] for r in selected['conditions']}
    if args.tables:
        ids &= {r['condition_id'] for t in args.tables for r in table_conditions(full,t)[1]}
    missing=[r for r in index['results'] if r['condition_id'] in ids and r['status']!='complete']
    if args.command=='verify':
        from reporting import verify_reference
        comparisons=verify_reference(paths['output'])
        different=[r for r in comparisons if r['status']=='different' and r['condition_id'] in ids]
        print(json.dumps({'selected':len(ids),'incomplete_or_conflicting':len(missing),
                          'reference_differences':different,'independent_gpu_rerun':False},indent=2))
        return 1 if missing or different else 0
    if missing and not args.allow_partial:
        print(f'{len(missing)} selected conditions missing/conflicting. Index saved receipts first, '
              'or use --allow-partial for marked draft tables.',file=sys.stderr)
        return 1
    from reporting import report
    manifest=report(paths['output'],full,tables=args.tables,plots=not args.no_plots)
    print(f"Reports: {paths['output']/'reports'} ({manifest['complete']}/{manifest['expected']} catalog conditions)")
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError) as error:
        print(f'Error: {error}',file=sys.stderr)
        raise SystemExit(2)
