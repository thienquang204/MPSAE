#!/usr/bin/env python3
"""One-file Linux NVIDIA setup for the two cached-embedding retrieval studies.

Requires Python 3, git, and an installed NVIDIA driver. No raw ImageNet,
compiler/toolkit installation, backbone extraction, or Hugging Face uploads.
Remaining CLI options are forwarded to mpsae.benchmark.retrieval_studies.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


def command(argv, cwd=None, env=None):
    print('\n>>>', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('retrieval-studies'))
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--skip-setup', action='store_true', help='Use an existing Python environment and --repo')
    parser.add_argument('--local-embeddings', action='store_true', help='Use existing archives without contacting HF')
    parser.add_argument('--device', default='cuda:0')
    args, forwarded = parser.parse_known_args(argv)
    workspace = args.workspace.expanduser().resolve()
    repo = (args.repo or workspace/'code/MPSAE').expanduser().resolve()
    if not args.skip_setup:
        if sys.platform != 'linux' or not shutil.which('git') or not shutil.which('nvidia-smi'):
            raise RuntimeError('Use a Linux server image with git, Python 3 and a working NVIDIA driver')
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            command(['git', 'clone', '--branch', 'refactor', '--single-branch',
                     'https://github.com/thienquang204/MPSAE.git', repo])
        else:
            branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
            if branch != 'refactor':
                raise RuntimeError(f'Expected refactor branch, found {branch!r}')
            if args.update:
                command(['git', '-C', repo, 'pull', '--ff-only'])
    if not (repo/'mpsae/benchmark/retrieval_studies.py').exists():
        raise RuntimeError('Push the retrieval-studies update to refactor first; use --update for an old checkout')
    env = dict(os.environ, HF_HOME=str(workspace/'cache/huggingface'),
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    python = Path(sys.executable)
    if not args.skip_setup:
        uv = shutil.which('uv') or str(Path.home()/'.local/bin/uv')
        if not Path(uv).exists():
            workspace.mkdir(parents=True, exist_ok=True)
            installer = workspace/'install-uv.sh'
            urllib.request.urlretrieve('https://astral.sh/uv/install.sh', installer)
            command(['sh', installer])
            installer.unlink()
            uv = str(Path.home()/'.local/bin/uv')
        environment = repo/'.venv-retrieval'
        python = environment/'bin/python'
        if not python.exists():
            command([uv, 'venv', '--python', '3.11', '--seed', environment])
        requirements = repo/'requirements-retrieval.txt'
        signature = hashlib.sha256(requirements.read_bytes()+b'torch-2.9.1-cu128-py311').hexdigest()
        stamp = environment/'.installation-signature'
        if not stamp.exists() or stamp.read_text() != signature:
            command([uv, 'pip', 'install', '--python', python, 'torch==2.9.1',
                     '--index-url', 'https://download.pytorch.org/whl/cu128'])
            command([uv, 'pip', 'install', '--python', python, '-r', requirements])
            command([uv, 'pip', 'check', '--python', python])
            stamp.write_text(signature)
        # Test real GPU kernels, not just torch.cuda.is_available().
        command([python, '-c',
                 'import sys,torch; d=sys.argv[1]; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(d)); '
                 'x=torch.randn(128,128,device=d,requires_grad=True); '
                 'y=(x@x.T).topk(8).values.logsumexp(-1).mean(); y.backward(); '
                 'torch.cuda.synchronize(d); assert torch.isfinite(x.grad).all(); print("CUDA kernel/backward check passed")',
                 args.device], env=env)
    cli = [python, '-u', '-m', 'mpsae.benchmark.retrieval_studies',
           '--workspace', workspace, '--source-root', workspace/'source', '--device', args.device]
    if not args.local_embeddings and not any(flag in forwarded for flag in ('--smoke', '--export-only', '--plan')):
        cli.append('--download-embeddings')
    command(cli+forwarded, cwd=repo, env=env)
    # The exact path is also recorded in studies-latest.json (including partial failures).
    import json
    root = workspace/'SMOKE_ONLY' if '--smoke' in forwarded else workspace
    latest = root/'studies-latest.json'
    if latest.exists():
        receipt = json.loads(latest.read_text())
        print(f'\nDownload: {receipt["archive"]}\nCompleted {receipt["completed"]}/{receipt["expected"]}')


if __name__ == '__main__':
    main()
