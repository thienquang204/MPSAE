"""Atomic hashed bundles; identical paths locally and in a private HF dataset."""
from pathlib import Path
import os

from ..io import atomic_json, read_json, sha256_file


def seal(folder):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha256_file(p)
             for p in sorted(folder.rglob('*')) if p.is_file()
             and p.name != 'bundle.json' and not p.name.endswith(('.partial', '.lock', '.tmp'))}
    atomic_json(folder/'bundle.json', {'files': files})


def verify(folder):
    folder = Path(folder)
    manifest = read_json(folder/'bundle.json')
    if manifest is None:
        return False
    for name, digest in manifest['files'].items():
        path = folder/name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('invalid bundle path')
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f'Corrupt artifact: {path}')
    return True


def save_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Hub:
    def __init__(self, root, repo=None, *, publish=False):
        self.root, self.repo, self.publish = Path(root), repo, publish
        self.files, self.revision = set(), None
        if repo:
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(repo)
            self.files = {f.rfilename for f in info.siblings}
            self.revision = info.sha

    def pull(self, relative):
        directory = self.root/relative
        prefix = f'scene-patch-v2/{relative}'
        try:
            if verify(directory):
                return directory
        except RuntimeError:
            if not (self.repo and f'{prefix}/bundle.json' in self.files):
                raise
            print(f'[restore] replacing incomplete/corrupt local bundle {relative}', flush=True)
        if self.repo and f'{prefix}/bundle.json' in self.files:
            from huggingface_hub import hf_hub_download
            import shutil
            # Download into HF cache, then atomically place only this bundle's files.
            manifest_path = hf_hub_download(self.repo, f'{prefix}/bundle.json', repo_type='dataset',
                                            revision=self.revision)
            manifest = read_json(manifest_path)
            for name in [*manifest['files'], 'bundle.json']:
                target = directory/name
                if not target.resolve().is_relative_to(directory.resolve()):
                    raise ValueError('invalid remote bundle path')
                source = hf_hub_download(self.repo, f'{prefix}/{name}', repo_type='dataset',
                                         revision=self.revision)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target.with_suffix(target.suffix+'.partial'))
                os.replace(target.with_suffix(target.suffix+'.partial'), target)
            verify(directory)
            print(f'[restore] {relative}', flush=True)
        return directory

    def push(self, relative, *, remote=True):
        directory = self.root/relative
        seal(directory)
        if self.repo and self.publish and remote:
            from huggingface_hub import HfApi
            manifest = read_json(directory/'bundle.json')
            HfApi().upload_folder(repo_id=self.repo, repo_type='dataset', folder_path=str(directory),
                path_in_repo=f'scene-patch-v2/{relative}',
                allow_patterns=[*manifest['files'], 'bundle.json'],
                commit_message=f'Checkpoint patch benchmark: {relative}')
            print(f'[publish] {relative}', flush=True)
