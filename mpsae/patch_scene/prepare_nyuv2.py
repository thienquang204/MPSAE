"""Attach a local labeled MAT without copying it; fetch two pinned small metadata files."""
from pathlib import Path
import argparse
import hashlib
import os
import urllib.request

from ..io import sha256_file
from .nyuv2 import METADATA_REVISION, METADATA_SHA256, load_local

HF_REPO = 'andrew-healey/nyuv2'
HF_REVISION = '742f4d8a35f024263c6a9f7f316954b06ec5a817'
LABELED_SHA256 = '2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa'


def download_labeled(destination):
    """Pinned public mirror; keep the resumable download local to the workspace."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=HF_REPO, repo_type='model',
        filename='nyu_depth_v2_labeled.mat', revision=HF_REVISION,
        local_dir=str(destination), token=False))
    print('Verifying NYUv2 labeled MAT checksum...', flush=True)
    if sha256_file(path) != LABELED_SHA256:
        raise ValueError('NYUv2 labeled MAT checksum mismatch')
    return path


def fetch_metadata(name, destination):
    destination = Path(destination)
    digest = METADATA_SHA256[name]
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError(f'Unexpected metadata contents: {destination}')
        return
    url = f'https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/{METADATA_REVISION}/{name}'
    request = urllib.request.Request(url, headers={'User-Agent': 'MPSAE-NYUv2-setup'})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(1024*1024+1)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'Unexpected download/hash for {name}; refusing to use it')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    matches = ([source] if source.is_file() and source.name == 'nyu_depth_v2_labeled.mat'
               else sorted(source.rglob('nyu_depth_v2_labeled.mat')) if source.is_dir() else [])
    if len(matches) != 1:
        raise FileNotFoundError('Attach exactly one nyu_depth_v2_labeled.mat to Kaggle (the ~2.8 GB labeled '
            f'dataset), then select its input directory. Found {len(matches)} under {source}. '
            'Official page: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html')
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'nyu_depth_v2_labeled.mat'
    if target.exists():
        if not target.samefile(matches[0]):
            raise ValueError(f'{target} already points to a different dataset')
    else:
        target.symlink_to(matches[0])  # Kaggle input stays read-only; no extra 2.8 GB copy
    for name in METADATA_SHA256:
        output = destination/name
        candidates = sorted((source if source.is_dir() else source.parent).rglob(name))
        if not output.exists() and len(candidates) == 1:
            if sha256_file(candidates[0]) != METADATA_SHA256[name]:
                raise ValueError(f'Attached metadata is not the pinned version: {candidates[0]}')
            output.symlink_to(candidates[0])
        fetch_metadata(name, output)
    data, _ = load_local(destination)
    print(f'NYUv2-40 ready: {len(data["train"])} train / {len(data["validation"])} official test images. '
          f'RGB only. Root: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', help='Attached dataset folder or labeled MAT file')
    source.add_argument('--from-hf', action='store_true', help='Download the pinned public labeled MAT; no token needed')
    parser.add_argument('--destination', required=True, help='Writable small metadata/link directory')
    args = parser.parse_args()
    source = download_labeled(Path(args.destination).parent/'hf_dataset') if args.from_hf else args.source
    prepare(source, args.destination)
