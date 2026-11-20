"""Pinned Chris1 mirror; original train/validation Parquets, never viewer/test data."""
import io
import hashlib
import shutil
import threading
import time
from bisect import bisect_right
from pathlib import Path
import numpy as np
from PIL import Image

from .cityscapes import CitySpec

from .config import CITYSCAPES_HF

REPO, REVISION, _ = CITYSCAPES_HF


class HFCitySpec(CitySpec):
    def remap(self, mask):
        raw = np.asarray(mask)
        # This mirror stores labelIds as lossless RGB PNGs with identical channels.
        # Never convert a color palette visualization to luminance/class IDs.
        if raw.ndim == 3:
            if raw.shape[2] != 3 or not (np.array_equal(raw[..., 0], raw[..., 1])
                                        and np.array_equal(raw[..., 0], raw[..., 2])):
                raise ValueError('Cityscapes mirror mask must contain identical label-ID channels')
            raw = raw[..., 0]
        return super().remap(raw)


def parquet_files(revision=REVISION):
    if revision != REVISION:
        raise ValueError('Unvalidated Cityscapes mirror revision; review before changing it')
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/data'
    return {split: [f'{base}/{split}-{i:05d}-of-{count:05d}.parquet' for i in range(count)]
            for split, count in [('train', 14), ('validation', 3)]}


def _download_status(cache, names, revision, started):
    """Report filesystem state, not an inferred network transfer rate."""
    repository = cache/'datasets--Chris1--cityscapes'
    snapshot = repository/'snapshots'/revision
    complete = sum((snapshot/name).is_file() for name in names)
    partials = []
    for path in (repository/'blobs').glob('*.incomplete'):
        try:
            partials.append(path.stat().st_size)
        except FileNotFoundError:  # downloader may have just renamed the file
            pass
    existing = cache
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    free = shutil.disk_usage(existing).free/2**30
    return (f'[cityscapes download] elapsed={time.monotonic()-started:.0f}s '
            f'cached={complete}/{len(names)} partial_files={len(partials)} '
            f'partial_logical_GiB={sum(partials)/2**30:.2f} free_GiB={free:.1f}')


def download_snapshot(names, revision=REVISION, *, heartbeat_seconds=30):
    from huggingface_hub import snapshot_download, constants
    cache = Path(constants.HF_HUB_CACHE)
    started = time.monotonic()
    stop = threading.Event()
    print(f'[cityscapes download] {REPO}@{revision}; {len(names)} train/validation shards; '
          f'2 workers; cache={cache}', flush=True)
    print('[cityscapes download] Status every 30s. Partial sizes are logical file sizes, '
          'not bytes transferred; unchanged sizes alone do not establish a stalled download.', flush=True)
    def status():
        try:
            print(_download_status(cache, names, revision, started), flush=True)
        except OSError as error:
            print(f'[cityscapes download] Cannot inspect cache: {error}', flush=True)
    def monitor():
        while not stop.wait(heartbeat_seconds):
            status()
    status()
    thread = threading.Thread(target=monitor, name='cityscapes-download-status', daemon=True)
    thread.start()
    try:
        root = Path(snapshot_download(REPO, repo_type='dataset', revision=revision,
                    allow_patterns=names, max_workers=2))
        print('[cityscapes download] All selected shards available; checking Parquet metadata next.', flush=True)
        return root
    except Exception:
        print('[cityscapes download] Download failed. Completed cache files are retained. '
              'Check the following traceback for network, disk or permission errors before retrying.', flush=True)
        raise
    finally:
        stop.set()
        thread.join(timeout=1)


class ParquetRows:
    """Read source Parquets directly, avoiding a second full Arrow copy on Kaggle SSD."""
    column_names = ('id', 'image', 'mask')

    def __init__(self, paths):
        import pyarrow.parquet as pq
        self.groups, self.ends = [], []
        self._ids, self._cache = None, None
        total = 0
        for path in paths:
            with pq.ParquetFile(path) as file:
                if set(file.schema_arrow.names) != {'image', 'semantic_segmentation'}:
                    raise ValueError('Unexpected Cityscapes mirror schema')
                for group in range(file.num_row_groups):
                    total += file.metadata.row_group(group).num_rows
                    self.groups.append((str(path), group))
                    self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getstate__(self):
        return {**self.__dict__, '_cache': None}

    def __getitem__(self, index):
        import pyarrow.parquet as pq
        if index == 'id':
            if self._ids is None:
                ids = []
                started = logged = time.monotonic()
                print(f'[cityscapes IDs] Hashing image bytes for {len(self)} rows (CPU/disk); no GPU work yet.', flush=True)
                for path, group in self.groups:
                    with pq.ParquetFile(path) as file:
                        for batch in file.iter_batches(batch_size=8, row_groups=[group], columns=['image']):
                            ids.extend(hashlib.sha256(row['image']['bytes']).hexdigest()
                                       for row in batch.to_pylist())
                    if time.monotonic()-logged >= 30:
                        print(f'[cityscapes IDs] {len(ids)}/{len(self)} rows; elapsed={time.monotonic()-started:.0f}s', flush=True)
                        logged = time.monotonic()
                self._ids = ids
                print(f'[cityscapes IDs] Complete: {len(ids)} rows in {time.monotonic()-started:.1f}s', flush=True)
            return self._ids
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect_right(self.ends, index)
        if self._cache is None or self._cache[0] != group:
            path, number = self.groups[group]
            with pq.ParquetFile(path) as file:
                self._cache = group, file.read_row_group(number)
        offset = index - (self.ends[group-1] if group else 0)
        row = self._cache[1].slice(offset, 1).to_pylist()[0]
        result = {'id': hashlib.sha256(row['image']['bytes']).hexdigest()}
        for original, name in [('image', 'image'), ('semantic_segmentation', 'mask')]:
            with Image.open(io.BytesIO(row[original]['bytes'])) as image:
                if image.size != (2048, 1024) or (name == 'mask' and image.format != 'PNG'):
                    raise ValueError('Expected native Cityscapes geometry and lossless PNG labels')
                result[name] = image.copy()
        return result


def load_hf(revision=REVISION):
    files = parquet_files(revision)
    names = {split: [f'data/{url.rsplit("/", 1)[1]}' for url in urls] for split, urls in files.items()}
    # One source copy on disk; no test split, raw ZIP extraction, or Arrow conversion.
    root = download_snapshot([name for paths in names.values() for name in paths], revision)
    data = {split: ParquetRows([root/name for name in paths]) for split, paths in names.items()}
    spec = HFCitySpec()
    for split, expected in [('train', spec.train_rows), ('validation', spec.validation_rows)]:
        if len(data[split]) != expected:
            raise ValueError(f'Cityscapes {split}: expected {expected}, got {len(data[split])}')
        print(f'[cityscapes validation] {split}: {len(data[split])} rows; decoding first image/mask.', flush=True)
        row = data[split][0]
        if row['image'].size != (2048, 1024) or row['mask'].size != row['image'].size:
            raise ValueError('Unexpected Cityscapes source geometry')
        spec.remap(row['mask'])
    return data


if __name__ == '__main__':
    data = load_hf()
    print(f'Cityscapes {REPO}@{REVISION}: ' + ', '.join(f'{s}={len(d)}' for s, d in data.items()), flush=True)
