"""Reusable, indexed packs of ORIGINAL JPEG bytes, not augmented tensors.

New servers can pack directly from pinned HF parquet. Existing ImageFolder
datasets can be packed without downloading again. Workers open packs lazily;
the OS page cache can share their bytes across workers and all nine runs.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import os
from pathlib import Path

import numpy as np

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, require_disk, sha256_file
from ..logs import Progress

FORMAT = "original-jpeg-packs-v1"
DTYPE = np.dtype([("pack", "<u4"), ("offset", "<u8"), ("length", "<u8"),
                  ("label", "<u4"), ("name", "S256")])


def _save_index(path, values):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_pack(directory, record):
    for key, suffix in (("bytes", ".jpegpack"), ("index_bytes", ".npy")):
        path = directory / (record["stem"] + suffix)
        if not path.is_file() or path.stat().st_size != record[key]:
            raise RuntimeError(f"incomplete/corrupt JPEG pack: {path}; restore or rebuild this cache")
    if sha256_file(directory / (record["stem"] + ".npy")) != record["index_sha256"]:
        raise RuntimeError("JPEG pack index checksum mismatch")


def _write_pack(directory, number, rows, identity):
    """Commit data and index before receipt. An interrupted pack is rewritten."""
    stem = f"pack-{number:05d}"
    receipt = directory / (stem + ".json")
    previous = read_json(receipt)
    if previous:
        if previous["identity"] != identity:
            raise RuntimeError("JPEG pack source changed; use a separate image-root/cache")
        _validate_pack(directory, previous)
        return previous
    path = directory / (stem + ".jpegpack")
    partial = path.with_suffix(".partial")
    index, digest = [], hashlib.sha256()
    with partial.open("wb") as handle:
        for data, label, name in rows:
            encoded = name.encode("utf-8")
            if not data or len(encoded) > 256 or label < 0:
                raise ValueError("invalid image bytes, label or name in JPEG pack")
            index.append((number, handle.tell(), len(data), label, encoded))
            handle.write(data)
            digest.update(data)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)
    index_path = directory / (stem + ".npy")
    _save_index(index_path, np.array(index, dtype=DTYPE))
    record = dict(stem=stem, identity=identity, rows=len(index), bytes=path.stat().st_size,
                  sha256=digest.hexdigest(), index_bytes=index_path.stat().st_size,
                  index_sha256=sha256_file(index_path))
    atomic_json(receipt, record)
    return record


def _publish(directory, records, classes, source, expected):
    arrays = [np.load(directory / (r["stem"] + ".npy"), allow_pickle=False) for r in records]
    index = np.concatenate(arrays)
    if len(index) != expected or set(index["label"].tolist()) != set(range(len(classes))):
        raise RuntimeError("JPEG pack rows or classes disagree with source")
    # Match ImageFolder's class/path ordering, including the validation split.
    index = index[np.lexsort((index["name"], index["label"]))]
    if len(index) > 1 and np.any((index["label"][1:] == index["label"][:-1]) &
                                 (index["name"][1:] == index["name"][:-1])):
        raise RuntimeError("duplicate image names in a JPEG pack split")
    path = directory / "index.npy"
    _save_index(path, index)
    manifest = dict(format=FORMAT, rows=len(index), classes=classes, source=source,
                    packs=records, index_sha256=sha256_file(path))
    atomic_json(directory / "manifest.json", manifest)
    return manifest


class PackedImages:
    def __init__(self, directory, *, verify=False):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if not self.manifest or self.manifest.get("format") != FORMAT:
            raise RuntimeError(f"no completed JPEG pack manifest: {self.directory}")
        self.classes = self.manifest["classes"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        if sha256_file(self.directory / "index.npy") != self.manifest["index_sha256"]:
            raise RuntimeError("JPEG pack global index checksum mismatch")
        for record in self.manifest["packs"]:
            _validate_pack(self.directory, record)
            if verify and sha256_file(self.directory / (record["stem"] + ".jpegpack")) != record["sha256"]:
                raise RuntimeError("JPEG pack data checksum mismatch")
        self._index, self._handles = None, OrderedDict()

    def __len__(self):
        return self.manifest["rows"]

    def __getstate__(self):
        return {**self.__dict__, "_index": None, "_handles": OrderedDict()}

    def __getitem__(self, position):
        from PIL import Image
        if self._index is None:
            self._index = np.load(self.directory / "index.npy", mmap_mode="r", allow_pickle=False)
        row = self._index[int(position)]
        pack = int(row["pack"])
        handle = self._handles.pop(pack, None)
        if handle is None:
            handle = (self.directory / f"pack-{pack:05d}.jpegpack").open("rb")
        self._handles[pack] = handle
        if len(self._handles) > 64:
            self._handles.popitem(last=False)[1].close()
        handle.seek(int(row["offset"]))
        data = handle.read(int(row["length"]))
        if len(data) != row["length"]:
            raise RuntimeError("truncated JPEG pack")
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB"), int(row["label"])

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if self._index is not None:
            self._index._mmap.close()
            self._index = None

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()


def pack_folder(folder, directory, *, expected, pack_rows=8192):
    from torchvision.datasets import ImageFolder
    from ..data.imagenet import imagenet_folder_stats
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / "prepare.lock", timeout=0):
        if (directory / "manifest.json").exists():
            return PackedImages(directory)
        stats = imagenet_folder_stats(folder)
        dataset = ImageFolder(str(folder))
        if len(dataset) != expected or len(dataset) != stats["images"]:
            raise RuntimeError("ImageFolder image count mismatch")
        source = {k: stats[k] for k in ("fingerprint", "fingerprint_kind", "images", "classes")}
        raw_split = "validation" if Path(folder).name == "val" else "train"
        receipt = read_json(Path(folder).parent / "_mpsae" / f"prepare_{raw_split}.json")
        if receipt and receipt.get("revision"):
            source["revision"] = receipt["revision"]
        identity = canonical_hash({"source": source, "pack_rows": pack_rows})
        completed_bytes = sum(p.stat().st_size for p in directory.glob("*.jpegpack"))
        require_disk(directory, max(0, stats["bytes"] - completed_bytes) + (512 << 20), what="JPEG packs")
        progress = Progress(expected, stage="finetune/pack-folder", what="images")
        records = []
        for number, start in enumerate(range(0, expected, pack_rows)):
            samples = dataset.samples[start:start + pack_rows]
            rows = ((Path(path).read_bytes(), label, Path(path).name) for path, label in samples)
            record = _write_pack(directory, number, rows, identity)
            records.append(record)
            progress.advance(record["rows"])
        _publish(directory, records, dataset.classes, source, expected)
    return PackedImages(directory)


def prepare_packs(root, *, token=None, download=False):
    """Offline reuse first; folder conversion or direct, bounded HF prefetch next."""
    from ..data import imagenet as source
    from ..embed.archive import hub_file, pinned_release
    root = Path(root)
    destination = root / "jpeg-packs-v1"
    datasets = {}
    release, classes = None, None
    for split, folder in source.SPLIT_FOLDER.items():
        directory = destination / folder
        if (directory / "manifest.json").exists():
            datasets[folder] = PackedImages(directory)
        elif (root / folder).is_dir():
            # Never pack a partial raw download as if it were complete.
            datasets[folder] = pack_folder(root / folder, directory, expected=source.EXPECTED_ROWS[split])
        else:
            if not download:
                raise FileNotFoundError(f"no ImageNet images/packs at {root}; use --download-images")
            directory.mkdir(parents=True, exist_ok=True)
            work = destination / "hub-cache"
            if release is None:
                release = pinned_release(source.IMAGENET_REPO, repo_type="dataset", revision="main",
                                          cache_dir=work, download=True, token=token)
                classes = source._synsets(release, cache_dir=work, download=True, token=token)
            with exclusive_lock(directory / "prepare.lock", timeout=0):
                shards = source._shards(release, split)
                provenance = dict(repository=source.IMAGENET_REPO, revision=release["revision"], split=split)
                identity = canonical_hash({**provenance, "shards": shards, "classes": classes})
                records, done_rows = [], 0
                progress = Progress(source.EXPECTED_ROWS[split], stage=f"finetune/pack-{split}", what="images")
                # At most the current shard and one future download exist in scratch.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = None
                    def fetch(name):
                        return hub_file(release, name, cache_dir=work, download=True, token=token, require_lfs=True)
                    for number, name in enumerate(shards):
                        receipt = directory / f"pack-{number:05d}.json"
                        if receipt.exists():
                            record = _write_pack(directory, number, (), identity)
                        else:
                            future = pending or pool.submit(fetch, name)
                            path, digest = future.result()
                            pending = None
                            if number + 1 < len(shards) and not (directory / f"pack-{number+1:05d}.json").exists():
                                pending = pool.submit(fetch, shards[number + 1])
                            require_disk(directory, path.stat().st_size * 2 + (512 << 20), what="next JPEG pack")
                            def rows():
                                index = done_rows
                                for images, labels in source._rows(path):
                                    if len(images) != len(labels):
                                        raise RuntimeError("ImageNet labels/images disagree")
                                    for entry, label in zip(images, labels):
                                        if not isinstance(label, int) or not 0 <= label < len(classes):
                                            raise RuntimeError("ImageNet class index out of range")
                                        filename = source._filename(entry, classes[label], index, split=split)
                                        yield entry["bytes"], label, filename
                                        index += 1
                            record = _write_pack(directory, number, rows(), identity)
                            source._drop_cached_copy(path, work)
                        records.append(record)
                        done_rows += record["rows"]
                        progress.advance(record["rows"])
                _publish(directory, records, classes, provenance, source.EXPECTED_ROWS[split])
            datasets[folder] = PackedImages(directory)
    if datasets["train"].class_to_idx != datasets["val"].class_to_idx:
        raise RuntimeError("packed ImageNet class mappings disagree")
    for split, dataset in datasets.items():
        if len(dataset.classes) != source.CLASSES or len(dataset) != source.EXPECTED_ROWS[
                "validation" if split == "val" else split]:
            raise RuntimeError("packed ImageNet count mismatch")
    index = np.load(destination / "val" / "index.npy", mmap_mode="r", allow_pickle=False)
    if not np.all(np.bincount(index["label"], minlength=source.CLASSES) ==
                  source.EXPECTED_ROWS["validation"] // source.CLASSES):
        raise RuntimeError("packed ImageNet validation class counts disagree")
    index._mmap.close()
    identity = {split: canonical_hash(dataset.manifest) for split, dataset in datasets.items()}
    return datasets, {"format": FORMAT, **identity}
