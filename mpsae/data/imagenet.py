"""Materialise ImageNet-1K from Hugging Face into PyTorch-style folders.

Both downstream recipes read pixels, not parquet: the MRL recipe uses
``torchvision.datasets.ImageFolder`` and the CSRv2 recipe uses an FFCV file
written from the same folders. So this stage writes the *original* JPEG bytes
out of the pinned parquet shards -- never a re-encode, because a second JPEG
pass would change the pixels both published recipes were measured on.

Resumability is per shard: a shard is recorded only after all of its rows are
on disk, and an interrupted shard is simply rewritten.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
from typing import Any, Iterator

from ..embed.archive import hub_file, pinned_release
from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, require_disk
from ..logs import Progress, get_logger, stage

IMAGENET_REPO = "ILSVRC/imagenet-1k"
#: Named in every access error: the repository is gated per user.
TERMS_URL = "https://huggingface.co/datasets/ILSVRC/imagenet-1k"
#: ``validation`` is the HF split name, ``val`` the folder name MRL/FFCV expect.
SPLIT_FOLDER = {"train": "train", "validation": "val"}
EXPECTED_ROWS = {"train": 1_281_167, "validation": 50_000}
CLASSES = 1000
#: Rows held in memory at once while copying a shard (~30 MB of JPEG).
ROW_BATCH = 256
_SYNSET = re.compile(r"n[0-9]{8}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:JPEG|jpeg|jpg|JPG)")


def _work_dir(root: str | Path) -> Path:
    """Our own scratch inside ``root``: hub cache plus the per-split receipts."""
    return Path(root) / "_mpsae"


def _access_error(what: str, error: BaseException) -> RuntimeError:
    return RuntimeError(
        f"cannot {what} ({type(error).__name__}: {error}); accept the access conditions at "
        f"{TERMS_URL} with the account owning HF_TOKEN, then authenticate with HF_TOKEN or "
        "`hf auth login`. No substitute dataset is used.")


def _synsets(release: dict[str, Any], *, cache_dir: Path, download: bool,
             token: str | None) -> list[str]:
    """The 1000 synset ids in label-index order, from the pinned revision.

    The parquet label column is a plain integer and the ClassLabel names are
    the human-readable descriptions, so the synsets come from ``classes.py``,
    whose ``IMAGENET2012_CLASSES`` is keyed by synset in label order. The file
    is parsed as a literal, never executed.
    """
    try:
        path, _ = hub_file(release, "classes.py", cache_dir=cache_dir, download=download,
                           token=token)
    except RuntimeError as error:
        raise RuntimeError(
            f"{IMAGENET_REPO}@{release['revision']} has no readable classes.py ({error}); "
            "the synset order cannot be derived from the parquet labels alone") from error
    mapping: dict[str, str] | None = None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "IMAGENET2012_CLASSES"
                   for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:  # OrderedDict({...})
            value = value.args[0]
        mapping = ast.literal_eval(value)
        break
    if not isinstance(mapping, dict):
        raise RuntimeError(f"{path} does not define IMAGENET2012_CLASSES as a dict literal")
    synsets = list(mapping)
    if len(synsets) != CLASSES or not all(_SYNSET.fullmatch(name) for name in synsets):
        raise RuntimeError(f"{path} lists {len(synsets)} keys; expected {CLASSES} synset ids")
    if synsets != sorted(synsets):
        # ImageFolder sorts its class directories, so the parquet label index
        # must be the sorted-synset index or the two label spaces disagree.
        raise RuntimeError(f"{path} lists synsets out of sorted order; the parquet label index "
                           "would not match ImageFolder's class order")
    return synsets


def _shards(release: dict[str, Any], split: str) -> list[str]:
    """Parquet files of one split, in their published order."""
    found = sorted(name for name in release["files"]
                   if name.endswith(".parquet") and re.match(rf"{split}[-.]", Path(name).name))
    if not found:
        raise RuntimeError(f"{IMAGENET_REPO}@{release['revision']} has no {split} parquet shards; "
                           f"files seen: {sorted(release['files'])[:8]}")
    return found


def _rows(path: Path) -> Iterator[tuple[list[dict[str, Any]], list[int]]]:
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as handle:
        for batch in handle.iter_batches(batch_size=ROW_BATCH, columns=["image", "label"]):
            yield batch.column("image").to_pylist(), batch.column("label").to_pylist()


def _filename(entry: dict[str, Any], synset: str, index: int, *, split: str) -> str:
    """Original file name when the shard carries one, else a stable synthetic one."""
    raw = Path(str(entry.get("path") or "")).name
    if not raw:
        return f"{synset}_{index:08d}.JPEG"
    if not _SAFE_NAME.fullmatch(raw):
        raise RuntimeError(f"{split} row {index} has an unusable image path {raw!r}")
    if split == "train" and (prefix := raw.split("_", 1)[0]) and _SYNSET.fullmatch(prefix):
        # Train file names begin with their own synset: a free, exact check that
        # our label -> synset mapping is the one the archive was built with.
        if prefix != synset:
            raise RuntimeError(f"{split} row {index}: file {raw} contradicts label synset {synset}")
    return raw


def _copy_shard(path: Path, directory: Path, synsets: list[str], *, split: str,
                start_index: int, counts: list[int], progress: Progress) -> int:
    """Write every row of one shard. Returns the number of rows written."""
    written = 0
    for entries, labels in _rows(path):
        if len(entries) != len(labels):
            raise RuntimeError(f"{path} has {len(entries)} images for {len(labels)} labels")
        for entry, label in zip(entries, labels):
            if not 0 <= int(label) < CLASSES:
                raise RuntimeError(f"{path} row {start_index + written}: label {label} "
                                   f"outside 0..{CLASSES - 1}")
            payload = entry.get("bytes")
            if not payload:
                raise RuntimeError(f"{path} row {start_index + written} carries no image bytes")
            synset = synsets[int(label)]
            name = _filename(entry, synset, start_index + written, split=split)
            target = directory / synset / name
            # Rows of an unfinished shard are rewritten unconditionally: only a
            # recorded shard is trusted, so a half-written file is never reused.
            temporary = target.with_name(target.name + ".partial")
            with temporary.open("wb") as handle:
                handle.write(payload)
            os.replace(temporary, target)
            counts[int(label)] += 1
            written += 1
            progress.advance()
    return written


def _materialise_split(split: str, root: Path, release: dict[str, Any], synsets: list[str], *,
                       download: bool, token: str | None) -> dict[str, Any]:
    """Copy every shard of one split into ``<root>/<folder>/<synset>/``."""
    log = get_logger(f"imagenet:{split}")
    directory = Path(root) / SPLIT_FOLDER[split]
    work = _work_dir(root)
    receipt_path = work / f"prepare_{split}.json"
    shards = _shards(release, split)
    with exclusive_lock(work / f"prepare_{split}.lock"):
        receipt = read_json(receipt_path) or {}
        identity = {"repository": IMAGENET_REPO, "revision": release["revision"], "split": split,
                    "shards": shards, "classes_sha256": canonical_hash(synsets)}
        if receipt and receipt.get("identity") != canonical_hash(identity):
            raise RuntimeError(f"{receipt_path} was written for a different ImageNet release; "
                               "remove the folder and the receipt to start over")
        done: dict[str, Any] = dict(receipt.get("done", {}))
        counts = list(receipt.get("counts") or [0] * CLASSES)
        if len(counts) != CLASSES:
            raise RuntimeError(f"{receipt_path} carries {len(counts)} class counters")
        for synset in synsets:
            (directory / synset).mkdir(parents=True, exist_ok=True)
        pending = [name for name in shards if name not in done]
        if pending:
            remaining = sum(int(release["files"][name].get("size") or 0) for name in pending)
            # JPEG bytes land next to the parquet copy; each parquet shard is
            # deleted once written, so the peak is one shard plus the output.
            require_disk(directory, int(remaining * 1.1), what=f"ImageNet {split} images")
        rows_done = sum(int(entry["rows"]) for entry in done.values())
        progress = Progress(EXPECTED_ROWS[split], stage=f"imagenet:{split}", what="images",
                            done=rows_done)
        for name in shards:
            if name in done:
                continue
            try:
                path, digest = hub_file(release, name, cache_dir=work, download=download,
                                        token=token, require_lfs=True)
            except RuntimeError as error:
                raise _access_error(f"download {name}", error) from error
            written = _copy_shard(path, directory, synsets, split=split, start_index=rows_done,
                                  counts=counts, progress=progress)
            rows_done += written
            # One barrier per shard instead of one fsync per image: the receipt
            # below is what makes a shard trusted, so the data must be on the
            # platter before it is recorded.
            os.sync()
            done[name] = {"rows": written, "sha256": digest}
            atomic_json(receipt_path, {"identity": canonical_hash(identity), **identity,
                                       "done": done, "counts": counts, "rows": rows_done})
            _drop_cached_copy(path, work)
            log.info("shard %s: %s rows (%s/%s shards)", Path(name).name, f"{written:,}",
                     len(done), len(shards))
    observed = _count_images(directory)
    present = {synset: counts[index] for index, synset in enumerate(synsets) if counts[index]}
    if rows_done != EXPECTED_ROWS[split]:
        raise RuntimeError(f"ImageNet {split}: wrote {rows_done:,} rows, expected "
                           f"{EXPECTED_ROWS[split]:,}")
    if len(present) != CLASSES:
        raise RuntimeError(f"ImageNet {split}: {len(present)} classes received images, "
                           f"expected {CLASSES}")
    if observed["images"] != rows_done or observed["classes"] != CLASSES:
        raise RuntimeError(f"{directory} holds {observed['images']:,} images in "
                           f"{observed['classes']} class folders, receipt says {rows_done:,} "
                           f"in {CLASSES}")
    if split == "validation" and set(counts) != {EXPECTED_ROWS[split] // CLASSES}:
        raise RuntimeError("ImageNet validation is not 50 images per class; the label mapping "
                           "or the shard set is wrong")
    log.info("%s ready: %s images, %s classes", directory, f"{rows_done:,}", CLASSES)
    return {"folder": str(directory), "rows": rows_done, "classes": CLASSES,
            "shards": len(done), "per_class_min": min(counts), "per_class_max": max(counts)}


def _drop_cached_copy(path: Path, work: Path) -> None:
    """Delete our cached parquet copy; it is re-downloadable from the pinned commit."""
    hub = (work / "hub").resolve()
    try:
        blob = path.resolve()
        if hub in blob.parents:
            blob.unlink(missing_ok=True)
        if path.is_symlink():
            path.unlink(missing_ok=True)
    except OSError as error:
        get_logger("imagenet").warning("could not drop the cached shard %s (%s)", path, error)


def _count_images(directory: Path) -> dict[str, int]:
    """Cheap verification pass: names only, no ``stat`` per file."""
    classes = images = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_dir() or not _SYNSET.fullmatch(entry.name):
                continue
            classes += 1
            with os.scandir(entry.path) as files:
                images += sum(1 for item in files if item.is_file()
                              and not item.name.endswith(".partial"))
    return {"classes": classes, "images": images}


def prepare_imagenet(root: str | Path, *, token: str | None = None, revision: str = "main",
                     download: bool = True, splits: tuple[str, ...] = ("train", "validation"),
                     state_dir: str | Path | None = None) -> dict[str, Any]:
    """Download ImageNet-1K and write ``<root>/{train,val}/n########/*.JPEG``.

    Returns the provenance every later stage records: the immutable dataset
    commit, the per-split row counts and the digest of the synset order that
    turns a parquet label into a class folder.
    """
    root = Path(root)
    work = _work_dir(root)
    work.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(splits) - set(SPLIT_FOLDER))
    if unknown or not splits:
        raise ValueError(f"splits must be a nonempty subset of {sorted(SPLIT_FOLDER)}, got {splits}")
    try:
        release = pinned_release(IMAGENET_REPO, repo_type="dataset", revision=revision,
                                 cache_dir=work, download=download, token=token)
    except RuntimeError as error:
        raise _access_error(f"resolve {IMAGENET_REPO}@{revision}", error) from error
    synsets = _synsets(release, cache_dir=work, download=download, token=token)
    identity = {"repository": IMAGENET_REPO, "revision": release["revision"],
                "root": str(root.resolve()), "splits": list(splits),
                "classes_sha256": canonical_hash(synsets)}

    def run() -> dict[str, Any]:
        return {"repository": IMAGENET_REPO, "revision": release["revision"],
                "requested_revision": revision, "root": str(root.resolve()),
                "classes": CLASSES, "classes_sha256": canonical_hash(synsets),
                "first_class": synsets[0], "last_class": synsets[-1],
                "splits": {split: _materialise_split(split, root, release, synsets,
                                                     download=download, token=token)
                           for split in splits}}

    if state_dir is None:
        return run()
    with stage("imagenet", state_dir, identity=identity,
               description=f"materialising {IMAGENET_REPO}@{release['revision'][:12]} -> {root}"
               ) as detail:
        if not (detail.get("skipped") and isinstance(detail.get("provenance"), dict)):
            detail["provenance"] = run()
        provenance = detail["provenance"]
    return provenance


def imagenet_folder_stats(directory: str | Path) -> dict[str, Any]:
    """Count images per synset and fingerprint the folder.

    The fingerprint (relative path plus size of every file, in sorted order) is
    what ties an embedding archive to the exact pixels it was extracted from;
    mtimes are deliberately excluded so copying the dataset does not invalidate
    an archive.
    """
    import hashlib

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not an ImageNet split folder")
    synsets = sorted(entry.name for entry in os.scandir(directory)
                     if entry.is_dir() and _SYNSET.fullmatch(entry.name))
    if not synsets:
        raise ValueError(f"{directory} contains no n######## class folders")
    digest = hashlib.sha256()
    per_class: dict[str, int] = {}
    total = 0
    bytes_total = 0
    progress = Progress(len(synsets), stage="imagenet:stats", what="classes")
    for synset in synsets:
        names = sorted(entry.name for entry in os.scandir(directory / synset)
                       if entry.is_file() and not entry.name.endswith(".partial"))
        per_class[synset] = len(names)
        for name in names:
            size = (directory / synset / name).stat().st_size
            digest.update(f"{synset}/{name}\0{size}\n".encode())
            bytes_total += size
        total += len(names)
        progress.advance()
    return {"root": str(directory.resolve()), "classes": len(synsets), "images": total,
            "bytes": bytes_total, "per_class": per_class,
            "per_class_min": min(per_class.values()), "per_class_max": max(per_class.values()),
            "fingerprint": digest.hexdigest(),
            "fingerprint_kind": "sorted_relative_paths_and_sizes"}
