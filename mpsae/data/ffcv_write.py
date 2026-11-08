"""Write the FFCV dataset the CSRv2 extraction recipe reads.

A direct port of ``CSRv2/image/dataset_preparation/write_imagenet.py`` with the
parameters its README pins (``max_resolution=500``, ``write_mode=proportion``,
``compress_probability=0.50``, ``jpeg_quality=90``). The file name encodes those
parameters exactly as the released pipeline names them, because the stored
resolution and JPEG quality are part of the recipe, not an implementation
detail: they decide what pixels the frozen backbone ever sees.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from ..io import atomic_json, canonical_hash, exclusive_lock, read_json, require_disk
from ..logs import get_logger

#: ``validation`` is the HF split name; the folder and the FFCV file say ``val``.
SPLIT_FOLDER = {"train": "train", "val": "val", "validation": "val"}
#: ``from_indexed_dataset`` chunk size in the released writer.
CHUNK_SIZE = 100
_SYNSET = re.compile(r"n[0-9]{8}")

FFCV_MISSING = ("ffcv is not importable. The CSRv2 recipe needs it to write and read the "
                "dataset; install the pinned environment with setup_server.sh "
                "(conda: opencv, libjpeg-turbo, pkg-config, then ffcv).")


def _image_root(image_root: str | Path, folder: str) -> Path:
    """Accept either the ImageNet root or one split folder, and say which it used."""
    image_root = Path(image_root)
    candidate = image_root / folder
    if candidate.is_dir():
        return candidate
    if not image_root.is_dir():
        raise FileNotFoundError(f"{image_root} does not exist; run the ImageNet stage first")
    if any(entry.is_dir() and _SYNSET.fullmatch(entry.name) for entry in os.scandir(image_root)):
        return image_root
    raise FileNotFoundError(
        f"{image_root} is neither an ImageNet root containing {folder}/ nor a split folder of "
        "n######## class directories")


def write_ffcv(image_root: str | Path, out_dir: str | Path, split: str, *,
               max_resolution: int = 500, compress_probability: float = 0.50,
               jpeg_quality: int = 90, num_workers: int | None = None) -> Path:
    """Write ``<out_dir>/<split>_500_0.50_90.ffcv``. Returns the output path."""
    if split not in SPLIT_FOLDER:
        raise ValueError(f"split must be one of {sorted(SPLIT_FOLDER)}, got {split!r}")
    if max_resolution < 1 or not 0.0 <= compress_probability <= 1.0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("degenerate FFCV write parameters")
    folder = SPLIT_FOLDER[split]
    directory = _image_root(image_root, folder)
    out_dir = Path(out_dir)
    out_path = out_dir / f"{folder}_{max_resolution}_{compress_probability:.2f}_{jpeg_quality}.ffcv"
    manifest_path = out_path.with_name(out_path.name + ".json")
    workers = int(num_workers) if num_workers else (os.cpu_count() or 16)
    log = get_logger("ffcv")

    from .imagenet import imagenet_folder_stats

    source = imagenet_folder_stats(directory)
    identity = {"writer": "csrv2-write_imagenet-port-v1", "split": folder,
                "source": {key: source[key] for key in ("classes", "images", "fingerprint")},
                "max_resolution": max_resolution, "compress_probability": compress_probability,
                "jpeg_quality": jpeg_quality, "write_mode": "proportion",
                "chunk_size": CHUNK_SIZE}
    signature = canonical_hash(identity)

    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out_path.with_name(out_path.name + ".lock")):
        manifest = read_json(manifest_path)
        if manifest and out_path.is_file() and manifest.get("identity") == signature:
            # Multi-GB file: identity plus length and mtime, not a rehash on
            # every pipeline run.
            status = out_path.stat()
            if (manifest.get("bytes") == status.st_size
                    and manifest.get("mtime_ns") == status.st_mtime_ns):
                log.info("skip %s (already written, %s images)", out_path.name,
                         f"{source['images']:,}")
                return out_path
            raise RuntimeError(f"{out_path} changed since its manifest was written "
                               f"({status.st_size:,} bytes vs {manifest.get('bytes'):,}); "
                               "delete both and write it again")
        try:
            from ffcv.fields import IntField, RGBImageField
            from ffcv.writer import DatasetWriter
        except ImportError as error:
            raise RuntimeError(FFCV_MISSING) from error
        import ffcv
        from PIL import ImageFile
        from torchvision.datasets import ImageFolder

        # A handful of ImageNet JPEGs are truncated; the released writer tolerates
        # them, and refusing them here would silently change the dataset size.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        dataset = ImageFolder(root=str(directory))
        if len(dataset.classes) != source["classes"] or len(dataset) != source["images"]:
            raise RuntimeError(f"{directory}: ImageFolder sees {len(dataset):,} images in "
                               f"{len(dataset.classes)} classes, the folder scan saw "
                               f"{source['images']:,} in {source['classes']}")
        # ~0.1 MB per stored image at these settings, with headroom.
        require_disk(out_dir, int(source["images"]) * (160 << 10), what=out_path.name)
        partial = out_path.with_name(out_path.name + ".partial")
        partial.unlink(missing_ok=True)
        log.info("writing %s: %s images from %s with %s workers", out_path.name,
                 f"{len(dataset):,}", directory, workers)
        writer = DatasetWriter(str(partial), {
            "image": RGBImageField(write_mode="proportion",
                                   max_resolution=max_resolution,
                                   compress_probability=compress_probability,
                                   jpeg_quality=jpeg_quality),
            "label": IntField(),
        }, num_workers=workers)
        writer.from_indexed_dataset(dataset, chunksize=CHUNK_SIZE)
        with partial.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, out_path)
        status = out_path.stat()
        payload: dict[str, Any] = {**identity, "identity": signature, "path": str(out_path),
                                   "image_root": str(directory.resolve()),
                                   "source_fingerprint_kind": source["fingerprint_kind"],
                                   "images": source["images"], "classes": source["classes"],
                                   "num_workers": workers, "bytes": status.st_size,
                                   "mtime_ns": status.st_mtime_ns,
                                   "ffcv": getattr(ffcv, "__version__", None)}
        atomic_json(manifest_path, payload)
        log.info("wrote %s (%s bytes)", out_path.name, f"{status.st_size:,}")
    return out_path
