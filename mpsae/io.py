"""Durable, resumable file primitives shared by every stage.

Ported from the previous ``repair_data``/``repair_runtime`` helpers. No torch
import at module scope so planning and reporting work without CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

CHUNK = 8 << 20


# ------------------------------------------------------------------ hashing

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Stable hash of a JSON-serialisable structure (sorted keys, no spaces)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_hash(values) -> str:
    """Hash a numpy array by dtype, shape and contiguous bytes."""
    import numpy as np

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


# -------------------------------------------------------------------- json

def atomic_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON through a temporary file, fsync, then rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                         prefix=path.name + ".", suffix=".tmp",
                                         encoding="utf-8")
    try:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def truncate_jsonl(path: str | Path, keep: int) -> int:
    """Drop records beyond ``keep`` so a resumed log never double-counts."""
    path = Path(path)
    records = read_jsonl(path)
    if len(records) <= keep:
        return len(records)
    with path.open("w", encoding="utf-8") as handle:
        for record in records[:keep]:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return keep


# ------------------------------------------------------------------- locks

@contextmanager
def exclusive_lock(path: str | Path, *, timeout: float | None = None) -> Iterator[None]:
    """Advisory lock so concurrent stages never write the same artefact."""
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    handle = open(path, "a+")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {path} within {timeout}s") from error
                time.sleep(0.25)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


# -------------------------------------------------------------------- disk

def require_disk(directory: str | Path, needed_bytes: int, *,
                 reserve_bytes: int = 512 << 20, what: str = "output") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed_bytes + reserve_bytes:
        raise RuntimeError(
            f"insufficient disk for {what} at {directory}: need {needed_bytes:,} bytes "
            f"plus {reserve_bytes:,} reserve, {free:,} free")


def human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(count) < 1024 or unit == "TiB":
            return f"{count:,.1f} {unit}" if unit != "B" else f"{int(count):,} B"
        count /= 1024
    return f"{count:.1f} TiB"


# ---------------------------------------------------------------- memmap npz

def inspect_npz(path: str | Path) -> dict[str, Any]:
    """Read shapes/dtypes of an uncompressed NPZ without loading the payload."""
    import zipfile

    import numpy as np

    info: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                shape, fortran, dtype = _read_npy_header(handle)
            info[member[:-4]] = {"shape": tuple(shape), "dtype": dtype.str,
                                 "fortran_order": bool(fortran)}
    return info


def _read_npy_header(handle) -> tuple[tuple[int, ...], bool, Any]:
    """Shape, order and dtype of a ``.npy`` stream, reading only its header.

    Dispatches on the format version through the documented
    ``read_array_header_*`` entry points. The private ``_read_array_header``
    these used to share was removed in NumPy 2.
    """
    import numpy as np

    major, minor = np.lib.format.read_magic(handle)
    reader = getattr(np.lib.format, f"read_array_header_{major}_{minor}", None)
    if reader is None:
        raise ValueError(f"unsupported .npy format version {major}.{minor}")
    return reader(handle)


def open_npz_memmap(path: str | Path, cache_dir: str | Path, *,
                    expected_sha256: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(data, labels, metadata)`` with ``data`` memory-mapped.

    An uncompressed NPZ member cannot be mmapped directly, so the payload is
    extracted once into ``cache_dir`` as a plain ``.npy`` and reused. The
    extraction is keyed by the archive digest, guarded by a lock, and only
    published after a successful fsync, so an interrupted run never leaves a
    half-written cache behind.
    """
    import numpy as np

    path, cache_dir = Path(path), Path(cache_dir)
    digest = expected_sha256 or sha256_file(path)
    if expected_sha256 is not None:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(f"{path} digest {observed} differs from the expected {expected_sha256}")
    target = cache_dir / digest
    features_npy, receipt_path = target / "data.npy", target / "cache.json"
    with exclusive_lock(target.with_suffix(".lock")):
        receipt = read_json(receipt_path)
        header = inspect_npz(path)
        if "data" not in header:
            raise ValueError(f"{path} has no 'data' member; keys: {sorted(header)}")
        label_key = "labels" if "labels" in header else ("label" if "label" in header else None)
        if label_key is None:
            raise ValueError(f"{path} has no 'labels' member; keys: {sorted(header)}")
        shape = header["data"]["shape"]
        if receipt is None or not features_npy.is_file() or receipt.get("shape") != list(shape):
            target.mkdir(parents=True, exist_ok=True)
            require_disk(target, int(np.prod(shape)) * 4, what=f"{path.name} feature cache")
            partial = features_npy.with_suffix(".partial")
            with np.load(path, mmap_mode=None, allow_pickle=False) as archive:
                source = archive["data"]
                written = np.lib.format.open_memmap(partial, mode="w+",
                                                    dtype=np.float32, shape=shape)
                step = max(1, (64 << 20) // max(1, shape[1] * 4))
                for start in range(0, shape[0], step):
                    written[start:start + step] = source[start:start + step]
                written.flush()
                del written
            with partial.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(partial, features_npy)
            receipt = {"source": str(path), "sha256": digest, "shape": list(shape),
                       "dtype": "float32", "label_key": label_key}
            atomic_json(receipt_path, receipt)
    data = np.load(features_npy, mmap_mode="r", allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive[receipt["label_key"]]).reshape(-1).astype(np.int64)
    if len(labels) != len(data):
        raise ValueError(f"{path}: {len(data):,} rows but {len(labels):,} labels")
    return data, labels, {"rows": len(data), "dim": int(data.shape[1]),
                          "sha256": digest, "label_key": receipt["label_key"],
                          "cache": str(features_npy)}


def save_npz(path: str | Path, *, data, labels, extra: dict[str, Any] | None = None) -> str:
    """Write ``{data, labels}`` uncompressed, atomically, and return its digest."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64).reshape(-1)
    if len(data) != len(labels):
        raise ValueError(f"{len(data):,} rows but {len(labels):,} labels")
    if not np.isfinite(data).all():
        raise ValueError("refusing to write nonfinite embeddings")
    partial = path.with_name(path.name + ".partial")
    payload = {"data": data, "labels": labels, **(extra or {})}
    with partial.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return sha256_file(path)
