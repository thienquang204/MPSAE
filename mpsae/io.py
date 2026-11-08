"""Durable, resumable file primitives shared by every stage.

Ported from the previous ``repair_data``/``repair_runtime`` helpers. No torch
import at module scope so planning and reporting work without CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
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
    """Read committed lines, tolerating only an interrupted trailing write.

    A killed training process can leave part of its last JSON record behind.
    Interior corruption or a malformed newline-terminated record remains an
    error rather than silently removing evidence from the experiment log.
    """
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                    break
                raise
    return records


def truncate_jsonl(path: str | Path, keep: int) -> int:
    """Drop records beyond ``keep`` so a resumed log never double-counts."""
    if keep < 0:
        raise ValueError("keep must be nonnegative")
    path = Path(path)
    records = read_jsonl(path)
    # Repair an interrupted final write even if every complete record is kept.
    # A valid last JSON value without a newline also needs its separator before
    # MetricWriter can append another record.
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size:
            handle.seek(-1, os.SEEK_END)
        terminated = not size or handle.read(1) == b"\n"
    if len(records) <= keep and terminated:
        return len(records)
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                         prefix=path.name + ".", suffix=".tmp",
                                         encoding="utf-8")
    try:
        with handle:
            for record in records[:keep]:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return min(len(records), keep)


# ------------------------------------------------------------------- locks

@contextmanager
def exclusive_lock(path: str | Path, *, timeout: float | None = None) -> Iterator[None]:
    """Advisory lock so concurrent stages never write the same artefact."""
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    handle = open(path, "a+b")
    acquired = False
    if os.name == "nt":
        import msvcrt

        # Windows byte-range locks need an actual byte and an explicit offset.
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
    else:
        import fcntl
    try:
        while True:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
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
            if acquired:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
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
    import zipfile

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
        dtype = np.dtype(header["data"]["dtype"])
        if len(shape) != 2 or shape[1] == 0 or dtype.kind not in "fiu":
            raise ValueError(f"{path}: embeddings must be a two-dimensional real numeric array")
        if receipt is None or not features_npy.is_file() or receipt.get("shape") != list(shape):
            target.mkdir(parents=True, exist_ok=True)
            convert = dtype != np.dtype(np.float32) or header["data"]["fortran_order"]
            payload_bytes = int(np.prod(shape))
            needed = payload_bytes * (4 + dtype.itemsize if convert else 4)
            require_disk(target, needed, what=f"{path.name} feature cache")
            partial = features_npy.with_suffix(".partial")
            source_path = features_npy.with_suffix(".source.partial") if convert else partial
            source = written = None
            try:
                # np.load(archive)['data'] materialises the complete embedding
                # matrix (10+ GB for ImageNet). Stream its NPY member instead;
                # ZIP_STORED and compressed NPZ archives both work this way.
                with zipfile.ZipFile(path) as archive, archive.open("data.npy") as incoming:
                    with source_path.open("wb") as output:
                        shutil.copyfileobj(incoming, output, length=CHUNK)
                source = np.load(source_path, mmap_mode="r", allow_pickle=False)
                if source.shape != shape or source.dtype != dtype:
                    raise ValueError(f"{path}: extracted embedding header disagrees with archive")
                if convert:
                    written = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
                step = max(1, (64 << 20) // max(1, shape[1] * max(4, dtype.itemsize)))
                for start in range(0, shape[0], step):
                    chunk = source[start:start + step]
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"{path}: refusing nonfinite embeddings")
                    if written is not None:
                        written[start:start + step] = chunk
                        if not np.isfinite(written[start:start + step]).all():
                            raise ValueError(f"{path}: embeddings overflow float32 storage")
                if written is not None:
                    written.flush()
            finally:
                for array in (written, source):
                    if array is not None:
                        array._mmap.close()
                if convert:
                    source_path.unlink(missing_ok=True)
            with partial.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(partial, features_npy)
            receipt = {"source": str(path), "sha256": digest, "shape": list(shape),
                       "dtype": "float32", "label_key": label_key}
            atomic_json(receipt_path, receipt)
    with np.load(path, allow_pickle=False) as archive:
        raw_labels = np.asarray(archive[receipt["label_key"]]).reshape(-1)
        if raw_labels.dtype.kind not in "iu":
            raise ValueError(f"{path}: labels must have an integer dtype")
        if raw_labels.dtype.kind == "u" and raw_labels.size and raw_labels.max() > np.iinfo(np.int64).max:
            raise ValueError(f"{path}: labels exceed int64 range")
        labels = raw_labels.astype(np.int64)
    if len(labels) != shape[0]:
        raise ValueError(f"{path}: {shape[0]:,} rows but {len(labels):,} labels")
    data = np.load(features_npy, mmap_mode="r", allow_pickle=False)
    if data.shape != shape or data.dtype != np.dtype(np.float32) or not data.flags.c_contiguous:
        data._mmap.close()
        raise ValueError(f"{features_npy}: invalid cached embedding shape, dtype or layout; remove this cache file and retry")
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
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("embeddings must be a two-dimensional array with nonzero width")
    if len(data) != len(labels):
        raise ValueError(f"{len(data):,} rows but {len(labels):,} labels")
    step = max(1, (64 << 20) // max(1, data.shape[1] * data.dtype.itemsize))
    for start in range(0, len(data), step):
        if not np.isfinite(data[start:start + step]).all():
            raise ValueError("refusing to write nonfinite embeddings")
    if extra and set(extra) & {"data", "labels"}:
        raise ValueError("extra metadata cannot replace data or labels")
    partial = path.with_name(path.name + ".partial")
    payload = {"data": data, "labels": labels, **(extra or {})}
    with partial.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return sha256_file(path)
