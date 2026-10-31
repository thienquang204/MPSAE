"""Bounded-memory NPZ access and integrity helpers for the repair research profile.

The split builder reads only the feature header and labels. The training cache
streams the feature NPY member once to disk and returns a read-only memory map.
No dataset download, model import, or feature extraction occurs here.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


PROTOCOL = "mpsae-repair-neighborhood-v1"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive_lock(path):
    """Hold a nonblocking OS lock, released automatically if a process exits.

    The small lock file is intentionally persistent: unlinking it allows a
    third process to lock a new inode while a second still holds the old one.
    Its first byte is the Windows locking region. PID metadata is cleared on
    normal release; after a crash it is informational, never an ownership test.
    File existence alone never means a writer is live.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
    stream = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            stream.write(b"\0")
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, PermissionError, OSError) as error:
            raise RuntimeError(f"Another process holds the writer lock: {path}") from error
        stream.seek(1)
        stream.write(f"pid={os.getpid()}\n".encode("ascii"))
        stream.truncate()
        yield
    finally:
        try:
            if acquired:
                try:
                    stream.truncate(1)
                finally:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            stream.close()


def _read_npy_header(stream):
    """Read only the NPY header using public NumPy APIs and the v3 format spec."""
    import numpy as np

    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    if version != (3, 0):
        raise ValueError(f"Unsupported NPY format version: {version}")

    # NumPy exposes no public v3 header reader. Its documented format uses a
    # four-byte little-endian length followed by a UTF-8 Python-literal dict.
    # Keep the same 10,000-byte header limit as the public v1/v2 readers.
    length_bytes = stream.read(4)
    if len(length_bytes) != 4:
        raise ValueError("Truncated NPY v3 header length")
    length = int.from_bytes(length_bytes, "little")
    if length > 10_000:
        raise ValueError("NPY v3 header exceeds the 10,000-byte limit")
    header_bytes = stream.read(length)
    if len(header_bytes) != length:
        raise ValueError("Truncated NPY v3 header")
    try:
        header = ast.literal_eval(header_bytes.decode("utf-8"))
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid NPY v3 header") from exc
    if not isinstance(header, dict) or set(header) != {"shape", "fortran_order", "descr"}:
        raise ValueError("NPY v3 header must contain shape, fortran_order, and descr")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("NPY v3 shape must be a tuple of nonnegative integers")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("NPY v3 fortran_order must be a boolean")
    try:
        dtype = np.lib.format.descr_to_dtype(header["descr"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid NPY v3 dtype descriptor") from exc
    return shape, header["fortran_order"], dtype


def inspect_npz(path, *, hash_archive=True):
    """Read data.npy's header, never its feature payload, and load numeric labels."""
    import numpy as np

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in ("data.npy", "label.npy"):
            if names.count(name) != 1:
                raise ValueError(f"Expected exactly one {name} member in {path}")
        with archive.open("data.npy") as stream:
            shape, fortran_order, dtype = _read_npy_header(stream)
        if len(shape) != 2 or dtype.hasobject or dtype.kind not in "fiu":
            raise ValueError("Expected a numeric, two-dimensional data array")
        with archive.open("label.npy") as stream:
            labels = np.load(stream, allow_pickle=False)
        if labels.shape not in ((shape[0],), (shape[0], 1)):
            raise ValueError("Labels must contain one value per feature row")
        labels = labels.reshape(-1)
        if labels.dtype.kind not in "fiu" or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("Labels must be finite integer class IDs")
        labels = labels.astype(np.int64, copy=False)
        metadata = {
            "path": str(path.resolve()), "archive_bytes": path.stat().st_size,
            "feature_shape": list(shape), "feature_dtype": dtype.str,
            "fortran_order": bool(fortran_order), "classes": int(np.unique(labels).size),
            "feature_npy_bytes": archive.getinfo("data.npy").file_size,
        }
    if hash_archive:
        metadata["sha256"] = sha256_file(path)
    return metadata, labels


def open_feature_cache(path, cache_dir, *, expected_sha256=None):
    """Return (read-only feature memmap, labels, source metadata).

    The archive hash is checked once per call. Call this once per process and
    reuse the returned mapping across arms. Compressed archives are streamed,
    not unpacked in RAM. Interrupted copies are not accepted as valid caches.
    """
    import numpy as np

    metadata, labels = inspect_npz(path)
    fingerprint = metadata["sha256"]
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("Training archive SHA-256 differs from the locked split source")
    directory = Path(cache_dir) / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    target, manifest_path = directory / "data.npy", directory / "cache.json"
    with exclusive_lock(directory / "cache.lock"):
        if target.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_sha256") != fingerprint or target.stat().st_size != metadata["feature_npy_bytes"]:
                raise ValueError("Existing feature-cache provenance or length differs; use a new cache directory")
            stat = target.stat()
            if stat.st_mtime_ns != manifest.get("cache_mtime_ns"):
                if sha256_file(target) != manifest.get("feature_npy_sha256"):
                    raise ValueError("Feature cache checksum mismatch")
        elif target.exists() or manifest_path.exists():
            raise ValueError("Incomplete feature cache: preserve it for inspection and use a new cache directory")
        else:
            fd, temporary = tempfile.mkstemp(prefix="data.", suffix=".tmp", dir=directory)
            digest = hashlib.sha256()
            try:
                with zipfile.ZipFile(path) as archive, archive.open("data.npy") as source, os.fdopen(fd, "wb") as output:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if Path(temporary).stat().st_size != metadata["feature_npy_bytes"]:
                    raise ValueError("Incomplete streamed NPY member")
                os.replace(temporary, target)
                atomic_json(manifest_path, {
                    "protocol": PROTOCOL, "source_sha256": fingerprint,
                    "feature_npy_sha256": digest.hexdigest(),
                    "cache_mtime_ns": target.stat().st_mtime_ns,
                    "feature_shape": metadata["feature_shape"],
                    "feature_dtype": metadata["feature_dtype"],
                })
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    features = np.load(target, mmap_mode="r", allow_pickle=False)
    if list(features.shape) != metadata["feature_shape"] or features.dtype.str != metadata["feature_dtype"]:
        raise ValueError("Feature cache header differs from source")
    return features, labels, metadata


def load_splits(directory, *, expected_source_sha256=None):
    import numpy as np

    directory = Path(directory)
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unrecognized split protocol")
    if expected_source_sha256 is not None and manifest["source"]["sha256"] != expected_source_sha256:
        raise ValueError("Split source fingerprint differs from the requested archive")
    path = directory / "split_indices.npz"
    if sha256_file(path) != manifest["indices_sha256"]:
        raise ValueError("Split index archive checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        indices = {name: arrays[name] for name in arrays.files}
    expected_names = {"fit_indices", "gallery_indices", "dev_cheap_indices", "dev_holdout_indices", "dev_full_indices", "anchor_indices"}
    if set(indices) != expected_names:
        raise ValueError("Split index names differ from the required protocol")
    rows = manifest["source"]["feature_shape"][0]
    for name, values in indices.items():
        if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) != manifest["counts"][name]:
            raise ValueError(f"Invalid split array {name}")
        if len(np.unique(values)) != len(values) or ((values < 0) | (values >= rows)).any():
            raise ValueError(f"Duplicate or out-of-range IDs in {name}")
    if not np.array_equal(indices["fit_indices"], indices["gallery_indices"]):
        raise ValueError("Pilot gallery must equal fitting rows in the same order")
    if not np.array_equal(np.sort(indices["dev_full_indices"]), np.sort(np.concatenate([indices["dev_cheap_indices"], indices["dev_holdout_indices"]]))):
        raise ValueError("Full development set is not the cheap/holdout union")
    distinct = np.concatenate([indices["fit_indices"], indices["dev_full_indices"], indices["anchor_indices"]])
    if len(np.unique(distinct)) != len(distinct):
        raise ValueError("Fitting, development, and landmark rows overlap")
    return indices, manifest
