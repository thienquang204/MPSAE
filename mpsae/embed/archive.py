"""Shared output layer for every embedding recipe.

Both recipes (MRL ``pytorch_inference`` and CSRv2 ``pretrained_embeddings``)
differ only in how a feature row is produced. Everything after that -- the
resumable cache, the row validation, the NPZ archive and the manifest -- lives
here so the two archives are byte-comparable in structure and provenance.

Release pinning helpers live here until ``mpsae/hub.py`` exists; ``data``
imports them so the dataset and the weights are pinned the same way.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

from ..io import (array_hash, atomic_json, canonical_hash, exclusive_lock, read_json,
                  require_disk, save_npz, sha256_file)
from ..logs import get_logger

#: The released CSRv2 trainer reads ``train_data.get('labels')`` while every
#: released producer writes ``label``, so the label-based contrastive loss sees
#: ``None`` and silently degenerates. We write ``labels``; the manifest records
#: the rename so nobody mistakes it for a difference in the algorithm.
LABEL_KEY_DEVIATION = ("label key renamed label->labels so CSRv2 label_based_cl receives real "
                       "labels (released loader/producer key mismatch)")

#: Precision contract of both recipes: compute under fp16 autocast like the
#: released scripts, keep the stored copy float32 like their ``np.asarray``.
PRECISION = "autocast_fp16_stored_float32"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


# ------------------------------------------------------------ hub provenance

def pinned_release(repo: str, *, repo_type: str = "model", revision: str = "main",
                   cache_dir: str | Path, download: bool = True,
                   token: str | None = None) -> dict[str, Any]:
    """Resolve ``repo@revision`` to an immutable commit once and cache it.

    A branch name moves; an extraction that cannot name the exact commit it read
    is not reproducible. The receipt also carries the Hub's per-file LFS digests
    so a later download can be verified without trusting the transport.
    """
    root = Path(cache_dir) / "releases"
    identity = {"repository": repo, "repo_type": repo_type, "requested_revision": revision}
    receipt = root / (canonical_hash(identity) + ".json")
    with exclusive_lock(receipt.with_suffix(".lock")):
        release = read_json(receipt)
        if release is not None:
            if any(release.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f"release receipt {receipt} describes a different repository")
        else:
            if not download:
                raise RuntimeError(
                    f"no pinned release receipt for {repo}@{revision} at {receipt}; "
                    "run once with download enabled to pin it")
            from huggingface_hub import HfApi

            try:
                info = HfApi().repo_info(repo, repo_type=repo_type, revision=revision,
                                         files_metadata=True, token=_token(token))
            except Exception as error:
                raise RuntimeError(
                    f"cannot resolve {repo_type} {repo}@{revision} ({type(error).__name__}: {error}); "
                    "check repository access and authenticate with HF_TOKEN or `hf auth login`"
                ) from error
            files: dict[str, dict[str, Any]] = {}
            for item in info.siblings or []:
                lfs = getattr(item, "lfs", None)
                digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                files[item.rfilename] = {"sha256": digest, "size": getattr(item, "size", None)}
            release = {**identity, "revision": str(info.sha), "files": files}
            if not _COMMIT.fullmatch(release["revision"]):
                raise RuntimeError(f"the Hub returned no immutable commit for {repo}@{revision}")
            atomic_json(receipt, release)
    if not _COMMIT.fullmatch(str(release.get("revision", ""))):
        raise RuntimeError(f"invalid pinned revision in {receipt}")
    return release


def hub_file(release: dict[str, Any], filename: str, *, cache_dir: str | Path,
             download: bool = True, token: str | None = None,
             require_lfs: bool = False) -> tuple[Path, str]:
    """Download one file of a pinned release and verify it. Returns path+sha256."""
    from huggingface_hub import hf_hub_download

    entry = release["files"].get(filename)
    if entry is None:
        raise RuntimeError(f"{filename} is absent from {release['repository']}@{release['revision']}")
    expected = entry.get("sha256")
    if require_lfs and not _SHA256.fullmatch(str(expected)):
        raise RuntimeError(f"{filename} has no Hub LFS sha256; refusing to use unverifiable weights")
    try:
        path = Path(hf_hub_download(release["repository"], filename,
                                    repo_type=release["repo_type"], revision=release["revision"],
                                    cache_dir=str(Path(cache_dir) / "hub"),
                                    local_files_only=not download, token=_token(token)))
    except Exception as error:
        raise RuntimeError(
            f"cannot fetch {filename} from {release['repository']}@{release['revision']} "
            f"({type(error).__name__}: {error})") from error
    if entry.get("size") is not None and path.stat().st_size != entry["size"]:
        raise RuntimeError(f"{path} is {path.stat().st_size:,} bytes, release says {entry['size']:,}")
    digest = sha256_file(path)
    if _SHA256.fullmatch(str(expected)) and digest != expected:
        raise RuntimeError(f"{path} hashes to {digest}, release says {expected}")
    return path, digest


def _token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or None


def library_versions() -> dict[str, str | None]:
    """Versions that can change a feature value, for the manifest."""
    versions: dict[str, str | None] = {}
    for module in ("numpy", "torch", "torchvision", "timm", "ffcv", "PIL", "huggingface_hub",
                   "pyarrow"):
        try:
            versions[module] = str(__import__(module).__version__)
        except Exception:
            versions[module] = None
    return versions


# --------------------------------------------------------------- validation

def validate_features(values, labels, *, dim: int, row_offset: int) -> tuple[Any, Any]:
    """Check one batch and return it as ``(float32[n,dim], int64[n])``.

    A nonfinite or zero-norm row is fatal rather than clipped: every downstream
    metric L2-normalises the embedding, which is undefined at zero, and a NaN
    would propagate into the SAE weights without a trace.
    """
    import numpy as np

    features = np.ascontiguousarray(values, dtype=np.float32)
    targets = np.ascontiguousarray(labels).reshape(-1)
    if features.ndim != 2 or features.shape[1] != dim:
        raise RuntimeError(f"rows {row_offset}..{row_offset + len(features)}: expected "
                           f"(n, {dim}) features, got {tuple(features.shape)}")
    if len(targets) != len(features):
        raise RuntimeError(f"rows {row_offset}..{row_offset + len(features)}: "
                           f"{len(features)} features but {len(targets)} labels")
    if not np.isfinite(features).all():
        bad = np.flatnonzero(~np.isfinite(features).all(axis=1))[:8] + row_offset
        raise RuntimeError(f"nonfinite feature rows at {bad.tolist()} (first few)")
    norms = np.square(features.astype(np.float64)).sum(axis=1)
    if (norms == 0).any():
        bad = np.flatnonzero(norms == 0)[:8] + row_offset
        raise RuntimeError(f"zero-norm feature rows at {bad.tolist()} (first few); normalised "
                           "exact retrieval would be undefined")
    if not np.issubdtype(targets.dtype, np.integer):
        if not np.all(np.equal(np.mod(targets.astype(np.float64), 1), 0)):
            raise RuntimeError(f"rows {row_offset}..{row_offset + len(features)}: "
                               f"non-integer labels of dtype {targets.dtype}")
    targets = targets.astype(np.int64, copy=False)
    if (targets < 0).any():
        bad = np.flatnonzero(targets < 0)[:8] + row_offset
        raise RuntimeError(f"negative labels at rows {bad.tolist()} (first few)")
    return features, targets


# -------------------------------------------------------------- feature sink

class FeatureSink:
    """Resumable float32 feature store with a durable row cursor.

    Features and labels share one cursor and one receipt, so an interrupted
    extraction can never re-align one against the other -- the failure mode of
    the released ``stack_emb.py``, which concatenated feature chunks in
    ``os.listdir`` order and dropped the labels entirely.

    The partial arrays are published under their final names only after the last
    row is in place, so a half-finished extraction is never mistaken for a
    complete one.
    """

    #: fsync and republish the cursor this often; a crash costs at most this
    #: many batches of recomputation.
    FLUSH_BATCHES = 25

    def __init__(self, cache_dir: str | Path, *, rows: int, dim: int,
                 identity: dict[str, Any], stage: str = "embed") -> None:
        if int(rows) < 1 or int(dim) < 1:
            raise ValueError(f"degenerate feature sink: rows={rows}, dim={dim}")
        self.rows, self.dim = int(rows), int(dim)
        self.identity = identity
        self.signature = canonical_hash(identity)
        self.directory = Path(cache_dir) / "features" / self.signature
        self.log = get_logger(stage)
        self.cursor = 0
        self._receipt: dict[str, Any] = {}
        self._features = None
        self._labels = None
        self._lock = None
        self._batches = 0
        self._started = 0.0
        self._prior_seconds = 0.0

    # -- paths ------------------------------------------------------------
    @property
    def _receipt_path(self) -> Path:
        return self.directory / "sink.json"

    @property
    def _final(self) -> tuple[Path, Path]:
        return self.directory / "features.npy", self.directory / "labels.npy"

    @property
    def _partial(self) -> tuple[Path, Path]:
        return self.directory / "features.partial.npy", self.directory / "labels.partial.npy"

    @property
    def remaining(self) -> int:
        return self.rows - self.cursor

    @property
    def complete(self) -> bool:
        return bool(self._receipt.get("complete")) and self.cursor == self.rows

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> "FeatureSink":
        import numpy as np

        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = exclusive_lock(self.directory / "sink.lock")
        self._lock.__enter__()
        receipt = read_json(self._receipt_path) or {}
        if receipt and receipt.get("signature") != self.signature:
            raise RuntimeError(f"feature cache {self.directory} belongs to another extraction")
        cursor = int(receipt.get("rows_complete", 0))
        if not 0 <= cursor <= self.rows:
            raise RuntimeError(f"invalid durable cursor {cursor} in {self._receipt_path}")
        features_final, labels_final = self._final
        features_partial, labels_partial = self._partial
        if receipt.get("complete"):
            if cursor != self.rows or not features_final.is_file() or not labels_final.is_file():
                raise RuntimeError(f"feature cache {self.directory} claims completion but is not whole")
            self._receipt, self.cursor = receipt, cursor
            self.log.info("feature cache hit: %s rows at %s", f"{self.rows:,}", features_final)
            return self
        if cursor and not (features_partial.is_file() and labels_partial.is_file()):
            raise RuntimeError(f"durable cursor is {cursor} but the partial arrays are missing "
                               f"in {self.directory}")
        if features_partial.is_file():
            self._features = np.load(features_partial, mmap_mode="r+", allow_pickle=False)
            self._labels = np.load(labels_partial, mmap_mode="r+", allow_pickle=False)
            if (self._features.shape != (self.rows, self.dim) or self._features.dtype != np.float32
                    or self._labels.shape != (self.rows,) or self._labels.dtype != np.int64):
                raise RuntimeError(f"malformed partial feature cache in {self.directory}")
        else:
            require_disk(self.directory, self.rows * (self.dim * 4 + 8),
                         what=f"feature cache {self.signature[:12]}")
            self._features = np.lib.format.open_memmap(
                features_partial, mode="w+", dtype=np.float32, shape=(self.rows, self.dim))
            self._labels = np.lib.format.open_memmap(
                labels_partial, mode="w+", dtype=np.int64, shape=(self.rows,))
        self.cursor = cursor
        self._prior_seconds = float(receipt.get("seconds", 0.0))
        self._started = time.monotonic()
        self._receipt = {"signature": self.signature, "identity": self.identity,
                         "rows": self.rows, "dim": self.dim, "rows_complete": cursor,
                         "seconds": self._prior_seconds, "complete": False}
        atomic_json(self._receipt_path, self._receipt)
        if cursor:
            self.log.info("resuming feature cache at row %s/%s", f"{cursor:,}", f"{self.rows:,}")
        return self

    def __exit__(self, kind, value, traceback) -> None:
        try:
            if value is not None and self._features is not None and self.cursor < self.rows:
                # Keep whatever is already correct so the next attempt resumes.
                self._checkpoint()
        finally:
            self._features = self._labels = None
            if self._lock is not None:
                self._lock.__exit__(None, None, None)
                self._lock = None

    # -- writing ----------------------------------------------------------
    def write(self, features, labels) -> int:
        """Append one validated batch. Returns the new cursor."""
        if self._features is None:
            raise RuntimeError("FeatureSink.write outside the context manager or after completion")
        values, targets = validate_features(features, labels, dim=self.dim, row_offset=self.cursor)
        stop = self.cursor + len(values)
        if stop > self.rows:
            raise RuntimeError(f"batch of {len(values)} rows overruns the sink: "
                               f"{self.cursor} + {len(values)} > {self.rows}")
        self._features[self.cursor:stop] = values
        self._labels[self.cursor:stop] = targets
        self.cursor = stop
        self._batches += 1
        if self._batches % self.FLUSH_BATCHES == 0 or self.cursor == self.rows:
            self._checkpoint()
        return self.cursor

    def _checkpoint(self) -> None:
        features_partial, labels_partial = self._partial
        for array, path in ((self._features, features_partial), (self._labels, labels_partial)):
            if array is None:
                continue
            array.flush()
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        self._receipt.update(rows_complete=self.cursor,
                             seconds=self._prior_seconds + time.monotonic() - self._started)
        atomic_json(self._receipt_path, self._receipt)

    def publish(self) -> tuple[Any, Any]:
        """Verify the sink is whole, rename it into place, return read-only views."""
        import numpy as np

        features_final, labels_final = self._final
        if not self.complete:
            if self.cursor != self.rows:
                raise RuntimeError(f"extraction ended early: {self.cursor:,}/{self.rows:,} rows")
            self._checkpoint()
            self._features = self._labels = None
            features_partial, labels_partial = self._partial
            os.replace(features_partial, features_final)
            os.replace(labels_partial, labels_final)
            self._receipt.update(complete=True, rows_complete=self.rows,
                                 features_bytes=features_final.stat().st_size,
                                 labels_bytes=labels_final.stat().st_size)
            atomic_json(self._receipt_path, self._receipt)
        features = np.load(features_final, mmap_mode="r", allow_pickle=False)
        labels = np.load(labels_final, mmap_mode="r", allow_pickle=False)
        if features.shape != (self.rows, self.dim) or labels.shape != (self.rows,):
            raise RuntimeError(f"published feature cache in {self.directory} has the wrong shape")
        return features, labels


# -------------------------------------------------------------- the archive

def archive_manifest_path(out_path: str | Path) -> Path:
    """``<archive>.npz`` -> ``<archive>.npz.json`` (the receipt travels with it)."""
    out_path = Path(out_path)
    return out_path.with_name(out_path.name + ".json")


def load_current_archive(out_path: str | Path, *, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Return the manifest when ``out_path`` already holds exactly this archive."""
    out_path = Path(out_path)
    manifest = read_json(archive_manifest_path(out_path))
    if not manifest or not out_path.is_file():
        return None
    if manifest.get("identity") != canonical_hash(identity):
        return None
    digest = sha256_file(out_path)
    if manifest.get("sha256") != digest:
        raise RuntimeError(f"{out_path} hashes to {digest} but its manifest says "
                           f"{manifest.get('sha256')}; move both aside and extract again")
    return manifest


def finalize_archive(out_path: str | Path, features, labels, *,
                     manifest: dict[str, Any]) -> dict[str, Any]:
    """Write ``{data, labels}`` plus the manifest that explains how it was made.

    ``manifest`` must already describe the recipe, the backbone, the weights and
    the preprocessing; this function adds only what it can measure itself.
    """
    import numpy as np

    out_path = Path(out_path)
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"{len(features):,} rows but {len(labels):,} labels")
    deviations = list(manifest.get("deviations", []))
    if LABEL_KEY_DEVIATION not in deviations:
        deviations.append(LABEL_KEY_DEVIATION)
    require_disk(out_path.parent, features.size * 4 + labels.size * 8, what=out_path.name)
    digest = save_npz(out_path, data=features, labels=labels)
    payload = {**manifest,
               "archive": out_path.name,
               "rows": int(len(features)),
               "dim": int(features.shape[1]),
               "keys": ["data", "labels"],
               "dtypes": {"data": "float32", "labels": "int64"},
               "precision": manifest.get("precision", PRECISION),
               "deviations": deviations,
               "labels_sha256": array_hash(np.ascontiguousarray(labels, dtype=np.int64)),
               "sha256": digest,
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "libraries": manifest.get("libraries") or library_versions()}
    atomic_json(archive_manifest_path(out_path), payload)
    get_logger(manifest.get("recipe", "embed")).info(
        "archive %s: %s x %s rows, sha256 %s", out_path.name, f"{payload['rows']:,}",
        payload["dim"], digest[:12])
    return payload
