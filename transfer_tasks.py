"""Frozen-crop downstream readouts for the October 2026 transfer specifications.

This module never fits an image encoder or compressor. Dense descriptors are
memory mapped, CSR coordinates survive the indexed first layer, and every fitted
quantity is estimated exclusively from ``fit_indices``. Importing it does not
start training or download anything. The orchestrator owns the experimental grid.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL_ID = "mrl-csr-frozen-transfer-2026-10-19-v1"
TASK_NAMES = {"A": "detection", "B": "mask", "C": "pose"}
POSE_CLASSES = ("aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair",
                "diningtable", "motorbike", "sofa", "train", "tvmonitor")
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def _task(name):
    name = TASK_NAMES.get(name, name)
    if name not in ("detection", "mask", "pose"):
        raise ValueError(f"Unknown transfer task: {name}")
    return name


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(data), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _array_hash(value):
    array = np.asarray(value, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device):
    result = {"peak_gpu_memory_bytes": 0, "peak_cpu_rss_bytes": None}
    if torch.device(device).type == "cuda":
        result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_cpu_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        # psutil RSS is current memory, so it must not be labelled peak memory.
        pass
    return result


@dataclass
class SparseBatch:
    indices: torch.Tensor
    values: torch.Tensor
    width: int


class FeatureStore:
    """Read-only FP32 dense .npy or indexed sparse cache, with bounded batches.

    Sparse directories contain metadata.json (``width``), values.npy [N,K],
    indices.npy [N,K] uint32 and nnz.npy [N] uint32. Nonzeros occupy each row's
    first nnz slots. Padded slots are masked, and their storage is counted. The
    sparse cache need not pretend its padded storage is 8*actual_nnz bytes.
    """

    def __init__(self, path, width=None):
        self.path = Path(path)
        self.sparse = self.path.is_dir()
        if self.sparse:
            self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
            if not self.metadata.get("complete") or not self.metadata.get("signature"):
                raise ValueError("Sparse cache has not completed compression")
            self.width = int(self.metadata["width"])
            if width is not None and int(width) != self.width:
                raise ValueError("A sparse code cannot be truncated by value rank")
            self.values = np.load(self.path / "values.npy", mmap_mode="r", allow_pickle=False)
            self.indices = np.load(self.path / "indices.npy", mmap_mode="r", allow_pickle=False)
            self.nnz = np.load(self.path / "nnz.npy", mmap_mode="r", allow_pickle=False)
            if (self.values.dtype != np.float32 or self.indices.dtype != np.uint32
                    or self.nnz.dtype != np.uint32 or self.values.ndim != 2
                    or self.indices.shape != self.values.shape
                    or self.nnz.shape != (len(self.values),)):
                raise ValueError("Invalid FP32/uint32 sparse cache layout")
            self.rows = len(self.values)
            if self.metadata.get("rows") != self.rows or self.metadata.get("rows_complete") != self.rows:
                raise ValueError("Sparse cache row metadata does not match stored arrays")
            self.serialized_bytes = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
            fingerprint_paths = [self.path / n for n in ("metadata.json", "values.npy", "indices.npy", "nnz.npy")]
        else:
            metadata_path = self.path.with_suffix(".json")
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not self.metadata.get("complete") or not self.metadata.get("signature"):
                raise ValueError("Dense cache has not completed feature extraction")
            self.values = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if self.values.dtype != np.float32 or self.values.ndim != 2:
                raise ValueError("Primary accuracy caches must contain 2-D FP32 arrays")
            self.width = int(width or self.values.shape[1])
            if not 0 < self.width <= self.values.shape[1]:
                raise ValueError("Requested prefix exceeds cached feature width")
            self.rows = len(self.values)
            if (self.metadata.get("rows") != self.rows or self.metadata.get("rows_complete") != self.rows
                    or self.metadata.get("parent", {}).get("dimension") != self.values.shape[1]):
                raise ValueError("Dense cache row/dimension metadata does not match stored array")
            self.serialized_bytes = self.path.stat().st_size
            fingerprint_paths = [self.path, metadata_path]
        self.fingerprint = _digest({"width": self.width, "files": [
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in fingerprint_paths]})

    def __len__(self):
        return self.rows

    def _values(self, row_indices):
        rows = np.asarray(row_indices, dtype=np.int64)
        values = np.array(self.values[rows, :self.width] if not self.sparse else self.values[rows],
                          dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite cached descriptors")
        if self.sparse:
            counts = np.asarray(self.nnz[rows], dtype=np.int64)
            if np.any(counts < 0) or np.any(counts > values.shape[1]):
                raise ValueError("Sparse row counts exceed stored support capacity")
            values[np.arange(values.shape[1])[None, :] >= counts[:, None]] = 0
        return values

    def batch(self, row_indices, device="cpu", scale=1.0):
        values = torch.from_numpy(self._values(row_indices)).to(device=device, dtype=torch.float32)
        values = values / float(scale)
        if not self.sparse:
            return values
        indices = np.array(self.indices[np.asarray(row_indices, dtype=np.int64)], dtype=np.int64, copy=True)
        if indices.size and (indices.min() < 0 or indices.max() >= self.width):
            raise ValueError("Sparse coordinate outside representation width")
        return SparseBatch(torch.from_numpy(indices).to(device), values, self.width)

    def scalar_normalization(self, row_indices, chunk_size=4096):
        rows = np.asarray(row_indices, dtype=np.int64)
        if not len(rows):
            raise ValueError("Cannot fit normalization on an empty training pool")
        squared = 0.0
        for start in range(0, len(rows), chunk_size):
            values = self._values(rows[start:start + chunk_size]).astype(np.float64)
            squared += float(np.square(values).sum(dtype=np.float64))
        return max(math.sqrt(squared / len(rows)), 1e-12)

    def cost_metadata(self):
        if not self.sparse:
            return {"stored_bytes_per_region": self.width * 4,
                    "feature_cache_bytes_on_disk": self.serialized_bytes,
                    "measured_nnz": None, "storage_layout": "dense-fp32-prefix-view"}
        total_nnz = 0
        for start in range(0, self.rows, 65536):
            total_nnz += int(np.asarray(self.nnz[start:start + 65536], dtype=np.uint64).sum())
        return {"stored_bytes_per_region": self.serialized_bytes / max(1, self.rows),
                "feature_cache_bytes_on_disk": self.serialized_bytes,
                "measured_nnz": total_nnz / max(1, self.rows),
                "storage_layout": "padded-fp32-uint32-with-uint32-row-count",
                "unpadded_payload_bytes_per_region": 8 * total_nnz / max(1, self.rows) + 4}


class IndexedLinear(nn.Module):
    """Affine map with identical coordinate semantics for dense and sparse input.

    embedding_bag sums weighted coordinate rows without building an N x q dense
    code or an N x k x output tensor. Its trainable table still contains q*out
    weights and its optimizer state is dense; sparse storage is not free capacity.
    """

    def __init__(self, input_width, output_width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_width, output_width))
        self.bias = nn.Parameter(torch.zeros(output_width))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, value):
        if isinstance(value, SparseBatch):
            if value.width != self.weight.shape[0]:
                raise ValueError("Sparse coordinate system and head width differ")
            n, k = value.values.shape
            if k == 0:
                return self.bias.expand(n, -1)
            offsets = torch.arange(n, device=value.indices.device, dtype=torch.long) * k
            return F.embedding_bag(value.indices.reshape(-1), self.weight, offsets,
                                   mode="sum", per_sample_weights=value.values.reshape(-1)) + self.bias
        return value @ self.weight + self.bias


def validate_readout_interface(device="cpu"):
    """Runtime protocol gate for coordinate-preserving sparse forward/backward.

    This runs only when the experiment launcher requests it. It uses tiny fixed
    inputs, including partially padded support and an all-zero code, and returns
    numerical evidence for the run provenance. CPU RNG is restored on exit;
    initialization occurs on CPU, so CUDA random generators are never consumed.
    """
    device = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026)
        sparse_layer = IndexedLinear(11, 7).to(device=device, dtype=torch.float32)
        dense_layer = IndexedLinear(11, 7).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            sparse_layer.bias.copy_(torch.linspace(-.2, .3, 7, device=device))
            dense_layer.load_state_dict(sparse_layer.state_dict())
        indices = torch.tensor([[9, 1, 5], [2, 8, 0], [10, 3, 6], [4, 7, 0]],
                               dtype=torch.long, device=device)
        values = torch.tensor([[.75, .25, .5], [1.25, 0., 0.], [.1, .8, .6], [0., 0., 0.]],
                              dtype=torch.float32, device=device)
        dense = torch.zeros((4, 11), dtype=torch.float32, device=device)
        dense.scatter_(1, indices, values)
        sparse_output = sparse_layer(SparseBatch(indices, values, 11))
        dense_output = dense_layer(dense)
        upstream = torch.linspace(-1., 1., 28, device=device).reshape(4, 7)
        sparse_output.backward(upstream)
        dense_output.backward(upstream)
        comparisons = {
            "forward": (sparse_output, dense_output),
            "weight_gradient": (sparse_layer.weight.grad, dense_layer.weight.grad),
            "bias_gradient": (sparse_layer.bias.grad, dense_layer.bias.grad),
            "zero_code_bias": (sparse_output[-1], sparse_layer.bias),
        }
        differences = {}
        for name, (actual, expected) in comparisons.items():
            if not (bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())):
                raise FloatingPointError(f"Nonfinite sparse-readout interface check: {name}")
            differences[name + "_max_abs_difference"] = float((actual - expected).abs().max().detach())
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"Sparse readout equivalence failed for {name}: {differences}")
    return {"status": "passed", "device": str(device), "precision": "fp32", "width": 11,
            "output_width": 7, "examples": 4, "rtol": 1e-5, "atol": 1e-6,
            "includes_all_zero_code": True, "includes_zero_padding": True, **differences}


class Readout(nn.Module):
    def __init__(self, width, outputs, hidden=None):
        super().__init__()
        self.first = IndexedLinear(width, outputs if hidden is None else hidden)
        self.last = None if hidden is None else nn.Linear(hidden, outputs)
        if self.last is not None:
            nn.init.normal_(self.last.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.last.bias)

    @property
    def output_bias(self):
        return self.first.bias if self.last is None else self.last.bias

    def forward(self, value):
        value = self.first(value)
        return value if self.last is None else self.last(F.relu(value))


class TransferHead(nn.Module):
    def __init__(self, task, width, head_family):
        super().__init__()
        self.task = _task(task)
        family = head_family.lower()
        if family in ("h0", "affine", "linear"):
            hidden = None
        elif family in ("h1", "mlp", "mlp128"):
            hidden = 128
        elif family in ("mlp256", "sensitivity256"):
            hidden = 256  # separately labelled, full-parent sensitivity only
        else:
            raise ValueError(f"Unknown head family {head_family}")
        if self.task == "detection":
            self.classifier = Readout(width, 21, hidden)
            self.regressor = Readout(width, 4, hidden)
        else:
            self.readout = Readout(width, 784 if self.task == "mask" else 72, hidden)
            if self.task == "pose":
                with torch.no_grad():
                    self.readout.output_bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.] * 12))

    def forward(self, value):
        if self.task == "detection":
            return self.classifier(value), self.regressor(value)
        value = self.readout(value)
        return value.reshape(-1, 28, 28) if self.task == "mask" else value.reshape(-1, 12, 6)


def rotations_from_6d(vectors, strict=False):
    """Return proper column-frame rotations and failure flags; never hide failure."""
    a1, a2 = vectors[..., :3], vectors[..., 3:]
    n1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    b1 = a1 / n1.clamp_min(1e-6)
    t2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    n2 = torch.linalg.vector_norm(t2, dim=-1, keepdim=True)
    b2 = t2 / n2.clamp_min(1e-6)
    matrix = torch.stack((b1, b2, torch.linalg.cross(b1, b2, dim=-1)), dim=-1)
    failed = (~torch.isfinite(vectors).all(dim=-1) | (n1[..., 0] < 1e-6)
              | (n2[..., 0] < 1e-6))
    # Invalid matrices are replaced only for safe downstream arithmetic. Their
    # failure mask is retained and evaluation assigns exactly 180 degrees.
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    safe = torch.where(failed[..., None, None], identity, matrix)
    orthogonal_error = (safe.transpose(-1, -2) @ safe - identity).abs().amax(dim=(-1, -2))
    determinant = torch.linalg.det(safe)
    failed = failed | (~torch.isfinite(matrix).all(dim=(-1, -2))) | (orthogonal_error > 1e-3) | ((determinant - 1).abs() > 1e-3)
    if strict and bool(failed.any()):
        raise FloatingPointError(f"Degenerate 6-D pose predictions: {int(failed.sum())}; fit halted")
    return safe, failed


def _box_iou(boxes, targets):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    extent = np.maximum(0., np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
                        - np.maximum(boxes[:, None, :2], targets[None, :, :2]))
    intersection = extent.prod(axis=-1)
    area_a = np.maximum(0., boxes[:, 2:] - boxes[:, :2]).prod(axis=-1)
    area_b = np.maximum(0., targets[:, 2:] - targets[:, :2]).prod(axis=-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-30)


def _box_targets(record):
    r = np.asarray(record["box"], dtype=np.float64)
    g = np.asarray(record["matched_box"], dtype=np.float64)
    rw, gw = r[2:] - r[:2], g[2:] - g[:2]
    if np.any(rw <= 0) or np.any(gw <= 0):
        raise ValueError("Invalid training proposal or matched ground-truth box")
    return np.r_[((g[:2] + g[2:]) - (r[:2] + r[2:])) / (2 * rw), np.log(gw / rw)]


def _category(record, task):
    if task == "pose":
        return int(record.get("pose_class_id", record["category"]))
    return int(record["category"])


def _record_identity(records):
    metadata = getattr(records, "metadata", {})
    manifest = getattr(records, "path", None)
    return {"length": len(records), "metadata": metadata, "fingerprint": getattr(records, "fingerprint", None),
            "path": None if manifest is None else str(manifest)}


class CachedMaskTargets:
    """Shared packed annotation cache: 12,544 bytes/crop, no float image cache.

    Construction is resumable, and only sampled rows are unpacked during fitting.
    Evaluation labels may be rasterized here, but never enter fitting statistics.
    The immutable row manifest is part of the cache identity.
    """

    def __init__(self, records, directory, target_provider=None):
        if target_provider is None:
            from transfer_data import mask_target
            target_provider = mask_target
        self.records = records
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = {"records": _record_identity(records), "format": "target-valid-packbits-big-224-v1"}
        signature = _digest(identity)
        metadata_path = self.directory / "metadata.json"
        path = self.directory / "targets.npy"
        shape = (len(records), 2, 224 * 224 // 8)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata and metadata.get("signature") != signature:
            raise ValueError("Packed mask targets belong to another manifest")
        start = int(metadata.get("rows_complete", 0))
        if not 0 <= start <= len(records):
            raise ValueError("Invalid packed-mask resume cursor")
        if path.exists():
            packed = np.load(path, mmap_mode="r" if metadata.get("complete") else "r+", allow_pickle=False)
            if packed.shape != shape or packed.dtype != np.uint8:
                raise ValueError("Malformed packed-mask array")
        else:
            if start or metadata.get("complete"):
                raise ValueError("Committed packed mask target array is missing")
            import shutil
            required = math.prod(shape) + (128 << 20)
            if shutil.disk_usage(self.directory).free < required:
                raise RuntimeError(f"Insufficient space for packed mask targets: need {required:,} bytes")
            packed = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
        if metadata.get("complete"):
            if start != len(records):
                raise ValueError("Packed mask completion metadata is inconsistent")
        else:
            for row in range(start, len(records)):
                target, valid = target_provider(records[row])
                values = np.stack((np.asarray(target, dtype=np.bool_), np.asarray(valid, dtype=np.bool_)))
                if values.shape != (2, 224, 224):
                    raise ValueError("Mask rasterizer must return 224x224 target and valid maps")
                packed[row] = np.packbits(values.reshape(2, -1), axis=-1, bitorder="big")
                if (row + 1) % 256 == 0 or row + 1 == len(records):
                    packed.flush()
                    metadata = {"signature": signature, "rows_complete": row + 1,
                                "rows": len(records), "complete": row + 1 == len(records),
                                "bytes_per_crop": 12544, "serialized_bytes": path.stat().st_size}
                    _atomic_json(metadata_path, metadata)
                    print(f"[mask_targets] {row + 1}/{len(records)} crops packed", flush=True)
            del packed
            packed = np.load(path, mmap_mode="r", allow_pickle=False)
        self.packed = packed
        self._record_lookup = None

    def batch(self, rows, device="cpu"):
        values = np.unpackbits(np.asarray(self.packed[np.asarray(rows, dtype=np.int64)]),
                               axis=-1, count=224 * 224, bitorder="big").reshape(-1, 2, 224, 224)
        tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        return tensor[:, 0], tensor[:, 1]

    def _lookup_rows(self, records):
        if self._record_lookup is None:
            self._record_lookup = {}
            for row in range(len(self.records)):
                item = self.records[row]
                key = (str(item["image_id"]), str(item["object_id"]), int(item["variant"]))
                if key in self._record_lookup:
                    raise ValueError("Duplicate object crop in packed-mask manifest")
                self._record_lookup[key] = row
        return [self._record_lookup[(str(record["image_id"]), str(record["object_id"]), int(record["variant"]))]
                for record in records]

    def batch_records(self, records, device="cpu"):
        return self.batch(self._lookup_rows(records), device)

    def get_record(self, record):
        target, valid = self.batch_records([record])
        return target[0], valid[0]


def presample_indices(task, records, fit_indices, seed, directory, max_updates=5000):
    """Save a shared, deterministic, disk-backed plan before any head is fitted."""
    task = _task(task)
    rows = np.asarray(fit_indices, dtype=np.int64)
    batch_size = 512 if task == "detection" else 256
    identity = {"protocol_id": PROTOCOL_ID, "task": task, "seed": int(seed),
                "fit_rows_sha256": _array_hash(rows), "updates": int(max_updates),
                "batch_size": batch_size, "records_hash": _digest(_record_identity(records))}
    key = _digest(identity)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}-seed{seed}-{key[:16]}.npy"
    metadata_path = path.with_suffix(".json")
    accepted_path = path.with_suffix(".accepted.npy")
    if path.exists() and metadata_path.exists() and accepted_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Presampling provenance mismatch")
        plan = np.load(path, mmap_mode="r", allow_pickle=False)
        accepted = np.load(accepted_path, mmap_mode="r", allow_pickle=False)
        if plan.shape != (max_updates, batch_size) or saved.get("accepted_rows_sha256") != _array_hash(accepted):
            raise ValueError("Incomplete presampling plan or accepted training pool")
        return plan, accepted, saved
    if task == "detection":
        positive, background = [], []
        for row in rows:
            label = int(records[int(row)]["label"])
            (positive if label > 0 else background if label == 0 else []).append(int(row))
        if not positive or not background:
            raise ValueError("Detector fitting requires both positive and background candidates")
        positive, background = np.asarray(positive), np.asarray(background)
        accepted = np.sort(np.r_[positive, background])
        categories = None
    else:
        objects = {}
        for row in rows:
            record = records[int(row)]
            key = (str(record["image_id"]), str(record["object_id"]))
            objects.setdefault((_category(record, task), key), {})[int(record["variant"])] = int(row)
        categories = {}
        for (category, object_key), variants in sorted(objects.items()):
            if set(variants) != set(range(5)):
                raise ValueError(f"Object {object_key} does not have all five crop variants")
            categories.setdefault(category, []).append([variants[v] for v in range(5)])
        if not categories:
            raise ValueError("No eligible extension training objects")
        accepted = np.sort(rows)
    identity["accepted_rows_sha256"] = _array_hash(accepted)
    temporary = path.with_name(path.name + ".tmp")
    dtype = np.uint32 if len(records) < 2 ** 32 else np.uint64
    plan = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype,
                                    shape=(max_updates, batch_size))
    rng = np.random.default_rng(seed)
    if categories is not None:
        class_ids = np.asarray(sorted(categories), dtype=np.int64)
        by_class = {c: np.asarray(categories[c], dtype=np.int64) for c in class_ids}
    for step in range(max_updates):
        if task == "detection":
            plan[step] = np.r_[rng.choice(positive, 128), rng.choice(background, 384)]
        else:
            sampled_classes = rng.choice(class_ids, batch_size)
            chosen = np.empty(batch_size, dtype=np.int64)
            for category in class_ids:
                positions = np.flatnonzero(sampled_classes == category)
                pool = by_class[category]
                objects_chosen = rng.integers(len(pool), size=len(positions))
                variants_chosen = rng.integers(5, size=len(positions))
                chosen[positions] = pool[objects_chosen, variants_chosen]
            plan[step] = chosen
    plan.flush()
    del plan
    os.replace(temporary, path)
    accepted_temp = accepted_path.with_name(accepted_path.name + ".tmp")
    with accepted_temp.open("wb") as stream:
        np.save(stream, accepted, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(accepted_temp, accepted_path)
    _atomic_json(metadata_path, identity)
    return np.load(path, mmap_mode="r", allow_pickle=False), accepted, identity


class TrainingTargets:
    """Shared numeric fitting targets, avoiding millions of JSON reads per fit."""

    def __init__(self, task, records, accepted, directory):
        self.task = task
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        signature = _digest({"task": task, "records": _record_identity(records),
                             "accepted_sha256": _array_hash(accepted), "format": "numeric-task-targets-v1"})
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata and metadata.get("signature") != signature:
            raise ValueError("Numeric fitting targets belong to a different accepted pool")
        specifications = ({"labels": ((len(records),), np.int8), "box_targets": ((len(records), 4), np.float32)}
                          if task == "detection" else
                          {"categories": ((len(records),), np.uint8), "rotations": ((len(records), 3, 3), np.float32)}
                          if task == "pose" else {})
        self.arrays = {}
        if metadata.get("complete"):
            for name, (shape, dtype) in specifications.items():
                array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError("Malformed numeric fitting target array")
                self.arrays[name] = array
            self.metadata = metadata
            return
        _atomic_json(metadata_path, {"signature": signature, "complete": False})
        for name, (shape, dtype) in specifications.items():
            self.arrays[name] = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        images, objects = set(), set()
        target_sum, target_squared, n_positive = np.zeros(4), np.zeros(4), 0
        for index, row in enumerate(accepted):
            row = int(row)
            record = records[row]
            images.add(str(record["image_id"]))
            if task == "detection":
                label = int(record["label"])
                self.arrays["labels"][row] = label
                if label > 0:
                    target = _box_targets(record)
                    self.arrays["box_targets"][row] = target
                    target_sum += target
                    target_squared += target * target
                    n_positive += 1
                    objects.add((str(record["image_id"]), str(record.get("matched_object_index", record["matched_box"]))))
            else:
                objects.add((str(record["image_id"]), str(record["object_id"])))
                if task == "pose":
                    self.arrays["categories"][row] = _category(record, task)
                    self.arrays["rotations"][row] = np.asarray(record["rotation"], dtype=np.float32)
            if (index + 1) % 50000 == 0:
                print(f"[{task}_targets] {index + 1}/{len(accepted)} training crops indexed", flush=True)
        metadata = {"signature": signature, "complete": True, "fit_images": len(images),
                    "fit_objects": len(objects), "fit_crops": len(accepted), "rows": len(records)}
        if task == "detection":
            mean = target_sum / n_positive
            std = np.maximum(np.sqrt(np.maximum(target_squared / n_positive - mean * mean, 0.)), 1e-3)
            metadata.update(box_target_mean=mean.tolist(), box_target_std=std.tolist())
        for name in list(self.arrays):
            self.arrays[name].flush()
            del self.arrays[name]
            self.arrays[name] = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        _atomic_json(metadata_path, metadata)
        self.metadata = metadata

    def batch(self, rows, device):
        result = {}
        for name, array in self.arrays.items():
            values = torch.from_numpy(np.array(array[np.asarray(rows, dtype=np.int64)], copy=True)).to(device)
            result[name] = values.long() if name in ("labels", "categories") else values.float()
        return result


def _save_checkpoint(path, head, optimizer, step, config, fit_info):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save({"model": head.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": int(step), "config": config, "fit_info": fit_info,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path, device):
    # Only self-produced checkpoints containing tensors/primitive containers.
    return torch.load(path, map_location=device, weights_only=True)


def train_head(task, features, records, fit_indices, run_dir, seed=0,
               head_family="mlp", updates=5000, device="cuda", target_provider=None,
               checkpoint_steps=(1000, 2500, 5000), resume=True, sampling_dir=None,
               max_updates=5000, log_every=25, checkpoint_every=250):
    """Fit a fresh task readout or resume an exactly matching saved configuration.

    No validation/test information participates in normalization, sampling, loss,
    or optimizer state. Development duration selection is a separate operation.
    """
    task = _task(task)
    device = torch.device(device)
    if not isinstance(features, FeatureStore):
        features = FeatureStore(features)
    if len(features) != len(records):
        raise ValueError("Feature rows and crop manifest differ")
    if not 0 < updates <= max_updates:
        raise ValueError("Requested head updates exceed the shared presampled plan")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan, accepted, sample_identity = presample_indices(
        task, records, fit_indices, seed, sampling_dir or run_dir / "sampling", max_updates)
    shared = Path(sampling_dir).parent if sampling_dir is not None else run_dir
    training_targets = TrainingTargets(task, records, accepted, shared / "numeric_targets")
    config = {"protocol_id": PROTOCOL_ID, "task": task, "width": features.width,
              "head_family": head_family, "seed": int(seed), "updates": int(updates),
              "feature_fingerprint": features.fingerprint,
              "sample_identity": sample_identity, "precision": "fp32",
              "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4}
    config_hash = _digest(config)
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Use a fresh run directory to fit a new head: {run_dir}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    head = TransferHead(task, features.width, head_family).to(device=device, dtype=torch.float32)
    matrix_parameters, bias_parameters = [], []
    for name, parameter in head.named_parameters():
        (bias_parameters if name.endswith("bias") else matrix_parameters).append(parameter)
    optimizer = torch.optim.AdamW([{"params": matrix_parameters, "weight_decay": 1e-4},
                                   {"params": bias_parameters, "weight_decay": 0.}],
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    start = 0
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, "cpu")
        if checkpoint["config"] != config:
            raise ValueError(f"Refusing incompatible head resume in {run_dir}")
        head.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        start, fit_info = int(checkpoint["step"]), checkpoint["fit_info"]
        if not 0 <= start <= updates:
            raise ValueError("Saved head update cursor is outside the requested schedule")
        del checkpoint
    else:
        scale = features.scalar_normalization(accepted)
        target_metadata = training_targets.metadata
        fit_info = {"config_hash": config_hash, "input_scale": scale,
                    "head_parameters": sum(p.numel() for p in head.parameters()),
                    "fit_images": target_metadata["fit_images"], "fit_objects": target_metadata["fit_objects"],
                    "fit_crops": len(accepted), "head_fit_time": 0., "update_count": 0,
                    "precision": "fp32", **features.cost_metadata()}
        if task == "detection":
            fit_info.update(box_target_mean=target_metadata["box_target_mean"],
                            box_target_std=target_metadata["box_target_std"])
    _atomic_json(run_dir / "config.json", config)
    if start >= updates:
        return fit_info
    packed_targets = None
    if task == "mask" and target_provider is None:
        packed_targets = CachedMaskTargets(records, shared / "target_cache")
    if task == "detection":
        target_mean = torch.tensor(fit_info["box_target_mean"], device=device, dtype=torch.float32)
        target_std = torch.tensor(fit_info["box_target_std"], device=device, dtype=torch.float32)
    _synchronize(device)
    started = time.perf_counter()
    prior_elapsed = float(fit_info["head_fit_time"])
    head.train()
    current_step = start
    try:
        with (run_dir / "training.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            for index in range(start, updates):
                rows = np.asarray(plan[index], dtype=np.int64)
                numeric_targets = training_targets.batch(rows, device)
                batch_records = [records[int(row)] for row in rows] if task == "mask" and target_provider is not None else None
                values = features.batch(rows, device, fit_info["input_scale"])
                prediction = head(values)
                optimizer.zero_grad(set_to_none=True)
                components = {}
                if task == "detection":
                    logits, offsets = prediction
                    labels = numeric_targets["labels"]
                    positive = labels > 0
                    targets = (numeric_targets["box_targets"][positive] - target_mean) / target_std
                    cls_loss = F.cross_entropy(logits, labels)
                    box_loss = F.smooth_l1_loss(offsets[positive], targets, beta=1., reduction="none").sum(1).mean()
                    loss = cls_loss + box_loss
                    components = {"classification_loss": float(cls_loss.detach()), "box_loss": float(box_loss.detach())}
                elif task == "mask":
                    if packed_targets is not None:
                        targets, valid = packed_targets.batch(rows, device)
                    else:
                        targets, valid = zip(*(target_provider(r) for r in batch_records))
                        targets = torch.stack([torch.as_tensor(t) for t in targets]).to(device=device, dtype=torch.float32)
                        valid = torch.stack([torch.as_tensor(v) for v in valid]).to(device=device, dtype=torch.float32)
                    counts = valid.sum(dim=(-1, -2))
                    if bool((counts == 0).any()):
                        raise ValueError("A fitting mask crop has no valid pixels")
                    logits = F.interpolate(prediction[:, None], size=(224, 224), mode="bilinear",
                                           align_corners=False)[:, 0]
                    loss = ((F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid)
                            .sum(dim=(-1, -2)) / counts).mean()
                else:
                    category = numeric_targets["categories"]
                    six = prediction[torch.arange(len(rows), device=device), category]
                    rotation, _ = rotations_from_6d(six, strict=True)
                    targets = numeric_targets["rotations"]
                    loss = F.mse_loss(rotation, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite task loss; refusing to advance fit")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                    raise FloatingPointError("Nonfinite readout gradients")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
                    raise FloatingPointError("Nonfinite parameters after optimizer update")
                current_step = index + 1
                if current_step % log_every == 0 or current_step == 1 or current_step == updates:
                    _synchronize(device)
                    progress = {"step": current_step, "loss": float(loss.detach()),
                                "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                                **components, **_peak_memory(device)}
                    progress["remaining_seconds_estimate"] = ((time.perf_counter() - started)
                                                               / (current_step - start) * (updates - current_step))
                    log.write(json.dumps(progress, allow_nan=False) + "\n")
                    _atomic_json(run_dir / "status.json", {"stage": "head_training", **progress})
                    print(f"[{task}/{head_family}/seed{seed}] {current_step}/{updates} loss={progress['loss']:.6f}", flush=True)
                if current_step % checkpoint_every == 0 or current_step in checkpoint_steps or current_step == updates:
                    _synchronize(device)
                    fit_info.update(update_count=current_step,
                                    head_fit_time=prior_elapsed + time.perf_counter() - started,
                                    **_peak_memory(device))
                    if current_step in checkpoint_steps:
                        _save_checkpoint(run_dir / f"checkpoint_{current_step}.pt", head, optimizer,
                                         current_step, config, fit_info)
                    # Commit milestones first: a crash must never advance last.pt
                    # past a missing development snapshot that resume would skip.
                    _save_checkpoint(checkpoint_path, head, optimizer, current_step, config, fit_info)
                    _atomic_json(run_dir / "fit.json", fit_info)
    except BaseException as error:
        # last.pt remains the last finite completed checkpoint. Never overwrite
        # it with degenerate pose parameters or a partially completed update.
        _atomic_json(run_dir / "failure.json", {"failed_after_update": current_step,
                                               "error_type": type(error).__name__,
                                               "message": str(error), "config_hash": config_hash})
        raise
    return fit_info


def _decode_mask(logits, record):
    """Place logits in the original expanded rectangle before clipping to image."""
    width, height = int(record["width"]), int(record["height"])
    left, top, right, bottom = map(int, record["crop_box"])
    if right <= left or bottom <= top:
        raise ValueError("Invalid mask crop rectangle")
    prediction = np.zeros((height, width), dtype=np.bool_)
    resized = F.interpolate(torch.as_tensor(logits, dtype=torch.float32).reshape(1, 1, 28, 28),
                            size=(bottom - top, right - left), mode="bilinear",
                            align_corners=False)[0, 0].numpy() >= 0.
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x1 > x0 and y1 > y0:
        prediction[y0:y1, x0:x1] = resized[y0 - top:y1 - top, x0 - left:x1 - left]
    return prediction


def _mask_iou(logits, record, original_provider):
    target, valid = original_provider(record)
    target, valid = np.asarray(target, dtype=np.bool_), np.asarray(valid, dtype=np.bool_)
    failed = not np.isfinite(logits).all()
    prediction = np.zeros_like(target) if failed else _decode_mask(logits, record)
    intersection = np.count_nonzero(prediction & target & valid)
    union = np.count_nonzero((prediction | target) & valid)
    return (float(intersection / union) if union else 0.), failed


def _prediction_identity(record, task):
    return {"image_id": str(record["image_id"]), "object_id": str(record["object_id"]),
            "category": _category(record, task), "variant": int(record.get("variant", 0)),
            "occluded": bool(record.get("occluded", False)),
            "truncated": bool(record.get("truncated", False))}


def _aggregate_variant(rows, task, required_classes=None, weights=None):
    classes = range(12 if task == "pose" else 20)
    by_class = {category: [] for category in classes}
    for row in rows:
        multiplicity = 1 if weights is None else int(weights.get(row["image_id"], 0))
        if multiplicity:
            by_class[row["category"]].extend([row] * multiplicity)
    per_class = {}
    for category, values in by_class.items():
        count = len(values)
        if task == "mask":
            metric = {"count": count, "mean_IoU": None if not count else float(np.mean([v["mask_iou"] for v in values]))}
        else:
            errors = np.asarray([v["error_degrees"] for v in values], dtype=np.float64)
            metric = {"count": count, "MedErr": None if not count else float(np.median(errors)),
                      "Acc30": None if not count else float(np.mean(errors < 30.)),
                      "Acc15": None if not count else float(np.mean(errors < 15.))}
        per_class[str(category)] = metric
    represented = [category for category in classes if by_class[category]]
    if required_classes is not None and set(represented) != set(required_classes):
        return None
    result = {"per_class": per_class, "macro_class_count": len(represented),
              "prediction_count": sum(len(v) for v in by_class.values()),
              "failed_predictions": sum(int(v.get("failed", False)) for values in by_class.values() for v in values)}
    if task == "mask":
        result["macro_IoU"] = float(np.mean([per_class[str(c)]["mean_IoU"] for c in represented])) if represented else None
        all_values = [v["mask_iou"] for values in by_class.values() for v in values]
        result["instance_weighted_IoU"] = float(np.mean(all_values)) if all_values else None
        result["primary_macro_metric"] = result["macro_IoU"]
    else:
        for key in ("Acc30", "Acc15", "MedErr"):
            result[f"macro_{key}"] = float(np.mean([per_class[str(c)][key] for c in represented])) if represented else None
        errors = [v["error_degrees"] for values in by_class.values() for v in values]
        result["pooled_MedErr"] = float(np.median(errors)) if errors else None
        result["primary_macro_metric"] = result["macro_Acc30"]
    return result


def _aggregate_condition(rows, task, condition="clean", required_classes=None, weights=None):
    variants = [0] if condition == "clean" else [1, 2, 3, 4]
    per_variant = {str(variant): _aggregate_variant(
        [r for r in rows if r["variant"] == variant], task, required_classes, weights)
        for variant in variants}
    if any(value is None for value in per_variant.values()):
        return None
    if condition == "clean":
        return per_variant["0"]
    # Medians of pooled translated predictions would be a different statistic.
    # Compute the stated mean of the four translated-crop condition results.
    result = {"per_variant": per_variant, "condition": "mean_of_r1_r2_r3_r4"}
    scalar_keys = ("macro_IoU", "instance_weighted_IoU", "primary_macro_metric") if task == "mask" else (
        "macro_Acc30", "macro_Acc15", "macro_MedErr", "primary_macro_metric")
    for key in scalar_keys:
        values = [entry[key] for entry in per_variant.values()]
        result[key] = float(np.mean(values)) if all(value is not None for value in values) else None
    result["prediction_count"] = sum(v["prediction_count"] for v in per_variant.values())
    result["failed_predictions"] = sum(v["failed_predictions"] for v in per_variant.values())
    result["macro_class_count"] = min(v["macro_class_count"] for v in per_variant.values())
    result["per_class"] = {}
    metric_keys = ("mean_IoU",) if task == "mask" else ("MedErr", "Acc30", "Acc15")
    for category in range(20 if task == "mask" else 12):
        entries = [v["per_class"][str(category)] for v in per_variant.values()]
        result["per_class"][str(category)] = {"count": sum(v["count"] for v in entries)}
        for key in metric_keys:
            values = [v[key] for v in entries]
            result["per_class"][str(category)][key] = float(np.mean(values)) if all(v is not None for v in values) else None
    return result


def summarize_object_predictions(rows, task):
    task = _task(task)
    subsets = {"primary": rows}
    if task == "pose":
        subsets = {"primary": [r for r in rows if not r["occluded"] and not r["truncated"]],
                   "all": rows, "occluded_or_truncated": [r for r in rows if r["occluded"] or r["truncated"]]}
    results = {name: {condition: _aggregate_condition(subset, task, condition)
                      for condition in ("clean", "translated")} for name, subset in subsets.items()}
    primary = results["primary"]
    return {"task": task, "clean": primary["clean"], "translated": primary["translated"],
            "subsets": results, "primary_macro_metric": primary["clean"]["primary_macro_metric"],
            "evaluation_images": len({r["image_id"] for r in rows}),
            "evaluation_objects": len({(r["image_id"], r["object_id"]) for r in rows}),
            "class_names": list(POSE_CLASSES if task == "pose" else VOC_CLASSES),
            "metric_direction": "higher_is_better"}


def _evaluate_objects(task, head, features, records, eval_indices, scale, output, device,
                      original_provider=None, prediction_function=None, batch_size=128):
    if task == "mask" and original_provider is None:
        from transfer_data import load_mask_original
        original_provider = load_mask_original
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(eval_indices, dtype=np.int64)
    if not len(rows):
        raise ValueError("Evaluation partition has no eligible crop rows")
    artifact_name = "mask_logits.npy" if task == "mask" else "rotations.npy"
    artifact_temp = output / (artifact_name + ".tmp")
    import shutil
    required_bytes = len(rows) * (28 * 28 if task == "mask" else 9) * 4 + (128 << 20)
    if shutil.disk_usage(output).free < required_bytes:
        raise RuntimeError(f"Insufficient space for prediction audit: need {required_bytes:,} bytes")
    artifact = np.lib.format.open_memmap(artifact_temp, mode="w+", dtype=np.float32,
                                         shape=(len(rows), 28, 28) if task == "mask" else (len(rows), 3, 3))
    audit_temp = output / "predictions.jsonl.tmp"
    summaries = []
    started = time.perf_counter()
    with torch.inference_mode(), audit_temp.open("w", encoding="utf-8", buffering=1) as audit:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch_records = [records[int(row)] for row in batch_rows]
            if prediction_function is not None:
                prediction = prediction_function(batch_records)
            else:
                prediction = head(features.batch(batch_rows, device, scale))
            if task == "mask":
                prediction = torch.as_tensor(prediction).detach().cpu().numpy()
                artifact[start:start + len(batch_rows)] = prediction
                for offset, (record, logits) in enumerate(zip(batch_records, prediction)):
                    iou, failed = _mask_iou(logits, record, original_provider)
                    row = {**_prediction_identity(record, task), "mask_iou": iou, "failed": failed,
                           "feature_row": int(batch_rows[offset]), "prediction_row": start + offset,
                           "crop_box": record["crop_box"], "instance_id": record.get("instance_id")}
                    summaries.append(row)
                    audit.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
            else:
                if prediction_function is None:
                    category = torch.tensor([_category(r, task) for r in batch_records], device=device)
                    six = prediction[torch.arange(len(batch_rows), device=device), category]
                    rotations, failed = rotations_from_6d(six.to(torch.float64))
                else:
                    rotations = torch.as_tensor(prediction, dtype=torch.float64, device=device)
                    identity = torch.eye(3, dtype=torch.float64, device=device)
                    failed = (~torch.isfinite(rotations).all(dim=(-1, -2))
                              | ((rotations.transpose(-1, -2) @ rotations - identity).abs().amax(dim=(-1, -2)) > 1e-6)
                              | ((torch.linalg.det(rotations) - 1).abs() > 1e-6))
                target = torch.tensor(np.asarray([r["rotation"] for r in batch_records]), device=device, dtype=torch.float64)
                cosine = ((rotations * target).sum(dim=(-1, -2)) - 1.) / 2.
                errors = torch.rad2deg(torch.acos(cosine.clamp(-1., 1.)))
                errors = torch.where(failed, torch.full_like(errors, 180.), errors)
                saved = rotations.detach().cpu().numpy()
                artifact[start:start + len(batch_rows)] = saved.astype(np.float32)
                for offset, record in enumerate(batch_records):
                    row = {**_prediction_identity(record, task), "error_degrees": float(errors[offset]),
                           "failed": bool(failed[offset]), "feature_row": int(batch_rows[offset]),
                           "prediction_row": start + offset, "rotation_gt": record["rotation"]}
                    summaries.append(row)
                    audit.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
            processed = start + len(batch_rows)
            if processed % 1024 == 0 or processed == len(rows):
                _atomic_json(output / "status.json", {"stage": "evaluation", "task": task,
                                                       "crops_complete": processed, "crops_total": len(rows),
                                                       "elapsed_seconds": time.perf_counter() - started})
                print(f"[{task}/evaluation] {processed}/{len(rows)} crops", flush=True)
    artifact.flush()
    del artifact
    os.replace(artifact_temp, output / artifact_name)
    os.replace(audit_temp, output / "predictions.jsonl")
    return {**summarize_object_predictions(summaries, task),
            "prediction_path": str((output / "predictions.jsonl").resolve()),
            "prediction_array_path": str((output / artifact_name).resolve())}


def _decode_boxes(boxes, offsets, width, height):
    boxes = np.asarray(boxes, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    size = boxes[:, 2:] - boxes[:, :2]
    center = (boxes[:, 2:] + boxes[:, :2]) * .5
    refined_center = center + offsets[:, :2] * size
    refined_size = size * np.exp(np.clip(offsets[:, 2:], -5., 5.))
    decoded = np.c_[refined_center - refined_size * .5, refined_center + refined_size * .5]
    decoded[:, [0, 2]] = np.clip(decoded[:, [0, 2]], 0., width)
    decoded[:, [1, 3]] = np.clip(decoded[:, [1, 3]], 0., height)
    valid = (np.isfinite(decoded).all(axis=1) & (decoded[:, 2] > decoded[:, 0])
             & (decoded[:, 3] > decoded[:, 1]))
    return decoded, valid


def _stable_nms(boxes, scores, proposal_indices, threshold=.3):
    # Per-image proposal pools are small. Stable NumPy ordering avoids CUDA NMS
    # tie differences and has no relation to the bulk retrieval distance path.
    order = np.lexsort((proposal_indices, -scores))
    kept = []
    while len(order):
        first = int(order[0])
        kept.append(first)
        if len(order) == 1:
            break
        overlap = _box_iou(boxes[first:first + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return np.asarray(kept, dtype=np.int64)


def _ground_truth_images(records, eval_indices):
    from transfer_data import load_voc_annotation
    metadata = getattr(records, "metadata", {}) or {}
    images = {}
    # Orchestrators combining fit/eval manifests can attach the original eval
    # manifest metadata as evaluation_metadata to preserve zero-proposal images.
    metadata = getattr(records, "evaluation_metadata", metadata)
    entries = metadata.get("images", [])
    if isinstance(entries, dict):
        entries = list(entries.values())
    for entry in entries:
        if isinstance(entry, dict) and entry.get("annotation_path"):
            annotation = load_voc_annotation(entry["annotation_path"])
            images[str(annotation.get("image_id", entry["image_id"]))] = annotation
    for path in metadata.get("annotation_paths", []):
        annotation = load_voc_annotation(path)
        images[str(annotation["image_id"])] = annotation
    for row in eval_indices:
        record = records[int(row)]
        image_id = str(record["image_id"])
        if image_id not in images:
            images[image_id] = load_voc_annotation(record["annotation_path"])
    if not images:
        raise ValueError("No VOC evaluation annotations")
    return images


def _voc_ap_from_database(connection, images, mode, overlap_threshold):
    per_class = {}
    for class_id in range(1, 21):
        ground_truth = {}
        n_positive = 0
        for image_id, annotation in images.items():
            objects = [o for o in annotation["objects"] if int(o["class_id"]) == class_id]
            boxes = np.asarray([o["box"] for o in objects], dtype=np.float64).reshape(-1, 4)
            difficult = np.asarray([o["difficult"] for o in objects], dtype=np.bool_)
            ground_truth[image_id] = (boxes, difficult, np.zeros(len(objects), dtype=np.bool_))
            n_positive += int((~difficult).sum())
        # 11 thresholds permit a streaming AP calculation; predictions remain
        # disk backed rather than accumulating half a million Python box objects.
        precision_at_recall = np.zeros(11, dtype=np.float64)
        recall_thresholds = np.linspace(0., 1., 11)
        true_positive = false_positive = ignored = 0
        query = ("SELECT image_id,x1,y1,x2,y2 FROM detections WHERE mode=? AND class_id=? "
                 "ORDER BY score DESC,image_id ASC,proposal_index ASC")
        for image_id, x1, y1, x2, y2 in connection.execute(query, (mode, class_id)):
            boxes, difficult, detected = ground_truth[image_id]
            overlap = _box_iou([[x1, y1, x2, y2]], boxes)[0]
            match = int(np.argmax(overlap)) if len(overlap) else -1
            # Original VOC devkit uses strict > at the evaluation threshold.
            if match >= 0 and overlap[match] > overlap_threshold:
                if difficult[match]:
                    ignored += 1
                    continue
                if not detected[match]:
                    detected[match] = True
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                false_positive += 1
            if n_positive:
                recall = true_positive / n_positive
                precision = true_positive / max(1, true_positive + false_positive)
                eligible = recall_thresholds <= recall
                precision_at_recall[eligible] = np.maximum(precision_at_recall[eligible], precision)
        per_class[str(class_id - 1)] = {"AP": float(precision_at_recall.mean()),
                                       "non_difficult_objects": n_positive,
                                       "true_positives": true_positive,
                                       "false_positives": false_positive,
                                       "ignored_difficult_detections": ignored}
    return float(np.mean([v["AP"] for v in per_class.values()])), per_class


def _evaluate_detection(head, features, records, eval_indices, fit_info, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = _ground_truth_images(records, eval_indices)
    rows_by_image = {image_id: [] for image_id in images}
    for row in eval_indices:
        record = records[int(row)]
        if record.get("is_gt", False):
            raise ValueError("Ground-truth proposal crops are forbidden at detection evaluation")
        rows_by_image[str(record["image_id"])].append(int(row))
    database_temp = output / "detections.sqlite.tmp"
    if database_temp.exists():
        database_temp.unlink()  # only this module's incomplete local artifact
    connection = sqlite3.connect(database_temp)
    connection.execute("CREATE TABLE detections (mode TEXT,class_id INTEGER,score REAL,image_id TEXT,"
                       "proposal_index INTEGER,x1 REAL,y1 REAL,x2 REAL,y2 REAL)")
    mean, std = np.asarray(fit_info["box_target_mean"]), np.asarray(fit_info["box_target_std"])
    gain_sum, gain_count, invalid_refined = 0., 0, 0
    recall_hits = {50: 0, 75: 0}
    gt_count = 0
    audit_temp = output / "proposal_diagnostics.jsonl.tmp"
    with torch.inference_mode(), audit_temp.open("w", encoding="utf-8") as audit:
        for image_number, image_id in enumerate(sorted(images), start=1):
            annotation = images[image_id]
            image_rows = rows_by_image[image_id]
            width, height = int(annotation["width"]), int(annotation["height"])
            objects = [o for o in annotation["objects"] if not o["difficult"]]
            gt_boxes = np.asarray([o["box"] for o in objects], dtype=np.float64).reshape(-1, 4)
            gt_count += len(objects)
            if not image_rows:
                audit.write(json.dumps({"image_id": image_id, "proposals": 0,
                                        "non_difficult_objects": len(objects)}) + "\n")
                continue
            probabilities, regression = [], []
            batch_records = [records[row] for row in image_rows]
            for start in range(0, len(image_rows), 256):
                logits, offsets = head(features.batch(image_rows[start:start + 256], device, fit_info["input_scale"]))
                probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
                regression.append(offsets.cpu().numpy())
            probabilities = np.concatenate(probabilities)
            offsets = np.concatenate(regression).astype(np.float64) * std + mean
            original = np.asarray([r["box"] for r in batch_records], dtype=np.float64)
            refined, valid = _decode_boxes(original, offsets, width, height)
            invalid_refined += int((~valid).sum())
            proposal_ids = np.asarray([r.get("proposal_index", i) for i, r in enumerate(batch_records)], dtype=np.int64)
            overlap = _box_iou(original, gt_boxes)
            if len(objects):
                best = overlap.argmax(axis=1)
                before = overlap[np.arange(len(overlap)), best]
                positive = before >= .5
                after = np.zeros(len(original), dtype=np.float64)
                valid_positive = np.flatnonzero(positive & valid)
                if len(valid_positive):
                    after[valid_positive] = _box_iou(refined[valid_positive], gt_boxes)[
                        np.arange(len(valid_positive)), best[valid_positive]]
                gain_sum += float((after[positive] - before[positive]).sum())
                gain_count += int(positive.sum())
                for threshold in recall_hits:
                    recall_hits[threshold] += int((overlap.max(axis=0) >= threshold / 100.).sum())
            else:
                best = np.full(len(original), -1)
                before = after = np.zeros(len(original))
                positive = np.zeros(len(original), dtype=np.bool_)
            audit.write(json.dumps({"image_id": image_id, "proposal_indices": proposal_ids.tolist(),
                                    "matched_nondifficult_object": best.tolist(),
                                    "original_IoU": before.tolist(), "refined_IoU": after.tolist(),
                                    "positive_diagnostic": positive.tolist(),
                                    "refined_valid": valid.tolist()}, allow_nan=False) + "\n")
            for mode, boxes, box_valid in (("refined", refined, valid),
                                           ("original", original, np.isfinite(original).all(axis=1))):
                detections = []
                for class_id in range(1, 21):
                    candidates = np.flatnonzero(box_valid & np.isfinite(probabilities[:, class_id])
                                               & (probabilities[:, class_id] >= .001))
                    if not len(candidates):
                        continue
                    kept = _stable_nms(boxes[candidates], probabilities[candidates, class_id], proposal_ids[candidates])
                    for index in candidates[kept]:
                        detections.append((float(probabilities[index, class_id]), class_id,
                                           int(proposal_ids[index]), boxes[index]))
                detections.sort(key=lambda d: (-d[0], d[1], d[2]))
                connection.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?)", [
                    (mode, category, score, image_id, proposal, *map(float, box))
                    for score, category, proposal, box in detections[:100]])
            connection.commit()
            if image_number % 100 == 0 or image_number == len(images):
                _atomic_json(output / "status.json", {"stage": "detection_evaluation",
                                                       "images_complete": image_number, "images_total": len(images)})
                print(f"[detection/evaluation] {image_number}/{len(images)} images", flush=True)
    connection.execute("CREATE INDEX ranked ON detections(mode,class_id,score DESC,image_id,proposal_index)")
    result = {"task": "detection", "evaluation_images": len(images), "evaluation_objects": gt_count,
              "conditional_IoU_gain": gain_sum / gain_count if gain_count else None,
              "conditional_positive_proposals": gain_count, "invalid_refined_boxes": invalid_refined,
              "proposal_recall": {f"recall{key}": value / gt_count if gt_count else None for key, value in recall_hits.items()},
              "per_class": {}, "AP_rule": "VOC2007_11_point_strict_IoU_match",
              "metric_direction": "higher_is_better"}
    for mode in ("refined", "original"):
        for threshold in (50, 75):
            ap, per_class = _voc_ap_from_database(connection, images, mode, threshold / 100.)
            key = f"AP{threshold}" + ("_no_refinement" if mode == "original" else "")
            result[key] = ap
            result["per_class"][key] = per_class
    connection.commit()
    connection.close()
    os.replace(database_temp, output / "detections.sqlite")
    os.replace(audit_temp, output / "proposal_diagnostics.jsonl")
    result.update(primary_macro_metric=result["AP50"],
                  prediction_path=str((output / "detections.sqlite").resolve()),
                  diagnostic_path=str((output / "proposal_diagnostics.jsonl").resolve()))
    return result


def evaluate_head(task, features, records, eval_indices, run_dir, device="cuda",
                  target_provider=None, checkpoint_step=None):
    """Evaluate one immutable checkpoint and retain auditable per-image output."""
    task = _task(task)
    if not isinstance(features, FeatureStore):
        features = FeatureStore(features)
    run_dir = Path(run_dir)
    checkpoint_name = "last.pt" if checkpoint_step is None else f"checkpoint_{checkpoint_step}.pt"
    output = run_dir / ("evaluation" if checkpoint_step is None else f"evaluation_step{checkpoint_step}")
    checkpoint = _load_checkpoint(run_dir / checkpoint_name, "cpu")
    config, fit_info = checkpoint["config"], checkpoint["fit_info"]
    if config["task"] != task or config["width"] != features.width or config["feature_fingerprint"] != features.fingerprint:
        raise ValueError("Evaluation cache does not match the trained representation")
    evaluation_identity = {"config_hash": _digest(config), "checkpoint_step": checkpoint["step"],
                           "eval_rows_sha256": _array_hash(eval_indices),
                           "records_hash": _digest(_record_identity(records))}
    if (output / "metrics.json").exists() and (output / "identity.json").exists():
        if json.loads((output / "identity.json").read_text(encoding="utf-8")) != evaluation_identity:
            raise ValueError("Existing evaluation has different inputs; use a separate run directory")
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    head = TransferHead(task, features.width, config["head_family"])
    head.load_state_dict(checkpoint["model"])
    del checkpoint
    head = head.to(device=device, dtype=torch.float32).eval()
    _synchronize(device)
    started = time.perf_counter()
    if task == "detection":
        result = _evaluate_detection(head, features, records, eval_indices, fit_info, output, device)
    else:
        result = _evaluate_objects(task, head, features, records, eval_indices, fit_info["input_scale"],
                                   output, device, original_provider=target_provider)
    _synchronize(device)
    result.update(evaluation_time=time.perf_counter() - started, update_count=fit_info["update_count"],
                  head_family=config["head_family"], head_seed=config["seed"],
                  head_parameters=fit_info["head_parameters"], **_peak_memory(device))
    result["supplied_test_information"] = {
        "detection": "RGB image and common annotation-independent EdgeBoxes proposals",
        "mask": "RGB crop from oracle visible-extent object box; category is not a head input",
        "pose": "RGB crop from supplied object box and true category selecting the pose branch"}[task]
    result["evaluation_manifest_fingerprint"] = getattr(records, "fingerprint", None)
    _atomic_json(output / "identity.json", evaluation_identity)
    _atomic_json(output / "metrics.json", result)
    return result


def select_update_count(full_parent_results):
    """Select one shared B/C duration from four seed-0 full-parent MLP runs.

    Entries contain parent_id, update_count, primary_macro_metric. Missing
    scores/parents are an error; a subset cannot silently determine the protocol.
    """
    candidates = (1000, 2500, 5000)
    table = {}
    for entry in full_parent_results:
        if int(entry.get("head_seed", entry.get("seed", 0))) != 0:
            raise ValueError("Schedule selection uses head seed 0 only")
        if entry.get("head_family", "mlp").lower() not in ("mlp", "mlp128", "h1"):
            raise ValueError("Schedule selection uses the primary width-128 MLP")
        step = int(entry["update_count"])
        parent = str(entry["parent_id"])
        score = entry["primary_macro_metric"]
        if step not in candidates or score is None or not math.isfinite(float(score)):
            raise ValueError("Missing/invalid full-parent development score")
        if (parent, step) in table:
            raise ValueError("Duplicate full-parent development result")
        table[(parent, step)] = float(score)
    parents = sorted({parent for parent, _ in table})
    if len(parents) != 4 or any((parent, step) not in table for parent in parents for step in candidates):
        raise ValueError("Schedule selection requires all four full parents at 1000, 2500, 5000 updates")
    scores = {step: float(np.mean([table[(parent, step)] for parent in parents])) for step in candidates}
    return max(candidates, key=lambda step: (scores[step], -step))


def _probability_logits(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1. - 1e-6)
    return (np.log(probability) - np.log1p(-probability)).astype(np.float32)


def _mask_cell_areas(record, target_provider):
    target, valid = target_provider(record)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(1, 1, 224, 224)
    valid = torch.as_tensor(valid, dtype=torch.float64).reshape(1, 1, 224, 224)
    foreground = F.avg_pool2d(target * valid, kernel_size=8)[0, 0].numpy()
    support = F.avg_pool2d(valid, kernel_size=8)[0, 0].numpy()
    return foreground, support


def _fit_mask_templates(records, fit_indices, target_provider=None, cached_targets=None):
    foreground = np.zeros((21, 28, 28), dtype=np.float64)
    support = np.zeros_like(foreground)
    counts = np.zeros(20, dtype=np.int64)
    rows = np.asarray(fit_indices, dtype=np.int64)
    for start in range(0, len(rows), 64):
        selected = rows[start:start + 64]
        batch_records = [records[int(row)] for row in selected]
        if cached_targets is None:
            targets, valid = zip(*(target_provider(record) for record in batch_records))
            targets, valid = torch.stack(list(targets)).float(), torch.stack(list(valid)).float()
        else:
            targets, valid = cached_targets.batch(selected)
        area = F.avg_pool2d((targets * valid)[:, None], kernel_size=8)[:, 0].numpy().astype(np.float64)
        valid = F.avg_pool2d(valid[:, None], kernel_size=8)[:, 0].numpy().astype(np.float64)
        categories = np.asarray([_category(record, "mask") for record in batch_records], dtype=np.int64)
        foreground[0] += area.sum(axis=0)
        support[0] += valid.sum(axis=0)
        np.add.at(foreground, categories + 1, area)
        np.add.at(support, categories + 1, valid)
        np.add.at(counts, categories, 1)
    probability = np.divide(foreground, support, out=np.zeros_like(foreground), where=support > 0)
    return probability, counts


def _fit_rotation_medoids(records, fit_indices, chunk_size=256):
    groups = {category: {} for category in range(12)}
    for row in fit_indices:
        record = records[int(row)]
        key = (str(record["image_id"]), str(record["object_id"]))
        groups[_category(record, "pose")][key] = np.asarray(record["rotation"], dtype=np.float64)
    medoids, provenance = {}, {}
    for category, objects in groups.items():
        if not objects:
            continue
        keys = sorted(objects)
        rotations = np.stack([objects[key] for key in keys]).reshape(-1, 9)
        distances = np.zeros(len(rotations), dtype=np.float64)
        # Frobenius inner product equals trace(R_a^T R_b); no N x N retained matrix.
        for start in range(0, len(rotations), chunk_size):
            sums = np.zeros(min(chunk_size, len(rotations) - start), dtype=np.float64)
            for other in range(0, len(rotations), chunk_size):
                cosine = (rotations[start:start + chunk_size] @ rotations[other:other + chunk_size].T - 1.) * .5
                sums += np.arccos(np.clip(cosine, -1., 1.)).sum(axis=1)
            distances[start:start + len(sums)] = sums / len(rotations)
        selected = int(np.argmin(distances))  # keys already give deterministic ID tie-break
        medoids[category] = rotations[selected].reshape(3, 3)
        provenance[str(category)] = {"object_id": list(keys[selected]), "fit_objects": len(keys),
                                     "mean_training_distance_radians": float(distances[selected])}
    return medoids, provenance


def evaluate_controls(task, records, fit_indices, eval_indices, run_dir, device="cpu"):
    """Fit annotation-only priors on training rows, then score identical crops.

    Resolution-reference masks explicitly use evaluation ground truth. They are
    named diagnostic artifacts, never passed to the learned readout or encoder.
    """
    task = _task(task)
    if task == "detection":
        raise ValueError("Detection controls are scored by evaluate_head with the same classifier")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_id": PROTOCOL_ID, "task": task,
                "fit_rows_sha256": _array_hash(fit_indices), "eval_rows_sha256": _array_hash(eval_indices),
                "records_hash": _digest(_record_identity(records))}
    if (directory / "controls.json").exists() and (directory / "identity.json").exists():
        if json.loads((directory / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Controls were fitted/evaluated with a different manifest")
        return json.loads((directory / "controls.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    results = {}
    if task == "mask":
        packed_targets = CachedMaskTargets(records, directory.parent / "target_cache")
        probabilities, class_counts = _fit_mask_templates(records, fit_indices, cached_targets=packed_targets)
        logits = _probability_logits(probabilities)
        centers = (np.arange(28) + .5) / 28
        central = ((centers[:, None] >= 1 / 12) & (centers[:, None] <= 11 / 12)
                   & (centers[None, :] >= 1 / 12) & (centers[None, :] <= 11 / 12))
        rectangle = _probability_logits(central)
        np.save(directory / "training_templates_probabilities.npy", probabilities, allow_pickle=False)
        _atomic_json(directory / "training_template_counts.json", {"class_crop_counts": class_counts})

        def selected_category(batch):
            categories = [_category(r, task) for r in batch]
            if any(class_counts[category] == 0 for category in categories):
                raise ValueError("True-category template has no fitting examples for an evaluated category")
            return np.stack([logits[category + 1] for category in categories])

        def resolution_reference(batch):
            targets, valid = packed_targets.batch_records(batch)
            foreground = F.avg_pool2d((targets * valid)[:, None], kernel_size=8)[:, 0].numpy()
            support = F.avg_pool2d(valid[:, None], kernel_size=8)[:, 0].numpy()
            probability = np.divide(foreground, support, out=np.zeros_like(foreground), where=support > 0)
            return _probability_logits(probability)

        predictors = {
            "fixed_central_rectangle": lambda batch: np.repeat(rectangle[None], len(batch), axis=0),
            "class_agnostic_mean_mask": lambda batch: np.repeat(logits[0:1], len(batch), axis=0),
            "true_category_mean_mask_privileged": selected_category,
            "ground_truth_28x28_resolution_reference": resolution_reference}
        for name, predictor in predictors.items():
            results[name] = _evaluate_objects(task, None, None, records, eval_indices, 1., directory / name,
                                              torch.device(device), prediction_function=predictor)
            results[name]["control_label"] = name
    else:
        medoids, provenance = _fit_rotation_medoids(records, fit_indices)
        _atomic_json(directory / "training_rotation_medoids.json", {"rotations": medoids, "provenance": provenance})

        def medoid_prediction(batch):
            categories = [_category(r, task) for r in batch]
            if any(category not in medoids for category in categories):
                raise ValueError("Known-category medoid has no training object in an evaluation category")
            return np.stack([medoids[category] for category in categories])

        name = "known_category_training_rotation_medoid"
        results[name] = _evaluate_objects(task, None, None, records, eval_indices, 1., directory / name,
                                          torch.device(device), prediction_function=medoid_prediction)
        results[name]["control_label"] = name
    for result in results.values():
        result["controls_total_time_seconds"] = time.perf_counter() - started
    _atomic_json(directory / "identity.json", identity)
    _atomic_json(directory / "controls.json", results)
    return results


def paired_bootstrap(compact_paths, parent_paths, task, metric=None, subset="primary",
                     condition="clean", replicates=1000, seed=2026):
    """Image-cluster paired interval for the mean of three signed seed deltas.

    All crop variants and objects of an image share its bootstrap multiplicity.
    Replicates missing a class are redrawn with the original class set fixed.
    These are conditional image-sampling intervals, not pretraining uncertainty.
    """
    task = _task(task)
    if task == "detection":
        raise ValueError("This specification defines image bootstrap for mask/pose tasks only")
    if len(compact_paths) != 3 or len(parent_paths) != 3:
        raise ValueError("Paired bootstrap requires exactly three matching head/compressor seeds")
    if condition not in ("clean", "translated") or subset not in ("primary", "all", "occluded_or_truncated"):
        raise ValueError("Invalid bootstrap condition/subset")
    metric = metric or ("macro_IoU" if task == "mask" else "macro_Acc30")
    permitted = {"macro_IoU", "instance_weighted_IoU"} if task == "mask" else {"macro_Acc30", "macro_Acc15", "macro_MedErr"}
    if metric not in permitted:
        raise ValueError("Unsupported task bootstrap metric")

    def read_rows(path):
        with Path(path).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if task == "pose":
            if subset == "primary":
                rows = [r for r in rows if not r["occluded"] and not r["truncated"]]
            elif subset == "occluded_or_truncated":
                rows = [r for r in rows if r["occluded"] or r["truncated"]]
        rows = [r for r in rows if (r["variant"] == 0 if condition == "clean" else r["variant"] in (1, 2, 3, 4))]
        rows.sort(key=lambda r: (r["image_id"], r["object_id"], r["variant"]))
        return rows

    compact, parent = [read_rows(p) for p in compact_paths], [read_rows(p) for p in parent_paths]
    identity = lambda rows: [(r["image_id"], r["object_id"], r["variant"], r["category"]) for r in rows]
    expected = identity(compact[0])
    if not expected or any(identity(rows) != expected for rows in compact + parent):
        raise ValueError("Paired predictions do not have identical objects, categories and crop variants")
    image_ids = sorted({r["image_id"] for r in compact[0]})
    classes = sorted({r["category"] for r in compact[0]})
    image_lookup = {image_id: i for i, image_id in enumerate(image_ids)}
    class_lookup = {category: i for i, category in enumerate(classes)}
    variants = [0] if condition == "clean" else [1, 2, 3, 4]
    variant_lookup = {variant: i for i, variant in enumerate(variants)}
    image_indices = np.asarray([image_lookup[r["image_id"]] for r in compact[0]], dtype=np.int64)
    class_indices = np.asarray([class_lookup[r["category"]] for r in compact[0]], dtype=np.int64)
    variant_indices = np.asarray([variant_lookup[r["variant"]] for r in compact[0]], dtype=np.int64)
    counts = np.zeros((len(variants), len(classes), len(image_ids)), dtype=np.float64)
    np.add.at(counts, (variant_indices, class_indices, image_indices), 1.)
    streams = compact + parent
    sums = None
    median_groups = []
    if metric != "macro_MedErr":
        sums = np.zeros((6, len(variants), len(classes), len(image_ids)), dtype=np.float64)
        for stream_index, rows in enumerate(streams):
            values = np.asarray([r["mask_iou"] if task == "mask" else
                                 float(r["error_degrees"] < (15. if metric == "macro_Acc15" else 30.))
                                 for r in rows], dtype=np.float64)
            np.add.at(sums[stream_index], (variant_indices, class_indices, image_indices), values)
    else:
        # Fixed sorted errors + image multiplicities give exact repeated-sample
        # medians without expanding image/object rows in each bootstrap draw.
        for stream_index, rows in enumerate(streams):
            errors = np.asarray([r["error_degrees"] for r in rows], dtype=np.float64)
            for variant in range(len(variants)):
                for category in range(len(classes)):
                    selected = np.flatnonzero((variant_indices == variant) & (class_indices == category))
                    ordered = selected[np.argsort(errors[selected], kind="stable")]
                    if not len(ordered):
                        raise ValueError("A required category is absent in a translated crop condition")
                    median_groups.append((stream_index, variant, category, errors[ordered], image_indices[ordered]))

    def scores_from_weights(weights):
        # weights: B x source_images. B is capped at 64, retaining bounded RAM.
        denominators = np.tensordot(weights, counts, axes=([1], [2]))  # B,V,C
        if np.any(denominators <= 0):
            raise ValueError("Bootstrap draw lost a required class")
        if sums is not None:
            numerator = np.tensordot(weights, sums, axes=([1], [3]))  # B,6,V,C
            if metric == "instance_weighted_IoU":
                scores = (numerator.sum(axis=3) / denominators.sum(axis=2)[:, None, :]).mean(axis=2)
            else:
                scores = (numerator / denominators[:, None, :, :]).mean(axis=(2, 3))
        else:
            medians = np.empty((len(weights), 6, len(variants), len(classes)), dtype=np.float64)
            for stream_index, variant, category, errors, image_order in median_groups:
                cumulative = np.cumsum(weights[:, image_order], axis=1, dtype=np.int64)
                total = cumulative[:, -1]
                lower = np.argmax(cumulative > ((total - 1) // 2)[:, None], axis=1)
                upper = np.argmax(cumulative > (total // 2)[:, None], axis=1)
                medians[:, stream_index, variant, category] = (errors[lower] + errors[upper]) * .5
            scores = medians.mean(axis=(2, 3))
        return (scores[:, :3] - scores[:, 3:]).mean(axis=1)

    observed_mean = float(scores_from_weights(np.ones((1, len(image_ids)), dtype=np.int64))[0])
    rng = np.random.default_rng(seed)
    deltas, attempts = [], 0
    max_attempts = max(10000, replicates * 1000)
    while len(deltas) < replicates:
        draw_count = min(64, replicates - len(deltas))
        attempts += draw_count
        if attempts > max_attempts:
            raise RuntimeError("Bootstrap cannot retain the declared classes; report subset coverage")
        multiplicities = rng.multinomial(len(image_ids), np.full(len(image_ids), 1 / len(image_ids)), size=draw_count)
        class_counts = np.tensordot(multiplicities, counts, axes=([1], [2]))
        multiplicities = multiplicities[(class_counts > 0).all(axis=(1, 2))]
        if not len(multiplicities):
            continue
        previous_count = len(deltas)
        deltas.extend(scores_from_weights(multiplicities).tolist())
        if len(deltas) // 100 != previous_count // 100 or len(deltas) == replicates:
            print(f"[bootstrap/{task}/{condition}] {len(deltas)}/{replicates}", flush=True)
    low, high = np.percentile(np.asarray(deltas), [2.5, 97.5])
    accuracy_metric = metric != "macro_MedErr"
    scale = 100. if accuracy_metric else 1.
    return {"task": task, "metric": metric, "subset": subset, "condition": condition,
            "mean_signed_difference": observed_mean, "percentile_95_interval": [float(low), float(high)],
            "display_mean_difference": observed_mean * scale,
            "display_percentile_95_interval": [float(low) * scale, float(high) * scale],
            "display_units": "percentage_points" if accuracy_metric else "degrees",
            "metric_direction": "higher_is_better" if accuracy_metric else "lower_is_better",
            "replicates": replicates, "redrawn_replicates": attempts - replicates,
            "seed": seed, "required_class_count": len(classes), "image_count": len(image_ids),
            "uncertainty_label": "paired_image_sampling_conditional_on_three_fitted_model_pairs"}
