"""CUDA endpoint evaluation for the separate MPSAE repair research profile.

Imports do not build an extension or start a benchmark. Compact encoding caches
are checkpoint/source/row keyed and resumable. Dense dictionary-width gallery
codes are never materialized. Cheap screening uses one exhaustive search;
the optional finalist benchmark uses the existing resident-search timer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from repair_data import atomic_json, canonical_hash, exclusive_lock, sha256_file
from mpsae_repair import encoding_start


def preflight(device, config):
    """Explicit backend/toolchain check before the first paid training stage."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Research production evaluation requires CUDA; no CPU fallback")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        raise ValueError("Selected CUDA device is not visible")
    import faiss
    import faiss.contrib.torch_utils
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("The production benchmark requires CUDA-enabled FAISS")
    from sparse_gpu_retrieval import preload_sparse_retrieval
    with torch.cuda.device(device):
        resources = faiss.StandardGpuResources()
        if not hasattr(resources, "setDefaultStream"):
            raise RuntimeError("FAISS must support the selected PyTorch CUDA stream")
        # PyTorch exposes an integer handle; FAISS's SWIG API needs cudaStream_t.
        stream = faiss.cast_integer_to_cudastream_t(torch.cuda.current_stream(device).cuda_stream)
        resources.setDefaultStream(device.index, stream)
        resources.setTempMemory(int(config["retrieval"].get("faiss_temp_memory_mib", 512)) * 2**20)
        preload_sparse_retrieval(device)
    return {"device": str(device), "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "faiss_version": getattr(faiss, "__version__", "unavailable"),
            "dense_backend": "faiss_gpu_flat_l2", "sparse_backend": "cuda_inverted_index"}


def _ids(values, length, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{name} must contain nonempty integer row IDs")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= length) or len(np.unique(result)) != len(result):
        raise ValueError(f"Duplicate or out-of-range {name}")
    return result


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _code_hash():
    digest = hashlib.sha256()
    for filename in ("repair_benchmark.py", "mpsae_repair.py", "mmrsae_dictionary.py",
                     "sparse_gpu_retrieval.py", "retrieval_timing.py",
                     "csrc/sparse_retrieval.cpp", "csrc/sparse_retrieval.cu"):
        path = Path(__file__).resolve().parent / filename
        digest.update(filename.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tensor_batch(features, rows, device):
    # Advanced indexing copies only the requested batch from a read-only map.
    values = np.asarray(features[rows], dtype=np.float32)
    x = torch.as_tensor(values, device=device, dtype=torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite input features during benchmark")
    return x


def _flush_maps(maps, directory):
    for name, array in maps.items():
        array.flush()
        with open(directory / f"{name}.npy", "r+b") as stream:
            os.fsync(stream.fileno())


@torch.no_grad()
def _encoded_cache(features, row_ids, sae, repair, config, cache_root, identity, role, device):
    sparse = sae is not None
    width = min(int(config["model"]["k"]), int(sae.decoder.shape[0])) if sparse else int(features.shape[1])
    encode_batch = int(config["retrieval"].get("encode_batch_size", 1024))
    if encode_batch < 1:
        raise ValueError("encode_batch_size must be positive")
    # Empty sparse codes stay zero rows (never retrieved; empty queries count as
    # misses) only when explicitly allowed; otherwise the benchmark aborts.
    allow_empty = bool(config["retrieval"].get("allow_empty_codes", False)) and sparse
    cache_identity = identity
    if role == "training_gallery" and isinstance(identity, dict):
        # Query pool/source does not change an already encoded training gallery.
        cache_identity = {key: value for key, value in identity.items() if key not in ("pool", "query_source")}
    specification = {
        "version": 1, "identity": cache_identity, "role": role,
        "rows_hash": _array_hash(row_ids), "rows": len(row_ids),
        "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
        "width": width, "sparse": sparse, "repair": repair is not None,
        "landmark_synthesis": getattr(sae, "dictionary", None) is not None,
        "inference_code_hash": _code_hash(), "normalization": "row_l2_float32",
        "empty_codes": "kept_zero" if allow_empty else "rejected",
        "encoding_runtime": {"device": str(device), "device_name": torch.cuda.get_device_name(device),
                             "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                             "amp": False, "tf32": False},
    }
    directory = Path(cache_root) / canonical_hash(specification)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    reused_rows = 0
    with exclusive_lock(directory / "writer.lock"):
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
            if state["specification"] != specification:
                raise ValueError("Encoded cache identity mismatch")
            reused_rows = int(state["completed_rows"])
            if not 0 <= reused_rows <= len(row_ids):
                raise ValueError("Invalid encoded-cache cursor")
            mode = "r+" if reused_rows < len(row_ids) else "r"
            maps = {"values": np.load(directory / "values.npy", mmap_mode=mode, allow_pickle=False)}
            if sparse:
                maps.update({name: np.load(directory / f"{name}.npy", mmap_mode=mode, allow_pickle=False)
                             for name in ("indices", "counts")})
            for name, previous in state.get("array_fingerprints", {}).items():
                path = directory / f"{name}.npy"
                stat = path.stat()
                if stat.st_size != previous["bytes"]:
                    raise ValueError("Encoded-cache array length changed")
                if stat.st_mtime_ns != previous["mtime_ns"] and sha256_file(path) != previous["sha256"]:
                    raise ValueError("Encoded-cache array checksum changed")
        else:
            if any(directory.glob("*.npy")):
                raise ValueError(f"Cache has arrays without a durable manifest: {directory}")
            maps = {"values": np.lib.format.open_memmap(directory / "values.npy", mode="w+",
                                                        dtype=np.float32, shape=(len(row_ids), width))}
            if sparse:
                maps["indices"] = np.lib.format.open_memmap(directory / "indices.npy", mode="w+",
                                                            dtype=np.int32, shape=(len(row_ids), width))
                maps["counts"] = np.lib.format.open_memmap(directory / "counts.npy", mode="w+",
                                                           dtype=np.uint32, shape=(len(row_ids),))
            state = {"specification": specification, "completed_rows": 0,
                     "encoding_wall_seconds": 0., "initial_encoding_seconds": 0.,
                     "correction_seconds": 0., "encoding_cuda_seconds": 0.,
                     "nonzero_sum": 0, "zero_final_codes": 0}
            _flush_maps(maps, directory)
            atomic_json(manifest_path, state)
        if maps["values"].shape != (len(row_ids), width) or maps["values"].dtype != np.float32:
            raise ValueError("Encoded values cache has the wrong shape or dtype")
        if sparse and (maps["indices"].shape != maps["values"].shape or maps["indices"].dtype != np.int32
                       or maps["counts"].shape != (len(row_ids),) or maps["counts"].dtype != np.uint32):
            raise ValueError("Encoded sparse index/count cache has the wrong format")
        start_event = torch.cuda.Event(enable_timing=True)
        initial_event = torch.cuda.Event(enable_timing=True)
        final_event = torch.cuda.Event(enable_timing=True)
        for first in range(reused_rows, len(row_ids), encode_batch):
            last = min(first + encode_batch, len(row_ids))
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            x = _tensor_batch(features, row_ids[first:last], device)
            start_event.record()
            initial = x if not sparse else encoding_start(sae, repair, x, width)
            initial_event.record()
            final = initial if repair is None else repair(sae, x, initial, width)
            final_event.record()
            if sparse:
                if not bool(torch.isfinite(final).all()) or bool((final < 0).any()) or bool(final.gt(0).sum(1).gt(width).any()):
                    raise ValueError("Final inference violates the finite nonnegative at-most-K code contract")
                values, indices = final.topk(width, dim=1, sorted=False)
                norms = values.norm(dim=1, keepdim=True)
                empty = norms.le(0).reshape(-1)
                if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or (
                        not allow_empty and bool(empty.any())):
                    raise ValueError("Invalid or zero-norm final sparse code; benchmark aborted")
                values = values / norms.clamp_min(torch.finfo(values.dtype).tiny)
                state["zero_final_codes"] += int(empty.sum())
                maps["values"][first:last] = values.cpu().numpy()
                maps["indices"][first:last] = indices.to(torch.int32).cpu().numpy()
                counts = values.gt(0).sum(1).cpu().numpy()
                maps["counts"][first:last] = counts
                state["nonzero_sum"] += int(counts.sum())
            else:
                norms = final.norm(dim=1, keepdim=True)
                if not bool(torch.isfinite(final).all()) or bool(norms.le(0).any()):
                    raise ValueError("Invalid or zero-norm protected parent feature")
                maps["values"][first:last] = (final / norms).cpu().numpy()
            final_event.synchronize()
            state["initial_encoding_seconds"] += start_event.elapsed_time(initial_event) / 1000.
            state["correction_seconds"] += initial_event.elapsed_time(final_event) / 1000.
            state["encoding_cuda_seconds"] += start_event.elapsed_time(final_event) / 1000.
            state["encoding_wall_seconds"] += time.perf_counter() - wall_start
            _flush_maps(maps, directory)
            state["completed_rows"] = last
            atomic_json(manifest_path, state)
        if "array_fingerprints" not in state:
            state["array_fingerprints"] = {
                name: {"bytes": (directory / f"{name}.npy").stat().st_size,
                       "mtime_ns": (directory / f"{name}.npy").stat().st_mtime_ns,
                       "sha256": sha256_file(directory / f"{name}.npy")}
                for name in maps}
            atomic_json(manifest_path, state)
        state["actual_directory_bytes"] = sum(path.stat().st_size for path in directory.iterdir()
                                               if path.is_file() and path.name != "writer.lock")
    # Never expose writable mappings to a caller preparing the retrieval index.
    readonly = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in maps}
    return readonly, dict(state, directory=str(directory), reused_rows=reused_rows,
                          current_invocation_encoded_rows=len(row_ids) - reused_rows)


@torch.no_grad()
def _resident_encoding_measurement(features, ids, sae, repair, k, batch_size, device, timed,
                                   allow_empty=False):
    if sae is None:
        return {"scope": "protected_cached_full_features", "encoding_seconds_per_example": 0.0,
                "encoder_extraction_measured": False}
    x = _tensor_batch(features, ids[:batch_size], device)
    warmup, repeats = (3, 5) if timed else (1, 1)
    def encode():
        initial = encoding_start(sae, repair, x, k)
        return initial if repair is None else repair(sae, x, initial, k)
    for _ in range(warmup):
        encode()
    torch.cuda.synchronize(device)
    records = []
    for _ in range(repeats):
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        start.record()
        initial = encoding_start(sae, repair, x, k)
        middle.record()
        final = initial if repair is None else repair(sae, x, initial, k)
        end.record()
        end.synchronize()
        records.append({"initial_seconds": start.elapsed_time(middle) / 1000.,
                        "correction_seconds": middle.elapsed_time(end) / 1000.,
                        "total_seconds": start.elapsed_time(end) / 1000.,
                        "wall_seconds": time.perf_counter() - wall_start})
        if not bool(torch.isfinite(final).all()) or (not allow_empty and bool(final.norm(dim=1).le(0).any())):
            raise ValueError("Invalid final code in encoding timing")
    median = statistics.median(row["total_seconds"] for row in records)
    return {"scope": "resident_feature_to_final_code_excludes_transfer_normalization_storage",
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "batch_size": len(x), "warmup": warmup, "repeats": repeats,
            "total_seconds": median, "encoding_seconds_per_example": median / len(x),
            "initial_seconds": statistics.median(row["initial_seconds"] for row in records),
            "correction_seconds": statistics.median(row["correction_seconds"] for row in records),
            "raw_records": records, "screening_only": not timed}


def _numpy_rows(array, start, stop, device, dtype):
    # Copy avoids handing PyTorch a writable view of a read-only mapped array.
    return torch.as_tensor(np.array(array[start:stop], copy=True), device=device, dtype=dtype).contiguous()


def _parameter_bytes(module):
    return 0 if module is None else sum(p.numel() * p.element_size() for p in module.parameters())


def _buffer_bytes(module):
    """Resident buffers, including derived keys not persisted in checkpoints."""
    return 0 if module is None else sum(b.numel() * b.element_size() for b in module.buffers())


@torch.no_grad()
def benchmark(features, labels, gallery_ids, query_ids, sae, repair, config,
              output_dir, seed, *, identity, device, query_features=None,
              query_labels=None, timed=False):
    """Evaluate one named query pool with exact normalized CUDA top-1 search.

    ``identity`` must identify the checkpoint and frozen training source; when
    query_features is supplied it must also identify that validation source.
    Use the same identity across cheap/holdout pools to share gallery encoding.
    Output directories identify pools (for example ``N/dev_cheap``). Parents
    use ``sae=None, repair=None`` and keep their full stored representation.
    """
    if not identity:
        raise ValueError("Benchmark requires explicit checkpoint/source identity")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production research retrieval requires CUDA; no CPU fallback")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    if sae is None and repair is not None:
        raise ValueError("A repair module requires its matching SAE")
    if len(labels) != len(features):
        raise ValueError("Training labels and features disagree")
    separate_queries = query_features is not None
    query_features = features if query_features is None else query_features
    query_labels = labels if query_labels is None else query_labels
    if len(query_labels) != len(query_features) or features.shape[1] != query_features.shape[1]:
        raise ValueError("Query source does not match its labels or parent feature dimension")
    gallery_ids = _ids(gallery_ids, len(features), "gallery IDs")
    query_ids = _ids(query_ids, len(query_features), "query IDs")
    if not separate_queries and np.intersect1d(gallery_ids, query_ids).size:
        raise ValueError("Pilot query IDs overlap the gallery")
    faiss_version = None
    if sae is None:
        import faiss
        faiss_version = getattr(faiss, "__version__", "unavailable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_identity = {"code_hash": _code_hash(), "retrieval": config["retrieval"],
                            "device": str(device), "device_name": torch.cuda.get_device_name(device),
                            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                            "faiss_version": faiss_version,
                            "gallery_ids_sha256": _array_hash(gallery_ids),
                            "query_ids_sha256": _array_hash(query_ids), "seed": int(seed),
                            "sparse": sae is not None, "repair": repair is not None,
                            "landmark_synthesis": getattr(sae, "dictionary", None) is not None,
                            "k": config["model"]["k"] if sae is not None else int(features.shape[1])}
    if (output_dir / "metrics.json").exists():
        existing = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError("Existing benchmark output belongs to a different checkpoint/source")
        if existing.get("measurement_identity") == measurement_identity and (existing.get("timed") or not timed):
            prediction_file = output_dir / "predictions.csv"
            if not prediction_file.exists() or sha256_file(prediction_file) != existing.get("prediction_sha256"):
                raise ValueError("Completed prediction artifact is missing or has changed")
            for name, wanted in (("gallery_ids", gallery_ids), ("query_ids", query_ids)):
                path = output_dir / f"{name}.npy"
                if not path.exists() or not np.array_equal(np.load(path, allow_pickle=False), wanted):
                    raise ValueError("Completed benchmark row identities have changed")
            return dict(existing, reused_completed_evaluation=True)
    settings = config["retrieval"]
    query_batch = int(settings.get("query_batch_size", 32))
    if query_batch < 1:
        raise ValueError("query_batch_size must be positive")
    cache_root = output_dir.parent / "encoded_cache"
    sparse = sae is not None
    k = int(config["model"]["k"]) if sparse else features.shape[1]
    modes = []
    for module in (sae, repair):
        if module is not None:
            if any(parameter.device != device for parameter in module.parameters()):
                raise ValueError("Model must already reside on the selected retrieval GPU")
            if any(parameter.dtype != torch.float32 for parameter in module.parameters()):
                raise ValueError("Research inference requires float32 model parameters")
            modes.append((module, module.training))
            module.eval()
    prior_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.cuda.device(device), torch.autocast(device_type="cuda", enabled=False), exclusive_lock(output_dir / "benchmark.lock"):
            gallery, gallery_encoding = _encoded_cache(features, gallery_ids, sae, repair, config, cache_root,
                                                        identity, "training_gallery", device)
            queries, query_encoding = _encoded_cache(query_features, query_ids, sae, repair, config, cache_root,
                                                       identity, "validation_queries" if separate_queries else "training_queries", device)
            np.save(output_dir / "gallery_ids.npy", gallery_ids, allow_pickle=False)
            np.save(output_dir / "query_ids.npy", query_ids, allow_pickle=False)
            allow_empty = bool(settings.get("allow_empty_codes", False)) and sparse
            encoding = _resident_encoding_measurement(features, gallery_ids, sae, repair, k,
                                                       int(settings.get("encode_batch_size", 1024)), device, timed,
                                                       allow_empty=allow_empty)
            # Compilation is a separate setup operation, never part of search.
            setup_start = time.perf_counter()
            if sparse:
                from sparse_gpu_retrieval import CudaSparseIndex, preload_sparse_retrieval
                preload_sparse_retrieval(device)
            else:
                import faiss
                import faiss.contrib.torch_utils  # enables resident tensor search
                if not hasattr(faiss, "StandardGpuResources"):
                    raise RuntimeError("Dense parent evaluation needs CUDA-enabled FAISS")
            backend_setup_seconds = time.perf_counter() - setup_start
            torch.cuda.synchronize(device)
            build_start = time.perf_counter()
            if sparse:
                width = int(gallery["values"].shape[1])
                score_bytes = int(settings.get("score_buffer_mib", 256)) * 2**20
                index = CudaSparseIndex(int(sae.decoder.shape[0]), device, max_score_buffer_bytes=score_bytes,
                                        memory_fraction=float(settings.get("gpu_memory_fraction", .8)))
                chunk_rows = min(int(settings.get("gallery_chunk_size", 65536)),
                                 (4 * 2**20) // width, score_bytes // (4 * query_batch))
                if chunk_rows < 1:
                    raise ValueError("Sparse index workspace cannot hold one row")
                for first in range(0, len(gallery_ids), chunk_rows):
                    last = min(first + chunk_rows, len(gallery_ids))
                    index.add(_numpy_rows(gallery["indices"], first, last, device, torch.int32),
                              _numpy_rows(gallery["values"], first, last, device, torch.float32))
                index.prepare(query_batch)
                index_info = {"backend": "cuda_inverted_index", "index_bytes": index.payload_bytes,
                              "workspace_bytes": index.workspace_bytes, "chunk_count": index.chunk_count,
                              "maximum_chunk_rows": index.max_chunk_rows, "nonzeros": index.nnz}
            else:
                resources = faiss.StandardGpuResources()
                from sparse_gpu_retrieval import check_cuda_memory_budget
                check_cuda_memory_budget(
                    device, 4 * len(gallery_ids) * features.shape[1]
                    + int(settings.get("faiss_temp_memory_mib", 512)) * 2**20,
                    float(settings.get("gpu_memory_fraction", .8)), "resident dense parent gallery")
                resources.setTempMemory(int(settings.get("faiss_temp_memory_mib", 512)) * 2**20)
                if hasattr(resources, "setDefaultStream"):
                    stream = faiss.cast_integer_to_cudastream_t(torch.cuda.current_stream(device).cuda_stream)
                    resources.setDefaultStream(device.index, stream)
                else:
                    raise RuntimeError("FAISS build cannot bind the current CUDA stream for valid event timing")
                index = faiss.index_cpu_to_gpu(resources, device.index, faiss.IndexFlatL2(features.shape[1]))
                for first in range(0, len(gallery_ids), 65536):
                    index.add(_numpy_rows(gallery["values"], first, first + 65536, device, torch.float32))
                index_info = {"backend": "faiss_gpu_flat_l2", "index_bytes": 4 * len(gallery_ids) * features.shape[1],
                              "workspace_bytes": int(settings.get("faiss_temp_memory_mib", 512)) * 2**20,
                              "index_bytes_scope": "vector_payload_excludes_faiss_internal_overhead"}
            torch.cuda.synchronize(device)
            build_seconds = time.perf_counter() - build_start
            timer = None
            if timed:
                from retrieval_timing import CudaSearchTimer
                timer = CudaSearchTimer(device, int(settings.get("warmup", 3)), int(settings.get("repeats", 5)))
            predictions_path = output_dir / "predictions.csv"
            temporary_predictions = output_dir / "predictions.csv.tmp"
            correct = 0
            empty_queries = 0
            products = 0
            nearest_distance_sum = 0.
            screening_search_seconds = 0.
            with temporary_predictions.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "query_id", "label", "prediction", "neighbor_id"])
                writer.writeheader()
                for first in range(0, len(query_ids), query_batch):
                    last = min(first + query_batch, len(query_ids))
                    values = _numpy_rows(queries["values"], first, last, device, torch.float32)
                    empty = np.zeros(last - first, dtype=bool)
                    if sparse:
                        indices = _numpy_rows(queries["indices"], first, last, device, torch.int32)
                        empty = values.norm(dim=1).le(0).cpu().numpy()
                        if not bool(torch.isfinite(values).all()) or (not allow_empty and empty.any()):
                            raise ValueError("Invalid cached sparse query")
                        products += index.query_product_count(indices, values)
                        search = lambda: index.search(indices, values)
                    else:
                        search = lambda: index.search(values, 1)
                    if timer is None:
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        result = search()
                        torch.cuda.synchronize(device)
                        screening_search_seconds += time.perf_counter() - start
                    else:
                        result = timer.measure(search, (last - first, values.shape[1]), last - first)
                    raw_distances, neighbor_positions = result
                    neighbor_positions = neighbor_positions.reshape(-1).cpu().numpy().astype(np.int64)
                    if np.any(neighbor_positions < 0) or np.any(neighbor_positions >= len(gallery_ids)):
                        raise RuntimeError("Retrieval backend returned an invalid gallery position")
                    distances = raw_distances.reshape(-1).cpu().numpy()
                    if not np.isfinite(distances).all():
                        raise RuntimeError("Nonfinite nearest-neighbor score")
                    distances = np.maximum(2. - 2. * distances if sparse else distances, 0.)
                    nearest_distance_sum += float(distances.sum(dtype=np.float64))
                    neighbors = gallery_ids[neighbor_positions]
                    predicted = np.asarray(labels)[neighbors].reshape(-1)
                    # An empty query scores zero against every gallery row, so its
                    # backend argmax is arbitrary: record it as an explicit miss.
                    neighbors = np.where(empty, -1, neighbors)
                    predicted = np.where(empty, -1, predicted)
                    empty_queries += int(empty.sum())
                    true = np.asarray(query_labels)[query_ids[first:last]].reshape(-1)
                    correct += int(np.count_nonzero(predicted == true))
                    for row_id, target, prediction, neighbor in zip(query_ids[first:last], true, predicted, neighbors):
                        writer.writerow({"seed": int(seed), "query_id": int(row_id), "label": int(target),
                                         "prediction": int(prediction), "neighbor_id": int(neighbor)})
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_predictions, predictions_path)
            dictionary = getattr(sae, "dictionary", None)
            parameter_bytes = _parameter_bytes(sae) + _parameter_bytes(repair)
            buffer_bytes = _buffer_bytes(sae) + _buffer_bytes(repair)
            metrics = {
                "protocol": config["protocol"], "identity": identity, "seed": int(seed),
                "measurement_identity": measurement_identity,
                "top1": 100. * correct / len(query_ids), "correct": correct, "total": len(query_ids),
                "gallery_count": len(gallery_ids), "query_count": len(query_ids),
                "gallery_ids_sha256": _array_hash(gallery_ids), "query_ids_sha256": _array_hash(query_ids),
                "predictions": str(predictions_path), "K": int(k), "device": str(device),
                "prediction_sha256": sha256_file(predictions_path),
                "device_name": torch.cuda.get_device_name(device), "dtype": "float32",
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "faiss_version": faiss_version,
                "query_batch_size": query_batch, "timed": bool(timed),
                "mean_nearest_neighbor_squared_l2": nearest_distance_sum / len(query_ids),
                "gallery_encoding": gallery_encoding, "query_encoding": query_encoding,
                "encoding_benchmark": encoding,
                "encoding_seconds_per_example": encoding["encoding_seconds_per_example"],
                "encoding_seconds_per_row": encoding["encoding_seconds_per_example"],
                "backend_setup_seconds": backend_setup_seconds, "index_build_seconds": build_seconds,
                "index": index_info, "index_bytes": index_info["index_bytes"],
                "gallery_label_bytes": int(np.asarray(labels)[gallery_ids].nbytes),
                "model_parameter_bytes": parameter_bytes,
                "model_buffer_bytes": buffer_bytes,
                "model_tensor_bytes": parameter_bytes + buffer_bytes,
                "model_memory_scope": "parameters and resident buffers; excludes gradients and optimizer state",
                "landmark_synthesis": dictionary is not None,
                "landmark_dictionary_anchors": 0 if dictionary is None else len(dictionary.features),
                "landmark_dictionary_buffer_bytes": _buffer_bytes(dictionary),
                "landmark_dictionary_parameter_bytes": _parameter_bytes(dictionary),
                "sae_parameters": 0 if sae is None else sum(p.numel() for p in sae.parameters()),
                "repair_parameters": 0 if repair is None else sum(p.numel() for p in repair.parameters()),
                "protected_full_path": sae is None,
                "encoder_extraction_measured": False,
            }
            if sparse:
                metrics.update({"empty_codes_policy": "kept_zero_query_counted_as_miss" if allow_empty else "rejected",
                                "empty_gallery_codes": int(gallery_encoding["zero_final_codes"]),
                                "empty_query_codes": empty_queries,
                                "empty_query_fraction": empty_queries / len(query_ids),
                                "actual_nnz_mean": gallery_encoding["nonzero_sum"] / len(gallery_ids),
                                "sparse_padded_payload_bytes_per_row": 8 * k + 4,
                                "sparse_product_count": products,
                                "sparse_product_fraction_of_dense_dictionary": products / (len(query_ids) * len(gallery_ids) * int(sae.decoder.shape[0]))})
            if timer is None:
                metrics["screening_search_wall_seconds"] = screening_search_seconds
                metrics["search_measurement_scope"] = "single_accuracy_pass_no_warm_repeated_performance_claim"
            else:
                metrics.update(timer.summary(len(query_ids)))
            atomic_json(output_dir / "metrics.json", metrics)
            return metrics
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prior_tf32_cudnn
        for module, training in modes:
            module.train(training)
