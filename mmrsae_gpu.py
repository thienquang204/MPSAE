"""Fail-fast GPU preflight for the manually configured ImageNet experiment.

Importing this module performs no compilation, GPU work or installation. The
runner explicitly calls ``preflight`` before downloading data or fitting a
model. Search failure is fatal; there is no CPU or dense-sparse fallback.
"""
from __future__ import annotations

import math

import torch


def _normalized_rows(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 2 or values.dtype != torch.float32 or not values.is_cuda:
        raise ValueError(f"{name} must be a float32 CUDA matrix")
    norms = values.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(norms).all()) or bool(norms.le(0).any()):
        raise ValueError(f"{name} has a nonfinite or zero-norm row")
    if bool(values.lt(0).any()):
        raise ValueError(f"{name} must contain nonnegative sparse values")
    return (values / norms).contiguous()


def _rejects(operation, description):
    try:
        operation()
    except (ValueError, TypeError):
        return
    raise RuntimeError(f"GPU retrieval preflight did not reject {description}")


@torch.no_grad()
def _retrieval_probe(device, settings):
    try:
        import faiss
        import faiss.contrib.torch_utils  # resident tensor add/search, no host copies
    except ImportError as exc:
        raise RuntimeError("MMRSAE requires CUDA-enabled FAISS and its PyTorch bridge. "
                           "Install the repository's GPU environment before starting the run.") from exc
    if not all(hasattr(faiss, name) for name in
               ("StandardGpuResources", "index_cpu_to_gpu", "GpuClonerOptions")):
        raise RuntimeError("The installed FAISS has no GPU backend; CPU FAISS is not supported")
    from sparse_gpu_retrieval import CudaSparseIndex, preload_sparse_retrieval

    preload_sparse_retrieval(device)
    gallery = _normalized_rows(torch.tensor([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [.6, .8, 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., .6, .8, 0., 0., 0., 0., 0.],
    ], device=device, dtype=torch.float32), "probe gallery")
    queries = _normalized_rows(torch.tensor([
        [.8, .6, 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0.],  # no gallery support overlap
    ], device=device, dtype=torch.float32), "probe queries")
    expected_scores, expected_ids = (queries @ gallery.T).max(dim=1)
    expected_distances = (2. - 2. * expected_scores).clamp_min(0.)

    resources = faiss.StandardGpuResources()
    if not hasattr(resources, "setDefaultStream"):
        raise RuntimeError("FAISS must support the current PyTorch CUDA stream")
    # PyTorch exposes an integer handle; FAISS's SWIG API needs cudaStream_t.
    stream = faiss.cast_integer_to_cudastream_t(torch.cuda.current_stream(device).cuda_stream)
    resources.setDefaultStream(device.index, stream)
    resources.setTempMemory(int(settings.get("faiss_temp_memory_mib", 512)) * 2**20)
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    dense = faiss.index_cpu_to_gpu(resources, device.index, faiss.IndexFlatL2(8), options)
    dense.add(gallery)
    distances, dense_ids = dense.search(queries, 1)
    torch.cuda.synchronize(device)
    if not isinstance(distances, torch.Tensor) or not isinstance(dense_ids, torch.Tensor):
        raise RuntimeError("FAISS did not return resident PyTorch tensors")
    if distances.device != device or dense_ids.device != device or distances.dtype != torch.float32:
        raise RuntimeError("FAISS search did not execute on the requested GPU in float32")
    if not bool(torch.isfinite(distances).all()) or not torch.allclose(
            distances[:, 0], expected_distances, atol=2e-5, rtol=2e-5):
        raise RuntimeError("FAISS GPU search failed the known-distance preflight")
    if not torch.equal(dense_ids[:3, 0], expected_ids[:3]):
        raise RuntimeError("FAISS GPU search returned an incorrect unique nearest neighbor")
    if bool(dense_ids.lt(0).any()) or bool(dense_ids.ge(len(gallery)).any()):
        raise RuntimeError("FAISS GPU search returned an invalid gallery row ID")

    values, indices = gallery.topk(2, dim=1, sorted=False)
    indices = indices.to(torch.int32).contiguous()
    values = values.contiguous()
    query_values, query_indices = queries.topk(2, dim=1, sorted=False)
    query_indices = query_indices.to(torch.int32).contiguous()
    query_values = query_values.contiguous()
    sparse = CudaSparseIndex(8, device, max_score_buffer_bytes=1024 * 1024,
                             memory_fraction=float(settings.get("gpu_memory_fraction", .8)))
    # Two chunks exercise winner merging and the no-overlap candidate path.
    sparse.add(indices[:2].contiguous(), values[:2].contiguous())
    sparse.add(indices[2:].contiguous(), values[2:].contiguous())
    sparse.prepare(len(queries))
    for batch_size in (len(queries), 1):
        scores, found_ids = sparse.search(query_indices[:batch_size].contiguous(),
                                         query_values[:batch_size].contiguous())
        torch.cuda.synchronize(device)
        if scores.device != device or found_ids.device != device or scores.dtype != torch.float32:
            raise RuntimeError("Sparse search returned output on the wrong device or at the wrong precision")
        if not bool(torch.isfinite(scores).all()) or not torch.allclose(
                scores, expected_scores[:batch_size], atol=2e-5, rtol=2e-5):
            raise RuntimeError("Sparse CUDA search failed the known-score preflight")
        if not torch.equal(found_ids, expected_ids[:batch_size]):
            raise RuntimeError("Sparse CUDA search failed nearest-neighbor/chunk/no-overlap handling")
    # Error guards are outside search timing. A malformed index must be stopped
    # before launching a kernel; zero vectors must not be silently normalized.
    _rejects(lambda: _normalized_rows(torch.zeros_like(queries[:1]), "zero probe"), "zero-norm codes")
    _rejects(lambda: sparse.search(query_indices.to(torch.int64), query_values), "non-int32 coordinates")
    invalid_index = CudaSparseIndex(8, device, max_score_buffer_bytes=1024 * 1024,
                                    memory_fraction=float(settings.get("gpu_memory_fraction", .8)))
    outside = indices[:1].clone()
    outside[0, 0] = 8
    _rejects(lambda: invalid_index.add(outside, values[:1].contiguous()), "out-of-range gallery coordinates")
    torch.cuda.synchronize(device)
    return {"faiss_version": getattr(faiss, "__version__", "unavailable"),
            "dense_backend": "faiss_gpu_flat_l2", "sparse_backend": "cuda_inverted_index",
            "gpu_search_probe": "passed", "probe_gallery_rows": len(gallery),
            "probe_query_rows": len(queries), "probe_sparse_chunks": sparse.chunk_count,
            "probe_checks": ["float32_gpu_outputs", "known_scores", "unique_neighbors",
                             "chunk_merging", "nonoverlap", "partial_query_batch",
                             "zero_vector_rejection", "coordinate_dtype", "coordinate_range"]}


def preflight(device, config, *, gallery_rows_estimate=0, input_dim=2048):
    """Check actual GPU search before data access; return provenance and estimates.

    ``config`` accepts the same ``model``/``retrieval`` dictionaries as the
    shared benchmark. Full dense gallery capacity is checked only when
    ``dense_reference`` is true. Estimates are conservative bounds for the
    specified index, not a guarantee against concurrent allocations or an
    estimate of measured runtime.
    """
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MMRSAE ImageNet requires a working CUDA GPU; CPU fallback is disabled")
    target = torch.device("cuda", target.index if target.index is not None else torch.cuda.current_device())
    if target.index < 0 or target.index >= torch.cuda.device_count():
        raise ValueError(f"Selected GPU {target} is not visible")
    settings = config.get("retrieval", {})
    model = config.get("model", {})
    for name, default in (("query_batch_size", 32), ("gallery_chunk_size", 65536),
                          ("score_buffer_mib", 256), ("encode_batch_size", 1024),
                          ("warmup", 3), ("repeats", 5)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"retrieval.{name} must be a positive integer")
    temporary_mib = settings.get("faiss_temp_memory_mib", 512)
    if isinstance(temporary_mib, bool) or not isinstance(temporary_mib, int) or temporary_mib < 0:
        raise ValueError("retrieval.faiss_temp_memory_mib must be a nonnegative integer")
    fraction = float(settings.get("gpu_memory_fraction", .8))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("retrieval.gpu_memory_fraction must be in (0, 1]")
    width, k = int(model.get("hidden_dim", 8192)), int(model.get("k", 8))
    if not 1 <= k <= width or width > torch.iinfo(torch.int32).max:
        raise ValueError("Retrieval requires 1 <= K <= hidden_dim <= int32 maximum")
    rows = int(gallery_rows_estimate)
    if rows < 0 or input_dim < 1:
        raise ValueError("Gallery estimate must be nonnegative and input_dim positive")
    landmark_dictionary = model.get("landmark_dictionary", True)
    if not isinstance(landmark_dictionary, bool):
        raise ValueError("model.landmark_dictionary must be a boolean")
    rank, anchors = model.get("rank", 32), model.get("anchor_count", 1024)
    context_references = model.get("context_reference_count", 0)
    if isinstance(context_references, bool) or not isinstance(context_references, int) or context_references < 0:
        raise ValueError("model.context_reference_count must be a nonnegative integer")
    if landmark_dictionary:
        for name, value in (("rank", rank), ("anchor_count", anchors)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
    score_cap = int(settings.get("score_buffer_mib", 256)) * 2**20
    query_batch = int(settings.get("query_batch_size", 32))
    chunk_rows = min(int(settings.get("gallery_chunk_size", 65536)),
                     (4 * 2**20) // k, score_cap // (4 * query_batch))
    if chunk_rows < 1:
        raise ValueError("The selected sparse score buffer cannot hold one gallery row")
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.cuda.device(target), torch.autocast(device_type="cuda", enabled=False):
            probe = _retrieval_probe(target, settings)
            # The tiny probe's local resource objects are released before this
            # guard, so its FAISS reservation is not counted a second time.
            from sparse_gpu_retrieval import check_cuda_memory_budget
            estimates = {}
            if rows:
                chunk = min(rows, chunk_rows)
                chunks = math.ceil(rows / chunk_rows)
                index_bytes = 8 * rows * k + 8 * chunks * (width + 1) + 8 * width
                workspace_bytes = 4 * query_batch * chunk
                construction_bytes = 96 * chunk * k + 24 * (width + 1)
                parameter_count = input_dim * width + width + input_dim
                dictionary_buffer_bytes = attention_workspace_bytes = 0
                if landmark_dictionary:
                    parameter_count += 2 * input_dim * rank + 1
                    # Frozen FP32 features and normalized keys, int64 anchor IDs,
                    # and the temperature/architecture scalar buffers.
                    dictionary_buffer_bytes = 8 * anchors * input_dim + 8 * anchors + 8
                    # Simultaneous FP32 attention logits and softmax weights.
                    attention_workspace_bytes = 8 * int(settings.get("encode_batch_size", 1024)) * anchors
                # The separate immutable DLMMSAE reference bank stays resident
                # during evaluation in this runner, though it is training-only.
                context_reference_bytes = context_references * (8 * input_dim + 8)
                model_buffer_bytes = 8 * width + dictionary_buffer_bytes + context_reference_bytes
                # Parameters, gradients and Adam's two moments. The persistent
                # input dictionary is included; the refreshed KL bank and full
                # training/encoding activation footprint are not estimated here.
                parameter_state_bytes = 16 * parameter_count
                peak_bound = parameter_state_bytes + model_buffer_bytes + index_bytes + max(
                    workspace_bytes, construction_bytes, attention_workspace_bytes)
                check_cuda_memory_budget(target, peak_bound, fraction,
                                         "MMRSAE parameters/Adam, input dictionary and sparse retrieval")
                estimates = {"gallery_rows": rows, "sparse_index_upper_bytes": index_bytes,
                             "sparse_score_workspace_bytes": workspace_bytes,
                             "sparse_build_workspace_estimate_bytes": construction_bytes,
                             "parameter_adam_gradient_bytes": parameter_state_bytes,
                             "landmark_dictionary": landmark_dictionary,
                             "landmark_dictionary_buffer_bytes": dictionary_buffer_bytes,
                             "context_reference_buffer_bytes": context_reference_bytes,
                             "landmark_attention_workspace_estimate_bytes": attention_workspace_bytes,
                             "model_buffer_bytes": model_buffer_bytes,
                             "estimate_excludes": "training activations, refreshed student KL bank, "
                                                  "non-attention encoding intermediates, library overhead"}
                if config.get("dense_reference", False):
                    dense_bytes = 4 * rows * input_dim + temporary_mib * 2**20
                    check_cuda_memory_budget(target, dense_bytes + parameter_state_bytes + model_buffer_bytes, fraction,
                                             "optional resident full-parent FAISS gallery")
                    estimates["dense_reference_payload_and_scratch_bytes"] = dense_bytes
            properties = torch.cuda.get_device_properties(target)
            return {"device": str(target), "device_name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "device_total_bytes": properties.total_memory,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "precision": "float32", "tf32": False, "cpu_fallback": False,
                    "gallery_estimates": estimates, **probe}
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise RuntimeError("MMRSAE GPU preflight failed before dataset preparation or training. "
                           "A working CUDA PyTorch installation, CUDA-enabled FAISS, and the "
                           "bundled sparse CUDA extension are required. No fallback was used. "
                           f"Cause: {exc}") from exc
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
