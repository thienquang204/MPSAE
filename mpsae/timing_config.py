"""Torch-free identity of the retrieval timing harness.

Changing its backend, tiling, precision or sampling settings invalidates a
cached timing, without invalidating a trained SAE or its accuracy measurement.
"""
from __future__ import annotations

PAPER_QUERY_BATCH = 512
PAPER_WARMUP = 100
PAPER_ROUNDS = 2000
DEFAULT_GALLERY_CHUNK = 32768
DEFAULT_BUILD_ENTRY_CAP = 4 * 1024 * 1024
TIMING_VERSION = "resident-sharded-csr-v1"
SPARSE_BACKEND = "torch_sparse_csr_sharded_int32_matmul"
TIMING_SCOPE = ("warm_resident_batch_search_repeated;"
                "excludes_encoding_normalization_transfers_and_index_build")


def timing_config(*, query_batch: int = PAPER_QUERY_BATCH,
                  warmup: int = PAPER_WARMUP, rounds: int = PAPER_ROUNDS,
                  gallery_chunk: int = DEFAULT_GALLERY_CHUNK,
                  build_entry_cap: int = DEFAULT_BUILD_ENTRY_CAP) -> dict:
    values = dict(query_batch=query_batch, warmup=warmup, rounds=rounds,
                  gallery_chunk=gallery_chunk, build_entry_cap=build_entry_cap)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"timing {name} must be a positive integer")
    return {"version": TIMING_VERSION, "backend": SPARSE_BACKEND,
            "precision": "float32_tf32_disabled", "timing_scope": TIMING_SCOPE,
            **values}


def runtime_fingerprint(device="cuda:0") -> dict:
    """Identify the runtime/device that a timing describes, without allocating.

    Torch is imported only when this function is called: configuration parsing
    and plan inspection remain torch-free. CPU-only/offline inspection returns
    an explicit unavailable CUDA record, which cannot equal a measured GPU
    runtime. Do not cache this helper across device selections or runtime tests.
    """
    requested = str(device)
    kind, _, ordinal = requested.partition(":")
    record = {"torch": None, "cuda": None, "device_type": kind,
              "device_index": int(ordinal) if ordinal.isdigit() else None,
              "available": False}
    try:
        import torch
    except ImportError:
        record["unavailable_reason"] = "torch_not_installed"
        return record
    selected = torch.device(device)
    record.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                  device_type=selected.type, device_index=selected.index)
    if selected.type != "cuda":
        record["available"] = selected.type == "cpu"
        return record
    if not torch.cuda.is_available():
        record["unavailable_reason"] = "cuda_unavailable"
        return record
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    record["device_index"] = index
    if not 0 <= index < torch.cuda.device_count():
        record["unavailable_reason"] = "device_not_visible"
        return record
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    if isinstance(uuid, bytes):
        uuid = uuid.hex()
    elif uuid is not None:
        uuid = str(uuid)
    record.update(available=True, device_name=str(properties.name),
                  capability=[int(properties.major), int(properties.minor)],
                  total_memory_bytes=int(properties.total_memory), uuid=uuid)
    return record
