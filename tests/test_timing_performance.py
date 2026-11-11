"""Bounded CSR construction/search and timing protocol identity."""
from __future__ import annotations

from types import SimpleNamespace
import builtins
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
from mpsae.evaluate import timing
from mpsae.timing_config import runtime_fingerprint, timing_config


def _codes(rows=13, dim=16, k=4):
    generator = torch.Generator().manual_seed(17)
    selected = torch.rand(rows, dim, generator=generator).topk(k, dim=1)
    values = selected.values
    values[0] = 0
    values[3, 2:] = 0
    return selected.indices.to(torch.int32), values


@pytest.mark.parametrize("chunk,cap", [(1, 4), (5, 8), (32, 20)])
def test_shards_and_global_top1_match_dense_reference(chunk, cap):
    codes = _codes()
    expected = torch.zeros(13, 16).scatter_(1, codes[0].long(), codes[1])
    shards, info = timing.build_csr_shards(codes, dim=16, device="cpu",
                                          gallery_chunk=chunk, build_entry_cap=cap, query_batch=3)
    assert all(s.crow_indices().dtype == s.col_indices().dtype == torch.int32 for s in shards)
    assert info["nnz"] == int(expected.ne(0).sum())
    assert info["index_bytes"] == sum(t.numel() * t.element_size() for s in shards
                                      for t in (s.crow_indices(), s.col_indices(), s.values()))
    assert info["build_block_rows"] * codes[0].shape[1] <= cap
    assert info["score_workspace_bytes"] == min(chunk, len(expected)) * 3 * 4
    assert torch.equal(torch.cat([s.to_dense() for s in shards]), expected)
    queries = torch.rand(3, 16, generator=torch.Generator().manual_seed(3))
    queries[0] = 0
    scores, positions = timing.sparse_sharded_top1(shards, queries)
    reference = (queries @ expected.T).max(1)
    torch.testing.assert_close(scores, reference.values)
    assert torch.equal(positions, reference.indices)


def test_zero_gallery_and_cross_shard_ties_keep_first_row():
    columns = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    values = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 0.]])
    shards, _ = timing.build_csr_shards((columns, values), dim=2, device="cpu",
                                       gallery_chunk=2, build_entry_cap=2)
    query = torch.tensor([[1., 0.], [0., 0.]])
    scores, positions = timing.sparse_sharded_top1(shards, query)
    assert positions.tolist() == [1, 0]
    assert scores.tolist() == [1., 0.]
    shards, info = timing.build_csr_shards((columns, values * 0), dim=2, device="cpu",
                                          gallery_chunk=2, build_entry_cap=2)
    scores, positions = timing.sparse_sharded_top1(shards, query)
    assert info["nnz"] == 0
    assert positions.tolist() == [0, 0]
    assert scores.tolist() == [0., 0.]


def test_single_matrix_compatibility_and_explicit_row_selection():
    codes = _codes()
    single, info = timing.build_csr_gallery(codes, dim=16, device="cpu", rows=7)
    assert single.shape == (7, 16)
    assert single.crow_indices().dtype == torch.int64
    shards, _ = timing.build_csr_shards(codes, dim=16, device="cpu", rows=7,
                                       gallery_chunk=3, build_entry_cap=8)
    assert torch.equal(single.to_dense(), torch.cat([s.to_dense() for s in shards]))
    assert info["rows"] == 7


def test_int32_overflow_is_bounded_locally_without_allocating_large_arrays():
    # Full ImageNet at K2048 has >2**31 nnz; never cast global offsets to int32.
    assert 1_281_167 * 2048 > torch.iinfo(torch.int32).max
    rows = timing._shard_rows(2048, 1_281_167)
    assert rows < 1_281_167
    assert rows * 2048 <= torch.iinfo(torch.int32).max
    with pytest.raises(ValueError, match="int32"):
        timing._shard_rows(torch.iinfo(torch.int32).max + 1, 1)


def test_preflight_runs_before_any_csr_allocation(monkeypatch):
    requested = []
    def refuse(device, needed_bytes, **kwargs):
        requested.append(needed_bytes)
        raise RuntimeError("test preflight insufficient VRAM")
    monkeypatch.setattr(timing, "require_vram", refuse)
    monkeypatch.setattr(torch, "sparse_csr_tensor", lambda *a, **kw: pytest.fail("allocated before preflight"))
    with pytest.raises(RuntimeError, match="preflight"):
        timing.build_csr_shards(_codes(), dim=16, device="cpu", gallery_chunk=5,
                                build_entry_cap=8, query_batch=3)
    assert requested and requested[0] > 0


@pytest.mark.parametrize("change", ["nan", "negative", "invalid_coordinate"])
def test_invalid_codes_fail_before_build(change):
    columns, values = _codes()
    if change == "nan":
        values[1, 0] = float("nan")
    elif change == "negative":
        values[1, 0] = -1
    else:
        columns[1, 0] = 16
    with pytest.raises(ValueError, match="finite nonnegative"):
        timing.build_csr_shards((columns, values), dim=16, device="cpu")


@pytest.mark.parametrize("kwargs", [{"rows": 0}, {"gallery_chunk": 0}, {"build_entry_cap": 3},
                                     {"query_batch": False}])
def test_invalid_builder_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        timing.build_csr_shards(_codes(), dim=16, device="cpu", **kwargs)


def test_dense_tiles_match_full_matmul_with_negative_scores_and_ties():
    gallery = torch.tensor([[1., 2.], [-1., -2.], [1., 2.], [-1., -2.]])
    queries = torch.tensor([[1., 1.], [-1., -1.], [0., 0.]])
    scores, positions = timing.dense_tiled_top1(gallery, queries, gallery_chunk=2)
    reference = (queries @ gallery.T).max(1)
    assert torch.equal(positions, reference.indices)
    torch.testing.assert_close(scores, reference.values)


def test_timing_identity_changes_with_resource_or_sampling_settings():
    import inspect
    from mpsae.benchmark.config import Campaign
    from mpsae.runs import RunSettings

    base = timing_config()
    settings = RunSettings()
    campaign = Campaign.from_mapping({})
    assert base["warmup"] == base["rounds"] == 5
    assert settings.timing_warmup == settings.timing_rounds == 5
    assert campaign.imagenet["timing_warmup"] == campaign.imagenet["timing_rounds"] == 5
    for entry in (timing.time_sparse_matmul, timing.time_dense_gemm):
        parameters = inspect.signature(entry).parameters
        assert parameters["warmup"].default == parameters["rounds"].default == 5
    for key in ("query_batch", "warmup", "rounds", "gallery_chunk", "build_entry_cap"):
        assert timing_config(**{key: base[key] + 1}) != base
        with pytest.raises(ValueError):
            timing_config(**{key: 0})
    assert base["version"] == timing.TIMING_VERSION


@pytest.mark.parametrize("warmup,rounds", [(2, 3), (5, 5)])
def test_progress_is_outside_measured_intervals(monkeypatch, warmup, rounds):
    state = {"active": False, "event": 0, "searches": 0}
    logs = []
    class Event:
        def __init__(self, **kwargs):
            self.start = state["event"] == 0
            state["event"] += 1
        def record(self, stream):
            state["active"] = self.start
        def synchronize(self):
            pass
        def elapsed_time(self, stop):
            return 3.0
    def log(*args):
        assert not state["active"]
        logs.append(args)
    def search():
        state["searches"] += 1
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(timing, "get_logger", lambda name: SimpleNamespace(info=log))
    result = timing._measure(search, device=torch.device("cuda"), warmup=warmup,
                              rounds=rounds, progress_seconds=1e-12)
    assert state["searches"] == warmup + rounds
    assert result["cuda_seconds"]["median"] == 0.003
    assert result["rounds"] == rounds
    assert len(logs) >= 5


def test_speedup_refuses_mixed_protocols():
    record = dict(queries_per_batch=512, index={"rows": 10, "index_bytes": 100},
                  cuda_seconds={"median": 0.1}, precision="float32_tf32_disabled",
                  timing_scope=timing.TIMING_SCOPE, warmup=5, rounds=5,
                  version=timing.TIMING_VERSION, config=timing_config(),
                  runtime={"device_name": "test GPU", "torch": "test", "cuda": "test"})
    assert timing.speedup(record, record)["dense_over_sparse"] == 1
    with pytest.raises(ValueError, match="rounds"):
        timing.speedup(record, {**record, "rounds": 50})
    with pytest.raises(ValueError, match="tiling"):
        timing.speedup(record, {**record, "config": timing_config(gallery_chunk=16)})
    with pytest.raises(ValueError, match="runtime/device"):
        timing.speedup(record, {**record, "runtime": {**record["runtime"], "device_name": "other GPU"}})
    missing = {key: value for key, value in record.items() if key != "runtime"}
    with pytest.raises(ValueError, match="runtime/device"):
        timing.speedup(missing, missing)


def test_runtime_fingerprint_tracks_hardware_runtime_and_current_device(monkeypatch):
    properties = [SimpleNamespace(name="RTX 5090", major=12, minor=0,
                                   total_memory=32 << 30, uuid=b"GPU-A"),
                  SimpleNamespace(name="H100", major=9, minor=0,
                                   total_memory=80 << 30, uuid="GPU-B")]
    fake_torch = SimpleNamespace(__version__="2.9-test", version=SimpleNamespace(cuda="12.8"),
                                  device=torch.device,
                                  cuda=SimpleNamespace(is_available=lambda: True,
                                                       current_device=lambda: 1,
                                                       device_count=lambda: 2,
                                                       get_device_properties=lambda i: properties[i]))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    first = runtime_fingerprint("cuda:0")
    assert first["available"] is True
    assert first["capability"] == [12, 0]
    assert first["total_memory_bytes"] == 32 << 30
    assert first["uuid"] == b"GPU-A".hex()
    assert runtime_fingerprint("cuda")["device_name"] == "H100"
    assert runtime_fingerprint("cuda:1") != first
    properties[0].uuid = "GPU-C"
    assert runtime_fingerprint("cuda:0") != first
    before_upgrade = runtime_fingerprint("cuda:0")
    fake_torch.__version__ = "2.10-test"
    assert runtime_fingerprint("cuda:0") != before_upgrade
    assert runtime_fingerprint("cuda:7")["unavailable_reason"] == "device_not_visible"


def test_runtime_fingerprint_handles_cpu_only_inspection(monkeypatch):
    fake_torch = SimpleNamespace(__version__="2.9-cpu", version=SimpleNamespace(cuda=None),
                                  device=torch.device, cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    record = runtime_fingerprint("cuda:0")
    assert record["available"] is False
    assert record["unavailable_reason"] == "cuda_unavailable"
    assert record["cuda"] is None
    assert runtime_fingerprint("cpu")["available"] is True


def test_runtime_fingerprint_handles_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    record = runtime_fingerprint("cuda:0")
    assert record["unavailable_reason"] == "torch_not_installed"
    assert record["available"] is False


def test_timing_config_module_does_not_import_torch(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("timing configuration imported torch eagerly")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source = Path(__file__).resolve().parents[1] / "mpsae" / "timing_config.py"
    namespace = {"__name__": "standalone_timing_config"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    assert namespace["timing_config"]()["query_batch"] == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation runs on the GPU server")
def test_cuda_int32_shards_scores_and_timed_metadata():
    codes = _codes(rows=17, dim=32, k=8)
    shards, info = timing.build_csr_shards(codes, dim=32, device="cuda",
                                          gallery_chunk=5, build_entry_cap=16, query_batch=3)
    query = torch.rand(3, 32, device="cuda")
    with timing.fp32_matmul(), torch.no_grad():
        actual = timing.sparse_sharded_top1(shards, query)
        dense = torch.zeros(17, 32).scatter_(1, codes[0].long(), codes[1]).cuda()
        expected = (query @ dense.T).max(1)
    torch.testing.assert_close(actual[0], expected.values, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[1], expected.indices)
    assert info["max_shard_rows"] == 5
    result = timing.time_sparse_matmul(codes, codes, dim=32, device="cuda", query_batch=3,
                                       warmup=1, rounds=2, gallery_chunk=5,
                                       build_entry_cap=16, progress_seconds=0)
    assert result["config"] == timing_config(query_batch=3, warmup=1, rounds=2,
                                              gallery_chunk=5, build_entry_cap=16)
    assert result["index"]["rows"] == 17
    assert result["cuda_seconds"]["rounds"] == 2
    assert result["runtime"] == runtime_fingerprint("cuda")
