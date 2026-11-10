"""The standalone profiler labels synthetic data and avoids campaign writes."""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mpsae import profile_train
from mpsae.config import BACKBONES


@pytest.fixture
def tiny_profile(monkeypatch):
    original = BACKBONES["ff2048"]
    preset = replace(original.preset, hidden_dim=32, k_aux=4, batch_size=4, epochs=2,
                     ot_microbatch=2, ot_max_iters=20, ot_max_refine_iters=100)
    spec = replace(original, dim=6, preset=preset, k_list=(2, 4), e2_anchors=(2, 4))
    monkeypatch.setitem(profile_train.BACKBONES, "ff2048", spec)


def test_cpu_cli_writes_explicit_synthetic_profile(tmp_path, tiny_profile):
    output = tmp_path / "profile.json"
    assert profile_train.main(["--device", "cpu", "--method", "csr", "--k", "2",
        "--warmup", "1", "--steps", "2", "--out", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scientific_result"] is False
    assert result["data"]["kind"] == "synthetic_resident_batch_repeated"
    assert result["samples"] == 8
    assert result["warmup_steps"] == 1
    assert result["measured_steps"] == 2
    assert result["samples_per_second"] > 0
    assert result["wall_step_seconds"]["p95"] >= result["wall_step_seconds"]["p50"] > 0
    assert result["cuda_compute_seconds"] is None
    assert result["memory"] is None
    assert result["trace"] is None
    assert result["preset"]["batch_size"] == 4
    assert result["precision"] == {"dtype": "float32", "autocast": False, "tf32": False}
    assert list(tmp_path.iterdir()) == [output]


def test_local_frozen_data_uses_loader_and_counts_actual_rows(tmp_path, tiny_profile, monkeypatch):
    from mpsae import features
    loaded = []
    rows = 10  # last short batch is retained and counted, not reported as full
    data = np.random.default_rng(123).normal(size=(rows, 6)).astype(np.float32)
    bundle = SimpleNamespace(train=SimpleNamespace(features=data, labels=np.arange(rows), rows=rows),
                             identity={"train": {"sha256": "fixed-test-archive", "rows": rows}})

    def load(spec, embeddings, cache, *, splits):
        loaded.append((embeddings, cache, splits))
        return bundle

    monkeypatch.setattr(features, "load_bundle", load)
    output = tmp_path / "local.json"
    args = profile_train.build_parser().parse_args([
        "--device", "cpu", "--method", "csr", "--k", "2", "--steps", "3", "--warmup", "0",
        "--out", str(output), "--embeddings-dir", str(tmp_path / "frozen"),
        "--cache-dir", str(tmp_path / "cache")])
    result = profile_train.run_profile(args)
    assert loaded == [(args.embeddings_dir, args.cache_dir, ("train",))]
    assert result["samples"] == rows
    assert result["data"]["kind"] == "local_frozen_embeddings"
    assert result["data"]["identity"] == bundle.identity
    assert result["schedule"]["total_steps"] == 6


def test_trace_window_does_not_add_to_benchmark_samples(tmp_path, tiny_profile):
    output, trace = tmp_path / "profile.json", tmp_path / "trace.json"
    args = profile_train.build_parser().parse_args([
        "--device", "cpu", "--method", "csr", "--k", "2", "--steps", "2", "--warmup", "1",
        "--out", str(output), "--trace", str(trace)])
    result = profile_train.run_profile(args)
    assert result["samples"] == 8
    assert result["measured_steps"] == 2
    assert result["trace"]["steps"] == 2
    assert result["trace"]["timed_in_benchmark"] is False
    assert result["trace"]["window"] == "after_benchmark"
    events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
    assert any(event.get("name") == "train/forward_loss" for event in events)
    assert json.loads(output.read_text(encoding="utf-8"))["trace"] == result["trace"]


@pytest.mark.parametrize("extra", [["--steps", "0"], ["--warmup", "-1"], ["--k", "0"],
                                   ["--start-step", "-1"], ["--seed", "-1"]])
def test_invalid_arguments_fail_before_output(tmp_path, tiny_profile, extra):
    output = tmp_path / "invalid.json"
    args = profile_train.build_parser().parse_args(["--device", "cpu", "--out", str(output)] + extra)
    with pytest.raises(ValueError):
        profile_train.run_profile(args)
    assert not output.exists()


def test_checkpoint_output_collision_is_rejected(tmp_path, tiny_profile):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"do not touch")
    args = profile_train.build_parser().parse_args([
        "--device", "cpu", "--k", "2", "--checkpoint", str(checkpoint), "--out", str(checkpoint)])
    with pytest.raises(ValueError, match="overwrite"):
        profile_train.run_profile(args)
    assert checkpoint.read_bytes() == b"do not touch"
