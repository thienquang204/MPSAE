"""Legacy pipeline guards run before expensive downloads or extraction."""
import builtins
from types import SimpleNamespace

import pytest

from mpsae import pipeline


@pytest.mark.parametrize("capability,runtime", [("8.0", "11.8"), ("8.6", "11.8"),
                                               ("9.0", "12.0"), ("12.0", "12.6"),
                                               ("10.3", "12.8")])
def test_architecture_with_zero_minor_version_still_checks_runtime(
        capability, runtime, tmp_path, monkeypatch):
    facts = {"cuda_available": True, "torch": "fixture", "cuda": runtime,
             "gpus": [{"index": 0, "capability": capability, "total_memory": 80 << 30}]}
    monkeypatch.setattr(pipeline, "log_environment", lambda *args: facts)
    monkeypatch.setattr(pipeline, "_driver_major", lambda: 580)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda *args: SimpleNamespace(free=1 << 40))
    monkeypatch.setenv("HF_TOKEN", "fixture")
    original = builtins.__import__

    def fake_dependencies(name, *args, **kwargs):
        if name in ("torch", "timm", "huggingface_hub", "datasets", "ffcv", "cupy"):
            return SimpleNamespace()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_dependencies)
    options = pipeline.PipelineOptions(paths=pipeline.Paths(tmp_path), backbones=("rn18",))
    with pytest.raises(RuntimeError, match="preflight problem"):
        pipeline.stage_env(options)


def test_legacy_smoke_does_not_upload_even_if_repo_is_configured(tmp_path):
    options = pipeline.PipelineOptions(paths=pipeline.Paths(tmp_path), smoke=True,
                                        hf_repo="account/MPSAE")
    assert pipeline.stage_publish(options)["uploaded"] is False


def test_source_extraction_selects_the_released_ffcv_recipe(tmp_path, monkeypatch):
    import mpsae.embed

    options = pipeline.PipelineOptions(paths=pipeline.Paths(tmp_path))
    options.paths.ffcv.mkdir(parents=True)
    # A different recipe sorts first and must not be selected via glob order.
    (options.paths.ffcv / "train_100_0.00_10.ffcv").touch()
    expected = options.paths.ffcv / "train_500_0.50_90.ffcv"
    expected.touch()
    seen = []

    def capture(spec, path, *args, **kwargs):
        seen.append(path)
        return {"path": str(path)}

    monkeypatch.setattr(mpsae.embed, "extract_timm", capture)
    pipeline._embed_one(pipeline.BACKBONES["rn18"], options, "train")
    assert seen == [expected]
