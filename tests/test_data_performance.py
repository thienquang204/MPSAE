"""Performance changes must preserve row order, tail batches and resume state."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mpsae.features import BatchSource, Bundle, Split
from mpsae.config import BACKBONES, Cell
from mpsae.train import FitSettings, fit, RUNTIME_IMPLEMENTATION


@pytest.mark.parametrize("residency", ["cpu", "mmap"])
@pytest.mark.parametrize("with_labels", [True, False])
def test_exact_numpy_order_tail_and_resume(residency, with_labels):
    features = np.arange(39, dtype=np.float32).reshape(13, 3)
    labels = np.arange(13, dtype=np.int64)
    source = BatchSource(features, labels, batch_size=4, seed=7, device="cpu",
                         residency=residency, with_labels=with_labels)
    for epoch in (0, 1):
        order = np.random.default_rng(np.random.SeedSequence([7, epoch])).permutation(13)
        for skip in (0, 2):
            batches = list(source.epoch(epoch, skip_batches=skip))
            assert [i for i, _, _ in batches] == list(range(skip, 4))
            actual = torch.cat([x for _, x, _ in batches]).numpy()
            np.testing.assert_array_equal(actual, features[order[skip * 4:]])
            if with_labels:
                np.testing.assert_array_equal(torch.cat([y for _, _, y in batches]), order[skip * 4:])
            else:
                assert source._labels is None
                assert all(y is None for _, _, y in batches)
    source.release()


def test_unused_labels_are_not_materialized_and_cpu_features_are_copied():
    class Unused:
        def __array__(self, *args, **kwargs):
            raise AssertionError("unsupervised training must not read labels")
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    expected = features.copy()
    source = BatchSource(features, Unused(), batch_size=4, seed=7, device="cpu",
                         residency="cpu", with_labels=False)
    features[:] = -1
    actual = torch.cat([x for _, x, _ in source.epoch(0)]).numpy()
    np.testing.assert_array_equal(actual, expected[source.permutation(0)])


def test_trainer_collects_metrics_only_for_log_checkpoint_or_final(tmp_path, monkeypatch):
    from mpsae import methods
    features = np.random.default_rng(2).normal(size=(24, 6)).astype(np.float32)
    labels = np.arange(24) % 3
    identity = {"train": {"sha256": hashlib.sha256(features.tobytes()).hexdigest(), "rows": 24}}
    split = Split("train", features, labels, {})
    bundle = Bundle("ff2048", split, split, identity)
    preset = replace(BACKBONES["ff2048"].preset, hidden_dim=32, k_aux=4, batch_size=4, epochs=1)
    spec = replace(BACKBONES["ff2048"], dim=6, preset=preset, k_list=(2, 4), e2_anchors=(2,))
    settings = FitSettings(device="cpu", residency="cpu", log_every=4, checkpoint_every=3)
    calls = []
    original = methods.compute_step

    def tracked(*args, **kwargs):
        calls.append(kwargs["collect_metrics"])
        return original(*args, **kwargs)

    monkeypatch.setattr(methods, "compute_step", tracked)
    result = fit(Cell("ff2048", "csr", 2, 7), bundle, tmp_path, settings,
                 spec=spec, code_version="test")
    assert calls == [False, False, True, True, False, True]
    assert result["final_metrics"]["loss_total"] > 0
    assert result["runtime_history"][-1]["runtime"]["implementation"] == RUNTIME_IMPLEMENTATION
    assert result["runtime_history"][-1]["runtime"]["hardware"]["device_type"] == "cpu"
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert checkpoint["last_metrics"] == result["final_metrics"]
    assert checkpoint["runtime_history"] == result["runtime_history"]
