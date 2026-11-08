"""Encoding and exact retrieval.

These guard the seam between ``mpsae.runs`` and ``mpsae.evaluate``: the two
were written against one another's signatures rather than against a running
system, and the first bug found there -- ``row_ids=None`` meaning "every row"
on one side and being an error on the other -- would have crashed every single
evaluation point after hours of training. Anything that crosses that boundary
gets a test here.
"""
from __future__ import annotations

import pytest

# Must precede the mpsae imports: these modules import torch at module scope.
pytest.importorskip("torch")

import numpy as np  # noqa: E402

from mpsae.evaluate import encode, knn  # noqa: E402
from mpsae.model import build_sae  # noqa: E402

DIM, HIDDEN, ROWS, K = 6, 32, 40, 3


def test_native_a100_extension_honors_explicit_architecture(monkeypatch):
    import os
    import torch
    from torch.utils import cpp_extension
    from mpsae.evaluate import timing

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(timing, "_EXTENSION", None)
    monkeypatch.setattr(timing, "_EXTENSION_FAILED", False)
    extension = object()
    seen = []

    def compile_extension(**kwargs):
        arches = os.environ["TORCH_CUDA_ARCH_LIST"]
        seen.append(arches)
        if arches != "8.0":
            raise RuntimeError("CUDA 12.6 does not support the requested architecture")
        return extension

    monkeypatch.setattr(cpp_extension, "load", compile_extension)
    assert timing.maybe_build_inverted_index() is extension
    assert seen == ["8.0"]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"


@pytest.fixture
def sae():
    return build_sae(DIM, HIDDEN, seed=1227)


@pytest.fixture
def features():
    generator = np.random.default_rng(11)
    return generator.normal(size=(ROWS, DIM)).astype(np.float32)


# ------------------------------------------------- row selection at the seam

def test_none_means_every_row(sae, features):
    """``mpsae.runs`` passes ``None`` for the whole split; it must not raise."""
    whole = encode.encode_split(sae, features, None, K, device="cpu")
    explicit = encode.encode_split(sae, features, np.arange(ROWS), K, device="cpu")
    assert whole["stats"]["rows"] == ROWS
    assert np.array_equal(whole["indices"], explicit["indices"])
    assert np.allclose(whole["values"], explicit["values"])


def test_none_means_every_row_for_the_dense_reference(features):
    """The same contract on the path that measures the un-compressed backbone."""
    whole = np.asarray(encode.encode_dense(features, None, device="cpu"))
    assert whole.shape == (ROWS, DIM)
    assert np.allclose(whole, features)


def test_negative_dense_rows_are_not_empty_codes():
    codes = np.array([[-1., -2.], [0., 0.], [0., -3.]], dtype=np.float32)
    source = encode.CodeSource(codes, dim=2, device="cpu")
    assert source.empty_mask().tolist() == [False, True, False]
    result = knn.dense_top1(codes[[0, 2]], [0, 1], codes[[0]], [0], device="cpu")
    assert result["top1"] == 100.0
    assert result["empty_queries"] == 0


def test_row_selection_is_validated(sae, features):
    for bad in (np.array([], dtype=np.int64),          # empty
                np.array([0, 0, 1]),                   # not strictly increasing
                np.array([2, 1, 0]),                   # descending
                np.array([0, ROWS])):                  # out of range
        with pytest.raises(ValueError):
            encode.encode_split(sae, features, bad, K, device="cpu")


# ------------------------------------------------------------- encode output

def test_codes_are_normalised_and_within_budget(sae, features):
    result = encode.encode_split(sae, features, None, K, device="cpu")
    indices, values = result["indices"], result["values"]
    assert indices.shape == values.shape == (ROWS, K)
    assert indices.dtype == np.int32 and values.dtype == np.float32
    assert indices.min() >= 0 and indices.max() < HIDDEN
    norms = np.linalg.norm(values, axis=1)
    nonempty = result["stats"]["rows"] - result["stats"]["empty_codes"]
    # A row is either L2-normalised or entirely zero; nothing in between.
    assert np.isclose(norms[norms > 0], 1.0, atol=1e-5).all()
    assert int((norms > 0).sum()) == nonempty


def test_stats_describe_the_codes(sae, features):
    stats = encode.encode_split(sae, features, None, K, device="cpu")["stats"]
    for field in ("rows", "k", "nnz_mean", "nnz_min", "empty_codes",
                  "empty_fraction", "seconds"):
        assert field in stats, field
    assert stats["k"] == K
    assert 0 <= stats["nnz_mean"] <= K
    assert stats["precision"] == "cpu_float32_compute_float32_storage"


def test_cache_round_trips(sae, features, tmp_path):
    """E2 re-reads the same codes many times; a wrong cache would be silent."""
    identity = {"test": "cache", "split": "train"}
    first = encode.encode_split(sae, features, None, K, device="cpu",
                                cache_dir=tmp_path, identity=identity)
    second = encode.encode_split(sae, features, None, K, device="cpu",
                                 cache_dir=tmp_path, identity=identity)
    assert np.array_equal(np.asarray(first["indices"]), np.asarray(second["indices"]))
    assert np.allclose(np.asarray(first["values"]), np.asarray(second["values"]))


# ----------------------------------------------------------- exact retrieval

def test_blocked_gemm_matches_bruteforce():
    """The search replaces FAISS, so it has to be exact, not merely close."""
    generator = np.random.default_rng(5)
    gallery = generator.normal(size=(97, 8)).astype(np.float32)
    queries = generator.normal(size=(23, 8)).astype(np.float32)
    report = knn.verify_against_bruteforce(gallery, queries, dim=8, device="cpu")
    assert report["max_score_difference"] < 1e-5
    assert report["position_mismatches"] == 0, report["mismatched_queries"]
    assert report["queries"] == 23


def test_ties_take_the_lowest_gallery_position():
    """``IndexFlatL2`` returns the first of equal neighbours; so must we."""
    gallery = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1))
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    _, positions, _ = knn.exact_top1(gallery, queries, dim=2, device="cpu",
                                     gallery_chunk=2, query_block=1)
    assert int(positions[0]) == 0


def test_empty_query_codes_are_counted_and_scored_as_misses():
    """An all-zero code carries no information; crediting it would flatter
    exactly the methods whose codes collapse."""
    indices = np.zeros((3, 2), dtype=np.int32)
    values = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    gallery = (np.zeros((2, 2), dtype=np.int32),
               np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    result = knn.top1_accuracy(gallery, np.array([0, 0]), (indices, values),
                               np.array([0, 0, 0]), dim=4, device="cpu")
    assert result["empty_queries"] == 1
    assert result["queries"] == 3
    # Two informative queries match their neighbour's label; the empty one cannot.
    assert result["correct"] == 2
    assert result["top1"] == pytest.approx(200.0 / 3)
