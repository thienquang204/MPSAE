"""Dataset identities and relevance partitions must survive HF repackaging."""
from __future__ import annotations

import pytest

from mpsae.benchmark.data import build_dataset_bundle


def _eurosat():
    return {name: [{"image_id": f"{name}-{index}", "label": index % 2}
                   for index in range(4)] for name in ("train", "validation", "test")}


def test_gallery_train_is_not_a_training_operation():
    bundle = build_dataset_bundle("eurosat", _eurosat(), revision="fixture-1",
                                   strict=False, audit_images=False)
    assert bundle.gallery.name == "train" and bundle.query.name == "test"
    assert not bundle.exclude_self
    assert bundle.manifest["training_on_target"] is False
    assert bundle.gallery.labels == bundle.query.labels


def test_identity_changes_when_row_alignment_changes():
    first = _eurosat()
    second = _eurosat()
    second["test"] = list(reversed(second["test"]))
    lhs = build_dataset_bundle("eurosat", first, revision="fixture-1", strict=False, audit_images=False)
    rhs = build_dataset_bundle("eurosat", second, revision="fixture-1", strict=False, audit_images=False)
    assert lhs.identity != rhs.identity


def test_overlapping_sample_ids_are_rejected():
    splits = _eurosat()
    splits["test"][0]["image_id"] = splits["train"][0]["image_id"]
    with pytest.raises(ValueError, match="overlap"):
        build_dataset_bundle("eurosat", splits, revision="fixture-1", strict=False, audit_images=False)


@pytest.mark.parametrize("invalid", [None, True, float("nan"), 1.5, "  "])
def test_missing_or_ambiguous_sample_ids_are_rejected(invalid):
    splits = _eurosat()
    splits["test"][0]["image_id"] = invalid
    with pytest.raises(ValueError, match="image_id"):
        build_dataset_bundle("eurosat", splits, revision="fixture-1", strict=False, audit_images=False)


def test_queries_without_positive_are_rejected():
    splits = _eurosat()
    splits["test"][0]["label"] = 2
    with pytest.raises(ValueError, match="no relevant"):
        build_dataset_bundle("eurosat", splits, revision="fixture-1", strict=False, audit_images=False)


def test_strict_mode_rejects_subset_masquerading_as_full_dataset():
    with pytest.raises(ValueError, match="expected 16,200"):
        build_dataset_bundle("eurosat", _eurosat(), revision="fixture-1", audit_images=False)


def _sop():
    return {name: [{"image_path": f"{name}/{item}_{view}.jpg", "item_id": item,
                    "category": "chair"}
                   for item in items for view in range(2)]
            for name, items in (("train", ("product-a", "product-b")),
                                ("test", ("product-c", "product-d")))}


def test_sop_uses_item_id_not_coarse_category():
    bundle = build_dataset_bundle("sop", _sop(), revision="fixture-1",
                                   strict=False, audit_images=False)
    assert bundle.exclude_self and bundle.gallery is bundle.query
    assert len(set(bundle.query.labels)) == 2
    assert bundle.manifest["positive_count_min"] == 1
    assert bundle.identity["relevance_key"] == "item_id"


def test_sop_train_test_products_must_be_disjoint():
    splits = _sop()
    splits["test"][0]["item_id"] = "product-a"
    with pytest.raises(ValueError, match="item IDs overlap"):
        build_dataset_bundle("sop", splits, revision="fixture-1", strict=False, audit_images=False)


def test_same_item_id_cannot_mean_two_categories():
    splits = _sop()
    splits["test"][0]["category"] = "bicycle"
    with pytest.raises(ValueError, match="multiple categories"):
        build_dataset_bundle("sop", splits, revision="fixture-1", strict=False, audit_images=False)


@pytest.mark.parametrize("key", ["item_id", "category", "image_path"])
def test_sop_null_metadata_is_rejected_before_string_conversion(key):
    splits = _sop()
    splits["test"][0][key] = None
    with pytest.raises(ValueError, match=key):
        build_dataset_bundle("sop", splits, revision="fixture-1", strict=False, audit_images=False)


def test_flowers_ids_are_content_based_and_detect_cross_split_leakage():
    image_module = pytest.importorskip("PIL.Image")
    splits = {name: [{"image": image_module.new("RGB", (4, 4), (offset + i, 10, 20)),
                      "label": i % 2} for i in range(4)]
              for name, offset in (("train", 0), ("validation", 10), ("test", 20))}
    bundle = build_dataset_bundle("flowers102", splits, revision="fixture-1", strict=False)
    assert "rgb-sha256:" in bundle.query.ids[0]
    splits["test"][0]["image"] = splits["train"][0]["image"].copy()
    with pytest.raises(ValueError, match="overlap"):
        build_dataset_bundle("flowers102", splits, revision="fixture-1", strict=False)


def _source_weights():
    return {"repository": "timm/resnet18.a1_in1k", "revision": "a" * 40,
            "file": "model.safetensors", "sha256": "b" * 64}


def test_source_weights_must_identify_an_immutable_original_backbone():
    from mpsae.benchmark.extract import _expected_weights
    from mpsae.config import BACKBONES

    expected = _source_weights()
    assert _expected_weights(BACKBONES["rn18"], expected) == expected
    for changed in ({"revision": "main"}, {"sha256": "unknown"},
                    {"repository": "timm/different-model"}, {"file": "different.pt"}):
        with pytest.raises(ValueError):
            _expected_weights(BACKBONES["rn18"], {**expected, **changed})


def test_network_pins_the_original_imagenet_revision_and_checks_digest(tmp_path, monkeypatch):
    from mpsae.benchmark.extract import _network
    from mpsae.config import BACKBONES
    from mpsae.embed import archive, csrv2_timm

    expected = _source_weights()
    calls = []

    def release(repository, *, revision, **kwargs):
        calls.append((repository, revision))
        return {"repository": repository, "revision": revision}

    monkeypatch.setattr(archive, "pinned_release", release)
    monkeypatch.setattr(archive, "hub_file", lambda *args, **kwargs: (tmp_path / "weights", "b" * 64))
    network = object()
    monkeypatch.setattr(csrv2_timm, "_timm_model", lambda *args: (network, {}, "resnet18", None))
    loaded, _, provenance = _network(BACKBONES["rn18"], tmp_path, "cpu", None, expected)
    assert calls == [(expected["repository"], expected["revision"])]
    assert loaded is network and provenance["sha256"] == expected["sha256"]
    monkeypatch.setattr(archive, "hub_file", lambda *args, **kwargs: (tmp_path / "weights", "c" * 64))
    with pytest.raises(RuntimeError, match="differ from the source"):
        _network(BACKBONES["rn18"], tmp_path, "cpu", None, expected)


def test_extraction_rechecks_mocked_loader_weight_provenance(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mpsae.benchmark import extract
    from mpsae.config import BACKBONES

    bundle = build_dataset_bundle("eurosat", _eurosat(), revision="fixture-1",
                                   strict=False, audit_images=False)
    monkeypatch.setattr(extract, "_network", lambda *args:
                        (object(), {}, {**_source_weights(), "sha256": "c" * 64}))
    with pytest.raises(RuntimeError, match="differ from the source"):
        extract.extract_bundle(BACKBONES["rn18"], bundle, tmp_path / "features", tmp_path / "cache",
                               device="cpu", expected_weights=_source_weights())


def test_synthetic_sop_reuses_test_gallery_without_self_only_positives(tmp_path):
    np = pytest.importorskip("numpy")
    from mpsae.benchmark.extract import make_synthetic_feature_bundle
    from mpsae.config import BACKBONES

    first = make_synthetic_feature_bundle(BACKBONES["rn18"], "sop", tmp_path / "out", tmp_path / "cache")
    second = make_synthetic_feature_bundle(BACKBONES["rn18"], "sop", tmp_path / "out", tmp_path / "cache")
    assert first.gallery is first.query
    assert first.exclude_self
    assert first.manifest["scientific"] is False
    assert first.gallery.features.shape == (60, 512)
    assert np.array_equal(first.gallery.features, second.gallery.features)
    assert first.identity == second.identity


def test_frozen_extraction_keeps_tail_rows_raw_values_and_source_alignment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    from mpsae.benchmark import extract
    from mpsae.config import BACKBONES

    splits = {name: [{"image_id": f"{name}-{index}", "label": index % 2,
                      "image": image_module.new("RGB", (4, 4), (offset + index, 10, 20))}
                     for index in range(3)]
              for name, offset in (("train", 30), ("validation", 40), ("test", 50))}
    source = build_dataset_bundle("eurosat", splits, revision="fixture-1", strict=False)

    class FrozenNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 512, bias=False)
            self.calls = 0
            with torch.no_grad():
                self.projection.weight.fill_(0.5)

        def forward_features(self, images):
            self.calls += 1
            return self.projection(images)

        def forward_head(self, features, *, pre_logits):
            assert pre_logits
            return features

    model = FrozenNetwork()
    initial = model.projection.weight.detach().clone()
    monkeypatch.setattr(extract, "_network", lambda *args: (model, {}, {"revision": "fixture-weights"}))
    monkeypatch.setattr(extract, "evaluation_transform", lambda *args:
                        (lambda image: torch.tensor(image.getpixel((0, 0)), dtype=torch.float32),
                         {"fixture": "raw_rgb_pixel"}))
    first = extract.extract_bundle(BACKBONES["rn18"], source, tmp_path / "features",
                                    tmp_path / "cache", device="cpu", batch_size=2)
    assert first.gallery.rows == first.query.rows == 3  # final batch of one is retained
    assert first.gallery.ids == source.gallery.ids
    assert first.query.ids == source.query.ids
    assert np.array_equal(first.gallery.labels, source.gallery.labels)
    assert np.allclose(first.gallery.features[:, 0], [30, 30.5, 31])
    assert not np.isclose(np.linalg.norm(first.gallery.features[0]), 1.0)
    assert torch.equal(model.projection.weight, initial)
    assert not model.training and not model.projection.weight.requires_grad
    assert model.projection.weight.grad is None
    calls = model.calls
    second = extract.extract_bundle(BACKBONES["rn18"], source, tmp_path / "features",
                                     tmp_path / "cache", device="cpu", batch_size=2)
    assert model.calls == calls  # a completed extraction is not re-evaluated
    assert first.identity == second.identity
    assert np.array_equal(first.query.features, second.query.features)
