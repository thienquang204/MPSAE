"""Joint-checkpoint transfer, leave-one-out voting, and feature cache isolation."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mpsae.config import BACKBONES
from mpsae.finetune.config import load_config
from mpsae.finetune import transfer


def test_sop_weighted_votes_match_independent_self_masked_oracle():
    import torch
    from mpsae.evaluate.classification import evaluate_knn
    from mpsae.evaluate.ranking import evaluate_retrieval
    generator = torch.Generator().manual_seed(18)
    codes = torch.nn.functional.normalize(torch.randn(24, 7, generator=generator), dim=1)
    labels = torch.arange(24) % 4
    ids = [f"product-image-{i}" for i in range(24)]
    similarity = codes @ codes.T
    similarity.fill_diagonal_(-torch.inf)
    scores, indices = similarity.topk(20, dim=1)
    arguments = dict(dim=7, device="cpu", gallery_ids=ids, query_ids=ids, exclude_self=True,
                     query_block=5, gallery_chunk=7)
    actual = evaluate_knn(codes, labels, codes, labels, **arguments)
    for n in (1, 20):
        one_hot = torch.nn.functional.one_hot(labels[indices[:, :n]], num_classes=4)
        votes = (one_hot * (scores[:, :n] / .07).exp()[..., None]).sum(1)
        prediction = votes.argmax(1)
        assert actual[f"knn_{n}_top1"] == pytest.approx(float((prediction == labels).double().mean()))
    assert actual["exclude_self"] and actual["knn_1_top1"] < 1
    retrieval = evaluate_retrieval(codes, labels, codes, labels, **arguments)
    assert retrieval["recall_at_1"] == actual["knn_1_top1"]
    with pytest.raises(ValueError, match="IDs"):
        evaluate_knn(codes, labels, codes, labels, dim=7, device="cpu", exclude_self=True)
    with pytest.raises(ValueError, match="support samples"):
        evaluate_knn(codes, labels, codes, labels, **arguments, n_neighbors=[24])


def test_self_masking_uses_ids_not_query_position():
    import torch
    from mpsae.evaluate.classification import evaluate_knn
    codes = torch.eye(3)
    result = evaluate_knn(codes, [0, 1, 2], codes[[2, 0]], [2, 0], dim=3, device="cpu",
        gallery_ids=["a", "b", "c"], query_ids=["c", "a"], exclude_self=True,
        n_neighbors=[1], gallery_chunk=1, query_block=1)
    assert result["knn_1_top1"] == 0


def test_joint_features_are_cached_per_checkpoint_and_sop_is_extracted_once(tmp_path):
    import torch
    from torch import nn
    from mpsae.benchmark.run import classifier_protocol
    config = load_config("configs/finetune.yaml")
    data = transfer.load_target("sop", tmp_path, config, True)
    spec = transfer.smoke_spec(BACKBONES["ff2048"])
    network = nn.Sequential(nn.Flatten(), nn.Linear(192, 16))
    calls = []
    handle = network.register_forward_hook(lambda module, inputs, result: calls.append(len(result)))
    arguments = dict(preprocessing={"synthetic": True}, precision="float32", device="cpu",
                     batch_size=7, workers=0)
    def extract(identity):
        return transfer.extract(network, spec, data, nn.Identity(), tmp_path,
                                checkpoint_id=identity, **arguments)
    first = extract("first-checkpoint")
    assert first.query is first.gallery and sum(calls) == 32
    values = first.gallery.features.copy()
    assert classifier_protocol(first)["evaluation"] == "sop_test_product_leave_one_out"
    with pytest.raises(ValueError, match="train support"):
        classifier_protocol(replace(first, exclude_self=False))
    transfer.close_features(first)
    calls.clear()
    cached = extract("first-checkpoint")
    assert not calls
    np.testing.assert_array_equal(cached.gallery.features, values)
    transfer.close_features(cached)
    with torch.no_grad():
        network[1].weight.mul_(2)
    changed = extract("second-checkpoint")
    assert sum(calls) == 32 and not np.array_equal(changed.gallery.features, values)
    transfer.close_features(changed)
    handle.remove()


def test_missing_joint_checkpoint_fails_before_downloading_targets(tmp_path, monkeypatch):
    config = load_config("configs/finetune.yaml")
    monkeypatch.setattr(transfer, "load_target", lambda *a, **kw: pytest.fail("unexpected dataset download"))
    with pytest.raises(FileNotFoundError, match="completed joint"):
        transfer.run_transfer(config, tmp_path, smoke=True, device="cpu", workers=0)


def test_transfer_options_change_evaluations_without_changing_fit_settings():
    from copy import deepcopy
    config = load_config("configs/finetune.yaml")
    changed = replace(config, transfer={**deepcopy(config.transfer),
                                      "knn": {"n_neighbors": [1, 5], "temperature": .1}})
    assert changed.plan()["plan_id"] != config.plan()["plan_id"]
    assert changed.cells() == config.cells()
    for cell in config.cells():
        assert changed.training_options(cell) == config.training_options(cell)
    assert transfer.expected_points(config) == 204
    assert transfer.expected_points(config, smoke=True) == 72


@pytest.mark.parametrize("setting", [
    {"enabled": "yes"}, {"datasets": ["sop", "sop"]}, {"datasets": ["typo"]}, {"datasets": [["sop"]]},
    {"knn": {"temperature": 0}}, {"knn": {"n_neighbors": [1, 1]}}, {"revisions": {"sop": ""}},
])
def test_invalid_transfer_config_rejected(tmp_path, setting):
    import yaml
    payload = yaml.safe_load(Path("configs/finetune.yaml").read_text())
    payload["transfer"].update(setting)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        load_config(path)
