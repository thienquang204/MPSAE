"""Replay preserves full-batch losses and stochastic backbone gradients."""
from copy import deepcopy

import pytest

from test_finetune import fixture_models, augment


@pytest.mark.parametrize("method", ["csr", "csrv2", "mpsae"])
def test_replay_matches_retained_graph_with_dropout_and_full_batch_loss(method):
    import torch
    from mpsae.finetune.config import DEFAULTS
    from mpsae.finetune.train import cached_backward, feature_loss, forward_features
    cell, spec, net, teacher, sae, head = fixture_models(method)
    replica, sae2, head2 = deepcopy(net), deepcopy(sae), deepcopy(head)
    images, labels = torch.randn(7, 3, 8, 8), torch.arange(7) % 4
    chunks = list(zip(images.split(3), labels.split(3)))
    options = {**DEFAULTS, "precision": "float32"}
    rng = torch.get_rng_state()
    features, targets = [], []
    for x, _ in chunks:
        with torch.no_grad():
            targets.append(forward_features(teacher, x, spec))
        features.append(forward_features(net, x, spec))
    loss, metrics = feature_loss(cell, spec, sae, head, torch.cat(features), torch.cat(targets), labels,
        options=options, step=0, total=6)
    loss.backward()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    replay_loss, replay_metrics = cached_backward(cell, spec, replica, teacher, sae2, head2, chunks,
        device="cpu", options=options, step=0, total=6)
    torch.testing.assert_close(loss, replay_loss)
    assert metrics == replay_metrics
    assert torch.equal(final_rng, torch.get_rng_state())
    for first, second in ((net, replica), (sae, sae2), (head, head2)):
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        for a, b in zip(first.buffers(), second.buffers()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
def test_loader_persistent_epochs_and_resume_match_original(workers, drop_last):
    import torch
    from torch.utils.data import TensorDataset
    from mpsae.finetune.data import TrainingLoader, loader
    dataset = TensorDataset(torch.randn(13, 3, 8, 8), torch.arange(13))
    shared = TrainingLoader(dataset, augment, seed=9, batch_size=5, microbatch=3,
                            workers=workers, drop_last=drop_last)
    try:
        for epoch, skip in ((0, 0), (1, 0), (1, 5), (2, 0)):
            expected = list(loader(dataset, augment, seed=9, epoch=epoch, batch_size=5,
                                   workers=0, skip=skip, training=True, drop_last=drop_last))
            actual = list(shared.batches(epoch, skip))
            assert len(expected) == len(actual)
            for (images, labels), chunks in zip(expected, actual):
                torch.testing.assert_close(images, torch.cat([x for x, _ in chunks]), atol=0, rtol=0)
                assert torch.equal(labels, torch.cat([y for _, y in chunks]))
                assert max(len(x) for x, _ in chunks) <= 3
    finally:
        shared.close()
