"""Numerical equivalence of the optimized training path to its references."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from mpsae import losses, methods
from mpsae.config import FF2048_PRESET, METHODS
from mpsae.model import build_sae


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("kind", ["random", "zero", "mixed", "near_epsilon"])
@pytest.mark.parametrize("label_kind", ["none", "mixed", "constant"])
@pytest.mark.parametrize("temperature", [0.2, 1.0])
def test_duplicate_view_loss_and_gradient_match_generic(batch, kind, label_kind, temperature):
    generator = torch.Generator().manual_seed(741)
    values = torch.rand(batch, 23, generator=generator)
    if kind == "zero":
        values.zero_()
    elif kind == "mixed":
        values[::2] = 0
    elif kind == "near_epsilon":
        values *= 1e-13
    labels = None if label_kind == "none" else (
        torch.arange(batch) % 3 if label_kind == "mixed" else torch.zeros(batch, dtype=torch.long))
    optimized = values.clone().requires_grad_(True)
    reference = values.clone().requires_grad_(True)
    actual = losses.duplicate_view_cl(optimized, labels, temperature)
    expected = (losses.inbatch_cl(reference, reference, temperature) if labels is None else
                losses.label_based_cl(reference, reference, labels, temperature))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual_gradient, = torch.autograd.grad(actual, optimized)
    expected_gradient, = torch.autograd.grad(expected, reference)
    # Near normalization's epsilon, gradients can be enormous. Compare their
    # common scale rather than accepting a large absolute tolerance for all.
    scale = max(1.0, float(expected_gradient.abs().max()))
    torch.testing.assert_close(actual_gradient / scale, expected_gradient / scale,
                               atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("sequence", [(3, 12, 3), (3, 12, 6)])
def test_reused_selections_preserve_activity_and_auxiliary_mask_with_ties(sequence):
    optimized = build_sae(5, 16, seed=7, dead_threshold=2)
    reference = deepcopy(optimized)
    pre = torch.tensor([
        [0., 0., 0., 0., 1., 1., 1., 1., -2., -2., 3., 3., 0., 1e-6, 1e-5, 2e-5],
        [1., 1., 1., 1., 0., 0., 0., 0., -1., -1., 2., 2., 0., 0., 0., 0.],
    ])
    for step in range(5):
        for index, k in enumerate(sequence):
            z, indices, values = optimized.gate_with_selection(pre, k)
            reference_z, reference_indices = reference.gate(pre, k)
            optimized.count_selected_activity(indices, values)
            reference.count_activity(pre, k)
            assert torch.equal(indices, reference_indices)
            assert torch.equal(z, reference_z)
            assert torch.equal(optimized.stats_last_nonzero, reference.stats_last_nonzero)
            if index == 0:
                assert torch.equal(optimized.auxiliary_latents(pre, 4),
                                   reference.auxiliary_latents(pre, 4))


def _preset():
    return replace(FF2048_PRESET, hidden_dim=32, k_aux=4, batch_size=8,
                   ot_microbatch=4, ot_max_iters=40, ot_max_refine_iters=300)


def _reference_cl_step(method, sae, x, labels, preset, step, total_steps):
    """Pre-optimization execution, using generic doubled contrastive views."""
    k = methods.plan_budget(method, 2, preset, step=step, total_steps=total_steps)
    pre = sae.pre_activations(x)
    main, _ = sae.gate(pre, k)
    sae.count_activity(pre, k)
    aux = sae.auxiliary_latents(pre, preset.k_aux)
    reconstruction = sae.decode(main)
    wide, _ = sae.gate(pre, min(4 * k, sae.hidden_dim))
    sae.count_activity(pre, min(4 * k, sae.hidden_dim))
    middle = None
    if method == "csr_multitopk":
        middle, _ = sae.gate(pre, min(2 * k, sae.hidden_dim))
    sae.count_activity(pre, min(2 * k, sae.hidden_dim) if middle is not None else k)
    result = losses.mse(reconstruction, x) + preset.nested_coef * losses.mse(sae.decode(wide), x)
    result = result + preset.aux_coef * losses.normalized_mse(
        sae.decode(aux), losses.auxk_target(x, reconstruction, sae.pre_bias))
    if middle is not None:
        result = result + preset.nested_coef * losses.mse(sae.decode(middle), x)
    contrastive = (losses.label_based_cl(main, main, labels, preset.cl_temperature)
                   if method == "csrv2" else losses.inbatch_cl(main, main, preset.cl_temperature))
    return result + preset.cl_coef * contrastive


@pytest.mark.parametrize("method", ["csr", "csrv2", "csr_multitopk"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimized_full_steps_preserve_gradients_updates_and_counters(method, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA numerical parity requires a GPU")
    preset = _preset()
    optimized = build_sae(6, preset.hidden_dim, seed=91, dead_threshold=2, device=device)
    reference = deepcopy(optimized)
    optimizers = [torch.optim.Adam(model.parameters(), lr=preset.lr,
                  weight_decay=preset.weight_decay, eps=preset.adam_eps, betas=preset.adam_betas)
                  for model in (optimized, reference)]
    generator = torch.Generator().manual_seed(19)
    labels = torch.arange(8, device=device) % 3
    # Includes wide and final annealed K, and activity counters across updates.
    for step in range(5):
        batch = torch.randn(8, 6, generator=generator).to(device)
        actual = methods.compute_step(method, optimized, batch, k=2, preset=preset,
                                      step=step, total_steps=5, labels=labels,
                                      collect_metrics=bool(step % 2)).loss
        expected = _reference_cl_step(method, reference, batch, labels, preset, step, 5)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        for optimizer, loss in zip(optimizers, (actual, expected)):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        for mine, theirs in zip(optimized.parameters(), reference.parameters()):
            torch.testing.assert_close(mine.grad, theirs.grad, atol=1e-6, rtol=1e-5)
        for optimizer in optimizers:
            optimizer.step()
        for mine, theirs in zip(optimized.parameters(), reference.parameters()):
            torch.testing.assert_close(mine, theirs, atol=1e-6, rtol=1e-5)
        assert torch.equal(optimized.stats_last_nonzero, reference.stats_last_nonzero)


@pytest.mark.parametrize("method", METHODS)
def test_metrics_collection_does_not_change_loss_gradient_or_activity(method, monkeypatch):
    preset = _preset()
    recorded = build_sae(6, preset.hidden_dim, seed=21)
    quiet = deepcopy(recorded)
    batch = torch.randn(8, 6, generator=torch.Generator().manual_seed(1))
    labels = torch.arange(8) % 3
    measured = methods.compute_step(method, recorded, batch, k=2, preset=preset,
                                    step=0, total_steps=1, labels=labels)

    def unwanted(*args, **kwargs):
        raise AssertionError("unlogged steps must not compute optional sparsity diagnostics")

    monkeypatch.setattr(quiet, "sparsity_tensors", unwanted)
    minimal = methods.compute_step(method, quiet, batch, k=2, preset=preset,
                                   step=0, total_steps=1, labels=labels, collect_metrics=False)
    assert minimal.metrics == {}
    assert measured.metrics["loss_total"] == pytest.approx(float(measured.loss.detach()))
    assert isinstance(measured.metrics["nnz_min"], int)
    assert isinstance(measured.metrics["active_units"], int)
    torch.testing.assert_close(measured.loss, minimal.loss, atol=1e-6, rtol=1e-6)
    measured.loss.backward()
    minimal.loss.backward()
    for mine, theirs in zip(recorded.parameters(), quiet.parameters()):
        torch.testing.assert_close(mine.grad, theirs.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(recorded.stats_last_nonzero, quiet.stats_last_nonzero)


def test_unlogged_step_still_rejects_nonfinite_loss():
    preset = _preset()
    model = build_sae(6, preset.hidden_dim, seed=21)
    bad = torch.full((8, 6), float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        methods.compute_step("csr", model, bad, k=2, preset=preset,
                             step=0, total_steps=1, collect_metrics=False)
