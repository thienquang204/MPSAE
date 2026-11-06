"""Our implementation versus the published CSR / CSRv2 source.

These are the tests that make the baseline claim defensible: the architecture,
the loss terms, the dead-latent bookkeeping and one optimiser step must agree
with ``tests/reference/csr_released.py``, which is a transcription of the
authors' code. Settings are ours; the algorithm is theirs.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

# Must precede every other import here: both the oracle and the mpsae modules
# import torch at module scope, so a skip mark alone would fail at collection.
pytest.importorskip("torch")

from reference import csr_released as released  # noqa: E402

from mpsae import losses, methods  # noqa: E402
from mpsae.config import FF2048_PRESET  # noqa: E402
from mpsae.model import build_sae  # noqa: E402

DIM, HIDDEN, BATCH, K, K_AUX = 12, 64, 16, 3, 8


def _paired_models(torch, *, seed=1227):
    """One of ours and one of theirs, sharing identical parameters."""
    ours = build_sae(DIM, HIDDEN, seed=seed, dead_threshold=30)
    theirs = released.CSR(n_latents=HIDDEN, topk=K, auxk=K_AUX, normalize=False,
                          n_inputs=DIM, dead_threshold=30)
    theirs.load_state_dict({
        "pre_bias": ours.pre_bias.detach().clone(),
        "encoder.weight": ours.encoder.weight.detach().clone(),
        "latent_bias": ours.latent_bias.detach().clone(),
        "stats_last_nonzero": ours.stats_last_nonzero.detach().clone(),
    }, strict=True)
    return ours, theirs


def test_parameter_shapes_and_initialisation_match(torch):
    ours, theirs = _paired_models(torch)
    assert ours.pre_bias.shape == theirs.pre_bias.shape
    assert ours.encoder.weight.shape == theirs.encoder.weight.shape
    assert ours.latent_bias.shape == theirs.latent_bias.shape
    assert torch.equal(ours.pre_bias, torch.zeros(DIM))
    assert torch.equal(ours.latent_bias, torch.zeros(HIDDEN))
    # Tied, and never renormalised.
    assert torch.equal(ours.decode(torch.zeros(1, HIDDEN)), ours.pre_bias[None, :])


def test_pre_activations_and_gate_match(torch, batch):
    ours, theirs = _paired_models(torch)
    assert torch.allclose(ours.pre_activations(batch), theirs.encode_pre_act(batch), atol=1e-6)
    mine, _ = ours.gate(ours.pre_activations(batch), K)
    theirs_k, _ = theirs.top_k(theirs.encode_pre_act(batch), K)
    assert torch.allclose(mine, theirs_k, atol=1e-6)
    assert int(mine.gt(0).sum(1).max()) <= K


def test_dead_latent_counter_matches_call_for_call(torch, batch):
    """The counter is state: it must advance identically, in the same order."""
    ours, theirs = _paired_models(torch)
    for _ in range(14):
        for budget in (K, 4 * K, K):
            pre_ours = ours.pre_activations(batch)
            ours.count_activity(pre_ours, budget)
            theirs.top_k(theirs.encode_pre_act(batch), budget)
        assert torch.equal(ours.stats_last_nonzero, theirs.stats_last_nonzero)
    assert torch.equal(ours.dead_mask(), theirs.stats_last_nonzero > 30)


def test_auxiliary_latents_match_including_the_inplace_mask(torch, batch):
    """Their mask is an untracked ``.data`` write; ours is straight-through."""
    ours, theirs = _paired_models(torch)
    # Drive some latents past the threshold first, so the mask is non-trivial.
    for _ in range(12):
        pre = ours.pre_activations(batch)
        ours.count_activity(pre, K)
        theirs.top_k(theirs.encode_pre_act(batch), K)
    pre_ours = ours.pre_activations(batch)
    ours.count_activity(pre_ours, K)
    mine = ours.auxiliary_latents(pre_ours, K_AUX)
    _, theirs_aux = theirs.top_k(theirs.encode_pre_act(batch), K)
    assert torch.allclose(mine, theirs_aux, atol=1e-6)


def test_auxiliary_gradient_matches_when_nothing_is_dead_yet(torch, batch):
    """A plain ``pre * mask`` would zero this gradient; the release does not.

    For the first ten or so steps no latent has crossed the threshold, so the
    masked tensor is all zeros and Top-K selects arbitrary positions. The
    release still back-propagates an unweighted scatter to those positions.
    """
    ours, _ = _paired_models(torch)
    pre = ours.pre_activations(batch)
    ours.count_activity(pre, K)
    assert not bool(ours.dead_mask().any())
    latents = ours.auxiliary_latents(pre, K_AUX)
    assert torch.equal(latents, torch.zeros_like(latents))
    grad, = torch.autograd.grad(latents.sum(), ours.encoder.weight, allow_unused=True)
    assert grad is not None and bool(grad.abs().sum() > 0)


def test_normalized_mse_matches(torch, batch):
    criterion = torch.nn.MSELoss()
    target = torch.randn_like(batch)
    prediction = torch.randn_like(batch)
    assert torch.allclose(
        losses.normalized_mse(prediction, target),
        released.normalized_mse(prediction, target, criterion).nan_to_num(0), atol=1e-6)


def test_normalized_mse_absorbs_a_constant_target(torch):
    constant = torch.ones(8, 4)
    assert float(losses.normalized_mse(torch.zeros(8, 4), constant)) == 0.0


def test_inbatch_cl_matches_the_release(torch):
    generator = torch.Generator().manual_seed(3)
    codes = torch.relu(torch.randn(16, HIDDEN, generator=generator))
    assert torch.allclose(losses.inbatch_cl(codes, codes),
                          released.inbatch_cl(codes, codes), atol=1e-5)
    assert torch.allclose(losses.inbatch_cl_released(codes, codes),
                          released.inbatch_cl(codes, codes), atol=1e-6)


def test_label_based_cl_matches_the_release(torch, labels):
    generator = torch.Generator().manual_seed(4)
    codes = torch.relu(torch.randn(16, HIDDEN, generator=generator))
    assert torch.allclose(losses.label_based_cl(codes, codes, labels),
                          released.label_based_cl(codes, codes, labels), atol=1e-5)


def test_label_based_cl_degenerates_on_constant_labels(torch):
    """Why our archives store ``labels``: the released key mismatch makes
    every label zero, and then the advertised supervision does nothing."""
    generator = torch.Generator().manual_seed(5)
    codes = torch.relu(torch.randn(16, HIDDEN, generator=generator))
    constant = torch.zeros(16, dtype=torch.long)
    assert float(released.label_based_cl(codes, codes, constant)) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("method,use_label_cl", [("csr", False), ("csrv2", True)])
def test_full_objective_matches_the_release(torch, batch, labels, method, use_label_cl):
    preset = replace(FF2048_PRESET, hidden_dim=HIDDEN, k_aux=K_AUX)
    ours, theirs = _paired_models(torch)
    # csrv2 anneals, so pin the step where its budget equals K to compare like
    # with like; the schedule itself is covered by test_schedules.
    total_steps = 10
    step = total_steps - 1
    budget = methods.plan_budget(method, K, preset, step=step, total_steps=total_steps)
    assert budget == K

    output = methods.compute_step(method, ours, batch, k=K, preset=preset,
                                  step=step, total_steps=total_steps, labels=labels)
    expected, parts = released.released_step(
        theirs, batch, topk=K, auxk_coef=preset.aux_coef, cl_coef=preset.cl_coef,
        use_label_cl=use_label_cl, target=labels)

    assert torch.allclose(output.loss, expected, atol=1e-5), \
        f"{float(output.loss)} vs {float(expected)}"
    assert output.metrics["loss_mse_k"] == pytest.approx(float(parts["loss_k"]), abs=1e-6)
    assert output.metrics["loss_mse_4k"] == pytest.approx(float(parts["loss_4k"]), abs=1e-6)
    assert output.metrics["loss_aux"] == pytest.approx(float(parts["loss_auxk"]), abs=1e-6)
    assert output.metrics["loss_contrastive"] == pytest.approx(float(parts["loss_cl"]), abs=1e-5)
    # Three gates per step, same as the release's three forward passes.
    assert torch.equal(ours.stats_last_nonzero, theirs.stats_last_nonzero)


def test_gradients_match_the_release(torch, batch, labels):
    preset = replace(FF2048_PRESET, hidden_dim=HIDDEN, k_aux=K_AUX)
    ours, theirs = _paired_models(torch)
    output = methods.compute_step("csr", ours, batch, k=K, preset=preset,
                                  step=0, total_steps=1)
    output.loss.backward()
    expected, _ = released.released_step(theirs, batch, topk=K, auxk_coef=preset.aux_coef,
                                         cl_coef=preset.cl_coef)
    expected.backward()
    for name in ("pre_bias", "latent_bias"):
        assert torch.allclose(getattr(ours, name).grad, getattr(theirs, name).grad, atol=1e-6), name
    assert torch.allclose(ours.encoder.weight.grad, theirs.encoder.weight.grad, atol=1e-6)


def test_three_adam_steps_match_the_release(torch, batch):
    """Coupled L2 on every parameter, eps 6.25e-10, constant rate."""
    preset = replace(FF2048_PRESET, hidden_dim=HIDDEN, k_aux=K_AUX)
    ours, theirs = _paired_models(torch)
    mine = torch.optim.Adam(ours.parameters(), lr=preset.lr, betas=preset.adam_betas,
                            eps=preset.adam_eps, weight_decay=preset.weight_decay)
    yours = torch.optim.Adam(theirs.parameters(), lr=preset.lr, betas=(0.9, 0.999),
                             eps=6.25e-10, weight_decay=preset.weight_decay)
    for step in range(3):
        output = methods.compute_step("csr", ours, batch, k=K, preset=preset,
                                      step=step, total_steps=3)
        mine.zero_grad(set_to_none=True)
        output.loss.backward()
        mine.step()
        expected, _ = released.released_step(theirs, batch, topk=K, auxk_coef=preset.aux_coef,
                                             cl_coef=preset.cl_coef)
        yours.zero_grad(set_to_none=True)
        expected.backward()
        yours.step()
        assert torch.allclose(ours.encoder.weight, theirs.encoder.weight, atol=1e-6), step
