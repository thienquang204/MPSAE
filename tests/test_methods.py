"""The five methods must differ from each other in exactly the intended way.

This is the test that protects the experimental design. If ``csr_multitopk``
accidentally stops matching ``csr`` everywhere except the 2K term, the ablation
no longer isolates anything, and the OT claim loses its control.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

# Must precede the mpsae imports: these modules import torch at module scope.
pytest.importorskip("torch")

from mpsae import methods  # noqa: E402
from mpsae.config import FF2048_PRESET, METHODS  # noqa: E402
from mpsae.model import build_sae  # noqa: E402

DIM, HIDDEN, K = 12, 64, 3


@pytest.fixture
def preset():
    return replace(FF2048_PRESET, hidden_dim=HIDDEN, k_aux=8, ot_microbatch=4,
                   ot_max_iters=40, ot_max_refine_iters=200)


def _model(torch, seed=1227):
    return build_sae(DIM, HIDDEN, seed=seed, dead_threshold=30)


def _run(torch, method, preset, batch, labels, *, step=0, total=1):
    model = _model(torch)
    output = methods.compute_step(method, model, batch, k=K, preset=preset,
                                  step=step, total_steps=total, labels=labels)
    return model, output


@pytest.mark.parametrize("method", METHODS)
def test_every_method_gates_exactly_three_times(torch, preset, batch, labels, method):
    """``dead_threshold`` is shared, so the gate count must be too.

    The release gates three times per step (K, 4K, and a duplicate K for the
    contrastive view). Ours spend the third gate on the 2K view they consume.
    """
    model, _ = _run(torch, method, preset, batch, labels)
    # Three gates, each incrementing every counter by one, with resets for the
    # latents that fired. A never-selected latent therefore reads exactly 3.
    assert int(model.stats_last_nonzero.max()) == 3


@pytest.mark.parametrize("method", METHODS)
def test_loss_is_finite_and_components_sum_to_it(torch, preset, batch, labels, method):
    _, output = _run(torch, method, preset, batch, labels)
    assert bool(torch.isfinite(output.loss))
    assert float(output.loss) == pytest.approx(
        float(sum(output.components.values())), rel=1e-6)
    assert output.training_k == K


def test_component_sets_are_as_designed(torch, preset, batch, labels):
    expected = {
        "csr": {"mse_k", "mse_4k", "aux", "contrastive"},
        "csrv2": {"mse_k", "mse_4k", "aux", "contrastive"},
        "mpsae": {"mse_k", "mse_4k", "aux", "ot"},
        "mpsaev2": {"mse_k", "mse_4k", "aux", "ot"},
        "csr_multitopk": {"mse_k", "mse_2k", "mse_4k", "aux", "contrastive"},
    }
    for method, names in expected.items():
        _, output = _run(torch, method, preset, batch, labels)
        assert set(output.components) == names, method


def test_multitopk_differs_from_csr_only_by_the_2k_term(torch, preset, batch, labels):
    """The ablation's whole purpose: same gates, same CL, one extra term."""
    _, plain = _run(torch, "csr", preset, batch, labels)
    _, ablation = _run(torch, "csr_multitopk", preset, batch, labels)
    for shared in ("mse_k", "mse_4k", "aux", "contrastive"):
        assert float(plain.components[shared]) == pytest.approx(
            float(ablation.components[shared]), rel=1e-6), shared
    extra = float(ablation.components["mse_2k"])
    assert extra > 0
    assert float(ablation.loss) == pytest.approx(float(plain.loss) + extra, rel=1e-6)


def test_mpsae_differs_from_multitopk_by_ot_versus_contrastive(torch, preset, batch, labels):
    """Same three gates and the same reconstruction spine; only the
    regulariser changes. That is the comparison the ablation supports."""
    _, ours = _run(torch, "mpsae", preset, batch, labels)
    _, ablation = _run(torch, "csr_multitopk", preset, batch, labels)
    for shared in ("mse_k", "mse_4k", "aux"):
        assert float(ours.components[shared]) == pytest.approx(
            float(ablation.components[shared]), rel=1e-6), shared


def test_annealed_methods_start_wide_and_end_at_the_target(torch, preset, batch, labels):
    total = 100
    for method in ("csrv2", "mpsaev2"):
        start = methods.plan_budget(method, K, preset, step=0, total_steps=total)
        end = methods.plan_budget(method, K, preset, step=total - 1, total_steps=total)
        assert start == methods.initial_topk(K, HIDDEN) > K, method
        assert end == K, method
    for method in ("csr", "mpsae", "csr_multitopk"):
        assert methods.plan_budget(method, K, preset, step=0, total_steps=total) == K, method


def test_mpsaev2_is_csrv2_with_ot_in_place_of_the_contrastive_term(torch, preset, batch, labels):
    """Variant B, as chosen: the *main* budget anneals in both, so the only
    difference is the regulariser. Variant A (fixed main budget, annealed OT
    views) is documented in docs/notes/mpsaev2_legacy_A.md and not implemented.
    """
    total = 10
    for step in (0, 3, 9):
        assert (methods.plan_budget("mpsaev2", K, preset, step=step, total_steps=total)
                == methods.plan_budget("csrv2", K, preset, step=step, total_steps=total))
    _, annealed = _run(torch, "mpsaev2", preset, batch, labels, step=0, total=total)
    _, fixed = _run(torch, "mpsae", preset, batch, labels, step=0, total=total)
    assert annealed.training_k > fixed.training_k
    assert set(annealed.components) == set(fixed.components)


def test_only_csrv2_requires_labels(torch, preset, batch, labels):
    assert methods.requires_labels("csrv2")
    for method in ("csr", "mpsae", "mpsaev2", "csr_multitopk"):
        assert not methods.requires_labels(method)
        _, output = _run(torch, method, preset, batch, None)
        assert bool(torch.isfinite(output.loss))
    with pytest.raises(ValueError, match="label"):
        methods.compute_step("csrv2", _model(torch), batch, k=K, preset=preset,
                             step=0, total_steps=1, labels=None)


def test_rejects_bad_inputs(torch, preset, batch, labels):
    model = _model(torch)
    with pytest.raises(ValueError):
        methods.compute_step("nope", model, batch, k=K, preset=preset, step=0, total_steps=1)
    with pytest.raises(ValueError):
        methods.compute_step("csr", model, batch, k=K, preset=preset, step=5, total_steps=1)
    with pytest.raises(TypeError):
        methods.compute_step("csr", model, batch.double(), k=K, preset=preset,
                             step=0, total_steps=1)


def test_describe_is_serialisable_and_records_the_objective(torch, preset):
    import json

    for method in METHODS:
        described = methods.describe(method, K, preset)
        json.dumps(described)
        assert described["method"] == method
        assert described["aux"]["gates_per_step"] == 3
        assert ("ot" in described) == (method in methods.OT_METHODS)
        assert ("contrastive" in described) == (method in methods.CL_METHODS)
        assert ("annealing" in described) == (method in methods.ANNEALED)
