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

from mpsae import losses, methods  # noqa: E402
from mpsae.config import FF2048_PRESET, IMPLEMENTED_METHODS as METHODS  # noqa: E402
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
        "mpsaev2": {"mse_k", "mse_4k", "aux", "ot", "contrastive"},
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
    """The common reconstruction terms match; the control also adds R_2K/8
    and exchanges the OT regularizer for instance contrastive learning."""
    _, ours = _run(torch, "mpsae", preset, batch, labels)
    _, ablation = _run(torch, "csr_multitopk", preset, batch, labels)
    for shared in ("mse_k", "mse_4k", "aux"):
        assert float(ours.components[shared]) == pytest.approx(
            float(ablation.components[shared]), rel=1e-6), shared


def test_annealed_methods_start_wide_and_end_at_the_target(torch, preset, batch, labels):
    total = 100
    for method in ("csrv2",):
        start = methods.plan_budget(method, K, preset, step=0, total_steps=total)
        end = methods.plan_budget(method, K, preset, step=total - 1, total_steps=total)
        assert start == methods.initial_topk(K, HIDDEN) > K, method
        assert end == K, method
    for method in ("csr", "mpsae", "mpsaev2", "csr_multitopk"):
        for step in (0, 35, 70, 99):
            assert methods.plan_budget(method, K, preset, step=step, total_steps=total) == K, method


@pytest.mark.parametrize("step", [0, 3, 9])
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("collect_metrics", [True, False])
def test_mpsaev2_adds_instance_cl_to_fixed_mpsae_in_loss_and_gradient(
        torch, preset, batch, labels, step, gamma, collect_metrics):
    """Both regularizers must contribute, with no curriculum or R_2K term."""
    preset = replace(preset, cl_coef=gamma)
    hybrid = _model(torch)
    reference = _model(torch)
    actual = methods.compute_step("mpsaev2", hybrid, batch, k=K, preset=preset,
                                   step=step, total_steps=10, labels=None,
                                   collect_metrics=collect_metrics)
    base = methods.compute_step("mpsae", reference, batch, k=K, preset=preset,
                                 step=step, total_steps=10, labels=None,
                                 collect_metrics=collect_metrics)
    # Use the generic doubled-view implementation, not the optimized helper.
    main, _ = reference.gate(reference.pre_activations(batch), K)
    contrastive = losses.inbatch_cl(main, main, preset.cl_temperature)
    expected = base.loss + gamma * contrastive
    assert actual.training_k == base.training_k == K
    assert set(actual.components) == set(base.components) | {"contrastive"}
    assert "mse_2k" not in actual.components
    for name in base.components:
        torch.testing.assert_close(actual.components[name], base.components[name])
    torch.testing.assert_close(actual.components["contrastive"], gamma * contrastive,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.loss, expected, atol=1e-6, rtol=1e-6)
    actual.loss.backward()
    expected.backward()
    for mine, theirs in zip(hybrid.parameters(), reference.parameters()):
        torch.testing.assert_close(mine.grad, theirs.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(hybrid.stats_last_nonzero, reference.stats_last_nonzero)
    if collect_metrics:
        assert actual.metrics["loss_contrastive"] == pytest.approx(float(contrastive.detach()))
        assert actual.metrics["loss_ot"] == base.metrics["loss_ot"]
        assert (actual.metrics["training_k"], actual.metrics["view_2k"],
                actual.metrics["view_4k"]) == (K, 2 * K, 4 * K)
    else:
        assert actual.metrics == {}


def test_mpsaev2_ignores_labels_and_records_the_fixed_hybrid_recipe(torch, preset, batch, labels):
    _, unsupervised = _run(torch, "mpsaev2", preset, batch, None, total=10)
    _, with_labels = _run(torch, "mpsaev2", preset, batch, labels, total=10)
    torch.testing.assert_close(unsupervised.loss, with_labels.loss, atol=0, rtol=0)
    described = methods.describe("mpsaev2", K, preset)
    assert described["variant"] == "fixed_ot_cl_v1"
    assert "annealing" not in described
    assert described["gate_sequence"] == ["k", "4k", "2k"]
    assert described["contrastive"]["loss"] == "inbatch_cl"
    assert described["contrastive"]["weight"] == preset.cl_coef
    assert described["ot"]["weight"] == preset.ot_weight
    assert described["reconstruction"] == ["L(k)", "L(4k)/8"]


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
