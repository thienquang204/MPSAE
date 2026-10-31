#!/usr/bin/env python3
"""Cheap CPU contract checks for the real research kernels and OT adapter.

No downloads, ImageNet files, GPU, torchvision, or training runs are needed.
Run: python check_reference.py   (or python -m unittest check_reference)
"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import unittest
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import mpsae_repair as core


def historical_symbols():
    """Load actual baseline definitions without importing its image stack."""
    path = Path(__file__).with_name("csr_vs_mmpot_imagenet.py")
    names = {"TopKSAE", "circular_variance_cost", "kl_constraint", "greenkhorn_mmpot",
             "entropy_term", "partial_matching_gap"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if len(selected) != len(names):
        raise AssertionError("baseline definitions changed; update explicit compatibility test")
    scope = dict(torch=torch, nn=nn, F=F, math=math, Dict=Dict, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def check_baseline_on_features(x, *, k=8, hidden_dim=8192, seed=1227):
    """Compare the actual adapter against historical code on supplied fit rows.

    This helper does not fetch data or select rows. Its caller must supply a
    training-only slice. CPU initialization has a private, restored RNG stream.
    Both models are fresh and identical; near-threshold activity counters make
    the ordering of the historical dead-feature mask observable.
    """
    if x.ndim != 2 or len(x) < 2 or not bool(torch.isfinite(x).all()):
        raise ValueError("baseline equivalence requires >=2 finite fitting rows")
    x = x.detach().float()
    historical = historical_symbols()
    legacy_type = historical["TopKSAE"]
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        actual_model = core.ResearchSAE(x.shape[1], hidden_dim, dead_steps=30)
        baseline = legacy_type(x.shape[1], hidden_dim, dead_steps=30)
    actual_model = actual_model.to(x.device)
    baseline = baseline.to(x.device)
    with torch.no_grad():
        actual_model.inactive_steps.copy_(torch.arange(hidden_dim, device=x.device) % 5 + 28)
    baseline.load_state_dict(actual_model.state_dict())
    baseline_reconstruction, _, raw = baseline.reconstruction_losses(
        x, k, 512, 1., 1/8, 1/32)
    historical_gap = historical["partial_matching_gap"]
    baseline_gap, _ = historical_gap(*raw, .9, .2, 100, 1e-4, min(32, len(x)))
    expected = baseline_reconstruction + 2*baseline_gap
    actual, _, _ = core.research_loss(actual_model, x, k, k_aux=512,
        microbatch=min(32, len(x)), require_feasible=False)
    # Sparse reductions and dense GEMM need not accumulate in the same order.
    # These are explicit float32 tolerances; the objective has no scale change.
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.backward()
    expected.backward()
    baseline_parameters = dict(baseline.named_parameters())
    differences = {}
    for name, parameter in actual_model.named_parameters():
        reference = baseline_parameters[name].grad
        observed = parameter.grad
        if reference is None or observed is None or not bool(torch.isfinite(observed).all()):
            raise AssertionError(f"missing/nonfinite baseline-equivalence gradient for {name}")
        torch.testing.assert_close(observed, reference, atol=2e-6, rtol=5e-4, msg=name)
        difference = (observed-reference).norm()
        norm = reference.norm()
        relative = float(difference / norm.clamp_min(1e-12))
        if float(norm) > 1e-8 and relative > 1e-3:
            raise AssertionError(f"relative gradient norm discrepancy for {name}: {relative}")
        differences[name] = {"max_absolute_difference": float((observed-reference).abs().max()),
                             "reference_norm": float(norm), "relative_l2_difference": relative}
    if not torch.equal(actual_model.inactive_steps, baseline.inactive_steps):
        raise AssertionError("actual training-slice baseline activity counters differ")
    return {"status": "passed", "rows": len(x), "input_dim": x.shape[1], "hidden_dim": hidden_dim,
            "K": k, "baseline_loss": float(expected.detach()), "adapter_loss": float(actual.detach()),
            "loss_absolute_difference": float((actual-expected).detach().abs()),
            "gradient_differences": differences, "activity_counters_identical": True,
            "initial_counter_values": [28, 29, 30, 31, 32],
            "tolerances": {"loss_atol": 2e-5, "loss_rtol": 2e-5,
                           "gradient_atol": 2e-6, "gradient_rtol": 5e-4,
                           "relative_gradient_l2": 1e-3},
            "scope": "actual fitting features, matched fresh SAE, historical unrefined solver; repair/masks/KL/new cost disabled"}


class ResearchKernelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.historical = historical_symbols()

    def setUp(self):
        torch.manual_seed(17)

    def model(self):
        model = core.ResearchSAE(6, 24, 3)
        with torch.no_grad():
            model.encoder_bias.fill_(.2)
        return model

    def test_signed_deterministic_projection_and_nested_gate(self):
        a = torch.tensor([[2., 2., -7., 1., 2.], [-1., -2., -3., -4., -5.]], requires_grad=True)
        z = core.nonnegative_topk(a, 2)
        torch.testing.assert_close(z, torch.tensor([[2., 2., 0., 0., 0.], [0., 0., 0., 0., 0.]]))
        torch.testing.assert_close(core.nonnegative_topk(core.nonnegative_topk(a, 4), 2), z)
        z.sum().backward()
        torch.testing.assert_close(a.grad, z.gt(0).float())

    def test_fixed_iht_and_zero_initialized_factor(self):
        sae = self.model()
        x = torch.randn(5, 6)
        alpha = .9 / float(torch.linalg.matrix_norm(sae.decoder, ord=2).square())
        alpha, record = core.backtrack_alpha(sae, x, 2, alpha)
        repair = core.RepairModule(6, 24, rank=3, alpha=alpha)
        raw = sae.encode(x, 2)
        expected = core.nonnegative_topk(raw + alpha*((x-sae.decode(raw))@sae.decoder.T), 2)
        torch.testing.assert_close(repair(sae, x, raw, 2), expected)
        self.assertLessEqual(record["fixed_iht_max_mse_increase"], 1e-7)
        loss = F.mse_loss(sae.decode(repair(sae, x, raw, 2)), x)
        loss.backward()
        self.assertGreater(float(repair.V.grad.norm()), 0.)
        self.assertEqual(float(repair.U.grad.norm()), 0.)

    def test_power_alpha_reproducible_and_rng_isolated(self):
        sae = self.model()
        state = torch.random.get_rng_state().clone()
        alpha = core.estimate_alpha(sae.decoder)
        self.assertEqual(alpha, core.estimate_alpha(sae.decoder))
        core.RepairModule(6, 24, seed=8)
        torch.testing.assert_close(torch.random.get_rng_state(), state)

    def test_mask_probabilities_and_stream(self):
        z = torch.ones(12000, 8)
        gen = torch.Generator().manual_seed(42)
        masked, stats = core.mask_start(z, 8, gen)
        self.assertAlmostEqual(float(stats["mask_erase_fraction"]), .2, delta=.006)
        self.assertAlmostEqual(float(stats["forced_clean_row_fraction"]), .2, delta=.012)
        self.assertAlmostEqual(float(stats["unchanged_start_fraction"]), .2+.8*.75**8, delta=.012)
        self.assertTrue(bool(((masked == 0) | (masked == 1)).all()))
        same, _ = core.mask_start(z, 8, torch.Generator().manual_seed(42))
        torch.testing.assert_close(same, masked)

    def test_disabled_components_match_real_baseline_loss_gradient_and_counters(self):
        legacy = self.historical["TopKSAE"](6, 24, dead_steps=3)
        research = self.model()
        with torch.no_grad():
            research.inactive_steps.copy_(torch.arange(24) % 7)
        legacy.load_state_dict(research.state_dict())
        x = torch.randn(7, 6)
        baseline_recon, _, views = legacy.reconstruction_losses(x, 2, 8, 1., 1/8, 1/32)
        baseline_gap, _ = self.historical["partial_matching_gap"](*views, .9, .2, 100, 1e-4, 3)
        expected = baseline_recon + 2*baseline_gap
        actual, _, _ = core.research_loss(research, x, 2, k_aux=8, microbatch=3, require_feasible=False)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        expected.backward()
        actual.backward()
        for (name, param), (_, ref) in zip(research.named_parameters(), legacy.named_parameters()):
            torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-5, msg=name)
        torch.testing.assert_close(research.inactive_steps, legacy.inactive_steps)

    def test_bank_and_diagnostic_encodings_have_no_activity_side_effect(self):
        sae = self.model()
        x = torch.randn(8, 6)
        snapshot = sae.inactive_steps.clone()
        bank = core.LandmarkBank(torch.arange(100, 108), x)
        bank.refresh(sae, None, 2, batch_size=3)
        core.encode_repaired(sae, None, x, 2)
        core.reconstruction_terms(sae, x, 2, 8, update_activity=False)
        torch.testing.assert_close(sae.inactive_steps, snapshot)

    def test_indexed_profiles_match_dense_and_detach_keys(self):
        sae = self.model()
        teacher = torch.randn(9, 6, requires_grad=True)
        bank = core.LandmarkBank(torch.arange(9), teacher)
        bank.refresh(sae, None, 2)
        z = sae.encode(torch.randn(4, 6), 2)
        indexed, _ = core.indexed_cosines(z, 2, bank.normalized_transposes[0], anchor_chunk=3)
        dense = F.normalize(z, dim=1) @ F.normalize(bank.codes[0], dim=1).T
        torch.testing.assert_close(indexed, dense)
        g1 = torch.autograd.grad(indexed.sum(), z, retain_graph=True)[0]
        g2 = torch.autograd.grad(dense.sum(), z)[0]
        # Hard Top-K paths only have derivatives on selected positive values.
        torch.testing.assert_close(g1[z > 0], g2[z > 0])
        self.assertFalse(bank.teacher_features.requires_grad)
        self.assertTrue(all(not key.requires_grad for key in bank.codes))
        self.assertIsNone(teacher.grad)

    def test_sparse_start_decode_matches_dense_values_and_gradients(self):
        sae = self.model()
        z = sae.encode(torch.randn(4, 6), 2)
        sparse = core.indexed_decode(z, sae.decoder, 2)
        dense = z @ sae.decoder
        torch.testing.assert_close(sparse, dense)
        sparse_grads = torch.autograd.grad(sparse.square().mean(), (z, sae.decoder), retain_graph=True)
        dense_grads = torch.autograd.grad(dense.square().mean(), (z, sae.decoder))
        torch.testing.assert_close(sparse_grads[0][z > 0], dense_grads[0][z > 0])
        torch.testing.assert_close(sparse_grads[1], dense_grads[1])

    def test_shared_minibatch_exclusion_and_bank_resume(self):
        sae = self.model()
        bank = core.LandmarkBank(torch.arange(8), torch.randn(8, 6))
        bank.refresh(sae, None, 2, step=25)
        x = torch.randn(3, 6)
        profiles = bank.profiles(x, torch.tensor([0, 3, 88]), sae.encode(x, 2), sae.encode(x, 4), k=2)
        self.assertEqual(profiles["teacher"].shape, (3, 6))
        self.assertEqual(profiles["allowed"].tolist(), [False, True, True, False, True, True, True, True])
        clone = core.LandmarkBank(torch.arange(8), bank.teacher_features)
        clone.load_state_dict(bank.state_dict())
        for a, b in zip(clone.codes, bank.codes):
            torch.testing.assert_close(a, b)
        self.assertEqual(clone.last_refresh_step, 25)
        with self.assertRaises(ValueError):
            bank.profiles(x, torch.arange(8), sae.encode(x, 2), sae.encode(x, 4), k=2)

    def test_affinity_properties_and_teacher_disagreement(self):
        student = torch.tensor([[.1, .1, .8]])
        teacher = torch.tensor([[.8, .1, .1]])
        self.assertAlmostEqual(float(core.neighborhood_cost(student.log(), student.log(), teacher.log())), .3, places=6)
        self.assertAlmostEqual(float(core.neighborhood_cost(student.log(), student.log(), student.log())), 0., places=6)
        probs = torch.randn(4, 6).softmax(1)
        cost = core.neighborhood_cost(probs.log(), probs.log(), probs.log())
        self.assertGreaterEqual(float(cost.min()), -1e-6)
        self.assertLessEqual(float(cost.max()), 1+1e-6)

    def test_no_overlap_and_k1_gradient_starvation(self):
        z = torch.tensor([[2., 1., 0., 0.]], requires_grad=True)
        bank = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
        similarities, overlaps = core.indexed_cosines(z, 2, bank)
        similarities.sum().backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z))
        self.assertFalse(bool(overlaps.any()))
        one = torch.tensor([[3., 0.]], requires_grad=True)
        core.indexed_cosines(one, 1, torch.eye(2))[0].sum().backward()
        self.assertAlmostEqual(float(one.grad[0, 0]), 0., places=6)

    def test_actual_solver_and_envelope_gradient(self):
        cost = torch.rand(3, 3, 3, requires_grad=True)*.5
        loss, stats, delta = core.matching_gap_from_cost(cost, tol=1e-5, max_refine_iters=4000)
        gradient = torch.autograd.grad(loss, cost)[0]
        torch.testing.assert_close(gradient, delta, atol=1e-7, rtol=1e-7)
        self.assertLessEqual(stats["mass_error"], 1e-5)
        self.assertLessEqual(stats["cap_violation"], 1e-5)
        plan, slacks, _ = self.historical["greenkhorn_mmpot"](cost.detach(), .9, .2, 4000, 1e-12)
        original = core.greenkhorn_mmpot(cost.detach(), .9, .2, 4000, 1e-12)[0]
        torch.testing.assert_close(plan, original, atol=1e-7, rtol=1e-6)
        for slack in slacks:
            self.assertTrue(bool(slack.ge(0).all()))

    def test_batched_solver_preserves_independent_problems(self):
        costs = torch.rand(3, 3, 3, 3)*.6
        plans, slacks, records = core.batched_greenkhorn_mmpot(costs, .9, .2, 2000, 1e-10)
        for index, cost in enumerate(costs):
            plan, slack, record = core.greenkhorn_mmpot(cost, .9, .2, 2000, 1e-10)
            torch.testing.assert_close(plans[index], plan, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(slacks[index], torch.stack(slack), atol=1e-5, rtol=1e-4)
            self.assertAlmostEqual(records[index]["mass"], record["mass"], delta=2e-5)

    def test_converged_actual_solver_finite_difference_full_gap(self):
        # Float32 KL stopping below machine precision is not a converged
        # reference and previously produced a misleading finite-difference fail.
        cost = torch.tensor([[[.11, .31], [.27, .48]], [[.42, .33], [.19, .14]]],
                            dtype=torch.float64, requires_grad=True)
        def objective(c):
            return core.matching_gap_from_cost(c, max_iters=10000, tol=1e-18,
                require_feasible=False, solver_dtype=torch.float64, stable_kl=True)[0]
        loss = objective(cost)
        derivative = torch.autograd.grad(loss, cost)[0]
        epsilon = .0002
        for index in ((0,0,0), (0,1,1), (1,0,1)):
            plus, minus = cost.detach().clone(), cost.detach().clone()
            plus[index] += epsilon
            minus[index] -= epsilon
            difference = (float(objective(plus))-float(objective(minus)))/(2*epsilon)
            self.assertAlmostEqual(difference, float(derivative[index]), delta=.0002)

    def test_mixed_cost_uses_shared_plan_for_component_gradients(self):
        codes = [torch.rand(4, 12, requires_grad=True) for _ in range(3)]
        logs = [torch.randn(4, 6, requires_grad=True).log_softmax(1) for _ in range(2)]
        profiles = {"student_logs": logs, "teacher_log": torch.randn(4, 6).log_softmax(1)}
        gap, _, context = core.mixed_matching_gap(*codes, profiles=profiles, mixture_weight=.5,
                                                microbatch=4, diagnostics=True)
        variables = [*codes, *logs]
        expected = torch.autograd.grad(gap, variables, retain_graph=True, allow_unused=True)
        actual = torch.autograd.grad(context["old_ot"]+context["new_ot"], variables, retain_graph=True, allow_unused=True)
        for a, b in zip(expected, actual):
            if a is not None:
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        self.assertIsNone(torch.autograd.grad(context["new_ot"], codes[2], allow_unused=True)[0])

    def test_schedule_has_one_auxiliary_and_fixed_served_budget(self):
        self.assertEqual(core.auxiliary_budget(8, 8192, 0, 250, "anneal"), 64)
        self.assertEqual(core.auxiliary_budget(8, 8192, 175, 250, "anneal"), 32)
        self.assertEqual(core.auxiliary_budget(8, 8192, 249, 250, "anneal"), 32)
        sae = self.model()
        x = torch.randn(4, 6)
        total, stats, views = core.reconstruction_terms(sae, x, 2, 16)
        torch.testing.assert_close(total, stats["mse_final_K"]+stats["mse_raw_aux"]/8+stats["loss_dead"]/32)
        self.assertTrue(bool(views["final_k"].gt(0).sum(1).le(2).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
