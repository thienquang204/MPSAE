"""Small CPU checks for invalid cost rejection and bounded objective recovery."""
import copy
import math
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import mpsae_repair as core


@unittest.skipIf(torch is None, "PyTorch is required for OT safety checks")
class TransportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cost = torch.ones(2, 2, 2)
        indices = torch.arange(2)
        self.cost[indices, indices, indices] = 0.

    def bad_solved_plan(self, *, mass_error=0.):
        # Fully feasible, but it matches the expensive off-diagonal triples.
        plan = torch.zeros_like(self.cost)
        plan[0, 1, 1] = .45
        plan[1, 0, 0] = .45
        slacks = [torch.full((2,), .05) for _ in range(3)]
        return plan, slacks, {"mass_error": mass_error, "cap_violation": 0., "iterations": 1}

    def test_invalid_costs_rejected_before_exponential_in_both_solvers(self):
        for bad in (-.001, float("nan"), float("inf"), -float("inf")):
            cost = self.cost.clone()
            cost[0, 0, 0] = bad
            for batched in (False, True):
                with self.subTest(bad=bad, batched=batched):
                    operation = core.batched_greenkhorn_mmpot if batched else core.greenkhorn_mmpot
                    argument = cost.unsqueeze(0) if batched else cost
                    with mock.patch.object(torch, "exp", side_effect=AssertionError("invalid cost reached kernel")):
                        with self.assertRaisesRegex(ValueError, "transport cost"):
                            operation(argument, .9, .2, 100, 1e-4)

    def test_nonfinite_solver_settings_are_rejected(self):
        for name in ("mass", "eta", "tol"):
            for value in (float("nan"), float("inf"), -float("inf")):
                settings = dict(mass=.9, eta=.2, max_iters=100, tol=1e-4)
                settings[name] = value
                for batched in (False, True):
                    with self.subTest(name=name, value=value, batched=batched):
                        operation = core.batched_greenkhorn_mmpot if batched else core.greenkhorn_mmpot
                        with self.assertRaises(ValueError):
                            operation(self.cost.unsqueeze(0) if batched else self.cost, **settings)

    def test_invalid_iteration_count_and_dtype_are_rejected(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.greenkhorn_mmpot(self.cost, .9, .2, value, 1e-4)
        with self.assertRaises(TypeError):
            core.greenkhorn_mmpot(self.cost.to(torch.int64), .9, .2, 100, 1e-4)
        with self.assertRaises(ValueError):
            core.greenkhorn_mmpot(self.cost, .9, .2, 100, 1e-4, solver_dtype=torch.float16)
        with self.assertRaises(ValueError):
            core.batched_greenkhorn_mmpot(self.cost.unsqueeze(0)[:0], .9, .2, 100, 1e-4)

    def test_feasible_suboptimal_plan_gets_objective_refinement(self):
        initial = self.bad_solved_plan()
        with mock.patch.object(core, "greenkhorn_mmpot", wraps=core.greenkhorn_mmpot) as solve:
            loss, record, _ = core.matching_gap_from_cost(self.cost, solved=initial)
        self.assertEqual(solve.call_count, 1)
        self.assertTrue(solve.call_args.kwargs["stable_kl"])
        self.assertGreaterEqual(float(loss), 0.)
        self.assertEqual(record["objective_refinement"], 1)
        self.assertGreater(record["refinement_iterations"], 0)
        self.assertLessEqual(record["mass_error"], 1e-4)
        self.assertLessEqual(record["cap_violation"], 1e-4)

    def test_genuinely_negative_gap_still_raises_after_one_bounded_retry(self):
        initial = self.bad_solved_plan()
        with mock.patch.object(core, "greenkhorn_mmpot", return_value=copy.deepcopy(initial)) as solve:
            with self.assertRaisesRegex(RuntimeError, "materially negative estimated OT gap"):
                core.matching_gap_from_cost(self.cost, solved=initial)
        self.assertEqual(solve.call_count, 1)

    def test_already_refined_negative_gap_is_not_retried_without_bound(self):
        initial = self.bad_solved_plan()
        initial[2]["refinement_iterations"] = 2000
        with mock.patch.object(core, "greenkhorn_mmpot") as solve:
            with self.assertRaisesRegex(RuntimeError, "materially negative estimated OT gap"):
                core.matching_gap_from_cost(self.cost, solved=initial)
        solve.assert_not_called()

    def test_objective_refinement_preserves_envelope_gradient(self):
        cost = self.cost.clone().requires_grad_()
        loss, _, delta = core.matching_gap_from_cost(cost, solved=self.bad_solved_plan())
        gradient = torch.autograd.grad(loss, cost)[0]
        torch.testing.assert_close(gradient, delta, atol=1e-7, rtol=1e-7)

    def test_nan_residual_does_not_pass_feasibility_comparison(self):
        with mock.patch.object(core, "greenkhorn_mmpot", wraps=core.greenkhorn_mmpot) as solve:
            loss, record, _ = core.matching_gap_from_cost(self.cost, solved=self.bad_solved_plan(mass_error=float("nan")))
        self.assertEqual(solve.call_count, 1)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(math.isfinite(record["mass_error"]))

    def test_unchecked_reference_mode_keeps_negative_gap_visible(self):
        with mock.patch.object(core, "greenkhorn_mmpot") as solve:
            loss, record, _ = core.matching_gap_from_cost(self.cost, solved=self.bad_solved_plan(), require_feasible=False)
        self.assertAlmostEqual(float(loss), -.9, places=6)
        self.assertTrue(record["negative_gap"])
        solve.assert_not_called()

    def test_nonfinite_gap_is_never_treated_as_nonnegative(self):
        plan, slacks, record = self.bad_solved_plan()
        plan[0, 1, 1] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "nonfinite estimated OT gap"):
            core.matching_gap_from_cost(self.cost, solved=(plan, slacks, record))

    def test_valid_default_scalar_and_batched_problems_still_agree(self):
        costs = torch.stack((self.cost, torch.full_like(self.cost, .25)))
        plans, _, records = core.batched_greenkhorn_mmpot(costs, .9, .2, 2000, 1e-10, stable_kl=True)
        for index, cost in enumerate(costs):
            plan, _, record = core.greenkhorn_mmpot(cost, .9, .2, 2000, 1e-10, stable_kl=True)
            torch.testing.assert_close(plans[index], plan, atol=2e-6, rtol=2e-5)
            self.assertAlmostEqual(records[index]["mass"], record["mass"], places=5)


if __name__ == "__main__":
    unittest.main()
