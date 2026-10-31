"""Small CPU checks for the neighborhood affinity cost and its gradients."""
from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import mpsae_repair as core
    from mmrsae import MMRSAEConfig, VARIANTS


@unittest.skipIf(torch is None, "PyTorch is required for numerical cost checks")
class NeighborhoodCostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9321)

    @staticmethod
    def reference(log_p, log_q, log_teacher):
        """Direct small-tensor probability formula, independent of the GEMM path."""
        p, q, teacher = (value.softmax(-1) for value in (log_p, log_q, log_teacher.detach()))
        products = p[:, None, None, :] * q[None, :, None, :] * teacher[None, None, :, :]
        return 1 - products.pow(1 / 3).sum(-1)

    def test_matches_direct_float64_probability_formula(self):
        values = [torch.randn(3, 7, dtype=torch.float64) for _ in range(3)]
        actual = core.neighborhood_cost(*values)
        torch.testing.assert_close(actual, self.reference(*values), atol=2e-14, rtol=2e-13)
        self.assertEqual(actual.dtype, torch.float64)

    def test_identical_wide_profiles_are_nonnegative(self):
        uniform = torch.full((3, 1024), -math.log(1024), dtype=torch.float32)
        peaked = torch.full((3, 1024), -10000., dtype=torch.float32)
        peaked[:, 0] = 0.
        random = torch.randn(3, 1024).log_softmax(-1)
        for profiles in (uniform, peaked, random):
            with self.subTest(distribution=profiles[0, 0].item()):
                cost = core.neighborhood_cost(profiles, profiles, profiles)
                self.assertTrue(bool(torch.isfinite(cost).all()))
                self.assertGreaterEqual(float(cost.min()), 0.)
                self.assertLessEqual(float(cost.max()), 1.)
                ids = torch.arange(len(profiles))
                torch.testing.assert_close(cost[ids, ids, ids], torch.zeros(3), atol=2e-6, rtol=0.)

    def test_row_log_offsets_do_not_change_the_distributions(self):
        profiles = [torch.randn(2, 9, dtype=torch.float64) for _ in range(3)]
        offsets = [torch.tensor([[10000.], [-10000.]], dtype=torch.float64),
                   torch.tensor([[-23.], [51.]], dtype=torch.float64),
                   torch.tensor([[100.], [200.]], dtype=torch.float64)]
        shifted = [value + shift for value, shift in zip(profiles, offsets)]
        torch.testing.assert_close(core.neighborhood_cost(*profiles), core.neighborhood_cost(*shifted),
                                   atol=2e-12, rtol=2e-12)

    def test_huge_equal_logits_still_define_uniform_profiles(self):
        # Subtracting logsumexp directly from 1e20 loses the log(1024)
        # normalizer in float32. Removing the common maximum first avoids it.
        huge = torch.full((2, 1024), 1e20, requires_grad=True)
        ordinary = torch.zeros(2, 1024)
        cost = core.neighborhood_cost(huge, huge, huge)
        torch.testing.assert_close(cost, core.neighborhood_cost(ordinary, ordinary, ordinary),
                                   atol=0., rtol=0.)
        self.assertLess(float(cost.max()), 2e-6)
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(cost.sum(), huge)[0]).all()))

    def test_exact_disjoint_support_has_unit_cost(self):
        p = torch.tensor([[0., -torch.inf, -torch.inf]])
        q = torch.tensor([[-torch.inf, 0., -torch.inf]])
        teacher = torch.tensor([[-torch.inf, -torch.inf, 0.]])
        torch.testing.assert_close(core.neighborhood_cost(p, q, teacher), torch.ones(1, 1, 1))

    def test_extreme_and_masked_profiles_keep_finite_gradients(self):
        p = torch.tensor([[10000., -10000., -torch.inf], [-1., -2., -torch.inf]], requires_grad=True)
        q = torch.tensor([[-10000., 10000., -torch.inf], [-3., -1., -torch.inf]], requires_grad=True)
        teacher = torch.tensor([[10000., -10000., -torch.inf], [-2., -1., -torch.inf]])
        cost = core.neighborhood_cost(p, q, teacher)
        gradients = torch.autograd.grad(cost.sum(), (p, q))
        self.assertTrue(bool(torch.isfinite(cost).all()))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertGreaterEqual(float(cost.min()), 0.)
        self.assertLessEqual(float(cost.max()), 1.)

    def test_teacher_detached_and_student_gradients_are_nonzero(self):
        p, q, teacher = [torch.randn(3, 7, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        gradients = torch.autograd.grad(core.neighborhood_cost(p, q, teacher).sum(),
                                        (p, q, teacher), allow_unused=True)
        self.assertIsNone(gradients[2])
        for value in gradients[:2]:
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertGreater(float(value.abs().max()), 1e-8)

    def test_autograd_matches_direct_formula_and_finite_differences(self):
        p, q = [torch.randn(2, 5, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        teacher = torch.randn(2, 5, dtype=torch.float64)
        actual = torch.autograd.grad(core.neighborhood_cost(p, q, teacher).sum(), (p, q))
        expected = torch.autograd.grad(self.reference(p, q, teacher).sum(), (p, q))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)
        self.assertTrue(torch.autograd.gradcheck(
            lambda first, second: core.neighborhood_cost(first, second, teacher),
            (p, q), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_nonfinite_or_zero_mass_rows_are_rejected(self):
        valid = torch.zeros(2, 4)
        for bad_value in (torch.nan, torch.inf, -torch.inf):
            bad = valid.clone()
            if bad_value == -torch.inf:
                bad[0] = bad_value
            else:
                bad[0, 0] = bad_value
            for position in range(3):
                values = [valid, valid, valid]
                values[position] = bad
                with self.subTest(value=bad_value, position=position), self.assertRaises(ValueError):
                    core.neighborhood_cost(*values)

    def test_invalid_shapes_and_integer_dtypes_are_rejected(self):
        valid = torch.zeros(2, 4)
        for invalid in (torch.zeros(4), torch.zeros(2, 5), torch.zeros(0, 4), torch.zeros(2, 0),
                        torch.zeros(2, 4, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.bool)):
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype), self.assertRaises((ValueError, TypeError)):
                core.neighborhood_cost(invalid, valid, valid)

    def test_different_devices_are_rejected_before_arithmetic(self):
        valid = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            core.neighborhood_cost(valid, valid, torch.empty(2, 4, device="meta"))

    def test_low_precision_inputs_promote_and_float64_is_preserved(self):
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(2, 8).to(dtype) for _ in range(3)]
            actual = core.neighborhood_cost(*inputs)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(actual).all()))
        inputs = [torch.randn(2, 8), torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 8)]
        self.assertEqual(core.neighborhood_cost(*inputs).dtype, torch.float64)

    def test_autocast_does_not_reduce_cost_precision(self):
        profiles = [torch.randn(3, 1024, requires_grad=True) for _ in range(3)]
        expected = core.neighborhood_cost(*profiles)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = core.neighborhood_cost(*profiles)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        gradients = torch.autograd.grad(actual.sum(), profiles[:2])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_rectangular_sample_counts_share_one_landmark_axis(self):
        profiles = [torch.randn(rows, 5) for rows in (2, 3, 4)]
        actual = core.neighborhood_cost(*profiles)
        self.assertEqual(actual.shape, (2, 3, 4))
        torch.testing.assert_close(actual, self.reference(*profiles), atol=3e-7, rtol=3e-6)

    def test_mixed_presets_default_to_point_two_and_controls_remain_angular(self):
        for variant in VARIANTS:
            config = MMRSAEConfig(input_dim=6, hidden_dim=16, k=2, variant=variant)
            self.assertEqual(config.mixture_weight, 0. if variant == "mmrsae-rel" else .2)
        for baseline in ("mpsae", "csr"):
            config = MMRSAEConfig(input_dim=6, hidden_dim=16, k=2, variant=baseline)
            self.assertEqual(config.mixture_weight, 0.)


if __name__ == "__main__":
    unittest.main()
