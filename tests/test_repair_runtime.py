"""Source-only authored checks for exact branching/resume; run explicitly when desired."""
import copy
import random
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

from repair_runtime import epoch_batches, named_parameters, pack_adam, restore_adam, restore_rng, rng_state


@unittest.skipIf(torch is None, "PyTorch and NumPy are required")
class RuntimeTests(unittest.TestCase):
    def test_adam_restores_by_name_after_parameter_reordering(self):
        source = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=4e-5, eps=6.25e-10, weight_decay=1e-4)
        batch = torch.tensor([[0.2, 0.7, -0.1], [0.4, 0.3, 0.5]])
        source(batch).square().mean().backward()
        source_optimizer.step()
        saved = pack_adam(source_optimizer, named_parameters(source))
        resumed = copy.deepcopy(source)
        # The model names are unchanged while optimizer iteration is reversed.
        resumed_optimizer = torch.optim.Adam(list(reversed(list(resumed.parameters()))), lr=4e-5, eps=6.25e-10, weight_decay=1e-4)
        restore_adam(resumed_optimizer, named_parameters(resumed), saved)
        for model, optimizer in ((source, source_optimizer), (resumed, resumed_optimizer)):
            optimizer.zero_grad(set_to_none=True)
            model(batch).square().mean().backward()
            optimizer.step()
        for name, parameter in source.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(resumed.named_parameters())[name]))

    def test_branch_new_repair_moments_zero_and_sae_moments_inherited(self):
        sae = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(sae.parameters(), lr=4e-5)
        sae(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        payload = pack_adam(optimizer, named_parameters(sae))
        repair = torch.nn.Linear(3, 2, bias=False)
        branch_optimizer = torch.optim.Adam(list(sae.parameters()) + list(repair.parameters()), lr=4e-5)
        all_named = named_parameters(sae, repair)
        restore_adam(branch_optimizer, all_named, payload, allow_new=True)
        for name, parameter in all_named.items():
            state = branch_optimizer.state[parameter]
            if name.startswith("repair."):
                self.assertEqual(state["step"].item(), 0)
                self.assertEqual(torch.count_nonzero(state["exp_avg"]).item(), 0)
                self.assertEqual(torch.count_nonzero(state["exp_avg_sq"]).item(), 0)
            else:
                self.assertTrue(torch.equal(state["exp_avg"], payload["state"][name]["exp_avg"]))
        bad_optimizer = torch.optim.Adam(list(sae.parameters()) + list(repair.parameters()), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            restore_adam(bad_optimizer, all_named, payload, allow_new=True)

    def test_mask_draws_do_not_change_data_order_and_resume_keeps_tail(self):
        ids = np.arange(17)
        first = list(epoch_batches(ids, seed=1227, epoch=2, batch_size=6))
        torch.rand(100)
        np.random.random(100)
        random.random()
        second = list(epoch_batches(ids, seed=1227, epoch=2, batch_size=6, offset=1))
        self.assertEqual([len(rows) for _, rows in first], [6, 6, 5])
        for (ordinal, rows), (other_ordinal, other_rows) in zip(first[1:], second):
            self.assertEqual(ordinal, other_ordinal)
            np.testing.assert_array_equal(rows, other_rows)
        self.assertFalse(np.array_equal(first[0][1], next(epoch_batches(ids, seed=1227, epoch=3, batch_size=6))[1]))

    def test_rng_roundtrip_includes_independent_mask_stream(self):
        mask = torch.Generator().manual_seed(22)
        diagnostic = torch.Generator().manual_seed(49)
        state = rng_state(mask, extra_generators={"diagnostic": diagnostic})
        first = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=mask), torch.rand(3, generator=diagnostic))
        restore_rng(state, mask, extra_generators={"diagnostic": diagnostic})
        second = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=mask), torch.rand(3, generator=diagnostic))
        self.assertEqual(first[:2], second[:2])
        for left, right in zip(first[2:], second[2:]):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
