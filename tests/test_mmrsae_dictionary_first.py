"""CPU behavioral checks for dictionary-before-Top-K MMRSAE and K annealing.

No ImageNet, CUDA, network access, or filesystem fixtures are used here.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

import mmrsae_imagenet as runner
import mpsae_repair as core
from mmrsae import MMRSAE, MMRSAEConfig
from repair_runtime import named_parameters, pack_adam, restore_adam


class DictionaryFirstTests(unittest.TestCase):
    def toy(self, **overrides):
        settings = dict(input_dim=2, hidden_dim=3, k=1, rank=2,
                        dictionary_temperature=1., dictionary_blend=.5,
                        relation_weight=0., ot_weight=0., mask=False, k_aux=0)
        settings.update(overrides)
        model = MMRSAE(MMRSAEConfig(**settings))
        model.initialize_dictionary(torch.eye(2), anchor_ids=[10, 20])
        with torch.no_grad():
            model.sae.decoder.copy_(torch.tensor([[1., 0.], [0., 1.], [.9, .9]]))
            model.sae.encoder_bias.zero_()
            model.sae.pre_bias.zero_()
        return model

    def test_landmark_embedding_is_constructed_before_the_only_sparse_gate(self):
        model = self.toy()
        x = torch.tensor([[1., 0.]])
        events = []
        handles = [model.sae.dictionary.register_forward_pre_hook(lambda *unused: events.append("dictionary"))]
        gate = model.sae.keep_topk

        def final_gate(values, k):
            events.append("topk")
            return gate(values, k)

        try:
            with mock.patch.object(model.sae, "keep_topk", side_effect=final_gate):
                z = model.encode(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(events, ["dictionary", "topk"])
        self.assertIsNone(model.repair)
        # Softmax over the two unit anchors at temperature 1 gives sigmoid(1).
        # Blending this context 50/50 with x changes the winning SAE atom:
        # plain x selects atom 0, whereas the synthesized embedding selects 2.
        torch.testing.assert_close(model.sae.dictionary(x), torch.tensor([[.8655293, .1344707]]))
        torch.testing.assert_close(z, torch.tensor([[0., 0., .9]]))
        torch.testing.assert_close(core.nonnegative_topk(x @ model.sae.decoder.T, 1),
                                   torch.tensor([[1., 0., 0.]]))

    def test_clean_views_and_reconstruction_use_one_frontend_and_original_target(self):
        model = self.toy()
        x = torch.tensor([[1., 0.], [.7, 1.2]])
        before = model.sae.inactive_steps.clone()
        _, stats, views = model.loss(x, update_activity=False)
        previous = torch.zeros_like(views["final_k"], dtype=torch.bool)
        for k in (1, 2, 3):
            encoded = model.encode(x, k)
            support = encoded.gt(0)
            self.assertTrue((~previous | support).all())
            self.assertTrue(support.sum(1).le(k).all())
            previous = support
        for name, k in (("final_k", 1), ("final_2k", 2),
                        ("raw_4k", 3), ("raw_auxiliary", 3)):
            torch.testing.assert_close(views[name], model.encode(x, k))
        decoded = model.sae.decode(views["final_k"])
        torch.testing.assert_close(stats["mse_final_K"], F.mse_loss(decoded, x))
        self.assertFalse(torch.isclose(stats["mse_final_K"], F.mse_loss(decoded, model.sae.dictionary(x))))
        torch.testing.assert_close(before, model.sae.inactive_steps)

    def test_masking_precedes_projection_and_keeps_clean_auxiliary_views(self):
        model = self.toy(mask=True, clean_probability=0., erase_probability=1.)
        x = torch.tensor([[1., .8], [.7, 1.2]])
        _, stats, views = model.loss(x, mask_generator=torch.Generator().manual_seed(11),
                                     update_activity=False)
        torch.testing.assert_close(views["final_k"], torch.zeros_like(views["final_k"]))
        torch.testing.assert_close(views["final_2k"], torch.zeros_like(views["final_2k"]))
        self.assertEqual(float(stats["mask_erase_fraction"]), 1.)
        torch.testing.assert_close(views["raw_4k"], model.encode(x, 3))

    def test_frontend_learns_while_training_anchor_features_stay_frozen(self):
        model = self.toy()
        with torch.no_grad():
            model.sae.dictionary.U.fill_(.2)
        x = torch.tensor([[1., .8], [.2, 1.4]])
        dictionary = model.sae.dictionary
        before = dictionary.features.clone()
        loss, _, _ = model.loss(x, update_activity=False)
        loss.backward()
        self.assertGreater(float(dictionary.V.grad.norm()), 0.)
        self.assertGreater(abs(float(dictionary.blend_logit.grad)), 0.)
        self.assertFalse(dictionary.features.requires_grad)
        self.assertIsNone(dictionary.features.grad)
        torch.testing.assert_close(before, dictionary.features)

    def test_dictionary_initialization_is_required_and_validates_training_anchors(self):
        settings = dict(input_dim=2, hidden_dim=3, k=1, rank=2, relation_weight=0.)
        model = MMRSAE(MMRSAEConfig(**settings))
        with self.assertRaisesRegex(RuntimeError, "[Ii]nitializ|[Dd]ictionary"):
            model.encode(torch.ones(1, 2))
        for features, ids in ((torch.zeros(1, 2), [0]),
                              (torch.tensor([[float("nan"), 1.]]), [0]),
                              (torch.eye(2), [0, 0]), (torch.eye(2), [0])):
            with self.assertRaises(ValueError):
                model.initialize_dictionary(features, anchor_ids=ids)
        model.initialize_dictionary(torch.eye(2), anchor_ids=[10, 20])
        with self.assertRaisesRegex(RuntimeError, "once|fixed|already"):
            model.initialize_dictionary(torch.eye(2), anchor_ids=[10, 20])

    def test_checkpoint_restores_dictionary_and_rejects_old_architecture(self):
        model = self.toy()
        state = copy.deepcopy(model.state_dict())
        restored = MMRSAE(model.config)
        restored.load_state_dict(state)
        self.assertTrue(restored.sae.dictionary.initialized)
        torch.testing.assert_close(restored.sae.dictionary.anchor_ids, torch.tensor([10, 20]))
        x = torch.tensor([[1., .8], [.7, 1.2]])
        for k in (1, 2, 3):
            torch.testing.assert_close(restored.encode(x, k), model.encode(x, k))
        del state["sae.dictionary.architecture_version"]
        with self.assertRaisesRegex(RuntimeError, "architecture|checkpoint"):
            MMRSAE(model.config).load_state_dict(state, strict=False)

    def test_annealed_endpoint_loss_and_gradients_match_fixed_mmrsae(self):
        settings = dict(input_dim=6, hidden_dim=32, k=2, rank=3,
                        anneal_start_k=8, relation_weight=0., mask=False, microbatch=4)
        fixed = MMRSAE(MMRSAEConfig(**settings, variant="mmrsae"))
        annealed = MMRSAE(MMRSAEConfig(**settings, variant="mmrsae-annealed"))
        x = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        for model in (fixed, annealed):
            model.initialize_dictionary(x, anchor_ids=[10, 20, 30, 40])
        fixed_loss, _, _ = fixed.loss(x, step=9, total_steps=10, update_activity=False)
        end_loss, stats, _ = annealed.loss(x, step=9, total_steps=10, update_activity=False)
        self.assertEqual(stats["training_k"], 2)
        self.assertEqual(stats["auxiliary_budget"], 8)
        torch.testing.assert_close(end_loss, fixed_loss)
        fixed_loss.backward()
        end_loss.backward()
        for (left_name, left), (right_name, right) in zip(fixed.named_parameters(), annealed.named_parameters()):
            self.assertEqual(left_name, right_name)
            torch.testing.assert_close(left.grad, right.grad)


class AnnealedLandmarkTests(unittest.TestCase):
    def model_and_bank(self):
        model = MMRSAE(MMRSAEConfig(input_dim=6, hidden_dim=32, k=2, rank=3,
            variant="mmrsae-annealed", anneal_start_k=8,
            mask=False, ot_weight=0., k_aux=0))
        features = torch.linspace(.1, 2.4, 24).reshape(4, 6)
        model.initialize_dictionary(features, anchor_ids=[100, 101, 102, 103])
        bank = core.LandmarkBank([100, 101, 102, 103], features, allow_empty_codes=True)
        return model, bank, features

    def assert_current_bank(self, model, bank, features, k):
        self.assertEqual(bank.k, k)
        for codes, budget in zip(bank.codes, (k, 2 * k)):
            torch.testing.assert_close(codes, model.encode(features, budget))

    def test_real_bank_tracks_schedule_and_resumed_state_without_extra_refreshes(self):
        model, bank, features = self.model_and_bank()
        args = SimpleNamespace(encode_batch_size=2)
        with mock.patch.object(runner, "log_wandb_metrics"):
            first = runner.refresh_training_bank(args, model, bank, 0, 10)
            self.assertEqual(first["training_k"], 8)
            self.assert_current_bank(model, bank, features, 8)
            self.assertIsNone(runner.refresh_training_bank(args, model, bank, 0, 10))
            self.assertEqual(bank.refresh_count, 1)

            resumed = core.LandmarkBank(bank.anchor_ids, features, allow_empty_codes=True)
            resumed.load_state_dict(copy.deepcopy(bank.state_dict()))
            self.assertIsNone(runner.refresh_training_bank(args, model, resumed, 0, 10))
            self.assertEqual(resumed.refresh_count, 1)
            changed = runner.refresh_training_bank(args, model, resumed, 3, 10)
            self.assertEqual(changed["training_k"], 5)
            self.assert_current_bank(model, resumed, features, 5)
            loss, stats, _ = model.loss(features, bank=resumed, row_ids=[0, 1, 2, 3],
                step=3, total_steps=10, update_activity=False)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(stats["training_k"], resumed.k)

            runner.refresh_training_bank(args, model, resumed, 10, 10)
            self.assert_current_bank(model, resumed, features, 2)
            count = resumed.refresh_count
            self.assertIsNone(runner.refresh_training_bank(args, model, resumed, 10, 10))
            runner.refresh_training_bank(args, model, resumed, 10, 10, force=True)
            self.assertEqual(resumed.refresh_count, count + 1)

    def test_stale_bank_budget_is_rejected_instead_of_mixing_training_views(self):
        model, bank, features = self.model_and_bank()
        bank.refresh(model.sae, model.repair, 8, batch_size=2)
        with self.assertRaisesRegex(ValueError, "[Bb]udget|[Bb]ank|[Kk]"):
            model.loss(features, bank=bank, row_ids=[0, 1, 2, 3], step=9,
                       total_steps=10, update_activity=False)

    def test_bank_refresh_validates_schedule_boundaries_and_supports_no_bank(self):
        model, _, _ = self.model_and_bank()
        args = SimpleNamespace(encode_batch_size=2)
        self.assertIsNone(runner.refresh_training_bank(args, model, None, 10, 10))
        for step, total in ((-1, 10), (11, 10), (0, 0)):
            with self.assertRaises(ValueError):
                runner.refresh_training_bank(args, model, None, step, total)

    def test_optimizer_resume_matches_uninterrupted_masked_kl_ot_training_across_k_changes(self):
        config = MMRSAEConfig(input_dim=6, hidden_dim=32, k=2, rank=3,
            variant="mmrsae-annealed", anneal_start_k=8, mask=True,
            relation_weight=1., ot_weight=2., microbatch=4, k_aux=8, dead_steps=2)
        data_generator = torch.Generator().manual_seed(407)
        anchors = torch.rand(8, 6, generator=data_generator) + .2
        batches = [torch.rand(4, 6, generator=data_generator) + .2 for _ in range(8)]
        anchor_ids = torch.arange(100, 108)
        args = SimpleNamespace(encode_batch_size=4)
        model = MMRSAE(config)
        model.initialize_dictionary(anchors, anchor_ids=anchor_ids)
        bank = core.LandmarkBank(anchor_ids, anchors, allow_empty_codes=True)
        masks = torch.Generator().manual_seed(833)

        def optimizer_for(current):
            parameters = named_parameters(current.sae, current.repair)
            return torch.optim.Adam(list(parameters.values()), lr=4e-5,
                                    eps=6.25e-10, weight_decay=1e-4)

        optimizer = optimizer_for(model)

        def train(current, adam, keys, generator, first, stop):
            trace = []
            for step in range(first, stop):
                runner.refresh_training_bank(args, current, keys, step, 8)
                adam.zero_grad(set_to_none=True)
                loss, stats, views = current.loss(batches[step], bank=keys,
                    row_ids=[0, 1, 2, 3], mask_generator=generator,
                    step=step, total_steps=8)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(stats["training_k"], keys.k)
                trace.append((stats["training_k"], views["final_k"].detach().clone()))
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all()
                    for parameter in current.parameters() if parameter.grad is not None))
                adam.step()
                if (step + 1) % 2 == 0:
                    runner.refresh_training_bank(args, current, keys, step + 1, 8, force=True)
            return trace

        with mock.patch.object(runner, "log_wandb_metrics"):
            train(model, optimizer, bank, masks, 0, 3)
            saved = copy.deepcopy({
                "model": model.state_dict(),
                "optimizer": pack_adam(optimizer, named_parameters(model.sae, model.repair)),
                "bank": bank.state_dict(),
                "masks": masks.get_state(),
            })
            restored = MMRSAE(config)
            # Anchor buffers are recovered from the checkpoint, not initialized
            # again from caller data, as in the actual runner's resume path.
            restored.load_state_dict(saved["model"])
            resumed_optimizer = optimizer_for(restored)
            restore_adam(resumed_optimizer, named_parameters(restored.sae, restored.repair),
                         saved["optimizer"])
            resumed_bank = core.LandmarkBank(anchor_ids, anchors, allow_empty_codes=True)
            resumed_bank.load_state_dict(saved["bank"])
            resumed_masks = torch.Generator()
            resumed_masks.set_state(saved["masks"])
            uninterrupted_trace = train(model, optimizer, bank, masks, 3, 8)
            resumed_trace = train(restored, resumed_optimizer, resumed_bank, resumed_masks, 3, 8)

        self.assertGreater(len({k for k, _ in resumed_trace}), 1)
        self.assertEqual(resumed_trace[-1][0], 2)
        for (left_k, left_code), (right_k, right_code) in zip(uninterrupted_trace, resumed_trace):
            self.assertEqual(left_k, right_k)
            torch.testing.assert_close(left_code, right_code, rtol=0., atol=0.)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.isfinite(value).all())
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0., atol=0.)
        for left_keys, right_keys in zip(bank.codes, resumed_bank.codes):
            torch.testing.assert_close(left_keys, right_keys, rtol=0., atol=0.)
        self.assertEqual(bank.refresh_count, resumed_bank.refresh_count)
        self.assertEqual(bank.last_refresh_step, resumed_bank.last_refresh_step)
        self.assertTrue(torch.equal(masks.get_state(), resumed_masks.get_state()))
        left_state = pack_adam(optimizer, named_parameters(model.sae, model.repair))["state"]
        right_state = pack_adam(resumed_optimizer, named_parameters(restored.sae, restored.repair))["state"]
        for name, moments in left_state.items():
            for field, value in moments.items():
                torch.testing.assert_close(value, right_state[name][field], rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
