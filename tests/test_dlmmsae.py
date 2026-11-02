"""CPU contracts for direct dictionary-context distillation and real K annealing.

The reference embeddings are fixed training data; no ImageNet, CUDA, network,
filesystem fixtures, or moving teacher is involved in these checks.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import numpy as np
from torch.nn import functional as F

import mmrsae_data as data
import mmrsae_imagenet as runner
from dlmmsae import ContextReferenceBank, context_distillation_loss
from mmrsae import METHODS, MMRSAE, MMRSAEConfig, LandmarkBank
from mmrsae_dictionary import LandmarkDictionary
from repair_runtime import named_parameters, pack_adam, restore_adam


def manual_context_kl(dictionary, clean, student, bank, row_ids, teacher_t=.4, student_t=.7):
    """Independent row-wise definition: discard each self key before softmax."""
    context = dictionary.synthesize(student, row_ids=row_ids, exclude_self=True)
    reference = F.normalize(bank.features.detach(), dim=1)
    teacher_logits = F.normalize(clean.detach(), dim=1) @ reference.T / teacher_t
    student_logits = F.normalize(context, dim=1) @ reference.T / student_t
    terms = []
    for index, row_id in enumerate(row_ids):
        keep = bank.ids.to(clean.device).ne(row_id)
        teacher = teacher_logits[index, keep].softmax(0)
        student_log = student_logits[index, keep].log_softmax(0)
        terms.append((teacher * (teacher.log() - student_log)).sum())
    return torch.stack(terms).mean()


class ContextDistillationTests(unittest.TestCase):
    def fixture(self):
        dictionary = LandmarkDictionary(3, rank=2, temperature=.5, blend=.1, seed=17)
        anchors = torch.tensor([[1., .1, .2], [.2, 1., .1], [.1, .2, 1.]])
        dictionary.initialize(anchors, anchor_ids=[100, 101, 102])
        with torch.no_grad():
            dictionary.U.fill_(.2)
        features = torch.tensor([[1., .6, .1], [.1, 1., .6], [.6, .1, 1.], [.3, .7, .9]],
                                requires_grad=True)
        bank = ContextReferenceBank([0, 1, 2, 3], features, anchor_ids=[100, 101, 102])
        x = torch.tensor([[.9, .4, .2], [.3, .5, 1.]], requires_grad=True)
        return dictionary, bank, x, features

    def loss(self, dictionary, x, bank, **kwargs):
        settings = dict(row_ids=[0, 8], noise_std=0., teacher_temperature=.4,
                        student_temperature=.7)
        settings.update(kwargs)
        return context_distillation_loss(dictionary, x, bank, **settings)

    def test_matches_teacher_to_raw_context_kl_with_identical_self_mask(self):
        dictionary, bank, x, _ = self.fixture()
        actual, _ = self.loss(dictionary, x, bank)
        expected = manual_context_kl(dictionary, x, x, bank, [0, 8])
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual))
        self.assertGreater(float(actual), 0.)
        unmasked = manual_context_kl(dictionary, x, x, bank, [8, 9])
        self.assertFalse(torch.isclose(actual, unmasked))

    def test_gradient_reaches_adapter_without_blend_or_sparse_gate_and_teacher_is_frozen(self):
        dictionary, bank, x, source = self.fixture()
        # A saturated bypass cannot disable the independently supervised context.
        with torch.no_grad():
            dictionary.blend_logit.fill_(-100.)
        actual, _ = self.loss(dictionary, x, bank)
        actual_v, actual_x, blend = torch.autograd.grad(
            actual, (dictionary.V, x, dictionary.blend_logit), allow_unused=True)
        self.assertGreater(float(actual_v.norm()), 0.)
        self.assertIsNone(blend)
        expected = manual_context_kl(dictionary, x, x, bank, [0, 8])
        expected_v = torch.autograd.grad(expected, dictionary.V)[0]
        torch.testing.assert_close(actual_v, expected_v)
        # Backbone embeddings are deliberately fixed on both sides; the query
        # adapter, rather than the upstream feature extractor, is optimized.
        self.assertIsNone(actual_x)
        self.assertFalse(bank.features.requires_grad)
        self.assertIsNone(source.grad)
        self.assertFalse(dictionary.features.requires_grad)

    def test_dictionary_auxiliary_lookup_excludes_own_anchor_by_id(self):
        dictionary, _, _, _ = self.fixture()
        x = dictionary.features[:1]
        logits = F.normalize(x, dim=1) @ dictionary.normalized_keys.T / dictionary.temperature
        expected = logits[:, 1:].softmax(1) @ dictionary.features[1:]
        actual = dictionary.synthesize(x, row_ids=[100], exclude_self=True)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, dictionary.synthesize(x)))

    def test_perturbation_uses_dedicated_rng_and_clean_teacher(self):
        dictionary, bank, x, _ = self.fixture()
        original = x.detach().clone()
        global_state = torch.random.get_rng_state().clone()
        generator = torch.Generator().manual_seed(771)
        identical = torch.Generator().manual_seed(771)
        with mock.patch.object(dictionary, "synthesize", wraps=dictionary.synthesize) as lookup:
            actual, _ = self.loss(dictionary, x, bank, noise_std=.25, generator=generator)
        corrupted = lookup.call_args.args[0]
        self.assertFalse(torch.equal(corrupted, x))
        torch.testing.assert_close(x.detach(), original)
        expected = manual_context_kl(dictionary, x, corrupted, bank, [0, 8])
        torch.testing.assert_close(actual, expected)
        repeated, _ = self.loss(dictionary, x, bank, noise_std=.25, generator=identical)
        torch.testing.assert_close(actual, repeated, rtol=0., atol=0.)
        self.assertTrue(torch.equal(generator.get_state(), identical.get_state()))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            self.loss(dictionary, x, bank, noise_std=.25)

    def test_reference_bank_rejects_overlap_duplicate_or_invalid_features(self):
        _, bank, _, _ = self.fixture()
        for ids, features, anchors in (
            ([0, 1, 1, 3], bank.features, [100]),
            ([0, 1, 2, 3], bank.features, [2]),
            ([0, 1, 2], bank.features, [100]),
            ([0, 1], torch.zeros(2, 3), [100]),
            ([0, 1], torch.full((2, 3), float("nan")), [100]),
        ):
            with self.assertRaises(ValueError):
                ContextReferenceBank(ids, features, anchor_ids=anchors)

    def test_reference_bank_checkpoint_cannot_replace_training_reference_identity(self):
        _, bank, _, _ = self.fixture()
        saved = copy.deepcopy(bank.state_dict())
        restored = ContextReferenceBank(bank.ids, bank.features, anchor_ids=[100, 101, 102])
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.features, bank.features, rtol=0., atol=0.)
        torch.testing.assert_close(restored.ids, bank.ids, rtol=0., atol=0.)
        incompatible = ContextReferenceBank([10, 11, 12, 13], bank.features)
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(saved)

    def test_degenerate_profile_support_is_rejected(self):
        dictionary, bank, x, _ = self.fixture()
        tiny = ContextReferenceBank([0, 1], bank.features[:2])
        with self.assertRaises(ValueError):
            self.loss(dictionary, x, tiny, row_ids=[0, 8])
        # Two references remain valid when neither is the current query.
        value, _ = self.loss(dictionary, x, tiny, row_ids=[8, 9])
        self.assertTrue(torch.isfinite(value))


class DLMMSAEMethodTests(unittest.TestCase):
    def fixture(self, variant="dlmmsae", **overrides):
        settings = dict(input_dim=3, hidden_dim=16, k=2, rank=2, variant=variant,
                        relation_weight=0., ot_weight=0., mask=False, k_aux=0,
                        context_noise_std=0., anneal_start_k=8)
        settings.update(overrides)
        model = MMRSAE(MMRSAEConfig(**settings))
        anchors = torch.tensor([[1., .1, .2], [.2, 1., .1], [.1, .2, 1.]])
        model.initialize_dictionary(anchors, anchor_ids=[100, 101, 102])
        bank = ContextReferenceBank([10, 11, 12, 13],
            torch.tensor([[1., .6, .1], [.1, 1., .6], [.6, .1, 1.], [.3, .7, .9]]),
            anchor_ids=[100, 101, 102])
        x = torch.tensor([[.9, .4, .2], [.3, .5, 1.], [.6, .9, .1]])
        return model, bank, x

    def test_context_sparse_relation_and_angular_ot_train_together(self):
        model, references, x = self.fixture(relation_weight=1., ot_weight=2., microbatch=3)
        sparse_bank = LandmarkBank(model.sae.dictionary.anchor_ids,
                                  model.sae.dictionary.features, allow_empty_codes=True)
        sparse_bank.refresh(model.sae, None, model.config.k)
        loss, stats, views = model.loss(x, row_ids=[0, 1, 2], bank=sparse_bank,
            context_bank=references, diagnostics=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats["gradient_context_adapter_norm"]), 0.)
        self.assertIn("old_ot", views["component_losses"])
        self.assertIn("new_ot", views["component_losses"])
        self.assertIn("relation", views["component_losses"])
        self.assertIn("context", views["component_losses"])
        self.assertNotIn("costnbr_mean", stats)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_new_variants_defaults_and_invalid_context_settings(self):
        self.assertIn("dlmmsae", METHODS)
        self.assertIn("dlmmsae-annealed", METHODS)
        for method in METHODS:
            config = MMRSAEConfig(input_dim=3, hidden_dim=16, k=2, variant=method)
            self.assertEqual(config.context_weight, 1. if method.startswith("dlmmsae") else 0.)
        for settings in ({"context_weight": -1.}, {"context_weight": float("nan")},
                         {"context_noise_std": -1.}, {"context_noise_std": float("inf")},
                         {"context_teacher_temperature": 0.}, {"context_student_temperature": 0.}):
            with self.assertRaises(ValueError):
                MMRSAEConfig(input_dim=3, hidden_dim=16, k=2, variant="dlmmsae", **settings)
        with self.assertRaises(ValueError):
            MMRSAEConfig(input_dim=3, hidden_dim=16, k=2, variant="mmrsae", context_weight=1.)

    def test_direct_context_term_adds_exactly_to_shared_objective(self):
        model, bank, x = self.fixture(context_weight=.7)
        baseline, _, _ = self.fixture(variant="mmrsae")
        self.assertTrue(model.needs_context)
        self.assertFalse(baseline.needs_context)
        with self.assertRaises(ValueError):
            model.loss(x, row_ids=[0, 1, 2], update_activity=False)
        value, stats, views = model.loss(x, context_bank=bank, row_ids=[0, 1, 2], update_activity=False)
        original, _, _ = baseline.loss(x, update_activity=False)
        context, _ = context_distillation_loss(model.sae.dictionary, x, bank,
            row_ids=[0, 1, 2], noise_std=0.,
            teacher_temperature=model.config.context_teacher_temperature,
            student_temperature=model.config.context_student_temperature)
        torch.testing.assert_close(value, original + .7 * context)
        torch.testing.assert_close(stats["loss_context_unweighted"], context)
        torch.testing.assert_close(stats["weighted_context"], .7 * context)
        torch.testing.assert_close(stats["loss_total"], value)
        torch.testing.assert_close(views["component_losses"]["context"], .7 * context)

    def test_zero_context_weight_is_an_exact_ablation_and_inference_needs_no_reference_bank(self):
        model, _, x = self.fixture(context_weight=0.)
        baseline, _, _ = self.fixture(variant="mmrsae")
        self.assertFalse(model.needs_context)
        value, _, _ = model.loss(x, update_activity=False)
        original, _, _ = baseline.loss(x, update_activity=False)
        torch.testing.assert_close(value, original, rtol=0., atol=0.)
        for k in (1, 2, 4, 8, 16):
            torch.testing.assert_close(model.encode(x, k), baseline.encode(x, k), rtol=0., atol=0.)

    def test_annealed_schedule_changes_real_codes_and_matches_fixed_endpoint(self):
        fixed, bank, x = self.fixture()
        annealed, _, _ = self.fixture(variant="dlmmsae-annealed")
        schedule = [annealed.config.training_k(step, 10) for step in range(10)]
        self.assertEqual(schedule[0], 8)
        self.assertEqual(schedule[6:], [2] * 4)
        self.assertEqual(schedule, sorted(schedule, reverse=True))
        for step in (0, 3, 9):
            _, stats, views = annealed.loss(x, context_bank=bank, row_ids=[0, 1, 2],
                step=step, total_steps=10, update_activity=False)
            k = schedule[step]
            self.assertEqual(stats["training_k"], k)
            self.assertEqual(stats["auxiliary_budget"], min(4 * k, 16))
            torch.testing.assert_close(views["final_k"], annealed.encode(x, k))
            torch.testing.assert_close(views["final_2k"], annealed.encode(x, min(2 * k, 16)))
        fixed_loss, _, _ = fixed.loss(x, context_bank=bank, row_ids=[0, 1, 2],
            step=9, total_steps=10, update_activity=False)
        end_loss, _, _ = annealed.loss(x, context_bank=bank, row_ids=[0, 1, 2],
            step=9, total_steps=10, update_activity=False)
        torch.testing.assert_close(fixed_loss, end_loss, rtol=0., atol=0.)
        fixed_loss.backward()
        end_loss.backward()
        for left, right in zip(fixed.parameters(), annealed.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0., atol=0.)

    def test_resume_preserves_noise_mask_optimizer_and_annealing(self):
        model, bank, x = self.fixture(variant="dlmmsae-annealed", mask=True, context_noise_std=.08)
        masks = torch.Generator().manual_seed(661)
        context_rng = torch.Generator().manual_seed(662)

        def optimizer_for(current):
            return torch.optim.Adam(list(named_parameters(current.sae, current.repair).values()), lr=1e-3)

        optimizer = optimizer_for(model)

        def train(current, references, adam, mask_rng, noise_rng, first, stop):
            trace = []
            for step in range(first, stop):
                adam.zero_grad(set_to_none=True)
                loss, stats, views = current.loss(x, row_ids=[0, 1, 2], context_bank=references,
                    mask_generator=mask_rng, context_generator=noise_rng, step=step, total_steps=4)
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all()
                    for parameter in current.parameters() if parameter.grad is not None))
                adam.step()
                trace.append((stats["training_k"], loss.detach().clone(), views["final_k"].detach().clone()))
            return trace

        train(model, bank, optimizer, masks, context_rng, 0, 1)
        saved = copy.deepcopy({"model": model.state_dict(), "bank": bank.state_dict(),
            "optimizer": pack_adam(optimizer, named_parameters(model.sae, model.repair)),
            "masks": masks.get_state(), "noise": context_rng.get_state()})
        restored = MMRSAE(model.config)
        restored.load_state_dict(saved["model"])
        restored_bank = ContextReferenceBank(bank.ids, bank.features, anchor_ids=[100, 101, 102])
        restored_bank.load_state_dict(saved["bank"])
        restored_optimizer = optimizer_for(restored)
        restore_adam(restored_optimizer, named_parameters(restored.sae, restored.repair), saved["optimizer"])
        restored_masks, restored_noise = torch.Generator(), torch.Generator()
        restored_masks.set_state(saved["masks"])
        restored_noise.set_state(saved["noise"])
        left = train(model, bank, optimizer, masks, context_rng, 1, 4)
        right = train(restored, restored_bank, restored_optimizer, restored_masks, restored_noise, 1, 4)
        self.assertGreater(left[0][0], left[-1][0])
        self.assertEqual(left[-1][0], 2)
        for actual, expected in zip(left, right):
            self.assertEqual(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1], rtol=0., atol=0.)
            torch.testing.assert_close(actual[2], expected[2], rtol=0., atol=0.)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0., atol=0.)
        self.assertTrue(torch.equal(masks.get_state(), restored_masks.get_state()))
        self.assertTrue(torch.equal(context_rng.get_state(), restored_noise.get_state()))


class TrainingReferenceAndTimingTests(unittest.TestCase):
    def test_all_training_rows_keeps_synthesis_anchors_and_queries_out_of_fit(self):
        labels = np.repeat(np.arange(3), 20)
        args = SimpleNamespace(evaluation="dev", train_size=0, dev_size=6,
                               anchor_count=6, data_seed=42)
        pools = data._select_pools(labels, args)
        self.assertEqual(len(pools["fit"]), len(labels) - 12)
        self.assertFalse(np.intersect1d(pools["fit"], pools["anchors"]).size)
        self.assertFalse(np.intersect1d(pools["query"], pools["fit"]).size)
        self.assertFalse(np.intersect1d(pools["query"], pools["anchors"]).size)

    def test_reference_selection_is_repeatable_train_only_and_validates_split_namespace(self):
        args = SimpleNamespace(data_seed=42, context_reference_count=4, evaluation="dev")
        bundle = SimpleNamespace(anchor_ids=np.array([100, 101, 102]),
            fit_ids=np.arange(20), gallery_ids=np.arange(20), query_ids=np.array([30, 31]))
        first = runner.context_reference_ids(args, bundle)
        second = runner.context_reference_ids(args, bundle)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(np.isin(first, bundle.fit_ids).all())
        bundle.query_ids = np.array([100])
        with self.assertRaises(ValueError):
            runner.context_reference_ids(args, bundle)
        # Official validation has its own row-ID namespace, so matching integer
        # indices do not imply a shared training feature.
        args.evaluation = "official"
        np.testing.assert_array_equal(first, runner.context_reference_ids(args, bundle))
        bundle.gallery_ids = np.array([100, 0])
        with self.assertRaises(ValueError):
            runner.context_reference_ids(args, bundle)

    def test_training_timer_accumulates_only_measured_scopes_and_can_resume(self):
        ticks = iter([10., 13., 20., 25.])
        synchronization = []
        timer = runner.TrainingTimer("cuda:0", seconds=7.,
            synchronize=lambda: synchronization.append("cuda:0"), clock=lambda: next(ticks))
        with timer.measure():
            pass
        self.assertEqual(timer.seconds, 10.)
        with timer.measure():
            pass
        self.assertEqual(timer.seconds, 15.)
        self.assertEqual(synchronization, ["cuda:0"] * 4)


if __name__ == "__main__":
    unittest.main()
