"""Harmless checks for token budgeting, source isolation and fixed teacher gates."""
import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_auxiliary_scorer as workflow
from traceguard import auxiliary_scorer as core


class Tokenizer:
    def encode(self, text, **kwargs):
        return [len(word) + 3 for word in text.split()]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def prepare_for_model(self, ids, pair_ids=None, **kwargs):
        sequence = [101] + list(ids) + [102] + list(pair_ids) + [102]
        return {"input_ids": sequence, "attention_mask": [1] * len(sequence),
                "token_type_ids": [0] * (len(ids) + 2) + [1] * (len(pair_ids) + 1)}

    def pad(self, rows, **kwargs):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        return {key: torch.tensor([r[key] + [0] * (length - len(r[key])) for r in rows]) for key in rows[0]}


class SentenceModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, features):
        import torch
        values = torch.zeros((len(features["input_ids"]), 384))
        values[:, 0] = features["input_ids"].sum(dim=1)
        values[:, 1] = features["attention_mask"].sum(dim=1)
        return {"sentence_embedding": values}


class Classifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        positive = .1 + .8 * np.asarray(values)[:, 0]
        return np.c_[1 - positive, positive]


class AuxiliaryScorerTests(unittest.TestCase):
    def fixture(self):
        membership = {"train": [], "validation": []}
        labels, features, groups = [], [], []
        for source in range(10):
            for condition in (1, 2):
                row = {"sample_id": f"s{source}-c{condition}", "source_id": f"s{source}", "split": "train",
                       "condition_id": condition, "activation_label": int(condition == 2)}
                membership["train"].append(row)
                groups.append(row["source_id"])
                labels.append({k: row[k] for k in ("sample_id", "source_id", "split")}
                              | {"auxiliary_targets": {h: int(condition == 2) for h in core.HEADS}})
                features.append([int(condition == 2), source])
        folds = [{"fold": i, "fit_indices": [j for j in range(20) if j // 4 != i],
                  "held_indices": list(range(i * 4, i * 4 + 4))} for i in range(5)]
        return membership, labels, np.asarray(features, dtype=float), folds, groups

    def fitter(self, features, targets, settings):
        self.assertEqual(set(targets), {0, 1})
        self.assertTrue(np.array_equal(features[:, 0].astype(int), targets))
        return Classifier()

    def test_pair_budget_reserves_response_and_exact_special_token_overhead(self):
        features, info = core.pair_features(Tokenizer(), "p " * 120, "answer " * 300)
        self.assertEqual(info["request_tokens_kept"], 64)
        self.assertEqual(info["response_tokens_kept"], 189)
        self.assertEqual(info["input_tokens"], 256)
        self.assertEqual(features["token_type_ids"], [0] * 66 + [1] * 190)
        _, short = core.pair_features(Tokenizer(), "one two three", "answer " * 300)
        self.assertEqual(short["response_tokens_kept"], 250)

    def test_pair_encoding_uses_only_text_and_is_batch_consistent(self):
        encoder = core.PairEncoder("unused", model=SentenceModel())
        rows = [{"prompt": "A harmless question.", "response": "A harmless answer.", "activation_label": 1, "source": "Hidden"},
                {"prompt": "Another question.", "response": "A longer harmless answer about colors."}]
        values = encoder.encode(rows)
        changed = copy.deepcopy(rows)
        changed[0]["activation_label"] = 0
        changed[0]["source"] = "Completely different"
        self.assertTrue(np.array_equal(values, encoder.encode(changed)))
        self.assertTrue(np.allclose(values[0], encoder.encode(rows[:1])[0], atol=1e-7))
        self.assertEqual(values.shape, (2, 384))
        self.assertTrue(np.allclose(np.linalg.norm(values, axis=1), 1.))
        self.assertFalse(encoder.model.training)

    def test_nested_plan_protects_outer_and_inner_source_groups(self):
        membership, labels, _, folds, groups = self.fixture()
        nested = core.nested_plan(membership, folds, groups)
        self.assertEqual(sum(len(r["inner_folds"]) for r in nested), 25)
        support = core.check_support_plan(labels, membership, nested)
        self.assertEqual(len(support), 30)
        self.assertTrue(all(r["passes"] for r in support))
        for outer in nested:
            held = {groups[i] for i in outer["held_indices"]}
            for inner in outer["inner_folds"]:
                self.assertFalse({groups[i] for i in inner["fit_indices"]} & held)

    def test_nested_plan_rejects_cross_partition_or_outer_leakage(self):
        membership, _, _, folds, groups = self.fixture()
        membership["validation"] = [{"source_id": "s0"}]
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            core.nested_plan(membership, folds, groups)
        membership["validation"] = []
        folds[0]["fit_indices"].append(0)
        with self.assertRaisesRegex(ValueError, "Invalid outer"):
            core.nested_plan(membership, folds, groups)

    def test_fit_ids_exclude_unknown_targets_and_forbidden_sources(self):
        _, labels, features, _, _ = self.fixture()
        labels[0]["auxiliary_targets"] = {h: None for h in core.HEADS}
        models, audits = core.fit_heads(features, labels, {f"s{i}" for i in range(8)}, {"s8", "s9"}, fitter=self.fitter)
        self.assertEqual(set(models), set(core.HEADS))
        for audit in audits:
            self.assertNotIn(labels[0]["sample_id"], audit["fit_sample_ids"])
            self.assertFalse(set(audit["fit_source_ids"]) & {"s8", "s9"})
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            core.fit_heads(features, labels, {"s0"}, {"s0"}, fitter=self.fitter)

    def test_insufficient_group_support_prevents_fitting(self):
        _, labels, features, _, _ = self.fixture()
        fitter = MagicMock()
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            core.fit_heads(features, labels, {"s0", "s1"}, fitter=fitter)
        fitter.assert_not_called()

    def test_oof_scores_come_only_from_fits_without_held_sources(self):
        membership, labels, features, folds, _ = self.fixture()
        before = copy.deepcopy(labels)
        settings = {**core.SETTINGS, "bootstrap_repetitions": 40}
        result = core.cross_validate(features, labels, membership, folds, settings, fitter=self.fitter)
        self.assertEqual(labels, before)
        self.assertEqual(len(result["fit_audits"]), 15)
        self.assertTrue(result["quality"]["all_heads_passed"])
        self.assertEqual(result["oof_scores"].shape, (20, 3))
        for audit in result["fit_audits"]:
            held = {membership["train"][i]["source_id"] for i in folds[audit["fold"]]["held_indices"]}
            self.assertFalse(held & set(audit["fit_source_ids"]))

    def test_oof_rejects_validation_labels_or_forged_training_identity(self):
        membership, labels, features, folds, _ = self.fixture()
        labels[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training-source"):
            core.cross_validate(features, labels, membership, folds, fitter=self.fitter)
        labels[0]["split"] = "train"
        labels[0]["sample_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "approved training"):
            core.cross_validate(features, labels, membership, folds, fitter=self.fitter)

    def test_fixed_gate_stops_all_augmentation_if_unsafe_head_fails(self):
        _, labels, features, _, _ = self.fixture()
        good = np.tile(.1 + .8 * features[:, :1], (1, 3))
        bad = good.copy()
        bad[:, 2] = .2
        quality = core.assess_quality(labels, bad, np.zeros_like(bad))
        self.assertFalse(quality["all_heads_passed"])
        self.assertTrue(quality["heads"]["refusal_present"]["passed"])
        self.assertFalse(quality["heads"]["unsafe_enabling_assistance"]["gates"]["class_1_recall"])
        self.assertEqual(quality["heads"]["unsafe_enabling_assistance"]["teacher_label_agreement"]["class_recalls"]["1"], 0.)

    def test_fixed_cutoff_and_unknown_exclusion_are_not_recalibrated(self):
        _, labels, features, _, _ = self.fixture()
        scores = np.tile(.1 + .8 * features[:, :1], (1, 3))
        labels[0]["auxiliary_targets"][core.HEADS[0]] = None
        quality = core.assess_quality(labels, scores, np.zeros_like(scores))
        self.assertEqual(quality["heads"][core.HEADS[0]]["teacher_label_agreement"]["count"], 19)
        point = core.teacher_metrics([0, 1], [.5, .4999999])
        self.assertEqual(point["confusion_matrix"], {"tn": 0, "fp": 1, "fn": 1, "tp": 0})

    def test_group_bootstrap_preserves_pairs_and_is_reproducible(self):
        _, labels, features, _, _ = self.fixture()
        values = np.tile(.1 + .8 * features[:, :1], (1, 3))
        settings = {**core.SETTINGS, "bootstrap_repetitions": 40}
        result = core.group_bootstrap(labels, values, settings)
        self.assertEqual(result, core.group_bootstrap(labels, values, settings))
        self.assertEqual(result["source_groups"], 10)
        for head in core.HEADS:
            self.assertEqual(result["heads"][head]["macro_f1"]["percentile_95_interval"], [1., 1.])

    def test_runtime_returns_three_auxiliary_scores_without_activation_label(self):
        state = {"models": {h: Classifier() for h in core.HEADS}}
        encoder = MagicMock()
        encoder.encode.return_value = np.array([[1., 0.]])
        values = core.predict_one(state, {"prompt": "A harmless prompt", "response": "Blue."}, encoder)
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.allclose(values, .9))

    def test_existing_stages_refuse_overwrite_or_refit(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.consensus, "verify") as verifier:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
            verifier.assert_not_called()
        for stage in (workflow.train, workflow.benchmark):
            with patch.object(workflow, "RUN", run), patch.object(workflow, "check_protocol", return_value={}), patch.object(workflow, "check_seal"):
                with self.assertRaisesRegex(ValueError, "refuse refit|refuse repeat"):
                    stage()


if __name__ == "__main__":
    unittest.main()
