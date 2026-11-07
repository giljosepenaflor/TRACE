"""Harmless checks for matched-view nested fusion and operating-point calibration."""
import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_end_to_end as workflow
from traceguard import end_to_end as core
from traceguard import auxiliary_scorer as auxiliary
from traceguard.detector import choose_threshold


class EndToEndTests(unittest.TestCase):
    def fixture(self):
        membership = {"train": [], "validation": []}
        labels, features, groups = [], [], []
        for group in range(10):
            for condition in (1, 2):
                row = {"sample_id": f"s{group}-c{condition}", "source_id": f"s{group}", "source": "AdvBench" if group % 2 else "HarmBench",
                       "split": "train", "condition_id": condition, "activation_label": int(condition == 2)}
                membership["train"].append(row)
                labels.append({k: row[k] for k in ("sample_id", "source_id", "split")}
                    | {"auxiliary_targets": {h: int(condition == 2) for h in auxiliary.HEADS}})
                features.append([float(condition == 2), group / 10])
                groups.append(row["source_id"])
        folds = [{"fold": i, "fit_indices": [j for j in range(20) if j // 4 != i], "held_indices": list(range(i*4, i*4+4))} for i in range(5)]
        nested = auxiliary.nested_plan(membership, folds, groups)
        base = np.asarray(features)
        generation = np.c_[base[:, 0], base[:, 1], .5 * base[:, 0]]
        return membership, labels, base, base.copy(), generation, nested

    def test_fast_max_f1_matches_existing_calibration_and_fixed_targets(self):
        rng = np.random.default_rng(7)
        for _ in range(8):
            y = np.r_[np.zeros(15, dtype=int), np.ones(5, dtype=int)]
            scores = rng.random(20)
            policies = core.thresholds(y, scores)
            self.assertEqual(policies["max_f1"], choose_threshold(y, scores))
            self.assertLessEqual(np.mean(scores[y == 0] >= policies["fpr_0.10"]), .10)
            self.assertGreaterEqual(np.mean(scores[y == 1] >= policies["recall_0.80"]), .80)

    def test_no_positive_budget_handles_saturated_scores(self):
        policies = core.thresholds([0, 0, 1], [1., 1., 1.])
        self.assertGreater(policies["fpr_0.10"], 1.)
        with self.assertRaisesRegex(ValueError, "both classes"):
            core.thresholds([0, 0], [.1, .2])

    def test_matched_behavioral_inputs_cannot_see_generation_features(self):
        behavior = np.arange(12).reshape(4, 3) / 12
        generation = np.zeros((4, 3))
        self.assertTrue(np.array_equal(core.combiner_input("behavioral_enhanced", generation, behavior), behavior))
        self.assertTrue(np.array_equal(core.combiner_input("behavioral_enhanced", generation + 100, behavior), behavior))
        values = core.combiner_input("full_linear", generation, behavior)
        self.assertEqual(values.shape, (4, 6))
        self.assertTrue(np.array_equal(values[:, 3:], behavior))
        with self.assertRaisesRegex(ValueError, "three matched"):
            core.combiner_input("full_linear", generation, behavior[:, :2])

    def test_components_exclude_failed_head_unknown_labels_and_forbidden_sources(self):
        membership, labels, base, semantic, _, _ = self.fixture()
        labels[0]["auxiliary_targets"]["refusal_present"] = None
        models, audits = core.fit_components(base, semantic, membership, labels, np.arange(16), {"s8", "s9"})
        self.assertNotIn("unsafe_enabling_assistance", models)
        for row in audits:
            self.assertFalse(set(row["fit_source_ids"]) & {"s8", "s9"})
            if row["component"] == "refusal_present":
                self.assertNotIn(labels[0]["sample_id"], row["fit_sample_ids"])
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            core.fit_components(base, semantic, membership, labels, np.arange(16), {"s0"})

    def test_duplicate_component_indices_are_rejected(self):
        membership, labels, base, semantic, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "indices"):
            core.fit_components(base, semantic, membership, labels, [0, 0, 1])

    def test_nested_activation_training_protects_every_outer_source(self):
        membership, labels, base, semantic, generation, folds = self.fixture()
        before = copy.deepcopy(labels)
        settings = {**core.SETTINGS, "tree": {**core.TREE, "max_iter": 2}}
        result = core.cross_validate(base, semantic, generation, membership, labels, folds, settings)
        self.assertEqual(labels, before)
        self.assertEqual(len(result["component_fit_audits"]), 90)
        self.assertTrue(all(np.isfinite(v).all() for v in result["oof_scores"].values()))
        for audit in result["component_fit_audits"]:
            outer = folds[audit["outer_fold"]]
            forbidden = {membership["train"][i]["source_id"] for i in outer["held_indices"]}
            if audit["stage"] == "inner":
                inner = outer["inner_folds"][audit["inner_fold"]]
                forbidden |= {membership["train"][i]["source_id"] for i in inner["held_indices"]}
            self.assertFalse(set(audit["fit_source_ids"]) & forbidden)
        for fold in result["fold_calibrations"]:
            held = {membership["train"][i]["sample_id"] for i in folds[fold["outer_fold"]]["held_indices"]}
            self.assertFalse(set(fold["fit_sample_ids"]) & held)

    def test_nested_cross_partition_or_inner_overlap_is_rejected(self):
        membership, labels, base, semantic, generation, folds = self.fixture()
        membership["validation"] = [{"source_id": "s0"}]
        with self.assertRaisesRegex(ValueError, "crosses development"):
            core.cross_validate(base, semantic, generation, membership, labels, folds)
        membership["validation"] = []
        folds[0]["inner_folds"][0]["fit_indices"].append(folds[0]["inner_folds"][0]["held_indices"][0])
        with self.assertRaisesRegex(ValueError, "Invalid inner"):
            core.cross_validate(base, semantic, generation, membership, labels, folds)

    def test_method_selection_ties_prefer_linear_without_validation(self):
        values = [{m: {"average_precision": .5, "auroc": .7} for m in core.METHODS} for _ in range(5)]
        selected = core.selection(values)
        self.assertEqual(selected["selected_full"], "full_linear")
        self.assertEqual(selected["best_all_methods"], "generation")
        self.assertFalse(selected["validation_or_test_used_for_model_selection"])

    def test_runtime_generation_needs_no_text_encoder(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[.2, .8]])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        state = {"method": "generation", "model": model, "scaler": scaler}
        record = {"signals": {"generated_token_ids": [1], "token_confidences": [.7], "token_entropies": [.5], "statistics_source": "full_vocabulary_generation_scores"}}
        self.assertEqual(core.predict_one(state, record), .8)

    def test_predicted_operating_policies_and_whole_source_bootstrap_are_preserved(self):
        membership, _, base, _, _, _ = self.fixture()
        scores = {m: .1 + .8 * base[:, 0] for m in core.METHODS}
        thresholds = {m: dict.fromkeys(core.POLICIES, .5) for m in core.METHODS}
        rows = core.prediction_rows(membership["train"], scores, thresholds)
        summary = core.summarize(rows)
        self.assertEqual(summary["train"]["max_f1"]["full_linear"]["f1"], 1.)
        settings = {**core.SETTINGS, "bootstrap_repetitions": 30}
        first = core.paired_intervals(rows, "full_linear", settings)
        self.assertEqual(first, core.paired_intervals(rows, "full_linear", settings))
        self.assertTrue(all(x["f1_difference"] == 0 for x in first.values()))
        with self.assertRaisesRegex(ValueError, "whole source pairs"):
            core.paired_intervals(rows[:-1], "full_linear", settings)

    def test_existing_stages_refuse_overwrite(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.previous, "verify") as verifier:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
            verifier.assert_not_called()
        for stage in (workflow.train, workflow.evaluate, workflow.benchmark):
            with patch.object(workflow, "RUN", run), patch.object(workflow, "check_protocol", return_value={}), patch.object(workflow, "check_seal"):
                with self.assertRaisesRegex(ValueError, "refuse refit|refuse repeat"):
                    stage()


if __name__ == "__main__":
    unittest.main()
