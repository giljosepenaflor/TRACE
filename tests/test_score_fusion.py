"""Harmless checks for isolated mixture selection and frozen inference."""
import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_score_fusion as workflow
from traceguard import score_fusion as core
from tests import test_reused_comparison as fixtures


class ScoreFusionTests(unittest.TestCase):
    def training_fixture(self):
        rows, groups = [], []
        for source in range(6):
            for condition in (1, 2):
                rows.append({"sample_id": f"s{source}-c{condition}", "source_id": f"s{source}", "condition_id": condition,
                             "split": "train", "activation_label": int(source % 2 == 0 and condition == 2)})
                groups.append(f"s{source}")
        folds = [{"fit_indices": [j for j in range(12) if j // 4 != i],
                  "held_indices": list(range(i * 4, i * 4 + 4))} for i in range(3)]
        labels = np.asarray([r["activation_label"] for r in rows])
        return rows, groups, folds, np.where(labels, .9, .1), np.where(labels, .8, .2)

    def test_mixture_endpoints_bounds_and_no_input_mutation(self):
        g, b = np.array([0., .4, 1.]), np.array([1., .8, 0.])
        self.assertTrue(np.array_equal(core.mix(g, b, 0), g))
        self.assertTrue(np.array_equal(core.mix(g, b, 1), b))
        self.assertTrue(np.allclose(core.mix(g, b, .25), [.25, .5, .75]))
        self.assertEqual(g.tolist(), [0., .4, 1.])
        for weight in (-.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                core.mix(g, b, weight)
        for invalid in ([0., float("nan"), 1.], [0., -1., 1.]):
            with self.assertRaises(ValueError):
                core.mix(invalid, b, .5)
        with self.assertRaisesRegex(ValueError, "shapes"):
            core.mix(g[:-1], b, .5)

    def test_training_ties_choose_smaller_weight_and_report_endpoint_winner(self):
        rows, groups, folds, g, b = self.training_fixture()
        results, selection, scores = core.select_training_mixture(g, b, rows, folds, groups)
        self.assertEqual(len(results), 8)
        self.assertEqual(selection["selected_fusion_weight"], .05)
        self.assertEqual(selection["best_overall_weight_including_endpoints"], 0)
        self.assertEqual(scores.shape, (8, 12))
        self.assertFalse(selection["used_validation_or_test_scores_for_selection"])
        self.assertEqual((results, selection), core.select_training_mixture(g, b, rows, folds, groups)[:2])

    def test_weight_selection_rejects_validation_test_and_clean_records(self):
        rows, groups, folds, g, b = self.training_fixture()
        for split in ("validation", "test_seen", "test_unseen"):
            changed = copy.deepcopy(rows)
            changed[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "Only training"):
                core.select_training_mixture(g, b, changed, folds, groups)
        rows[0]["condition_id"] = 3
        with self.assertRaisesRegex(ValueError, "condition"):
            core.select_training_mixture(g, b, rows, folds, groups)

    def test_fold_leakage_repetition_and_missing_rows_rejected(self):
        rows, groups, folds, g, b = self.training_fixture()
        changed = copy.deepcopy(folds)
        changed[0]["held_indices"] = [0, 2, 4, 6]
        changed[0]["fit_indices"] = [j for j in range(12) if j not in changed[0]["held_indices"]]
        with self.assertRaisesRegex(ValueError, "group leaks"):
            core.select_training_mixture(g, b, rows, changed, groups)
        for changed in (folds + folds[:1], folds[:-1]):
            with self.assertRaisesRegex(ValueError, "coverage"):
                core.select_training_mixture(g, b, rows, changed, groups)
        with self.assertRaisesRegex(ValueError, "membership"):
            core.select_training_mixture(g[:-1], b, rows, folds, groups)

    def test_invalid_mixture_grid_rejected(self):
        rows, groups, folds, g, b = self.training_fixture()
        for weights in ((0., 1.), (0., .1, .1, 1.), (.1, .5, 1.)):
            with self.assertRaisesRegex(ValueError, "grid"):
                core.select_training_mixture(g, b, rows, folds, groups, weights)

    def test_only_validation_calibrates_threshold_and_higher_tie_is_used(self):
        rows, _, _, _, b = self.training_fixture()
        for row in rows:
            row["split"] = "validation"
        threshold, result = core.calibrate_validation(b, rows)
        self.assertEqual(threshold, .8)
        self.assertEqual(result["f1"], 1.)
        rows[0]["split"] = "test_seen"
        with self.assertRaisesRegex(ValueError, "Only validation"):
            core.calibrate_validation(b, rows)

    def retrospective_fixture(self):
        labels, old, base, signals, states = fixtures.ReusedComparisonTests().fixture()
        previous = fixtures.core.score_candidates(base, signals, labels, old, states)
        thresholds = {m: .5 if m != "score_fusion" else .3 for m in core.METHODS}
        return previous, thresholds

    def test_existing_predictions_are_preserved_without_any_classifier_call(self):
        previous, thresholds = self.retrospective_fixture()
        before = copy.deepcopy(previous)
        rows = core.add_frozen_fusion(previous, .25, thresholds)
        self.assertEqual(previous, before)
        self.assertEqual(core.summarize(rows, thresholds)["test_seen"]["score_fusion"]["f1"], 1.)
        for row, old in zip(rows, previous):
            for method in core.METHODS:
                if method != "score_fusion":
                    self.assertEqual(row["scores"][method], old["scores"][method])
                    self.assertEqual(row["predictions"][method], old["predictions"][method])
        thresholds["original_full"] = .1
        with self.assertRaisesRegex(ValueError, "changed"):
            core.add_frozen_fusion(previous, .25, thresholds)

    def test_saved_threshold_decisions_cannot_be_modified(self):
        previous, thresholds = self.retrospective_fixture()
        rows = core.add_frozen_fusion(previous, .25, thresholds)
        rows[0]["predictions"]["score_fusion"] = 1
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            core.summarize(rows, thresholds)

    def test_bootstrap_resamples_complete_source_pairs_reproducibly(self):
        previous, thresholds = self.retrospective_fixture()
        rows = core.add_frozen_fusion(previous, .25, thresholds)
        first = core.paired_intervals(rows, 40, 9)
        self.assertEqual(first, core.paired_intervals(rows, 40, 9))
        self.assertEqual(first["score_fusion_minus_generation"]["f1_difference"], 1.)
        self.assertEqual(first["score_fusion_minus_generation"]["source_groups"], 4)
        with self.assertRaisesRegex(ValueError, "complete source pairs"):
            core.paired_intervals(rows[:-1], 40, 9)

    def test_fusion_uses_frozen_classifiers_one_encoding_and_both_components(self):
        g = {"view": "generation", "threshold": .5, "scaler": fixtures.FrozenScaler(), "model": fixtures.FrozenModel(), "projection": None}
        b = {"view": "behavioral", "threshold": .5, "projection": None, "model": fixtures.FrozenModel()}
        state = {"kind": "score_fusion", "generation_component": g, "behavioral_component": b,
                 "behavioral_weight": .25, "threshold": .65}
        encoder = MagicMock()
        encoder.encode_base.return_value = np.array([[.6, .4]])
        record = {"signals": {"generated_token_ids": [1, 2], "token_confidences": [.8, .8], "token_entropies": [.7, .7],
                               "statistics_source": "full_vocabulary_generation_scores"}}
        score, decision = core.predict_one(state, record, encoder)
        self.assertAlmostEqual(score, .675)
        self.assertEqual(decision, 1)
        encoder.encode_base.assert_called_once()
        self.assertEqual(g["model"].calls, 1)
        self.assertEqual(b["model"].calls, 1)

    def test_generation_endpoint_skips_behavioral_encoder_and_classifier(self):
        g = {"view": "generation", "threshold": .5, "scaler": fixtures.FrozenScaler(), "model": fixtures.FrozenModel(), "projection": None}
        state = {"kind": "score_fusion", "generation_component": g, "behavioral_component": MagicMock(),
                 "behavioral_weight": 0., "threshold": .5}
        record = {"signals": {"generated_token_ids": [1], "token_confidences": [.8], "token_entropies": [.7],
                               "statistics_source": "full_vocabulary_generation_scores"}}
        self.assertEqual(core.predict_one(state, record)[1], 1)
        state["behavioral_component"].assert_not_called()

    def test_existing_stages_refuse_overwrite(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.previous, "verify") as verifier:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
            verifier.assert_not_called()
        for stage in (workflow.develop, workflow.evaluate, workflow.benchmark):
            with patch.object(workflow, "RUN", run), patch.object(workflow, "check_protocol", return_value={}), patch.object(workflow, "check_seal"):
                with self.assertRaisesRegex(ValueError, "refuse repeat"):
                    stage()


if __name__ == "__main__":
    unittest.main()
