"""Harmless fixtures for fixed-model dataset reuse and validation benchmarking."""
import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_reused_evaluation as workflow
from scripts.trace_error_efficiency import validate_benchmark_membership
from traceguard import reused_comparison as core


class FrozenModel:
    def __init__(self):
        self.calls = 0

    def fit(self, *args, **kwargs):
        raise AssertionError("Refitting is forbidden")

    def predict_proba(self, values):
        self.calls += 1
        score = np.asarray(values)[:, 0]
        return np.c_[1 - score, score]


class OriginalModel(FrozenModel):
    def predict_proba(self, values):
        raise AssertionError("Original test predictions must be reused")


class FrozenScaler:
    def fit(self, *args, **kwargs):
        raise AssertionError("Scaler refitting is forbidden")

    def transform(self, values):
        return np.asarray(values)


class ReusedComparisonTests(unittest.TestCase):
    def fixture(self):
        labels, old = [], []
        for split in ("test_seen", "test_unseen"):
            for group, pattern in enumerate(("0100", "0000")):
                for condition in (1, 2):
                    label = int(pattern == "0100" and condition == 2)
                    sid = f"{split}-{group}"
                    row = {"sample_id": f"{sid}-condition-{condition}", "source_id": sid,
                           "split": split, "condition_id": condition, "activation_label": label,
                           "four_condition_pattern": pattern}
                    labels.append(row)
                    old.append({k: row[k] for k in ("sample_id", "source_id", "split", "activation_label")}
                               | {"scores": {v: 0.3 if label else 0.2 for v in ("full", "behavioral", "generation")},
                                  "predictions": {v: 0 for v in ("full", "behavioral", "generation")}})
        base = np.array([[0.8 if r["activation_label"] else 0.2, 0.4] for r in labels])
        signals = np.ones((len(labels), 3)) * 0.1
        states = {m: {"view": d["view"], "threshold": 0.5, "projection": None, "scaler": FrozenScaler(),
                      "model": OriginalModel() if d["original_view"] else FrozenModel()}
                  for m, d in core.METHODS.items()}
        return labels, old, base, signals, states

    def test_only_new_frozen_models_are_scored_and_baselines_are_preserved(self):
        labels, old, base, signals, states = self.fixture()
        before = copy.deepcopy(old)
        rows = core.score_candidates(base, signals, labels, old, states)
        self.assertEqual(old, before)
        self.assertEqual(len(rows), len(labels))
        for method, definition in core.METHODS.items():
            if definition["original_view"]:
                self.assertEqual(states[method]["model"].calls, 0)
                self.assertEqual([r["scores"][method] for r in rows], [r["scores"][definition["original_view"]] for r in old])
            else:
                self.assertEqual(states[method]["model"].calls, 1)
        self.assertTrue(all(s["threshold"] == 0.5 for s in states.values()))

    def test_label_pattern_clean_condition_duplicate_and_development_leak_rejected(self):
        labels, _, _, _, _ = self.fixture()
        for field, value in (("condition_id", 3), ("four_condition_pattern", "0110"), ("activation_label", 1)):
            changed = copy.deepcopy(labels)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                core.validate_test_membership(changed)
        with self.assertRaisesRegex(ValueError, "Development source"):
            core.validate_test_membership(labels, {labels[0]["source_id"]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            core.validate_test_membership(labels + labels[:1])

    def test_original_label_identity_and_threshold_changes_rejected(self):
        labels, old, base, signals, states = self.fixture()
        changed = copy.deepcopy(old)
        changed[0]["source_id"] = "different-source"
        with self.assertRaisesRegex(ValueError, "membership"):
            core.score_candidates(base, signals, labels, changed, states)
        states["original_full"]["threshold"] = 0.1
        with self.assertRaisesRegex(ValueError, "threshold changed"):
            core.score_candidates(base, signals, labels, old, states)

    def test_missing_original_predictions_and_incomplete_pairs_rejected(self):
        labels, old, base, signals, states = self.fixture()
        with self.assertRaisesRegex(ValueError, "coverage"):
            core.score_candidates(base, signals, labels, old[:-1], states)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            core.validate_test_membership(labels[:-1])

    def test_saved_decisions_must_match_existing_thresholds(self):
        labels, old, base, signals, states = self.fixture()
        rows = core.score_candidates(base, signals, labels, old, states)
        thresholds = {m: s["threshold"] for m, s in states.items()}
        report = core.summarize(rows, thresholds)
        self.assertEqual(report["test_seen"]["candidate_full"]["f1"], 1.0)
        self.assertEqual(report["test_seen"]["original_full"]["f1"], 0.0)
        rows[0]["predictions"]["candidate_full"] = 1
        with self.assertRaisesRegex(ValueError, "thresholds"):
            core.summarize(rows, thresholds)

    def test_bootstrap_keeps_source_pairs_and_is_reproducible(self):
        labels, old, base, signals, states = self.fixture()
        rows = core.score_candidates(base, signals, labels, old, states)
        first = core.paired_intervals(rows, repetitions=30, seed=11)
        self.assertEqual(first, core.paired_intervals(rows, repetitions=30, seed=11))
        self.assertEqual(first["candidate_full_minus_original_full"]["f1_difference"], 1.0)
        self.assertEqual(first["candidate_full_minus_original_full"]["source_groups"], 4)
        with self.assertRaisesRegex(ValueError, "source pairs"):
            core.paired_intervals(rows[:-1], repetitions=30, seed=11)

    def test_generation_only_never_requires_an_encoder(self):
        state = {"view": "generation", "threshold": 0.5, "projection": None,
                 "scaler": FrozenScaler(), "model": FrozenModel()}
        record = {"signals": {"generated_token_ids": [1, 2], "token_confidences": [0.8, 0.8],
                               "token_entropies": [0.7, 0.7], "statistics_source": "full_vocabulary_generation_scores"}}
        score, prediction = core.predict_one(state, record)
        self.assertAlmostEqual(score, 0.7)
        self.assertEqual(prediction, 1)

    def test_projected_and_raw_features_use_distinct_frozen_dimensions(self):
        _, _, base, signals, _ = self.fixture()
        raw = {"view": "full", "projection": None, "scaler": FrozenScaler()}
        projected = {**raw, "projection": np.array([[1.0], [0.0]])}
        self.assertEqual(core.features_for_state(raw, base, signals).shape[1], 5)
        self.assertEqual(core.features_for_state(projected, base, signals).shape[1], 4)

    def test_test_data_cannot_enter_validation_benchmark(self):
        labels, _, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Test/unexpected"):
            validate_benchmark_membership(labels, expected_count=len(labels))

    def test_existing_output_stages_refuse_repeat(self):
        run = MagicMock()
        run.exists.return_value = True
        with patch.object(workflow, "RUN", run), patch.object(workflow.representation, "check_protocol") as checker:
            with self.assertRaisesRegex(ValueError, "refuse overwrite"):
                workflow.prepare()
            checker.assert_not_called()
        with patch.object(workflow, "RUN", run), patch.object(workflow, "check_protocol", return_value={}):
            with self.assertRaisesRegex(ValueError, "refuse repeat"):
                workflow.evaluate()
            with self.assertRaisesRegex(ValueError, "refuse automatic repeat"):
                workflow.benchmark()


if __name__ == "__main__":
    unittest.main()
