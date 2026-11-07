"""Offline held-out workflow tests with harmless text and fabricated software fixtures.

Fixture scores are not experiment results; no real StrongREJECT responses are
generated, reviewed or evaluated in these tests.
"""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import trace_strongreject_evaluate as workflow


class ReviewSealTests(unittest.TestCase):
    def fixture(self, pattern="0100", other=None):
        mapping, a, b = {}, {}, {}
        for condition in (1, 2, 3, 4):
            rid = f"R{condition}"
            mapping[rid] = {"review_id": rid, "source_id": "STRONGREJECT-001", "condition_id": condition,
                            "split": "test_seen", "raw_path": f"fixture-{condition}.json", "raw_sha256": "fixture",
                            "question_sha256": workflow.common.text_sha("Why do rainbows form?"),
                            "response_sha256": workflow.common.text_sha("Light refracts and disperses.")}
            a[rid] = {"task_success": pattern[condition - 1], "brief_reason": "Reviewer explanation.", "reviewer": "Alice"}
            b[rid] = {"task_success": (other or pattern)[condition - 1], "brief_reason": "Reviewer explanation.", "reviewer": "Bob"}
        return mapping, a, b

    def test_all_binary_patterns_only_0000_0100_admit_adapted_labels(self):
        for number in range(16):
            pattern = f"{number:04b}"
            labels, groups = workflow.derive_labels(*self.fixture(pattern))
            if pattern in ("0000", "0100"):
                self.assertEqual([r["activation_label"] for r in labels], [0, int(pattern == "0100")])
                self.assertEqual([r["condition_id"] for r in labels], [1, 2])
                self.assertEqual(groups[0]["status"], "eligible")
            else:
                self.assertFalse(labels)
                self.assertEqual(groups[0]["status"], "pattern_not_admitted")

    def test_uncertainty_disagreement_and_incomplete_groups_are_not_forced(self):
        for a, b, status in (("0U00", "0U00", "uncertain_review"), ("0100", "0000", "review_disagreement")):
            labels, groups = workflow.derive_labels(*self.fixture(a, b))
            self.assertFalse(labels)
            self.assertEqual(groups[0]["status"], status)
        mapping, a, b = self.fixture()
        mapping.pop("R4")
        with self.assertRaisesRegex(ValueError, "all four"):
            workflow.derive_labels(mapping, a, b)

    def completed(self):
        mapping, _, _ = self.fixture()
        rows = [{"review_id": rid, "source_question": "Why do rainbows form?", "model_response": "Light refracts and disperses.",
                 "task_success_0_1_U": "0", "brief_reason": "Does not perform the requested task.", "reviewer": "Alice"}
                for rid in mapping]
        return mapping, rows

    def test_review_requires_exact_text_full_coverage_reasons_and_identity(self):
        mapping, rows = self.completed()
        with patch.object(Path, "read_bytes", return_value=json.dumps(rows).encode()):
            reviews, name, _ = workflow.read_review(Path("completed.json"), mapping)
        self.assertEqual(len(reviews), 4)
        self.assertEqual(name, "alice")
        for column, value in (("source_question", "Changed."), ("model_response", "Changed."),
                              ("task_success_0_1_U", "11"), ("brief_reason", ""), ("reviewer", "")):
            bad = copy.deepcopy(rows)
            bad[0][column] = value
            with self.subTest(column=column), patch.object(Path, "read_bytes", return_value=json.dumps(bad).encode()), self.assertRaises(ValueError):
                workflow.read_review(Path("completed.json"), mapping)


class FrozenScoringTests(unittest.TestCase):
    def fixture(self, *, only_seen=False, single_class=False):
        labels, records = [], []
        for i in range(4):
            split = "test_seen" if i < 2 or only_seen else "test_unseen"
            labels.append({"sample_id": f"Q{i}-condition-2", "source_id": f"Q{i}", "condition_id": 2,
                           "split": split, "activation_label": 0 if single_class else i % 2})
            records.append({"sample_id": labels[-1]["sample_id"], "prompt": "Why do rainbows form?",
                            "response": "Light refracts and disperses.", "generation_vector": np.array([.2 + .2 * i, .8, .01])})

        class Encoder:
            def transform(self, rows):
                matrix = np.zeros((len(rows), 64))
                matrix[:, 0] = [.2, .8, .3, .9]
                return matrix
            def fit_transform(self, *args, **kwargs):
                raise AssertionError("Evaluation must not fit the encoder")

        class Scaler:
            def transform(self, matrix): return matrix
            def fit(self, *args): raise AssertionError("Evaluation must not fit the scaler")

        class Model:
            def predict_proba(self, matrix): return np.c_[1 - matrix[:, 0], matrix[:, 0]]
            def fit(self, *args): raise AssertionError("Evaluation must not fit a classifier")

        artifact = {"encoder": Encoder(), "scaler": Scaler(), "models": {name: Model() for name in workflow.VIEWS},
                    "thresholds": {"full": .8, "behavioral": .5, "generation": .4}}
        return artifact, records, labels

    def test_all_views_predict_using_unchanged_thresholds_without_fitting_or_selection(self):
        artifact, records, labels = self.fixture()
        original = dict(artifact["thresholds"])
        with patch("traceguard.detector.choose_threshold", side_effect=AssertionError("No held-out threshold tuning")):
            report, rows = workflow.score_frozen(artifact, records, labels)
        self.assertEqual(artifact["thresholds"], original)
        self.assertEqual(set(report), {"test_seen", "test_unseen"})
        self.assertEqual(report["test_seen"]["count"], 2)
        self.assertEqual(report["test_unseen"]["count"], 2)
        self.assertEqual(set(rows[0]["scores"]), set(workflow.VIEWS))
        self.assertEqual(rows[1]["predictions"]["full"], 1)  # equality uses >=

    def test_no_eligible_partition_and_single_class_auc_are_not_fabricated(self):
        artifact, records, labels = self.fixture(only_seen=True, single_class=True)
        report, _ = workflow.score_frozen(artifact, records, labels)
        self.assertEqual(report["test_unseen"]["status"], "no_eligible_examples")
        self.assertIsNone(report["test_unseen"]["views"])
        self.assertIsNone(report["test_seen"]["views"]["full"]["auroc"])

    def test_clean_controls_and_changed_sample_order_cannot_be_scored(self):
        artifact, records, labels = self.fixture()
        labels[0]["condition_id"] = 3
        with self.assertRaisesRegex(ValueError, "adapted test"):
            workflow.score_frozen(artifact, records, labels)
        artifact, records, labels = self.fixture()
        records.reverse()
        with self.assertRaisesRegex(ValueError, "sample order"):
            workflow.score_frozen(artifact, records, labels)

    def test_existing_evaluation_stops_before_reading_or_scoring(self):
        run = MagicMock()
        (run / "evaluation").exists.return_value = True
        with patch.object(workflow.handoff, "RUN", run), patch.object(workflow, "checked_labels") as load:
            with self.assertRaisesRegex(ValueError, "already exists"):
                workflow.evaluate(Path("labels"), Path("evidence.json"))
            load.assert_not_called()


class ReadinessTests(unittest.TestCase):
    def test_real_workspace_remains_missing_responses_without_model_evaluation(self):
        result = workflow.status()
        self.assertEqual(result["stage"], "awaiting_saved_responses")
        self.assertEqual(result["missing_responses"], 1252)
        self.assertFalse(result["evaluation_exists"])
        self.assertFalse(result["fitting_performed"])
        self.assertEqual(result["target_model_calls"], 0)

    def test_signal_scope_cannot_be_accepted_without_collection_protocol_linkage(self):
        with patch.object(workflow.handoff, "read", return_value={}), self.assertRaisesRegex(ValueError, "not bound"):
            workflow.signal_evidence(Path("not-real.json"), "fixture-digest")


if __name__ == "__main__":
    unittest.main()
